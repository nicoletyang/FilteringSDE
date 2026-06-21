import logging
import os
import math
import shutil
from typing import List, Optional, Sequence

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal
from scipy.stats import wasserstein_distance

import torchsde
from sdeint_obs import sdeint_obs

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

#
class StochasticLorenz96(object):
    """Stochastic Lorenz-96 (Ito SDE), diagonal multiplicative noise.

    dx_i = ((x_{i+1} - x_{i-2}) * x_{i-1}) - x_i + F + b_i * x_i * dW_i
    with cyclic indices.
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, dim: int = 15, F: float = 8.0, b: Sequence = None):
        super(StochasticLorenz96, self).__init__()
        self.dim = int(dim)
        self.F = float(F)
        if b is None:
            self.b = torch.tensor([0.2] * self.dim, dtype=torch.float32)
        else:
            b = torch.as_tensor(b, dtype=torch.float32)
            assert b.numel() == self.dim, "len(b) must equal dim"
            self.b = b

    def f(self, t, y):
        # y: (B, D)
        y_ip1 = torch.roll(y, shifts=-1, dims=1)
        y_im1 = torch.roll(y, shifts=+1, dims=1)
        y_im2 = torch.roll(y, shifts=+2, dims=1)
        drift = (y_ip1 - y_im2) * y_im1 - y + self.F
        return drift

    def g(self, t, y):
        return y * self.b.to(y.device).unsqueeze(0)

    @torch.no_grad()
    def sample(self, x0, ts):
        xs = torchsde.sdeint(self, x0, ts)  # (S,B,D)
        return xs


# -----------------------------------------------------------------------------
# Stochastic Lorenz-63 SDE (Ground Truth Physics)
# -----------------------------------------------------------------------------
class StochasticLorenz63(nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self, 
        theta1: float = 10.0, theta2: float = 28.0, theta3: float = 8.0/3.0,
        beta1: float = 0.1, beta2: float = 0.28, beta3: float = 0.3
    ):
        super().__init__()
        self.theta1, self.theta2, self.theta3 = float(theta1), float(theta2), float(theta3)
        self.register_buffer('beta', torch.tensor([beta1, beta2, beta3], dtype=torch.float32))

    def f(self, t, y):
        x1, x2, x3 = y[..., 0], y[..., 1], y[..., 2]
        dx1 = self.theta1 * (x2 - x1)
        dx2 = (x1 * (self.theta2 - x3)) - x2
        dx3 = (x1 * x2) - (self.theta3 * x3)
        return torch.stack([dx1, dx2, dx3], dim=-1)

    def g(self, t, y):
        if self.beta.device != y.device: self.beta = self.beta.to(y.device)
        return y * self.beta

    @torch.no_grad()
    def sample(self, x0, ts):
        return torchsde.sdeint(self, x0, ts)

# -----------------------------------------------------------------------------
# Observation Model 
# -----------------------------------------------------------------------------
class LorenzAtanObsModel:
    def __init__(self, obs_dim: int, data_dim: int, sigma: float):
        self.obs_dim = int(obs_dim)
        self.data_dim = int(data_dim)
        self.sigma = float(sigma)
        self._Rinv = (1.0 / (sigma ** 2))
        self._const = -0.5 * self.obs_dim * math.log(2 * math.pi * sigma * sigma)

    def loglik(self, y: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (Particles, B, D) or (B, D)
        # y: (B, D_obs) or (T, B, D_obs) - usually single step here
        # mask: (B,) or (B, D_obs) - Optional
        
        if x.ndim == 3 and y.ndim == 2:
             y = y.unsqueeze(0).expand(x.shape[0], -1, -1)
             if mask is not None and mask.ndim == 2:
                 mask = mask.unsqueeze(0).expand(x.shape[0], -1, -1)
        
        y_obs = y[..., :self.obs_dim]
        x_obs = x[..., :self.obs_dim]
        
        mean = torch.atan(x_obs)
        resid = y_obs - mean
        quad = -0.5 * self._Rinv * (resid**2).sum(dim=-1)
        
        if mask is not None:
            # If mask is provided, zero out likelihood for missing data.
            # Assuming mask is 1 for observed, 0 for missing.
            # If mask is per-dimension, we should have applied it inside the sum.
            # If mask is per-timestep (scalar), apply here.
            # Simplified: assuming scalar mask per sample for PF step
            if mask.ndim == quad.ndim:
                quad = quad * mask
            elif mask.ndim == quad.ndim + 1: # if mask is per-dim
                 # Re-calculate quad per dim
                 quad_per_dim = -0.5 * self._Rinv * (resid**2)
                 quad = (quad_per_dim * mask[..., :self.obs_dim]).sum(dim=-1)

        return self._const + quad

# -----------------------------------------------------------------------------
# Dataset Helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def make_dataset(t0, t1, batch_size, noise_std, train_dir, device,
                 normalize=True, steps=400, D=15):
    data_path = os.path.join(train_dir, 'l96_train.pth')
    if os.path.exists(data_path):
        data = torch.load(data_path, map_location=device)
        return data['xs'].to(device), data['ts'].to(device), data.get('mean'), data.get('std')

    ts = torch.linspace(t0, t1, steps=steps, device=device)
    x0 = torch.randn(batch_size, D, device=device)
    xs_raw = StochasticLorenz96(dim=D).sample(x0, ts)

    if normalize:
        mean = xs_raw.mean(dim=(0, 1))
        std = xs_raw.std(dim=(0, 1))
        xs = (xs_raw - mean) / (std + 1e-8)
    else:
        mean = torch.zeros(D, device=device)
        std = torch.ones(D, device=device)
        xs = xs_raw

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    torch.save({'xs': xs.cpu(), 'ts': ts.cpu(), 
                'mean': mean.cpu(), 'std': std.cpu()}, data_path)
    return xs, ts, mean, std



@torch.no_grad()
def make_testdataset(t0, t1, batch_size, noise_std, train_dir, device,
                     normalize=True, steps=400, D=15):
    data_path = os.path.join(train_dir, 'l96_test.pth')
    if os.path.exists(data_path):
        data = torch.load(data_path, map_location=device)
        return data['xs'].to(device), data['ts'].to(device)

    train_path = os.path.join(train_dir, 'l96_train.pth')
    if not os.path.exists(train_path):
        raise RuntimeError("Train data missing.")
    train_data = torch.load(train_path, map_location=device)
    mean, std = train_data['mean'].to(device), train_data['std'].to(device)

    ts = torch.linspace(t0, t1, steps=steps, device=device)
    x0 = torch.randn(batch_size, D, device=device)
    xs_raw = StochasticLorenz96(dim=D).sample(x0, ts)

    if normalize:
        xs = (xs_raw - mean) / (std + 1e-8)
    else:
        xs = xs_raw

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    torch.save({'xs': xs.cpu(), 'ts': ts.cpu()}, data_path)
    return xs, ts

# -----------------------------------------------------------------------------
# Neural SDE Components (Updated for Masking)
# -----------------------------------------------------------------------------
# ===================== CAUSAL VERSION (drop-in replacement) =====================
# This updates ONLY the encoder + LatentSDE parts so that:
#   - No reverse-time GRU (no look-ahead).
#   - Attention is strictly causal (no attending to future times).
#   - The per-time embedding used inside the drift uses only y_{0:t}.
#   - q(z0 | y) uses only y_0 (strictest causality).
#
# You can paste these classes over your existing Encoder / ObsEncoderWithAttention / LatentSDE.
# Everything else in your script can remain unchanged.

import math
import torch
from torch import nn


class EncoderCausal(nn.Module):
    """
    Forward GRU only (causal). Input is (T, B, D).
    Output is (T, B, context_size), where output[t] depends only on inputs[:t].
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = torch.nan_to_num(inp, nan=0.0)
        out, _ = self.gru(inp)  # (T,B,H)
        return self.lin(out)    # (T,B,output_size)


class ObsEncoderWithCausalAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, output_size)

    @staticmethod
    def _causal_attn_mask(T: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward_per_time(self, inp: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        inp = torch.nan_to_num(inp, nan=0.0)
        B, T, _ = inp.shape
        tokens = self.input_proj(inp)  # (B,T,H)
        attn_mask = self._causal_attn_mask(T, tokens.device)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)
        attn_out, _ = self.self_attention(
            tokens, tokens, tokens,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )  # (B,T,H)
        return self.output_proj(attn_out)  # (B,T,output_size)

    def forward_global_from_t0(self, inp: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        seq = self.forward_per_time(inp, mask=mask)  # (B,T,C)
        return seq[:, 0, :]  # (B,C)

    def forward_global_last(self, inp: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        seq = self.forward_per_time(inp, mask=mask)
        return seq[:, -1, :]


class LatentSDE(nn.Module):
    """
    CAUSAL LatentSDE:
      - EncoderCausal (forward GRU only).
      - ObsEncoderWithCausalAttention (no future access).
      - _embed_obs_at_time(t) uses only obs context up to current time index.
      - q(z0|y) uses only y_0 (strictest causality). If you prefer, swap to last-token
        summary by using forward_global_last instead.
    """
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(
        self,
        data_size,
        latent_size,
        context_size,
        hidden_size,
        obs_size,       
        ctxobs_size=16,
        num_heads=2,
        time_d=32,
        causal=True,
    ):
        super().__init__()
        self.data_dim = data_size
        self.latent_size = latent_size
        self.context_size = context_size
        self.ctxobs_size = ctxobs_size
        self.causal = True  # enforce

        # Causal encoders
        self.encoder = EncoderCausal(data_size, hidden_size, context_size)
        self.obs_encoder = ObsEncoderWithCausalAttention(data_size, hidden_size, ctxobs_size, num_heads=num_heads)

        # z0 recognition net (strict causal: from obs at time 0)
        self.qz0_net = nn.Linear(ctxobs_size, latent_size * 2)

        # Prior + control drift; you can interpret control_net as u_theta
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size + ctxobs_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, latent_size),
        )
        self.control_net = nn.Sequential(
            nn.Linear(latent_size + ctxobs_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size+ ctxobs_size, hidden_size), nn.Tanh(), # 
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, latent_size),
        )

        # Diagonal diffusion
        self.g_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden_size), nn.Softplus(), nn.Linear(hidden_size, 1), nn.Sigmoid())
            for _ in range(latent_size)
        ])

        # Decoder to data space
        self.projector = nn.Linear(latent_size, data_size)

        # Prior on z0
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        # Cache
        self._ctx = None                # (ts, ctx_seq, obs_ctx_global0)
        self._obs_ctx_seq = None        # (T,B,Cobs)
        self._ts_cached = None
        self.time_d = time_d
        self.time_feat = nn.Sequential(nn.Linear(1, time_d), nn.Tanh(), nn.Linear(time_d, time_d))
        self.time_key = nn.Linear(ctxobs_size, time_d)
        self.time_val = nn.Linear(ctxobs_size, ctxobs_size)
        self.mix_gate = nn.Sequential(nn.Linear(ctxobs_size * 2, 64), nn.Tanh(), nn.Linear(64, 1))

        self._K = None  # (T,B,time_d)
        self._V = None  # (T,B,ctxobs_size)

    def _encode_obs_context(self, obs, ts, mask=None):
        """
        obs: (T,B,D)
        mask: (T,B,D) or (T,B) or None.
        We convert to time-wise mask (B,T) for attention.
        """
        obs_clean = torch.nan_to_num(obs, nan=0.0)

        # GRU context: (T,B,context_size), causal
        ctx_seq = self.encoder(obs_clean)

        # Attention expects (B,T,D)
        obs_b = obs_clean.permute(1, 0, 2)  # (B,T,D)

        mask_bt = None
        if mask is not None:
            if mask.ndim == 3:
                # (T,B,D) -> (B,T) time observed if any dim observed
                mask_bt = (mask.permute(1, 0, 2).sum(dim=-1) > 0).float()
            elif mask.ndim == 2:
                # (T,B) -> (B,T)
                mask_bt = mask.permute(1, 0).float()
            else:
                raise ValueError("mask must be (T,B,D) or (T,B)")

        # Per-time obs context (B,T,Cobs), causal attention
        obs_ctx_seq_btc = self.obs_encoder.forward_per_time(obs_b, mask=mask_bt)

        # Strict causal global for z0: take context at t=0
        obs_ctx_global = self.obs_encoder.forward_global_from_t0(obs_b, mask=mask_bt)  # (B,Cobs)

        # Cache as (T,B,Cobs)
        self._obs_ctx_seq = obs_ctx_seq_btc.permute(1, 0, 2).contiguous()  # (T,B,Cobs)
        self._ts_cached = ts
        self._ctx = (ts, ctx_seq, obs_ctx_global)
        self._K = self.time_key(self._obs_ctx_seq)  # (T,B,time_d)
        self._V = self.time_val(self._obs_ctx_seq)  # (T,B,Cobs)

        return ctx_seq, obs_ctx_global

    @torch.no_grad()
    def condition_on_obs(self, obs, ts, mask=None):
        _, obs_ctx_global = self._encode_obs_context(obs, ts, mask)
        qz0_mean, qz0_logstd = self.qz0_net(obs_ctx_global).chunk(2, dim=1)
        return qz0_mean, qz0_logstd

    def _embed_obs_at_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Return e_t (B,Cobs) using only observation context up to current time index.
        Uses:
          - linear interpolation between nearest cached grid points (causal),
          - plus optional attention over PAST indices only.
        """
        ts = self._ts_cached               # (T,)
        E = self._obs_ctx_seq              # (T,B,Cobs)
        K = self._K                        # (T,B,time_d)
        V = self._V                        # (T,B,Cobs)

        T, B, Cobs = E.shape
        Dk = K.size(-1)

        # Find bracket indices
        i1 = int(torch.searchsorted(ts, t, right=True).item())
        i0 = max(i1 - 1, 0)
        i1 = min(i1, T - 1)

        # Local interpolation (only uses i0,i1 <= current time)
        denom = (ts[i1] - ts[i0]).clamp_min(1e-8)
        w_loc = ((t - ts[i0]) / denom).clamp(0, 1)
        e_lin = (1.0 - w_loc) * E[i0] + w_loc * E[i1]  # (B,Cobs)

        # Past-only attention over indices [0..i1]
        past_len = i1 + 1
        Kp = K[:past_len]  # (past_len,B,Dk)
        Vp = V[:past_len]  # (past_len,B,Cobs)

        t_in = t.expand(B, 1)
        q = self.time_feat(t_in)  # (B,time_d)

        # scores: (B,past_len)
        scores = torch.bmm(
            q.unsqueeze(1),                  # (B,1,Dk)
            Kp.permute(1, 2, 0),             # (B,Dk,past_len)
        ).squeeze(1) / math.sqrt(Dk)

        w_all = torch.softmax(scores, dim=-1)  # (B,past_len)
        e_attn = torch.bmm(
            w_all.unsqueeze(1),              # (B,1,past_len)
            Vp.permute(1, 0, 2),             # (B,past_len,Cobs)
        ).squeeze(1)                          # (B,Cobs)

        alpha = torch.sigmoid(self.mix_gate(torch.cat([e_lin, e_attn], dim=1)))
        e_t = (1.0 - alpha) * e_lin + alpha * e_attn
        return e_t

    # --- torchsde drift/diffusion ---
    # NOTE: torchsde.sdeint will call f(t,z,obs) if you pass obs as "args".
    def f(self, t, z, obs):
        ts, ctx_seq, _ = self._ctx
        i = int(torch.searchsorted(ts, t, right=True).item()) - 1
        i = max(0, min(i, len(ts) - 1))
        e_t = self._embed_obs_at_time(t)
        prior = self.f_net(torch.cat((z, ctx_seq[i], e_t), dim=1))
        return torch.nan_to_num(prior, nan=0.0, posinf=1e6, neginf=-1e6)

    def h(self, t, z, obs):
        e_t = self._embed_obs_at_time(t)
        out = self.h_net(torch.cat((z, e_t), dim=1))
        # out = self.h_net(z)  # no obs context for h, to keep it simpler and more stable
        ctrl = self.control_net(torch.cat((z, e_t), dim=1))
        out = out + self.g(t, z) * ctrl
        return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)




    def g(self, t, z):
        chunks = torch.split(z, 1, dim=1)
        out = [g_net(c) for g_net, c in zip(self.g_nets, chunks)]
        return torch.nan_to_num(torch.cat(out, dim=1), nan=0.0, posinf=1e6, neginf=-1e6)


    # --- obs_loglik, forward, euler_maruyama, sample_posterior_paths_em below ---

    def obs_loglik(self, x_seq, y_seq, R, mask=None):
        # unchanged from your code (kept here for completeness if you want it inline)
        T, B, D = x_seq.shape
        Dy = y_seq.size(-1)
        R_chol = torch.linalg.cholesky(R)
        log_det = 2.0 * torch.log(torch.diag(R_chol)).sum()
        const = -0.5 * Dy * math.log(2 * math.pi)

        total = 0.0
        for t in range(T):
            x_obs = x_seq[t][..., :Dy]
            mean_y = torch.atan(x_obs)
            resid = y_seq[t] - mean_y

            v = torch.cholesky_solve(resid.unsqueeze(-1), R_chol).squeeze(-1)
            quad_unmasked = -0.5 * (resid * v).sum(dim=-1) + const - 0.5 * log_det

            if mask is not None:
                mask_t = mask[t]
                if mask_t.ndim == 1:
                    quad = quad_unmasked * mask_t
                else:
                    quad_elem = -0.5 * (resid * v)
                    const_elem = const / Dy
                    log_det_elem = 0.5 * log_det / Dy
                    term_elem = quad_elem + const_elem - log_det_elem
                    quad = (term_elem * mask_t).sum(dim=-1)
            else:
                quad = quad_unmasked

            total += quad.mean()
        return total

    def forward(
        self, xs, obs, ts, noise_std, H, R, mask=None,
        adjoint=False, method="euler",
        x_recon_weight=1.0, y_ll_weight=1.0
    ):
        # Encode (causal)
        _, obs_ctx_global = self._encode_obs_context(obs, ts, mask)

        # Strict causal z0 uses only y0 context
        qz0_mean, qz0_logstd = self.qz0_net(obs_ctx_global).chunk(2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        if adjoint:
            zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=1e-2, logqp=True, method=method)
        else:
            zs, log_ratio = sdeint_obs(self, z0, obs, ts, dt=1e-2, logqp=True, method=method)
        x_hat = self.projector(zs)

        log_pxs = torch.tensor(0.0, device=x_hat.device)
        if xs is not None and x_recon_weight > 0:
            xs_dist = torch.distributions.Normal(x_hat, noise_std)
            log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0) * x_recon_weight

        log_py = self.obs_loglik(x_hat, obs, R, mask) * y_ll_weight

        qz0 = torch.distributions.Normal(qz0_mean, qz0_logstd.exp())
        pz0 = torch.distributions.Normal(self.pz0_mean, self.pz0_logstd.exp())
        kl0 = torch.distributions.kl_divergence(qz0, pz0).sum(1).mean()
        kl_path = log_ratio.sum(0).mean()

        return log_pxs, log_py, (kl0 + kl_path), x_hat

    @torch.no_grad()
    def inverse_projector(self, xs):
        A = self.projector.weight
        b = self.projector.bias
        A_pinv = torch.linalg.pinv(A, rcond=1e-10)
        return (xs - b) @ A_pinv.T

    @torch.no_grad()
    def likelihood_force(self, z, y_i, H, R, mask_i=None):
        A_proj = self.projector.weight
        b_proj = self.projector.bias
        x = z @ A_proj.T + b_proj

        Dy = y_i.size(-1)
        x_obs = x[..., :Dy]
        
        atan_x = torch.atan(x_obs)
        resid = y_i - atan_x
        
        # Apply Mask to Resid
        if mask_i is not None:
            resid = resid * mask_i
            
        R_chol = torch.linalg.cholesky(R)
        v = torch.cholesky_solve(resid.unsqueeze(-1), R_chol).squeeze(-1) 

        jac = 1.0 / (1.0 + x_obs ** 2)
        score_obs = v * jac
        score_x = torch.zeros_like(x)
        score_x[..., :Dy] = score_obs
        force_z = score_x @ A_proj
        return torch.nan_to_num(force_z, nan=0.0)



    def euler_maruyama(self, z0, obs, ts, H, R, gain=1.0, mask=None):
        S = len(ts)
        z = torch.zeros(S, *z0.shape, device=z0.device, dtype=z0.dtype)
        z[0] = z0
        for i in range(1, S):
            t_prev = ts[i-1]
            dt = (ts[i] - t_prev).clamp_min(1e-8)
            sqrt_dt = torch.sqrt(dt)
            
            y_i = obs[i] if i < obs.size(0) else obs[-1]
            mask_i = mask[i] if (mask is not None and i < mask.size(0)) else None
            
            drift = self.h(t_prev, z[i-1], obs)
            like = self.likelihood_force(z[i-1], y_i, H, R, mask_i)
            g_diag = self.g(t_prev, z[i-1])
            dW = torch.randn_like(z0) * sqrt_dt
            
            z[i] = z[i-1] + (drift + gain * like) * dt + g_diag * dW
        return z


    @torch.no_grad()
    def sample_posterior_paths_em(self, obs, ts_obs, ts_sim, H, R, L=256, gain=1.0, x0=None, mask=None):
        qz0_mean, qz0_logstd = self.condition_on_obs(obs, ts_obs, mask)

        samples = []
        for _ in range(L):
            if x0 is not None:
                z0 = self.inverse_projector(x0)
            else:
                z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

            # IMPORTANT: refresh cache for ts_sim if different from ts_obs
            # If ts_sim differs, re-encode with ts_sim grid by interpolating obs to that grid
            # For now assume ts_obs == ts_sim in your usage.
            z_path = self.euler_maruyama(z0, obs, ts_sim, H, R, gain=gain, mask=mask)
            x_path = self.projector(z_path)
            samples.append(x_path)

        return torch.stack(samples, dim=0)


@torch.no_grad()
def posterior_mean_rmse(x_samples: torch.Tensor, x_truth: torch.Tensor) -> float:
    mu = x_samples.mean(dim=0)
    return torch.sqrt(torch.mean((mu - x_truth) ** 2)).item()

def wasserstein_1_empirical(a: torch.Tensor, b: torch.Tensor) -> float:
    a_np = a.reshape(-1).detach().cpu().numpy()
    b_np = b.reshape(-1).detach().cpu().numpy()
    if a_np.size == 0 or b_np.size == 0:
        return float("nan")
    return float(wasserstein_distance(a_np, b_np))



class NormalizedPhysicsWrapper:
    def __init__(self, raw_physics, mean, std):
        self.raw = raw_physics
        self.mean = mean.view(1, -1)
        self.std = std.view(1, -1) + 1e-8
    def f(self, t, z):
        x = z * self.std + self.mean
        dx = self.raw.f(t, x)
        return dx / self.std
    def g(self, t, z):
        x = z * self.std + self.mean
        gx = self.raw.g(t, x)
        return gx / self.std

# -----------------------------------------------------------------------------
# Particle Filter & Smoother (Mask Aware)
# -----------------------------------------------------------------------------
class BootstrapParticleFilter:
    def __init__(self, dynamics_model, obs_model, num_particles=500):
        self.dynamics = dynamics_model
        self.obs_model = obs_model
        self.num_particles = num_particles

    @torch.no_grad()
    def run(self, y_seq, ts, x0_mean, x0_std, device, mask=None):
        T, B, D = y_seq.shape
        particles = x0_mean.unsqueeze(0) + x0_std * torch.randn(self.num_particles, B, D, device=device)
        weights = torch.ones(self.num_particles, B, device=device) / self.num_particles
        history = torch.zeros(T, self.num_particles, B, D, device=device)
        history[0] = particles
        
        for t in range(1, T):
            dt = ts[t] - ts[t-1]
            t_prev = ts[t-1]
            
            # Physics Propagate
            p_flat = particles.reshape(-1, D)
            f_val = self.dynamics.f(t_prev, p_flat).reshape(self.num_particles, B, D)
            g_val = self.dynamics.g(t_prev, p_flat).reshape(self.num_particles, B, D)
            dW = torch.randn_like(particles) * torch.sqrt(dt)
            particles = particles + f_val * dt + g_val * dW
            
            # Likelihood with Mask
            # mask[t]: (B, D) or (B,)
            m_t = None
            if mask is not None:
                m_t = mask[t].unsqueeze(0).expand(self.num_particles, -1, -1) if mask.ndim==3 else mask[t].unsqueeze(0)
            
            log_lik = self.obs_model.loglik(y_seq[t], particles, mask=m_t)
            
            # Standard re-weighting
            log_weights = torch.log(weights + 1e-16) + log_lik
            max_log_w = torch.max(log_weights, dim=0, keepdim=True)[0]
            weights = torch.exp(log_weights - max_log_w)
            weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-16)
            
            indices = torch.multinomial(weights.T, self.num_particles, replacement=True).T
            new_particles = torch.zeros_like(particles)
            for b in range(B):
                new_particles[:, b, :] = particles[indices[:, b], b, :]
            particles = new_particles
            weights = torch.ones_like(weights) / self.num_particles
            history[t] = particles
        return history

class ParticleSmoother:
    def __init__(self, dynamics_model, obs_model, num_particles=200):
        self.dynamics = dynamics_model
        self.obs_model = obs_model
        self.num_particles = num_particles

    @torch.no_grad()
    def run_smoother(self, y_seq, ts, x0_mean, x0_std, device, mask=None):
        T, B, D = y_seq.shape
        trajectories = torch.zeros(T, self.num_particles, B, D, device=device)
        particles = x0_mean.unsqueeze(0) + x0_std * torch.randn(self.num_particles, B, D, device=device)
        weights = torch.ones(self.num_particles, B, device=device) / self.num_particles
        trajectories[0] = particles
        
        for t in range(1, T):
            dt = ts[t] - ts[t-1]
            t_prev = ts[t-1]
            p_flat = particles.reshape(-1, D)
            f_val = self.dynamics.f(t_prev, p_flat).reshape(self.num_particles, B, D)
            g_val = self.dynamics.g(t_prev, p_flat).reshape(self.num_particles, B, D)
            dW = torch.randn_like(particles) * torch.sqrt(dt)
            pred_particles = particles + f_val * dt + g_val * dW
            
            m_t = None
            if mask is not None:
                m_t = mask[t].unsqueeze(0).expand(self.num_particles, -1, -1) if mask.ndim==3 else mask[t].unsqueeze(0)

            log_lik = self.obs_model.loglik(y_seq[t], pred_particles, mask=m_t)
            
            log_weights = torch.log(weights + 1e-16) + log_lik
            max_log_w = torch.max(log_weights, dim=0, keepdim=True)[0]
            weights = torch.exp(log_weights - max_log_w)
            weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-16)
            
            indices = torch.multinomial(weights.T, self.num_particles, replacement=True).T
            resampled_particles = torch.zeros_like(pred_particles)
            for b in range(B):
                resampled_particles[:, b, :] = pred_particles[indices[:, b], b, :]
            
            new_traj = trajectories.clone()
            for b in range(B):
                 new_traj[:, :, b, :] = trajectories[:, indices[:, b], b, :]
            trajectories = new_traj
            trajectories[t] = resampled_particles
            particles = resampled_particles
            weights = torch.ones_like(weights) / self.num_particles
            
        return trajectories

# -----------------------------------------------------------------------------
# Metric & Plotting
# -----------------------------------------------------------------------------
def compute_w1_to_truth(samples, truth, subsample=5):
    s_np = samples.detach().cpu().numpy()
    t_np = truth.detach().cpu().numpy()
    L, T, B, D = s_np.shape
    total_w1 = 0.0
    count = 0
    for t in range(0, T, subsample):
        for b in range(min(B, 3)):
            for d in range(D):
                dist_samples = s_np[:, t, b, d]
                point_truth = [t_np[t, b, d]]
                total_w1 += wasserstein_distance(dist_samples, point_truth)
                count += 1
    return total_w1 / max(count, 1)

def plot_step(epoch, test_xs, testobs, test_xs_post, ts, train_dir, sample_idx, seed_idx, pf_samples=None, smoother_samples=None, mask=None):
    # Set global font size for better readability
    plt.rcParams.update({'font.size': 14})
    
    Dshow = min(test_xs.shape[-1], 3)
    t = ts.detach().cpu().numpy()
    
    # Increase figsize slightly to accommodate larger fonts and external legend
    fig = plt.figure(figsize=(20, 6))

    # 1. Model
    ax1 = plt.subplot(1, 3, 1)
    for d in range(Dshow):
        # We only assign labels once to avoid redundant legend entries
        l1, = plt.plot(t, test_xs[:, sample_idx, d].cpu().numpy(), color='black', linestyle='-', alpha=0.4, label='True')
        
        Samps = test_xs_post[:, :, sample_idx, d]
        mu = Samps.mean(dim=0).detach().cpu().numpy()
        lo = Samps.quantile(0.05, dim=0).detach().cpu().numpy()
        hi = Samps.quantile(0.95, dim=0).detach().cpu().numpy()
        l2, = plt.plot(t, mu, linestyle='--', color='tab:blue', label='Estimated (Mean)')
        plt.fill_between(t, lo, hi, color='tab:blue', alpha=0.2)
    plt.title('True vs Model', fontsize=18)
    plt.xlabel('t')

    # 2. Filter
    plt.subplot(1, 3, 2)
    if pf_samples is not None:
        for d in range(Dshow):
            plt.plot(t, test_xs[:, sample_idx, d].cpu().numpy(), color='black', linestyle='-', alpha=0.4)
            Samps = pf_samples[:, :, sample_idx, d] 
            mu = Samps.mean(dim=0).detach().cpu().numpy()
            lo = Samps.quantile(0.05, dim=0).detach().cpu().numpy()
            hi = Samps.quantile(0.95, dim=0).detach().cpu().numpy()
            plt.plot(t, mu, linestyle='--', color='tab:orange')
            plt.fill_between(t, lo, hi, color='tab:orange', alpha=0.2)
    plt.title('True vs Particle Filter', fontsize=18)
    plt.xlabel('t')

    # 3. Smoother
    plt.subplot(1, 3, 3)
    if smoother_samples is not None:
        for d in range(Dshow):
            plt.plot(t, test_xs[:, sample_idx, d].cpu().numpy(), color='black', linestyle='-', alpha=0.4)
            Samps = smoother_samples[:, :, sample_idx, d] 
            mu = Samps.mean(dim=0).detach().cpu().numpy()
            lo = Samps.quantile(0.05, dim=0).detach().cpu().numpy()
            hi = Samps.quantile(0.95, dim=0).detach().cpu().numpy()
            plt.plot(t, mu, linestyle='--', color='tab:green')
            plt.fill_between(t, lo, hi, color='tab:green', alpha=0.2)
    plt.title('True vs Particle Smoother', fontsize=18)
    plt.xlabel('t')

    # Create one global legend
    # bbox_to_anchor moves it outside; loc='upper center' centers it
    fig.legend(handles=[l1, l2], labels=['True State', 'Posterior Estimate (90% CI)'], 
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), fontsize=16)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make room for the legend
    os.makedirs(train_dir, exist_ok=True)
    plt.savefig(os.path.join(train_dir, f'finalrobust_l63_{epoch}_seed{seed_idx}.pdf'), bbox_inches='tight')
    plt.close()

def ensure_dir(path: str):
    if not os.path.exists(path): os.makedirs(path, exist_ok=True)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(
    data_dim: int = 15,
    obs_dim: int = 15,
    t0: float = 0.0,
    t1: float = 3.0,
    steps_train: int = 300,
    steps_test: int = 300,
    batch_size: int = 256,
    seed: int = 0,
    obs_noise_std: float = 0.15,

    missing_rate_train: float = 0.2,
    missing_rate_test: float = 0.5,

    latent_size: int = 16,
    context_size: int = 256,
    hidden_size: int = 512,
    lr_init: float = 1e-3,
    num_iters: int = 5000,
    save_every: int = 300,
    test_horizon_mult: float = 2.0,
    pf_x0_std: float = 0.1,
    l_samples: int = 256,

    train_dir: str = "./dump/l96/",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    ensure_dir(train_dir)

    # 1. Data (xs_train is normalized)
    xs_train, ts_train, train_mean, train_std = make_dataset(t0, t1, batch_size, 0.01, train_dir, device, steps=steps_train, D=data_dim)
    if train_mean is not None:
        train_mean = train_mean.to(device)
        train_std = train_std.to(device)

    t1_test = t1 * test_horizon_mult
    steps_test_ext = int(steps_test * test_horizon_mult)
    xs_test, ts_test = make_testdataset(t0, t1_test, batch_size, 0.01, train_dir, device, steps=steps_test_ext, D=data_dim)

    # 2. Observations & Mask Generation
    ys_train = torch.atan(xs_train) + obs_noise_std * torch.randn_like(xs_train)
    ys_test = torch.atan(xs_test) + obs_noise_std * torch.randn_like(xs_test)

    mask_train = torch.bernoulli(torch.full_like(ys_train, 1 - missing_rate_train))
    mask_test = torch.bernoulli(torch.full_like(ys_test, 1 - missing_rate_test))

    ys_train = ys_train * mask_train
    ys_test = ys_test * mask_test

    H = torch.eye(data_dim, device=device)
    R = (obs_noise_std ** 2) * torch.eye(data_dim, device=device)

    # 3. Model
    model = LatentSDE(data_dim, latent_size, context_size, hidden_size, steps_train).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_init)

    # 4. Train — load latest checkpoint if one exists
    import glob as _glob
    existing = sorted(_glob.glob(os.path.join(train_dir, "model_step_*.pth")))
    if existing:
        modelpath = existing[-1]
        model.load_state_dict(torch.load(modelpath, map_location=device))
        logging.info(f"Loaded existing model from {modelpath}")
    else:
        import time as _time
        for step in tqdm.tqdm(range(1, num_iters + 1)):
            model.zero_grad()
            log_pxs, log_py, log_kl, _ = model(
                xs_train, ys_train, ts_train, obs_noise_std, H, R, mask=mask_train
            )
            loss = -(log_pxs) + log_kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % save_every == 0:
                logging.info(f"Step {step} | Loss: {loss.item():.3f}")
                with open(os.path.join(train_dir, "training_log.txt"), "a") as _logf:
                    _logf.write(f"[{_time.strftime('%Y-%m-%d %H:%M:%S')}] Step {step} | Loss: {loss.item():.6f}\n")
                plot_step(
                    step, xs_train, ys_train, model.sample_posterior_paths_em(
                        ys_train, ts_train, ts_train, H, R, L=l_samples, gain=1.0, x0=xs_train[0], mask=mask_train
                    ), ts_train, train_dir, sample_idx=0, seed_idx=seed
                )
                model_path = os.path.join(train_dir, f'model_step_{step}.pth')
                torch.save(model.state_dict(), model_path)
                logging.info(f"Saved model to {model_path}")

    # --------------------------------------------------------------------------
    # 5. FINAL EVALUATION: Normalized Comparison
    # --------------------------------------------------------------------------
    logging.info("Starting Final Evaluation...")
    with torch.no_grad():
        # A. Latent SDE
        x_samples_model = model.sample_posterior_paths_em(
            ys_test, ts_test, ts_test, H, R, L=l_samples, gain=1.0, x0=xs_test[0], mask=mask_test
        )

        # B. PF/Smoother Setup
        raw_physics = StochasticLorenz96()#.to(device)
        norm_physics = NormalizedPhysicsWrapper(raw_physics, train_mean, train_std)
        pf_obs = LorenzAtanObsModel(obs_dim, data_dim, obs_noise_std)
        
        pf_filter = BootstrapParticleFilter(norm_physics, pf_obs, num_particles=512)
        pf_smoother = ParticleSmoother(norm_physics, pf_obs, num_particles=512)
        
        x0_norm = xs_test[0]
        x0_std_norm = torch.tensor(float(pf_x0_std), device=device)
        
        # C. Run Filter (pass mask)
        logging.info("Running Particle Filter...")
        pf_hist_norm = pf_filter.run(ys_test, ts_test, x0_norm, x0_std_norm, device, mask=mask_test)
        pf_samples_norm = pf_hist_norm.permute(1, 0, 2, 3) 
        
        # D. Run Smoother (pass mask)
        logging.info("Running Particle Smoother...")
        sm_hist_norm = pf_smoother.run_smoother(ys_test, ts_test, x0_norm, x0_std_norm, device, mask=mask_test)
        sm_samples_norm = sm_hist_norm.permute(1, 0, 2, 3)
        
        # E. Metrics
        rmse_model = posterior_mean_rmse(x_samples_model, xs_test)
        rmse_pf = posterior_mean_rmse(pf_samples_norm, xs_test)
        rmse_sm = posterior_mean_rmse(sm_samples_norm, xs_test)
        
        w1_model = compute_w1_to_truth(x_samples_model, xs_test)
        w1_pf = compute_w1_to_truth(pf_samples_norm, xs_test)
        w1_sm = compute_w1_to_truth(sm_samples_norm, xs_test)
        
        logging.info("-" * 40)
        logging.info(f"EVALUATION METRICS (Missing Rate: {missing_rate_test})")
        logging.info("-" * 40)
        logging.info(f"RMSE (Model)    : {rmse_model:.4f}")
        logging.info(f"RMSE (Filter)   : {rmse_pf:.4f}")
        logging.info(f"RMSE (Smoother) : {rmse_sm:.4f}")
        logging.info("-" * 40)
        logging.info(f"W1 (Model)      : {w1_model:.4f}")
        logging.info(f"W1 (Filter)     : {w1_pf:.4f}")
        logging.info(f"W1 (Smoother)   : {w1_sm:.4f}")
        logging.info("-" * 40)

        import time as _time
        results_path = os.path.join(train_dir, "results.txt")
        with open(results_path, "a") as _f:
            _f.write(f"=== {_time.strftime('%Y-%m-%d %H:%M:%S')} | seed={seed} | train_miss={missing_rate_train} | test_miss={missing_rate_test} ===\n")
            _f.write(f"RMSE  Ours={rmse_model:.4f}  PF={rmse_pf:.4f}  PG={rmse_sm:.4f}\n")
            _f.write(f"W1    Ours={w1_model:.4f}  PF={w1_pf:.4f}  PG={w1_sm:.4f}\n")
            _f.write("\n")
        logging.info(f"Results appended to {results_path}")
        with open(os.path.join(train_dir, "training_log.txt"), "a") as _logf:
            _logf.write(f"\n=== EVALUATION [{_time.strftime('%Y-%m-%d %H:%M:%S')}] | seed={seed} | test_miss={missing_rate_test} ===\n")
            _logf.write(f"RMSE  Ours={rmse_model:.4f}  PF={rmse_pf:.4f}  PG={rmse_sm:.4f}\n")
            _logf.write(f"W1    Ours={w1_model:.4f}  PF={w1_pf:.4f}  PG={w1_sm:.4f}\n\n")
        
        plot_step(
            f"final_benchmark_missing", 
            xs_test, ys_test, x_samples_model, ts_test, train_dir, 
            sample_idx=0, 
            seed_idx = seed,
            pf_samples=pf_samples_norm, 
            smoother_samples=sm_samples_norm
        )
        logging.info(f"Saved 3-subplot benchmark plot to {train_dir}")

if __name__ == "__main__":
    fire.Fire(main)

