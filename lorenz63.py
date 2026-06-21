import glob
import logging
import os
import math
import re
from typing import Dict, Tuple, Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from scipy.stats import wasserstein_distance
from torch import nn, optim
from torch.distributions import Normal

import torchsde
from sdeint_obs import sdeint_obs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# =========================
# Utilities
# =========================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def wasserstein_1_empirical(a: torch.Tensor, b: torch.Tensor) -> float:
    a_np = a.reshape(-1).detach().cpu().numpy()
    b_np = b.reshape(-1).detach().cpu().numpy()
    if a_np.size == 0 or b_np.size == 0:
        return float("nan")
    return float(wasserstein_distance(a_np, b_np))


@torch.no_grad()
def posterior_mean_rmse(x_samples: torch.Tensor, x_truth: torch.Tensor) -> float:
    mu = x_samples.mean(dim=0)
    return torch.sqrt(torch.mean((mu - x_truth) ** 2)).item()


@torch.no_grad()
def w1_to_truth_per_time(samples: torch.Tensor, truth: torch.Tensor,
                         subsample_t: int = 5, max_b: int = 8) -> float:
    """
    Per-time W1 between the posterior samples and the ground-truth path.
    samples: (L, T, B, D)   truth: (T, B, D)
    """
    s = samples.detach().cpu().numpy()
    x = truth.detach().cpu().numpy()
    L, T, B, D = s.shape
    Bcap = min(B, max_b)
    tot, n = 0.0, 0
    for t in range(0, T, subsample_t):
        for b in range(Bcap):
            for d in range(D):
                tot += wasserstein_distance(s[:, t, b, d], [x[t, b, d]])
                n += 1
    return tot / max(n, 1)


@torch.no_grad()
def w1_pooled(samples: torch.Tensor, truth: torch.Tensor) -> float:
    return wasserstein_1_empirical(samples.reshape(-1), truth.reshape(-1))


def _mean_se(vals) -> Tuple[float, float]:
    a = np.asarray(vals, dtype=np.float64)
    n = a.size
    if n == 0:
        return float("nan"), float("nan")
    m = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return m, se


# =========================
# Stochastic Lorenz-63
# =========================
class StochasticLorenz63(nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        theta1: float = 10.0,
        theta2: float = 28.0,
        theta3: float = 8.0 / 3.0,
        beta1: float = 0.1,
        beta2: float = 0.28,
        beta3: float = 0.3,
    ):
        super().__init__()
        self.theta1 = float(theta1)
        self.theta2 = float(theta2)
        self.theta3 = float(theta3)
        self.register_buffer("beta", torch.tensor([beta1, beta2, beta3], dtype=torch.float32))

    def f(self, t, y):
        x1, x2, x3 = y[..., 0], y[..., 1], y[..., 2]
        dx1 = self.theta1 * (x2 - x1)
        dx2 = x1 * (self.theta2 - x3) - x2
        dx3 = x1 * x2 - self.theta3 * x3
        return torch.stack([dx1, dx2, dx3], dim=-1)

    def g(self, t, y):
        if self.beta.device != y.device:
            self.beta = self.beta.to(y.device)
        return y * self.beta

    @torch.no_grad()
    def sample(self, x0, ts, dt: float = 1e-3, method: str = "euler"):
        return torchsde.sdeint(self, x0, ts, dt=dt, method=method)

    @torch.no_grad()
    def euler_step(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2 = x.reshape(-1, 3)
        dt_ = dt.to(device=x2.device, dtype=x2.dtype)
        drift = self.f(None, x2)
        diff = self.g(None, x2)
        sqrt_dt = torch.sqrt(dt_.clamp_min(1e-12))
        noise = diff * torch.randn_like(x2) * sqrt_dt
        x_next = x2 + drift * dt_ + noise
        return x_next.reshape(orig_shape)


# =========================
# Observation model on normalized state:
# y = atan(x_norm) + noise
# =========================
class LorenzAtanObsModel(nn.Module):
    def __init__(self, obs_sigma: float = 0.05):
        super().__init__()
        self.obs_sigma = float(obs_sigma)

    def mean(self, x_norm: torch.Tensor) -> torch.Tensor:
        return torch.atan(x_norm)

    def log_prob(self, y: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
        mu = self.mean(x_norm)
        dist = Normal(loc=mu, scale=self.obs_sigma)
        return dist.log_prob(y).sum(dim=-1)

    @torch.no_grad()
    def sample(self, x_norm: torch.Tensor) -> torch.Tensor:
        return self.mean(x_norm) + self.obs_sigma * torch.randn_like(x_norm)


# =========================
# Dataset
# =========================
@torch.no_grad()
def make_dataset(
    t0: float,
    t1: float,
    steps: int,
    batch_size: int,
    device: torch.device,
    cache_path: str,
    l63_params: Dict,
    sim_dt: float = 1e-3,
    sim_method: str = "euler",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if os.path.exists(cache_path):
        d = torch.load(cache_path, map_location=device)
        logging.warning(f"Loaded dataset: {cache_path}")
        return d["xs_raw"].to(device), d["ts"].to(device)

    ts = torch.linspace(t0, t1, steps=steps, device=device)
    x0 = torch.randn(batch_size, 3, device=device)
    sde = StochasticLorenz63(**l63_params)
    xs_raw = sde.sample(x0, ts, dt=sim_dt, method=sim_method)

    ensure_dir(os.path.dirname(cache_path))
    torch.save({"xs_raw": xs_raw.detach().cpu(), "ts": ts.detach().cpu()}, cache_path)
    logging.warning(f"Stored dataset: {cache_path}")
    return xs_raw, ts


@torch.no_grad()
def compute_norm_stats(xs_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = xs_raw.mean(dim=(0, 1))
    std = xs_raw.std(dim=(0, 1)).clamp_min(1e-8)
    return mean, std


@torch.no_grad()
def normalize_x(xs_raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (xs_raw - mean.view(1, 1, -1)) / std.view(1, 1, -1)


# =========================
# Causal encoders
# =========================
class EncoderCausal(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = torch.nan_to_num(inp, nan=0.0)
        out, _ = self.gru(inp)
        return self.lin(out)


class ObsEncoderGRU(nn.Module):
    """
    GRU observation encoder
    Per-step input features: [y * mask, mask, time_encoding(t)].
      - Concatenating the mask lets the encoder distinguish a genuinely observed
        value near zero from a missing entry that was zero-filled.
      - A learned time encoding makes it aware of (possibly irregular) timestamps.
    """
    def __init__(self, input_size, hidden_size, output_size, causal=True, use_time_encoding=True):
        super().__init__()
        self.use_time_encoding = use_time_encoding
        self.causal = causal
        actual_input = input_size * 2 + (32 if use_time_encoding else 0)
        if use_time_encoding:
            self.time_encoder = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32))
        gru_hidden = hidden_size if causal else hidden_size // 2
        self.gru = nn.GRU(
            input_size=actual_input,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=not causal,
        )
        self.output_proj = nn.Linear(hidden_size if causal else gru_hidden * 2, output_size)

    def _prepare_input(self, inp, mask, times):
        B, T, _ = inp.shape
        inp = torch.nan_to_num(inp, nan=0.0)
        if mask is None:
            mask = torch.ones_like(inp)
        feats = [inp * mask, mask]
        if self.use_time_encoding and times is not None:
            t_enc = self.time_encoder(times.view(T, 1))      # (T,32)
            t_enc = t_enc.unsqueeze(0).expand(B, T, -1)       # (B,T,32)
            feats.append(t_enc)
        return torch.cat(feats, dim=-1)

    def forward(self, inp, mask=None, times=None):
        # Global context for q(z0 | .): masked mean-pool over time -> (B, output_size)
        combined = self._prepare_input(inp, mask, times)
        out, _ = self.gru(combined)
        if mask is not None:
            mask_t = (mask.sum(dim=-1) > 0).float().unsqueeze(-1)   # (B,T,1)
            pooled = (out * mask_t).sum(dim=1) / mask_t.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = out.mean(dim=1)
        return self.output_proj(pooled)

    def forward_per_time(self, inp, mask=None, times=None, causal: bool = True):
        # Per-time context -> (T, B, output_size); causal via the unidirectional GRU.
        combined = self._prepare_input(inp, mask, times)
        out, _ = self.gru(combined)
        per_time = self.output_proj(out)                      # (B,T,output_size)
        return per_time.permute(1, 0, 2)                      # (T,B,output_size)


# =========================
# Latent SDE
# =========================
class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(
        self,
        data_size: int,
        latent_size: int,
        context_size: int,
        hidden_size: int,
        ctxobs_size: int = 16,
        num_heads: int = 2,
        time_d: int = 32,
        obs_noise_std: float = 0.15,
        use_ctrl: bool = True,
    ):
        super().__init__()
        self.data_dim = int(data_size)
        self.latent_size = int(latent_size)
        self.context_size = int(context_size)
        self.ctxobs_size = int(ctxobs_size)
        self._obs_noise_std = float(obs_noise_std)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_ctrl = bool(use_ctrl)

        self.encoder = EncoderCausal(data_size, hidden_size, context_size)
        self.obs_encoder = ObsEncoderGRU(data_size, hidden_size, ctxobs_size, causal=True, use_time_encoding=True)

        self.qz0_net = nn.Linear(ctxobs_size, latent_size * 2)

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

        self.g_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden_size), nn.Softplus(), nn.Linear(hidden_size, 1), nn.Sigmoid())
            for _ in range(latent_size)
        ])

        self.projector = nn.Linear(latent_size, data_size)
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None
        self._obs_ctx_seq = None
        self._ts_cached = None
        self._K = None
        self._V = None

        self.time_d = time_d
        self.time_feat = nn.Sequential(nn.Linear(1, time_d), nn.Tanh(), nn.Linear(time_d, time_d))
        self.time_key = nn.Linear(ctxobs_size, time_d)
        self.time_val = nn.Linear(ctxobs_size, ctxobs_size)
        self.mix_gate = nn.Sequential(nn.Linear(ctxobs_size * 2, 64), nn.Tanh(), nn.Linear(64, 1))

    def _encode_x_context(self, xs, ts):
        xs_clean = torch.nan_to_num(xs, nan=0.0)
        x_ctx_rev = self.encoder(torch.flip(xs_clean, dims=(0,)))
        return torch.flip(x_ctx_rev, dims=(0,))

    def _encode_obs_context(self, obs, ts, mask=None):
        # obs: (T,B,D). Keep a full per-coordinate mask so the GRU sees it as a feature.
        if mask is None:
            mask = torch.ones_like(obs)
        elif mask.ndim == 2:                                  # (T,B) -> (T,B,D)
            mask = mask.unsqueeze(-1).expand_as(obs)

        obs_b = obs.permute(1, 0, 2)                          # (B,T,D)
        mask_b = mask.permute(1, 0, 2)                        # (B,T,D)

        # Per-time causal context (T,B,Cobs) and whole-window global context (B,Cobs).
        obs_ctx_seq = self.obs_encoder.forward_per_time(
            obs_b, mask=mask_b, times=ts, causal=self.obs_encoder.causal
        )
        obs_ctx_global = self.obs_encoder(obs_b, mask=mask_b, times=ts)

        self._obs_ctx_seq = obs_ctx_seq                       # already (T,B,Cobs)
        self._ts_cached = ts
        self._K = self.time_key(self._obs_ctx_seq)
        self._V = self.time_val(self._obs_ctx_seq)
        return obs_ctx_global

    @torch.no_grad()
    def condition_on_obs(self, obs, ts, mask=None):
        obs_ctx_global = self._encode_obs_context(obs, ts, mask)
        return self.qz0_net(obs_ctx_global).chunk(2, dim=1)

    @torch.no_grad()
    def sample_prior_z0(self, batch_size: int) -> torch.Tensor:
        eps = torch.randn(batch_size, self.latent_size, device=self.pz0_mean.device, dtype=self.pz0_mean.dtype)
        return self.pz0_mean + self.pz0_logstd.exp() * eps

    def _embed_obs_at_time(self, t: torch.Tensor) -> torch.Tensor:
        ts = self._ts_cached
        E = self._obs_ctx_seq
        K = self._K
        V = self._V

        T, B, _ = E.shape
        Dk = K.size(-1)

        i1 = int(torch.searchsorted(ts, t, right=True).item())
        i0 = max(i1 - 1, 0)
        i1 = min(i1, T - 1)

        denom = (ts[i1] - ts[i0]).clamp_min(1e-8)
        w_loc = ((t - ts[i0]) / denom).clamp(0, 1)
        e_lin = (1.0 - w_loc) * E[i0] + w_loc * E[i1]

        past_len = i1 + 1
        Kp = K[:past_len]
        Vp = V[:past_len]

        t_in = t.expand(B, 1)
        q = self.time_feat(t_in)

        scores = torch.bmm(q.unsqueeze(1), Kp.permute(1, 2, 0)).squeeze(1) / math.sqrt(Dk)
        w_all = torch.softmax(scores, dim=-1)
        e_attn = torch.bmm(w_all.unsqueeze(1), Vp.permute(1, 0, 2)).squeeze(1)

        alpha = torch.sigmoid(self.mix_gate(torch.cat([e_lin, e_attn], dim=1)))
        return (1.0 - alpha) * e_lin + alpha * e_attn

    def f(self, t, z, obs):
        ts, x_ctx_seq = self._ctx
        i = int(torch.searchsorted(ts, t, right=True).item()) - 1
        i = max(0, min(i, len(ts) - 1))
        e_t = self._embed_obs_at_time(t)
        prior = self.f_net(torch.cat((z, x_ctx_seq[i], e_t), dim=1))
        return torch.nan_to_num(prior, nan=0.0, posinf=1e6, neginf=-1e6)

    def h(self, t, z, obs):
        e_t = self._embed_obs_at_time(t)
        out = self.h_net(torch.cat((z, e_t), dim=1))
        # out = self.h_net(z)
        if self.use_ctrl:
            ctrl = self.control_net(torch.cat((z, e_t), dim=1))
            out = out + self.g(t, z) * ctrl
        return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

    def g(self, t, z):
        chunks = torch.split(z, 1, dim=1)
        out = [g_net(c) for g_net, c in zip(self.g_nets, chunks)]
        return torch.nan_to_num(torch.cat(out, dim=1), nan=0.0, posinf=1e6, neginf=-1e6)

    def obs_loglik(self, x_seq, y_seq, R, mask=None):
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
        x_ctx_seq = self._encode_x_context(xs, ts)
        obs_ctx_global = self._encode_obs_context(obs, ts, mask)
        self._ctx = (ts, x_ctx_seq)

        qz0_mean, qz0_logstd = self.qz0_net(obs_ctx_global).chunk(2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)

        if adjoint:
            zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=1e-2, logqp=True, method=method)
        else:
            zs, log_ratio = sdeint_obs(self, z0, obs, ts, dt=1e-2, logqp=True, method=method)

        x_hat = self.projector(zs)

        log_pxs = torch.tensor(0.0, device=x_hat.device)
        if xs is not None and x_recon_weight > 0:
            xs_dist = Normal(x_hat, noise_std)
            log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0) * x_recon_weight

        log_py = self.obs_loglik(x_hat, obs, R, mask) * y_ll_weight

        qz0 = Normal(qz0_mean, qz0_logstd.exp())
        pz0 = Normal(self.pz0_mean, self.pz0_logstd.exp())
        kl0 = torch.distributions.kl_divergence(qz0, pz0).sum(1).mean()
        kl_path = log_ratio.sum(0).mean()

        return log_pxs, log_py, (kl0 + kl_path), x_hat

    @torch.no_grad()
    def inverse_projector(self, xs: torch.Tensor) -> torch.Tensor:
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
            t_prev = ts[i - 1]
            dt = (ts[i] - t_prev).clamp_min(1e-8)
            sqrt_dt = torch.sqrt(dt)

            y_i = obs[i] if i < obs.size(0) else obs[-1]
            mask_i = mask[i] if (mask is not None and i < mask.size(0)) else None

            drift = self.h(t_prev, z[i - 1], obs)
            like = self.likelihood_force(z[i - 1], y_i, H, R, mask_i)
            g_diag = self.g(t_prev, z[i - 1])
            dW = torch.randn_like(z0) * sqrt_dt

            z[i] = z[i - 1] + (drift + gain * like) * dt + g_diag * dW
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

            z_path = self.euler_maruyama(z0, obs, ts_sim, H, R, gain=gain, mask=mask)
            x_path = self.projector(z_path)
            samples.append(x_path)

        return torch.stack(samples, dim=0)


# =========================
# PF / Smoother baselines in normalized space
# =========================
class NormalizedPhysicsWrapper:
    def __init__(self, raw_physics: StochasticLorenz63, mean: torch.Tensor, std: torch.Tensor):
        self.raw = raw_physics
        self.mean = mean.view(1, -1)
        self.std = std.view(1, -1).clamp_min(1e-8)

    def f(self, t, z):
        x = z * self.std + self.mean
        dx = self.raw.f(t, x)
        return dx / self.std

    def g(self, t, z):
        x = z * self.std + self.mean
        gx = self.raw.g(t, x)
        return gx / self.std


class BootstrapParticleFilter:
    def __init__(self, dynamics_model, obs_model, num_particles=512):
        self.dynamics = dynamics_model
        self.obs_model = obs_model
        self.num_particles = int(num_particles)

    @torch.no_grad()
    def run(self, y_seq, ts, x0_mean, x0_std, device, mask=None):
        T, B, D = y_seq.shape
        N = self.num_particles

        particles = x0_mean.unsqueeze(0) + x0_std * torch.randn(N, B, D, device=device)
        weights = torch.ones(N, B, device=device) / N
        history = torch.zeros(T, N, B, D, device=device)
        history[0] = particles

        for t in range(1, T):
            dt = ts[t] - ts[t - 1]
            p_flat = particles.reshape(-1, D)
            drift = self.dynamics.f(ts[t - 1], p_flat).reshape(N, B, D)
            diff = self.dynamics.g(ts[t - 1], p_flat).reshape(N, B, D)
            dW = torch.randn_like(particles) * torch.sqrt(dt.clamp_min(1e-12))
            particles = particles + drift * dt + diff * dW

            y_t = y_seq[t].unsqueeze(0).expand(N, -1, -1)
            mean = torch.atan(particles)
            resid = y_t - mean

            if mask is not None:
                resid = resid * mask[t].unsqueeze(0)

            var = self.obs_model.obs_sigma ** 2
            log_lik = -0.5 * (resid ** 2).sum(dim=-1) / var
            log_lik = log_lik - 0.5 * D * math.log(2 * math.pi * var)

            logw = torch.log(weights + 1e-16) + log_lik
            max_logw = torch.max(logw, dim=0, keepdim=True)[0]
            weights = torch.exp(logw - max_logw)
            weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-16)

            idx = torch.multinomial(weights.T, N, replacement=True).T
            new_particles = torch.zeros_like(particles)
            for b in range(B):
                new_particles[:, b, :] = particles[idx[:, b], b, :]
            particles = new_particles
            weights = torch.ones_like(weights) / N
            history[t] = particles

        return history


class ParticleSmoother:
    def __init__(self, dynamics_model, obs_model, num_particles=512):
        self.dynamics = dynamics_model
        self.obs_model = obs_model
        self.num_particles = int(num_particles)

    @torch.no_grad()
    def run_smoother(self, y_seq, ts, x0_mean, x0_std, device, mask=None):
        T, B, D = y_seq.shape
        N = self.num_particles

        trajectories = torch.zeros(T, N, B, D, device=device)
        particles = x0_mean.unsqueeze(0) + x0_std * torch.randn(N, B, D, device=device)
        weights = torch.ones(N, B, device=device) / N
        trajectories[0] = particles

        for t in range(1, T):
            dt = ts[t] - ts[t - 1]
            p_flat = particles.reshape(-1, D)
            drift = self.dynamics.f(ts[t - 1], p_flat).reshape(N, B, D)
            diff = self.dynamics.g(ts[t - 1], p_flat).reshape(N, B, D)
            dW = torch.randn_like(particles) * torch.sqrt(dt.clamp_min(1e-12))
            pred_particles = particles + drift * dt + diff * dW

            y_t = y_seq[t].unsqueeze(0).expand(N, -1, -1)
            mean = torch.atan(pred_particles)
            resid = y_t - mean

            if mask is not None:
                resid = resid * mask[t].unsqueeze(0)

            var = self.obs_model.obs_sigma ** 2
            log_lik = -0.5 * (resid ** 2).sum(dim=-1) / var
            log_lik = log_lik - 0.5 * D * math.log(2 * math.pi * var)

            logw = torch.log(weights + 1e-16) + log_lik
            max_logw = torch.max(logw, dim=0, keepdim=True)[0]
            weights = torch.exp(logw - max_logw)
            weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-16)

            idx = torch.multinomial(weights.T, N, replacement=True).T

            resampled_particles = torch.zeros_like(pred_particles)
            new_traj = trajectories.clone()
            for b in range(B):
                resampled_particles[:, b, :] = pred_particles[idx[:, b], b, :]
                new_traj[:, :, b, :] = trajectories[:, idx[:, b], b, :]
            trajectories = new_traj
            trajectories[t] = resampled_particles
            particles = resampled_particles
            weights = torch.ones_like(weights) / N

        return trajectories


# =========================
# Plotting
# =========================
def plot_model_ci(epoch, test_xs, test_xs_post, ts, train_dir, sample_idx, seed_idx, title="Latent SDE"):
    plt.rcParams.update({'font.size': 13})
    Dshow = min(test_xs.shape[-1], 3)
    t = ts.detach().cpu().numpy()
    fig = plt.figure(figsize=(9, 4.5))

    for d in range(Dshow):
        l1, = plt.plot(t, test_xs[:, sample_idx, d].cpu().numpy(), color='black', linestyle='-', alpha=0.45, label='True')
        samps = test_xs_post[:, :, sample_idx, d]
        mu = samps.mean(dim=0).detach().cpu().numpy()
        lo = samps.quantile(0.05, dim=0).detach().cpu().numpy()
        hi = samps.quantile(0.95, dim=0).detach().cpu().numpy()
        l2, = plt.plot(t, mu, linestyle='--', linewidth=1.6, label='Estimated (Mean)')
        plt.fill_between(t, lo, hi, alpha=0.2)

    plt.title(title)
    fig.legend(handles=[l1, l2], labels=['True State', 'Posterior Estimate (90% CI)'],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    ensure_dir(train_dir)
    plt.savefig(os.path.join(train_dir, f'lor63_{epoch}_seed{seed_idx}.pdf'), bbox_inches='tight')
    plt.close()


def plot_benchmark_triptych(
    out_path: str,
    ts: torch.Tensor,
    x_true: torch.Tensor,
    x_model: torch.Tensor,
    x_pf: torch.Tensor,
    x_sm: torch.Tensor,
    title_left: str = "True vs Model",
    title_mid: str = "True vs Particle Filter",
    title_right: str = "True vs Smoother",
):
    plt.rcParams.update({"font.size": 12})
    t = ts.detach().cpu().numpy()
    Dshow = min(x_true.shape[-1], 3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    panels = [
        (axes[0], x_model, title_left),
        (axes[1], x_pf, title_mid),
        (axes[2], x_sm, title_right),
    ]

    for ax, x_est, ttl in panels:
        for d in range(Dshow):
            ax.plot(t, x_true[:, d].detach().cpu().numpy(), linewidth=1.7, alpha=0.8, label=f"x{d+1} True")
            ax.plot(t, x_est[:, d].detach().cpu().numpy(), linestyle="--", linewidth=1.7, alpha=0.9, label=f"x{d+1} Model" if ttl == title_left else f"x{d+1} Est")
        ax.set_title(ttl)
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.25)

    axes[0].legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_ablation_compare(
    out_path: str,
    ts: torch.Tensor,
    x_true: torch.Tensor,
    x_full: torch.Tensor,
    x_noctrl: torch.Tensor,
    title_left: str = "Latent SDE (with ctrl)",
    title_right: str = "Latent SDE (no ctrl)",
):
    plt.rcParams.update({"font.size": 12})
    t = ts.detach().cpu().numpy()
    Dshow = min(x_true.shape[-1], 3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for ax, est, ttl in [(axes[0], x_full, title_left), (axes[1], x_noctrl, title_right)]:
        for d in range(Dshow):
            ax.plot(t, x_true[:, d].detach().cpu().numpy(), linewidth=1.8, alpha=0.8, label=f"x{d+1} True")
            ax.plot(t, est[:, d].detach().cpu().numpy(), linestyle="--", linewidth=1.8, alpha=0.9, label=f"x{d+1} Est")
        ax.set_title(ttl)
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.25)

    axes[0].legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# =========================
# Training helper
# =========================
def train_or_load_model(
    model: LatentSDE,
    model_path: str,
    optimizer: optim.Optimizer,
    xs_train: torch.Tensor,
    ys_train: torch.Tensor,
    ts_train: torch.Tensor,
    H: torch.Tensor,
    R: torch.Tensor,
    mask_train: torch.Tensor,
    obs_noise_std: float,
    num_iters: int,
    save_every: int,
    train_dir: str,
    seed: int,
    plot_prefix: str,
):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=xs_train.device))
        logging.info(f"Loaded existing model from {model_path}")
        return

    for step in tqdm.tqdm(range(1, num_iters + 1)):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        log_pxs, log_py, log_kl, _ = model(
            xs_train, ys_train, ts_train, obs_noise_std, H, R, mask=mask_train
        )

        loss = -(log_pxs) + log_kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % save_every == 0:
            logging.info(
                f"[{plot_prefix}] Step {step} | Loss: {loss.item():.3f} | "
                f"log_pxs={float(log_pxs):.3f} | log_py={float(log_py):.3f} | kl={float(log_kl):.3f}"
            )

            with torch.no_grad():
                x_samples = model.sample_posterior_paths_em(
                    ys_train, ts_train, ts_train, H, R, L=128, gain=1.0, mask=mask_train
                )
                plot_model_ci(
                    epoch=f"{plot_prefix}_{step}",
                    test_xs=xs_train,
                    test_xs_post=x_samples,
                    ts=ts_train,
                    train_dir=train_dir,
                    sample_idx=0,
                    seed_idx=seed,
                    title=f"{plot_prefix} train sample"
                )

            torch.save(model.state_dict(), model_path)
            logging.info(f"Saved model to {model_path}")


# =========================
# Inference-only multi-seed evaluation
# =========================
def _discover_seed_dirs(base_dir: str) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    """Auto-discover seed subdirectories under base_dir named seed<N>."""
    dirs, seeds = [], []
    for d in sorted(glob.glob(os.path.join(base_dir, "seed*"))):
        m = re.search(r"seed(\d+)$", d)
        if m and os.path.isdir(d):
            dirs.append(d + os.sep)
            seeds.append(int(m.group(1)))
    return tuple(dirs), tuple(seeds)


def _build_latent_sde(data_dim, latent_size, context_size, hidden_size,
                      ctxobs_size, obs_noise_std, use_ctrl, device):
    return LatentSDE(
        data_size=data_dim,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
        ctxobs_size=ctxobs_size,
        num_heads=2,
        time_d=32,
        obs_noise_std=obs_noise_std,
        use_ctrl=use_ctrl,
    ).to(device)


def evaluate_seeds(
    base_dir: str = "./dump/l63/",
    seed_dirs: Tuple[str, ...] = (),
    seeds: Tuple[int, ...] = (),
    data_dim: int = 3,
    obs_noise_std: float = 0.15,
    missing_rate_test: float = 0.0,
    latent_size: int = 4,
    context_size: int = 128,
    hidden_size: int = 256,
    ctxobs_size: int = 64,
    theta1: float = 10.0,
    theta2: float = 28.0,
    theta3: float = 8.0 / 3.0,
    beta1: float = 0.1,
    beta2: float = 0.28,
    beta3: float = 0.3,
    pf_num_particles: int = 512,
    pf_x0_std: float = 0.10,
    l_samples: int = 64,
    gain: float = 1.0,
    subsample_t: int = 5,
    max_b: int = 8,
    out_path: str = "./dump/l63_eval_summary.txt",
):
    """
    Inference-only evaluation over pre-trained seeds.

    For each seed directory we load the cached test data and the two checkpoints
    (model_ctrl.pth, model_noctrl.pth), regenerate the noisy observations
    deterministically, run inference for the latent SDE (ctrl / no-ctrl) and the
    particle baselines (PF / PG, given the true Lorenz dynamics), and report RMSE,
    the corrected per-time W1-to-truth, and the legacy pooled W1. Results are
    aggregated as mean +/- standard error over seeds, with a paired control-term
    ablation. No training is performed.
    """
    if isinstance(seed_dirs, str):
        seed_dirs = (seed_dirs,)
    if isinstance(seeds, int):
        seeds = (seeds,)
    if not seed_dirs:
        seed_dirs, seeds = _discover_seed_dirs(base_dir)
        if not seed_dirs:
            logging.error(f"No seed*/  subdirectories found under {base_dir}. Train first.")
            return None
    assert len(seed_dirs) == len(seeds), "seed_dirs and seeds must have equal length"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l63_params = dict(theta1=theta1, theta2=theta2, theta3=theta3,
                      beta1=beta1, beta2=beta2, beta3=beta3)

    methods = ["ours_ctrl", "ours_noctrl", "pf", "pg"]
    agg = {m: {"rmse": [], "w1t": [], "w1p": []} for m in methods}
    per_seed = []

    for sd, seed in zip(seed_dirs, seeds):
        ckpt_ctrl = os.path.join(sd, "model_ctrl.pth")
        ckpt_noctrl = os.path.join(sd, "model_noctrl.pth")
        train_cache = os.path.join(sd, "l63_train_raw.pth")
        test_cache = os.path.join(sd, "l63_test_raw.pth")
        missing = [p for p in (ckpt_ctrl, ckpt_noctrl, train_cache, test_cache) if not os.path.exists(p)]
        if missing:
            logging.warning(f"[seed {seed}] missing {missing}; skipping {sd}.")
            continue

        set_seed(seed)

        # --- load cached data ---
        d_tr = torch.load(train_cache, map_location=device)
        xs_train_raw = d_tr["xs_raw"].to(device)
        d_te = torch.load(test_cache, map_location=device)
        xs_test_raw = d_te["xs_raw"].to(device)
        ts_test = d_te["ts"].to(device)

        train_mean, train_std = compute_norm_stats(xs_train_raw)
        train_mean = train_mean.to(device)
        train_std = train_std.to(device)
        xs_test = normalize_x(xs_test_raw, train_mean, train_std)

        # --- noisy observations + mask (deterministic given seed) ---
        obs_model = LorenzAtanObsModel(obs_sigma=obs_noise_std).to(device)
        ys_test = obs_model.sample(xs_test)
        mask_test = torch.bernoulli(torch.full_like(ys_test, 1 - missing_rate_test))
        ys_test = ys_test * mask_test

        H = torch.eye(data_dim, device=device)
        R = (obs_noise_std ** 2) * torch.eye(data_dim, device=device)

        # --- load both models ---
        model = _build_latent_sde(data_dim, latent_size, context_size, hidden_size,
                                  ctxobs_size, obs_noise_std, True, device)
        model_noctrl = _build_latent_sde(data_dim, latent_size, context_size, hidden_size,
                                         ctxobs_size, obs_noise_std, False, device)
        model.load_state_dict(torch.load(ckpt_ctrl, map_location=device))
        model_noctrl.load_state_dict(torch.load(ckpt_noctrl, map_location=device))
        model.eval()
        model_noctrl.eval()

        with torch.no_grad():
            xs_ctrl = model.sample_posterior_paths_em(
                ys_test, ts_test, ts_test, H, R, L=l_samples, gain=gain, mask=mask_test
            )
            xs_noc = model_noctrl.sample_posterior_paths_em(
                ys_test, ts_test, ts_test, H, R, L=l_samples, gain=gain, mask=mask_test
            )

            # particle baselines with the TRUE dynamics (oracle for PF/PG)
            raw_physics = StochasticLorenz63(**l63_params).to(device)
            norm_physics = NormalizedPhysicsWrapper(raw_physics, train_mean, train_std)
            pf = BootstrapParticleFilter(norm_physics, obs_model, num_particles=pf_num_particles)
            sm = ParticleSmoother(norm_physics, obs_model, num_particles=pf_num_particles)
            x0_norm = xs_test[0]
            x0_std_norm = torch.tensor(float(pf_x0_std), device=device)
            pf_s = pf.run(ys_test, ts_test, x0_norm, x0_std_norm, device, mask=mask_test).permute(1, 0, 2, 3)
            pg_s = sm.run_smoother(ys_test, ts_test, x0_norm, x0_std_norm, device, mask=mask_test).permute(1, 0, 2, 3)

        results = {"ours_ctrl": xs_ctrl, "ours_noctrl": xs_noc, "pf": pf_s, "pg": pg_s}
        row = {"seed": seed}
        for m, S in results.items():
            r = posterior_mean_rmse(S, xs_test)
            w1t = w1_to_truth_per_time(S, xs_test, subsample_t=subsample_t, max_b=max_b)
            w1p = w1_pooled(S, xs_test)
            agg[m]["rmse"].append(r)
            agg[m]["w1t"].append(w1t)
            agg[m]["w1p"].append(w1p)
            row[m] = (r, w1t, w1p)
        per_seed.append(row)
        logging.info(
            f"[seed {seed}] "
            + " | ".join(f"{m}: RMSE={row[m][0]:.4f} W1t={row[m][1]:.4f} W1p={row[m][2]:.4f}"
                         for m in methods)
        )

    if not per_seed:
        logging.error("No seeds evaluated -- check seed_dirs / checkpoints.")
        return None

    # --- aggregate ---
    lines = []
    lines.append("=" * 78)
    lines.append(f"Lorenz-63 multi-seed evaluation (inference only)")
    lines.append(f"n_seeds={len(per_seed)} | L(ours)={l_samples} | particles(PF/PG)={pf_num_particles} | gain={gain}")
    lines.append("W1_time : per-time W1 to the truth path, avg over (t,b,d)  (= MAE-to-truth; the corrected metric).")
    lines.append("W1_pool : legacy pooled-marginal W1 over all (t,b,d)       (reproduced by matching the stationary marginal).")
    lines.append("PF/PG are given the TRUE Lorenz dynamics; Ours learns them from data.")
    lines.append("-" * 78)
    lines.append(f"{'method':<14}{'RMSE (mean±SE)':<24}{'W1_time (mean±SE)':<24}{'W1_pool (mean±SE)':<24}")
    for m in methods:
        rm, rse = _mean_se(agg[m]["rmse"])
        tm, tse = _mean_se(agg[m]["w1t"])
        pm, pse = _mean_se(agg[m]["w1p"])
        lines.append(f"{m:<14}{rm:.4f} ± {rse:.4f}        {tm:.4f} ± {tse:.4f}        {pm:.4f} ± {pse:.4f}")
    lines.append("-" * 78)

    # --- paired control-term ablation ---
    c_r = np.asarray(agg["ours_ctrl"]["rmse"]); n_r = np.asarray(agg["ours_noctrl"]["rmse"])
    c_t = np.asarray(agg["ours_ctrl"]["w1t"]);  n_t = np.asarray(agg["ours_noctrl"]["w1t"])
    if len(c_r) > 1:
        try:
            from scipy import stats as _stats
            p_r = float(_stats.ttest_rel(n_r, c_r).pvalue)
            p_t = float(_stats.ttest_rel(n_t, c_t).pvalue)
        except Exception:
            p_r, p_t = float("nan"), float("nan")
        d_r = n_r - c_r  # > 0  => ctrl has lower error => control helps
        d_t = n_t - c_t
        dr_m, dr_se = _mean_se(d_r)
        dt_m, dt_se = _mean_se(d_t)
        lines.append("Ablation (paired over seeds, Δ = noctrl − ctrl; positive ⇒ control helps):")
        lines.append(f"  RMSE     Δ = {dr_m:+.4f} ± {dr_se:.4f}   ctrl better in {int((d_r > 0).sum())}/{len(d_r)} seeds   (paired t p={p_r:.3f})")
        lines.append(f"  W1_time  Δ = {dt_m:+.4f} ± {dt_se:.4f}   ctrl better in {int((d_t > 0).sum())}/{len(d_t)} seeds   (paired t p={p_t:.3f})")
    lines.append("=" * 78)

    summary = "\n".join(lines)
    print(summary)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        ensure_dir(out_dir)
    with open(out_path, "w") as f:
        f.write(summary + "\n\n")
        f.write("Per-seed values:\n")
        for row in per_seed:
            f.write(
                f"seed {row['seed']}: "
                + " | ".join(f"{m} RMSE={row[m][0]:.4f} W1t={row[m][1]:.4f} W1p={row[m][2]:.4f}"
                             for m in methods)
                + "\n"
            )
    logging.info(f"Wrote summary to {out_path}")
    return agg


# =========================
# Inference-only budget sweep (sample / particle efficiency)
# =========================
def _load_seed_eval_data(sd, seed, data_dim, obs_noise_std, missing_rate_test,
                         sweep_max_batch, device):
    """Load cached test states for a seed dir and build the inference tensors.
    Deterministic given `seed`. Optionally caps the batch (columns) for memory."""
    d_tr = torch.load(os.path.join(sd, "l63_train_raw.pth"), map_location=device)
    xs_train_raw = d_tr["xs_raw"].to(device)
    d_te = torch.load(os.path.join(sd, "l63_test_raw.pth"), map_location=device)
    xs_test_raw = d_te["xs_raw"].to(device)
    ts_test = d_te["ts"].to(device)

    train_mean, train_std = compute_norm_stats(xs_train_raw)
    train_mean = train_mean.to(device)
    train_std = train_std.to(device)
    xs_test = normalize_x(xs_test_raw, train_mean, train_std)

    obs_model = LorenzAtanObsModel(obs_sigma=obs_noise_std).to(device)
    ys_test = obs_model.sample(xs_test)
    mask_test = torch.bernoulli(torch.full_like(ys_test, 1 - missing_rate_test))
    ys_test = ys_test * mask_test

    if sweep_max_batch is not None and xs_test.size(1) > int(sweep_max_batch):
        B = int(sweep_max_batch)
        xs_test = xs_test[:, :B].contiguous()
        ys_test = ys_test[:, :B].contiguous()
        mask_test = mask_test[:, :B].contiguous()

    return xs_test, ys_test, mask_test, ts_test, train_mean, train_std, obs_model


def plot_budget_curves(results, methods, out_path):
    """results[method][budget] = {'rmse': [...over seeds], 'w1t': [...]}"""
    plt.rcParams.update({"font.size": 12})
    style = {
        "ours_ctrl":   ("tab:blue",   "o", "Ours (ctrl)"),
        "ours_noctrl": ("tab:cyan",   "s", "Ours (no ctrl)"),
        "pf":          ("tab:orange", "^", "PF (true dyn.)"),
        "pg":          ("tab:green",  "D", "PG (true dyn.)"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, ttl in [(axes[0], "rmse", "RMSE to truth"),
                            (axes[1], "w1t", r"per-time $W_1$ to truth")]:
        for m in methods:
            budgets = sorted(results[m].keys())
            means, ses = [], []
            for b in budgets:
                mu, se = _mean_se(results[m][b][metric])
                means.append(mu); ses.append(se)
            c, mk, lab = style.get(m, ("gray", "o", m))
            ax.errorbar(budgets, means, yerr=ses, marker=mk, color=c,
                        capsize=3, lw=1.8, markersize=6, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"inference budget  (Ours: # samples $L$;  PF/PG: # particles $N$)")
        ax.set_ylabel(ttl)
        ax.set_title(ttl)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path) or ".")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def budget_sweep(
    base_dir: str = "./dump/l63/",
    seed_dirs: Tuple[str, ...] = (),
    seeds: Tuple[int, ...] = (),
    sample_grid: Tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512),         # Ours: # posterior samples L
    particle_grid: Tuple[int, ...] = (16, 32, 64, 128, 256, 512),  # PF/PG: # particles N
    data_dim: int = 3,
    obs_noise_std: float = 0.15,
    missing_rate_test: float = 0.0,
    latent_size: int = 4,
    context_size: int = 128,
    hidden_size: int = 256,
    ctxobs_size: int = 64,
    theta1: float = 10.0,
    theta2: float = 28.0,
    theta3: float = 8.0 / 3.0,
    beta1: float = 0.1,
    beta2: float = 0.28,
    beta3: float = 0.3,
    pf_x0_std: float = 0.10,
    gain: float = 1.0,
    subsample_t: int = 5,
    max_b: int = 8,
    sweep_max_batch: Optional[int] = None,   # cap batch (columns) to limit memory at large N
    out_dir: str = "./dump/l63_budget/",
):
    """
    Inference-only sample/particle-efficiency sweep over pre-trained seeds.

    For each seed we draw the latent-SDE posterior once at L=max(sample_grid) and
    evaluate every smaller L by subsampling the i.i.d. draws (free); the particle
    baselines are RE-RUN at each N (their weight degeneracy is N-dependent and
    cannot be obtained by subsampling). Reports RMSE and per-time W1-to-truth vs
    budget (mean +/- SE over seeds), writes a CSV, a per-method "saturation budget"
    (smallest budget within 5% of the method's own best error), and the budget
    curves.
    """
    if isinstance(seed_dirs, str):
        seed_dirs = (seed_dirs,)
    if isinstance(seeds, int):
        seeds = (seeds,)
    if not seed_dirs:
        seed_dirs, seeds = _discover_seed_dirs(base_dir)
        if not seed_dirs:
            logging.error(f"No seed*/  subdirectories found under {base_dir}. Train first.")
            return None
    assert len(seed_dirs) == len(seeds), "seed_dirs and seeds must have equal length"
    sample_grid = tuple(sorted({int(x) for x in sample_grid}))
    particle_grid = tuple(sorted({int(x) for x in particle_grid}))
    Lmax = max(sample_grid)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l63_params = dict(theta1=theta1, theta2=theta2, theta3=theta3,
                      beta1=beta1, beta2=beta2, beta3=beta3)

    results: Dict[str, Dict[int, Dict[str, list]]] = {}

    def _slot(method, budget):
        results.setdefault(method, {}).setdefault(budget, {"rmse": [], "w1t": []})
        return results[method][budget]

    for sd, seed in zip(seed_dirs, seeds):
        ckpt_ctrl = os.path.join(sd, "model_ctrl.pth")
        ckpt_noctrl = os.path.join(sd, "model_noctrl.pth")
        needed = [ckpt_ctrl, ckpt_noctrl,
                  os.path.join(sd, "l63_train_raw.pth"), os.path.join(sd, "l63_test_raw.pth")]
        miss = [p for p in needed if not os.path.exists(p)]
        if miss:
            logging.warning(f"[seed {seed}] missing {miss}; skipping {sd}.")
            continue

        set_seed(seed)
        xs_test, ys_test, mask_test, ts_test, train_mean, train_std, obs_model = _load_seed_eval_data(
            sd, seed, data_dim, obs_noise_std, missing_rate_test, sweep_max_batch, device
        )
        H = torch.eye(data_dim, device=device)
        R = (obs_noise_std ** 2) * torch.eye(data_dim, device=device)

        # --- Ours: draw L=Lmax once per model, subsample for smaller L ---
        for tag, use_ctrl, ckpt in [("ours_ctrl", True, ckpt_ctrl),
                                    ("ours_noctrl", False, ckpt_noctrl)]:
            m = _build_latent_sde(data_dim, latent_size, context_size, hidden_size,
                                  ctxobs_size, obs_noise_std, use_ctrl, device)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m.eval()
            with torch.no_grad():
                S_full = m.sample_posterior_paths_em(
                    ys_test, ts_test, ts_test, H, R, L=Lmax, gain=gain, mask=mask_test
                )  # (Lmax, T, B, D)
            for L in sample_grid:
                S = S_full[:L]
                _slot(tag, L)["rmse"].append(posterior_mean_rmse(S, xs_test))
                _slot(tag, L)["w1t"].append(
                    w1_to_truth_per_time(S, xs_test, subsample_t=subsample_t, max_b=max_b)
                )
            del S_full, m
            if device.type == "cuda":
                torch.cuda.empty_cache()
        logging.info(f"[seed {seed}] Ours sweep done over L={sample_grid}")

        # --- PF / PG: re-run at each N (degeneracy is N-dependent) ---
        raw_physics = StochasticLorenz63(**l63_params).to(device)
        norm_physics = NormalizedPhysicsWrapper(raw_physics, train_mean, train_std)
        x0_norm = xs_test[0]
        x0_std_norm = torch.tensor(float(pf_x0_std), device=device)
        for N in particle_grid:
            with torch.no_grad():
                pf = BootstrapParticleFilter(norm_physics, obs_model, num_particles=N)
                pf_s = pf.run(ys_test, ts_test, x0_norm, x0_std_norm, device, mask=mask_test).permute(1, 0, 2, 3)
                _slot("pf", N)["rmse"].append(posterior_mean_rmse(pf_s, xs_test))
                _slot("pf", N)["w1t"].append(
                    w1_to_truth_per_time(pf_s, xs_test, subsample_t=subsample_t, max_b=max_b)
                )
                del pf_s
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                sm = ParticleSmoother(norm_physics, obs_model, num_particles=N)
                pg_s = sm.run_smoother(ys_test, ts_test, x0_norm, x0_std_norm, device, mask=mask_test).permute(1, 0, 2, 3)
                _slot("pg", N)["rmse"].append(posterior_mean_rmse(pg_s, xs_test))
                _slot("pg", N)["w1t"].append(
                    w1_to_truth_per_time(pg_s, xs_test, subsample_t=subsample_t, max_b=max_b)
                )
                del pg_s
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            logging.info(f"[seed {seed}] PF/PG done at N={N}")

    if not results:
        logging.error("No seeds evaluated -- check seed_dirs / checkpoints.")
        return None

    methods = [m for m in ["ours_ctrl", "ours_noctrl", "pf", "pg"] if m in results]
    ensure_dir(out_dir)

    # --- CSV of aggregated curves ---
    csv_path = os.path.join(out_dir, "budget_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("method,budget,n_seeds,rmse_mean,rmse_se,w1t_mean,w1t_se\n")
        for m in methods:
            for b in sorted(results[m].keys()):
                rm, rse = _mean_se(results[m][b]["rmse"])
                tm, tse = _mean_se(results[m][b]["w1t"])
                ns = len(results[m][b]["rmse"])
                f.write(f"{m},{b},{ns},{rm:.6f},{rse:.6f},{tm:.6f},{tse:.6f}\n")
    logging.info(f"Wrote {csv_path}")

    # --- saturation budget per method (smallest budget within 5% of its own best) ---
    sat_lines = ["Saturation budget = smallest budget within 5% (>=0.95x best, lower=better) of the method's own best mean:"]
    for metric, label in [("rmse", "RMSE"), ("w1t", "W1_time")]:
        sat_lines.append(f"  [{label}]")
        for m in methods:
            budgets = sorted(results[m].keys())
            means = [_mean_se(results[m][b][metric])[0] for b in budgets]
            best = min(means)
            sat = next((b for b, mn in zip(budgets, means) if mn <= 1.05 * best), budgets[-1])
            sat_lines.append(f"    {m:<12} best={best:.4f}  saturates at budget={sat}")
    sat_txt = "\n".join(sat_lines)
    print(sat_txt)
    with open(os.path.join(out_dir, "saturation.txt"), "w") as f:
        f.write(sat_txt + "\n")

    # --- figures ---
    plot_budget_curves(results, methods, os.path.join(out_dir, "budget_curves.pdf"))
    plot_budget_curves(results, methods, os.path.join(out_dir, "budget_curves.png"))
    logging.info(f"Saved budget figures + CSV to {out_dir}")


# =========================
# Main
# =========================
def main(
    data_dim: int = 3,
    obs_dim: int = 3,
    t0: float = 0.0,
    t1: float = 2.0,
    steps_train: int = 100,
    steps_test: int = 100,
    batch_size: int = 256,
    seed: int = 0,
    obs_noise_std: float = 0.15,

    missing_rate_train: float = 0.0,
    missing_rate_test: float = 0.0,

    latent_size: int = 4,
    context_size: int = 128,
    hidden_size: int = 256,
    ctxobs_size: int = 64,
    lr_init: float = 1e-3,
    num_iters: int = 500,
    save_every: int = 100,

    sim_dt: float = 1e-3,
    sim_method: str = "euler",

    theta1: float = 10.0,
    theta2: float = 28.0,
    theta3: float = 8.0 / 3.0,
    beta1: float = 0.1,
    beta2: float = 0.28,
    beta3: float = 0.3,

    pf_num_particles: int = 512,
    pf_x0_std: float = 0.10,
    l_samples: int = 64,

    base_dir: str = "",
    train_dir: str = "",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    if not train_dir:
        _base = base_dir if base_dir else "./dump/l63/"
        train_dir = os.path.join(_base, f"seed{seed}")
    ensure_dir(train_dir)

    l63_params = dict(
        theta1=theta1, theta2=theta2, theta3=theta3,
        beta1=beta1, beta2=beta2, beta3=beta3
    )

    xs_train_raw, ts_train = make_dataset(
        t0, t1, steps_train, batch_size, device,
        cache_path=os.path.join(train_dir, "l63_train_raw.pth"),
        l63_params=l63_params, sim_dt=sim_dt, sim_method=sim_method,
    )
    xs_test_raw, ts_test = make_dataset(
        t0, t1, steps_test, batch_size, device,
        cache_path=os.path.join(train_dir, "l63_test_raw.pth"),
        l63_params=l63_params, sim_dt=sim_dt, sim_method=sim_method,
    )

    train_mean, train_std = compute_norm_stats(xs_train_raw)
    train_mean = train_mean.to(device)
    train_std = train_std.to(device)

    xs_train = normalize_x(xs_train_raw, train_mean, train_std)
    xs_test = normalize_x(xs_test_raw, train_mean, train_std)

    obs_model = LorenzAtanObsModel(obs_sigma=obs_noise_std).to(device)

    ys_train = obs_model.sample(xs_train)
    ys_test = obs_model.sample(xs_test)

    mask_train = torch.bernoulli(torch.full_like(ys_train, 1 - missing_rate_train))
    mask_test = torch.bernoulli(torch.full_like(ys_test, 1 - missing_rate_test))

    ys_train = ys_train * mask_train
    ys_test = ys_test * mask_test

    H = torch.eye(data_dim, device=device)
    R = (obs_noise_std ** 2) * torch.eye(data_dim, device=device)

    # Full model
    model = LatentSDE(
        data_size=data_dim,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
        ctxobs_size=ctxobs_size,
        num_heads=2,
        time_d=32,
        obs_noise_std=obs_noise_std,
        use_ctrl=True,
    ).to(device)

    # Ablation model: no control network contribution in drift
    model_noctrl = LatentSDE(
        data_size=data_dim,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
        ctxobs_size=ctxobs_size,
        num_heads=2,
        time_d=32,
        obs_noise_std=obs_noise_std,
        use_ctrl=False,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr_init)
    optimizer_noctrl = optim.Adam(model_noctrl.parameters(), lr=lr_init)

    model_path = os.path.join(train_dir, "model_ctrl.pth")
    model_noctrl_path = os.path.join(train_dir, "model_noctrl.pth")

    if os.path.exists(model_path) and os.path.exists(model_noctrl_path):
        logging.info(f"Found existing models, skipping training.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model_noctrl.load_state_dict(torch.load(model_noctrl_path, map_location=device))
    else:
        train_or_load_model(
            model=model,
            model_path=model_path,
            optimizer=optimizer,
            xs_train=xs_train,
            ys_train=ys_train,
            ts_train=ts_train,
            H=H,
            R=R,
            mask_train=mask_train,
            obs_noise_std=obs_noise_std,
            num_iters=num_iters,
            save_every=save_every,
            train_dir=train_dir,
            seed=seed,
            plot_prefix="ctrl",
        )

        train_or_load_model(
            model=model_noctrl,
            model_path=model_noctrl_path,
            optimizer=optimizer_noctrl,
            xs_train=xs_train,
            ys_train=ys_train,
            ts_train=ts_train,
            H=H,
            R=R,
            mask_train=mask_train,
            obs_noise_std=obs_noise_std,
            num_iters=num_iters,
            save_every=save_every,
            train_dir=train_dir,
            seed=seed,
            plot_prefix="noctrl",
        )

    logging.info("Starting Final Evaluation...")
    with torch.no_grad():
        model.eval()
        model_noctrl.eval()

        x_samples_model = model.sample_posterior_paths_em(
            ys_test, ts_test, ts_test, H, R, L=l_samples, gain=1.0, mask=mask_test
        )
        x_samples_noctrl = model_noctrl.sample_posterior_paths_em(
            ys_test, ts_test, ts_test, H, R, L=l_samples, gain=1.0, mask=mask_test
        )

        rmse_model = posterior_mean_rmse(x_samples_model, xs_test)
        rmse_noctrl = posterior_mean_rmse(x_samples_noctrl, xs_test)
        w1_model = w1_to_truth_per_time(x_samples_model, xs_test)
        w1_noctrl = w1_to_truth_per_time(x_samples_noctrl, xs_test)

        # PF / smoother baselines
        raw_physics = StochasticLorenz63(**l63_params).to(device)
        norm_physics = NormalizedPhysicsWrapper(raw_physics, train_mean, train_std)

        pf = BootstrapParticleFilter(norm_physics, obs_model, num_particles=pf_num_particles)
        sm = ParticleSmoother(norm_physics, obs_model, num_particles=pf_num_particles)

        x0_norm = xs_test[0]
        x0_std_norm = torch.tensor(float(pf_x0_std), device=device)

        pf_hist = pf.run(ys_test, ts_test, x0_norm, x0_std_norm, device, mask=mask_test)
        sm_hist = sm.run_smoother(ys_test, ts_test, x0_norm, x0_std_norm, device, mask=mask_test)

        pf_samples = pf_hist.permute(1, 0, 2, 3)
        sm_samples = sm_hist.permute(1, 0, 2, 3)

        rmse_pf = posterior_mean_rmse(pf_samples, xs_test)
        rmse_sm = posterior_mean_rmse(sm_samples, xs_test)
        w1_pf = w1_to_truth_per_time(pf_samples, xs_test)
        w1_sm = w1_to_truth_per_time(sm_samples, xs_test)

        logging.info("-" * 60)
        logging.info("FINAL EVALUATION (per-time W1 to truth)")
        logging.info("-" * 60)
        logging.info(f"Latent SDE (ctrl)    : RMSE={rmse_model:.4f} | W1_time={w1_model:.4f}")
        logging.info(f"Latent SDE (no ctrl) : RMSE={rmse_noctrl:.4f} | W1_time={w1_noctrl:.4f}")
        logging.info(f"Particle Filter      : RMSE={rmse_pf:.4f} | W1_time={w1_pf:.4f}")
        logging.info(f"Particle Smoother    : RMSE={rmse_sm:.4f} | W1_time={w1_sm:.4f}")
        logging.info("-" * 60)

        import time as _time
        results_path = os.path.join(train_dir, "results.txt")
        with open(results_path, "a") as _f:
            _f.write(f"=== {_time.strftime('%Y-%m-%d %H:%M:%S')} | seed={seed} | particles={pf_num_particles} | l_samples={l_samples} ===\n")
            _f.write(f"RMSE     Ctrl={rmse_model:.4f}  NoCtrl={rmse_noctrl:.4f}  PF={rmse_pf:.4f}  PG={rmse_sm:.4f}\n")
            _f.write(f"W1_time  Ctrl={w1_model:.4f}  NoCtrl={w1_noctrl:.4f}  PF={w1_pf:.4f}  PG={w1_sm:.4f}\n")
            _f.write("\n")
        logging.info(f"Results appended to {results_path}")

        # Means for screenshot-style trajectory plots
        b = 0
        x_true = xs_test[:, b, :]
        x_model_mu = x_samples_model.mean(dim=0)[:, b, :]
        x_noctrl_mu = x_samples_noctrl.mean(dim=0)[:, b, :]
        x_pf_mu = pf_samples.mean(dim=0)[:, b, :]
        x_sm_mu = sm_samples.mean(dim=0)[:, b, :]

        # CI plot for full model
        plot_model_ci(
            epoch="final_model",
            test_xs=xs_test,
            test_xs_post=x_samples_model,
            ts=ts_test,
            train_dir=train_dir,
            sample_idx=0,
            seed_idx=seed,
            title="Latent SDE (with ctrl)"
        )
        plot_benchmark_triptych(
            out_path=os.path.join(train_dir, "benchmark_triptych.png"),
            ts=ts_test,
            x_true=x_true,
            x_model=x_model_mu,
            x_pf=x_pf_mu,
            x_sm=x_sm_mu,
        )
        plot_ablation_compare(
            out_path=os.path.join(train_dir, "ablation_ctrl_vs_noctrl.png"),
            ts=ts_test,
            x_true=x_true,
            x_full=x_model_mu,
            x_noctrl=x_noctrl_mu,
        )

        logging.info(f"Saved plots to {train_dir}")


# =========================
# Multi-seed training
# =========================
def train_seeds(
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4),
    base_dir: str = "./dump/l63/",
    data_dim: int = 3,
    obs_dim: int = 3,
    t0: float = 0.0,
    t1: float = 2.0,
    steps_train: int = 100,
    steps_test: int = 100,
    batch_size: int = 256,
    obs_noise_std: float = 0.15,
    missing_rate_train: float = 0.0,
    missing_rate_test: float = 0.0,
    latent_size: int = 4,
    context_size: int = 128,
    hidden_size: int = 256,
    ctxobs_size: int = 64,
    lr_init: float = 1e-3,
    num_iters: int = 500,
    save_every: int = 100,
    sim_dt: float = 1e-3,
    sim_method: str = "euler",
    theta1: float = 10.0,
    theta2: float = 28.0,
    theta3: float = 8.0 / 3.0,
    beta1: float = 0.1,
    beta2: float = 0.28,
    beta3: float = 0.3,
    pf_num_particles: int = 512,
    pf_x0_std: float = 0.10,
    l_samples: int = 64,
):
    """Train independent seeds sequentially, each into {base_dir}/seed<N>/."""
    if isinstance(seeds, int):
        seeds = (seeds,)
    for s in seeds:
        logging.info(f"=== Training seed {s} ===")
        main(
            data_dim=data_dim, obs_dim=obs_dim, t0=t0, t1=t1,
            steps_train=steps_train, steps_test=steps_test, batch_size=batch_size,
            seed=s, obs_noise_std=obs_noise_std,
            missing_rate_train=missing_rate_train, missing_rate_test=missing_rate_test,
            latent_size=latent_size, context_size=context_size, hidden_size=hidden_size,
            ctxobs_size=ctxobs_size, lr_init=lr_init, num_iters=num_iters,
            save_every=save_every, sim_dt=sim_dt, sim_method=sim_method,
            theta1=theta1, theta2=theta2, theta3=theta3,
            beta1=beta1, beta2=beta2, beta3=beta3,
            pf_num_particles=pf_num_particles, pf_x0_std=pf_x0_std, l_samples=l_samples,
            base_dir=base_dir,
        )


if __name__ == "__main__":
    # `python lorenz63.py train`        -> single-seed training (auto-names dir by seed)
    # `python lorenz63.py train_seeds`  -> train all seeds sequentially
    # `python lorenz63.py eval`         -> inference-only multi-seed evaluation (auto-discovers seeds)
    # `python lorenz63.py sweep`        -> sample/particle-efficiency sweep (auto-discovers seeds)
    fire.Fire({"train": main, "train_seeds": train_seeds,
               "eval": evaluate_seeds, "sweep": budget_sweep})
