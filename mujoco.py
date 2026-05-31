import os
os.environ["MUJOCO_GL"] = "egl"

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import fire
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

import torchsde
from sdeint_obs import sdeint_obs


# -----------------------------------------------------------------------------
# Irregular-grid helpers
# -----------------------------------------------------------------------------
def _make_irregular_time_grid(
    rng: np.random.RandomState,
    T_obs: int,
    irregularity_strength: float = 0.8,
    min_dt_scale: float = 0.15,
    max_dt_scale: float = 3.0,
    max_time: float = 1.0,
) -> np.ndarray:
    assert T_obs >= 2
    low = max(1e-4, 1.0 - irregularity_strength)
    high = 1.0 + irregularity_strength

    dt = rng.uniform(low=low, high=high, size=T_obs - 1).astype(np.float64)
    dt = np.clip(dt, min_dt_scale, max_dt_scale)

    t = np.concatenate([[0.0], np.cumsum(dt)])
    t = (t / t[-1]) * max_time
    return t.astype(np.float32)


def _build_time_grid_bank(
    num_grids: int,
    T_obs: int,
    seed: int,
    irregularity_strength: float = 0.8,
    max_time: float = 1.0,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    bank = []
    for _ in range(num_grids):
        ts = _make_irregular_time_grid(
            rng=rng,
            T_obs=T_obs,
            irregularity_strength=irregularity_strength,
            max_time=max_time,
        )
        bank.append(ts)
    return np.stack(bank, axis=0)


def _time_grid_to_internal_indices(
    ts_obs: np.ndarray,
    internal_len: int,
    max_time: float = 1.0,
) -> np.ndarray:
    idx = np.round((ts_obs / max_time) * (internal_len - 1)).astype(np.int64)
    idx[0] = 0
    idx[-1] = internal_len - 1

    for i in range(1, len(idx)):
        idx[i] = max(idx[i], idx[i - 1] + 1)
    idx[-1] = min(idx[-1], internal_len - 1)

    for i in range(len(idx) - 2, -1, -1):
        idx[i] = min(idx[i], idx[i + 1] - 1)

    idx[0] = 0
    idx[-1] = internal_len - 1

    if np.any(np.diff(idx) <= 0):
        raise RuntimeError("Internal index mapping is not strictly increasing.")
    return idx


# -----------------------------------------------------------------------------
# HopperPhysics irregular data
# -----------------------------------------------------------------------------
def _sample_mode_action_schedule_internal(
    rng: np.random.RandomState,
    internal_len: int,
    action_dim: int,
    mode: int,
    pulse_start_min_frac: float,
    pulse_start_max_frac: float,
    pulse_len_min_frac: float,
    pulse_len_max_frac: float,
    pulse_scale: float,
) -> np.ndarray:
    actions = np.zeros((internal_len, action_dim), dtype=np.float32)

    start_lo = max(2, int(round(pulse_start_min_frac * internal_len)))
    start_hi = max(start_lo + 1, int(round(pulse_start_max_frac * internal_len)))
    start_hi = min(start_hi, internal_len - 2)

    pulse_len_lo = max(4, int(round(pulse_len_min_frac * internal_len)))
    pulse_len_hi = max(pulse_len_lo + 1, int(round(pulse_len_max_frac * internal_len)))

    start = int(rng.randint(start_lo, start_hi + 1))
    pulse_len = int(rng.randint(pulse_len_lo, min(pulse_len_hi, internal_len - start) + 1))
    end = min(internal_len, start + pulse_len)

    if mode == 0:
        vec = np.zeros(action_dim, dtype=np.float32)
        amp = 0.0
    elif mode == 1:
        base = np.array([0.7, 1.0, 0.5, 0.0], dtype=np.float32)
        vec = base[:action_dim]
        amp = pulse_scale * rng.uniform(0.8, 1.2)
    elif mode == 2:
        base = np.array([-0.7, -0.5, 0.9, -0.3], dtype=np.float32)
        vec = base[:action_dim]
        amp = pulse_scale * rng.uniform(0.8, 1.2)
    elif mode == 3:
        base = np.array([0.0, 1.0, -1.0, 0.7], dtype=np.float32)
        vec = base[:action_dim]
        amp = pulse_scale * rng.uniform(0.8, 1.2)
    else:
        sign = -1.0 if mode % 2 == 0 else 1.0
        base = np.array([0.4, 0.9, -0.8, 0.2], dtype=np.float32) * sign
        vec = base[:action_dim]
        amp = pulse_scale * rng.uniform(0.7, 1.3)

    if amp > 0.0:
        tau = np.linspace(0.0, 1.0, end - start, dtype=np.float32)
        env = np.sin(np.pi * tau).astype(np.float32)
        if mode == 3:
            env = env * np.sign(np.sin(4.0 * np.pi * tau)).astype(np.float32)
        actions[start:end] = amp * env[:, None] * vec[None, :]

    return actions


def generate_hopperphysics_irregular(
    data_dir: str,
    num_trajs: int = 10000,
    T_obs: int = 100,
    internal_len: int = 400,
    num_time_grids: int = 16,
    seed: int = 0,
    num_modes: int = 4,
    pulse_start_min_frac: float = 0.30,
    pulse_start_max_frac: float = 0.55,
    pulse_len_min_frac: float = 0.10,
    pulse_len_max_frac: float = 0.22,
    pulse_scale: float = 0.75,
    qpos_noise_std: float = 0.1,
    qvel_noise_std: float = 0.1,
    irregularity_strength: float = 0.8,
    max_time: float = 1.0,
) -> str:
    os.makedirs(data_dir, exist_ok=True)
    fpath = os.path.join(
        data_dir,
        f"training_multimodal_irregular_Tobs{T_obs}_Tint{internal_len}_G{num_time_grids}_H{max_time}.pt",
    )
    if os.path.exists(fpath):
        logging.info(f"Irregular HopperPhysics dataset already exists: {fpath}")
        return fpath

    from dm_control import suite

    logging.info(
        f"Generating irregular HopperPhysics dataset "
        f"({num_trajs} trajs, T_obs={T_obs}, internal_len={internal_len}, grids={num_time_grids}) ..."
    )
    rng = np.random.RandomState(seed)
    env = suite.load(domain_name="hopper", task_name="stand")
    physics = env.physics
    D = int(physics.state().shape[0])
    action_dim = int(env.action_spec().shape[0])

    ts_bank = _build_time_grid_bank(
        num_grids=num_time_grids,
        T_obs=T_obs,
        seed=seed + 12345,
        irregularity_strength=irregularity_strength,
        max_time=max_time,
    )
    obs_index_bank = np.stack(
        [_time_grid_to_internal_indices(ts_bank[g], internal_len, max_time) for g in range(num_time_grids)],
        axis=0,
    )

    all_trajs = np.empty((num_trajs, T_obs, D), dtype=np.float32)
    all_modes = np.empty((num_trajs,), dtype=np.int64)
    all_actions = np.empty((num_trajs, T_obs, action_dim), dtype=np.float32)
    all_grid_ids = np.empty((num_trajs,), dtype=np.int64)

    mode_order = np.arange(num_modes, dtype=np.int64)
    mode_order = np.resize(mode_order, num_trajs)
    rng.shuffle(mode_order)

    grid_ids = np.arange(num_time_grids, dtype=np.int64)
    grid_ids = np.resize(grid_ids, num_trajs)
    rng.shuffle(grid_ids)

    for i in range(num_trajs):
        env.reset()

        qpos0 = physics.data.qpos.copy() + rng.randn(physics.data.qpos.shape[0]) * qpos_noise_std
        qvel0 = rng.randn(physics.data.qvel.shape[0]) * qvel_noise_std
        physics.data.qpos[:] = qpos0
        physics.data.qvel[:] = qvel0
        physics.forward()

        mode = int(mode_order[i])
        gid = int(grid_ids[i])
        obs_idx = obs_index_bank[gid]

        actions_fine = _sample_mode_action_schedule_internal(
            rng=rng,
            internal_len=internal_len,
            action_dim=action_dim,
            mode=mode,
            pulse_start_min_frac=pulse_start_min_frac,
            pulse_start_max_frac=pulse_start_max_frac,
            pulse_len_min_frac=pulse_len_min_frac,
            pulse_len_max_frac=pulse_len_max_frac,
            pulse_scale=pulse_scale,
        )

        traj_fine = np.empty((internal_len, D), dtype=np.float32)
        traj_fine[0] = physics.state().astype(np.float32)

        for t in range(1, internal_len):
            env.step(actions_fine[t - 1])
            traj_fine[t] = physics.state().astype(np.float32)

        traj_obs = traj_fine[obs_idx]
        actions_obs = actions_fine[obs_idx]

        all_trajs[i] = traj_obs
        all_modes[i] = mode
        all_actions[i] = actions_obs
        all_grid_ids[i] = gid

        if (i + 1) % 1000 == 0:
            binc = np.bincount(all_modes[: i + 1], minlength=num_modes)
            ginc = np.bincount(all_grid_ids[: i + 1], minlength=num_time_grids)
            logging.info(
                f"  generated {i + 1}/{num_trajs} | mode_counts={binc.tolist()} | "
                f"grid_usage_minmax=({int(ginc.min())}, {int(ginc.max())})"
            )

    payload = {
        "x": torch.from_numpy(all_trajs),
        "mode": torch.from_numpy(all_modes),
        "actions": torch.from_numpy(all_actions),
        "ts_bank": torch.from_numpy(ts_bank),
        "grid_id": torch.from_numpy(all_grid_ids),
        "meta": {
            "T_obs": T_obs,
            "internal_len": internal_len,
            "num_modes": num_modes,
            "num_time_grids": num_time_grids,
            "pulse_start_min_frac": pulse_start_min_frac,
            "pulse_start_max_frac": pulse_start_max_frac,
            "pulse_len_min_frac": pulse_len_min_frac,
            "pulse_len_max_frac": pulse_len_max_frac,
            "pulse_scale": pulse_scale,
            "irregularity_strength": irregularity_strength,
            "max_time": max_time,
        },
    }
    torch.save(payload, fpath)
    logging.info(f"Saved irregular HopperPhysics dataset to: {fpath} | shape={tuple(payload['x'].shape)}")
    return fpath


def load_hopperphysics_irregular(
    data_dir: str,
    num_trajs: int = 10000,
    T_obs: int = 100,
    internal_len: int = 400,
    num_time_grids: int = 16,
    seed: int = 0,
    num_modes: int = 4,
    pulse_start_min_frac: float = 0.30,
    pulse_start_max_frac: float = 0.55,
    pulse_len_min_frac: float = 0.10,
    pulse_len_max_frac: float = 0.22,
    pulse_scale: float = 0.75,
    irregularity_strength: float = 0.8,
    max_time: float = 1.0,
) -> Dict[str, torch.Tensor]:
    fpath = generate_hopperphysics_irregular(
        data_dir=data_dir,
        num_trajs=num_trajs,
        T_obs=T_obs,
        internal_len=internal_len,
        num_time_grids=num_time_grids,
        seed=seed,
        num_modes=num_modes,
        pulse_start_min_frac=pulse_start_min_frac,
        pulse_start_max_frac=pulse_start_max_frac,
        pulse_len_min_frac=pulse_len_min_frac,
        pulse_len_max_frac=pulse_len_max_frac,
        pulse_scale=pulse_scale,
        irregularity_strength=irregularity_strength,
        max_time=max_time,
    )

    try:
        payload = torch.load(fpath, weights_only=True)
    except TypeError:
        payload = torch.load(fpath)

    payload["x"] = payload["x"].float()
    payload["mode"] = payload["mode"].long()
    payload["actions"] = payload["actions"].float()
    payload["ts_bank"] = payload["ts_bank"].float()
    payload["grid_id"] = payload["grid_id"].long()
    return payload


def compute_stats(x_all: torch.Tensor) -> Dict[str, torch.Tensor]:
    flat = x_all.reshape(-1, x_all.shape[-1])
    return {"mean": flat.mean(dim=0), "std": flat.std(dim=0).clamp_min(1e-6)}


# -----------------------------------------------------------------------------
# Dataset / config
# -----------------------------------------------------------------------------
@dataclass
class HopperConfig:
    obs_noise_std: float = 0.35
    keep_time_prob: float = 0.95
    drop_dim_prob: float = 0.15
    T_use: Optional[int] = 100

    num_hidden_windows: int = 2
    hidden_window_min_frac: float = 0.05
    hidden_window_max_frac: float = 0.12

    irregular_times: bool = True
    irregularity_strength: float = 0.8
    state_noise_std: float = 0.08

    force_observe_initial: bool = True
    force_observe_final: bool = False
    force_observe_prefix_frac: float = 0.25
    force_mask_future_window: bool = True
    future_window_min_frac: float = 0.68
    future_window_max_frac: float = 0.72


class HopperPhysicsIrregularDataset(Dataset):
    def __init__(
        self,
        payload: Dict[str, torch.Tensor],
        stats: Dict[str, torch.Tensor],
        cfg: HopperConfig,
        seed: int = 0,
        normalize: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.seed = seed

        x_all = payload["x"]
        mode_all = payload["mode"]
        ts_bank = payload["ts_bank"]
        grid_id = payload["grid_id"]

        if cfg.T_use is not None:
            x_all = x_all[:, : cfg.T_use, :]
            ts_bank = ts_bank[:, : cfg.T_use]

        self.x_all = x_all
        self.mode_all = mode_all
        self.ts_bank = ts_bank
        self.grid_id = grid_id

        self.N, self.T, self.D = x_all.shape
        self.num_grids = int(ts_bank.shape[0])

        if normalize:
            self.mean = stats["mean"].view(1, 1, self.D)
            self.std = stats["std"].view(1, 1, self.D)
        else:
            self.mean = torch.zeros(1, 1, self.D)
            self.std = torch.ones(1, 1, self.D)

    def __len__(self) -> int:
        return self.N

    def _sample_hidden_windows_time(
        self,
        ts: torch.Tensor,
        rng_local: np.random.RandomState,
    ) -> torch.Tensor:
        hidden_t = torch.zeros(self.T, dtype=torch.bool)

        num_windows = int(self.cfg.num_hidden_windows)
        prefix_cutoff = float(self.cfg.force_observe_prefix_frac)

        if num_windows > 0:
            for _ in range(num_windows):
                dur_lo = self.cfg.hidden_window_min_frac
                dur_hi = self.cfg.hidden_window_max_frac
                duration = float(rng_local.uniform(dur_lo, dur_hi))

                start_lo = prefix_cutoff
                start_hi = max(start_lo + 1e-3, 1.0 - duration - 1e-3)
                start = float(rng_local.uniform(start_lo, start_hi))
                end = min(1.0, start + duration)

                hidden_t |= ((ts >= start) & (ts <= end))

        if self.cfg.force_mask_future_window:
            f0 = float(self.cfg.future_window_min_frac)
            f1 = float(self.cfg.future_window_max_frac)
            hidden_t |= ((ts >= f0) & (ts <= f1))

        hidden_t[ts <= prefix_cutoff] = False
        hidden_t[0] = False
        if self.cfg.force_observe_final:
            hidden_t[-1] = False

        return hidden_t

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng_local = np.random.RandomState(self.seed * 1000003 + idx)

        gid = int(self.grid_id[idx].item())
        ts = self.ts_bank[gid].clone()

        x = self.x_all[idx: idx + 1]
        x = (x - self.mean) / self.std
        x = x.squeeze(0)

        if self.cfg.state_noise_std > 0.0:
            x = x + self.cfg.state_noise_std * torch.randn_like(x)

        mask = torch.ones(self.T, self.D, dtype=torch.float32)

        if self.cfg.keep_time_prob < 1.0:
            keep_t = rng_local.rand(self.T) < self.cfg.keep_time_prob
            keep_t = torch.tensor(keep_t, dtype=torch.bool)

            prefix_cutoff = float(self.cfg.force_observe_prefix_frac)
            keep_t[ts <= prefix_cutoff] = True

            if self.cfg.force_observe_initial:
                keep_t[0] = True
            if self.cfg.force_observe_final:
                keep_t[-1] = True

            mask[~keep_t, :] = 0.0

        if self.cfg.drop_dim_prob > 0.0:
            drop = rng_local.rand(self.T, self.D) < self.cfg.drop_dim_prob
            drop = torch.tensor(drop, dtype=torch.bool)
            time_kept = (mask.sum(dim=1) > 0).unsqueeze(1)
            drop = drop & time_kept
            mask[drop] = 0.0

        hidden_t = self._sample_hidden_windows_time(ts=ts, rng_local=rng_local)
        hidden_mask = hidden_t.unsqueeze(-1).expand(self.T, self.D)
        mask[hidden_mask] = 0.0

        prefix_cutoff = float(self.cfg.force_observe_prefix_frac)
        mask[ts <= prefix_cutoff, :] = 1.0
        if self.cfg.force_observe_initial:
            mask[0, :] = 1.0
        if self.cfg.force_observe_final:
            mask[-1, :] = 1.0

        noise = self.cfg.obs_noise_std * torch.randn_like(x)
        y_full = torch.atan(x) + noise
        y = torch.where(mask > 0, y_full, torch.full_like(y_full, float("nan")))

        return {
            "ts": ts,
            "x": x,
            "y": y,
            "mask": mask,
            "hidden_mask": hidden_mask.float(),
            "mode": self.mode_all[idx].long(),
            "grid_id": torch.tensor(gid, dtype=torch.long),
        }


def collate_hopper_irregular(batch: List[Dict[str, torch.Tensor]]):
    ts0 = batch[0]["ts"]
    gid0 = int(batch[0]["grid_id"].item())

    for b in batch[1:]:
        gid = int(b["grid_id"].item())
        if gid != gid0:
            raise RuntimeError("Mixed grid_id in batch. Use GridBatchSampler.")
        if not torch.allclose(b["ts"], ts0):
            raise RuntimeError("Mixed timestamps in batch. Use GridBatchSampler.")

    xs = torch.stack([b["x"] for b in batch], dim=1)
    ys = torch.stack([b["y"] for b in batch], dim=1)
    mask = torch.stack([b["mask"] for b in batch], dim=1)
    hidden_mask = torch.stack([b["hidden_mask"] for b in batch], dim=1)
    modes = torch.stack([b["mode"] for b in batch], dim=0)
    grid_ids = torch.stack([b["grid_id"] for b in batch], dim=0)
    return {
        "times": ts0,
        "xs": xs,
        "ys": ys,
        "mask": mask,
        "hidden_mask": hidden_mask,
        "modes": modes,
        "grid_ids": grid_ids,
    }


class GridBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        grid_ids: torch.Tensor,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.grid_ids = grid_ids.cpu().numpy().astype(np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)

        self.groups = {}
        for idx, gid in enumerate(self.grid_ids):
            self.groups.setdefault(int(gid), []).append(idx)

        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self._epoch)
        group_batches = []

        gids = list(self.groups.keys())
        if self.shuffle:
            rng.shuffle(gids)

        for gid in gids:
            idxs = np.array(self.groups[gid], dtype=np.int64)
            if self.shuffle:
                rng.shuffle(idxs)

            n_full = len(idxs) // self.batch_size
            used = n_full * self.batch_size

            if used > 0:
                arr = idxs[:used].reshape(n_full, self.batch_size)
                for row in arr:
                    group_batches.append(row.tolist())

            if not self.drop_last and used < len(idxs):
                group_batches.append(idxs[used:].tolist())

        if self.shuffle:
            rng.shuffle(group_batches)

        for batch in group_batches:
            yield batch

    def __len__(self):
        total = 0
        for _, idxs in self.groups.items():
            n = len(idxs)
            total += n // self.batch_size
            if not self.drop_last and (n % self.batch_size != 0):
                total += 1
        return total


# -----------------------------------------------------------------------------
# Model definitions
# -----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_time_diff=True):
        super().__init__()
        self.use_time_diff = use_time_diff
        actual_input = input_size * 2 + (1 if use_time_diff else 0)
        self.gru = nn.GRU(input_size=actual_input, hidden_size=hidden_size, batch_first=False)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp, mask=None, times=None):
        T, B, _ = inp.shape
        if mask is None:
            mask = torch.ones_like(inp)
        inp = torch.nan_to_num(inp, nan=0.0)
        feats = [inp * mask, mask]
        if self.use_time_diff and times is not None:
            dt = torch.zeros(T, device=inp.device, dtype=inp.dtype)
            dt[1:] = times[1:] - times[:-1]
            dt = dt.view(T, 1, 1).expand(T, inp.shape[1], 1)
            feats.append(dt)
        combined = torch.cat(feats, dim=-1)
        out, _ = self.gru(combined)
        return self.lin(out)


class ObsEncoderGRU(nn.Module):
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
            t_enc = self.time_encoder(times.view(T, 1))
            t_enc = t_enc.unsqueeze(0).expand(B, T, -1)
            feats.append(t_enc)

        return torch.cat(feats, dim=-1)

    def forward(self, inp, mask=None, times=None):
        combined = self._prepare_input(inp, mask, times)
        out, _ = self.gru(combined)

        if mask is not None:
            mask_t = (mask.sum(dim=-1) > 0).float().unsqueeze(-1)
            pooled = (out * mask_t).sum(dim=1) / mask_t.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = out.mean(dim=1)

        return self.output_proj(pooled)

    def forward_per_time(self, inp, mask=None, times=None, causal: bool = True):
        combined = self._prepare_input(inp, mask, times)
        out, _ = self.gru(combined)
        per_time = self.output_proj(out)
        return per_time.permute(1, 0, 2)


class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(
        self,
        data_size,
        latent_size,
        context_size,
        hidden_size,
        ctxobs_size=64,
        num_heads=4,
        causal=True,
        time_d=32,
        decoder_hidden=256,
        diffusion_floor=0.20,
        learn_mixture_prior=False,
        mixture_components=4,
    ):
        super().__init__()
        self.data_size = data_size
        self.latent_size = latent_size
        self.context_size = context_size
        self.ctxobs_size = ctxobs_size
        self.causal = causal
        self.diffusion_floor = diffusion_floor
        self.learn_mixture_prior = learn_mixture_prior
        self.mixture_components = mixture_components

        self.encoder = Encoder(data_size, hidden_size, context_size, use_time_diff=True)
        self.obs_encoder = ObsEncoderGRU(
            data_size, hidden_size, ctxobs_size, causal=causal, use_time_encoding=True
        )

        self.qz0_net = nn.Linear(ctxobs_size, 2 * latent_size)

        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size + ctxobs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size + ctxobs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, latent_size),
        )

        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1),
                )
                for _ in range(latent_size)
            ]
        )

        # Must match code 1 checkpoint architecture.
        self.control_net = nn.Sequential(
            nn.Linear(latent_size + ctxobs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, latent_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, decoder_hidden),
            nn.LayerNorm(decoder_hidden),
            nn.Tanh(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.LayerNorm(decoder_hidden),
            nn.Tanh(),
            nn.Linear(decoder_hidden, data_size),
        )
        self.x0_encoder = nn.Sequential(
            nn.Linear(data_size, decoder_hidden),
            nn.Tanh(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.Tanh(),
            nn.Linear(decoder_hidden, latent_size),
        )

        if self.learn_mixture_prior:
            self.pz0_logits = nn.Parameter(torch.zeros(mixture_components))
            self.pz0_means = nn.Parameter(torch.randn(mixture_components, latent_size) * 0.15)
            self.pz0_logstds = nn.Parameter(torch.zeros(mixture_components, latent_size))
        else:
            self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
            self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None
        self._obs_ctx_seq = None
        self._ts_cached = None
        self._K = None
        self._V = None

        self.time_feat = nn.Sequential(nn.Linear(1, time_d), nn.Tanh(), nn.Linear(time_d, time_d))
        self.time_key = nn.Linear(ctxobs_size, time_d)
        self.time_val = nn.Linear(ctxobs_size, ctxobs_size)
        self.mix_gate = nn.Sequential(nn.Linear(ctxobs_size * 2, 64), nn.Tanh(), nn.Linear(64, 1))

    def decode_x(self, z):
        return self.decoder(z)

    def encode_x0(self, x0):
        return self.x0_encoder(x0)

    def _sample_prior_z0(self, B, device, dtype):
        if self.learn_mixture_prior:
            probs = torch.softmax(self.pz0_logits, dim=0)
            comps = torch.multinomial(probs, num_samples=B, replacement=True)
            means = self.pz0_means[comps].to(device=device, dtype=dtype)
            stds = self.pz0_logstds.exp()[comps].to(device=device, dtype=dtype)
            return means + stds * torch.randn_like(means)
        return self.pz0_mean + self.pz0_logstd.exp() * torch.randn(B, self.latent_size, device=device, dtype=dtype)

    def _encode_x_context(self, xs, ts):
        x_mask = torch.ones_like(xs)
        xs_rev = torch.flip(xs, dims=(0,))
        mask_rev = torch.flip(x_mask, dims=(0,))
        ts_rev = ts[-1] - torch.flip(ts, dims=(0,))
        ctx_rev = self.encoder(xs_rev, mask=mask_rev, times=ts_rev)
        ctx_seq = torch.flip(ctx_rev, dims=(0,))
        return ctx_seq

    def _encode_obs_context(self, obs, ts, mask=None):
        if mask is None:
            mask = torch.ones_like(obs)
        obs_b = obs.permute(1, 0, 2)
        mask_b = mask.permute(1, 0, 2)
        obs_ctx_seq = self.obs_encoder.forward_per_time(obs_b, mask=mask_b, times=ts, causal=self.causal)
        obs_ctx_global = self.obs_encoder(obs_b, mask=mask_b, times=ts)

        self._obs_ctx_seq = obs_ctx_seq
        self._ts_cached = ts
        self._K = self.time_key(obs_ctx_seq)
        self._V = self.time_val(obs_ctx_seq)
        return obs_ctx_seq, obs_ctx_global

    @torch.no_grad()
    def condition_on_obs(self, obs, ts, mask=None):
        _, obs_ctx_global = self._encode_obs_context(obs, ts, mask=mask)
        return self.qz0_net(obs_ctx_global).chunk(2, dim=1)

    def _embed_obs_at_time(self, t):
        ts = self._ts_cached
        E = torch.nan_to_num(self._obs_ctx_seq, nan=0.0)
        K = torch.nan_to_num(self._K, nan=0.0)
        V = torch.nan_to_num(self._V, nan=0.0)

        T, B, _ = E.shape
        i1 = int(torch.searchsorted(ts, t, right=True).item())
        i0 = max(i1 - 1, 0)
        i1 = min(i1, T - 1)
        t0, t1 = ts[i0], ts[i1]
        denom = (t1 - t0).clamp_min(1e-8)
        w = ((t - t0) / denom).clamp(0, 1)
        e_lin = (1.0 - w) * E[i0] + w * E[i1]

        t_in = t.expand(B, 1)
        q = self.time_feat(t_in)
        Kbt = K.permute(1, 0, 2)
        Vbt = V.permute(1, 0, 2)
        Dk = Kbt.size(-1)

        scores = torch.bmm(q.unsqueeze(1), Kbt.transpose(1, 2)).squeeze(1) / math.sqrt(Dk)
        if self.causal:
            causal_mask = (ts.unsqueeze(0) <= t_in)
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        scores = scores.clamp(min=-1e9, max=1e9)
        w_all = torch.softmax(scores, dim=-1)
        w_all = torch.nan_to_num(w_all, nan=1.0 / scores.size(-1))
        e_attn = torch.bmm(w_all.unsqueeze(1), Vbt).squeeze(1)

        alpha = torch.sigmoid(self.mix_gate(torch.cat([e_lin, e_attn], dim=1)))
        return (1.0 - alpha) * e_lin + alpha * e_attn

    def f(self, t, z, obs=None):
        ts, ctx_seq, _ = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        e_t = self._embed_obs_at_time(t)
        base = self.f_net(torch.cat([z, ctx_seq[i], e_t], dim=1))
        return torch.nan_to_num(base, nan=0.0, posinf=1e6, neginf=-1e6)

    def h(self, t, z, obs=None):
        e_t = self._embed_obs_at_time(t)
        out = self.h_net(torch.cat([z, e_t], dim=1))
        ctrl = self.control_net(torch.cat([z, e_t], dim=1))
        return torch.nan_to_num(out + ctrl, nan=0.0, posinf=1e6, neginf=-1e6)

    def g(self, t, z):
        chunks = torch.split(z, 1, dim=1)
        out = []
        for g_net, c in zip(self.g_nets, chunks):
            raw = g_net(c)
            out.append(torch.nn.functional.softplus(raw) + self.diffusion_floor)
        return torch.nan_to_num(torch.cat(out, dim=1), nan=self.diffusion_floor).clamp_min(self.diffusion_floor)

    def obs_score_z(self, z, y_t, mask_t, R_scalar):
        with torch.enable_grad():
            z_req = z.detach().requires_grad_(True)
            x = self.decode_x(z_req)
            mean_y = torch.atan(x)

            var_d = torch.full((self.data_size,), float(R_scalar), device=z.device, dtype=z.dtype).view(1, -1)
            y_clean = torch.where(torch.isnan(y_t), mean_y.detach(), y_t)

            ll = -0.5 * (
                ((y_clean - mean_y) ** 2) / var_d
                + torch.log(2.0 * torch.tensor(math.pi, device=z.device, dtype=z.dtype) * var_d)
            )
            ll = (ll * mask_t).sum()

            grad = torch.autograd.grad(
                outputs=ll,
                inputs=z_req,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

        return grad.detach()

    def euler_maruyama_posterior(
        self,
        obs,
        ts,
        mask,
        R_scalar: float,
        L: int = 64,
        gain: float = 1.0,
        use_explicit_likelihood: bool = True,
        x0: Optional[torch.Tensor] = None,
    ):
        self._encode_obs_context(obs, ts, mask=mask)

        T, B, _ = obs.shape
        Dz = self.latent_size

        x_samples = []
        for _ in range(L):
            if x0 is not None:
                z0 = self.encode_x0(x0)
            else:
                z0 = self._sample_prior_z0(B, obs.device, obs.dtype)

            z_path = torch.zeros(T, B, Dz, device=obs.device, dtype=obs.dtype)
            z_path[0] = z0

            for i in range(1, T):
                t_prev = ts[i - 1]
                dt = (ts[i] - ts[i - 1]).clamp_min(1e-8)
                sqrt_dt = torch.sqrt(dt)

                z_prev = z_path[i - 1]

                with torch.no_grad():
                    drift = self.h(t_prev, z_prev, obs)
                    g_diag = self.g(t_prev, z_prev)
                    dW = torch.randn_like(z0) * sqrt_dt

                if use_explicit_likelihood:
                    like = self.obs_score_z(z_prev.detach(), obs[i], mask[i], R_scalar)
                    like = torch.clamp(like, min=-100.0, max=100.0)
                    with torch.no_grad():
                        z_path[i] = z_prev + (drift + gain * like) * dt + g_diag * dW
                else:
                    with torch.no_grad():
                        z_path[i] = z_prev + drift * dt + g_diag * dW

            x_samples.append(self.decode_x(z_path))

        return torch.stack(x_samples, dim=0)


class GRUARFilter(nn.Module):
    def __init__(self, D: int, H: int = 128):
        super().__init__()
        self.D = D
        self.gru = nn.GRU(input_size=3 * D + 1, hidden_size=H, batch_first=False)
        self.head_mu = nn.Linear(H, D)
        self.head_logvar = nn.Linear(H, D)

    def step(self, y_t, mask_t, x_prev, dt_t, h=None):
        inp_t = torch.cat([y_t * mask_t, mask_t, x_prev, dt_t], dim=-1).unsqueeze(0)
        out, h = self.gru(inp_t, h)
        out = out.squeeze(0)
        mu_t = self.head_mu(out)
        logvar_t = self.head_logvar(out).clamp(-8.0, 5.0)
        return mu_t, logvar_t, h

    def forward_autoregressive(
        self,
        y: torch.Tensor,
        mask: torch.Tensor,
        times: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        feedback: str = "sample",
        detach_feedback: bool = False,
        warm_start_from_obs: bool = False,
    ):
        assert feedback in {"sample", "mean"}

        T, B, D = y.shape
        y0 = torch.nan_to_num(y, nan=0.0)
        device = y.device
        dtype = y.dtype

        dts = torch.zeros(T, 1, device=device, dtype=dtype)
        dts[1:, 0] = times[1:] - times[:-1]
        dts = dts.unsqueeze(1).expand(T, B, 1)

        if x0 is not None:
            x_prev = x0
        elif warm_start_from_obs:
            x_prev = torch.tan(torch.clamp(y0[0], min=-1.3, max=1.3)) * mask[0]
        else:
            x_prev = torch.zeros(B, D, device=device, dtype=dtype)

        h = None
        mus = []
        logvars = []
        xs_feedback = []

        for t in range(T):
            mu_t, logvar_t, h = self.step(y0[t], mask[t], x_prev, dts[t], h=h)

            if t == 0 and x0 is not None:
                mu_t = x0
                logvar_t = torch.full_like(mu_t, -12.0)
                x_t = x0
            else:
                std_t = torch.exp(0.5 * logvar_t)
                if feedback == "sample":
                    x_t = mu_t + std_t * torch.randn_like(mu_t)
                else:
                    x_t = mu_t

            x_prev = x_t.detach() if detach_feedback else x_t

            mus.append(mu_t)
            logvars.append(logvar_t)
            xs_feedback.append(x_t)

        mu_seq = torch.stack(mus, dim=0)
        logvar_seq = torch.stack(logvars, dim=0)
        x_rollout = torch.stack(xs_feedback, dim=0)
        return mu_seq, logvar_seq, x_rollout

    @torch.no_grad()
    def sample_posterior(
        self,
        y: torch.Tensor,
        mask: torch.Tensor,
        times: torch.Tensor,
        L: int = 128,
        feedback: str = "sample",
        warm_start_from_obs: bool = False,
        x0: Optional[torch.Tensor] = None,
    ):
        all_samples = []
        for _ in range(L):
            _, _, x_rollout = self.forward_autoregressive(
                y=y,
                mask=mask,
                times=times,
                x0=x0,
                feedback=feedback,
                detach_feedback=False,
                warm_start_from_obs=warm_start_from_obs,
            )
            all_samples.append(x_rollout)

        return torch.stack(all_samples, dim=0)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def _set_physics_state_from_vec14(physics, state14: np.ndarray):
    state14 = state14.astype(np.float64, copy=False)
    try:
        physics.set_state(state14)
        physics.forward()
    except Exception:
        nq = physics.data.qpos.shape[0]
        physics.data.qpos[:] = state14[:nq]
        physics.data.qvel[:] = state14[nq:nq + physics.data.qvel.shape[0]]
        physics.forward()


def render_truth_recon_frames(
    out_path: str,
    x_true_14: np.ndarray,
    x_recon_14: np.ndarray,
    frame_indices: np.ndarray,
    title: str = "Truth vs Reconstructed (Hopper)",
    width: int = 240,
    height: int = 240,
    x_recon2_14: Optional[np.ndarray] = None,
    recon_label: str = "LatentSDE Recon",
    recon2_label: str = "GRU-AR Recon",
):
    from dm_control import suite

    env = suite.load(domain_name="hopper", task_name="stand")
    physics = env.physics

    frames_true, frames_recon = [], []

    for k in frame_indices:
        _set_physics_state_from_vec14(physics, x_true_14[k])
        frames_true.append(physics.render(height=height, width=width, camera_id=0))

    for k in frame_indices:
        _set_physics_state_from_vec14(physics, x_recon_14[k])
        frames_recon.append(physics.render(height=height, width=width, camera_id=0))

    frames_recon2 = []
    if x_recon2_14 is not None:
        for k in frame_indices:
            _set_physics_state_from_vec14(physics, x_recon2_14[k])
            frames_recon2.append(physics.render(height=height, width=width, camera_id=0))

    n = len(frame_indices)
    nrows = 3 if len(frames_recon2) > 0 else 2
    fig, axes = plt.subplots(nrows, n, figsize=(2.2 * n, 2.4 * nrows))
    if n == 1:
        axes = axes.reshape(nrows, 1)

    for j in range(n):
        axes[0, j].imshow(frames_true[j])
        axes[0, j].axis("off")
        if j == 0:
            axes[0, j].set_title("Truth", fontsize=12)

        axes[1, j].imshow(frames_recon[j])
        axes[1, j].axis("off")
        if j == 0:
            axes[1, j].set_title(recon_label, fontsize=12)

        if len(frames_recon2) > 0:
            axes[2, j].imshow(frames_recon2[j])
            axes[2, j].axis("off")
            if j == 0:
                axes[2, j].set_title(recon2_label, fontsize=12)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_spaghetti(
    out_path: str,
    ts: torch.Tensor,
    x_true: torch.Tensor,
    x_samples: torch.Tensor,
    mask: torch.Tensor,
    hidden_mask: torch.Tensor,
    dims=(0, 1, 2),
    max_paths: int = 50,
    title: str = "Posterior sample spaghetti",
):
    ts_np = ts.detach().cpu().numpy()
    L = x_samples.shape[0]
    take = min(L, max_paths)

    fig, axes = plt.subplots(len(dims), 1, figsize=(12, 3 * len(dims)), sharex=True)
    if len(dims) == 1:
        axes = [axes]

    for ax, d in zip(axes, dims):
        xt = x_true[:, d].detach().cpu().numpy()
        mk = mask[:, d].detach().cpu().numpy().astype(bool)
        hk = hidden_mask[:, d].detach().cpu().numpy().astype(bool)

        for i in range(take):
            xs = x_samples[i, :, d].detach().cpu().numpy()
            ax.plot(ts_np, xs, alpha=0.18, linewidth=1.0)

        ax.plot(ts_np, xt, color="black", linewidth=2.0, label="Truth")
        ax.plot(ts_np[mk], xt[mk], "k.", markersize=4, label="Observed")

        if hk.any():
            hidden_idx = np.where(hk)[0]
            splits = np.split(hidden_idx, np.where(np.diff(hidden_idx) != 1)[0] + 1)
            for seg in splits:
                if len(seg) > 0:
                    ax.axvspan(ts_np[seg[0]], ts_np[seg[-1]], color="gray", alpha=0.12)

        ax.set_ylabel(f"dim {d}")
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("t")
    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_histograms_hidden_times(
    out_path: str,
    ts: torch.Tensor,
    x_true: torch.Tensor,
    sde_samples: torch.Tensor,
    gru_samples: Optional[torch.Tensor],
    hidden_mask: torch.Tensor,
    dims=(0, 1, 2),
    bins: int = 40,
    max_times: int = 4,
    title: str = "Posterior histograms at hidden future times",
):
    hidden_any = (hidden_mask.sum(dim=1) > 0).detach().cpu().numpy().astype(bool)
    hidden_idx = np.where(hidden_any)[0]

    if len(hidden_idx) == 0:
        raise ValueError("No hidden times found to plot histograms.")

    pick_idx = np.linspace(0, len(hidden_idx) - 1, min(max_times, len(hidden_idx))).round().astype(int)
    time_ids = hidden_idx[pick_idx]

    nrows = len(dims)
    ncols = len(time_ids)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows), squeeze=False)

    ts_np = ts.detach().cpu().numpy()

    for r, d in enumerate(dims):
        for c, t_id in enumerate(time_ids):
            ax = axes[r, c]

            sde_vals = sde_samples[:, t_id, d].detach().cpu().numpy()
            ax.hist(sde_vals, bins=bins, alpha=0.50, density=True, label="LatentSDE")

            if gru_samples is not None:
                gru_vals = gru_samples[:, t_id, d].detach().cpu().numpy()
                ax.hist(gru_vals, bins=bins, alpha=0.50, density=True, label="GRU-AR")

            true_val = x_true[t_id, d].detach().cpu().item()
            ax.axvline(true_val, color="black", linewidth=2.0, linestyle="-", label="Truth")

            ax.set_title(f"dim {d}, t={ts_np[t_id]:.3f} (idx={t_id})")
            ax.grid(True, alpha=0.25)

            if r == 0 and c == 0:
                ax.legend()

    plt.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_true_mode_histograms(
    out_path: str,
    x_all: torch.Tensor,
    mode_all: torch.Tensor,
    ts: torch.Tensor,
    dims=(0, 1, 2),
    time_fracs=(0.38, 0.52, 0.68, 0.90),
    bins: int = 50,
    title: str = "True state histograms by mode and pooled",
):
    x_np = x_all.detach().cpu().numpy()
    mode_np = mode_all.detach().cpu().numpy()
    ts_np = ts.detach().cpu().numpy()

    uniq_modes = np.unique(mode_np)
    time_ids = [int(round(frac * (x_np.shape[1] - 1))) for frac in time_fracs]

    nrows = len(dims)
    ncols = len(time_ids)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)

    for r, d in enumerate(dims):
        for c, t_id in enumerate(time_ids):
            ax = axes[r, c]

            pooled_vals = x_np[:, t_id, d]
            ax.hist(pooled_vals, bins=bins, density=True, histtype="step", linewidth=2.0, label="all modes")

            for m in uniq_modes:
                vals_m = x_np[mode_np == m, t_id, d]
                ax.hist(vals_m, bins=bins, density=True, alpha=0.35, label=f"mode {m}")

            ax.set_title(f"dim {d}, t={ts_np[t_id]:.3f} (idx={t_id})")
            ax.grid(True, alpha=0.25)

            if r == 0 and c == 0:
                ax.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Aggregate hidden-distribution analysis
# -----------------------------------------------------------------------------
@torch.no_grad()
def collect_hidden_hist_data_over_testset(
    model: "LatentSDE",
    gru: Optional["GRUARFilter"],
    test_loader: DataLoader,
    device: torch.device,
    R_scalar: float,
    L_samples: int,
    gain: float,
    use_explicit_likelihood: bool,
    gru_feedback_eval: str,
    gru_warm_start_from_obs: bool,
    dims=(1, 3, 5, 7, 9),
    time_fracs=(0.38, 0.52, 0.68, 0.90),
    max_eval_batches: int = 4,
):
    model.eval()
    if gru is not None:
        gru.eval()

    out = None

    for batch_idx, batch_te in enumerate(test_loader):
        if batch_idx >= max_eval_batches:
            break

        ts_te = batch_te["times"].to(device)
        xs_te = batch_te["xs"].to(device)
        ys_te = batch_te["ys"].to(device)
        mask_te = batch_te["mask"].to(device)
        hidden_mask_te = batch_te["hidden_mask"].to(device)

        T, B, D = xs_te.shape
        time_ids = [int(round(frac * (T - 1))) for frac in time_fracs]

        if out is None:
            out = {}
            ts_np = ts_te.detach().cpu().numpy()
            for d in dims:
                for t_id in time_ids:
                    out[(d, t_id)] = {
                        "true": [],
                        "sde": [],
                        "gru": [],
                        "t": float(ts_np[t_id]),
                    }

        x_samps_sde = model.euler_maruyama_posterior(
            obs=ys_te,
            ts=ts_te,
            mask=mask_te,
            R_scalar=R_scalar,
            L=L_samples,
            gain=gain,
            use_explicit_likelihood=use_explicit_likelihood,
            x0=None,
        )

        x_samps_gru = None
        if gru is not None:
            x_samps_gru = gru.sample_posterior(
                y=ys_te,
                mask=mask_te,
                times=ts_te,
                L=L_samples,
                feedback=gru_feedback_eval,
                warm_start_from_obs=gru_warm_start_from_obs,
                x0=None,
            )

        for d in dims:
            for t_id in time_ids:
                hidden_b = hidden_mask_te[t_id, :, d] > 0
                if hidden_b.sum().item() == 0:
                    continue

                true_vals = xs_te[t_id, hidden_b, d].detach().cpu().tolist()
                sde_vals = x_samps_sde[:, t_id, hidden_b, d].reshape(-1).detach().cpu().tolist()

                out[(d, t_id)]["true"].extend(true_vals)
                out[(d, t_id)]["sde"].extend(sde_vals)

                if x_samps_gru is not None:
                    gru_vals = x_samps_gru[:, t_id, hidden_b, d].reshape(-1).detach().cpu().tolist()
                    out[(d, t_id)]["gru"].extend(gru_vals)

    return out


def plot_aggregate_hidden_histograms(
    out_path: str,
    hist_data: dict,
    dims=(0, 1, 2),
    bins: int = 50,
    title: str = "Aggregated hidden-state histograms over masked test set",
):
    keys = list(hist_data.keys())
    time_ids = sorted(list(set(k[1] for k in keys)))

    nrows = len(dims)
    ncols = len(time_ids)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows), squeeze=False)

    for r, d in enumerate(dims):
        for c, t_id in enumerate(time_ids):
            ax = axes[r, c]
            rec = hist_data[(d, t_id)]

            true_vals = np.array(rec["true"], dtype=float)
            sde_vals = np.array(rec["sde"], dtype=float)
            gru_vals = np.array(rec["gru"], dtype=float)

            if len(true_vals) > 0:
                ax.hist(true_vals, bins=bins, density=True, histtype="step", alpha=0.99, label="True hidden test")
            if len(sde_vals) > 0:
                ax.hist(sde_vals, bins=bins, density=True, alpha=0.45, label="LatentSDE")
            if len(gru_vals) > 0:
                ax.hist(gru_vals, bins=bins, density=True, alpha=0.35, label="GRU-AR")

            ax.set_title(f"dim {d}, t={rec['t']:.3f} (idx={t_id})")
            ax.grid(True, alpha=0.25)

            if r == 0 and c == 0:
                ax.legend()

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"[plot] wrote {out_path}")


@torch.no_grad()
def _sample_w1_1d(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.flatten().sort().values
    y = y.flatten().sort().values

    n = min(x.numel(), y.numel())
    if n == 0:
        return float("nan")

    if x.numel() != n:
        idx = torch.linspace(0, x.numel() - 1, steps=n, device=x.device).round().long()
        x = x[idx]
    if y.numel() != n:
        idx = torch.linspace(0, y.numel() - 1, steps=n, device=y.device).round().long()
        y = y[idx]

    return torch.mean(torch.abs(x - y)).item()


@torch.no_grad()
def _sample_energy_score_1d(x_true: torch.Tensor, x_model: torch.Tensor, max_pairs: int = 2048) -> float:
    x_true = x_true.flatten()
    x_model = x_model.flatten()

    if x_true.numel() == 0 or x_model.numel() == 0:
        return float("nan")

    term1 = torch.abs(x_model.unsqueeze(1) - x_true.unsqueeze(0)).mean()

    L = x_model.numel()
    if L <= 1:
        return term1.item()

    num_pairs = min(max_pairs, L * (L - 1) // 2)
    rng = np.random.RandomState(0)
    diffs = []
    for _ in range(num_pairs):
        i = int(rng.randint(0, L))
        j = int(rng.randint(0, L - 1))
        if j >= i:
            j += 1
        diffs.append(torch.abs(x_model[i] - x_model[j]))

    term2 = torch.stack(diffs).mean()
    return (term1 - 0.5 * term2).item()


@torch.no_grad()
def evaluate_hidden_distribution_metrics(
    hist_data: dict,
    dims=(1, 3, 5, 7, 9),
):
    metrics = {}

    for key, rec in hist_data.items():
        d, t_id = key

        if d not in dims:
            continue

        true_vals = torch.tensor(rec["true"], dtype=torch.float32)
        sde_vals = torch.tensor(rec["sde"], dtype=torch.float32)
        gru_vals = torch.tensor(rec["gru"], dtype=torch.float32)

        metrics[key] = {
            "t": rec["t"],
            "w1_sde": _sample_w1_1d(sde_vals, true_vals),
            "w1_gru": _sample_w1_1d(gru_vals, true_vals) if gru_vals.numel() > 0 else float("nan"),
            "es_sde": _sample_energy_score_1d(true_vals, sde_vals),
            "es_gru": _sample_energy_score_1d(true_vals, gru_vals) if gru_vals.numel() > 0 else float("nan"),
        }

    return metrics


def write_hidden_distribution_metrics(
    out_path: str,
    metrics: dict,
    dims=(1, 3, 5, 7, 9),
):
    time_ids = sorted(list(set(k[1] for k in metrics.keys())))

    with open(out_path, "w") as f:
        f.write("dim,time_idx,t,w1_sde,w1_gru,es_sde,es_gru\n")
        for d in dims:
            for t_id in time_ids:
                if (d, t_id) not in metrics:
                    continue
                rec = metrics[(d, t_id)]
                f.write(
                    f"{d},{t_id},{rec['t']:.6f},"
                    f"{rec['w1_sde']:.6f},{rec['w1_gru']:.6f},"
                    f"{rec['es_sde']:.6f},{rec['es_gru']:.6f}\n"
                )

    logging.info(f"[metrics] wrote {out_path}")


@torch.no_grad()
def pathwise_wasserstein_to_truth(
    x_samples: torch.Tensor,   # (L, T, B, D) or (L, T, D)
    x_true: torch.Tensor,      # (T, B, D) or (T, D)
    eval_time_mask: Optional[torch.Tensor] = None,
    p: int = 2,
) -> float:
    if x_samples.dim() == 3:
        x_samples = x_samples.unsqueeze(2)   # (L,T,1,D)
    if x_true.dim() == 2:
        x_true = x_true.unsqueeze(1)         # (T,1,D)

    L, T, B, D = x_samples.shape

    if eval_time_mask is None:
        use = torch.ones(T, B, D, device=x_samples.device, dtype=torch.bool)
    else:
        if eval_time_mask.dim() == 1:
            use = eval_time_mask.view(T, 1, 1).expand(T, B, D) > 0
        elif eval_time_mask.dim() == 2:
            use = eval_time_mask.unsqueeze(-1).expand(T, B, D) > 0
        else:
            use = eval_time_mask > 0

    vals = []
    for b in range(B):
        use_b = use[:, b, :]
        if use_b.sum().item() == 0:
            continue
        xb = x_true[:, b, :][use_b]
        samp_b = x_samples[:, :, b, :][:, use_b]
        d = torch.norm(samp_b - xb.unsqueeze(0), p=2, dim=1)
        if p == 1:
            vals.append(d.mean())
        elif p == 2:
            vals.append(torch.sqrt((d ** 2).mean()))
        else:
            vals.append(((d ** p).mean()) ** (1.0 / p))

    if len(vals) == 0:
        return float("nan")
    return torch.stack(vals).mean().item()


@torch.no_grad()
def path_energy_score(
    x_samples: torch.Tensor,   # (L, T, B, D) or (L, T, D)
    x_true: torch.Tensor,      # (T, B, D) or (T, D)
    eval_time_mask: Optional[torch.Tensor] = None,
    max_pairs: int = 64,
) -> float:
    if x_samples.dim() == 3:
        x_samples = x_samples.unsqueeze(2)
    if x_true.dim() == 2:
        x_true = x_true.unsqueeze(1)

    L, T, B, D = x_samples.shape

    if eval_time_mask is None:
        use = torch.ones(T, B, D, device=x_samples.device, dtype=torch.bool)
    else:
        if eval_time_mask.dim() == 1:
            use = eval_time_mask.view(T, 1, 1).expand(T, B, D) > 0
        elif eval_time_mask.dim() == 2:
            use = eval_time_mask.unsqueeze(-1).expand(T, B, D) > 0
        else:
            use = eval_time_mask > 0

    scores = []
    rng = np.random.RandomState(0)

    for b in range(B):
        use_b = use[:, b, :]
        if use_b.sum().item() == 0:
            continue

        xb = x_true[:, b, :][use_b]
        samp_b = x_samples[:, :, b, :][:, use_b]

        term1 = torch.norm(samp_b - xb.unsqueeze(0), dim=1).mean()

        if L <= 1:
            scores.append(term1)
            continue

        num_pairs = min(max_pairs, L * (L - 1) // 2)
        pair_vals = []
        for _ in range(num_pairs):
            i = int(rng.randint(0, L))
            j = int(rng.randint(0, L - 1))
            if j >= i:
                j += 1
            pair_vals.append(torch.norm(samp_b[i] - samp_b[j], p=2))
        term2 = torch.stack(pair_vals).mean()
        scores.append(term1 - 0.5 * term2)

    if len(scores) == 0:
        return float("nan")
    return torch.stack(scores).mean().item()


@torch.no_grad()
def pathwise_w2_per_dim(
    x_samples: torch.Tensor,   # (L,T,D) or (L,T,B,D)
    x_true: torch.Tensor,      # (T,D) or (T,B,D)
    eval_time_mask: Optional[torch.Tensor] = None,
):
    if x_samples.dim() == 3:
        x_samples = x_samples.unsqueeze(2)
    if x_true.dim() == 2:
        x_true = x_true.unsqueeze(1)

    L, T, B, D = x_samples.shape

    if eval_time_mask is None:
        use_t = torch.ones(T, B, device=x_samples.device, dtype=torch.bool)
    else:
        if eval_time_mask.dim() == 1:
            use_t = eval_time_mask.view(T, 1).expand(T, B) > 0
        else:
            use_t = eval_time_mask > 0

    out = {}

    for d in range(D):
        vals_d = []
        for b in range(B):
            mask_tb = use_t[:, b]
            if mask_tb.sum().item() == 0:
                continue

            xb = x_true[:, b, d][mask_tb]
            samp_b = x_samples[:, :, b, d][:, mask_tb]

            dist = torch.norm(samp_b - xb.unsqueeze(0), p=2, dim=1)
            vals_d.append(torch.sqrt((dist ** 2).mean()))

        out[d] = torch.stack(vals_d).mean().item() if len(vals_d) > 0 else float("nan")

    return out


# -----------------------------------------------------------------------------
# Main evaluation-only script
# -----------------------------------------------------------------------------
def main(
    model_ckpt: str = "./hopper_pathwise_runs_multimodal_irregular_hctrl5/model_step_08000.pth",
    gru_ckpt: str = "./hopper_pathwise_runs_multimodal_irregular_hctrl5/gru_ar_step_08000.pth",
    data_dir: str = "./train_hopperphysics_pathwise_multimodal_irregular_hctrl5",
    out_dir: str = "./future_mask_eval_irregulartestlong5hctrl_irr8000_s0",
    seed: int = 0,
    device: str = "",
    sample_index_in_batch: int = 0,
    batch_size: int = 128,
    L_samples: int = 512,

    num_modes: int = 3,
    pulse_start_min_frac: float = 0.30,
    pulse_start_max_frac: float = 0.55,
    pulse_len_min_frac: float = 0.10,
    pulse_len_max_frac: float = 0.22,
    pulse_scale: float = 0.51,

    T_use: int = 100,
    internal_len: int = 400,
    num_time_grids: int = 16,

    obs_noise_std: float = 0.35,
    keep_time_prob: float = 0.7,
    drop_dim_prob: float = 0.15,
    num_hidden_windows: int = 4,
    hidden_window_min_frac: float = 0.1,
    hidden_window_max_frac: float = 0.15,
    force_observe_prefix_frac: float = 0.25,
    force_mask_future_window: bool = True,
    future_window_min_frac: float = 0.58,
    future_window_max_frac: float = 0.75,
    irregular_times: bool = True,
    irregularity_strength: float = 0.8,
    state_noise_std: float = 0.08,
    max_time: float = 1.0,

    latent_size: int = 15,
    context_size: int = 32,
    hidden_size: int = 512,
    ctxobs_size: int = 32,
    num_heads: int = 4,
    causal: bool = True,
    decoder_hidden: int = 64,
    diffusion_floor: float = 0.2,
    learn_mixture_prior: bool = True,
    mixture_components: int = 3,

    gru_hidden: int = 256,
    gru_feedback_eval: str = "sample",
    gru_warm_start_from_obs: bool = False,

    gain: float = 1.0,
    use_explicit_likelihood: bool = True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    os.makedirs(out_dir, exist_ok=True)

    payload = load_hopperphysics_irregular(
        data_dir=data_dir,
        num_trajs=20000,
        T_obs=T_use,
        internal_len=internal_len,
        num_time_grids=num_time_grids,
        seed=seed,
        num_modes=num_modes,
        pulse_start_min_frac=pulse_start_min_frac,
        pulse_start_max_frac=pulse_start_max_frac,
        pulse_len_min_frac=pulse_len_min_frac,
        pulse_len_max_frac=pulse_len_max_frac,
        pulse_scale=pulse_scale,
        irregularity_strength=irregularity_strength,
        max_time=max_time,
    )
    x_all = payload["x"]
    stats = compute_stats(x_all)

    idx = np.arange(x_all.shape[0])
    strat_labels = (
        payload["mode"].cpu().numpy().astype(np.int64) * 1000
        + payload["grid_id"].cpu().numpy().astype(np.int64)
    )

    _, te_idx = train_test_split(
        idx,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
        stratify=strat_labels,
    )

    payload_test = {
        "x": payload["x"][te_idx],
        "mode": payload["mode"][te_idx],
        "actions": payload["actions"][te_idx],
        "ts_bank": payload["ts_bank"],
        "grid_id": payload["grid_id"][te_idx],
    }

    cfg = HopperConfig(
        obs_noise_std=obs_noise_std,
        keep_time_prob=keep_time_prob,
        drop_dim_prob=drop_dim_prob,
        T_use=T_use,
        num_hidden_windows=num_hidden_windows,
        hidden_window_min_frac=hidden_window_min_frac,
        hidden_window_max_frac=hidden_window_max_frac,
        irregular_times=irregular_times,
        irregularity_strength=irregularity_strength,
        state_noise_std=state_noise_std,
        force_observe_initial=True,
        force_observe_final=False,
        force_observe_prefix_frac=force_observe_prefix_frac,
        force_mask_future_window=force_mask_future_window,
        future_window_min_frac=future_window_min_frac,
        future_window_max_frac=future_window_max_frac,
    )

    test_ds = HopperPhysicsIrregularDataset(payload_test, stats=stats, cfg=cfg, seed=seed + 1, normalize=True)

    test_batch_sampler = GridBatchSampler(
        grid_ids=payload_test["grid_id"],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        seed=seed + 1,
    )

    test_loader = DataLoader(
        test_ds,
        batch_sampler=test_batch_sampler,
        collate_fn=collate_hopper_irregular,
    )

    D = int(x_all.shape[-1])

    model = LatentSDE(
        data_size=D,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
        ctxobs_size=ctxobs_size,
        num_heads=num_heads,
        causal=causal,
        decoder_hidden=decoder_hidden,
        diffusion_floor=diffusion_floor,
        learn_mixture_prior=learn_mixture_prior,
        mixture_components=mixture_components,
    ).to(device)

    if not os.path.exists(model_ckpt):
        raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt}")

    state = torch.load(model_ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    logging.info(f"Loaded LatentSDE checkpoint: {model_ckpt}")

    gru = None
    if gru_ckpt:
        if os.path.exists(gru_ckpt):
            gru = GRUARFilter(D=D, H=gru_hidden).to(device)
            state_g = torch.load(gru_ckpt, map_location=device)
            if isinstance(state_g, dict) and "state_dict" in state_g:
                state_g = state_g["state_dict"]
            gru.load_state_dict(state_g, strict=True)
            gru.eval()
            logging.info(f"Loaded GRU-AR checkpoint: {gru_ckpt}")
        else:
            logging.warning(f"GRU checkpoint not found, skipping GRU tests: {gru_ckpt}")

    batch = next(iter(test_loader))
    ts = batch["times"].to(device)
    xs = batch["xs"].to(device)
    ys = batch["ys"].to(device)
    mask = batch["mask"].to(device)
    hidden_mask = batch["hidden_mask"].to(device)
    modes = batch["modes"]
    grid_ids = batch["grid_ids"]

    b = int(sample_index_in_batch)
    if b >= xs.shape[1]:
        raise ValueError(f"sample_index_in_batch={b} but batch has size {xs.shape[1]}")

    R_scalar = obs_noise_std ** 2

    with torch.no_grad():
        sde_samples = model.euler_maruyama_posterior(
            obs=ys[:, b:b+1, :],
            ts=ts,
            mask=mask[:, b:b+1, :],
            R_scalar=R_scalar,
            L=L_samples,
            gain=gain,
            use_explicit_likelihood=use_explicit_likelihood,
            x0=None,
        ).squeeze(2)

        gru_samples = None
        if gru is not None:
            gru_samples = gru.sample_posterior(
                y=ys[:, b:b+1, :],
                mask=mask[:, b:b+1, :],
                times=ts,
                L=L_samples,
                feedback=gru_feedback_eval,
                warm_start_from_obs=gru_warm_start_from_obs,
                x0=None,
            ).squeeze(2)

    hidden_time_mask_1d = (hidden_mask[:, b, :].sum(dim=-1) > 0).float()

    sde_w2_path = pathwise_wasserstein_to_truth(
        x_samples=sde_samples,
        x_true=xs[:, b, :],
        eval_time_mask=None,
        p=2,
    )
    sde_w2_hidden = pathwise_wasserstein_to_truth(
        x_samples=sde_samples,
        x_true=xs[:, b, :],
        eval_time_mask=hidden_time_mask_1d,
        p=2,
    )
    sde_energy_path = path_energy_score(
        x_samples=sde_samples,
        x_true=xs[:, b, :],
        eval_time_mask=None,
    )
    sde_energy_hidden = path_energy_score(
        x_samples=sde_samples,
        x_true=xs[:, b, :],
        eval_time_mask=hidden_time_mask_1d,
    )

    gru_w2_path = float("nan")
    gru_w2_hidden = float("nan")
    gru_energy_path = float("nan")
    gru_energy_hidden = float("nan")

    if gru_samples is not None:
        gru_w2_path = pathwise_wasserstein_to_truth(
            x_samples=gru_samples,
            x_true=xs[:, b, :],
            eval_time_mask=None,
            p=2,
        )
        gru_w2_hidden = pathwise_wasserstein_to_truth(
            x_samples=gru_samples,
            x_true=xs[:, b, :],
            eval_time_mask=hidden_time_mask_1d,
            p=2,
        )
        gru_energy_path = path_energy_score(
            x_samples=gru_samples,
            x_true=xs[:, b, :],
            eval_time_mask=None,
        )
        gru_energy_hidden = path_energy_score(
            x_samples=gru_samples,
            x_true=xs[:, b, :],
            eval_time_mask=hidden_time_mask_1d,
        )

    x_true = xs[:, b, :]
    mask_b = mask[:, b, :]
    hidden_b = hidden_mask[:, b, :]
    mode_b = int(modes[b].item())
    gid_b = int(grid_ids[b].item())

    sde_w2_per_dim_hidden = pathwise_w2_per_dim(
        x_samples=sde_samples,
        x_true=x_true,
        eval_time_mask=hidden_time_mask_1d,
    )

    gru_w2_per_dim_hidden = None
    if gru_samples is not None:
        gru_w2_per_dim_hidden = pathwise_w2_per_dim(
            x_samples=gru_samples,
            x_true=x_true,
            eval_time_mask=hidden_time_mask_1d,
        )

    mean = stats["mean"].to(device=device)
    std = stats["std"].to(device=device)

    x_true_den = (x_true * std + mean).detach().cpu().numpy()
    x_sde_mu_den = (sde_samples.mean(dim=0) * std + mean).detach().cpu().numpy()
    x_gru_mu_den = None
    if gru_samples is not None:
        x_gru_mu_den = (gru_samples.mean(dim=0) * std + mean).detach().cpu().numpy()

    frame_idx = np.linspace(0, x_true_den.shape[0] - 1, max(1, 6)).round().astype(int)

    render_truth_recon_frames(
        out_path=os.path.join(out_dir, "frames_compare.png"),
        x_true_14=x_true_den,
        x_recon_14=x_sde_mu_den,
        frame_indices=frame_idx,
        title=f"Hopper sample b={b}, mode={mode_b}, grid={gid_b}",
        width=240,
        height=240,
        x_recon2_14=x_gru_mu_den,
        recon_label="LatentSDE Recon",
        recon2_label="GRU-AR Recon",
    )

    plot_spaghetti(
        out_path=os.path.join(out_dir, "spaghetti_sde.png"),
        ts=ts,
        x_true=x_true,
        x_samples=sde_samples,
        mask=mask_b,
        hidden_mask=hidden_b,
        dims=(0, 1, 2),
        max_paths=min(50, L_samples),
        title="LatentSDE posterior spaghetti",
    )

    if gru_samples is not None:
        plot_spaghetti(
            out_path=os.path.join(out_dir, "spaghetti_gru.png"),
            ts=ts,
            x_true=x_true,
            x_samples=gru_samples,
            mask=mask_b,
            hidden_mask=hidden_b,
            dims=(0, 1, 2),
            max_paths=min(50, L_samples),
            title="GRU-AR posterior spaghetti",
        )

    plot_histograms_hidden_times(
        out_path=os.path.join(out_dir, "hidden_hist_sample.png"),
        ts=ts,
        x_true=x_true,
        sde_samples=sde_samples,
        gru_samples=gru_samples,
        hidden_mask=hidden_b,
        dims=(0, 1, 2),
        bins=40,
        max_times=4,
        title="Posterior histograms at hidden times",
    )

    plot_true_mode_histograms(
        out_path=os.path.join(out_dir, "true_mode_histograms.png"),
        x_all=payload_test["x"],
        mode_all=payload_test["mode"],
        ts=payload_test["ts_bank"][gid_b],
        dims=(0, 1, 2),
        time_fracs=(0.38, 0.52, 0.68, 0.90),
        bins=50,
        title="True state histograms by mode and pooled",
    )

    hidden_times = torch.where(hidden_time_mask_1d > 0)[0].detach().cpu().tolist()

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"sample_index_in_batch: {b}\n")
        f.write(f"mode: {mode_b}\n")
        f.write(f"grid_id: {gid_b}\n")
        f.write(f"num_hidden_times: {len(hidden_times)}\n")
        f.write(f"hidden_time_indices: {hidden_times}\n")
        f.write(f"selected_time_grid: {ts.cpu().numpy().tolist()}\n")
        f.write(f"force_mask_future_window: {force_mask_future_window}\n")
        f.write(f"future_window_min_frac: {future_window_min_frac}\n")
        f.write(f"future_window_max_frac: {future_window_max_frac}\n")
        f.write(f"num_hidden_windows: {num_hidden_windows}\n")
        f.write(f"hist_dims: [0, 1, 2]\n")
        f.write(f"L_samples: {L_samples}\n")
        f.write(f"model_ckpt: {model_ckpt}\n")
        f.write(f"gru_ckpt: {gru_ckpt if gru_ckpt else 'None'}\n")
        f.write(f"sde_w2_path: {sde_w2_path:.6f}\n")
        f.write(f"sde_w2_hidden: {sde_w2_hidden:.6f}\n")
        f.write(f"sde_energy_path: {sde_energy_path:.6f}\n")
        f.write(f"sde_energy_hidden: {sde_energy_hidden:.6f}\n")
        f.write(f"gru_w2_path: {gru_w2_path:.6f}\n")
        f.write(f"gru_w2_hidden: {gru_w2_hidden:.6f}\n")
        f.write(f"gru_energy_path: {gru_energy_path:.6f}\n")
        f.write(f"gru_energy_hidden: {gru_energy_hidden:.6f}\n")
        f.write("sde_w2_per_dim_hidden:\n")
        for d, val in sde_w2_per_dim_hidden.items():
            f.write(f"  dim_{d}: {val:.6f}\n")

        if gru_w2_per_dim_hidden is not None:
            f.write("gru_w2_per_dim_hidden:\n")
            for d, val in gru_w2_per_dim_hidden.items():
                f.write(f"  dim_{d}: {val:.6f}\n")

    hist_data = collect_hidden_hist_data_over_testset(
        model=model,
        gru=gru,
        test_loader=test_loader,
        device=device,
        R_scalar=R_scalar,
        L_samples=min(128, L_samples),
        gain=gain,
        use_explicit_likelihood=use_explicit_likelihood,
        gru_feedback_eval=gru_feedback_eval,
        gru_warm_start_from_obs=gru_warm_start_from_obs,
        dims=(1, 3, 5, 7, 9),
        time_fracs=(0.38, 0.52, 0.68, 0.90),
        max_eval_batches=3,
    )

    plot_aggregate_hidden_histograms(
        out_path=os.path.join(out_dir, "agg_hidden_hist_final.png"),
        hist_data=hist_data,
        dims=(1, 3, 5, 7, 9),
        bins=50,
        title="Aggregated hidden-state histograms over masked test set (final)",
    )

    metrics = evaluate_hidden_distribution_metrics(
        hist_data=hist_data,
        dims=(1, 3, 5, 7, 9),
    )
    write_hidden_distribution_metrics(
        out_path=os.path.join(out_dir, "hidden_distribution_metrics.csv"),
        metrics=metrics,
        dims=(1, 3, 5, 7, 9),
    )

    logging.info(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
