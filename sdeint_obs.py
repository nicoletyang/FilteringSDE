# Copyright 2024 (see LICENSE in this repository)
#
# sdeint_obs.py
# Self-contained Euler-Maruyama integrator for observation-conditioned SDEs.
#
# This replaces the vendor-patched torchsde used in development.  The only
# change relative to the published torchsde package is that the drift receives
# the observation tensor `obs` at every solver step:
#
#   sde.f(t, y, obs)   -- posterior drift
#   sde.h(t, y, obs)   -- prior drift  (only if logqp=True)
#   sde.g(t, y)        -- diffusion    (unchanged; no obs needed)
#
# Supported: Ito SDEs, diagonal noise, fixed-step Euler-Maruyama only.
# For the prior SDE (no observation conditioning) use the standard
# torchsde.sdeint as usual.

from __future__ import annotations

import torch
from torch import Tensor
from torchsde._brownian import BrownianInterval


def sdeint_obs(
    sde,
    y0: Tensor,
    obs: Tensor,
    ts: Tensor,
    *,
    dt: float = 1e-3,
    logqp: bool = False,
    method: str = "euler",
    **unused_kwargs,
):
    """Integrate an observation-conditioned Ito SDE with Euler-Maruyama.

    The SDE class must expose:
        sde.f(t, y, obs)  -> Tensor (B, D)   posterior drift
        sde.g(t, y)       -> Tensor (B, D)   diagonal diffusion
        sde.h(t, y, obs)  -> Tensor (B, D)   prior drift (needed iff logqp=True)
        sde.noise_type == "diagonal"
        sde.sde_type      == "ito"

    Arguments:
        sde:    SDE object (see above).
        y0:     (B, D) initial state tensor.
        obs:    Observation tensor passed as-is to sde.f / sde.h at each step.
        ts:     1-D Tensor of query times in strictly increasing order.
        dt:     Fixed Euler step size.
        logqp:  If True, also return the incremental KL path penalty.
                The returned log_ratio has shape (T-1, B), matching the
                convention of torchsde.sdeint(..., logqp=True).
                Calling code accumulates it with ``log_ratio.sum(0).mean()``.
        method: Must be "euler" (only Euler-Maruyama is implemented).

    Returns:
        ys          : Tensor (T, B, D)  -- states at each query time.
        log_ratio   : Tensor (T-1, B)   -- per-interval KL increments
                      (only returned when logqp=True).
    """
    if method != "euler":
        raise ValueError(
            f"sdeint_obs only supports method='euler', got '{method}'."
        )
    assert getattr(sde, "noise_type", None) == "diagonal", (
        "sdeint_obs requires sde.noise_type == 'diagonal'."
    )
    assert getattr(sde, "sde_type", None) == "ito", (
        "sdeint_obs requires sde.sde_type == 'ito'."
    )

    if not torch.is_tensor(ts):
        ts = torch.tensor(ts, dtype=y0.dtype, device=y0.device)

    bm = BrownianInterval(
        t0=ts[0],
        t1=ts[-1],
        size=y0.shape,
        dtype=y0.dtype,
        device=y0.device,
        levy_area_approximation="none",
    )

    return _euler_integrate(sde, y0, obs, ts, float(dt), logqp, bm)


# ---------------------------------------------------------------------------
# Internal integrator
# ---------------------------------------------------------------------------

def _euler_integrate(sde, y0, obs, ts, dt, logqp, bm):
    B, D = y0.shape
    device, dtype = y0.device, y0.dtype

    curr_t = ts[0]
    curr_y = y0
    # Accumulated KL penalty per batch element (scalar per B).
    curr_lr = torch.zeros(B, device=device, dtype=dtype) if logqp else None

    ys = [y0]                                       # list of (B, D)
    lr_vals = [curr_lr.clone()] if logqp else None  # list of (B,)

    prev_t = curr_t
    prev_y = curr_y

    for out_t in ts[1:]:
        # Take fixed Euler steps until we reach out_t.
        while curr_t < out_t:
            next_t = min(curr_t + dt, out_t)
            step_dt = next_t - curr_t

            dW = bm(curr_t, next_t)          # (B, D) Brownian increment

            f = sde.f(curr_t, curr_y, obs)   # (B, D)
            g = sde.g(curr_t, curr_y)        # (B, D)

            if logqp:
                h = sde.h(curr_t, curr_y, obs)  # (B, D)
                # Stable division (avoid /0 where diffusion is tiny).
                safe_g = g.abs().clamp_min(1e-8) * g.sign().masked_fill(g == 0, 1.0)
                u = (f - h) / safe_g                # (B, D)
                # Deterministic KL rate (g_logqp = 0 in SDELogqp).
                curr_lr = curr_lr + 0.5 * (u ** 2).sum(dim=1) * step_dt

            prev_t, prev_y = curr_t, curr_y
            curr_y = curr_y + f * step_dt + g * dW
            curr_t = next_t

        # Linear interpolation to the exact query time.
        span = curr_t - prev_t
        if span > 0:
            alpha = (out_t - prev_t) / span
            interp_y = prev_y + alpha * (curr_y - prev_y)
        else:
            interp_y = curr_y

        ys.append(interp_y)
        if logqp:
            lr_vals.append(curr_lr.clone())

    ys_tensor = torch.stack(ys, dim=0)  # (T, B, D)

    if logqp:
        lr_tensor = torch.stack(lr_vals, dim=0)  # (T, B)
        # Return per-interval increments, mirroring torchsde's convention.
        log_ratio_increments = lr_tensor[1:] - lr_tensor[:-1]  # (T-1, B)
        return ys_tensor, log_ratio_increments

    return ys_tensor
