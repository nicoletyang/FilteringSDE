# FilteringSDEs

Neural path filtering of stochastic dynamical systems under partial observations. Code for the paper:

> Yang, Nicole Tianjiao. "Pathwise Learning of Stochastic Dynamical Systems with Partial Observations." arXiv:2601.21860 (2026).

We learn a conditional generative model that produces posterior distributions over latent trajectories, amortized over observation paths. The method handles noisy, partial, and irregularly-timed observations, and supports both filtering (causal) and smoothing (non-causal) inference at test time without retraining.

Experiments cover stochastic **Lorenz-63**, **Lorenz-96**, and **MuJoCo Hopper** data.

---

## Results

### Lorenz-63: posterior mean vs. ground truth (Ours vs. PF vs. PG smoother)

![L63 benchmark](figs/l63_triptych.png)

### Lorenz-63: inference-budget efficiency curve (RMSE and W₁ vs. # samples / particles)

Ours matches particle-filter accuracy with far fewer samples because inference is amortized.

![L63 budget curves](figs/l63_budget.png)

---

### Lorenz-96: learned posterior on 15-dimensional stochastic system

The model is trained on the time interval [0, 2] with observation model y_t = tanh(x_t) + N(0, 0.15²) and 20% missing observations during training, 50% at test time. Inference is performed on an extended test horizon [0, 4].

**Marginal posterior distributions** at t = 0.5, 1.0, 1.5 for dimension 1 — posterior samples (histogram) tightly bracket the ground-truth value (blue dashed line) across all three snapshots:

![L96 marginal histograms](figs/hist_vs_truth-1.png)

**True vs. inferred trajectories** for the first 3 dimensions, with 90% credible intervals:

![L96 trajectory comparison](figs/comparenew_final-1.png)

---

## Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

All scripts are run from the repo root.

### Lorenz-63

**Train 5 seeds in one command** (each seed saved to `./dump/l63/seed<N>/`):

```bash
python lorenz63.py train_seeds \
    --seeds "[0,1,2,3,4]" \
    --base_dir ./dump/l63/ \
    --num_iters 500
```

**Train a single seed** (auto-names directory by seed):

```bash
python lorenz63.py train \
    --seed 0 \
    --base_dir ./dump/l63/ \
    --num_iters 500
```

---

### Lorenz-96

```bash
python lorenz96.py \
    --seed 0 \
    --missing_rate_train 0.2 \
    --missing_rate_test 0.5 \
    --train_dir ./dump/l96/
```

---

### MuJoCo Hopper (evaluation only — requires a trained checkpoint)

```bash
python mujoco.py \
    --model_ckpt ./path/to/model.pth \
    --data_dir   ./hopper_data/ \
    --out_dir    ./hopper_eval/
```

---

## Custom posterior integrator (`sdeint_obs.py`)

Standard `torchsde.sdeint` does not support passing observations into the drift at each solver step. `sdeint_obs` is a self-contained Euler-Maruyama integrator that does exactly this, acting as a drop-in replacement for `torchsde.sdeint(..., logqp=True)`:

```python
from sdeint_obs import sdeint_obs

# posterior path: observation tensor is passed to sde.f and sde.h at every step
zs, log_ratio = sdeint_obs(sde, z0, obs, ts, dt=1e-2, logqp=True, method="euler")

# prior path: unchanged, uses plain torchsde
zs = torchsde.sdeint(sde, z0, ts, dt=1e-2, method="euler")
```

The SDE class must expose:
- `f(t, y, obs)` — posterior (obs-conditioned) drift
- `h(t, y, obs)` — prior drift (only needed when `logqp=True`)
- `g(t, y)` — diagonal diffusion (no obs)
- `noise_type = "diagonal"`, `sde_type = "ito"`

---

## References

[1] Yang, Nicole Tianjiao. "Pathwise Learning of Stochastic Dynamical Systems with Partial Observations." arXiv:2601.21860 (2026).

[2] Li, Xuechen, et al. "Scalable gradients for stochastic differential equations." AISTATS 2020.
