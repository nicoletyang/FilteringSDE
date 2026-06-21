# FilteringSDEs

Neural path filtering of stochastic dynamical systems under partial observations. Code for the paper:

> Yang, Nicole Tianjiao. "Pathwise Learning of Stochastic Dynamical Systems with Partial Observations." arXiv:2601.21860 (2026).

We learn a conditional generative model that produces posterior distributions over latent trajectories, amortized over observation paths. The method handles noisy, partial, and irregularly-timed observations, and supports estimation of path functionals and the corresponding uncertainty quantification for new observations on updated time windows without retraining.

Experiments cover stochastic **Lorenz-63**, **Lorenz-96**, and **MuJoCo Hopper** data.

---

## Example

### Lorenz-96: learned posterior on 15-dimensional stochastic system

The model is trained on the time interval [0, 2] with observation model y_t = tanh(x_t) + N(0, 0.15²) and 20% missing observations during training.

**Quantitative comparison across missing rates** — model trained on [0, 3] with 20% observations masked; evaluated over 10 seeds against particle filter (PF, 512 particles) and particle smoother (PG, 512 particles). Our method uses only 64 posterior samples and yields better results in the comparison below.

<table>
<thead>
<tr>
  <th>Missing Rate</th>
  <th>Method</th>
  <th>RMSE</th>
  <th>W₁</th>
</tr>
</thead>
<tbody>
<tr>
  <td rowspan="3">10%</td>
  <td>Ours</td>
  <td><span style="color:green">0.141 ± 0.002</span></td>
  <td><span style="color:green">0.104 ± 0.002</span></td>
</tr>
<tr>
  <td>PF</td>
  <td>0.223 ± 0.004</td>
  <td>0.128 ± 0.010</td>
</tr>
<tr>
  <td>PG</td>
  <td>0.219 ± 0.005</td>
  <td>0.106 ± 0.006</td>
</tr>
<tr>
  <td rowspan="3">30%</td>
  <td>Ours</td>
  <td><span style="color:green">0.159 ± 0.002</span></td>
  <td><span style="color:green">0.117 ± 0.002</span></td>
</tr>
<tr>
  <td>PF</td>
  <td>0.281 ± 0.004</td>
  <td>0.182 ± 0.016</td>
</tr>
<tr>
  <td>PG</td>
  <td>0.280 ± 0.004</td>
  <td>0.153 ± 0.020</td>
</tr>
<tr>
  <td rowspan="3">50%</td>
  <td>Ours</td>
  <td><span style="color:green">0.195 ± 0.002</span></td>
  <td><span style="color:green">0.137 ± 0.004</span></td>
</tr>
<tr>
  <td>PF</td>
  <td>0.377 ± 0.004</td>
  <td>0.237 ± 0.021</td>
</tr>
<tr>
  <td>PG</td>
  <td>0.378 ± 0.003</td>
  <td>0.242 ± 0.026</td>
</tr>
</tbody>
</table>

---


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

### MuJoCo Hopper

**Train** (generates data automatically via `dm_control`, saves checkpoints to `train_dir`):

```bash
python mujoco.py train \
    --data_dir  ./hopper_data/ \
    --train_dir ./hopper_runs/ \
    --num_iters 8000
```

Checkpoints are saved as `model_step_NNNNN.pth` / `gru_ar_step_NNNNN.pth` every `eval_every` steps, plus `model_best.pth` for the best W₂ on the hidden window.

**Evaluate** a trained checkpoint:

```bash
python mujoco.py evaluate \
    --model_ckpt ./hopper_runs/model_best.pth \
    --gru_ckpt   ./hopper_runs/gru_ar_best.pth \
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

[2] Li, Xuechen and Wong, Ting-Kam Leonard and Chen, Ricky T. Q. and Duvenaud, David. "Scalable gradients for stochastic differential equations." Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, PMLR 108:3870-3882, 2020.
