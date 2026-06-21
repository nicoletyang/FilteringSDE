# FilteringSDEs
Neural path filtering of stochastic dynamical systems under partial observations. Code accompanying the paper [1]

We explore robust and accurate surrogate modeling of stochastic dynamical systems with noisy and partial observations, with the capability of uncertainty quantification and online inference for both filtering marginals and trajectory-dependent functionals. The method is a conditional generative model that gives posterior distribution of the system amortized over observation paths. The code is modified based on Latent SDEs [2].

Experiments on learning stochastic Lorenz-63, Lorenz-96 and MuJoCo simulator data from only noisy and partial observations.

## Dependencies

```bash
pip install -r requirements.txt
```


## Usage

Stochastic Lorenz 96 experiment

```bash
cd /home/tyang23/Documents/AmortizedDA

python lorenz96.py \
    --seed 0 \
    --missing_rate_train 0.2 \
    --missing_rate_test 0.5 \
    --train_dir ./dump/my_run/
```

### Posterior sampler `sdeint_obs.py`

In lorenz96.py, the posterior SDE integrator needs to pass observation context `obs` into the drift
`f(t, z, obs)` at every solver step. This is implemented as a Euler-Maruyama integrator in `sdeint_obs.py`.

**Prior SDE** (no observations needed): uses standard `torchsde.sdeint`.  
**Posterior SDE** (obs-conditioned drift): uses `sdeint_obs` from this repo.

The API is a drop-in match for `torchsde.sdeint(..., logqp=True)`:

```python
from sdeint_obs import sdeint_obs

# posterior path with observation conditioning
zs, log_ratio = sdeint_obs(sde, z0, obs, ts, dt=1e-2, logqp=True, method="euler")

# prior path — unchanged, uses plain torchsde
zs = torchsde.sdeint(sde, x0, ts, dt=1e-2, method="euler")
```

The SDE class must expose:
- `f(t, y, obs)` — posterior (obs-conditioned) drift
- `h(t, y, obs)` — prior drift (only needed when `logqp=True`)
- `g(t, y)` — diagonal diffusion (no obs needed)
- `noise_type = "diagonal"`, `sde_type = "ito"`

# References

[1] Yang, Nicole Tianjiao. "Pathwise Learning of Stochastic Dynamical Systems with Partial Observations." arXiv preprint arXiv:2601.21860 (2026). 

[2] Li, Xuechen, Ting-Kam Leonard Wong, Ricky TQ Chen, and David Duvenaud. "Scalable gradients for stochastic differential equations." In International conference on artificial intelligence and statistics, pp. 3870-3882. PMLR, 2020.
