---
status: draft
version: 0.2.0
---

# Layer 0 — Primitive Examples

Pure JAX functions. No NumPyro, no xarray — just arrays in, arrays out.

---

## GEV Log-Density

### Evaluate the GEV density directly

```python
import jax.numpy as jnp
from xtremax._src.gev import gev_log_prob, gev_cdf, gev_return_level

# Pure function call — no Distribution class needed
x = jnp.array([30.0, 35.0, 40.0])
lp = gev_log_prob(x, loc=30.0, scale=2.5, shape=0.1)  # (3,)

# CDF
p = gev_cdf(x, loc=30.0, scale=2.5, shape=0.1)  # (3,)

# Differentiable — gradient of log-prob w.r.t. location
grad_loc = jax.grad(lambda mu: gev_log_prob(35.0, mu, 2.5, 0.1))(30.0)
```

---

## Return Levels

### T-year return level as a pure function

```python
from xtremax._src.gev import gev_return_level

# 100-year return level: z_T = μ + σ/ξ * [(-log(1-1/T))^(-ξ) - 1]
rl_100 = gev_return_level(period=100, loc=30.0, scale=2.5, shape=0.1)
rl_50 = gev_return_level(period=50, loc=30.0, scale=2.5, shape=0.1)

# Vectorize over parameters (e.g., posterior samples)
locs = jnp.array([29.5, 30.0, 30.5])
scales = jnp.array([2.3, 2.5, 2.7])
shapes = jnp.array([0.08, 0.10, 0.12])
rl_batch = jax.vmap(lambda l, s, sh: gev_return_level(100, l, s, sh))(locs, scales, shapes)
```

---

## GPD Survival Function

### Exceedance probability for threshold exceedances

```python
from xtremax._src.gpd import gpd_survival, gpd_return_level

# P(Y > 5) where Y is the excess above threshold
p_exceed = gpd_survival(y=5.0, scale=2.0, shape=0.15)

# POT return level: z_T = u + σ/ξ * [(λT)^ξ - 1]
rl_100 = gpd_return_level(period=100, scale=2.0, shape=0.15, rate=10.0)
```

---

## Point Process Likelihood

### Hawkes self-exciting process — pure log-likelihood

```python
from xtremax._src.point_process import hawkes_log_likelihood

event_times = jnp.array([1.2, 3.5, 3.8, 7.1, 7.3, 7.5, 12.0])

# Pure JAX — no Distribution class
ll = hawkes_log_likelihood(
    event_times, background_rate=0.5, alpha=0.3, beta=1.0,
)

# Differentiable — gradient w.r.t. background rate
grad_mu = jax.grad(
    lambda mu: hawkes_log_likelihood(event_times, mu, 0.3, 1.0)
)(0.5)
```

---

## Max-Stable Extremal Coefficients

### Dependence measures as pure functions

```python
from xtremax._src.max_stable import brown_resnick_extremal_coeff
from xtremax._src.variogram import matern_variogram

# Compute extremal coefficient θ(h) for a range of distances
distances = jnp.linspace(0, 500, 100)  # km
gamma_h = matern_variogram(distances, range=150.0, smoothness=1.5, sill=2.0)
theta_h = brown_resnick_extremal_coeff(distances, lambda h: gamma_h)

# θ=1: perfect dependence, θ=2: independence
# Researchers can plot θ(h) without fitting a model
```

---

## Composing L0 Functions

### Build custom likelihoods from primitives

```python
from xtremax._src.gev import gev_log_prob
from xtremax._src.gpd import gpd_log_prob

# Custom composite likelihood — GEV for maxima + GPD for exceedances
def joint_log_lik(maxima, exceedances, loc, scale, shape, gpd_scale, gpd_shape):
    ll_gev = jnp.sum(gev_log_prob(maxima, loc, scale, shape))
    ll_gpd = jnp.sum(gpd_log_prob(exceedances, gpd_scale, gpd_shape))
    return ll_gev + ll_gpd

# Fully differentiable — use with any optimizer or sampler
grad_fn = jax.grad(joint_log_lik, argnums=(2, 3, 4, 5, 6))
```
