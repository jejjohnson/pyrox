---
status: draft
version: 0.1.0
---

# Models — Model Zoo

Pre-built NumPyro model functions for common extreme value workflows. Each model is a standard NumPyro `model()` function that can be passed directly to `numpyro.infer.MCMC` or `numpyro.infer.SVI`.


## Stationary GEV

The simplest block maxima model. Assumes iid annual maxima from a fixed GEV.

```python
import numpyro
from numpyro.infer import MCMC, NUTS
from xtremax.models import stationary_gev

kernel = NUTS(stationary_gev)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000)
mcmc.run(rng_key, obs=annual_maxima)
```

Priors: `loc ~ Normal`, `scale ~ HalfNormal`, `concentration ~ Normal(0, 0.3)`.


## Nonstationary GEV

GEV parameters vary as functions of covariates (time, GMST, elevation, etc.):

```
μ(t) = μ₀ + μ₁·x(t)
log σ(t) = σ₀ + σ₁·x(t)
ξ(t) = ξ₀  (or optionally covariate-dependent)
```

```python
from xtremax.models import nonstationary_gev

mcmc.run(rng_key, obs=annual_maxima, covariates=gmst_anomaly)
```


## Spatial GEV with Gaussian Processes (GPJax)

Borrows strength across locations by modeling GEV parameters as spatial Gaussian process fields. Uses **GPJax** for all GP functionality — no hand-rolled GPs.

```
μ(s) = X(s)ᵀβ_μ + f_μ(s),    f_μ ~ GP(0, k_μ)
log σ(s) = X(s)ᵀβ_σ + f_σ(s), f_σ ~ GP(0, k_σ)
ξ(s) = X(s)ᵀβ_ξ + f_ξ(s),    f_ξ ~ GP(0, k_ξ)
```

Where X(s) can include elevation, latitude, distance-to-coast, etc. See api/components.md §GP Layer for the full GP layer API.

```python
from xtremax.models import spatial_gev

mcmc.run(
    rng_key,
    obs=block_maxima,        # (n_sites, n_years)
    coordinates=site_coords, # (n_sites, 2)
    covariates=elevation,    # (n_sites, n_covariates)
)
```


## Peaks-Over-Threshold GPD

Threshold exceedance modeling. Combines a Poisson rate for exceedance frequency with GPD for exceedance magnitudes. Works with any threshold from the three-tier system (see api/components.md §Threshold Selection): constant, trend, or parametric.

```python
from xtremax.models import pot_gpd
from xtremax.xarray import quantile_threshold, trend_quantile_threshold

# Constant threshold POT
u = quantile_threshold(da, q=0.95)
mcmc.run(rng_key, exceedances=excess_values, threshold=u, n_total=n_obs)

# Nonstationary threshold POT — threshold tracks a warming trend
u_trend = trend_quantile_threshold(da, q=0.95, time_dim="time", covariates=gmst)
mcmc.run(rng_key, exceedances=excess_values, threshold=u_trend, n_total=n_obs)
```


## Point Process Extremes

The point process characterization of extremes — unifies block maxima and POT into a single likelihood. Exceedances of a high threshold are modeled as a Poisson process with GEV-parameterized intensity.

```python
from xtremax.models import point_process_extreme

mcmc.run(rng_key, obs=daily_data, threshold=threshold, n_years=n_years)
```
