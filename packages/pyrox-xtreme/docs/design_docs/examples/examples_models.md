---
status: draft
version: 0.1.0
---

# Layer 2 — Model Examples

Model zoo workflows with NumPyro inference. *(P1: NumPyro-native, P4: composable)*

---

## Stationary GEV Block Maxima

### Simplest extreme value model — iid annual maxima

```python
from xtremax.models import stationary_gev
from xtremax.xarray import block_maxima
from numpyro.infer import MCMC, NUTS

# Extract annual maxima
annual_max = block_maxima.temporal_block_maxima(ds["temperature"], block="year")

# P1: model is a standard NumPyro function — plug into MCMC
kernel = NUTS(stationary_gev)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
mcmc.run(key, obs=annual_max.values)

# Posterior return levels
from numpyro.infer import Predictive
from xtremax.distributions import GEVD

samples = mcmc.get_samples()
gev_posterior = GEVD(samples["loc"], samples["scale"], samples["concentration"])
rl_100_samples = gev_posterior.return_level(100)  # posterior distribution of 100-yr level
```

---

## Nonstationary GEV with Covariates

### Location parameter depends on GMST warming trend

```python
from xtremax.models import nonstationary_gev

# GEV with μ(t) = β₀ + β₁·GMST(t)
mcmc.run(key, obs=annual_max.values, covariates=gmst_anomaly.values)
```

---

## Peaks-Over-Threshold with GPD

### Threshold exceedances with trend-following threshold

```python
from xtremax.models import pot_gpd
from xtremax.xarray import threshold, decluster

# Trend-following threshold
u = threshold.trend_quantile_threshold(ds["temperature"], q=0.95, covariates=gmst)

# Extract and decluster exceedances
exceedances = threshold.threshold_exceedances(ds["temperature"], u)
declustered = decluster.decluster_runs(exceedances, run_length=3)

# Fit GPD
mcmc.run(key, exceedances=declustered.values, threshold=u.values, n_total=len(ds.time))
```

---

## Spatial GEV with Gaussian Process

### Borrow strength across stations via GP spatial pooling

```python
from xtremax.models import spatial_gev

# GEV parameters vary smoothly over space:
# μ(s) = X(s)ᵀβ_μ + f_μ(s),  f_μ ~ GP(0, k_μ)
mcmc.run(
    key,
    obs=block_maxima_all_stations,   # (n_sites, n_years)
    coordinates=station_coords,       # (n_sites, 2)
    covariates=elevation,             # (n_sites, n_covariates)
)
```

---

## Composition Patterns

| Pattern | Components | Use Case |
|---|---|---|
| Stationary block maxima | `block_maxima` + `GEVD` + MCMC | Standard extreme value analysis |
| Nonstationary GEV | `GEVD` + covariate regression + SVI | Trend detection in extremes |
| POT analysis | `threshold_exceedances` + `decluster` + `GPD` | Threshold-based modeling |
| Spatial pooling | `GEVD` + `VariationalGP` + MCMC | Borrowing strength across locations |
| Point process | `HawkesProcess` / `PoissonProcess` + MCMC | Temporal clustering of extremes |
| End-to-end climate | `geo_toolz` preprocess → `xtremax` model → return levels | Full pipeline |
