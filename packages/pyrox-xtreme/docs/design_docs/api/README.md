---
status: draft
version: 0.1.0
---

# API Overview

## Surface Inventory

### Layer 0 — Primitives (`xtremax._src`)

Pure JAX functions. Stateless, differentiable, no NumPyro dependency.

| Submodule | Key Functions | Description |
|-----------|---------------|-------------|
| `_src.gev` | `gev_log_prob`, `gev_cdf`, `gev_icdf`, `gev_return_level`, `gev_mean`, `gev_variance` | GEV density, CDF, quantile, return level, moments |
| `_src.gpd` | `gpd_log_prob`, `gpd_cdf`, `gpd_survival`, `gpd_return_level`, `gpd_mean_excess` | GPD density, CDF, survival, return level, mean excess |
| `_src.point_process` | `poisson_log_likelihood`, `inhomogeneous_poisson_log_likelihood`, `hawkes_log_likelihood`, `extremal_index` | Point process likelihoods and diagnostics |
| `_src.max_stable` | `brown_resnick_extremal_coeff`, `smith_extremal_coeff`, `schlather_extremal_coeff`, `pairwise_log_likelihood`, `madogram` | Max-stable dependence functions and composite likelihood |
| `_src.variogram` | `power_variogram`, `matern_variogram` | Semivariogram functions for spatial models |

See: [primitives.md](primitives.md)

### Layer 1 — Components

NumPyro `Distribution` subclasses (wrapping L0 functions), xarray utilities, plotting, simulations, datasets, GP wrappers.

**Distributions** (NumPyro wrappers around L0 math):

| Class | Wraps | Description |
|-------|-------|-------------|
| `GEVD` | `gev_*` functions | Full three-parameter GEV distribution |
| `GPD` | `gpd_*` functions | Generalized Pareto for threshold exceedances |
| `Gumbel` | `gev_*` with ξ=0 | Light-tailed extreme value |
| `Frechet` | `gev_*` with ξ>0 | Heavy-tailed extreme value |
| `Weibull` | `gev_*` with ξ<0 | Bounded upper tail |
| `HawkesProcess` | `hawkes_log_likelihood` | Self-exciting point process |
| `HomogeneousPoissonProcess` | `poisson_log_likelihood` | Constant-rate events |
| `BrownResnickProcess` | `pairwise_log_likelihood` + `brown_resnick_extremal_coeff` | Max-stable spatial process |
| `SmithProcess`, `SchlatherProcess`, `ExtremalTProcess` | Respective L0 functions | Other max-stable models |

**xarray utilities, plotting, simulations, datasets, GP:**

| Module | Key Exports | Description |
|--------|-------------|-------------|
| `xarray.block_maxima` | `temporal_block_maxima`, `spatial_block_maxima`, `sliding_block_maxima`, `r_largest_block_maxima` | Block maxima extraction |
| `xarray.threshold` | `quantile_threshold`, `trend_quantile_threshold`, `parametric_quantile_threshold` | Three-tier threshold selection |
| `xarray.decluster` | `decluster_runs`, `decluster_separation`, `estimate_extremal_index` | Declustering utilities |
| `xarray.masks` | `land_sea_mask`, `coverage_mask`, `quality_mask`, `combine_masks` | Spatial/temporal/quality masking |
| `plotting.*` | `plot_qq`, `plot_return_level`, `plot_return_level_map`, `plot_posterior`, ... | Diagnostic, spatial, temporal, model plots |
| `simulations.*` | `generate_gmst_trajectory`, `create_domain`, `simulate_temp_extremes`, ... | Synthetic data generation |
| `datasets.*` | `ghcnd`, `gsod`, `ndbc`, `uhslc`, `gesla`, `coops`, `covariates` | Climate/ocean observation loaders |
| `gp` | `VariationalGP`, `SparseVariationalGP` | GPJax integration wrappers |

See: [components.md](components.md)

### Layer 2 — Models (`xtremax.models`)

| Module | Key Exports | Description |
|--------|-------------|-------------|
| `models.stationary_gev` | `stationary_gev` | iid GEV block maxima model |
| `models.nonstationary_gev` | `nonstationary_gev` | GEV with covariate-dependent parameters |
| `models.spatial_gev` | `spatial_gev` | GEV + GPJax spatial pooling |
| `models.pot_gpd` | `pot_gpd` | Peaks-over-threshold with GPD |
| `models.point_process` | `point_process_extreme` | Point process extreme value models |
| `models.max_stable` | `max_stable_composite` | Max-stable process fitting (composite likelihood) |

See: [models.md](models.md)

## Conventions

- All distributions subclass `numpyro.distributions.Distribution`
- All models are standard NumPyro `model()` functions compatible with `numpyro.infer.MCMC` and `numpyro.infer.SVI`
- All xarray utilities accept an optional `mask` parameter and preserve coordinates/metadata
- All dataset loaders return `xr.Dataset` with standardized units and coordinate names

## Notation

| Symbol | Meaning |
|---|---|
| $\mu$ (loc) | Location parameter |
| $\sigma$ (scale) | Scale parameter ($\sigma > 0$) |
| $\xi$ (concentration/shape) | Shape parameter ($\xi > 0$: heavy tail, $\xi = 0$: Gumbel, $\xi < 0$: bounded) |
| $z_T$ | $T$-year return level |
| $\theta$ | Extremal index ($0 < \theta \leq 1$) |
| $u$ | Threshold for POT analysis |
| $\lambda$ | Exceedance rate (expected exceedances per block) |

## Import Conventions

```python
# Layer 0 — Pure JAX functions (no NumPyro)
from xtremax._src.gev import gev_log_prob, gev_cdf, gev_return_level
from xtremax._src.gpd import gpd_log_prob, gpd_survival, gpd_return_level
from xtremax._src.point_process import hawkes_log_likelihood
from xtremax._src.max_stable import brown_resnick_extremal_coeff, pairwise_log_likelihood

# Layer 1 — NumPyro distributions (wrap L0)
from xtremax.distributions import GEVD, GPD, Gumbel, Frechet, Weibull
from xtremax.distributions.point_process import HawkesProcess, HomogeneousPoissonProcess
from xtremax.distributions.max_stable import BrownResnickProcess

# Layer 1 — xarray utilities
from xtremax.xarray import block_maxima, threshold, decluster

# Layer 1 — datasets
from xtremax.datasets import ghcnd, ndbc
from xtremax.datasets.covariates import global_warming, enso

# Layer 2 — Models (NumPyro model functions)
from xtremax.models import stationary_gev, nonstationary_gev, spatial_gev, pot_gpd

# Inference (from NumPyro — not xtremax)
from numpyro.infer import MCMC, NUTS, SVI, Predictive
```

---

## Detail Files

| File | Covers |
|---|---|
| [primitives.md](primitives.md) | Layer 0 — pure JAX functions (GEV, GPD, point process, max-stable, variogram) |
| [components.md](components.md) | Layer 1 — NumPyro distributions, xarray utilities, plotting, simulations, datasets, GP, copulas |
| [models.md](models.md) | Layer 2 — model zoo (stationary, nonstationary, spatial GEV, POT, PP) |

---

*For usage patterns, see [../examples/](../examples/) — organized by layer to match this directory.*
