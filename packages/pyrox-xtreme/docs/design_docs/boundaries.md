---
status: draft
version: 0.1.0
---

# Boundaries and Ecosystem

## Overview

xtremax owns extreme value distributions, Bayesian models, and xarray extraction utilities. It delegates inference to NumPyro, spatial GP modeling to GPJax, and data preprocessing to geo_toolz.

---

## Ownership Map

| Concern | Owner | Notes |
|---------|-------|-------|
| EVT distributions (GEV, GPD, Gumbel, Frechet, Weibull) | **xtremax** | NumPyro Distribution subclasses |
| Point process distributions (Hawkes, Poisson, renewal) | **xtremax** | NumPyro Distribution subclasses |
| Max-stable process models (Smith, Schlather, Brown-Resnick, Extremal-t) | **xtremax** | Composite pairwise likelihood |
| Model zoo (stationary/nonstationary/spatial GEV, POT, PP) | **xtremax** | NumPyro model functions |
| Block maxima extraction | **xtremax** | xarray utilities |
| Threshold selection (constant, trend, parametric) | **xtremax** | xarray utilities |
| Declustering and extremal index | **xtremax** | xarray utilities |
| EVT diagnostic plots (QQ, PP, return level, 4-panel) | **xtremax** | matplotlib-based |
| Simulation tools (synthetic extremes, GMST, spatial fields) | **xtremax** | For testing and demos |
| Dataset loaders (GHCN-D, GSOD, NDBC, UHSLC, GESLA, etc.) | **xtremax** | Standard climate/ocean obs |
| Climate covariate loaders (GMST, ONI, NAO, etc.) | **xtremax** | For nonstationary models |
| GP spatial pooling engine | **GPJax** | xtremax wraps via `gp/` module |
| Inference (MCMC, SVI, Predictive) | **NumPyro** | Never reimplemented |
| Data preprocessing (regrid, detrend, subset) | **geo_toolz** | Upstream of xtremax |
| Structured linear algebra | **gaussx** | Optional for GP covariance |
| ODE/SDE integration for simulations | **diffrax** | Optional for energy balance models |

---

## Decision Table

| Scenario | Recommendation |
|----------|---------------|
| Fit stationary GEV to annual maxima | `block_maxima` → `stationary_gev` → MCMC |
| Detect trend in extremes | `nonstationary_gev` with GMST covariate → SVI |
| Estimate 100-year return level | Fit GEV → `gev.return_level(100)` from posterior |
| Model threshold exceedances | `threshold_exceedances` → `decluster` → `pot_gpd` → MCMC |
| Spatial extreme analysis | `spatial_gev` with GPJax spatial pooling → MCMC |
| Model clustered extreme events | `HawkesProcess` or `point_process_extreme` → MCMC |
| Preprocess raw station data | geo_toolz (upstream) |
| Compare return levels across models | xtremax posterior samples → geo_toolz metrics |

---

## Ecosystem Interactions

| External Package | Integration Point | Pattern |
|---|---|---|
| **numpyro** | `Distribution` base class, inference | All xtremax distributions and models |
| **GPJax** | GP kernels, variational inference | `xtremax.gp` wraps GPJax for spatial models |
| **xarray** | Data interface for all utilities | Block maxima, thresholds, datasets |
| **geo_toolz** | Preprocessing upstream, evaluation downstream | Preprocess → xtremax → evaluate |
| **diffrax** | ODE/SDE solvers | Energy balance model in simulations |
| **gaussx** | Structured covariance for GP layers | Optional for large-scale spatial models |

---

## Scope

### In Scope

- Extreme value distributions (GEV family, GPD, point processes, max-stable)
- Bayesian extreme value models (stationary, nonstationary, spatial, POT, PP)
- xarray utilities for extreme extraction (block maxima, thresholds, declustering)
- EVT diagnostic and spatial plotting
- Synthetic extreme data simulation
- Climate observation dataset loaders
- Climate covariate loaders (GMST, ENSO, NAO, etc.)
- GP-based spatial pooling (via GPJax)
- Copula utilities for multivariate extremes

### Out of Scope

- Climate models / physics-based simulation — somax, ESMs
- General-purpose statistics — scipy.stats, statsmodels
- Full R-EVT feature parity — Bayesian-first, not frequentist-complete
- Geoprocessing / regridding — geo_toolz
- Dashboard / interactive visualization — user code
- GP library — pyrox_gp / GPJax (xtremax wraps GPJax)

---

## Testing Strategy

| Category | What it tests | Example |
|----------|---------------|---------|
| **Distribution contract** | NumPyro Distribution interface | `GEVD.log_prob`, `sample`, `mean`, `variance`, `cdf` match analytical |
| **Numerical accuracy** | Distribution math vs scipy reference | `GEVD.log_prob(x)` matches `scipy.stats.genextreme.logpdf(x)` |
| **Return levels** | Quantile inversion | `return_level(T)` matches `cdf^{-1}(1 - 1/T)` |
| **xarray preservation** | Coordinates, dims, attrs survive | `block_maxima(ds).coords` contains expected dims |
| **Model inference** | Models converge on synthetic data | `stationary_gev` recovers known parameters via MCMC |
| **JAX transforms** | Compatible with jit, vmap | `jax.vmap(GEVD.log_prob)(batch_x)` works |
| **Plotting** | Diagnostic plots don't crash | QQ, PP, return level plots produce figures |

### Test Priorities

1. **Distribution correctness** — log_prob, cdf, return_level match analytical/scipy
2. **NumPyro compatibility** — all distributions work in MCMC and SVI
3. **xarray round-trip** — extraction utilities preserve metadata
4. **Model convergence** — model zoo recovers known parameters on synthetic data

---

## Migration Plan from `jej_vc_snippets/extremes`

The existing code in `jej_vc_snippets/extremes/` is the primary source material. The migration is a restructuring + cleanup, not a rewrite.

| Source (`jej_vc_snippets/extremes/`) | Destination (`xtremax/`) | Notes |
|--------------------------------------|--------------------------|-------|
| `gevd.py` | `distributions/gevd.py` | Direct move, clean imports |
| `gpd.py` | `distributions/gpd.py` | Direct move |
| `gumbel.py` | `distributions/gumbel.py` | Direct move |
| `frechet.py` | `distributions/frechet.py` | Direct move |
| `weibull.py` | `distributions/weibull.py` | Direct move |
| `point_process/*.py` | `distributions/point_process/*.py` | Consolidate Poisson variants |
| `block_maxima.py` | `xarray/block_maxima.py` | Direct move |
| `threshold.py` | `xarray/threshold.py` | Direct move |
| `decluster.py` | `xarray/decluster.py` | Direct move |
| `models/temp_gevd_*.py` | `models/spatial_gev.py` | Generalize from temperature-specific |
| `simulations/*.py` | `simulations/*.py` | Split and reorganize |

---

## Roadmap

| Phase | Focus | Depends on |
|-------|-------|------------|
| v0.1 | Core distributions (GEV, GPD) + block maxima + stationary/nonstationary models | numpyro stable |
| v0.2 | Point processes + POT + spatial GEV (GPJax) + max-stable | v0.1 + GPJax |
| v0.3 | Dataset loaders + simulation tools + plotting | v0.1 |
| v0.4+ | r-Pareto processes, copulas, additional point process models | v0.2 |

---

## Open Questions

1. **Composite likelihood adjustment** — Pairwise composite likelihood underestimates posterior uncertainty. Should we implement the Chandler-Bate sandwich adjustment? Proposal: provide a `sandwich_adjustment()` utility.

2. **Max-stable simulation** — Exact simulation of Brown-Resnick is expensive. Approximate via truncated spectral representation? How many terms?

3. **Spatial plotting backend** — Plain matplotlib by default, cartopy optional for georeferenced maps?

4. **r-Pareto processes** — Conditional spatial analog of max-stable processes. Defer to v0.2?
