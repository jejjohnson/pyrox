---
status: draft
version: 0.1.0
---

# Vision

## One-Liner

> **xtremax** is a JAX/NumPyro-native library for extreme value modeling — custom distributions, Bayesian models, and spatiotemporal data extraction in one place.

---

## Motivation

Extreme value analysis is central to climate risk, engineering reliability, finance, and insurance — yet the Python ecosystem lacks a cohesive library that combines custom extreme value distributions, flexible Bayesian models, and spatiotemporal data extraction in one place. Practitioners cobble together scipy stats, hand-rolled log-likelihoods, and ad hoc xarray scripts. The result is fragile, hard to extend, and disconnected from modern probabilistic programming.

xtremax provides three things:

1. **Custom NumPyro distributions** — GEV, GPD, Gumbel, Frechet, Weibull, and point process likelihoods, all fully differentiable and compatible with NumPyro's MCMC/SVI inference.
2. **A model zoo** — ready-to-use Bayesian models for common extreme value workflows: stationary and nonstationary GEV, peaks-over-threshold GPD, spatial pooling with Gaussian processes, and point process characterizations.
3. **xarray utilities** — functions to extract extremes from spatiotemporal data: block maxima, threshold exceedances, declustering, and threshold selection methods.

---

## User Stories

**Climate risk researcher** — "I have 50 years of daily temperature data at 200 stations. I want to fit a spatial GEV that borrows strength across locations via a GP, with GMST as a nonstationary covariate. I need return levels with uncertainty."

**Coastal engineer** — "I have tide gauge records and need to estimate the 100-year storm surge. I want peaks-over-threshold with a trend-following threshold that accounts for sea level rise."

**Insurance actuary** — "I need to model the frequency and severity of extreme wind events using a point process + GPD. The model should be Bayesian so I get full posterior uncertainty on the return period."

**Student / newcomer** — "I want `GEVD(loc, scale, concentration)` as a NumPyro distribution that I can plug into MCMC. I shouldn't need to hand-code the GEV log-density."

---

## Design Principles

1. **NumPyro-native** — Distributions subclass `numpyro.distributions.Distribution`. Models are NumPyro model functions. No custom inference — NUTS, SVI, Predictive all from NumPyro.

2. **JAX all the way down** — All distribution math is JAX. Automatic differentiation, JIT, vmap, GPU/TPU for free.

3. **xarray as data interface** — Spatiotemporal data flows as `xr.DataArray` / `xr.Dataset`. Extraction utilities preserve coordinates, dimensions, and metadata.

4. **Composable, not monolithic** — Each layer (distributions, models, xarray utilities) is independently useful. Import only what you need.

5. **Theory-aware API** — Method names follow EVT conventions: `loc`, `scale`, `concentration`/`shape` for distributions; `return_level()`, `extremal_index()`, `tail_index()` as first-class methods. Docstrings connect to the mathematics.

---

## Identity

### What xtremax IS

- Custom NumPyro distributions for extreme values (GEV, GPD, point processes, max-stable)
- A model zoo of Bayesian extreme value models (stationary, nonstationary, spatial, POT, PP)
- xarray utilities for extracting extremes (block maxima, threshold exceedances, declustering)
- Simulation tools for generating synthetic extreme datasets
- Dataset loaders for common climate/ocean observation networks
- Diagnostic and spatial plotting for EVT

### What xtremax is NOT

| Not this | Use instead |
|----------|-------------|
| Climate model (physics-based simulation) | somax, diffrax, ESMs |
| General-purpose statistics library | scipy.stats, statsmodels |
| Feature-complete R-EVT port (extRemes, ismev) | xtremax is Bayesian-first, not frequentist-complete |
| Visualization/dashboard framework | matplotlib / user code (xtremax ships diagnostic plots only) |
| Geoprocessing / regridding | geo_toolz |
| Gaussian process library | pyrox_gp / GPJax (xtremax wraps GPJax for spatial models) |

---

## Migration Context

### Internal

The `jej_vc_snippets/extremes/` directory contains the existing codebase: GEV/GPD distributions, point process implementations, block maxima extraction, threshold selection, and spatial models. xtremax restructures and generalizes this into a proper library. See `boundaries.md` for the full migration map.

### External

| Tool | Language | Limitation xtremax addresses |
|------|----------|------------------------------|
| `extRemes` (R) | R | Not Python; frequentist-only; no GPU |
| `ismev` (R) | R | Not Python; limited spatial support |
| `scipy.stats.genextreme` | Python | No Bayesian inference; no spatial; no point processes |
| `pyextremes` | Python | Limited to univariate; no JAX; no spatial GP |
| `climada` | Python | Risk framework, not EVT library; heavy dependencies |

### Key References

- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.
- Davison, A. C. & Huser, R. (2015). Statistics of extremes. *Annual Review of Statistics*.
- Cooley, D., Nychka, D. & Naveau, P. (2007). Bayesian spatial modeling of extreme precipitation return levels. *JASA*.
- Padoan, S. A., Ribatet, M. & Sisson, S. A. (2010). Likelihood-based inference for max-stable processes. *JASA*.

---

## Connection to Ecosystem

```
                    ┌──────────────┐
                    │  raw climate  │  Station obs, reanalysis, tide gauges
                    │  data         │
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │      geo_toolz          │  Preprocess (regrid, detrend, subset)
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │      xtremax            │  Extract extremes → fit models → return levels
              │  distributions + models │
              └────────┬────┬───────────┘
                       │    │
            ┌──────────┘    └──────────┐
     ┌──────▼──────┐           ┌──────▼──────┐
     │   numpyro   │           │   GPJax     │
     │  (inference)│           │  (spatial   │
     │             │           │   GP engine)│
     └─────────────┘           └─────────────┘

Downstream consumers:
    User risk analysis, impact modeling, insurance pricing

Optional ecosystem integration:
    gaussx (structured covariance for GP layers)
    geo_toolz (pre/post-processing of climate data)
```
