---
status: draft
version: 0.1.0
---

# Architecture

## Three-Layer Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 2 — Models                                                       │
│  NumPyro model functions. Ready-to-use Bayesian workflows.              │
│  stationary_gev, nonstationary_gev, spatial_gev, pot_gpd,               │
│  point_process_extreme, max_stable_composite                            │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 1 — Components                                                   │
│  NumPyro Distribution classes: GEVD, GPD, Gumbel, Frechet, Weibull,    │
│    HawkesProcess, BrownResnickProcess, SmithProcess, ...                │
│  xarray utilities: block_maxima, threshold, decluster, masks            │
│  plotting: diagnostic, spatial, temporal, model                         │
│  simulations: temporal, spatial, climate_signal, extremes, ODE          │
│  datasets: ghcnd, gsod, ndbc, uhslc, gesla, coops, covariates          │
│  GP wrappers: VariationalGP, SparseVariationalGP (via GPJax)           │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 0 — Primitives                                                   │
│  Pure JAX functions. Stateless, differentiable, no NumPyro.             │
│  gev_log_prob, gev_cdf, gev_icdf, gev_return_level, gev_mean           │
│  gpd_log_prob, gpd_cdf, gpd_survival, gpd_return_level                 │
│  poisson_log_likelihood, hawkes_log_likelihood, extremal_index          │
│  brown_resnick_extremal_coeff, pairwise_log_likelihood, madogram        │
│  power_variogram, matern_variogram                                      │
└─────────────────────────────────────────────────────────────────────────┘

Foundation (not owned by xtremax):
┌───────────────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐
│  jax              │  │  numpyro  │  │  xarray  │  │  gpjax   │
│  (L0 math)        │  │  (L1 dist │  │  (L1     │  │  (L1 GP  │
│                   │  │   + L2    │  │   xarray │  │   engine)│
│                   │  │   models) │  │   utils) │  │          │
└───────────────────┘  └───────────┘  └──────────┘  └──────────┘
```

**Layer 0** is pure JAX math. Every function takes arrays and returns arrays. `gev_log_prob(x, loc, scale, shape)` implements the GEV log-density; `gev_return_level(T, loc, scale, shape)` computes the T-year return level. These are the equations from Coles (2001) translated to JAX. A researcher can use L0 without NumPyro, without xarray — just JAX arrays.

**Layer 1** wraps L0 functions into richer interfaces. NumPyro `Distribution` subclasses wire `log_prob`, `sample`, `cdf`, `return_level` together with parameter storage. xarray utilities operate on labeled spatiotemporal data (block maxima, thresholds, declustering). Plotting, simulations, datasets, and GP wrappers are also L1.

**Layer 2** composes L1 distributions and utilities into complete NumPyro model functions. `stationary_gev(obs)` declares priors, creates a GEVD distribution, and defines the likelihood — ready for `MCMC(NUTS(stationary_gev))`.


## Package Layout

```
src/xtremax/
├── __init__.py
├── distributions/
│   ├── __init__.py
│   ├── gevd.py              # Generalized Extreme Value Distribution
│   ├── gpd.py               # Generalized Pareto Distribution
│   ├── gumbel.py            # Gumbel (Type I, ξ=0)
│   ├── frechet.py           # Frechet (Type II, ξ>0)
│   ├── weibull.py           # Weibull (Type III, ξ<0)
│   ├── max_stable/
│   │   ├── __init__.py
│   │   ├── smith.py         # Smith (Gaussian storm) model
│   │   ├── schlather.py     # Schlather model
│   │   ├── brown_resnick.py # Brown-Resnick model
│   │   └── extremal_t.py    # Extremal-t model
│   └── point_process/
│       ├── __init__.py
│       ├── poisson.py       # Homogeneous & inhomogeneous Poisson
│       ├── hawkes.py        # Self-exciting Hawkes process
│       ├── renewal.py       # Renewal processes
│       ├── spatial.py       # Spatial point processes
│       └── spatiotemporal.py # Spatio-temporal point processes
├── models/
│   ├── __init__.py
│   ├── stationary_gev.py    # iid GEV block maxima
│   ├── nonstationary_gev.py # GEV with covariate-dependent parameters
│   ├── spatial_gev.py       # GEV + GPJax spatial pooling
│   ├── pot_gpd.py           # Peaks-over-threshold with GPD
│   ├── point_process.py     # Point process extreme value models
│   └── max_stable.py        # Max-stable process fitting
├── gp/
│   ├── __init__.py
│   ├── variational.py       # Variational GP wrapper around GPJax
│   ├── sparse.py            # Sparse variational GP with inducing points
│   └── utils.py             # Kernel helpers, parameter storage
├── plotting/
│   ├── __init__.py
│   ├── temporal.py          # Time series extreme plots
│   ├── spatial.py           # Map-based extreme plots
│   ├── diagnostic.py        # QQ, PP, return level, probability plots
│   └── model.py             # Posterior & convergence plots
├── xarray/
│   ├── __init__.py
│   ├── masks.py             # Spatial, temporal, and quality masks
│   ├── block_maxima.py      # Block maxima extraction
│   ├── threshold.py         # Threshold selection (constant, trend, parametric)
│   └── decluster.py         # Declustering & extremal index
├── datasets/
│   ├── __init__.py
│   ├── _cache.py             # Download caching & local storage
│   ├── _units.py             # Unit normalization (°C, m/s, mm, m)
│   ├── ghcnd.py              # GHCN-Daily (temperature, precipitation)
│   ├── gsod.py               # Global Summary of Day (wind, temperature)
│   ├── isd.py                # Integrated Surface Database (sub-daily)
│   ├── ecad.py               # ECA&D European stations
│   ├── ndbc.py               # NDBC buoys (SST, waves, wind)
│   ├── uhslc.py              # UHSLC tide gauges (sea level)
│   ├── gesla.py              # GESLA storm surge / extreme sea level
│   ├── coops.py              # NOAA CO-OPS tide gauges (U.S.)
│   └── covariates/
│       ├── __init__.py
│       ├── global_warming.py # GMST, CO₂, radiative forcing, TSI
│       ├── enso.py           # ONI, Nino3.4, SOI, MEI
│       ├── oscillations.py   # NAO, PDO, AMO, AO, SAM, PNA, IOD
│       └── tropical.py       # MJO, QBO, ACE, monsoon indices
└── simulations/
    ├── __init__.py
    ├── temporal.py           # GMST, energy balance ODE, TemporalFeatureExtractor
    ├── spatial.py            # Domain masks, fractal terrain, SpatialFeatureExtractor
    ├── climate_signal.py     # Physics-informed spatiotemporal mean fields
    ├── extremes.py           # Variable-specific extreme generators (temp, precip, wind)
    └── ode.py                # Diffrax ODE/SDE utilities for temporal dynamics
```


## Dependencies

| Package | Version | Role |
|---------|---------|------|
| `jax` | ≥ 0.4 | Array computation, autodiff, JIT |
| `jaxlib` | ≥ 0.4 | JAX backend |
| `numpyro` | ≥ 0.15 | Probabilistic programming, MCMC, SVI |
| `xarray` | ≥ 2024.0 | Spatiotemporal data structures |
| `numpy` | ≥ 1.26 | Array utilities |

**Optional:**

| Package | Role |
|---------|------|
| `gpjax` | Gaussian process kernels/inference for spatial models |
| `matplotlib` | Plotting (diagnostic, spatial, temporal plots) |
| `statsmodels` | Quantile regression in threshold selection |
| `diffrax` | ODE/SDE solvers for temporal dynamics (energy balance, etc.) |
| `equinox` | Module-style JAX for advanced model composition |
| `cartopy` | Map projections for spatial plots |
| `erddapy` | ERDDAP client for NDBC buoy data |
| `requests` | HTTP access for GHCN-D, GSOD, ECA&D, UHSLC, GESLA |

---

## CI / Quality Gates

| Check | Command | Scope |
|-------|---------|-------|
| Tests | `uv run pytest tests -x` | Full suite |
| Lint | `uv run ruff check .` | Entire repo |
| Format | `uv run ruff format --check .` | Entire repo |
| Typecheck | `uv run ty check src/xtremax` | Package only |

All four must pass before merge. GitHub Actions on push/PR.
Conventional commits required (`feat:`, `fix:`, `docs:`, `test:`, etc.).

**Build system:** hatchling (PEP 621), `src/` layout
**Python:** >= 3.12, < 3.14
**License:** MIT
