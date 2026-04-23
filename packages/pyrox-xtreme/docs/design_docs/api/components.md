---
status: draft
version: 0.1.0
---

# Components

## xarray Utilities

Functions for extracting extremes from spatiotemporal `xr.DataArray` inputs. All preserve coordinates and metadata.

### Masking (`xtremax.xarray.masks`)

Real-world spatiotemporal data is irregular. Satellite orbits have gaps. Stations go offline. Ocean data needs land masked out. Quality flags mark suspect observations. If you compute a 95th-percentile threshold on data that includes fill values, land pixels, or sensor artifacts, you get garbage. Masking is therefore a prerequisite to every downstream operation — block maxima, thresholds, and declustering all need to operate on clean, regular data.

`xtremax` provides mask construction utilities and a convention: **every xarray function in `xtremax.xarray` accepts an optional `mask` parameter.** When provided, masked-out values are treated as NaN before any computation. The mask is a boolean `xr.DataArray` (True = valid, False = masked) that can be spatial, temporal, or spatiotemporal.

#### Mask construction

```python
from xtremax.xarray.masks import (
    land_sea_mask,
    coverage_mask,
    quality_mask,
    seasonal_availability_mask,
    combine_masks,
)

# Spatial mask: ocean only (True where ocean)
ocean_mask = land_sea_mask(ds, variable="sftlf", threshold=50)  # land fraction < 50%

# Temporal coverage mask: require ≥ 80% non-NaN per year per gridcell
cov_mask = coverage_mask(da, time_dim="time", freq="YS", min_coverage=0.8)

# Quality flag mask: keep only QC flag == 0 or 1
qc_mask = quality_mask(ds["quality_flag"], valid_values=[0, 1])

# Seasonal availability: only keep DJF months
djf_mask = seasonal_availability_mask(da, seasons=["DJF"], time_dim="time")

# Combine: intersection of all masks (all must be True)
mask = combine_masks([ocean_mask, cov_mask, qc_mask, djf_mask])
```

#### Mask types

| Function | Dims | Use case |
|----------|------|----------|
| `land_sea_mask` | spatial (lat, lon) | Remove land/ocean gridcells |
| `coverage_mask` | spatiotemporal | Drop gridcell-years with too many NaNs |
| `quality_mask` | same as input | Filter by quality/status flag |
| `seasonal_availability_mask` | temporal | Restrict to specific seasons/months |
| `elevation_mask` | spatial | Altitude band selection |
| `bounding_box_mask` | spatial | Rectangular lat/lon region |
| `distance_mask` | spatial | Within radius of a point (e.g., coastal) |
| `combine_masks` | any | Logical AND of multiple masks |

#### How masks flow through the pipeline

All downstream functions accept `mask` as an optional keyword. Internally, masked values become NaN before computation, and outputs preserve the mask in their coordinates for provenance.

```python
from xtremax.xarray import (
    temporal_block_maxima,
    quantile_threshold,
    decluster_runs,
)

# Build mask once
mask = combine_masks([ocean_mask, cov_mask])

# Every step respects the same mask
annual_max = temporal_block_maxima(da, freq="YS", min_periods=300, mask=mask)
u = quantile_threshold(da, q=0.95, mask=mask)
peaks = decluster_runs(da, threshold=u, reduction="max", mask=mask)
```

Alternatively, apply the mask upfront and work with the masked array:

```python
from xtremax.xarray.masks import apply_mask

da_clean = apply_mask(da, mask)  # sets masked values to NaN

# Now all functions operate on clean data without needing mask= everywhere
annual_max = temporal_block_maxima(da_clean, freq="YS", min_periods=300)
```

#### Coverage diagnostics

After masking, you often need to know what's left. How many valid gridcells? How many years per station?

```python
from xtremax.xarray.masks import coverage_summary

summary = coverage_summary(da, mask=mask, time_dim="time", freq="YS")
# Returns Dataset with:
#   valid_fraction  — (lat, lon) fraction of years with sufficient data
#   n_valid_years   — (lat, lon) count of valid years
#   total_valid_obs — scalar, total non-NaN observations after masking
```

### Block Maxima

Block maxima functions respect masks — blocks with insufficient valid observations (below `min_periods`) return NaN rather than a misleading maximum from sparse data.

```python
from xtremax.xarray import (
    temporal_block_maxima,
    spatial_block_maxima,
    sliding_block_maxima,
    declustered_block_maxima,
    r_largest_block_maxima,
)

# Annual maxima from daily data
annual_max = temporal_block_maxima(da, freq="YS", min_periods=300)

# r-largest order statistics per year
r_largest = r_largest_block_maxima(da, r=3, freq="YS")

# Sliding window with overlap
sliding = sliding_block_maxima(da, window=365, stride=30)
```

### Threshold Selection

Threshold selection is a first-class concept in `xtremax`. Three tiers, all quantile-based, with increasing flexibility:

**Tier 1 — Constant quantile.** A single quantile computed over the full dataset. The simplest and most common choice.

```python
from xtremax.xarray import quantile_threshold

# 95th percentile — one number for the whole dataset
u = quantile_threshold(da, q=0.95)

# Per-location constant (quantile over time only)
u_spatial = quantile_threshold(da, q=0.95, dim="time")
```

**Tier 2 — Trend quantile.** A linearly time-varying threshold via quantile regression. Captures nonstationarity (e.g., warming trends) so that "extreme" is defined relative to the evolving climate, not a fixed baseline.

```python
from xtremax.xarray import trend_quantile_threshold

# Linear trend in the 95th percentile
u_trend = trend_quantile_threshold(da, q=0.95, time_dim="time")

# With external covariates (e.g., GMST drives the trend)
u_trend = trend_quantile_threshold(da, q=0.95, time_dim="time", covariates=gmst)
```

Fits τ-quantile regression: `Q_τ(Y|t) = β₀ + β₁·t` (or `β₀ + β₁·x(t)` with covariates). Uses `statsmodels.QuantReg` under the hood.

**Tier 3 — Parametric function quantile.** The threshold is a user-supplied function of covariates, fit via quantile regression. Handles seasonal cycles, polynomial trends, splines, or any design matrix the user provides.

```python
from xtremax.xarray import parametric_quantile_threshold

# Seasonal cycle: threshold = β₀ + β₁·sin(2πt/365) + β₂·cos(2πt/365)
u_seasonal = parametric_quantile_threshold(
    da,
    q=0.95,
    design_matrix=seasonal_design_matrix,  # (n_time, n_basis)
    time_dim="time",
)

# Polynomial trend + seasonal harmonics
import numpy as np
t = np.arange(len(da.time))
X = np.column_stack([
    t, t**2,                                    # quadratic trend
    np.sin(2 * np.pi * t / 365.25),            # annual cycle
    np.cos(2 * np.pi * t / 365.25),
    np.sin(4 * np.pi * t / 365.25),            # semi-annual
    np.cos(4 * np.pi * t / 365.25),
])
u_full = parametric_quantile_threshold(da, q=0.95, design_matrix=X)

# Spline basis (via patsy or manual B-spline construction)
u_spline = parametric_quantile_threshold(da, q=0.95, design_matrix=spline_basis)
```

Fits τ-quantile regression: `Q_τ(Y|X) = Xβ` where X is the user-provided design matrix. This subsumes Tier 2 (linear trend is just a single-column design matrix) but is kept separate for ergonomics.

**Convenience wrappers** for common temporal groupings (these are lighter-weight alternatives that don't use quantile regression — they compute empirical quantiles within groups):

```python
from xtremax.xarray import (
    temporal_threshold,
    rolling_threshold,
    seasonal_threshold,
)

# Seasonal empirical quantile (groupby, not regression)
u_season = temporal_threshold(da, q=0.95, groupby="season")

# Smooth rolling window
u_rolling = rolling_threshold(da, q=0.95, window_size=30)
```

### Declustering

```python
from xtremax.xarray import (
    decluster_runs,
    decluster_separation,
    estimate_extremal_index,
)

# Runs declustering — one peak per cluster of consecutive exceedances
peaks = decluster_runs(da, threshold=u, reduction="max")

# Enforce minimum 3-day separation between peaks
peaks = decluster_separation(da, threshold=u, min_separation=3)

# Estimate clustering strength
theta = estimate_extremal_index(da, threshold=u, method="runs")
# θ = 1: independent; θ < 1: clustered
```


## Plotting

`xtremax` ships standard diagnostic and visualization functions for extreme value analysis. These are **not** a visualization framework — they are opinionated, publication-ready matplotlib functions that cover the plots practitioners actually need. Every function returns `(fig, ax)` tuples for further customization.

### Temporal Plots (`xtremax.plotting.temporal`)

Plots for time series of extremes.

```python
from xtremax.plotting.temporal import (
    plot_block_maxima,
    plot_threshold_exceedances,
    plot_extremal_index,
    plot_cluster_identification,
    plot_annual_maxima_trend,
    plot_return_level_time_series,
)

# Annotated time series with block maxima highlighted
fig, ax = plot_block_maxima(da, annual_max, freq="YS")

# Threshold exceedances with declustered peaks marked
fig, ax = plot_threshold_exceedances(da, threshold=u, peaks=peaks)

# Extremal index over a range of thresholds
fig, ax = plot_extremal_index(da, quantiles=np.linspace(0.90, 0.99, 20))

# Trend in annual maxima with CI band
fig, ax = plot_annual_maxima_trend(annual_max, covariates=gmst)
```

### Spatial / Geographic Plots (`xtremax.plotting.spatial`)

Map-based plots for spatial extreme value fields. Works with xarray objects that have `lat`/`lon` coordinates.

```python
from xtremax.plotting.spatial import (
    plot_return_level_map,
    plot_gev_parameter_maps,
    plot_exceedance_probability_map,
    plot_station_maxima,
    plot_coverage_map,
    plot_spatial_trend,
    plot_extremal_coefficient_map,
)

# Return level surface over a spatial domain
fig, ax = plot_return_level_map(return_levels, period=100, cmap="YlOrRd")

# GEV parameter fields (μ, σ, ξ) as a triptych
fig, axes = plot_gev_parameter_maps(posterior_params, coordinates)

# Station locations sized/colored by observed maximum
fig, ax = plot_station_maxima(ds, variable="tasmax")

# Spatial coverage after masking — which gridcells have enough data?
fig, ax = plot_coverage_map(coverage_summary, variable="valid_fraction")

# Spatial field of extremal dependence strength
fig, ax = plot_extremal_coefficient_map(theta_field, coordinates)
```

### Diagnostic Plots (`xtremax.plotting.diagnostic`)

Standard EVT diagnostic plots for model checking.

```python
from xtremax.plotting.diagnostic import (
    plot_qq,
    plot_pp,
    plot_return_level,
    plot_probability,
    plot_density_histogram,
    plot_mean_residual_life,
    plot_threshold_stability,
    plot_hill,
    plot_gev_diagnostic_4panel,
    plot_gpd_diagnostic_4panel,
)

# Classic 4-panel GEV diagnostic: QQ, PP, return level, density
fig, axes = plot_gev_diagnostic_4panel(data, loc=mu, scale=sigma, shape=xi)

# Return level plot with confidence intervals
fig, ax = plot_return_level(data, params, ci=0.95)

# Mean residual life plot for threshold selection
fig, ax = plot_mean_residual_life(data, thresholds=np.linspace(u_lo, u_hi, 50))

# Threshold stability plot (GPD scale & shape vs threshold)
fig, axes = plot_threshold_stability(data, thresholds)

# Hill plot for tail index estimation
fig, ax = plot_hill(data, k_range=(10, 500))

# Compare threshold tiers: constant vs trend vs parametric
from xtremax.plotting.diagnostic import plot_threshold_comparison
fig, ax = plot_threshold_comparison(
    da, thresholds={"constant": u, "trend": u_trend, "seasonal": u_param}
)
```

### Model Evaluation Plots (`xtremax.plotting.model`)

Posterior diagnostics and model comparison for Bayesian fits.

```python
from xtremax.plotting.model import (
    plot_posterior,
    plot_trace,
    plot_posterior_return_level,
    plot_posterior_predictive,
    plot_gp_spatial_field,
    plot_nonstationary_parameters,
)

# Posterior density + trace for GEV parameters
fig, axes = plot_trace(mcmc_samples, var_names=["loc", "scale", "concentration"])

# Return level plot with full posterior uncertainty
fig, ax = plot_posterior_return_level(mcmc_samples, periods=[2, 5, 10, 25, 50, 100])

# Posterior predictive check: observed vs simulated
fig, ax = plot_posterior_predictive(mcmc_samples, observed=annual_max)

# GP spatial field: posterior mean + uncertainty
fig, axes = plot_gp_spatial_field(gp_posterior, coordinates, parameter="loc")

# Nonstationary parameters over covariate range
fig, axes = plot_nonstationary_parameters(mcmc_samples, covariate=gmst)
```


## Simulations (`xtremax.simulations`)

Synthetic data generators for testing, demonstration, and benchmarking. The simulations module has four layers: **temporal** (GMST trajectories via ODEs), **spatial** (procedural domains and terrain features), **climate signal** (physics-informed spatiotemporal mean fields), and **variable-specific extreme generators** (temperature, precipitation, wind).

### Package Layout

```
src/xtremax/simulations/
├── __init__.py
├── temporal.py          # GMST, energy balance ODE, TemporalFeatureExtractor
├── spatial.py           # Domain masks, fractal terrain, feature extraction
├── climate_signal.py    # Physics-informed spatiotemporal mean fields
├── extremes.py          # Variable-specific extreme value generators
└── ode.py               # Diffrax ODE utilities for temporal dynamics
```

### Temporal — GMST & Energy Balance (`simulations.temporal`)

Two levels of complexity for generating GMST trajectories.

**Simple trajectories** — deterministic trend + AR(1) red noise:

```python
from xtremax.simulations.temporal import generate_gmst_trajectory

# Linear warming with internal variability
gmst = generate_gmst_trajectory(
    n_years=50, start_year=1981,
    trend_type="linear",     # or "exponential", "logistic"
    noise_std=0.05,
    seed=42,
)
# xr.DataArray(year), units °C anomaly
```

**Physics-based trajectories** — 0-D Energy Balance Model solved as an ODE via **diffrax**:

```
C · dT/dt = F(t) − λ · T(t)
```

Where F(t) = F_ghg(t) + F_solar(t) + F_volc(t) + ε(t) is a superposition of:
- **GHG forcing**: F_ghg = 5.35 · ln(CO₂(t)/CO₂₀), logistic CO₂ growth
- **Solar cycles**: F_solar = A · sin(2πt/11), 11-year Schwabe cycle
- **Volcanic eruptions**: Stochastic negative impulses with exponential decay
- **Internal variability**: Ornstein-Uhlenbeck red noise

```python
from xtremax.simulations.temporal import generate_physical_gmst

ds = generate_physical_gmst(
    n_years=120,
    start_year=1900,
    climate_sensitivity=3.0,      # ECS: °C per 2×CO₂
    ocean_heat_capacity=10.0,     # W·yr·m⁻²·K⁻¹
    seed=42,
)
# xr.Dataset with:
#   gmst, forcing_total, forcing_ghg, forcing_volcanic,
#   forcing_solar, forcing_stochastic
```

### ODE Utilities — Diffrax Integration (`simulations.ode`)

General-purpose ODE solver utilities wrapping **diffrax** for temporal dynamics. Used by the energy balance model but available for any user-defined temporal ODE.

```python
from xtremax.simulations.ode import solve_temporal_ode

# Solve any ODE: dy/dt = f(t, y)
def energy_balance(t, y, args):
    """C · dT/dt = F(t) - λ · T"""
    T = y
    C, lam, forcing_fn = args
    return (forcing_fn(t) - lam * T) / C

solution = solve_temporal_ode(
    vector_field=energy_balance,
    y0=jnp.array([0.0]),           # initial condition
    t0=0.0,
    t1=120.0,                       # years
    dt0=0.1,                        # initial step size
    args=(C, lam, forcing_fn),
    solver="tsit5",                 # or "dopri5", "euler", "heun"
    saveat=jnp.linspace(0, 120, 1440),  # monthly output
)
# solution.ys: (n_steps, n_vars) JAX array
```

Supported solvers (via diffrax): `Tsit5` (default, RK45-like), `Dopri5`, `Euler`, `Heun`, `Midpoint`, `ImplicitEuler` (stiff systems).

**Stochastic differential equations** are also supported for noise-driven dynamics:

```python
from xtremax.simulations.ode import solve_temporal_sde

# Ornstein-Uhlenbeck process: dε = -θ·ε·dt + σ·dW
def ou_drift(t, y, args):
    theta = args
    return -theta * y

def ou_diffusion(t, y, args):
    return jnp.array([0.2])

solution = solve_temporal_sde(
    drift=ou_drift,
    diffusion=ou_diffusion,
    y0=jnp.array([0.0]),
    t0=0.0, t1=100.0, dt0=0.01,
    args=(0.5,),
    key=rng_key,
)
```

### Spatial — Domain & Terrain (`simulations.spatial`)

Procedural generation of realistic spatial domains with physical geography.

**Domain masks** — generate land/ocean boundaries from geometric primitives:

```python
from xtremax.simulations.spatial import create_domain

# Procedural Iberian Peninsula with Balearic Islands
ds = create_domain(
    region="iberia",
    resolution=0.1,               # degrees (~10 km)
)
# xr.Dataset with: mask (bool), elevation (m)
#   coords: lat, lon

# Custom bounding box with fractal terrain
ds = create_domain(
    bounds=(-10, 5, 36, 44),
    resolution=0.05,
    terrain_seed=42,
)
```

**Fractal terrain** — coherent elevation via Fractal Brownian Motion (fBm):

```python
from xtremax.simulations.spatial import generate_fractal_terrain

terrain = generate_fractal_terrain(
    shape=(80, 150),
    octaves=4,                     # layers of noise
    persistence=0.5,               # amplitude decay per octave
    lacunarity=2.0,                # frequency growth per octave
    seed=42,
)
# np.ndarray (n_lat, n_lon), normalized 0–1
```

Terrain can be biased toward known geography (e.g., Pyrenees in the north, Sierra Nevada in the south) via additive Gaussian bumps.

**Spatial feature extraction** — derived physical covariates from a DEM:

```python
from xtremax.simulations.spatial import SpatialFeatureExtractor

extractor = SpatialFeatureExtractor()

# Distance to coast (Euclidean distance transform, km)
dist = extractor.get_distance_to_coast(ds)

# Slope (degrees) and aspect (compass bearing, 0=N, 180=S)
slope, aspect = extractor.get_slope_and_aspect(ds)

# Terrain roughness index (local elevation std dev)
roughness = extractor.get_roughness(ds, window_size=3)

# Pipeline: augment dataset with all features at once
ds = extractor.augment(ds)
# Adds: dist_to_coast, slope, aspect, roughness (masked to land)
```

These features are physically meaningful covariates for extreme value models:
- **Distance to coast** → continentality effect on temperature extremes
- **Slope** → orographic precipitation enhancement
- **Aspect** → solar insolation (south-facing = warmer)
- **Roughness** → local wind acceleration, turbulence

### Temporal Feature Extraction (`simulations.temporal`)

The temporal analog of `SpatialFeatureExtractor`. Extracts time-domain covariates from climate time series — seasonal cycles, trends, anomalies, harmonic components, and variability measures. These become covariates for nonstationary EVA models.

```python
from xtremax.simulations.temporal import TemporalFeatureExtractor

extractor = TemporalFeatureExtractor()
```

**Trend extraction** — isolate the long-term signal:

```python
# Linear trend (slope + intercept)
trend = extractor.get_linear_trend(da, time_dim="time")
# xr.DataArray: fitted trend line, same coords as input

# Lowess / locally-weighted trend (nonparametric)
trend = extractor.get_lowess_trend(da, time_dim="time", frac=0.3)

# Detrended residuals
residuals = extractor.detrend(da, method="linear")  # or "lowess"
```

**Seasonal cycle decomposition:**

```python
# Climatological mean seasonal cycle
clim = extractor.get_climatology(da, freq="dayofyear", smoothing=15)
# xr.DataArray(dayofyear) — 365-day smooth climatology

# Anomalies (observations minus climatology)
anom = extractor.get_anomalies(da, freq="dayofyear", smoothing=15)

# Seasonal amplitude (max - min of climatology per year)
amp = extractor.get_seasonal_amplitude(da, time_dim="time")
```

**Harmonic decomposition** — Fourier basis for periodic signals:

```python
# Annual + semi-annual harmonics
harmonics = extractor.get_harmonics(da, periods=[365.25, 182.625], time_dim="time")
# xr.Dataset: annual_sin, annual_cos, semiannual_sin, semiannual_cos, residual

# Design matrix for quantile regression (Tier 3 threshold selection)
X = extractor.harmonic_design_matrix(da, periods=[365.25, 182.625])
# np.ndarray (n_time, n_basis) — plug into parametric_quantile_threshold
```

**Variability measures** — characterize changing variance:

```python
# Rolling standard deviation (captures changing variability)
rolling_std = extractor.get_rolling_std(da, window=365, time_dim="time")

# Interannual variability (std of annual means)
iav = extractor.get_interannual_variability(da, time_dim="time")

# Coefficient of variation (rolling)
cv = extractor.get_rolling_cv(da, window=365, time_dim="time")
```

**Extreme-specific temporal features:**

```python
# Number of threshold exceedances per year (exceedance rate)
rate = extractor.get_exceedance_rate(da, threshold=u, freq="YS")
# xr.DataArray(year) — how often extremes occur per year

# Growing season length (consecutive days above threshold)
gsl = extractor.get_spell_duration(da, threshold=25.0, above=True, freq="YS")

# Frost-free season (consecutive days above 0°C)
ffs = extractor.get_spell_duration(da, threshold=0.0, above=True, freq="YS")

# Dry spell length (consecutive days below 1mm)
dry = extractor.get_spell_duration(da, threshold=1.0, above=False, freq="YS")
```

**Pipeline — augment a time series dataset with all features:**

```python
ds = extractor.augment(da, features=["trend", "climatology", "anomalies", "harmonics"])
# xr.Dataset with original data + all requested temporal features
```

These features connect directly to the rest of xtremax:
- **Trend** → covariate for `nonstationary_gev` (is the distribution shifting?)
- **Harmonics** → design matrix for `parametric_quantile_threshold` (Tier 3)
- **Anomalies** → deseasonalized input for POT/declustering
- **Exceedance rate** → diagnostic for threshold selection (is the rate stable?)
- **Spell durations** → compound event analysis (heat waves, droughts)

**Station-based domains** — random point locations for station network simulations:

```python
from xtremax.simulations.spatial import generate_station_network

ds = generate_station_network(
    n_sites=158,
    bounds=(-9, 3, 36, 44),       # Iberian box
    seed=42,
)
# xr.Dataset: coords(site), vars(lon, lat, elevation)
```

### Climate Signal — Physics-Informed Mean Fields (`simulations.climate_signal`)

Combines spatial features with temporal covariates (GMST) to produce realistic spatiotemporal mean fields. These become the location parameter μ(s, t) for distribution-based extreme simulation.

**Generic signal composition:**

```python
from xtremax.simulations.climate_signal import compute_climate_signal

# μ(s,t) = base + c₁·elev + c₂·lat + c₃·GMST + c₄·GMST×elev
mu_temp = compute_climate_signal(
    spatial_ds=ds,
    gmst_da=gmst,
    base_val=35.0,
    coeffs={
        "elevation": -6.5,         # lapse rate: -6.5 °C/km
        "lat": -0.5,               # -0.5 °C per degree north
        "gmst": 1.5,               # +1.5 °C per °C GMST
        "interaction": 0.1,        # elevation-dependent warming
    },
)
# xr.DataArray(year, site)
```

**Advanced physics composition** (gridded domains with terrain features):

```python
from xtremax.simulations.climate_signal import compute_advanced_climate_signal

ds_signal = compute_advanced_climate_signal(spatial_ds=ds, gmst_da=gmst)
# xr.Dataset with:
#   mu_tmax  — temperature mean field with:
#               lapse rate, continentality, latitude gradient,
#               aspect-driven insolation, land-ocean contrast amplification
#   mu_precip — precipitation intensity field with:
#               elevation boost, orographic slope effect,
#               Clausius-Clapeyron thermodynamic scaling (+6%/°C)
```

Physics captured:
- **Lapse rate**: −6.5 °C/km
- **Continentality**: +1.5 °C per 100 km from coast (land warms faster)
- **Insolation**: +1 °C on south-facing slopes, −1 °C on north-facing
- **Orographic precipitation**: steeper slopes → more rainfall
- **Clausius-Clapeyron**: +6–7% precipitation intensity per °C warming
- **Land-ocean contrast**: warming amplification increases with distance from coast

### Variable-Specific Extreme Generators (`simulations.extremes`)

Given a spatiotemporal mean field, generate synthetic block maxima from the appropriate extreme value distribution.

**Temperature extremes** — GEV-distributed annual maxima:

```python
from xtremax.simulations.extremes import simulate_temp_extremes

ds_temp = simulate_temp_extremes(
    mu=mu_temp,                    # location parameter (year, site)
    scale=2.0,                     # σ (constant or array)
    shape=-0.1,                    # ξ < 0: Weibull domain (bounded upper tail)
    seed=42,
)
# xr.Dataset: tmax (year, site), mu_tmax (year, site)
```

**Precipitation extremes** — Gamma intensity (Rx1day) + Poisson duration (CWD):

```python
from xtremax.simulations.extremes import simulate_precip_extremes

ds_precip = simulate_precip_extremes(
    spatial_ds=ds,
    gmst_da=gmst,
    seed=42,
)
# xr.Dataset:
#   rx1day (year, site) — annual max daily rainfall (mm), Gamma-distributed
#                          with Clausius-Clapeyron scaling (+7%/°C)
#   cwd (year, site)    — max consecutive wet days, Poisson-distributed
#                          with drying trend in Mediterranean
```

**Wind speed extremes** — Weibull-distributed annual max gusts:

```python
from xtremax.simulations.extremes import simulate_wind_extremes

ds_wind = simulate_wind_extremes(
    spatial_ds=ds,
    gmst_da=gmst,
    seed=42,
)
# xr.Dataset: wind_max (year, site), Weibull(k=2, λ=f(elevation, GMST))
```

### End-to-End Orchestration

```python
from xtremax.simulations import (
    temporal, spatial, climate_signal, extremes
)

# 1. Temporal driver
gmst = temporal.generate_gmst_trajectory(n_years=50, trend_type="linear")

# 2. Spatial domain (station network or gridded)
ds_space = spatial.generate_station_network(n_sites=100, bounds=(-9, 3, 36, 44))

# 3. Climate signal
mu_temp = climate_signal.compute_climate_signal(
    ds_space, gmst, base_val=35.0,
    coeffs={"elevation": -6.5, "lat": -0.5, "gmst": 1.5, "interaction": 0.1},
)

# 4. Generate extremes
ds_temp = extremes.simulate_temp_extremes(mu_temp, scale=2.0, shape=-0.1)
ds_precip = extremes.simulate_precip_extremes(ds_space, gmst)
ds_wind = extremes.simulate_wind_extremes(ds_space, gmst)

# 5. Ready for EVA — pipe directly into xtremax models
from xtremax.models import nonstationary_gev
mcmc.run(rng_key, obs=ds_temp["tmax"].values, covariates=gmst.values)
```


## Datasets (`xtremax.datasets`)

Real extreme value analysis needs real data. `xtremax.datasets` provides download-and-load utilities for open-source, observation-based datasets covering the four core application domains: **land surface temperature**, **wind speed**, **precipitation**, and **ocean extremes**. All loaders return xarray objects with standardized units, coordinates, and quality flags — ready to pipe into the masking and extraction utilities.

### Design Principles

**Observations only.** Every dataset is station/buoy/gauge-based. No reanalysis, no model output. Users get raw measurements with known provenance.

**Standardized output.** Regardless of source, every loader returns an `xr.Dataset` with:
- Consistent units (°C, m/s, mm, m) — raw units are converted automatically
- Standard coordinate names (`time`, `station_id`, `lat`, `lon`, `elevation`)
- Quality flags preserved as a coordinate variable (`qc_flag`)
- Source metadata in `attrs` (dataset name, version, citation, license)

**Lazy download + local cache.** Data is downloaded on first access and cached locally (`~/.cache/xtremax/` by default, configurable via `XTREMAX_DATA_DIR`). Subsequent calls read from cache. No data ships with the package.

**Thin wrappers.** Each loader is a thin wrapper around the provider's API or FTP. We don't reprocess or grid the data — we fetch, parse, normalize units/coordinates, and return. The user applies their own masks and extraction.

### Land Surface — Temperature Extremes

#### GHCN-Daily (`xtremax.datasets.ghcnd`)

The workhorse for global station-based temperature (and precipitation). ~100,000 stations, daily observations, some records back to the 1700s.

```python
from xtremax.datasets import ghcnd

# Load daily max temperature for a set of stations
ds = ghcnd.load_temperature(
    stations=["USW00094728", "USW00023174"],  # NYC Central Park, LA Airport
    variables=["TMAX", "TMIN"],
    start_date="1950-01-01",
    end_date="2023-12-31",
)
# ds has dims: (time, station_id)
# ds["TMAX"] in °C, ds["qc_flag"] per observation

# Search for stations by region
stations = ghcnd.find_stations(
    bbox=(-75, 40, -73, 42),           # lon_min, lat_min, lon_max, lat_max
    variables=["TMAX"],
    min_years=50,                       # at least 50 years of data
)
# Returns DataFrame: station_id, name, lat, lon, elevation, start_year, end_year

# Load and go straight to block maxima
ds = ghcnd.load_temperature(stations=stations.station_id.tolist(), variables=["TMAX"])
from xtremax.xarray import temporal_block_maxima
annual_tmax = temporal_block_maxima(ds["TMAX"], freq="YS", min_periods=300)
```

**Provider:** NOAA NCEI | **Access:** CDO Web API (token-based REST) + FTP fallback | **License:** Public Domain
**Quirks:** Temperature in tenths of °C (auto-converted). Missing = NaN. Station metadata merged automatically.

#### ECA&D (`xtremax.datasets.ecad`)

European stations with long records (some back to 1760s). ~2,000 stations across Europe and the Mediterranean.

```python
from xtremax.datasets import ecad

# Daily temperature for European stations
ds = ecad.load_temperature(
    stations=["DE_000001"],             # Berlin-Dahlem
    variables=["TX", "TN"],             # daily max, daily min
    start_date="1900-01-01",
)

# Search by country
stations = ecad.find_stations(country="NL", variables=["TX"], min_years=100)
```

**Provider:** KNMI | **Access:** HTTP direct download | **License:** CC-BY 4.0
**Quirks:** European-only. No official API — files downloaded and parsed.

### Land Surface — Wind Speed Extremes

#### GSOD (`xtremax.datasets.gsod`)

Global Summary of Day. ~9,000 stations with daily max wind speed and gust.

```python
from xtremax.datasets import gsod

# Daily wind data
ds = gsod.load_wind(
    stations=["720534-00164"],          # Station ID (USAF-WBAN)
    variables=["MXSPD", "GUST"],        # max sustained, gust
    start_date="1980-01-01",
)
# ds["MXSPD"] in m/s (auto-converted from knots)
# ds["GUST"] in m/s

# Search coastal stations for offshore wind studies
stations = gsod.find_stations(
    bbox=(-80, 25, -70, 45),            # U.S. East Coast
    variables=["GUST"],
    min_years=30,
)
```

**Provider:** NOAA NCEI | **Access:** FTP/HTTP | **License:** Public Domain
**Quirks:** Wind units vary by station (knots, m/s) — auto-normalized to m/s. Gust data not available at all stations.

#### ISD (`xtremax.datasets.isd`)

Integrated Surface Database. ~35,000 stations with **hourly** observations — the only dataset here with sub-daily resolution.

```python
from xtremax.datasets import isd

# Hourly wind for sub-daily extreme analysis
ds = isd.load_hourly(
    stations=["720534"],
    variables=["wind_speed", "wind_gust"],
    start_date="2000-01-01",
)
# Hourly resolution enables POT on sub-daily timescales

# Aggregate to daily max for compatibility with daily workflows
ds_daily = isd.load_daily_max(
    stations=["720534"],
    variables=["wind_speed", "wind_gust"],
    start_date="2000-01-01",
)
```

**Provider:** NOAA NCEI | **Access:** FTP (parsed from fixed-format files) | **License:** Public Domain
**Quirks:** Complex fixed-format parsing (handled internally). Per-element quality flags (0–9). Hourly data is large — downloads can be slow.

### Land Surface — Precipitation Extremes

#### GHCN-Daily Precipitation

Same dataset as temperature, different variables.

```python
from xtremax.datasets import ghcnd

# Daily precipitation
ds = ghcnd.load_precipitation(
    stations=["USW00094728"],
    variables=["PRCP"],                  # daily total precipitation
    start_date="1950-01-01",
)
# ds["PRCP"] in mm (auto-converted from tenths of mm)
# Trace amounts ("T") → 0.0 with trace_flag=True in coords

# Snowfall
ds = ghcnd.load_precipitation(
    stations=["USW00094728"],
    variables=["PRCP", "SNOW", "SNWD"],  # precip, snowfall, snow depth
)
```

#### ISD Sub-Daily Precipitation

For extreme rainfall intensity (flash flood analysis).

```python
from xtremax.datasets import isd

# Sub-daily precipitation accumulations
ds = isd.load_hourly(
    stations=["720534"],
    variables=["precipitation_1h", "precipitation_6h"],
    start_date="2000-01-01",
)
```

### Ocean — Sea Surface Temperature

#### NDBC Buoys (`xtremax.datasets.ndbc`)

NOAA buoy network. ~1,000 moorings with SST, wave, and wind observations.

```python
from xtremax.datasets import ndbc

# SST from a specific buoy
ds = ndbc.load_sst(
    stations=["41002", "41004"],         # South Atlantic buoys
    start_date="1990-01-01",
)
# ds["sea_surface_temperature"] in °C, hourly

# Search buoys by region
buoys = ndbc.find_stations(
    bbox=(-90, 25, -80, 30),            # Gulf of Mexico
    variables=["sea_surface_temperature"],
    min_years=20,
)
```

**Provider:** NOAA NDBC | **Access:** ERDDAP (via `erddapy`) | **License:** Public Domain
**Quirks:** Real-time data differs from QC'd archive. Sensor biofouling can cause drift. Use `qc_flag` filtering.

### Ocean — Wave Height Extremes

#### NDBC Buoy Waves

Same buoy network, wave variables.

```python
from xtremax.datasets import ndbc

# Significant wave height
ds = ndbc.load_waves(
    stations=["46025", "46042"],         # Southern California buoys
    variables=["WVHT", "DPD", "MWD"],   # sig. wave height, period, direction
    start_date="1990-01-01",
)
# ds["WVHT"] in meters

# Full pipeline: load → mask → extract → fit
from xtremax.xarray import quantile_threshold, decluster_separation
from xtremax.xarray.masks import coverage_mask

mask = coverage_mask(ds["WVHT"], time_dim="time", freq="YS", min_coverage=0.8)
u = quantile_threshold(ds["WVHT"], q=0.95, mask=mask)
peaks = decluster_separation(ds["WVHT"], threshold=u, min_separation=48)  # 48 hours
```

### Ocean — Sea Level Extremes

#### UHSLC Tide Gauges (`xtremax.datasets.uhslc`)

University of Hawaii Sea Level Center. ~1,300 tide gauge stations globally.

```python
from xtremax.datasets import uhslc

# Hourly sea level
ds = uhslc.load_sea_level(
    stations=["057"],                    # Honolulu
    frequency="hourly",                  # or "daily"
    start_date="1950-01-01",
)
# ds["sea_level"] in meters relative to station datum

# Search Pacific gauges
gauges = uhslc.find_stations(
    bbox=(120, -40, -70, 40),           # Pacific basin
    min_years=50,
)
```

**Provider:** University of Hawaii | **Access:** HTTP download | **License:** Public Domain (citation required)
**Quirks:** Vertical datum varies by station. Tidal signal included — use `uhslc.detide()` for non-tidal residuals (storm surge).

#### GESLA (`xtremax.datasets.gesla`)

Global Extreme Sea Level Analysis. ~500 tide gauges with pre-computed non-tidal residuals and annual maxima — purpose-built for EVA.

```python
from xtremax.datasets import gesla

# Pre-computed annual maxima (ready for GEV fitting)
ds = gesla.load_annual_maxima(
    stations=["newlyn-p038"],
    start_date="1920-01-01",
)
# ds["annual_max_sea_level"] in meters — pipe directly to stationary_gev

# Non-tidal residuals (storm surge) for POT analysis
ds = gesla.load_surge(
    stations=["newlyn-p038"],
)
# ds["surge"] = observed - predicted tide, in meters

# Full EVA pipeline
from xtremax.models import stationary_gev
from numpyro.infer import MCMC, NUTS
mcmc = MCMC(NUTS(stationary_gev), num_warmup=1000, num_samples=2000)
mcmc.run(rng_key, obs=ds["annual_max_sea_level"].values)
```

**Provider:** NOC / University of Liverpool | **Access:** HTTP download | **License:** CC-BY 4.0
**Quirks:** Pre-processed residuals (quality varies by site). Annual maxima provided separately — ideal for block maxima EVA.

#### NOAA CO-OPS (`xtremax.datasets.coops`)

U.S. tide gauges with 6-minute resolution. Best temporal resolution of any sea level dataset.

```python
from xtremax.datasets import coops

# 6-minute water level observations
ds = coops.load_water_level(
    station="8518750",                   # The Battery, NYC
    start_date="2000-01-01",
    end_date="2023-12-31",
    datum="MSL",                         # mean sea level reference
)
# ds["water_level"] in meters, 6-min resolution

# Tidal predictions (for computing surge residuals)
ds_pred = coops.load_predictions(station="8518750", datum="MSL")
surge = ds["water_level"] - ds_pred["predicted_water_level"]
```

**Provider:** NOAA | **Access:** REST API (`api.tidesandcurrents.noaa.gov`) | **License:** Public Domain
**Quirks:** U.S. only. API rate limits (not aggressive). Real-time data available before QC.

### Common API Pattern

Every dataset module follows the same interface:

```python
# Search for stations
stations_df = module.find_stations(
    bbox=(lon_min, lat_min, lon_max, lat_max),  # optional spatial filter
    variables=["VAR1", "VAR2"],                  # required variables
    min_years=30,                                # minimum record length
    country="US",                                # optional country filter
)
# Returns: pandas DataFrame with station_id, name, lat, lon, elevation, years

# Load data
ds = module.load_<variable_group>(
    stations=["ID1", "ID2"],             # station IDs from find_stations
    variables=["VAR1"],                  # specific variables
    start_date="1980-01-01",             # ISO date string
    end_date="2023-12-31",              # optional
)
# Returns: xr.Dataset with dims (time, station_id)

# Cache management
from xtremax.datasets import cache_info, clear_cache
cache_info()    # shows cached datasets and disk usage
clear_cache()   # removes all cached data
```

### Unit Normalization

All loaders auto-convert to SI-adjacent standard units:

| Variable | Standard Unit | Raw Sources |
|----------|--------------|-------------|
| Temperature | °C | tenths of °C (GHCN-D), °F (some GSOD), K |
| Wind speed | m/s | knots (GSOD, NDBC), mph, 0.1 m/s |
| Precipitation | mm | tenths of mm (GHCN-D), inches (GSOD) |
| Sea level | m | mm (UHSLC), feet (CO-OPS), cm |
| Wave height | m | (already standard) |

The original units are stored in `ds.attrs["original_units"]` for traceability.

### Application Domain Summary

| Domain | Primary Dataset | Secondary | Variables | Typical EVA |
|--------|----------------|-----------|-----------|-------------|
| Temperature extremes | GHCN-D | ECA&D | TMAX, TMIN | Annual maxima → GEV, heat/cold wave POT |
| Wind speed extremes | GSOD | ISD (sub-daily) | MXSPD, GUST | Annual max gust → GEV, storm POT |
| Precipitation extremes | GHCN-D | ISD (sub-daily) | PRCP | Annual max daily rainfall → GEV, intensity POT |
| SST extremes | NDBC | — | SST | Marine heatwave POT, seasonal max → GEV |
| Wave height extremes | NDBC | — | WVHT | Design wave → GEV, storm wave POT |
| Sea level extremes | GESLA | UHSLC, CO-OPS | surge, water level | Annual max surge → GEV, storm surge POT |

### Covariates (`xtremax.datasets.covariates`)

Nonstationary extreme value models need covariates — external drivers that explain why the distribution of extremes changes over time. `xtremax.datasets.covariates` provides loaders for the standard climate indices used in EVA and attribution studies. Every loader returns an `xr.DataArray` with a `time` coordinate, ready to plug into `nonstationary_gev`, `trend_quantile_threshold`, or any model that accepts covariate arguments.

#### Global warming indicators (`covariates.global_warming`)

```python
from xtremax.datasets.covariates import global_warming

# Global Mean Surface Temperature anomaly (the most common EVA covariate)
gmst = global_warming.load_gmst(source="noaa")  # or "gistemp", "hadcrut"
# xr.DataArray, monthly, 1880–present, °C anomaly vs 1901–2000 baseline

# Atmospheric CO₂ (Mauna Loa)
co2 = global_warming.load_co2(frequency="monthly")  # or "annual"
# xr.DataArray, monthly, 1958–present, ppm

# Total Solar Irradiance
tsi = global_warming.load_tsi()
# xr.DataArray, monthly, 1978–present, W/m²

# Effective Radiative Forcing (IPCC AR6)
erf = global_warming.load_radiative_forcing()
# xr.DataArray, annual, 1750–present, W/m²
```

| Index | Abbrev | Temporal | Span | Source | What it measures |
|-------|--------|----------|------|--------|-----------------|
| Global Mean Surface Temperature | GMST | monthly | 1880– | NOAA NCEI | Land+ocean temperature anomaly |
| Global Surface Air Temperature | GSAT | monthly | 1850– | NASA GISS | 2m air temperature anomaly |
| CO₂ concentration | CO₂ | monthly | 1958– | NOAA GML (Mauna Loa) | Atmospheric CO₂ in ppm |
| Total Solar Irradiance | TSI | monthly | 1978– | NOAA/PMOD | Solar energy at top of atmosphere |
| Effective Radiative Forcing | ERF | annual | 1750– | IPCC AR6 | Net anthropogenic+natural forcing |

#### ENSO indices (`covariates.enso`)

```python
from xtremax.datasets.covariates import enso

# Oceanic Niño Index (official ENSO definition)
oni = enso.load_oni()
# xr.DataArray, 3-month running mean, 1950–present

# Niño 3.4 SST anomaly
nino34 = enso.load_nino34()
# xr.DataArray, monthly, 1950–present, °C

# Southern Oscillation Index (atmospheric ENSO)
soi = enso.load_soi()
# xr.DataArray, monthly, 1866–present

# Multivariate ENSO Index (5-variable EOF)
mei = enso.load_mei()
# xr.DataArray, bimonthly, 1950–present

# Convenience: load all ENSO indices as a Dataset
ds_enso = enso.load_all()
# xr.Dataset with oni, nino34, soi, mei as variables
```

| Index | Abbrev | Temporal | Span | Source | What it measures |
|-------|--------|----------|------|--------|-----------------|
| Oceanic Niño Index | ONI | 3-month mean | 1950– | NOAA CPC | Official ENSO state (Niño 3.4 running mean) |
| Niño 3.4 | NINO34 | monthly | 1950– | NOAA PSL | SST anomaly in equatorial Pacific (5°N–5°S, 120°–170°W) |
| Southern Oscillation Index | SOI | monthly | 1866– | NOAA PSL | SLP difference Tahiti–Darwin (atmospheric ENSO) |
| Multivariate ENSO Index | MEI | bimonthly | 1950– | NOAA PSL | Combined EOF of SLP, SST, wind, OLR |

#### Oscillation indices (`covariates.oscillations`)

```python
from xtremax.datasets.covariates import oscillations

# North Atlantic Oscillation
nao = oscillations.load_nao()
# xr.DataArray, monthly, 1864–present

# Pacific Decadal Oscillation
pdo = oscillations.load_pdo()
# xr.DataArray, monthly, 1900–present

# Atlantic Multidecadal Oscillation
amo = oscillations.load_amo()
# xr.DataArray, monthly, 1870–present

# Indian Ocean Dipole / Dipole Mode Index
dmi = oscillations.load_dmi()
# xr.DataArray, monthly, 1950–present

# Arctic Oscillation
ao = oscillations.load_ao()
# xr.DataArray, monthly, 1950–present

# Southern Annular Mode / Antarctic Oscillation
sam = oscillations.load_sam()
# xr.DataArray, monthly, 1950–present

# Pacific/North American pattern
pna = oscillations.load_pna()
# xr.DataArray, monthly, 1950–present

# Load all oscillation indices at once
ds_osc = oscillations.load_all()
```

| Index | Abbrev | Temporal | Span | Source | What it measures |
|-------|--------|----------|------|--------|-----------------|
| North Atlantic Oscillation | NAO | monthly | 1864– | NOAA PSL | SLP dipole: Azores–Iceland. Drives European weather |
| Pacific Decadal Oscillation | PDO | monthly | 1900– | NOAA NCEI | Leading EOF of North Pacific SST. Multi-decadal regime shifts |
| Atlantic Multidecadal Oscillation | AMO | monthly | 1870– | NOAA NCEI | Low-frequency North Atlantic SST. Affects hurricane activity, Sahel rain |
| Indian Ocean Dipole | DMI | monthly | 1950– | NOAA PSL | SST gradient across Indian Ocean. Drives monsoon variability |
| Arctic Oscillation | AO | monthly | 1950– | NOAA CPC | Arctic polar vortex strength. Cold outbreaks when negative |
| Southern Annular Mode | SAM | monthly | 1950– | NOAA PSL | Southern Hemisphere jet position. Affects Antarctic/Aus weather |
| Pacific/North American | PNA | monthly | 1950– | NOAA CPC | Rossby wave train: Pacific→N. America. Modulates temperature/precip |

#### Tropical & subseasonal indices (`covariates.tropical`)

```python
from xtremax.datasets.covariates import tropical

# Madden-Julian Oscillation (RMM1/RMM2 amplitudes + phase)
mjo = tropical.load_mjo()
# xr.Dataset with variables: rmm1, rmm2, phase, amplitude
# Daily, 1979–present

# Quasi-Biennial Oscillation (stratospheric zonal wind)
qbo = tropical.load_qbo(level=50)  # 50 hPa, or 30, 20
# xr.DataArray, monthly, 1953–present, m/s

# Accumulated Cyclone Energy (Atlantic hurricane activity)
ace = tropical.load_ace()
# xr.DataArray, annual, 1950–present

# Indian Monsoon Index (All-India Rainfall)
imi = tropical.load_indian_monsoon()
# xr.DataArray, monthly/seasonal, 1871–present
```

| Index | Abbrev | Temporal | Span | Source | What it measures |
|-------|--------|----------|------|--------|-----------------|
| Madden-Julian Oscillation | MJO | daily | 1979– | NOAA CPC | 30–90 day tropical convection envelope (phase + amplitude) |
| Quasi-Biennial Oscillation | QBO | monthly | 1953– | NOAA PSL | Stratospheric wind reversal (~28-month cycle) |
| Accumulated Cyclone Energy | ACE | annual | 1950– | NOAA NHC | Integrated hurricane intensity × duration per season |
| Indian Monsoon Index | IMI | seasonal | 1871– | IMD/NOAA | Southwest monsoon rainfall strength |

#### Common API pattern

All covariate loaders share the same interface:

```python
# Individual index
idx = module.load_<index>(
    start_date="1950-01-01",    # optional time filter
    end_date="2023-12-31",      # optional
    detrend=False,              # optionally remove linear trend
    normalize=False,            # optionally standardize to zero mean, unit variance
)
# Returns: xr.DataArray with time coordinate

# All indices from a category
ds = module.load_all(start_date="1950-01-01")
# Returns: xr.Dataset with one variable per index

# Metadata
module.describe("nao")
# Prints: name, source URL, temporal resolution, citation, physical description
```

#### Integration with nonstationary models

Covariates plug directly into the model zoo and threshold utilities:

```python
from xtremax.datasets.covariates import global_warming, enso
from xtremax.models import nonstationary_gev
from xtremax.xarray import trend_quantile_threshold

# Load covariates
gmst = global_warming.load_gmst()
oni = enso.load_oni()

# GMST as covariate for nonstationary GEV (climate attribution)
mcmc.run(rng_key, obs=annual_maxima, covariates=gmst)

# ENSO-modulated threshold for tropical precipitation
u_enso = trend_quantile_threshold(da_precip, q=0.95, time_dim="time", covariates=oni)

# Multiple covariates via xr.Dataset
import xarray as xr
covariates = xr.Dataset({"gmst": gmst, "oni": oni, "pdo": pdo})
mcmc.run(rng_key, obs=annual_maxima, covariates=covariates)
```

#### Data source: NOAA PSL

Most indices come from the same place — NOAA Physical Sciences Laboratory (`psl.noaa.gov/data/correlation/`). These are plain-text files with a consistent format: year in column 1, monthly values in columns 2–13. The loaders handle the parsing, missing value codes (-99.9, -999), and conversion to xarray.


## GP Layer — GPJax Integration (`xtremax.gp`)

All Gaussian process functionality uses **GPJax** directly. The `xtremax.gp` module provides thin wrappers that handle parameter storage, integration with NumPyro models, and the two main inference modes: variational GP and sparse variational GP.

### Design Decisions

- **GPJax is the GP engine.** No hand-rolled kernels or GP math. GPJax provides kernels, mean functions, likelihoods, and variational families.
- **Helpers for parameter persistence.** GP hyperparameters (kernel lengthscale, variance, inducing locations) need to be stored, loaded, and passed between training and prediction. `xtremax.gp.utils` provides `save_gp_params()` / `load_gp_params()` for this.
- **Two inference modes.** Variational GP for moderate data (< ~5k sites). Sparse variational GP with inducing points for large spatial domains.

### Variational GP (`xtremax.gp.variational`)

Full variational inference over the GP posterior. Uses GPJax's `VariationalGaussianProcess` under the hood.

```python
from xtremax.gp import VariationalGP

# Build a variational GP for a GEV parameter field
vgp = VariationalGP(
    kernel="matern32",             # GPJax kernel name or kernel instance
    mean_function="zero",          # or "linear", or a callable
    likelihood="gaussian",
    jitter=1e-6,
)

# Fit: optimizes ELBO over variational parameters + kernel hyperparams
vgp_state = vgp.fit(
    X=coordinates,                 # (n_sites, 2) — lon/lat
    y=loc_estimates,               # (n_sites,) — e.g. GEV location at each site
    optimizer="adam",
    learning_rate=0.01,
    num_iters=2000,
    key=rng_key,
)

# Predict at new locations
pred_mean, pred_var = vgp.predict(vgp_state, X_new=grid_coordinates)

# Access fitted parameters
print(vgp_state.kernel_params)     # {"lengthscale": ..., "variance": ...}
print(vgp_state.variational_params) # {"mean": ..., "scale_tril": ...}
```

### Sparse Variational GP (`xtremax.gp.sparse`)

For large spatial domains. Selects M inducing points and approximates the full GP posterior. Supports minibatch training.

```python
from xtremax.gp import SparseVariationalGP

svgp = SparseVariationalGP(
    kernel="matern52",
    n_inducing=100,                # number of inducing points
    inducing_init="kmeans",        # or "random", or explicit (M, 2) array
    whiten=True,                   # numerically stable whitened parameterization
)

# Fit with minibatching for large datasets
svgp_state = svgp.fit(
    X=coordinates,                 # (n_sites, 2)
    y=loc_estimates,               # (n_sites,)
    batch_size=256,                # minibatch size
    optimizer="adam",
    learning_rate=0.01,
    num_iters=5000,
    key=rng_key,
)

# Predict — same interface as VariationalGP
pred_mean, pred_var = svgp.predict(svgp_state, X_new=grid_coordinates)

# Inducing point locations (may have moved during optimization)
print(svgp_state.inducing_locations)  # (100, 2)
```

### Parameter Utilities (`xtremax.gp.utils`)

Helpers for storing and recovering GP state across sessions.

```python
from xtremax.gp.utils import save_gp_params, load_gp_params, summarize_gp

# Save fitted GP state to disk
save_gp_params(svgp_state, path="gp_loc_field.json")

# Load and resume
svgp_state = load_gp_params(path="gp_loc_field.json")

# Human-readable summary of fitted GP
summarize_gp(svgp_state)
# Kernel: Matern52(lengthscale=2.34, variance=0.87)
# Inducing points: 100 (whitened)
# ELBO: -234.5
# Noise variance: 0.12
```

### Integration with Spatial Models

The GP wrappers plug directly into the spatial GEV model:

```python
from xtremax.gp import SparseVariationalGP
from xtremax.models import spatial_gev

# Pre-train GP fields for each GEV parameter (warm start)
gp_loc = SparseVariationalGP(kernel="matern32", n_inducing=50)
gp_logscale = SparseVariationalGP(kernel="matern32", n_inducing=50)
gp_shape = SparseVariationalGP(kernel="matern32", n_inducing=30)

# Pass into spatial model — uses GP posterior as spatial prior
mcmc.run(
    rng_key,
    obs=block_maxima,
    coordinates=site_coords,
    covariates=elevation,
    gp_loc=gp_loc,
    gp_logscale=gp_logscale,
    gp_shape=gp_shape,
)
```


## Copula Utilities

NumPyro already provides copula distributions (`numpyro.distributions.copulas`). Rather than re-implementing, `xtremax` provides thin helpers that make it easy to combine NumPyro copulas with extreme value marginals.

```python
from xtremax.distributions import GEVD
from numpyro.distributions.copulas import GaussianCopula
import numpyro

def spatial_gev_copula_model(obs, correlation_matrix):
    """GEV marginals + Gaussian copula for spatial dependence."""
    loc = numpyro.sample("loc", dist.Normal(0, 10))
    scale = numpyro.sample("scale", dist.HalfNormal(5))
    shape = numpyro.sample("shape", dist.Normal(0, 0.3))

    marginal = GEVD(loc=loc, scale=scale, concentration=shape)

    # Use NumPyro's Gaussian copula directly
    copula = GaussianCopula(
        marginal_dist=marginal,
        correlation_matrix=correlation_matrix,
    )
    numpyro.sample("obs", copula, obs=obs)
```
