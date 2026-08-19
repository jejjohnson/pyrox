# Simulations

Synthetic extremes generators for testing, teaching, and method development —
data with a *known* ground truth, so you can check whether an estimator
recovers the parameters you put in.

Unlike the rest of the package these are NumPy-based and take an integer
`seed` rather than a JAX PRNG key.

## Extreme-value simulators

Per-variable simulators with physically sensible defaults: temperature block
maxima from a GEV, precipitation extremes, and wind gusts from a Weibull. Each
salts its RNG with a per-variable constant, so the same `seed` across
variables still yields independent fields rather than rank-correlated ones.

::: xtremax.simulations
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - simulate_temp_extremes
        - simulate_precip_extremes
        - simulate_wind_extremes

## Climate signals & GMST

Covariate trajectories for non-stationary EVT — a synthetic global mean
surface temperature series to regress the GEV parameters against.
`generate_gmst_trajectory` is the cheap statistical version;
`generate_physical_gmst` runs a 0-D energy-balance model for a trajectory with
physical structure rather than a fitted curve.

::: xtremax.simulations
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - generate_gmst_trajectory
        - generate_physical_gmst
        - compute_climate_signal
        - compute_advanced_climate_signal

## Spatial domains & terrain

A worked spatial testbed: a procedural Iberian Peninsula with fractal
Brownian-motion terrain, plus the feature extractors (elevation, distance to
coast, and friends) that turn that geometry into covariates for spatial
pooling.

::: xtremax.simulations
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - create_iberian_domain
        - generate_iberia_mask
        - generate_fractal_terrain
        - generate_spatial_field
        - augment_spatial_features
        - SpatialFeatureExtractor
