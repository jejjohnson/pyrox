"""Variable-specific extreme value generators.

GEV temperature extremes, Gamma/Poisson precipitation intensities and
durations, and Weibull wind gusts, composed with spatial fields and a
GMST trajectory to produce Clausius-Clapeyron-scaled covariate response.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import genextreme, weibull_min


# ==============================================================================
# SPATIAL MODULE: FIELDS & COVARIATES
# ==============================================================================


def generate_spatial_field(
    n_sites: int = 158,
    bounds: tuple[float, float, float, float] = (-9, 3, 36, 44),  # Spain Box
    seed: int = 42,
) -> xr.Dataset:
    """
    Generates static spatial data: Lat, Lon, Elevation, Distance to Coast.

    Args:
        bounds: (lon_min, lon_max, lat_min, lat_max)
    """
    rng = np.random.default_rng(seed)
    lon_min, lon_max, lat_min, lat_max = bounds

    # Random spatial locations
    lon = rng.uniform(lon_min, lon_max, n_sites)
    lat = rng.uniform(lat_min, lat_max, n_sites)

    # Synthetic Elevation (correlated with distance from center for 'mountain' effect)
    # Simple heuristic: higher in the north/center
    center_lat = (lat_min + lat_max) / 2
    dist_factor = np.abs(lat - center_lat)
    elevation = rng.exponential(500, n_sites) + (dist_factor * 200)
    elevation = np.clip(elevation, 0, 3400)  # Clip to realistic max (Mulhacen approx)

    ds = xr.Dataset(
        coords={"site": np.arange(n_sites)},
        data_vars={
            "lon": (("site",), lon),
            "lat": (("site",), lat),
            "elevation": (("site",), elevation),
        },
    )
    return ds


# ==============================================================================
# 3. SPATIOTEMPORAL MODULE: SIGNAL COMBINATION
# ==============================================================================


def compute_climate_signal(
    spatial_ds: xr.Dataset,
    gmst_da: xr.DataArray,
    base_val: float,
    coeffs: dict[str, float],
) -> xr.DataArray:
    """
    Combines Spatial and Temporal fields to create a mean climate signal (Mu).

    Formula: Mu = Base + C1*Elev + C2*Lat + C3*GMST + (C4*GMST*Elev)
    """
    # Broadcast GMST to (year, site) and Spatial to (year, site)
    # Xarray handles broadcasting automatically by dimension name

    # 1. Static Effects
    elev_effect = coeffs.get("elevation", 0.0) * (spatial_ds["elevation"] / 1000.0)
    lat_effect = coeffs.get("lat", 0.0) * (spatial_ds["lat"] - spatial_ds["lat"].mean())

    # 2. Temporal Effects (The Signal)
    time_effect = coeffs.get("gmst", 0.0) * gmst_da

    # 3. Interaction Effects (e.g., higher elevations warm faster — elevation-dependent)
    interaction = (
        coeffs.get("interaction", 0.0) * gmst_da * (spatial_ds["elevation"] / 1000.0)
    )

    # Sum components
    mu = base_val + elev_effect + lat_effect + time_effect + interaction

    # Ensure correct dimension order
    return mu.transpose("year", "site")


# ==============================================================================
# 4. EXTREMES MODULE: DISTRIBUTION GENERATORS
# ==============================================================================


def simulate_temp_extremes(
    mu: xr.DataArray,
    scale: float = 1.5,
    # Negative shape = bounded (Weibull-like); positive = heavy tail (Frechet).
    shape: float = -0.1,
    seed: int = 42,
) -> xr.Dataset:
    """
    Simulates Block Maxima Temperature via the GEV distribution.

    Args:
        mu: Location parameter (varying over time/space).
        scale: Scale parameter (assumed constant here, but could be array).
        shape: Shape parameter (xi).
    """
    shape_dim = mu.shape

    # Scipy uses shape 'c' where c = -xi (sign flip vs typical EVT notation).
    # However, standard GEV: mu + (sigma/xi)* ((p^-xi) - 1)
    # We will use simple inversion sampling or scipy's rvs

    # Note: Scipy genextreme args are (c, loc, scale). c = -shape parameter (xi)
    # if xi < 0 (Weibull domain), scipy c > 0
    c = -shape

    data = genextreme.rvs(
        c, loc=mu.values, scale=scale, size=shape_dim, random_state=seed
    )

    ds = mu.to_dataset(name="mu_tmax")
    ds["tmax"] = (("year", "site"), data)
    ds.attrs["distribution"] = "GEV"
    return ds


def simulate_precip_extremes(
    spatial_ds: xr.Dataset, gmst_da: xr.DataArray, seed: int = 42
) -> xr.Dataset:
    """
    Simulates Precipitation Extremes:
    1. Intensity (Rx1day): Gamma or GEV distribution.
    2. Duration (Consecutive Wet Days - CWD): Poisson/Geometric approach.
    """
    if "year" not in gmst_da.sizes:
        raise ValueError("gmst_da must include a 'year' dimension.")
    if "site" not in spatial_ds.sizes:
        raise ValueError("spatial_ds must include a 'site' dimension.")
    rng = np.random.default_rng(seed)

    # --- A. Intensity (Rx1day - Annual Max Precip) ---
    # Physical intuition: Warmer air holds more moisture (Clausius-Clapeyron ~7%/K)
    # Base precip depends on elevation
    base_intensity = 40.0 + 0.01 * spatial_ds["elevation"]  # mm
    cc_scaling = 1.0 + 0.07 * gmst_da  # Simple 7% per degree scaling

    # Expected intensity location parameter
    loc_intensity = base_intensity * cc_scaling

    # Generate using Gamma distribution (Shape/Scale parameterization)
    # Mean = shape * scale. Let's fix shape, vary scale.
    gamma_shape = 4.0
    gamma_scale = loc_intensity / gamma_shape

    # Values must be broadcast manually for numpy sampling if using arrays
    # But Xarray math handles the broadcasting for parameters
    # We sample:
    rx1day = rng.gamma(
        shape=gamma_shape,
        scale=gamma_scale.values.transpose(),  # (year, site)
    )

    # --- B. Duration (CWD - Max consecutive wet days) ---
    # Duration may decrease slightly with GMST in the Mediterranean (drier summers).
    base_duration = 10.0 + 0.001 * spatial_ds["elevation"]  # days
    trend_duration = 1.0 - 0.2 * gmst_da

    lambda_param = base_duration * trend_duration
    lambda_param = np.maximum(1.0, lambda_param)  # Ensure positive

    # Poisson for count data
    cwd = rng.poisson(lambda_param.values.transpose())

    ds = xr.Dataset(
        coords={"year": gmst_da.year, "site": spatial_ds.site},
        data_vars={
            "rx1day": (("year", "site"), rx1day),
            "cwd": (("year", "site"), cwd),
            "lon": spatial_ds.lon,
            "lat": spatial_ds.lat,
            "elevation": spatial_ds.elevation,
        },
    )
    return ds


def simulate_wind_extremes(
    spatial_ds: xr.Dataset, gmst_da: xr.DataArray, seed: int = 42
) -> xr.Dataset:
    """
    Simulates Extreme Wind Speeds (Gusts) using Weibull distribution.
    """
    # Wind often higher at higher elevation and near coast (ignored coast for now)
    base_wind = 15.0 + 0.01 * spatial_ds["elevation"]  # m/s

    # Trend: Maybe slight increase or noise. Let's assume just noise around trend.
    # Note: Weibull is defined by Shape (k) and Scale (lambda)
    # Mean = lambda * Gamma(1 + 1/k)

    k_shape = 2.0  # Rayleigh-like
    w_scale = base_wind  # Scale parameter varies by site

    # Broadcast to time
    w_scale_time = w_scale + (0.5 * gmst_da)  # Slight increase with warming?

    # Sample
    # Scipy weibull_min takes 'c' as shape parameter
    wind_max = weibull_min.rvs(
        c=k_shape,
        scale=w_scale_time.values.transpose(),
        loc=0,
        random_state=seed,
    )

    ds = xr.Dataset(
        coords={"year": gmst_da.year, "site": spatial_ds.site},
        data_vars={
            "wind_max": (("year", "site"), wind_max),
            "lon": spatial_ds.lon,
            "lat": spatial_ds.lat,
            "elevation": spatial_ds.elevation,
        },
    )
    return ds
