"""Synthetic spatial-domain generators.

Procedural Iberian Peninsula mask, fractal terrain on a regular grid,
terrain-feature extraction (slope, aspect, coast distance), and
spatiotemporal climate-signal composition.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt, gaussian_filter


# ==============================================================================
# 1. DOMAIN GENERATION: THE IBERIAN MASK
# ==============================================================================


def generate_iberia_mask(lat_grid: np.ndarray, lon_grid: np.ndarray) -> np.ndarray:
    """
    Procedurally generates a boolean mask approximating the Iberian Peninsula
    and Balearic Islands using geometrical primitives.
    """
    # 1. Main Peninsula (Rough approximation using rotated ellipse/intersection)
    # Center approx (40, -4)
    dx = (lon_grid - (-3.5)) * 0.85
    dy = lat_grid - 40.0

    # Rotated squared Euclidean distance (simple implicit function)
    # Rotated slightly counter-clockwise
    angle = np.radians(10)
    dx_rot = dx * np.cos(angle) - dy * np.sin(angle)
    dy_rot = dx * np.sin(angle) + dy * np.cos(angle)

    # Super-ellipseish shape for the blocky peninsula
    peninsula = (np.abs(dx_rot / 5.5) ** 4 + np.abs(dy_rot / 3.5) ** 4) <= 1.0

    # 2. Add Balearic Islands (Mallorca, Menorca, Ibiza)
    # Simple circles
    def make_island(lat0, lon0, rad):
        return ((lat_grid - lat0) ** 2 + (lon_grid - lon0) ** 2) < rad**2

    mallorca = make_island(39.6, 2.9, 0.4)
    menorca = make_island(40.0, 4.1, 0.2)
    ibiza = make_island(38.9, 1.4, 0.2)

    # Combine
    land_mask = peninsula | mallorca | menorca | ibiza
    return land_mask


# ==============================================================================
# 2. TOPOGRAPHY GENERATOR: FRACTAL NOISE
# ==============================================================================


def generate_fractal_terrain(
    shape: tuple[int, int],
    scale: float = 100.0,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generates coherent terrain using Fractal Brownian Motion (fBm).
    Instead of random elevations, this creates mountain ranges and valleys.
    """
    # Use a local RNG. `np.random.seed(...)` would mutate NumPy's global
    # state and leak reproducibility coupling into any downstream code
    # that draws from NumPy afterwards.
    rng = np.random.default_rng(seed)
    terrain = np.zeros(shape)

    # Add layers of noise (octaves)
    for i in range(octaves):
        freq = lacunarity**i
        amp = persistence**i

        # Generate random noise grid
        noise = rng.normal(0, 1, shape)

        # Smooth it to create correlation (Cheap Perlin proxy via Gaussian)
        # Higher frequency = smaller sigma (rougher features)
        sigma = (shape[0] / freq) / 2.0
        smoothed = gaussian_filter(noise, sigma=sigma)

        # Normalize and add
        smoothed = (smoothed - smoothed.mean()) / (smoothed.std() + 1e-6)
        terrain += smoothed * amp

    # Normalize to 0-1
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    return terrain


def create_iberian_domain(
    res_deg: float = 0.1,  # Grid resolution (approx 10km)
    bounds: tuple[float, float, float, float] = (-10, 5, 36, 44),
    seed: int = 42,
) -> xr.Dataset:
    """
    Creates the master spatial dataset with Lat, Lon, Elevation, and Land Mask.
    """
    lon_min, lon_max, lat_min, lat_max = bounds
    # `np.linspace(min, max, n)` puts points at both endpoints, giving
    # spacing `(max-min)/(n-1)` not `res_deg` (e.g. 36–44 at 0.1 would
    # yield ~0.1013). Use `np.arange` so realized spacing equals the
    # requested resolution exactly.
    lats = np.arange(lat_min, lat_max, res_deg)
    lons = np.arange(lon_min, lon_max, res_deg)
    n_lat = lats.size
    n_lon = lons.size
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # 1. Generate Mask
    mask = generate_iberia_mask(lat_grid, lon_grid)

    # 2. Generate Topography (Fractal)
    raw_topo = generate_fractal_terrain((n_lat, n_lon), seed=seed)

    # 3. Shape Topography to Physical Geography
    # Bias towards Pyrenees (North) and Sierra Nevada (South)
    # Simple linear ramp + bumps
    pyrenees_bias = np.exp(-((lat_grid - 42.5) ** 2) / 0.5) * 1.5
    sierra_bias = np.exp(-((lat_grid - 37.0) ** 2 + (lon_grid + 3.0) ** 2) / 0.8) * 1.5
    central_bias = np.exp(-((lat_grid - 40.5) ** 2 + (lon_grid + 4.0) ** 2) / 2.0) * 0.5

    # Apply bias and clip to land
    elevation = (
        raw_topo * 1500
        + (pyrenees_bias * 1500)
        + (sierra_bias * 2000)
        + (central_bias * 800)
    )
    elevation = np.where(mask, elevation, -100)  # Ocean is -100m
    elevation = np.clip(elevation, -100, 3400)  # Max Mulhacen/Aneto height

    ds = xr.Dataset(
        coords={"lat": lats, "lon": lons},
        data_vars={
            "mask": (("lat", "lon"), mask),
            "elevation": (("lat", "lon"), elevation),
        },
    )
    return ds


# ==============================================================================
# 3. MODULAR FEATURE EXTRACTORS
# ==============================================================================


class SpatialFeatureExtractor:
    """
    Static methods to derive physical features from a Digital Elevation Model (DEM).
    """

    @staticmethod
    def get_distance_to_coast(ds: xr.Dataset) -> xr.DataArray:
        """
        Calculates Euclidean distance to the nearest coast (mask boundary).
        Uses SciPy's Distance Transform.
        """
        mask = ds["mask"].values
        # distance_transform_edt calculates distance to background (0)
        # We want distance from land (1) to ocean (0)
        dist_grid = distance_transform_edt(mask)

        # Convert pixels to approx km (assuming 0.1 deg ~ 11km)
        # This is an approximation, real geodesy would require Haversine
        res_deg = ds.lat[1] - ds.lat[0]
        dist_km = dist_grid * float(res_deg) * 111.0

        return xr.DataArray(
            dist_km, coords=ds.coords, dims=ds.dims, name="dist_to_coast"
        )

    @staticmethod
    def get_slope_and_aspect(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Calculates local Slope (magnitude) and Aspect (direction).
        Slope is critical for orographic precipitation.
        Aspect is critical for solar insolation (South vs North facing).
        """
        elev = ds["elevation"].values
        res_deg = float(ds.lat[1] - ds.lat[0])
        dx_meters = res_deg * 111000.0

        # Numpy gradient: returns (dy, dx)
        grad_y, grad_x = np.gradient(elev, dx_meters)

        # Slope (Magnitude) in degrees
        slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        slope_deg = np.degrees(slope_rad)

        # Aspect (Direction)
        # 0 = North, 90 = East, 180 = South, 270 = West
        aspect_rad = np.arctan2(-grad_x, grad_y)  # Mathematical convention to compass
        aspect_deg = (np.degrees(aspect_rad) + 360) % 360

        da_slope = xr.DataArray(slope_deg, coords=ds.coords, dims=ds.dims, name="slope")
        da_aspect = xr.DataArray(
            aspect_deg, coords=ds.coords, dims=ds.dims, name="aspect"
        )

        return da_slope, da_aspect

    @staticmethod
    def get_roughness(ds: xr.Dataset, window_size: int = 3) -> xr.DataArray:
        """
        Calculates Terrain Roughness Index (TRI).
        Roughness = Std Dev of elevation in a local window.
        """
        elev = ds["elevation"].values
        # We use a uniform filter to get local mean, then compute variance
        local_mean = gaussian_filter(elev, sigma=window_size)
        local_sq_mean = gaussian_filter(elev**2, sigma=window_size)

        # Var = E[x^2] - (E[x])^2
        roughness = np.sqrt(np.maximum(0, local_sq_mean - local_mean**2))

        return xr.DataArray(roughness, coords=ds.coords, dims=ds.dims, name="roughness")


# ==============================================================================
# 4. ADVANCED SIGNAL COMPOSITION
# ==============================================================================


def augment_spatial_features(ds: xr.Dataset) -> xr.Dataset:
    """Pipeline to run all extractors and merge into Dataset."""
    extractor = SpatialFeatureExtractor()

    ds["dist_to_coast"] = extractor.get_distance_to_coast(ds)
    ds["slope"], ds["aspect"] = extractor.get_slope_and_aspect(ds)
    ds["roughness"] = extractor.get_roughness(ds)

    # Mask out ocean for clarity (optional, setting features to NaN)
    for var in ["dist_to_coast", "slope", "aspect", "roughness"]:
        ds[var] = ds[var].where(ds["mask"])

    return ds


def compute_advanced_climate_signal(
    spatial_ds: xr.Dataset,
    gmst_da: xr.DataArray,
) -> xr.Dataset:
    """
    Generates Tmax and Precip Mean fields using advanced physics proxies.

    New Physics:
    1. Continentality: Tmax increases with dist_to_coast.
    2. Orographic Lift: Precip increases with Slope * Wind_Direction alignment.
    3. Insolation: Tmax higher on South-facing slopes (Aspect ~ 180).
    """
    # Create empty container

    # --- 1. Temperature Mean (Mu) ---
    # Base: 30C
    # Lapse Rate: -6.5 C / km
    # Continentality: +1.5 C per 100km from coast (Summer effect)
    # Latitude: -0.5 C per degree North
    # Aspect: +1.0 C if South Facing (180 deg), -1.0 if North
    # GMST: +1.5 * GMST

    aspect_factor = -np.cos(
        np.radians(spatial_ds["aspect"])
    )  # +1 at 180 (S), -1 at 0 (N)

    mu_temp_space = (
        30.0
        - 6.5 * (spatial_ds["elevation"] / 1000.0)
        + 0.015 * spatial_ds["dist_to_coast"]
        - 0.5 * (spatial_ds["lat"] - 36)
        + 1.0 * aspect_factor
    )

    # Combine with Time (GMST)
    # Continentality effect may *intensify* with GMST (land warms faster than ocean).
    land_ocean_contrast = 1.0 + (0.2 * spatial_ds["dist_to_coast"] / 100.0)

    mu_temp = mu_temp_space + (gmst_da * 1.5 * land_ocean_contrast)

    # --- 2. Precipitation Intensity Scale (Mu_Precip) ---
    # Base: 30mm
    # Elevation: Increase with height
    # Orographic: Increase if Slope is high (simple proxy)
    # Drying Trend: -5% per degree GMST

    mu_precip_space = (
        30.0
        + 0.02 * spatial_ds["elevation"]
        + 0.5 * spatial_ds["slope"]  # Steeper slopes = more orographic rain
    )

    # Thermodynamic scaling (Clausius Clapeyron + Dynamics)
    # Warmer = more intensity (CC), but circulation changes (dynamics) might reduce it.
    precip_scaling = 1.0 + 0.06 * gmst_da

    mu_precip = mu_precip_space * precip_scaling

    # Pack into Dataset
    ds_out = xr.Dataset()
    ds_out["mu_tmax"] = mu_temp.where(spatial_ds["mask"])
    ds_out["mu_precip"] = mu_precip.where(spatial_ds["mask"])

    return ds_out
