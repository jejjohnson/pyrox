"""Synthetic extreme value data generators.

Temporal trajectories (GMST, energy balance), spatial domain/terrain
generators, and variable-specific extreme generators for temperature,
precipitation, and wind.
"""

from __future__ import annotations

from xtremax.simulations.extremes import (
    compute_climate_signal,
    generate_spatial_field,
    simulate_precip_extremes,
    simulate_temp_extremes,
    simulate_wind_extremes,
)
from xtremax.simulations.spatial import (
    SpatialFeatureExtractor,
    augment_spatial_features,
    compute_advanced_climate_signal,
    create_iberian_domain,
    generate_fractal_terrain,
    generate_iberia_mask,
)
from xtremax.simulations.temporal import (
    generate_gmst_trajectory,
    generate_physical_gmst,
)


__all__ = [
    "SpatialFeatureExtractor",
    "augment_spatial_features",
    "compute_advanced_climate_signal",
    "compute_climate_signal",
    "create_iberian_domain",
    "generate_fractal_terrain",
    "generate_gmst_trajectory",
    "generate_iberia_mask",
    "generate_physical_gmst",
    "generate_spatial_field",
    "simulate_precip_extremes",
    "simulate_temp_extremes",
    "simulate_wind_extremes",
]
