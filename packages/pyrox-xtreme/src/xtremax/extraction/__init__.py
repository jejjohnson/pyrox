"""Extraction utilities for extreme values on xarray data.

Block maxima, threshold selection, and declustering — the three standard
preprocessing steps for extreme value analysis on spatiotemporal data.
"""

from __future__ import annotations

from xtremax.extraction.block_maxima import (
    declustered_block_maxima,
    r_largest_block_maxima,
    sliding_block_maxima,
    spatial_block_maxima,
    temporal_block_maxima,
)
from xtremax.extraction.decluster import (
    decluster_runs,
    decluster_separation,
    estimate_extremal_index,
)
from xtremax.extraction.threshold import (
    constant_threshold,
    quantile_threshold,
    rolling_threshold,
    seasonal_threshold,
    temporal_threshold,
)


__all__ = [
    "constant_threshold",
    "decluster_runs",
    "decluster_separation",
    "declustered_block_maxima",
    "estimate_extremal_index",
    "quantile_threshold",
    "r_largest_block_maxima",
    "rolling_threshold",
    "seasonal_threshold",
    "sliding_block_maxima",
    "spatial_block_maxima",
    "temporal_block_maxima",
    "temporal_threshold",
]

# Quantile-regression threshold selection needs scikit-learn, which is an
# optional `[threshold]` extra. Only re-export when available so the core
# `import xtremax` path works without sklearn installed.
try:
    from xtremax.extraction.quantile_regression import (
        XarrayQuantileRegressor,
        quantile_regression_threshold,
    )
except ImportError:
    pass
else:
    __all__ += ["XarrayQuantileRegressor", "quantile_regression_threshold"]
