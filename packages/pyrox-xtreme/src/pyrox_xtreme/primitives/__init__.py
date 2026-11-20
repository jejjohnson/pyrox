"""Pure-JAX primitive functions for extreme value distributions.

Every function takes arrays and returns arrays. No NumPyro, no class
state, no side effects — compatible with ``jax.jit``, ``jax.grad``, and
``jax.vmap``.

The Distribution classes in :mod:`pyrox_xtreme.distributions` are thin
wrappers that route ``log_prob`` / ``cdf`` / ``icdf`` / ``mean`` /
``return_level`` to these primitives.
"""

from __future__ import annotations

from pyrox_xtreme.primitives.frechet import (
    frechet_cdf,
    frechet_icdf,
    frechet_log_prob,
    frechet_mean,
    frechet_return_level,
)
from pyrox_xtreme.primitives.gev import (
    gev_cdf,
    gev_icdf,
    gev_log_prob,
    gev_log_survival,
    gev_mean,
    gev_return_level,
    gev_survival,
)
from pyrox_xtreme.primitives.gpd import (
    gpd_cdf,
    gpd_icdf,
    gpd_log_prob,
    gpd_log_survival,
    gpd_mean,
    gpd_return_level,
    gpd_survival,
)
from pyrox_xtreme.primitives.gumbel import (
    gumbel_cdf,
    gumbel_icdf,
    gumbel_log_prob,
    gumbel_mean,
    gumbel_return_level,
)
from pyrox_xtreme.primitives.nonstationary import (
    assemble_nonstationary_gev_fields,
    expected_exceedances,
    nonstationary_return_level,
    nonstationary_return_period,
)
from pyrox_xtreme.primitives.spatial import (
    design_matrix,
    pairwise_distances,
    two_range_correlation,
)
from pyrox_xtreme.primitives.weibull import (
    weibull_cdf,
    weibull_icdf,
    weibull_log_prob,
    weibull_mean,
    weibull_return_level,
)


__all__ = [
    "assemble_nonstationary_gev_fields",
    "design_matrix",
    "expected_exceedances",
    # Fréchet (GEV, ξ > 0)
    "frechet_cdf",
    "frechet_icdf",
    "frechet_log_prob",
    "frechet_mean",
    "frechet_return_level",
    # GEV
    "gev_cdf",
    "gev_icdf",
    "gev_log_prob",
    "gev_log_survival",
    "gev_mean",
    "gev_return_level",
    "gev_survival",
    # GPD
    "gpd_cdf",
    "gpd_icdf",
    "gpd_log_prob",
    "gpd_log_survival",
    "gpd_mean",
    "gpd_return_level",
    "gpd_survival",
    # Gumbel (GEV, ξ = 0)
    "gumbel_cdf",
    "gumbel_icdf",
    "gumbel_log_prob",
    "gumbel_mean",
    "gumbel_return_level",
    "nonstationary_return_level",
    "nonstationary_return_period",
    "pairwise_distances",
    "two_range_correlation",
    # Weibull (GEV, ξ < 0)
    "weibull_cdf",
    "weibull_icdf",
    "weibull_log_prob",
    "weibull_mean",
    "weibull_return_level",
]
