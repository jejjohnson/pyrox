"""Pure-JAX primitive functions for extreme value distributions.

Every function takes arrays and returns arrays. No NumPyro, no class
state, no side effects — compatible with ``jax.jit``, ``jax.grad``, and
``jax.vmap``.

The Distribution classes in :mod:`xtremax.distributions` are thin
wrappers that route ``log_prob`` / ``cdf`` / ``icdf`` / ``mean`` /
``return_level`` to these primitives.
"""

from __future__ import annotations

from xtremax.primitives.frechet import (
    frechet_cdf,
    frechet_icdf,
    frechet_log_prob,
    frechet_mean,
    frechet_return_level,
)
from xtremax.primitives.gev import (
    gev_cdf,
    gev_icdf,
    gev_log_prob,
    gev_mean,
    gev_return_level,
)
from xtremax.primitives.gpd import (
    gpd_cdf,
    gpd_icdf,
    gpd_log_prob,
    gpd_mean,
    gpd_return_level,
)
from xtremax.primitives.gumbel import (
    gumbel_cdf,
    gumbel_icdf,
    gumbel_log_prob,
    gumbel_mean,
    gumbel_return_level,
)
from xtremax.primitives.weibull import (
    weibull_cdf,
    weibull_icdf,
    weibull_log_prob,
    weibull_mean,
    weibull_return_level,
)


__all__ = [
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
    "gev_mean",
    "gev_return_level",
    # GPD
    "gpd_cdf",
    "gpd_icdf",
    "gpd_log_prob",
    "gpd_mean",
    "gpd_return_level",
    # Gumbel (GEV, ξ = 0)
    "gumbel_cdf",
    "gumbel_icdf",
    "gumbel_log_prob",
    "gumbel_mean",
    "gumbel_return_level",
    # Weibull (GEV, ξ < 0)
    "weibull_cdf",
    "weibull_icdf",
    "weibull_log_prob",
    "weibull_mean",
    "weibull_return_level",
]
