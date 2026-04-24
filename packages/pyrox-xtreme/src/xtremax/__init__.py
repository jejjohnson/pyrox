"""xtremax: JAX/NumPyro-native library for extreme value modeling."""

from __future__ import annotations

from xtremax import distributions, extraction, point_processes, primitives, simulations
from xtremax.distributions import (
    FrechetType2GEVD,
    GeneralizedExtremeValueDistribution,
    GeneralizedParetoDistribution,
    GumbelType1GEVD,
    WeibullType3GEVD,
)
from xtremax.primitives import (
    frechet_cdf,
    frechet_icdf,
    frechet_log_prob,
    frechet_mean,
    frechet_return_level,
    gev_cdf,
    gev_icdf,
    gev_log_prob,
    gev_mean,
    gev_return_level,
    gpd_cdf,
    gpd_icdf,
    gpd_log_prob,
    gpd_mean,
    gpd_return_level,
    gumbel_cdf,
    gumbel_icdf,
    gumbel_log_prob,
    gumbel_mean,
    gumbel_return_level,
    weibull_cdf,
    weibull_icdf,
    weibull_log_prob,
    weibull_mean,
    weibull_return_level,
)


__version__: str = "0.0.0"


__all__ = [
    "FrechetType2GEVD",
    "GeneralizedExtremeValueDistribution",
    "GeneralizedParetoDistribution",
    "GumbelType1GEVD",
    "WeibullType3GEVD",
    "distributions",
    "extraction",
    "frechet_cdf",
    "frechet_icdf",
    "frechet_log_prob",
    "frechet_mean",
    "frechet_return_level",
    "gev_cdf",
    "gev_icdf",
    "gev_log_prob",
    "gev_mean",
    "gev_return_level",
    "gpd_cdf",
    "gpd_icdf",
    "gpd_log_prob",
    "gpd_mean",
    "gpd_return_level",
    "gumbel_cdf",
    "gumbel_icdf",
    "gumbel_log_prob",
    "gumbel_mean",
    "gumbel_return_level",
    "point_processes",
    "primitives",
    "simulations",
    "weibull_cdf",
    "weibull_icdf",
    "weibull_log_prob",
    "weibull_mean",
    "weibull_return_level",
]
