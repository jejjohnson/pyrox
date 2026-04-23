"""NumPyro-compatible extreme value distributions."""

from __future__ import annotations

from xtremax.distributions.frechet import FrechetType2GEVD
from xtremax.distributions.gevd import GeneralizedExtremeValueDistribution
from xtremax.distributions.gpd import GeneralizedParetoDistribution
from xtremax.distributions.gumbel import GumbelType1GEVD
from xtremax.distributions.weibull import WeibullType3GEVD


__all__ = [
    "FrechetType2GEVD",
    "GeneralizedExtremeValueDistribution",
    "GeneralizedParetoDistribution",
    "GumbelType1GEVD",
    "WeibullType3GEVD",
]
