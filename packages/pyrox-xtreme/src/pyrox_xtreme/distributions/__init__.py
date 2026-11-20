"""NumPyro-compatible extreme value distributions."""

from __future__ import annotations

from pyrox_xtreme.distributions.frechet import FrechetType2GEVD
from pyrox_xtreme.distributions.gevd import GeneralizedExtremeValueDistribution
from pyrox_xtreme.distributions.gpd import GeneralizedParetoDistribution
from pyrox_xtreme.distributions.gumbel import GumbelType1GEVD
from pyrox_xtreme.distributions.weibull import WeibullType3GEVD


__all__ = [
    "FrechetType2GEVD",
    "GeneralizedExtremeValueDistribution",
    "GeneralizedParetoDistribution",
    "GumbelType1GEVD",
    "WeibullType3GEVD",
]
