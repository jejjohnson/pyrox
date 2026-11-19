"""Gaussian NumPyro distributions on gaussx's structured operators.

The multivariate normals, the linear-Gaussian state-space models, the Markov
chain over states and the GMRF wrapper, as ``numpyro.distributions``
classes whose covariance or precision is a lineax operator. The numerics are
gaussx's; this package gives them NumPyro's support, shapes and seeding.
``pyrox_gauss`` imports gaussx and NumPyro and never imports ``pyrox``.
"""

# Registers numpyro kl_divergence for the two MVN classes.
from pyrox_gauss import _numpyro_kl  # noqa: F401
from pyrox_gauss._gmrf import GMRFDistribution
from pyrox_gauss._lgssm import LGSSM, LGSSMFactory, MaskedLGSSM
from pyrox_gauss._markov_gaussian import MarkovGaussian
from pyrox_gauss._mvn import MultivariateNormal
from pyrox_gauss._mvn_base import AbstractMultivariateNormal
from pyrox_gauss._mvn_prec import MultivariateNormalPrecision


__version__ = "0.0.0"  # x-release-please-version

__all__ = [
    "LGSSM",
    "AbstractMultivariateNormal",
    "GMRFDistribution",
    "LGSSMFactory",
    "MarkovGaussian",
    "MaskedLGSSM",
    "MultivariateNormal",
    "MultivariateNormalPrecision",
]
