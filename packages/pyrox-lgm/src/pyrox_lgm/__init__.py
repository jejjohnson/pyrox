"""Latent Gaussian models: GMRF components, PC priors and ``inla()``.

Precision-form latent Gaussian models on top of gaussx and kernellib:
temporal, areal and SPDE components, penalised-complexity priors, ``inla()``
and its diagnostics. ``pyrox_lgm`` never imports ``pyrox_gp``.
"""

from pyrox_lgm._components import (
    AR1,
    BYM2,
    CAR,
    IID,
    RW1,
    RW2,
    SPDE,
    AbstractComponent,
    Besag,
    Generic,
    Kronecker,
    Leroux,
    MarkovGP,
    Replicate,
)
from pyrox_lgm._condition import ConditionedLGM, ConditionedNonGaussLGM
from pyrox_lgm._diagnostics import Diagnostics, diagnostics
from pyrox_lgm._formula import f
from pyrox_lgm._inla import inla
from pyrox_lgm._likelihood import (
    AbstractObservation,
    Bernoulli,
    Binomial,
    Gaussian,
    NegativeBinomial,
    Poisson,
)
from pyrox_lgm._model import LGM, FixedEffects, LGMPrior
from pyrox_lgm._numpyro import lgm_factor, lgm_laplace_factor
from pyrox_lgm._priors import (
    PCAR1Rho,
    PCBYM2Phi,
    PCMatern,
    PCPrecision,
    StructureSpectrum,
    structure_spectrum,
)
from pyrox_lgm._result import INLAResult, Summary


__version__ = "0.1.5"  # x-release-please-version

__all__ = [
    "AR1",
    "BYM2",
    "CAR",
    "IID",
    "LGM",
    "RW1",
    "RW2",
    "SPDE",
    "AbstractComponent",
    "AbstractObservation",
    "Bernoulli",
    "Besag",
    "Binomial",
    "ConditionedLGM",
    "ConditionedNonGaussLGM",
    "Diagnostics",
    "FixedEffects",
    "Gaussian",
    "Generic",
    "INLAResult",
    "Kronecker",
    "LGMPrior",
    "Leroux",
    "MarkovGP",
    "NegativeBinomial",
    "PCAR1Rho",
    "PCBYM2Phi",
    "PCMatern",
    "PCPrecision",
    "Poisson",
    "Replicate",
    "StructureSpectrum",
    "Summary",
    "__version__",
    "diagnostics",
    "f",
    "inla",
    "lgm_factor",
    "lgm_laplace_factor",
    "structure_spectrum",
]
