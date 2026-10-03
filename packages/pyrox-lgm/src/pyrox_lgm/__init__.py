"""Latent Gaussian models: GMRF components, PC priors and ``inla()``.

Precision-form latent Gaussian models on top of gaussx and kernellib. This
release is the package scaffold (P6); the components, priors and inference
land in P7-P9. ``pyrox_lgm`` never imports ``pyrox_gp``.
"""

from pyrox_lgm._priors import (
    PCAR1Rho,
    PCBYM2Phi,
    PCMatern,
    PCPrecision,
    StructureSpectrum,
    structure_spectrum,
)


__version__ = "0.0.0"  # x-release-please-version

__all__ = [
    "PCAR1Rho",
    "PCBYM2Phi",
    "PCMatern",
    "PCPrecision",
    "StructureSpectrum",
    "__version__",
    "structure_spectrum",
]
