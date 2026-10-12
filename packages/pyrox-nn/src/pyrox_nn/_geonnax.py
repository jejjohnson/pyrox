"""Deterministic neural-network primitives re-exported from ``geonnax``.

These were originally defined in ``pyrox_nn._layers`` /
``pyrox_nn._conditioning`` / etc., but with the package split the
deterministic forwards now live in ``geonnax``. ``pyrox_nn`` keeps the
names available so existing user imports
(``from pyrox_nn import SIREN``) keep working.
"""

from __future__ import annotations

from geonnax import (
    SIREN,
    AbstractConditioner,
    AffineModulation,
    Cartesian3DEncoder,
    ConcatConditioner,
    ConditionedINR,
    CyclicEncoder,
    Deg2Rad,
    FiLM,
    FourierFilter,
    FourierNet,
    GaborFilter,
    GaborNet,
    GeoFNO,
    HybridSphericalSlepianEncoder,
    HyperLinear,
    HyperSIREN,
    LaplaceRandomFeatureCovariance,
    LearnedDeformation,
    LonLatScale,
    ManifoldNO,
    NCPContinuousPerturb,
    OrthogonalRandomFeatures,
    SirenDense,
    SlepianEncoder,
    SphericalHarmonicEncoder,
    mfn_forward,
)
from geonnax.geo import (
    cyclic_encode,
    deg2rad,
    lonlat_scale,
    lonlat_to_cartesian3d,
    spherical_harmonic_encode,
)
from geonnax.layers import BasisTransform


__all__ = [
    "SIREN",
    "AbstractConditioner",
    "AffineModulation",
    "BasisTransform",
    "Cartesian3DEncoder",
    "ConcatConditioner",
    "ConditionedINR",
    "CyclicEncoder",
    "Deg2Rad",
    "FiLM",
    "FourierFilter",
    "FourierNet",
    "GaborFilter",
    "GaborNet",
    "GeoFNO",
    "HybridSphericalSlepianEncoder",
    "HyperLinear",
    "HyperSIREN",
    "LaplaceRandomFeatureCovariance",
    "LearnedDeformation",
    "LonLatScale",
    "ManifoldNO",
    "NCPContinuousPerturb",
    "OrthogonalRandomFeatures",
    "SirenDense",
    "SlepianEncoder",
    "SphericalHarmonicEncoder",
    "cyclic_encode",
    "deg2rad",
    "lonlat_scale",
    "lonlat_to_cartesian3d",
    "mfn_forward",
    "spherical_harmonic_encode",
]
