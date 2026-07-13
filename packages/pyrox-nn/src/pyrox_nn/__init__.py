"""Bayesian, spectral, and coordinate-encoding neural network layers.

After the ``geonnax`` split, ``pyrox_nn`` consists of:

* **Deterministic primitives** (re-exported from ``geonnax`` via
  `pyrox_nn._geonnax`): SIREN, MFN, conditioning bases, encoders,
  Slepian, orthogonal RFF, SNGP covariance container, etc.
* **Bayesian wrappers** (per-family modules): ``_dense``, ``_features``,
  ``_siren``, ``_mfn``, ``_vssgp``, ``_heteroscedastic``, ``_ensemble``,
  ``_sngp``, ``_conditioning``, ``_bnf``, ``_slepian``. Each holds a
  ``geonnax`` core and registers NumPyro sample / param sites.
* **BNF + pandas-free helpers** (`pyrox_nn._bnf`,
  `pyrox_nn._features`).

``MCDropout`` and ``LinearCore`` were dropped — use
``equinox.nn.Dropout(p=rate, inference=False)`` and
``equinox.nn.Linear`` directly.
"""

from geonnax.geo import (
    cyclic_encode,
    deg2rad,
    lonlat_scale,
    lonlat_to_cartesian3d,
    spherical_harmonic_encode,
)

from pyrox_nn._bnf import (
    BayesianNeuralField,
    FourierFeatures,
    InteractionFeatures,
    SeasonalFeatures,
    Standardization,
)
from pyrox_nn._conditioning import (
    BayesianAffineModulation,
    BayesianConcatConditioner,
    BayesianHyperLinear,
    ConditionedRFFNet,
    HyperFourierFeatures,
)
from pyrox_nn._dense import (
    DenseDVI,
    DenseFlipout,
    DenseHierarchical,
    DenseNCP,
    DenseReparameterization,
    DenseVariational,
    DenseVariationalDropout,
    NCPNormalOutput,
)
from pyrox_nn._ensemble import DenseRank1, LayerNormEnsemble, MultiHeadAttentionBE
from pyrox_nn._features import (
    ArcCosineFourierFeatures,
    HSGPFeatures,
    LaplaceCosineFeatures,
    LaplaceFourierFeatures,
    MaternCosineFeatures,
    MaternFourierFeatures,
    RandomKitchenSinks,
    RBFCosineFeatures,
    RBFFourierFeatures,
    VariationalFourierFeatures,
    fourier_features,
    interaction_features,
    seasonal_features,
    seasonal_frequencies,
    standardize,
    unstandardize,
)
from pyrox_nn._geonnax import (
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
    HybridSphericalSlepianEncoder,
    HyperLinear,
    HyperSIREN,
    LaplaceRandomFeatureCovariance,
    LonLatScale,
    NCPContinuousPerturb,
    OrthogonalRandomFeatures,
    SirenDense,
    SlepianEncoder,
    SphericalHarmonicEncoder,
    mfn_forward,
)
from pyrox_nn._heteroscedastic import MCSigmoidDenseFA, MCSoftmaxDenseFA
from pyrox_nn._mfn import BayesianFourierNet, BayesianGaborNet
from pyrox_nn._siren import BayesianSIREN
from pyrox_nn._slepian import BayesianSlepianEncoder
from pyrox_nn._sngp import RandomFeatureGaussianProcess
from pyrox_nn._vssgp import DeepVSSGP


__version__ = "0.1.0"  # x-release-please-version

__all__ = [
    "SIREN",
    "AbstractConditioner",
    "AffineModulation",
    "ArcCosineFourierFeatures",
    "BayesianAffineModulation",
    "BayesianConcatConditioner",
    "BayesianFourierNet",
    "BayesianGaborNet",
    "BayesianHyperLinear",
    "BayesianNeuralField",
    "BayesianSIREN",
    "BayesianSlepianEncoder",
    "Cartesian3DEncoder",
    "ConcatConditioner",
    "ConditionedINR",
    "ConditionedRFFNet",
    "CyclicEncoder",
    "DeepVSSGP",
    "Deg2Rad",
    "DenseDVI",
    "DenseFlipout",
    "DenseHierarchical",
    "DenseNCP",
    "DenseRank1",
    "DenseReparameterization",
    "DenseVariational",
    "DenseVariationalDropout",
    "FiLM",
    "FourierFeatures",
    "FourierFilter",
    "FourierNet",
    "GaborFilter",
    "GaborNet",
    "HSGPFeatures",
    "HybridSphericalSlepianEncoder",
    "HyperFourierFeatures",
    "HyperLinear",
    "HyperSIREN",
    "InteractionFeatures",
    "LaplaceCosineFeatures",
    "LaplaceFourierFeatures",
    "LaplaceRandomFeatureCovariance",
    "LayerNormEnsemble",
    "LonLatScale",
    "MCSigmoidDenseFA",
    "MCSoftmaxDenseFA",
    "MaternCosineFeatures",
    "MaternFourierFeatures",
    "MultiHeadAttentionBE",
    "NCPContinuousPerturb",
    "NCPNormalOutput",
    "OrthogonalRandomFeatures",
    "RBFCosineFeatures",
    "RBFFourierFeatures",
    "RandomFeatureGaussianProcess",
    "RandomKitchenSinks",
    "SeasonalFeatures",
    "SirenDense",
    "SlepianEncoder",
    "SphericalHarmonicEncoder",
    "Standardization",
    "VariationalFourierFeatures",
    "__version__",
    "cyclic_encode",
    "deg2rad",
    "fourier_features",
    "interaction_features",
    "lonlat_scale",
    "lonlat_to_cartesian3d",
    "mfn_forward",
    "seasonal_features",
    "seasonal_frequencies",
    "spherical_harmonic_encode",
    "standardize",
    "unstandardize",
]
