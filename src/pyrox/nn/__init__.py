"""Bayesian, spectral, and coordinate-encoding neural network layers.

Dense / Bayesian-linear layers (``pyrox.nn._layers``):

* :class:`Deg2Rad` — element-wise degrees-to-radians conversion.
* :class:`LonLatScale` — affine lon/lat scaling into ``[-1, 1]``.
* :class:`Cartesian3DEncoder` — lon/lat lift to unit Cartesian
  coordinates on :math:`S^2`.
* :class:`CyclicEncoder` — periodic cos/sin encoding.
* :class:`SphericalHarmonicEncoder` — real spherical-harmonic features.
* :class:`DenseReparameterization` — weight-space Bayesian linear via
  the reparameterization trick.
* :class:`DenseFlipout` — variance-reduced Bayesian linear via the
  Flipout estimator.
* :class:`DenseVariational` — user-supplied prior + posterior
  callables for full flexibility.
* :class:`DenseVariationalDropout` — sparse variational dropout dense
  layer with per-weight learnable dropout rates (Molchanov et al.,
  2017).
* :class:`MCDropout` — always-on dropout for Monte Carlo uncertainty.
* :class:`DenseNCP` — Noise Contrastive Prior: deterministic backbone
  + scaled stochastic perturbation.
* :class:`MCSoftmaxDenseFA` — heteroscedastic multi-class output head
  with input-dependent low-rank+diag logit noise (Collier et al. 2021).
* :class:`MCSigmoidDenseFA` — same noise model with sigmoid output
  for multi-label classification.
* :class:`DenseRank1` — rank-1 ensemble dense layer (BatchEnsemble /
  rank-1 BNN, Wen et al. 2020 / Dusenberry et al. 2020).
* :class:`NCPContinuousPerturb` — input perturbation for NCP.
* :class:`NCPNormalOutput` — output-side NCP KL regulariser
  (Hafner et al. 2018).
* :class:`RBFFourierFeatures` — SSGP-style [cos, sin] RFF (Gaussian).
* :class:`RBFCosineFeatures` — cos(Wx + b) RFF variant (Gaussian).
* :class:`MaternFourierFeatures` — SSGP-style RFF (Student-t).
* :class:`MaternCosineFeatures` — cos(Wx + b) RFF variant (Student-t).
* :class:`LaplaceFourierFeatures` — SSGP-style RFF (Cauchy).
* :class:`LaplaceCosineFeatures` — cos(Wx + b) RFF variant (Cauchy).
* :class:`ArcCosineFourierFeatures` — arc-cosine / ReLU features.
* :class:`RandomKitchenSinks` — RFF + learned linear head.
* :class:`SirenDense` — single sine-activated dense layer (SIREN).
* :class:`SIREN` — multi-layer sinusoidal representation network.
* :class:`BayesianSIREN` — SIREN with regime-scaled Normal priors.
* :class:`DeepVSSGP` — deep random feature expansion for variational
  SSGP (Cutajar et al. 2017).
* :class:`RandomFeatureGaussianProcess` — SNGP output head with RFF
  feature map and Laplace covariance over linear weights (Liu et al.,
  2020).
* :class:`LaplaceRandomFeatureCovariance` — pure-functional precision
  container used by SNGP.

Bayesian Neural Field stack (``pyrox.nn._bnf``):

* :class:`Standardization` — affine normalization with fixed mean/std.
* :class:`FourierFeatures` — dyadic-frequency cos/sin basis per input.
* :class:`SeasonalFeatures` — period-and-harmonic cos/sin basis.
* :class:`InteractionFeatures` — element-wise products on column pairs.
* :class:`BayesianNeuralField` — full BNF MLP with Logistic(0, 1) priors.

Geographic / spherical helpers (``pyrox.nn._geo``):

* :func:`deg2rad`, :func:`lonlat_scale`, :func:`lonlat_to_cartesian3d`,
  :func:`cyclic_encode`, :func:`spherical_harmonic_encode` —
  pure-JAX preprocessing helpers for lon/lat inputs.

Pure-JAX feature helpers (``pyrox.nn._features``):

* :func:`fourier_features`, :func:`seasonal_features`,
  :func:`interaction_features`, :func:`standardize`,
  :func:`unstandardize` — pandas-free building blocks the BNF layers
  wrap.
"""

from pyrox.nn._bnf import (
    BayesianNeuralField,
    FourierFeatures,
    InteractionFeatures,
    SeasonalFeatures,
    Standardization,
)
from pyrox.nn._conditioning import (
    AbstractConditioner,
    AffineModulation,
    BayesianAffineModulation,
    BayesianConcatConditioner,
    BayesianHyperLinear,
    ConcatConditioner,
    ConditionedINR,
    ConditionedRFFNet,
    FiLM,
    HyperFourierFeatures,
    HyperLinear,
    HyperSIREN,
)
from pyrox.nn._ensemble import DenseRank1
from pyrox.nn._features import (
    fourier_features,
    interaction_features,
    seasonal_features,
    seasonal_frequencies,
    standardize,
    unstandardize,
)
from pyrox.nn._geo import (
    cyclic_encode,
    deg2rad,
    lonlat_scale,
    lonlat_to_cartesian3d,
    spherical_harmonic_encode,
)
from pyrox.nn._heteroscedastic import MCSigmoidDenseFA, MCSoftmaxDenseFA
from pyrox.nn._layers import (
    SIREN,
    ArcCosineFourierFeatures,
    BayesianSIREN,
    Cartesian3DEncoder,
    CyclicEncoder,
    DeepVSSGP,
    Deg2Rad,
    DenseFlipout,
    DenseNCP,
    DenseReparameterization,
    DenseVariational,
    DenseVariationalDropout,
    HSGPFeatures,
    LaplaceCosineFeatures,
    LaplaceFourierFeatures,
    LonLatScale,
    MaternCosineFeatures,
    MaternFourierFeatures,
    MCDropout,
    NCPContinuousPerturb,
    NCPNormalOutput,
    OrthogonalRandomFeatures,
    RandomKitchenSinks,
    RBFCosineFeatures,
    RBFFourierFeatures,
    SirenDense,
    SphericalHarmonicEncoder,
    VariationalFourierFeatures,
)
from pyrox.nn._sngp import (
    LaplaceRandomFeatureCovariance,
    RandomFeatureGaussianProcess,
)


__all__ = [
    "SIREN",
    "AbstractConditioner",
    "AffineModulation",
    "ArcCosineFourierFeatures",
    "BayesianAffineModulation",
    "BayesianConcatConditioner",
    "BayesianHyperLinear",
    "BayesianNeuralField",
    "BayesianSIREN",
    "Cartesian3DEncoder",
    "ConcatConditioner",
    "ConditionedINR",
    "ConditionedRFFNet",
    "CyclicEncoder",
    "DeepVSSGP",
    "Deg2Rad",
    "DenseFlipout",
    "DenseNCP",
    "DenseRank1",
    "DenseReparameterization",
    "DenseVariational",
    "DenseVariationalDropout",
    "FiLM",
    "FourierFeatures",
    "HSGPFeatures",
    "HyperFourierFeatures",
    "HyperLinear",
    "HyperSIREN",
    "InteractionFeatures",
    "LaplaceCosineFeatures",
    "LaplaceFourierFeatures",
    "LaplaceRandomFeatureCovariance",
    "LonLatScale",
    "MCDropout",
    "MCSigmoidDenseFA",
    "MCSoftmaxDenseFA",
    "MaternCosineFeatures",
    "MaternFourierFeatures",
    "NCPContinuousPerturb",
    "NCPNormalOutput",
    "OrthogonalRandomFeatures",
    "RBFCosineFeatures",
    "RBFFourierFeatures",
    "RandomFeatureGaussianProcess",
    "RandomKitchenSinks",
    "SeasonalFeatures",
    "SirenDense",
    "SphericalHarmonicEncoder",
    "Standardization",
    "VariationalFourierFeatures",
    "cyclic_encode",
    "deg2rad",
    "fourier_features",
    "interaction_features",
    "lonlat_scale",
    "lonlat_to_cartesian3d",
    "seasonal_features",
    "seasonal_frequencies",
    "spherical_harmonic_encode",
    "standardize",
    "unstandardize",
]
