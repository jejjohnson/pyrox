"""Bayesian and uncertainty-aware neural network layers.

Dense / Bayesian-linear layers (``pyrox.nn._layers``):

* :class:`DenseReparameterization` — weight-space Bayesian linear via
  the reparameterization trick.
* :class:`DenseFlipout` — variance-reduced Bayesian linear via the
  Flipout estimator.
* :class:`DenseVariational` — user-supplied prior + posterior
  callables for full flexibility.
* :class:`MCDropout` — always-on dropout for Monte Carlo uncertainty.
* :class:`DenseNCP` — Noise Contrastive Prior: deterministic backbone
  + scaled stochastic perturbation.
* :class:`NCPContinuousPerturb` — input perturbation for NCP.
* :class:`RBFFourierFeatures` — SSGP-style [cos, sin] RFF (Gaussian).
* :class:`RBFCosineFeatures` — cos(Wx + b) RFF variant (Gaussian).
* :class:`MaternFourierFeatures` — SSGP-style RFF (Student-t).
* :class:`LaplaceFourierFeatures` — SSGP-style RFF (Cauchy).
* :class:`ArcCosineFourierFeatures` — arc-cosine / ReLU features.
* :class:`RandomKitchenSinks` — RFF + learned linear head.

Multiplicative Filter Networks (``pyrox.nn._layers``):

* :class:`FourierFilter` — single Fourier filter primitive.
* :class:`GaborFilter` — single Gabor filter primitive.
* :class:`FourierNet` — multiplicative Fourier filter network.
* :class:`GaborNet` — multiplicative Gabor filter network.
* :class:`BayesianFourierNet` — FourierNet with NumPyro priors.
* :class:`BayesianGaborNet` — GaborNet with NumPyro priors.
* :func:`mfn_forward` — pure-JAX MFN forward helper.

Bayesian Neural Field stack (``pyrox.nn._bnf``):

* :class:`Standardization` — affine normalization with fixed mean/std.
* :class:`FourierFeatures` — dyadic-frequency cos/sin basis per input.
* :class:`SeasonalFeatures` — period-and-harmonic cos/sin basis.
* :class:`InteractionFeatures` — element-wise products on column pairs.
* :class:`BayesianNeuralField` — full BNF MLP with Logistic(0, 1) priors.

Pure-JAX feature helpers (``pyrox.nn._features``):

* :func:`fourier_features`, :func:`seasonal_features`,
  :func:`interaction_features`, :func:`standardize`,
  :func:`unstandardize` — pandas-free building blocks the layers wrap.
"""

from pyrox.nn._bnf import (
    BayesianNeuralField,
    FourierFeatures,
    InteractionFeatures,
    SeasonalFeatures,
    Standardization,
)
from pyrox.nn._features import (
    fourier_features,
    interaction_features,
    seasonal_features,
    seasonal_frequencies,
    standardize,
    unstandardize,
)
from pyrox.nn._layers import (
    ArcCosineFourierFeatures,
    BayesianFourierNet,
    BayesianGaborNet,
    DenseFlipout,
    DenseNCP,
    DenseReparameterization,
    DenseVariational,
    FourierFilter,
    FourierNet,
    GaborFilter,
    GaborNet,
    HSGPFeatures,
    LaplaceFourierFeatures,
    MaternFourierFeatures,
    MCDropout,
    NCPContinuousPerturb,
    OrthogonalRandomFeatures,
    RandomKitchenSinks,
    RBFCosineFeatures,
    RBFFourierFeatures,
    VariationalFourierFeatures,
    mfn_forward,
)


__all__ = [
    "ArcCosineFourierFeatures",
    "BayesianFourierNet",
    "BayesianGaborNet",
    "BayesianNeuralField",
    "DenseFlipout",
    "DenseNCP",
    "DenseReparameterization",
    "DenseVariational",
    "FourierFeatures",
    "FourierFilter",
    "FourierNet",
    "GaborFilter",
    "GaborNet",
    "HSGPFeatures",
    "InteractionFeatures",
    "LaplaceFourierFeatures",
    "MCDropout",
    "MaternFourierFeatures",
    "NCPContinuousPerturb",
    "OrthogonalRandomFeatures",
    "RBFCosineFeatures",
    "RBFFourierFeatures",
    "RandomKitchenSinks",
    "SeasonalFeatures",
    "Standardization",
    "VariationalFourierFeatures",
    "fourier_features",
    "interaction_features",
    "mfn_forward",
    "seasonal_features",
    "seasonal_frequencies",
    "standardize",
    "unstandardize",
]
