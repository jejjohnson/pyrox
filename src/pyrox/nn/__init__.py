"""Bayesian and uncertainty-aware neural network layers.

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
"""

from pyrox.nn._layers import (
    ArcCosineFourierFeatures,
    DenseFlipout,
    DenseNCP,
    DenseReparameterization,
    DenseVariational,
    LaplaceFourierFeatures,
    MaternFourierFeatures,
    MCDropout,
    NCPContinuousPerturb,
    RandomKitchenSinks,
    RBFCosineFeatures,
    RBFFourierFeatures,
)


__all__ = [
    "ArcCosineFourierFeatures",
    "DenseFlipout",
    "DenseNCP",
    "DenseReparameterization",
    "DenseVariational",
    "LaplaceFourierFeatures",
    "MCDropout",
    "MaternFourierFeatures",
    "NCPContinuousPerturb",
    "RBFCosineFeatures",
    "RBFFourierFeatures",
    "RandomKitchenSinks",
]
