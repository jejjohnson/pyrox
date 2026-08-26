"""Gaussian process building blocks.

* Pure kernel *functions* in `pyrox_gp._src.kernels` — closed-form
  math primitives (RBF, Matern, Periodic, Linear, RationalQuadratic,
  Polynomial, Cosine, White, Constant).
* `Parameterized` kernel classes that wrap those functions with
  constraints, priors, and guide metadata — re-exported from this
  module.
* Multi-output GP structures — `LMCKernel`, `ICMKernel`,
  `OILMMKernel`, and shared inducing-point helpers for explicit
  cross-output structure without monolithic model classes — plus the
  model layer on top: `MultiOutputGPPrior` /
  `MultiOutputConditionedGP` / `mo_gp_factor` for exact
  inference, `MultiOutputSparseGPPrior` /
  `mo_svgp_elbo` / `mo_svgp_factor` for the inducing-input
  variational path (reusing the single-output guides), and
  `OILMMGPPrior` / `ConditionedOILMMGP` for the
  orthogonal-projection exact workflow.
* Collapsed latent-factor regression — `collapsed_lfr_log_prob` /
  `decoder_posterior` / `lfr_predictive_moments`: the linear decoder
  carries a fixed unit-normal prior and is marginalized analytically,
  so the likelihood costs `O(NQP)` in the output dimension instead of
  the dense vec-GP `O(P^3 N^3)`.
* Abstract protocols (`Kernel`, `Guide`,
  `Likelihood`) plus five concrete sparse
  variational guides — `FullRankGuide`, `MeanFieldGuide`,
  `WhitenedGuide`, `NaturalGuide`, `DeltaGuide`.
  Natural-parameter conversion and damped-update primitives live in
  ``gaussx`` (`gaussx.mean_cov_to_natural`,
  `gaussx.natural_to_mean_cov`,
  `gaussx.damped_natural_update`) — `NaturalGuide`
  delegates its math there, and so will the future natural-gradient /
  CVI inference paths.
* Model-facing entry points — `GPPrior`, `ConditionedGP`,
  `SparseGPPrior`, `PathwiseSampler`,
  `DecoupledPathwiseSampler`, `gp_factor`,
  `gp_sample` — the NumPyro-aware shell on top of gaussx linear
  algebra.

*Scalable matrix construction* and *solver strategies* — numerically
stable matrix assembly, implicit operators, batched matvec, Cholesky /
CG / BBMM / LSMR, etc. — live in ``gaussx``. pyrox model entry points
accept any ``gaussx.AbstractSolverStrategy``; the default is
``gaussx.DenseSolver()``.
"""

# State-space (SDE) kernels live in `gaussx._ssm` since gaussx
# 0.0.11; re-export the public names here so ``pyrox_gp.MaternSDE`` etc.
# keep resolving for downstream callers.
from gaussx import (
    ConstantSDE,
    CosineSDE,
    IntegratedWienerSDE,
    MaternSDE,
    PeriodicSDE,
    ProductSDE,
    QuasiPeriodicSDE,
    SDEParams,
    SumSDE,
)

from pyrox_gp._guides import (
    DeltaGuide,
    FullRankGuide,
    MeanFieldGuide,
    NaturalGuide,
    WhitenedGuide,
)
from pyrox_gp._inducing import (
    DecoupledInducingFeatures,
    FourierInducingFeatures,
    InducingFeatures,
    LaplacianInducingFeatures,
    SlepianInducingFeatures,
    SphericalHarmonicInducingFeatures,
    funk_hecke_coefficients,
)
from pyrox_gp._inference import (
    ConjugateVI,
    svgp_elbo,
    svgp_factor,
)
from pyrox_gp._inference_nongauss import (
    ExpectationPropagation,
    GaussNewtonInference,
    LaplaceInference,
    NonGaussConditionedGP,
    PosteriorLinearization,
    QuasiNewtonInference,
)
from pyrox_gp._inference_nongauss_markov import (
    ExpectationPropagationMarkov,
    GaussNewtonMarkovInference,
    LaplaceMarkovInference,
    NonGaussConditionedMarkovGP,
    PosteriorLinearizationMarkov,
)
from pyrox_gp._kernels import (
    RBF,
    Constant,
    Cosine,
    Linear,
    Matern,
    Periodic,
    Polynomial,
    RationalQuadratic,
    White,
)
from pyrox_gp._latent_factor import (
    collapsed_lfr_log_prob,
    decoder_posterior,
    lfr_predictive_moments,
    warp_to_base,
    warped_decoder_posterior,
    warped_lfr_log_prob,
)
from pyrox_gp._latent_factor_models import (
    ConditionedLatentFactorGP,
    LatentFactorGPPrior,
    latent_total_correlation,
    lfr_factor,
    lfr_model,
)
from pyrox_gp._likelihoods import (
    BernoulliLikelihood,
    DistLikelihood,
    GaussianLikelihood,
    HeteroscedasticGaussianLikelihood,
    PoissonLikelihood,
    SoftmaxLikelihood,
    StudentTLikelihood,
)
from pyrox_gp._markov import (
    ConditionedMarkovGP,
    MarkovGPPrior,
    markov_gp_factor,
    markov_gp_sample,
)
from pyrox_gp._markov_flow import (
    NormalizingKalmanPrior,
    normalizing_kalman_factor,
)
from pyrox_gp._models import (
    ConditionedGP,
    GPPrior,
    gp_factor,
    gp_sample,
)
from pyrox_gp._multi_output import (
    ICMKernel,
    LMCKernel,
    MultiOutputInducingVariables,
    OILMMKernel,
    SharedInducingPoints,
)
from pyrox_gp._multi_output_models import (
    ConditionedOILMMGP,
    MultiOutputConditionedGP,
    MultiOutputGPPrior,
    MultiOutputSparseGPPrior,
    OILMMGPPrior,
    mo_gp_factor,
    mo_svgp_elbo,
    mo_svgp_factor,
)
from pyrox_gp._pathwise import (
    DecoupledPathwiseSampler,
    PathwiseFunction,
    PathwiseSampler,
)
from pyrox_gp._protocols import (
    Guide,
    Kernel,
    Likelihood,
    SDEKernel,
)
from pyrox_gp._sparse import SparseGPPrior
from pyrox_gp._sparse_markov import (
    SparseConditionedMarkovGP,
    SparseMarkovGPPrior,
    sparse_markov_elbo,
    sparse_markov_factor,
)
from pyrox_gp._warped import (
    WarpedGaussianLikelihood,
    warped_predictive_moments,
)


__version__ = "0.1.3"  # x-release-please-version

__all__ = [
    "RBF",
    "BernoulliLikelihood",
    "ConditionedGP",
    "ConditionedLatentFactorGP",
    "ConditionedMarkovGP",
    "ConditionedOILMMGP",
    "ConjugateVI",
    "Constant",
    "ConstantSDE",
    "Cosine",
    "CosineSDE",
    "DecoupledInducingFeatures",
    "DecoupledPathwiseSampler",
    "DeltaGuide",
    "DistLikelihood",
    "ExpectationPropagation",
    "ExpectationPropagationMarkov",
    "FourierInducingFeatures",
    "FullRankGuide",
    "GPPrior",
    "GaussNewtonInference",
    "GaussNewtonMarkovInference",
    "GaussianLikelihood",
    "Guide",
    "HeteroscedasticGaussianLikelihood",
    "ICMKernel",
    "InducingFeatures",
    "IntegratedWienerSDE",
    "Kernel",
    "LMCKernel",
    "LaplaceInference",
    "LaplaceMarkovInference",
    "LaplacianInducingFeatures",
    "LatentFactorGPPrior",
    "Likelihood",
    "Linear",
    "MarkovGPPrior",
    "Matern",
    "MaternSDE",
    "MeanFieldGuide",
    "MultiOutputConditionedGP",
    "MultiOutputGPPrior",
    "MultiOutputInducingVariables",
    "MultiOutputSparseGPPrior",
    "NaturalGuide",
    "NonGaussConditionedGP",
    "NonGaussConditionedMarkovGP",
    "NormalizingKalmanPrior",
    "OILMMGPPrior",
    "OILMMKernel",
    "PathwiseFunction",
    "PathwiseSampler",
    "Periodic",
    "PeriodicSDE",
    "PoissonLikelihood",
    "Polynomial",
    "PosteriorLinearization",
    "PosteriorLinearizationMarkov",
    "ProductSDE",
    "QuasiNewtonInference",
    "QuasiPeriodicSDE",
    "RationalQuadratic",
    "SDEKernel",
    "SDEParams",
    "SharedInducingPoints",
    "SlepianInducingFeatures",
    "SoftmaxLikelihood",
    "SparseConditionedMarkovGP",
    "SparseGPPrior",
    "SparseMarkovGPPrior",
    "SphericalHarmonicInducingFeatures",
    "StudentTLikelihood",
    "SumSDE",
    "WarpedGaussianLikelihood",
    "White",
    "WhitenedGuide",
    "__version__",
    "collapsed_lfr_log_prob",
    "decoder_posterior",
    "funk_hecke_coefficients",
    "gp_factor",
    "gp_sample",
    "latent_total_correlation",
    "lfr_factor",
    "lfr_model",
    "lfr_predictive_moments",
    "markov_gp_factor",
    "markov_gp_sample",
    "mo_gp_factor",
    "mo_svgp_elbo",
    "mo_svgp_factor",
    "normalizing_kalman_factor",
    "sparse_markov_elbo",
    "sparse_markov_factor",
    "svgp_elbo",
    "svgp_factor",
    "warp_to_base",
    "warped_decoder_posterior",
    "warped_lfr_log_prob",
    "warped_predictive_moments",
]
