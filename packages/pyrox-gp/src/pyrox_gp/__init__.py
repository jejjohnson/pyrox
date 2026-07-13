"""Gaussian process building blocks.

* Pure kernel *functions* in `pyrox_gp._src.kernels` — closed-form
  math primitives (RBF, Matern, Periodic, Linear, RationalQuadratic,
  Polynomial, Cosine, White, Constant).
* `Parameterized` kernel classes that wrap those functions with
  constraints, priors, and guide metadata — re-exported from this
  module.
* Multi-output GP structures — `LMCKernel`, `ICMKernel`,
  `OILMMKernel`, and shared inducing-point helpers for explicit
  cross-output structure without monolithic model classes.
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


__version__ = "0.1.0"  # x-release-please-version

__all__ = [
    "RBF",
    "BernoulliLikelihood",
    "ConditionedGP",
    "ConditionedMarkovGP",
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
    "Kernel",
    "LMCKernel",
    "LaplaceInference",
    "LaplaceMarkovInference",
    "LaplacianInducingFeatures",
    "Likelihood",
    "Linear",
    "MarkovGPPrior",
    "Matern",
    "MaternSDE",
    "MeanFieldGuide",
    "MultiOutputInducingVariables",
    "NaturalGuide",
    "NonGaussConditionedGP",
    "NonGaussConditionedMarkovGP",
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
    "White",
    "WhitenedGuide",
    "__version__",
    "funk_hecke_coefficients",
    "gp_factor",
    "gp_sample",
    "markov_gp_factor",
    "markov_gp_sample",
    "sparse_markov_elbo",
    "sparse_markov_factor",
    "svgp_elbo",
    "svgp_factor",
]
