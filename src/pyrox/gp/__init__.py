"""Gaussian process building blocks.

* Pure kernel *functions* in :mod:`pyrox.gp._src.kernels` — closed-form
  math primitives (RBF, Matern, Periodic, Linear, RationalQuadratic,
  Polynomial, Cosine, White, Constant).
* :class:`Parameterized` kernel classes that wrap those functions with
  constraints, priors, and guide metadata — re-exported from this
  module.
* Multi-output GP structures — :class:`LMCKernel`, :class:`ICMKernel`,
  :class:`OILMMKernel`, and shared inducing-point helpers for explicit
  cross-output structure without monolithic model classes.
* Abstract protocols (:class:`Kernel`, :class:`Guide`,
  :class:`Integrator`, :class:`Likelihood`) plus five concrete sparse
  variational guides — :class:`FullRankGuide`, :class:`MeanFieldGuide`,
  :class:`WhitenedGuide`, :class:`NaturalGuide`, :class:`DeltaGuide`.
  Natural-parameter conversion and damped-update primitives live in
  ``gaussx`` (:func:`gaussx.mean_cov_to_natural`,
  :func:`gaussx.natural_to_mean_cov`,
  :func:`gaussx.damped_natural_update`) — :class:`NaturalGuide`
  delegates its math there, and so will the future natural-gradient /
  CVI inference paths.
* Model-facing entry points — :class:`GPPrior`, :class:`ConditionedGP`,
  :class:`SparseGPPrior`, :class:`PathwiseSampler`,
  :class:`DecoupledPathwiseSampler`, :func:`gp_factor`,
  :func:`gp_sample` — the NumPyro-aware shell on top of gaussx linear
  algebra.

*Scalable matrix construction* and *solver strategies* — numerically
stable matrix assembly, implicit operators, batched matvec, Cholesky /
CG / BBMM / LSMR, etc. — live in ``gaussx``. pyrox model entry points
accept any ``gaussx.AbstractSolverStrategy``; the default is
``gaussx.DenseSolver()``.
"""

from pyrox.gp._guides import (
    DeltaGuide,
    FullRankGuide,
    MeanFieldGuide,
    NaturalGuide,
    WhitenedGuide,
)
from pyrox.gp._inducing import (
    DecoupledInducingFeatures,
    FourierInducingFeatures,
    InducingFeatures,
    LaplacianInducingFeatures,
    SphericalHarmonicInducingFeatures,
    funk_hecke_coefficients,
)
from pyrox.gp._inference import (
    ConjugateVI,
    svgp_elbo,
    svgp_factor,
)
from pyrox.gp._kernels import (
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
from pyrox.gp._likelihoods import (
    DistLikelihood,
    GaussianLikelihood,
)
from pyrox.gp._models import (
    ConditionedGP,
    GPPrior,
    gp_factor,
    gp_sample,
)
from pyrox.gp._multi_output import (
    ICMKernel,
    LMCKernel,
    MultiOutputInducingVariables,
    OILMMKernel,
    SharedInducingPoints,
)
from pyrox.gp._pathwise import (
    DecoupledPathwiseSampler,
    PathwiseFunction,
    PathwiseSampler,
)
from pyrox.gp._protocols import (
    Guide,
    Integrator,
    Kernel,
    Likelihood,
    SDEKernel,
)
from pyrox.gp._sde_kernels import (
    ConstantSDE,
    CosineSDE,
    MaternSDE,
    PeriodicSDE,
    ProductSDE,
    QuasiPeriodicSDE,
    SumSDE,
)
from pyrox.gp._sparse import SparseGPPrior


__all__ = [
    "RBF",
    "ConditionedGP",
    "ConjugateVI",
    "Constant",
    "ConstantSDE",
    "Cosine",
    "CosineSDE",
    "DecoupledInducingFeatures",
    "DecoupledPathwiseSampler",
    "DeltaGuide",
    "DistLikelihood",
    "FourierInducingFeatures",
    "FullRankGuide",
    "GPPrior",
    "GaussianLikelihood",
    "Guide",
    "ICMKernel",
    "InducingFeatures",
    "Integrator",
    "Kernel",
    "LMCKernel",
    "LaplacianInducingFeatures",
    "Likelihood",
    "Linear",
    "Matern",
    "MaternSDE",
    "MeanFieldGuide",
    "MultiOutputInducingVariables",
    "NaturalGuide",
    "OILMMKernel",
    "PathwiseFunction",
    "PathwiseSampler",
    "Periodic",
    "PeriodicSDE",
    "Polynomial",
    "ProductSDE",
    "QuasiPeriodicSDE",
    "RationalQuadratic",
    "SDEKernel",
    "SharedInducingPoints",
    "SparseGPPrior",
    "SphericalHarmonicInducingFeatures",
    "SumSDE",
    "White",
    "WhitenedGuide",
    "funk_hecke_coefficients",
    "gp_factor",
    "gp_sample",
    "svgp_elbo",
    "svgp_factor",
]
