"""Gaussian process building blocks.

Wave 2 ships:

* Pure kernel *functions* in :mod:`pyrox.gp._src.kernels` — closed-form
  math primitives (RBF, Matern, Periodic, Linear, RationalQuadratic,
  Polynomial, Cosine, White, Constant).
* :class:`Parameterized` kernel classes that wrap those functions with
  constraints, priors, and guide metadata — re-exported from this
  module.
* Abstract protocols (:class:`Kernel`, :class:`Guide`,
  :class:`Integrator`, :class:`Likelihood`) — concrete implementations of
  the non-kernel protocols land in later waves.
* Model-facing entry points — :class:`GPPrior`, :class:`ConditionedGP`,
  :func:`gp_factor`, :func:`gp_sample` — the NumPyro-aware shell on top
  of gaussx linear algebra.

*Scalable matrix construction* and *solver strategies* — numerically
stable matrix assembly, implicit operators, batched matvec, Cholesky /
CG / BBMM / LSMR, etc. — live in ``gaussx``. pyrox model entry points
accept any ``gaussx.AbstractSolverStrategy``; the default is
``gaussx.DenseSolver()``.
"""

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
from pyrox.gp._models import (
    ConditionedGP,
    GPPrior,
    gp_factor,
    gp_sample,
)
from pyrox.gp._protocols import (
    Guide,
    Integrator,
    Kernel,
    Likelihood,
)


__all__ = [
    "RBF",
    "ConditionedGP",
    "Constant",
    "Cosine",
    "GPPrior",
    "Guide",
    "Integrator",
    "Kernel",
    "Likelihood",
    "Linear",
    "Matern",
    "Periodic",
    "Polynomial",
    "RationalQuadratic",
    "White",
    "gp_factor",
    "gp_sample",
]
