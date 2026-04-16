"""Gaussian process building blocks.

Wave 2 ships:

* Pure kernel *functions* in :mod:`pyrox.gp._src.kernels` — closed-form
  math primitives (RBF, Matern, Periodic, Linear, RationalQuadratic,
  Polynomial, Cosine, White, Constant).
* :class:`Parameterized` kernel classes that wrap those functions with
  constraints, priors, and guide metadata — re-exported from this
  module.
* Abstract protocols (:class:`Kernel`, :class:`Solver`, :class:`Guide`,
  :class:`Integrator`, :class:`Likelihood`) — concrete implementations of
  the non-kernel protocols land in later waves.

*Scalable matrix construction* — numerically stable matrix assembly,
implicit operators, batched matvec, Cholesky-with-jitter, solver
strategies — lives in ``gaussx``. Concrete model-facing entry points
(``GPPrior``, ``ConditionedGP``, ``gp_factor``, ``gp_sample``) land in
Wave 2 Epic 2.B (#22).
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
from pyrox.gp._protocols import (
    Guide,
    Integrator,
    Kernel,
    Likelihood,
    Solver,
)


__all__ = [
    "RBF",
    "Constant",
    "Cosine",
    "Guide",
    "Integrator",
    "Kernel",
    "Likelihood",
    "Linear",
    "Matern",
    "Periodic",
    "Polynomial",
    "RationalQuadratic",
    "Solver",
    "White",
]
