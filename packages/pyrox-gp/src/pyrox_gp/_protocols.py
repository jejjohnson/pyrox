"""Layer 1 — abstract protocol classes for GP components.

Three orthogonal pyrox-local protocols that compose into a GP model.
`Kernel` has concrete implementations in `pyrox_gp._kernels`
(`pyrox_gp.RBF`, etc.). `SDEKernel` is the state-space
face of stationary 1-D kernels and now lives in `gaussx._ssm`
(``gaussx.MaternSDE``, ``gaussx.SumSDE``, ``gaussx.PeriodicSDE``, ...);
this module re-exports it so the ``pyrox_gp.SDEKernel`` alias still
resolves. Concrete subclasses feed the Kalman-based
`pyrox_gp.MarkovGPPrior`.

Solver strategies intentionally live in `gaussx`, not here. Use
``gaussx.AbstractSolverStrategy`` (combined solve + logdet),
``AbstractSolveStrategy``, or ``AbstractLogdetStrategy`` — with concretes
like ``gaussx.DenseSolver``, ``gaussx.CGSolver``, ``gaussx.BBMMSolver``,
and ``gaussx.ComposedSolver``. The pyrox model entry points
(``GPPrior``, ``gp_factor``, ``gp_sample``) accept any solver strategy.

Gaussian-expectation integrators live in `gaussx` too — use
``gaussx.AbstractIntegrator`` (and its concretes
``GaussHermiteIntegrator``, ``MonteCarloIntegrator``,
``UnscentedIntegrator``, ``TaylorIntegrator``) wherever pyrox needs to
take expectations against a ``GaussianState``.

* `Kernel` — covariance structure, ``(X1, X2) -> Gram``; an alias of
  ``kernellib.AbstractKernel``.
* `SDEKernel` — re-exported from `gaussx._ssm`.
* `Guide` — variational posterior structure; re-exported from
  `pyrox.likelihoods`.
* `Likelihood` — observation model; re-exported from `pyrox.likelihoods`,
  with ``likelihood_accepts_inputs``.
"""

from __future__ import annotations

from gaussx import SDEKernel as SDEKernel
from kernellib import AbstractKernel
from pyrox.likelihoods import (
    Guide as Guide,
    Likelihood as Likelihood,
    likelihood_accepts_inputs as likelihood_accepts_inputs,
)


# ``Kernel`` is kernellib's ``AbstractKernel``: same abstract ``__call__``,
# same ``gram`` / ``diag`` defaults, so every subclass here is unaffected.
# kernellib adds ``pairwise`` / ``is_pointwise`` (Gram-only by default) and
# ``+`` / ``*`` composition, which pyrox kernels inherit.
Kernel = AbstractKernel


# ``SDEKernel`` lives in `gaussx._ssm` since gaussx 0.0.11; we
# re-export it at the top of this module so ``pyrox_gp.SDEKernel`` keeps
# resolving for downstream callers.
