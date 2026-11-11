"""Layer 1 — abstract protocol classes for GP components.

Three orthogonal pyrox-local protocols that compose into a GP model.
`Kernel` has concrete implementations in `pyrox_gp._kernels`
(`pyrox_gp.RBF`, etc.). The state-space protocol ``SDEKernel`` and the
Kalman-based Markov GP moved to pyrox-ssm (``pyrox_ssm.SDEKernel``,
``pyrox_ssm.MarkovGPPrior``).

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
* `Guide` — variational posterior structure; re-exported from
  `pyrox.likelihoods`.
* `Likelihood` — observation model; re-exported from `pyrox.likelihoods`,
  with ``likelihood_accepts_inputs``.
"""

from __future__ import annotations

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
