r"""A gaussx solver strategy for exact GPs at large ``n`` (P4).

With $\hat K = K + \sigma^2 I$ and $\alpha = \hat K^{-1}y$,

$$
\log p(y) = -\tfrac12 y^\top\alpha - \tfrac12\log|\hat K| - \tfrac n2\log 2\pi .
$$

The solve is conjugate gradients preconditioned by a low-rank
approximation of $K$ (gaussx G13); the log-determinant is stochastic
Lanczos quadrature; gradients flow through both (the preconditioner only
changes the iteration count, so gaussx stops its gradient). The
preconditioner is built from $\hat K - \mu I$ with shift $\mu$, never from
$\hat K$ itself, so the noise is not counted twice (gaussx#345).
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import gaussx as gx
import jax
import lineax as lx


class _ShiftedNystrom(gx.AbstractPreconditioner):
    """Nyström preconditioner of ``A - shift I`` with ``shift``, per solve."""

    rank: int = eqx.field(static=True)
    shift: float = eqx.field(static=True)
    key: jax.Array

    def as_operator(self, operator=None):
        if operator is None:
            return None
        identity = lx.IdentityLinearOperator(operator.in_structure())
        psd_part = lx.TaggedLinearOperator(
            operator - self.shift * identity, lx.positive_semidefinite_tag
        )
        return gx.NystromPreconditioner.from_operator(
            psd_part, self.rank, shift=self.shift, key=self.key
        ).as_operator()


def preconditioned_cg_solver(
    *,
    shift: float,
    preconditioner: Literal["nystrom", "rpcholesky"] = "nystrom",
    rank: int = 200,
    logdet: Literal["slq", "nystrom"] = "slq",
    key: jax.Array,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_steps: int = 1000,
    num_probes: int = 20,
    lanczos_order: int = 30,
) -> gx.AbstractSolverStrategy:
    """A gaussx solver strategy for exact GPs: preconditioned CG plus SLQ.

    Pass it as ``GPPrior(..., solver=...)``; every solve, log-determinant and
    gradient of the marginal likelihood then goes through it.

    Args:
        shift: $\\mu$, a lower bound on the noise variance in the system
            $K + \\sigma^2 I$ (e.g. the smallest noise the prior allows).
            The preconditioner is built from $K + (\\sigma^2 - \\mu) I$ and
            targets the system, so ``shift`` must not exceed $\\sigma^2$.
        preconditioner: ``"nystrom"`` (randomized Nyström) or
            ``"rpcholesky"`` (randomly pivoted partial Cholesky), both of
            rank ``rank`` and rebuilt from the operator at each solve.
        rank: Preconditioner rank; aim for a few times the effective
            dimension of $K$ at noise level $\\mu$.
        logdet: ``"slq"`` (stochastic Lanczos quadrature). ``"nystrom"``
            waits for gaussx's tier-2 ``NystromLogdet``.
        key: PRNG key for the sketch / the random pivots.
        rtol: CG relative tolerance.
        atol: CG absolute tolerance.
        max_steps: CG iteration cap.
        num_probes: SLQ probe vectors.
        lanczos_order: SLQ Lanczos steps per probe.

    Returns:
        A `gaussx.CGSolver` with the preconditioner attached.

    Raises:
        ValueError: For an unknown preconditioner, a non-positive shift or
            ``logdet="nystrom"``.

    Examples:
        ```python
        import jax.random as jr
        import pyrox_gp as px

        solver = px.preconditioned_cg_solver(shift=1e-2, rank=500, key=jr.key(0))
        prior = px.GPPrior(px.Matern(nu=1.5), X, solver=solver)  # X: (1e5, 2)
        # inside a NumPyro model: px.gp_factor("y", prior, y, noise_var)
        ```
    """
    if logdet == "nystrom":
        raise ValueError(
            'logdet="nystrom" needs gaussx\'s NystromLogdet, which is not '
            'released yet; use logdet="slq"'
        )
    if logdet != "slq":
        raise ValueError(f"logdet must be 'slq', got {logdet!r}")
    if not shift > 0:
        raise ValueError(f"shift must be positive, got {shift}")
    if preconditioner == "nystrom":
        precond: gx.AbstractPreconditioner = _ShiftedNystrom(rank, shift, key)
    elif preconditioner == "rpcholesky":
        precond = gx.PartialCholeskyPreconditioner(
            rank=rank, shift=shift, pivoting="random", key=key
        )
    else:
        raise ValueError(
            f"preconditioner must be 'nystrom' or 'rpcholesky', got {preconditioner!r}"
        )
    return gx.CGSolver(
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        num_probes=num_probes,
        lanczos_order=lanczos_order,
        preconditioner=precond,
    )
