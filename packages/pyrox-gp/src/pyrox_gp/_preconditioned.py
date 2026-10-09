r"""A gaussx solver strategy for exact GPs at large ``n`` (P4).

With $\hat K = K + \sigma^2 I$ and $\alpha = \hat K^{-1}y$,

$$
\log p(y) = -\tfrac12 y^\top\alpha - \tfrac12\log|\hat K| - \tfrac n2\log 2\pi .
$$

The solve is conjugate gradients preconditioned by a low-rank
approximation of $K$ (gaussx G13); the log-determinant is stochastic
Lanczos quadrature; gradients flow through both (the preconditioner only
changes the iteration count, so gaussx stops its gradient). The
preconditioner is built from $K$ with the noise as its shift, never from
$\hat K$ itself, so the noise is not counted twice (gaussx#345):

- for ``GPPrior(..., matrix_free=True)`` the system is the sum
  $K_{\text{op}} + (\text{jitter} + \sigma^2) I$, and the noise is read
  from it exactly;
- for any other operator (one dense $\hat K$) the preconditioner is built
  from $\hat K - \mu I$ with a user lower bound $\mu \le \sigma^2$.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import gaussx as gx
import jax
import lineax as lx
from jaxtyping import Array, Float

from pyrox_gp._kernel_operator import split_noise


class _NoiseSplitPreconditioner(gx.AbstractPreconditioner):
    """Low-rank preconditioner of ``K + s I``, rebuilt from the system per solve.

    If the system is the sum ``K_op + s I`` (`GPPrior` with
    ``matrix_free=True``), the low-rank part is built from ``K_op`` with the
    exact ``s``. Otherwise it is built from ``A - shift I`` with the user
    bound ``shift``.
    """

    kind: Literal["nystrom", "rpcholesky"] = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    shift: float | None = eqx.field(static=True)
    key: jax.Array

    def _psd_part_and_shift(
        self, operator: lx.AbstractLinearOperator
    ) -> tuple[lx.AbstractLinearOperator, float | Float[Array, ""]]:
        split = split_noise(operator)
        if split is not None:
            return split
        if self.shift is None:
            raise ValueError(
                "preconditioned_cg_solver: the system is not a sum K + s I, so "
                "the noise cannot be read from it. Pass shift= (a lower bound "
                "on the noise variance) or use GPPrior(..., matrix_free=True)."
            )
        identity = lx.IdentityLinearOperator(operator.in_structure())
        return operator - self.shift * identity, self.shift

    def as_operator(self, operator=None):
        if operator is None:
            return None
        psd_part, shift = self._psd_part_and_shift(operator)
        psd_part = lx.TaggedLinearOperator(psd_part, lx.positive_semidefinite_tag)
        if self.kind == "nystrom":
            precond: gx.AbstractPreconditioner = gx.NystromPreconditioner.from_operator(
                psd_part, self.rank, shift=shift, key=self.key
            )
        else:
            precond = gx.PartialCholeskyPreconditioner.from_operator(
                psd_part, self.rank, shift=shift, pivoting="random", key=self.key
            )
        return precond.as_operator()


def preconditioned_cg_solver(
    *,
    shift: float | None = None,
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
        shift: Optional. For ``GPPrior(..., matrix_free=True)`` the system
            is the sum $K_{\\text{op}} + (\\text{jitter} + \\sigma^2) I$ and
            the preconditioner takes the noise from it exactly, so ``shift``
            is ignored. For a dense system it is $\\mu$, a lower bound on
            the noise variance (e.g. the smallest noise the prior allows):
            the preconditioner is built from $K + (\\sigma^2 - \\mu) I$, so
            ``shift`` must not exceed $\\sigma^2$. A dense system without
            ``shift`` raises at solve time.
        preconditioner: ``"nystrom"`` (randomized Nyström) or
            ``"rpcholesky"`` (randomly pivoted partial Cholesky), both of
            rank ``rank`` and rebuilt from the operator at each solve.
            Nyström sketches all ``rank`` columns in one batched matvec;
            partial Cholesky takes one matvec per pivot, so prefer
            ``"nystrom"`` for a matrix-free system.
        rank: Preconditioner rank; aim for a few times the effective
            dimension of $K$ at noise level $\\mu$.
        logdet: ``"slq"`` (stochastic Lanczos quadrature). ``"nystrom"``
            waits for gaussx's tier-2 ``NystromLogdet``.
        key: PRNG key for the sketch / the random pivots.
        rtol: CG relative tolerance.
        atol: CG absolute tolerance.
        max_steps: CG iteration cap.
        num_probes: SLQ probe vectors.
        lanczos_order: SLQ Lanczos steps per probe. The SLQ runs on the
            unpreconditioned system, so it needs more steps as
            $K + \\sigma^2 I$ gets worse conditioned: for a Matérn-3/2 GP at
            $n = 3000$, lengthscale 0.2 and noise 0.04, 30 steps bias the
            log-determinant gradient by about 8 % and 60 steps by under 1 %.

    Returns:
        A `gaussx.CGSolver` with the preconditioner attached.

    Raises:
        ValueError: For an unknown preconditioner, a non-positive shift or
            ``logdet="nystrom"``; at solve time, for a dense system without
            ``shift``.

    Examples:
        ```python
        import jax.random as jr
        import pyrox_gp as px

        solver = px.preconditioned_cg_solver(rank=500, key=jr.key(0))
        # X: (1e5, 2); K is never formed, and the noise comes from the system
        prior = px.GPPrior(px.Matern(nu=1.5), X, solver=solver, matrix_free=True)
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
    if shift is not None and not shift > 0:
        raise ValueError(f"shift must be positive, got {shift}")
    if preconditioner not in ("nystrom", "rpcholesky"):
        raise ValueError(
            f"preconditioner must be 'nystrom' or 'rpcholesky', got {preconditioner!r}"
        )
    precond = _NoiseSplitPreconditioner(preconditioner, rank, shift, key)
    return gx.CGSolver(
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        num_probes=num_probes,
        lanczos_order=lanczos_order,
        preconditioner=precond,
    )
