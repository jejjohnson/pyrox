r"""A gaussx solver strategy for exact GPs at large ``n`` (P4).

With $\hat K = K + \sigma^2 I$ and $\alpha = \hat K^{-1}y$,

$$
\log p(y) = -\tfrac12 y^\top\alpha - \tfrac12\log|\hat K| - \tfrac n2\log 2\pi .
$$

The solve is conjugate gradients preconditioned by a low-rank
approximation of $K$ (gaussx G13); the log-determinant is stochastic
Lanczos quadrature, either on $\hat K$ itself or on $\hat K$
preconditioned by a Nyström approximation (gaussx G17,
`gaussx.NystromLogdet`); gradients flow through both (the preconditioner only
changes the iteration count, so gaussx stops its gradient). The
preconditioner is built from $K$ with the diagonal as its shift, never from
$\hat K$ itself, so the noise is not counted twice (gaussx#345). The system
is read back with `gaussx.split_diagonal`:

- for ``GPPrior(..., matrix_free=True)`` the system is the sum
  $K_{\text{op}} + (\text{jitter} + \sigma^2) I$, and the scalar noise is
  read from it exactly;
- for a sum $K + \mathrm{diag}(d)$ whose diagonal varies by point (site
  variances, heteroscedastic noise), the preconditioner is built by gaussx's
  per-point shift: a preconditioner $P_B$ of
  $B = D^{-1/2} \hat K D^{-1/2}$, $D = \mathrm{diag}(d)$, with unit shift,
  pulled back as $P^{-1} = D^{-1/2} P_B^{-1} D^{-1/2}$ (gaussx#878);
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


def _scalar_noise(
    operator: lx.AbstractLinearOperator,
) -> tuple[lx.AbstractLinearOperator, Float[Array, ""]] | None:
    """$(K, s)$ for a sum of exactly $K$ and $s I$, by type, or ``None``.

    A Gaussian-noise system keeps the scalar shift it has always had, so its
    preconditioner is bit-identical to the one before per-point shifts
    (pyrox#555). $K$ must hold no diagonal of its own, or the sum is a
    per-point $K' + \\mathrm{diag}(d)$ for `gaussx.split_diagonal`.
    """
    while isinstance(operator, lx.TaggedLinearOperator):
        operator = operator.operator
    if not isinstance(operator, lx.AddLinearOperator):
        return None
    for psd_part, other in (
        (operator.operator1, operator.operator2),
        (operator.operator2, operator.operator1),
    ):
        is_scaled_identity = isinstance(other, lx.MulLinearOperator) and isinstance(
            other.operator, lx.IdentityLinearOperator
        )
        has_diagonal = lx.is_diagonal(psd_part) or (
            gx.split_diagonal(psd_part) is not None
        )
        if is_scaled_identity and not has_diagonal:
            return psd_part, other.scalar
    return None


def _split_system(
    operator: lx.AbstractLinearOperator,
) -> tuple[lx.AbstractLinearOperator, Float[Array, ""] | Float[Array, " N"]] | None:
    r"""$(K, s)$ for $K + s I$, $(K, d)$ for $K + \mathrm{diag}(d)$, else ``None``.

    The split is by type (`gaussx.split_diagonal`), so it is the same under
    ``jit`` with the diagonal traced.
    """
    scalar_split = _scalar_noise(operator)
    if scalar_split is not None:
        # K + s I: the exact scalar, so the Gaussian-noise path is unchanged
        return scalar_split
    split = gx.split_diagonal(operator)
    if split is None:
        return None
    # K + diag(d): a per-point shift, by symmetric scaling in gaussx
    return split.base, split.diagonal


class _NoiseSplitPreconditioner(gx.AbstractPreconditioner):
    """Low-rank preconditioner of ``K + diag(d)``, rebuilt from the system per solve.

    If the system is the sum ``K_op + s I`` (`GPPrior` with
    ``matrix_free=True``), the low-rank part is built from ``K_op`` with the
    exact ``s``. If it is ``K + diag(d)`` with a diagonal that varies by
    point, it is built from ``K`` with the per-point shift ``d``. Otherwise
    it is built from ``A - shift I`` with the user bound ``shift``.
    """

    kind: Literal["nystrom", "rpcholesky"] = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    shift: float | None = eqx.field(static=True)
    key: jax.Array

    def _psd_part_and_shift(
        self, operator: lx.AbstractLinearOperator
    ) -> tuple[
        lx.AbstractLinearOperator,
        float | Float[Array, ""] | Float[Array, " N"],
    ]:
        split = _split_system(operator)
        if split is not None:
            return split
        if self.shift is None:
            raise ValueError(
                "preconditioned_cg_solver: the system is not a sum K + diag(d), "
                "so the diagonal cannot be read from it. Pass shift= (a lower "
                "bound on the noise variance) or use GPPrior(..., matrix_free=True)."
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


class _NoiseSplitNystromLogdet(gx.AbstractLogdetStrategy):
    """`gaussx.NystromLogdet` of ``K + diag(d)``, ``d`` read from the system per call.

    `gaussx.NystromLogdet` needs the shift contained in the system at
    construction, and a learned noise changes it at every call. If the
    system is the sum ``K_op + s I`` (`GPPrior` with ``matrix_free=True``),
    the shift is the exact scalar of that sum; if it is ``K + diag(d)``, the
    per-point ``d``. Otherwise it is the user bound ``shift`` (a lower bound
    on the noise variance is still valid, since the strategy only needs
    ``A - shift I`` to be PSD; a shift above the noise makes it indefinite
    and the estimate NaN, gaussx#878).
    """

    rank: int = eqx.field(static=True)
    shift: float | None = eqx.field(static=True)
    num_probes: int = eqx.field(static=True)
    lanczos_order: int = eqx.field(static=True)
    key: jax.Array

    def logdet(
        self,
        operator: lx.AbstractLinearOperator,
        *,
        key: jax.Array | None = None,
    ) -> Float[Array, ""]:
        split = _split_system(operator)
        shift: float | Float[Array, ""] | Float[Array, " N"]
        if split is not None:
            _, shift = split
        elif self.shift is not None:
            shift = self.shift
        else:
            raise ValueError(
                'preconditioned_cg_solver(logdet="nystrom"): the system is not a '
                "sum K + diag(d), so the diagonal cannot be read from it. Pass "
                "shift= (a lower bound on the noise variance) or use "
                "GPPrior(..., matrix_free=True)."
            )
        strategy = gx.NystromLogdet(
            shift=shift,
            rank=self.rank,
            num_probes=self.num_probes,
            lanczos_order=self.lanczos_order,
        )
        # no key from the caller: the solver's own key, so the estimate is a
        # fixed function of the hyperparameters (common random numbers)
        logdet_key = self.key if key is None else key
        return strategy.logdet(operator, key=logdet_key)


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
    """A gaussx solver for exact GPs: preconditioned CG plus a stochastic logdet.

    Pass it as ``GPPrior(..., solver=...)``; every solve, log-determinant and
    gradient of the marginal likelihood then goes through it. It also solves
    a system $K + \\mathrm{diag}(d)$ whose diagonal varies by point, the
    system of a non-Gaussian or heteroscedastic GP (pyrox#555).

    Args:
        shift: Optional, and used only for a system that is one dense
            matrix. A sum $K + \\mathrm{diag}(d)$ is read with
            `gaussx.split_diagonal`, so ``shift`` is ignored there: for
            ``GPPrior(..., matrix_free=True)`` the system is
            $K_{\\text{op}} + (\\text{jitter} + \\sigma^2) I$ and the
            preconditioner takes the scalar noise from it exactly, and for
            a per-point $d$ it takes $d$ and builds the preconditioner by
            gaussx's symmetric scaling. For a dense system ``shift`` is
            $\\mu$, a lower bound on the noise variance (e.g. the smallest
            noise the prior allows): the preconditioner is built from
            $K + (\\sigma^2 - \\mu) I$, so ``shift`` must not exceed
            $\\sigma^2$. A dense system without ``shift`` raises at solve
            time.
        preconditioner: ``"nystrom"`` (randomized Nyström) or
            ``"rpcholesky"`` (randomly pivoted partial Cholesky), both of
            rank ``rank`` and rebuilt from the operator at each solve.
            Nyström sketches all ``rank`` columns in one batched matvec;
            partial Cholesky takes one matvec per pivot, so prefer
            ``"nystrom"`` for a matrix-free system.
        rank: Preconditioner rank; aim for a few times the effective
            dimension of $K$ at noise level $\\mu$.
        logdet: ``"slq"`` (stochastic Lanczos quadrature on the system) or
            ``"nystrom"`` (`gaussx.NystromLogdet`: the exact log-determinant
            of a rank-``rank`` Nyström preconditioner $P = \\hat K_{\\text{lr}}
            + s I$, plus SLQ on $P^{-1/2} \\hat K P^{-1/2}$, which is close to
            the identity, so its variance is much lower at the same probe
            count). Its shift is read from the system like the
            preconditioner's: the exact noise for ``matrix_free=True``, the
            per-point $d$ of a sum $K + \\mathrm{diag}(d)$, and ``shift``
            for a dense system, which raises at solve time without one. Its
            sketch has the preconditioner's ``rank`` but is drawn separately
            (a shared sketch would save one factorisation and is not
            implemented).
        key: PRNG key for the sketch / the random pivots. With
            ``logdet="nystrom"`` and no key passed to ``logdet``, the
            log-determinant's sketch and probes are split from it as well,
            so the estimate is a fixed function of the hyperparameters.
        rtol: CG relative tolerance.
        atol: CG absolute tolerance.
        max_steps: CG iteration cap.
        num_probes: SLQ probe vectors (for either ``logdet``).
        lanczos_order: SLQ Lanczos steps per probe. The SLQ runs on the
            unpreconditioned system, so it needs more steps as
            $K + \\sigma^2 I$ gets worse conditioned: for a Matérn-3/2 GP at
            $n = 3000$, lengthscale 0.2 and noise 0.04, 30 steps bias the
            log-determinant gradient by about 8 % and 60 steps by under 1 %.

    Returns:
        A `gaussx.CGSolver` with the preconditioner attached for
        ``logdet="slq"``; for ``logdet="nystrom"``, a `gaussx.ComposedSolver`
        of that CG solve and the Nyström log-determinant.

    Raises:
        ValueError: For an unknown preconditioner or ``logdet``, or a
            non-positive shift; at solve time, for a dense system without
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
    if logdet not in ("slq", "nystrom"):
        raise ValueError(f"logdet must be 'slq' or 'nystrom', got {logdet!r}")
    if shift is not None and not shift > 0:
        raise ValueError(f"shift must be positive, got {shift}")
    if preconditioner not in ("nystrom", "rpcholesky"):
        raise ValueError(
            f"preconditioner must be 'nystrom' or 'rpcholesky', got {preconditioner!r}"
        )
    precond = _NoiseSplitPreconditioner(preconditioner, rank, shift, key)
    cg = gx.CGSolver(
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        num_probes=num_probes,
        lanczos_order=lanczos_order,
        preconditioner=precond,
    )
    if logdet == "slq":
        return cg
    nystrom_logdet = _NoiseSplitNystromLogdet(
        rank, shift, num_probes, lanczos_order, key
    )
    return gx.ComposedSolver(solve_strategy=cg, logdet_strategy=nystrom_logdet)
