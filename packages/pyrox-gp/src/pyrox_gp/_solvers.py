r"""The linear-algebra choices of a GP prior, and two ready-made sets of them.

A prior carries three separate choices (pyrox#550): the ``solver`` that
applies $A^{-1}$, the ``preconditioner`` that speeds it up, and the
``logdet`` that estimates $\log \det A$. `compose_solver_fields` turns
them into the one `gaussx.AbstractSolverStrategy` every entry point uses,
through `gaussx.compose_solver`.

`matfree_solver` and `bbmm_solver` (pyrox#557) are the two families people
know, as one call each, on the system $A = K_{XX} + \mathrm{diag}(d)$:

$$
\text{matfree: } x = A^{-1} b \text{ by preconditioned CG}, \qquad
\log \det A \approx \log \det P
  + \frac{1}{N_\text{probe}} \sum_i \lVert u^{(i)} \rVert_2^2
    e_1^\top \log(T^{(i)}) e_1 ,
$$

$$
\text{BBMM: both from one block CG over }
[\, b \mid z^{(1)} \ldots z^{(N_\text{probe})} \,],
\quad z^{(i)} = P^{1/2} u^{(i)} ,
$$

with $T^{(i)}$ the Lanczos tridiagonal of probe $i$ and
$u^{(i)} \sim \mathcal{N}(0, I)$.
"""

from __future__ import annotations

import warnings
from typing import Literal

import gaussx as gx
import jax
import jax.random as jr

from pyrox_gp._preconditioned import (
    _NoiseSplitNystromLogdet,
    _NoiseSplitPreconditioner,
)


def _field_name(value: object) -> str:
    return "None" if value is None else type(value).__name__


def compose_solver_fields(
    owner: str,
    solver: gx.AbstractSolveStrategy | None,
    preconditioner: gx.AbstractPreconditioner | None,
    logdet: gx.AbstractLogdetStrategy | None,
) -> gx.AbstractSolverStrategy:
    """One strategy from a prior's ``solver``, ``preconditioner`` and ``logdet``.

    All three ``None`` is `gaussx.DenseSolver`, and a ``solver`` alone is
    returned unchanged, so a prior that sets none of the new fields keeps
    its strategy exactly. The checks are on types only, so this runs under
    ``jit``.

    Args:
        owner: The class whose fields these are, named in the error.
        solver: The solve strategy; ``None`` is `gaussx.DenseSolver`.
        preconditioner: The preconditioner of the solve and, for a
            `gaussx.BBMMSolver`, of its log-determinant.
        logdet: A separate log-determinant strategy; ``None`` keeps the
            solver's own.

    Returns:
        The composed `gaussx.AbstractSolverStrategy`.

    Raises:
        ValueError: For a combination `gaussx.compose_solver` rejects (a
            preconditioner with an exact solver, two preconditioners, ...),
            naming the three fields.
    """
    try:
        return gx.compose_solver(solver, preconditioner=preconditioner, logdet=logdet)
    except ValueError as error:
        raise ValueError(
            f"{owner}(solver={_field_name(solver)}, "
            f"preconditioner={_field_name(preconditioner)}, "
            f"logdet={_field_name(logdet)}) cannot be combined: {error}"
        ) from error


_FITC_MESSAGE = (
    'preconditioner="fitc" needs inducing inputs and a kernel, which a system '
    "does not carry, so no preset can build it. Pass the preconditioner object "
    "itself to the prior instead: GPPrior(..., solver=gaussx.CGSolver(), "
    "preconditioner=...), with a FITC preconditioner once gaussx#824 lands."
)


def _check_preconditioner(preconditioner: str) -> None:
    if preconditioner == "fitc":
        raise ValueError(_FITC_MESSAGE)
    if preconditioner not in ("nystrom", "rpcholesky"):
        raise ValueError(
            "preconditioner must be 'nystrom', 'rpcholesky' or 'fitc', "
            f"got {preconditioner!r}"
        )


def _matfree_solver(
    *,
    rank: int,
    preconditioner: Literal["nystrom", "rpcholesky", "fitc"],
    logdet: Literal["nystrom", "slq"],
    num_probes: int,
    lanczos_order: int,
    rtol: float | None,
    atol: float | None,
    max_steps: int,
    throw: bool,
    shift: float | None,
    key: jax.Array,
) -> gx.AbstractSolverStrategy:
    if logdet not in ("slq", "nystrom"):
        raise ValueError(f"logdet must be 'slq' or 'nystrom', got {logdet!r}")
    _check_preconditioner(preconditioner)
    assert preconditioner != "fitc"
    built = _NoiseSplitPreconditioner(preconditioner, rank, shift, key)
    cg = gx.CGSolver(
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        num_probes=num_probes,
        lanczos_order=lanczos_order,
        throw=throw,
    )
    if logdet == "slq":
        # CG's own SLQ on the system, unpreconditioned
        return gx.compose_solver(cg, preconditioner=built)
    # the same key as the preconditioner's sketch, so the estimate is a
    # fixed function of the hyperparameters (common random numbers)
    nystrom_logdet = _NoiseSplitNystromLogdet(
        rank, shift, num_probes, lanczos_order, key
    )
    return gx.compose_solver(cg, preconditioner=built, logdet=nystrom_logdet)


def matfree_solver(
    *,
    rank: int = 200,
    preconditioner: Literal["nystrom", "rpcholesky", "fitc"] = "nystrom",
    logdet: Literal["nystrom", "slq"] = "nystrom",
    num_probes: int = 50,
    lanczos_order: int = 30,
    rtol: float = 1e-4,
    atol: float | None = None,
    max_steps: int = 1000,
    throw: bool = True,
    key: jax.Array,
) -> gx.AbstractSolverStrategy:
    r"""CG and a matfree-backed log-determinant, preconditioned (pyrox#557).

    The solve is lineax conjugate gradients, preconditioned by a rank-``rank``
    approximation of the kernel part $K$ of the system
    $A = K + \mathrm{diag}(d)$, rebuilt from the system at every solve so it
    follows the hyperparameters and, in a non-Gaussian fit, the site
    variances $d$. The log-determinant is stochastic Lanczos quadrature
    through matfree. The library matfree 0.6.2 has no conjugate gradients,
    so "matfree" here means lineax CG plus gaussx's matfree-backed
    log-determinant.

    The preconditioner reads the system with `gaussx.split_diagonal`: the
    scalar noise of ``GPPrior(..., matrix_free=True)`` exactly, or a
    per-point $d$ by gaussx's symmetric scaling. A system that is one dense
    matrix does not split, and raises at solve time. Building it costs
    $O(N R^2)$ plus $R$ kernel matvecs, at every solve; reusing a sketch
    across Newton steps is not implemented. Timing is unmeasured.

    Args:
        rank: Rank $R$ of the preconditioner (and of the Nyström
            log-determinant). 200 is a starting point, measured at
            $N = 400$ only.
        preconditioner: ``"nystrom"`` (randomized Nyström, one batched
            matvec per build) or ``"rpcholesky"`` (randomly pivoted partial
            Cholesky, one matvec per pivot). ``"fitc"`` needs inducing
            inputs and a kernel, which a system does not carry, so it
            raises; pass a preconditioner object to the prior instead.
        logdet: ``"nystrom"`` (the default, `gaussx.NystromLogdet` with the
            same shift: the exact $\log \det P$ plus SLQ on the
            near-identity $P^{-1/2} A P^{-1/2}$, the more accurate at equal
            rank) or ``"slq"`` (CG's own SLQ on the system, with probes from
            its fixed seed).
        num_probes: Probe vectors $N_\text{probe}$ of the SLQ.
        lanczos_order: Lanczos steps per probe.
        rtol: CG relative tolerance.
        atol: CG absolute tolerance; ``None`` is gaussx's default.
        max_steps: CG iteration cap.
        throw: Raise when CG does not converge within ``max_steps``.
        key: PRNG key of the preconditioner's sketch and of the Nyström
            log-determinant's sketch and probes, so the strategy is a fixed,
            reproducible function of the hyperparameters. Inside NUTS a
            fixed key means the sampler targets a fixed approximation of the
            posterior (pyrox#550).

    Returns:
        One ordinary `gaussx.AbstractSolverStrategy` (a `gaussx.CGSolver`
        for ``logdet="slq"``, a `gaussx.ComposedSolver` for
        ``logdet="nystrom"``), so `gaussx.compose_solver` and
        `gaussx.KeyedSolver` work on it.

    Raises:
        ValueError: For an unknown ``preconditioner`` or ``logdet``, and for
            ``preconditioner="fitc"``.

    Examples:
        ```python
        import jax.random as jr
        import pyrox_gp as px

        # X: (100_000, 2); K is never formed
        solver = px.matfree_solver(rank=300, key=jr.key(0))
        prior = px.GPPrior(px.Matern(nu=1.5), X, solver=solver, matrix_free=True)
        log_marginal = prior.log_marginal(y, noise_var)
        ```

    References:
        Frangella, Z., Tropp, J. A. & Udell, M. (2023). Randomized Nyström
        preconditioning. SIAM J. Matrix Anal. Appl. 44(2), 718–752.

        Ubaru, S., Chen, J. & Saad, Y. (2017). Fast estimation of tr(f(A)) via
        stochastic Lanczos quadrature. SIAM J. Matrix Anal. Appl. 38(4),
        1075–1099.
    """
    return _matfree_solver(
        rank=rank,
        preconditioner=preconditioner,
        logdet=logdet,
        num_probes=num_probes,
        lanczos_order=lanczos_order,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        throw=throw,
        shift=None,
        key=key,
    )


def bbmm_solver(
    *,
    rank: int = 200,
    preconditioner: Literal["nystrom", "rpcholesky", "fitc"] = "nystrom",
    num_probes: int = 50,
    lanczos_order: int = 100,
    rtol: float = 1e-4,
    max_steps: int = 1000,
    throw: bool = True,
    key: jax.Array,
) -> gx.AbstractSolverStrategy:
    r"""BBMM with a low-rank preconditioner (pyrox#557).

    `gaussx.BBMMSolver` with the preconditioner of `matfree_solver` (Nyström
    by default, so both presets build the same one): the solve is CG with
    $P^{-1}$, and the log-determinant comes from one modified batched CG
    pass over the right-hand side and probes $z^{(i)} \sim \mathcal{N}(0, P)$
    drawn from $P$ itself, so only the near-identity remainder
    $\log \det(P^{-1/2} A P^{-1/2})$ is estimated. The probes run to
    convergence, not to ``lanczos_order`` steps, so the pass can cost more
    matvecs than a separate solve and SLQ; timing is unmeasured. The
    gradient does not get the preconditioner's variance reduction
    (gaussx#826).

    Args:
        rank: Rank $R$ of the preconditioner; 200 is a starting point,
            measured at $N = 400$ only.
        preconditioner: ``"nystrom"`` or ``"rpcholesky"``, as in
            `matfree_solver`; ``"fitc"`` raises.
        num_probes: Probe vectors $N_\text{probe}$.
        lanczos_order: Lanczos order of the quadrature.
        rtol: CG relative tolerance.
        max_steps: CG iteration cap.
        throw: Raise when CG does not converge within ``max_steps``.
        key: PRNG key, split into the preconditioner's sketch and the probes
            (bound with `gaussx.KeyedSolver`), so the estimate is a fixed,
            reproducible function of the hyperparameters. Inside NUTS a fixed
            key means the sampler targets a fixed approximation of the
            posterior (pyrox#550).

    Returns:
        A `gaussx.KeyedSolver` around the preconditioned `gaussx.BBMMSolver`.

    Raises:
        ValueError: For an unknown ``preconditioner``, and for
            ``preconditioner="fitc"``.

    Examples:
        ```python
        import jax.random as jr
        import pyrox_gp as px

        solver = px.bbmm_solver(rank=300, key=jr.key(0))
        prior = px.GPPrior(px.Matern(nu=1.5), X, solver=solver, matrix_free=True)
        log_marginal = prior.log_marginal(y, noise_var)
        ```

    References:
        Gardner, J. R., Pleiss, G., Weinberger, K. Q., Bindel, D. & Wilson,
        A. G. (2018). GPyTorch: blackbox matrix-matrix Gaussian process
        inference with GPU acceleration. NeurIPS, arXiv:1809.11165.

        Wenger, J., Pleiss, G., Hennig, P., Cunningham, J. P. & Gardner, J. R.
        (2022). Preconditioning for scalable Gaussian process hyperparameter
        optimization. ICML, PMLR 162, 23751–23780.
    """
    _check_preconditioner(preconditioner)
    assert preconditioner != "fitc"
    sketch_key, probe_key = jr.split(key)
    built = _NoiseSplitPreconditioner(preconditioner, rank, None, sketch_key)
    bbmm = gx.BBMMSolver(
        rtol=rtol,
        max_steps=max_steps,
        lanczos_order=lanczos_order,
        num_probes=num_probes,
        throw=throw,
    )
    return gx.KeyedSolver(gx.compose_solver(bbmm, preconditioner=built), probe_key)


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
    """Deprecated alias of `matfree_solver`, with its old arguments and defaults.

    Kept for one release (pyrox#557). It builds exactly the strategy it
    built before, which is `matfree_solver` with these defaults, plus
    ``shift``: a lower bound on the noise variance, used only for a system
    that is one dense matrix. On a sum $K + \\mathrm{diag}(d)$ the diagonal
    is read from the system, and a ``shift`` passed anyway is ignored with
    a warning.

    Args:
        shift: Lower bound $\\mu$ on the noise variance, for a dense system.
        preconditioner: As in `matfree_solver`.
        rank: As in `matfree_solver`.
        logdet: As in `matfree_solver`, with ``"slq"`` the old default.
        key: As in `matfree_solver`.
        rtol: CG relative tolerance.
        atol: CG absolute tolerance.
        max_steps: CG iteration cap.
        num_probes: SLQ probe vectors.
        lanczos_order: SLQ Lanczos steps per probe.

    Returns:
        The `matfree_solver` strategy.

    Raises:
        ValueError: For a non-positive ``shift``, and as `matfree_solver`.

    Examples:
        ```python
        import jax.random as jr
        import pyrox_gp as px

        # before: px.preconditioned_cg_solver(rank=500, key=jr.key(0))
        solver = px.matfree_solver(rank=500, logdet="slq", key=jr.key(0))
        ```
    """
    warnings.warn(
        "preconditioned_cg_solver is deprecated and will be removed in the "
        "next release; use pyrox_gp.matfree_solver (or pyrox_gp.bbmm_solver) "
        "instead (pyrox#557).",
        DeprecationWarning,
        stacklevel=2,
    )
    if shift is not None and not shift > 0:
        raise ValueError(f"shift must be positive, got {shift}")
    return _matfree_solver(
        rank=rank,
        preconditioner=preconditioner,
        logdet=logdet,
        num_probes=num_probes,
        lanczos_order=lanczos_order,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        throw=True,
        shift=shift,
        key=key,
    )
