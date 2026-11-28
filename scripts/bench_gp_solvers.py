#!/usr/bin/env python3
r"""Benchmark the solver, preconditioner and log-determinant choices of an exact GP.

The measurements behind ``docs/notebooks/gp_solver_choices.ipynb`` (pyrox#562).
Each row of the output CSV is one (likelihood, kernel, $N$, configuration):
a Matérn-3/2 GP on $N$ inputs drawn uniformly on the unit square, with a
Gaussian likelihood (the collapsed evidence ``GPPrior.log_marginal``) or a
Bernoulli one (the Laplace evidence of ``LaplaceInference``), evaluated at a
smooth ($\ell = 0.3$) or a rough ($\ell = 0.03$) lengthscale.

For each row the script measures, on the CPU it runs on:

- the wall-clock time of one evidence (for Bernoulli, one full Laplace fit)
  and of one evidence gradient, after compilation (ahead-of-time compiled
  executables, ``jax.block_until_ready``), as the median over the keys;
- the number of kernel passes per evidence and per gradient: calls of
  ``KernelOperator.mv``, counted by a host callback in the timed executable
  (``dense`` forms $K$ and has none). A pass evaluates $K$ once, block by
  block, and applies it to one vector or, under ``vmap``, to a batch of them
  (probes, sketch columns), so it is the unit of cost of a matrix-free GP;
- the error of the evidence against the dense one over the keys (bias,
  standard deviation, RMS and relative RMS), the relative RMS error of the
  gradient per hyperparameter, and the relative error of the predictive mean
  (and, for Gaussian, variance) at 64 test points;
- XLA's temporary allocation of the compiled evidence and gradient (peak
  working memory of the executable, ``memory_analysis().temp_size_in_bytes``);
- the effective dimension $d_\text{eff} = \mathrm{tr}(K (K + \mathrm{diag}(d))^{-1})$
  at the evaluation point, $d$ the noise or the Laplace site variances.

The plan is fixed, so one call reproduces the saved table (``--resume``
keeps the rows already in ``out`` and runs the rest):

- every size runs ``dense``, ``matfree`` and ``bbmm`` (the presets at rank
  200, Nyström) and, below ``large_n``, ``cg`` (unpreconditioned CG and its
  own SLQ), at both lengthscales;
- on the Gaussian likelihood below ``large_n``, and on the Bernoulli one
  below ``sweep_n``, it adds ``cg-nystrom`` (CG, a lazy rank-200 Nyström
  preconditioner and a preconditioned ``SLQLogdet``, whose gradient carries
  the control variate of gaussx#826);
- at ``sweep_n``, on the Gaussian likelihood, it adds ``matfree-fitc``,
  ``cg-nystrom-fixedP`` (the same preconditioner as ``cg-nystrom``, built
  once at the evaluation point, so the control variate has no gradient),
  ``matfree-r50``, ``matfree-r800`` and ``matfree-lanczos60``;
- from ``large_n`` up it runs the Gaussian likelihood only, with
  ``large_keys`` keys, plus ``matfree-lanczos60`` on the rough kernel, whose
  first key gives the same probes as ``matfree``'s, so the difference of the
  two is the truncation bias of 30 Lanczos steps.

Usage (one process, four cores, about half an hour on the reference VM):

    taskset -c 0-3 uv run --group docs python scripts/bench_gp_solvers.py \
        --out docs/notebooks/gp_solver_choices.csv
"""

from __future__ import annotations

import contextlib
import csv
import math
import platform
import statistics
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

import cyclopts
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import kernellib as kl
import numpy as np
import pyrox_gp as px
from loguru import logger
from pyrox_gp._kernel_operator import KernelOperator


jax.config.update("jax_enable_x64", True)

#: Gaussian noise variance, and the GP's own jitter (`GPPrior` default)
NOISE_VAR = 0.01
JITTER = 1e-6
#: the two kernels: Matérn-3/2 at a smooth and a rough lengthscale
LENGTHSCALES = {"smooth": 0.3, "rough": 0.03}
RANK = 200
NUM_TEST = 64
#: LaplaceInference tolerance: ten times looser than the CG rtol (1e-4)
LAPLACE_TOL = 1e-3

COLUMNS = (
    "id",
    "likelihood",
    "kernel",
    "lengthscale",
    "n",
    "config",
    "rank",
    "n_keys",
    "logz_dense",
    "logz_mean",
    "logz_bias",
    "logz_std",
    "logz_rms",
    "logz_rel_rms",
    "grad_rel_rms_lengthscale",
    "grad_rel_rms_variance",
    "grad_rel_rms_noise",
    "grad_rel_rms_max",
    "pred_mean_rel_err",
    "pred_var_rel_err",
    "t_evidence_s",
    "t_grad_s",
    "passes_evidence",
    "passes_grad",
    "mem_evidence_mb",
    "mem_grad_mb",
    "d_eff",
    "newton_iters",
    "device",
)


class Problem(NamedTuple):
    """One data set and the hyperparameters the evidence is evaluated at."""

    likelihood: str
    kernel: str
    X: jax.Array
    y: jax.Array
    X_star: jax.Array
    theta: jax.Array


class Run(NamedTuple):
    """What one configuration computes, as functions of ``(theta, solver)``."""

    evidence: Callable[..., Any]
    gradient: Callable[..., Any]
    predict: Callable[..., Any]


def make_problem(likelihood: str, kernel: str, n: int) -> Problem:
    """The inputs, observations and evaluation point of one row.

    The latent function is $f(x) = \\sin(6 x_1) + \\cos(4 x_2)$, the same for
    both lengthscales, so only the kernel's roughness changes between them.
    """
    X = jr.uniform(jr.key(0), (n, 2))
    X_star = jr.uniform(jr.key(1), (NUM_TEST, 2))
    f = jnp.sin(6.0 * X[:, 0]) + jnp.cos(4.0 * X[:, 1])
    log_ls = math.log(LENGTHSCALES[kernel])
    if likelihood == "gaussian":
        y = f + math.sqrt(NOISE_VAR) * jr.normal(jr.key(2), (n,))
        theta = jnp.array([log_ls, 0.0, math.log(NOISE_VAR)])
    else:
        y = jr.bernoulli(jr.key(2), jax.nn.sigmoid(2.0 * f)).astype(X.dtype)
        theta = jnp.array([log_ls, 0.0])
    return Problem(likelihood, kernel, X, y, X_star, theta)


def _kernel(theta: jax.Array) -> kl.Matern:
    return kl.Matern(nu=1.5, lengthscale=jnp.exp(theta[0]), variance=jnp.exp(theta[1]))


def make_run(problem: Problem, matrix_free: bool) -> Run:
    """The evidence, its gradient and the prediction of one problem."""
    X, y, X_star = problem.X, problem.y, problem.X_star

    def prior(theta: jax.Array, solver: Any) -> px.GPPrior:
        return px.GPPrior(_kernel(theta), X, solver=solver, matrix_free=matrix_free)

    if problem.likelihood == "gaussian":

        def evidence(theta: jax.Array, solver: Any) -> jax.Array:
            return prior(theta, solver).log_marginal(y, jnp.exp(theta[2]))

        def predict(theta: jax.Array, solver: Any) -> tuple[jax.Array, jax.Array]:
            conditioned = prior(theta, solver).condition(y, jnp.exp(theta[2]))
            return conditioned.predict(X_star)

        return Run(evidence, jax.value_and_grad(evidence), predict)

    strategy = px.LaplaceInference(tol=LAPLACE_TOL)

    def fit(theta: jax.Array, solver: Any) -> Any:
        return strategy.fit(prior(theta, solver), px.BernoulliLikelihood(), y)

    def laplace_evidence(theta: jax.Array, solver: Any) -> jax.Array:
        return fit(theta, solver).log_marginal_approx

    def fit_outputs(theta: jax.Array, solver: Any) -> tuple[jax.Array, ...]:
        result = fit(theta, solver)
        mean = result.predict_mean(X_star)
        return result.log_marginal_approx, mean, result.n_iter, result.site_nat2

    return Run(fit_outputs, jax.value_and_grad(laplace_evidence), fit_outputs)


def build_solver(config: str, key: jax.Array, problem: Problem) -> Any:
    """The solver of one configuration, its randomness drawn from ``key``."""
    sketch_key, probe_key = jr.split(key)
    cg = gx.CGSolver(rtol=1e-4, throw=False)
    if config == "dense":
        return None
    # a rank above N only arises in the smoke test
    rank = min(config_rank(config) or RANK, problem.X.shape[0])
    if config == "cg":
        return gx.KeyedSolver(cg, probe_key)
    if config in ("cg-nystrom", "cg-nystrom-fixedP"):
        if config == "cg-nystrom":
            preconditioner = gx.NystromPreconditioner.lazy(rank, key=sketch_key)
        else:
            # built once at the evaluation point: P has no gradient, so the
            # control variate of gaussx#826 contributes none
            operator = KernelOperator(_kernel(problem.theta), problem.X)
            shift = jnp.exp(problem.theta[2]) + JITTER
            preconditioner = gx.NystromPreconditioner.from_operator(
                operator, rank, shift=shift, key=sketch_key
            )
        composed = gx.compose_solver(
            cg, preconditioner=preconditioner, logdet=gx.SLQLogdet(num_probes=50)
        )
        return gx.KeyedSolver(composed, probe_key)
    if config in ("matfree", "matfree-r50", "matfree-r800"):
        return px.matfree_solver(rank=rank, key=key, throw=False)
    if config == "matfree-fitc":
        return px.matfree_solver(rank=rank, preconditioner="fitc", key=key, throw=False)
    if config == "matfree-lanczos60":
        return px.matfree_solver(rank=rank, lanczos_order=60, key=key, throw=False)
    if config == "bbmm":
        return px.bbmm_solver(rank=rank, key=key, throw=False)
    raise ValueError(f"unknown configuration {config!r}")


def config_rank(config: str) -> int | None:
    """The preconditioner rank of a configuration; ``None`` for none."""
    if config in ("dense", "cg"):
        return None
    if config.endswith("-r50"):
        return 50
    if config.endswith("-r800"):
        return 800
    return RANK


def plan(
    n: int, likelihood: str, kernel: str, *, large_n: int, sweep_n: int
) -> list[str]:
    """The configurations run at one size, likelihood and kernel."""
    if n >= large_n:
        if likelihood != "gaussian":
            return []
        configs = ["dense", "matfree", "bbmm"]
        # the Lanczos truncation bias, on the kernel that needs the most steps
        return configs + (["matfree-lanczos60"] if kernel == "rough" else [])
    configs = ["dense", "matfree", "bbmm", "cg"]
    if likelihood == "bernoulli":
        return configs + (["cg-nystrom"] if n < sweep_n else [])
    configs += ["cg-nystrom"]
    if n == sweep_n:
        configs += ["matfree-fitc", "cg-nystrom-fixedP", "matfree-r50"]
        configs += ["matfree-r800", "matfree-lanczos60"]
    return configs


_PASSES = [0]


def _tally() -> None:
    _PASSES[0] += 1


@contextlib.contextmanager
def counting_passes() -> Iterator[None]:
    """Trace ``KernelOperator.mv`` with a host callback that counts its calls.

    The callback takes no argument, so under ``vmap`` it runs once per
    batched call, not once per vector.
    """
    original = KernelOperator.mv

    def mv(self: KernelOperator, vector: jax.Array) -> jax.Array:
        jax.debug.callback(_tally)
        return original(self, vector)

    KernelOperator.mv = mv
    try:
        yield
    finally:
        KernelOperator.mv = original


class Compiled(NamedTuple):
    """An ahead-of-time compiled function over ``(theta, solver arrays)``."""

    call: Callable[..., Any]
    temp_mb: float


def compile_fn(fn: Callable[..., Any], theta: jax.Array, solver: Any) -> Compiled:
    """Compile ``fn(theta, solver)`` with the solver's arrays as arguments."""
    arrays, static = eqx.partition(solver, eqx.is_array)

    def over_arrays(theta: jax.Array, arrays: Any) -> Any:
        return fn(theta, eqx.combine(arrays, static))

    compiled = jax.jit(over_arrays).lower(theta, arrays).compile()
    analysis = compiled.memory_analysis()
    temp = getattr(analysis, "temp_size_in_bytes", float("nan"))

    def call(theta: jax.Array, solver: Any) -> Any:
        return compiled(theta, eqx.partition(solver, eqx.is_array)[0])

    return Compiled(call, temp / 2**20)


def timed(call: Callable[..., Any], *args: Any) -> tuple[Any, float]:
    """``call(*args)`` and its wall-clock time, outputs ready."""
    start = time.perf_counter()
    out = jax.block_until_ready(call(*args))
    return out, time.perf_counter() - start


def effective_dimension(theta: jax.Array, X: jax.Array, d: jax.Array) -> float:
    """$\\mathrm{tr}(K A^{-1}) = N - \\sum_n d_n (A^{-1})_{nn}$, densely."""
    K = _kernel(theta)(X, X) + JITTER * jnp.eye(X.shape[0], dtype=X.dtype)
    A = K + jnp.diag(jnp.broadcast_to(d, X.shape[:1]))
    L = jnp.linalg.cholesky(A)
    L_inv = jax.scipy.linalg.solve_triangular(L, jnp.eye(X.shape[0]), lower=True)
    diag_inv = jnp.sum(L_inv**2, axis=0)
    return float(X.shape[0] - jnp.sum(d * diag_inv))


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


class Reference(NamedTuple):
    """The dense results a configuration is compared with."""

    logz: float
    grad: np.ndarray
    mean: np.ndarray
    var: np.ndarray | None


def measure(
    problem: Problem,
    config: str,
    *,
    keys: int,
    reference: Reference | None,
) -> tuple[dict[str, Any], Reference]:
    """One row of the table: run ``config`` on ``problem`` for every key."""
    n = problem.X.shape[0]
    dense = config == "dense"
    run = make_run(problem, matrix_free=not dense)
    gaussian = problem.likelihood == "gaussian"
    n_keys = 1 if dense else keys
    solvers = [build_solver(config, jr.key(100 + k), problem) for k in range(n_keys)]
    theta = problem.theta

    with counting_passes():
        evidence = compile_fn(run.evidence, theta, solvers[0])
        gradient = compile_fn(run.gradient, theta, solvers[0])
    values, grads, t_evidence, t_grad, iters = [], [], [], [], []
    passes_evidence = passes_grad = 0
    outputs: list[Any] = []
    for solver in solvers:
        _PASSES[0] = 0
        out, seconds = timed(evidence.call, theta, solver)
        passes_evidence = _PASSES[0]
        t_evidence.append(seconds)
        if gaussian:
            values.append(float(out))
        else:
            values.append(float(out[0]))
            iters.append(int(out[2]))
        outputs.append(out)
        _PASSES[0] = 0
        (_, grad), seconds = timed(gradient.call, theta, solver)
        passes_grad = _PASSES[0]
        t_grad.append(seconds)
        grads.append(np.asarray(grad))

    first_out = outputs[0]
    if gaussian:
        mean, var = (
            np.asarray(a)
            for a in compile_fn(run.predict, theta, solvers[0]).call(theta, solvers[0])
        )
    else:
        mean, var = np.asarray(first_out[1]), None

    d_eff = float("nan")
    if dense:
        d = jnp.asarray(NOISE_VAR) if gaussian else 1.0 / jnp.asarray(first_out[3])
        d_eff = effective_dimension(theta, problem.X, d)
        reference = Reference(values[0], grads[0], mean, var)
    assert reference is not None

    values_arr = np.asarray(values)
    errors = values_arr - reference.logz
    grads_arr = np.stack(grads)
    grad_rel = np.sqrt(np.mean((grads_arr - reference.grad) ** 2, axis=0)) / np.abs(
        reference.grad
    )
    names = ("lengthscale", "variance", "noise")
    row: dict[str, Any] = {
        "id": f"{problem.likelihood}-{problem.kernel}-{n}-{config}",
        "likelihood": problem.likelihood,
        "kernel": problem.kernel,
        "lengthscale": LENGTHSCALES[problem.kernel],
        "n": n,
        "config": config,
        "rank": config_rank(config) or "",
        "n_keys": n_keys,
        "logz_dense": reference.logz,
        "logz_mean": float(values_arr.mean()),
        "logz_bias": float(errors.mean()),
        "logz_std": float(values_arr.std(ddof=1)) if n_keys > 1 else 0.0,
        "logz_rms": float(np.sqrt(np.mean(errors**2))),
        "logz_rel_rms": float(np.sqrt(np.mean(errors**2)) / abs(reference.logz)),
        "grad_rel_rms_max": float(grad_rel.max()),
        "pred_mean_rel_err": _rel(mean, reference.mean),
        "pred_var_rel_err": (
            _rel(var, reference.var)
            if var is not None and reference.var is not None
            else ""
        ),
        "t_evidence_s": statistics.median(t_evidence),
        "t_grad_s": statistics.median(t_grad),
        "passes_evidence": "" if dense else passes_evidence,
        "passes_grad": "" if dense else passes_grad,
        "mem_evidence_mb": evidence.temp_mb,
        "mem_grad_mb": gradient.temp_mb,
        "d_eff": d_eff if dense else "",
        "newton_iters": statistics.median(iters) if iters else "",
        "device": f"{jax.devices()[0].platform}:{platform.machine()}",
    }
    for name, value in zip(names, grad_rel, strict=False):
        row[f"grad_rel_rms_{name}"] = float(value)
    for name in names[len(grad_rel) :]:
        row[f"grad_rel_rms_{name}"] = ""
    return row, reference


app = cyclopts.App(help=__doc__.splitlines()[0], result_action="return_value")


@app.default
def main(
    *,
    out: Path = Path("docs/notebooks/gp_solver_choices.csv"),
    sizes: tuple[int, ...] = (500, 2000, 8000),
    likelihoods: tuple[str, ...] = ("gaussian", "bernoulli"),
    kernels: tuple[str, ...] = ("smooth", "rough"),
    keys: int = 3,
    large_keys: int = 1,
    large_n: int = 4000,
    sweep_n: int = 2000,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Run the plan and write one CSV row per configuration.

    Args:
        out: The CSV to write; rows are appended to it as they finish.
        sizes: Numbers of training points $N$.
        likelihoods: ``"gaussian"`` and / or ``"bernoulli"``.
        kernels: ``"smooth"`` and / or ``"rough"``.
        keys: Keys per stochastic configuration below ``large_n``.
        large_keys: Keys per stochastic configuration from ``large_n`` up.
        large_n: The size from which only the presets and the dense
            reference run, on the Gaussian likelihood, with ``large_keys``
            keys.
        sweep_n: The size of the rank, preconditioner and Lanczos-order sweep.
        resume: Keep the rows already in ``out`` and run only the others.

    Returns:
        The rows written.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, Any]] = {}
    if resume and out.is_file():
        with out.open(newline="") as handle:
            done = {row["id"]: row for row in csv.DictReader(handle)}
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for n in sizes:
            large = n >= large_n
            for likelihood in likelihoods:
                for kernel in kernels:
                    configs = plan(
                        n, likelihood, kernel, large_n=large_n, sweep_n=sweep_n
                    )
                    problem = make_problem(likelihood, kernel, n)
                    reference = None
                    for config in configs:
                        row_id = f"{likelihood}-{kernel}-{n}-{config}"
                        pending = any(
                            f"{likelihood}-{kernel}-{n}-{c}" not in done
                            for c in configs
                        )
                        if row_id in done and not (config == "dense" and pending):
                            row = done[row_id]
                        else:
                            row, reference = measure(
                                problem,
                                config,
                                keys=large_keys if large else keys,
                                reference=reference,
                            )
                            # a dense row rerun only for its reference is kept
                            row = done.get(row_id, row)
                            logger.info(
                                "{} logZ rel {:.2e} grad rel {:.2e} "
                                "t {:.2f}s / {:.2f}s passes {} ({:.0f}s so far)",
                                row["id"],
                                float(row["logz_rel_rms"]),
                                float(row["grad_rel_rms_max"]),
                                float(row["t_evidence_s"]),
                                float(row["t_grad_s"]),
                                row["passes_evidence"],
                                time.perf_counter() - start,
                            )
                        writer.writerow(row)
                        handle.flush()
                        rows.append(row)
    logger.success("{} rows -> {}", len(rows), out)
    return rows


if __name__ == "__main__":
    app()
