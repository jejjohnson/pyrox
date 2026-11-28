"""Tests for `pyrox_gp.preconditioned_cg_solver` (P4)."""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
from pyrox_gp import GPPrior, Matern, gp_factor, preconditioned_cg_solver
from pyrox_gp._kernel_operator import KernelOperator
from pyrox_gp._preconditioned import _NoiseSplitPreconditioner, _split_system


jax.config.update("jax_enable_x64", True)

N = 600
X = jr.uniform(jr.key(0), (N, 2))
Y = jnp.sin(6 * X[:, 0]) * jnp.cos(4 * X[:, 1]) + 0.2 * jr.normal(jr.key(1), (N,))


def _lml(lengthscale, noise_var, solver):
    prior = GPPrior(Matern(nu=1.5, init_lengthscale=lengthscale), X, solver=solver)
    op = prior._noisy_operator(noise_var)
    return gx.log_marginal_likelihood(prior.mean(X), op, Y, solver=solver)


_value_and_grad = jax.value_and_grad(_lml, argnums=(0, 1))


def test_validation():
    with pytest.raises(ValueError, match="logdet must be"):
        preconditioned_cg_solver(
            logdet="exact",  # ty: ignore[invalid-argument-type]
            key=jr.key(0),
        )
    with pytest.raises(ValueError, match="preconditioner must be"):
        preconditioned_cg_solver(
            shift=0.1,
            preconditioner="jacobi",  # ty: ignore[invalid-argument-type]
            key=jr.key(0),
        )
    with pytest.raises(ValueError, match="shift must be positive"):
        preconditioned_cg_solver(shift=0.0, key=jr.key(0))


@pytest.mark.slow
@pytest.mark.parametrize("preconditioner", ["nystrom", "rpcholesky"])
def test_marginal_likelihood_and_gradient_match_dense(preconditioner):
    # Matern-3/2 exact GP, n = 600, noise 0.04 = shift. The solve is exact
    # to CG tolerance, so the gradients match dense within 2 % (measured
    # 1.2 %); the value carries SLQ's error on log|K + s I| with 20 probes
    # (measured 1.8 nats), bounded at 3.
    dense, dense_grad = _value_and_grad(0.3, 0.04, gx.DenseSolver())
    solver = preconditioned_cg_solver(
        shift=0.04, preconditioner=preconditioner, rank=150, key=jr.key(2)
    )
    value, grad = _value_and_grad(0.3, 0.04, solver)
    assert abs(float(value) - float(dense)) < 3.0
    assert np.allclose(np.asarray(grad), np.asarray(dense_grad), rtol=0.02)


@pytest.mark.slow
def test_preconditioning_converges_where_plain_cg_does_not():
    # 60 CG steps: enough with a rank-150 preconditioner, not without one
    # (lineax raises when CG does not converge).
    solver = preconditioned_cg_solver(shift=0.04, rank=150, max_steps=60, key=jr.key(2))
    value, _ = _value_and_grad(0.3, 0.04, solver)
    assert np.isfinite(float(value))
    with pytest.raises(Exception, match=r"(?i)max_steps|converge|maximum"):
        _value_and_grad(0.3, 0.04, gx.CGSolver(rtol=1e-6, atol=1e-6, max_steps=60))


# --- logdet="nystrom" (pyrox#556) -------------------------------------------

N_LOGDET = 400


def _matrix_free_system(n, lengthscale, noise_var):
    prior = GPPrior(
        Matern(nu=1.5, init_lengthscale=lengthscale), X[:n], matrix_free=True
    )
    return prior._noisy_operator(noise_var)


def test_nystrom_logdet_builds_and_is_more_accurate():
    # Matérn-3/2 at N = 400, noise 0.04, rank 50, 20 probes, over 20 keys:
    # RMS error about 2 nats for "nystrom" against 10 for "slq" (measured).
    # Before pyrox#556 the solver refused to build with logdet="nystrom".
    solver = preconditioned_cg_solver(logdet="nystrom", key=jr.key(0))
    assert isinstance(solver, gx.ComposedSolver)
    operator = _matrix_free_system(N_LOGDET, 0.3, 0.04)
    exact = jnp.linalg.slogdet(operator.as_matrix())[1]
    keys = jr.split(jr.key(100), 20)

    def estimate(logdet, key):
        solver = preconditioned_cg_solver(rank=50, logdet=logdet, key=key)
        return solver.logdet(operator, key=key)

    rms = {}
    for logdet in ("slq", "nystrom"):
        estimates = jax.vmap(lambda key, logdet=logdet: estimate(logdet, key))(keys)
        rms[logdet] = float(jnp.sqrt(jnp.mean(jnp.square(estimates - exact))))
    assert rms["nystrom"] < 0.5 * rms["slq"]


def test_nystrom_logdet_reads_the_noise_from_the_system():
    # The shift is the scalar of the matrix-free sum, never shift=: a shift
    # above the noise would make the sketch indefinite (gaussx#878).
    operator = _matrix_free_system(200, 0.3, 0.04)
    exact = jnp.linalg.slogdet(operator.as_matrix())[1]
    solver = preconditioned_cg_solver(
        shift=10.0, rank=100, logdet="nystrom", key=jr.key(0)
    )
    estimate = solver.logdet(operator)
    assert np.isfinite(float(estimate))
    assert abs(float(estimate - exact)) < 0.01 * abs(float(exact))


def test_nystrom_logdet_dense_system_needs_shift():
    dense = GPPrior(Matern(nu=1.5, init_lengthscale=0.3), X[:200])
    operator = dense._noisy_operator(0.04)
    exact = jnp.linalg.slogdet(operator.as_matrix())[1]
    with pytest.raises(ValueError, match="not a sum"):
        preconditioned_cg_solver(logdet="nystrom", key=jr.key(0)).logdet(operator)
    solver = preconditioned_cg_solver(
        shift=0.04, rank=100, logdet="nystrom", key=jr.key(0)
    )
    estimate = solver.logdet(operator)
    assert abs(float(estimate - exact)) < 0.01 * abs(float(exact))


@pytest.mark.slow
def test_nystrom_logdet_gradient_matches_dense():
    # 10 keys at N = 400, rank 50: the mean gradient is within 3 standard
    # errors of the dense one (measured 1.7 for both hyperparameters).
    x, y = X[:N_LOGDET], Y[:N_LOGDET]
    dense = _mf_value_and_grad(x, y, 0.3, 0.04, gx.DenseSolver(), matrix_free=False)
    dense_grad = np.asarray(dense[1])
    grads = []
    for index in range(10):
        solver = preconditioned_cg_solver(
            rank=50, logdet="nystrom", key=jr.key(200 + index)
        )
        _, grad = _mf_value_and_grad(x, y, 0.3, 0.04, solver, matrix_free=True)
        grads.append(np.asarray(grad))
    grads = np.stack(grads)
    assert np.all(np.isfinite(grads))
    standard_error = grads.std(axis=0, ddof=1) / np.sqrt(len(grads))
    assert np.all(np.abs(grads.mean(axis=0) - dense_grad) < 3 * standard_error)


# --- A diagonal that varies by point (pyrox#555) -----------------------------


def _heteroscedastic_system(site_variance):
    # the reproducer of pyrox#555: K_op + diag(d), d different at every point
    kernel_op = KernelOperator(Matern(nu=1.5).frozen(), X[:60])
    return lx.TaggedLinearOperator(
        kernel_op + lx.DiagonalLinearOperator(site_variance),
        lx.positive_semidefinite_tag,
    )


SITE_VARIANCE = jnp.linspace(1.0, 4.0, 60)
SITE_TARGET = (X[:60, 0] > 0.5).astype(float)


@pytest.mark.parametrize("preconditioner", ["nystrom", "rpcholesky"])
def test_heteroscedastic_diagonal(preconditioner):
    # Before pyrox#555 this raised "the system is not a sum K + s I". The CG
    # solve matches the dense one at CG's tolerance (rtol 1e-6 here, checked
    # at the issue's 1e-5).
    system = _heteroscedastic_system(SITE_VARIANCE)
    expected = jnp.linalg.solve(system.as_matrix(), SITE_TARGET)
    solver = preconditioned_cg_solver(
        rank=20, preconditioner=preconditioner, key=jr.key(1)
    )
    solution = solver.solve(system, SITE_TARGET)
    np.testing.assert_allclose(solution, expected, rtol=1e-5)


def test_heteroscedastic_diagonal_under_jit():
    # The diagonal is traced, so the split and the per-point shift are read
    # by type, never by value.
    solver = preconditioned_cg_solver(rank=20, key=jr.key(1))

    @jax.jit
    def solve(site_variance):
        return solver.solve(_heteroscedastic_system(site_variance), SITE_TARGET)

    for site_variance in (SITE_VARIANCE, SITE_VARIANCE[::-1] * 2.0):
        system = _heteroscedastic_system(site_variance)
        expected = jnp.linalg.solve(system.as_matrix(), SITE_TARGET)
        np.testing.assert_allclose(solve(site_variance), expected, rtol=1e-5)


def test_heteroscedastic_nystrom_logdet():
    # logdet="nystrom" takes the per-point d too: rank 50 of 60 leaves a
    # near-identity remainder, so 1 % of the exact value holds with margin.
    system = _heteroscedastic_system(SITE_VARIANCE)
    exact = jnp.linalg.slogdet(system.as_matrix())[1]
    solver = preconditioned_cg_solver(rank=50, logdet="nystrom", key=jr.key(0))
    estimate = solver.logdet(system)
    assert abs(float(estimate - exact)) < 0.01 * abs(float(exact))


@pytest.mark.parametrize("kind", ["nystrom", "rpcholesky"])
def test_scalar_noise_unchanged(kind):
    # A Gaussian-noise system K_op + s I keeps the scalar shift, so its
    # preconditioner is the one built before pyrox#555, bit for bit.
    operator = _matrix_free_system(80, 0.3, 0.04)
    split = _split_system(operator)
    assert split is not None
    kernel_op, noise = split
    key = jr.key(3)
    psd_part = lx.TaggedLinearOperator(kernel_op, lx.positive_semidefinite_tag)
    if kind == "nystrom":
        reference = gx.NystromPreconditioner.from_operator(
            psd_part, 30, shift=noise, key=key
        )
    else:
        reference = gx.PartialCholeskyPreconditioner.from_operator(
            psd_part, 30, shift=noise, pivoting="random", key=key
        )
    got = _NoiseSplitPreconditioner(kind, 30, None, key).as_operator(operator)
    assert got is not None
    expected = reference.as_operator()
    assert expected is not None
    np.testing.assert_array_equal(got.as_matrix(), expected.as_matrix())


# --- Matrix-free GPPrior (pyrox#277) ----------------------------------------


def _mf_lml(X, y, lengthscale, noise_var, solver, *, matrix_free):
    prior = GPPrior(
        Matern(nu=1.5, init_lengthscale=lengthscale),
        X,
        solver=solver,
        matrix_free=matrix_free,
    )
    op = prior._noisy_operator(noise_var)
    return gx.log_marginal_likelihood(prior.mean(X), op, y, solver=solver)


_mf_value_and_grad = jax.value_and_grad(_mf_lml, argnums=(2, 3))


def test_matrix_free_operator_is_a_noise_sum():
    x = X[:50]
    v = jr.normal(jr.key(3), (50,))
    noisy = GPPrior(Matern(nu=1.5), x, matrix_free=True, block_size=16)
    dense = GPPrior(Matern(nu=1.5), x)
    op = noisy._noisy_operator(0.04)
    np.testing.assert_allclose(
        op.mv(v), dense._noisy_operator(0.04).mv(v), rtol=1e-12, atol=1e-12
    )
    split = _split_system(op)
    assert split is not None
    K_op, diagonal = split
    assert isinstance(K_op, KernelOperator)
    # the scalar of K_op + s I, not a vector: the Gaussian-noise path
    assert jnp.ndim(diagonal) == 0
    np.testing.assert_allclose(diagonal, 0.04 + noisy.jitter)
    np.testing.assert_allclose(lx.diagonal(K_op), jnp.diag(K_op.as_matrix()))
    assert _split_system(dense._noisy_operator(0.04)) is None
    np.testing.assert_allclose(
        noisy._prior_operator().as_matrix(), dense._prior_operator().as_matrix()
    )


def test_matrix_free_dense_solver_matches_dense():
    x, y = X[:80], Y[:80]
    ref = _mf_value_and_grad(x, y, 0.3, 0.04, gx.DenseSolver(), matrix_free=False)
    got = _mf_value_and_grad(x, y, 0.3, 0.04, gx.DenseSolver(), matrix_free=True)
    np.testing.assert_allclose(got[0], ref[0], rtol=1e-10)
    np.testing.assert_allclose(np.asarray(got[1]), np.asarray(ref[1]), rtol=1e-8)


def test_dense_system_without_shift_raises():
    solver = preconditioned_cg_solver(rank=10, key=jr.key(0))
    with pytest.raises(ValueError, match="not a sum"):
        _mf_lml(X[:40], Y[:40], 0.3, 0.04, solver, matrix_free=False)


def test_gp_factor_under_svi_matrix_free():
    # The kernel is frozen once per call, so its sites register outside the
    # matvec's lax.map, and SVI reaches the same optimum as the dense path.
    x, y = X[:60], Y[:60]
    solver = preconditioned_cg_solver(rank=20, key=jr.key(0))

    def model(matrix_free):
        prior = GPPrior(
            Matern(nu=1.5, init_lengthscale=0.3),
            x,
            solver=solver if matrix_free else None,
            matrix_free=matrix_free,
        )
        noise_var = numpyro.param(
            "noise_var", 0.1, constraint=dist.constraints.positive
        )
        gp_factor("y", prior, y, noise_var)

    params = {}
    for matrix_free in (False, True):
        svi = SVI(model, lambda matrix_free: None, Adam(1e-2), Trace_ELBO())
        result = svi.run(jr.key(0), 40, matrix_free, progress_bar=False)
        params[matrix_free] = result.params
    for name, value in params[False].items():
        np.testing.assert_allclose(params[True][name], value, rtol=0.05)


@pytest.mark.slow
@pytest.mark.parametrize("preconditioner", ["nystrom", "rpcholesky"])
def test_matrix_free_without_shift_matches_dense(preconditioner):
    # Same check as above with the noise read from K_op + s I: no shift.
    dense, dense_grad = _value_and_grad(0.3, 0.04, gx.DenseSolver())
    solver = preconditioned_cg_solver(
        preconditioner=preconditioner, rank=150, key=jr.key(2)
    )
    value, grad = _mf_value_and_grad(X, Y, 0.3, 0.04, solver, matrix_free=True)
    assert abs(float(value) - float(dense)) < 3.0
    assert np.allclose(np.asarray(grad), np.asarray(dense_grad), rtol=0.02)


# --- Large n, matrix-free (pyrox#277; integration tier) ----------------------

N_LARGE, N_SUBSET = 20_000, 3000
X_LARGE = jr.uniform(jr.key(0), (N_LARGE, 2))
Y_LARGE = jnp.sin(6 * X_LARGE[:, 0]) * jnp.cos(4 * X_LARGE[:, 1]) + 0.2 * jr.normal(
    jr.key(1), (N_LARGE,)
)
# lanczos_order=80: at lengthscale 0.2 the default 30 Lanczos steps are too
# few. On the n = 3000 subset they bias the SLQ log-determinant gradient by
# ~8 %, and at n = 20 000 they flip the sign of the noise gradient; 60+
# steps bring the n = 3000 bias below 1 %. The backward pass through the
# Lanczos runs dominates the cost, which grows with the order, not the probe
# count (this test took ~33 min on 16 heavily shared CPU cores).
_LARGE_SOLVER = preconditioned_cg_solver(rank=200, lanczos_order=80, key=jr.key(2))


@pytest.mark.slow
def test_matrix_free_n20000_matches_dense_subset():
    # Matern-3/2, lengthscale 0.2, noise 0.04. On the n = 3000 subset the
    # matrix-free path matches DenseSolver to SLQ tolerance (measured: value
    # 8 nats, gradients 0.9 % / 1.5 %). At n = 20 000 (K alone would be
    # 3.2 GB) it runs in O(block_size n) memory and its gradient has the
    # dense subset's signs.
    x, y = X_LARGE[:N_SUBSET], Y_LARGE[:N_SUBSET]
    dense, dense_grad = _mf_value_and_grad(
        x, y, 0.2, 0.04, gx.DenseSolver(), matrix_free=False
    )
    value, grad = _mf_value_and_grad(x, y, 0.2, 0.04, _LARGE_SOLVER, matrix_free=True)
    assert abs(float(value) - float(dense)) < 25.0
    assert np.allclose(np.asarray(grad), np.asarray(dense_grad), rtol=0.03)

    value, grad = _mf_value_and_grad(
        X_LARGE, Y_LARGE, 0.2, 0.04, _LARGE_SOLVER, matrix_free=True
    )
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.array_equal(np.sign(grad), np.sign(dense_grad))


def _cg_steps(n, preconditioned):
    x, y = X_LARGE[:n], Y_LARGE[:n]
    op = GPPrior(Matern(nu=1.5, init_lengthscale=0.2), x, matrix_free=True)
    op = op._noisy_operator(0.04)
    options = {}
    if preconditioned:
        precond = _LARGE_SOLVER.preconditioner
        assert precond is not None
        options["preconditioner"] = precond.as_operator(op)
    cg = lx.CG(rtol=1e-6, atol=1e-6, max_steps=2000)
    solution = lx.linear_solve(op, y, cg, options=options, throw=False)
    return int(solution.stats["num_steps"])


@pytest.mark.slow
def test_cg_iterations_stay_within_budget_as_n_grows():
    # Rank-200 Nystrom preconditioner, noise read from K_op + s I. Measured
    # 20 / 35 / 51 steps at n = 2000 / 8000 / 20 000; plain CG needs
    # 179 steps already at n = 1000 and 339 at n = 4000.
    steps = [_cg_steps(n, preconditioned=True) for n in (2000, 8000, N_LARGE)]
    assert max(steps) <= 60, steps
    assert _cg_steps(2000, preconditioned=False) > 3 * steps[0]
