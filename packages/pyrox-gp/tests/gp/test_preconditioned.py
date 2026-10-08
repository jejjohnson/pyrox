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
from pyrox_gp._kernel_operator import KernelOperator, split_noise


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
    with pytest.raises(ValueError, match="NystromLogdet"):
        preconditioned_cg_solver(shift=0.1, logdet="nystrom", key=jr.key(0))
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
    split = split_noise(op)
    assert split is not None
    K_op, diagonal = split
    assert isinstance(K_op, KernelOperator)
    np.testing.assert_allclose(diagonal, 0.04 + noisy.jitter)
    np.testing.assert_allclose(lx.diagonal(K_op), jnp.diag(K_op.as_matrix()))
    assert split_noise(dense._noisy_operator(0.04)) is None
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
