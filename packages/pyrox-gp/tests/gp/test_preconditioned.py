"""Tests for `pyrox_gp.preconditioned_cg_solver` (P4)."""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from pyrox_gp import GPPrior, Matern, preconditioned_cg_solver


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
