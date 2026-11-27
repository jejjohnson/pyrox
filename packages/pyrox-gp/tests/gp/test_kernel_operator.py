"""Tests for the lineax structure functions of `KernelOperator` (pyrox#554)."""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest
from pyrox_gp import Matern
from pyrox_gp._kernel_operator import KernelOperator, matrix_free_operator


jax.config.update("jax_enable_x64", True)

X = jr.uniform(jr.key(0), (60, 2))
KERNEL = Matern(nu=1.5).frozen()


def test_lineax_structure_functions():
    operator = KernelOperator(KERNEL, X)
    dense = KERNEL(X, X)
    # lx.conj raised NotImplementedError before pyrox#554
    assert lx.conj(operator) is operator
    noisy = matrix_free_operator(KERNEL, X, jnp.asarray(0.1))
    np.testing.assert_allclose(
        lx.conj(noisy).as_matrix(), noisy.as_matrix(), rtol=1e-12
    )
    assert lx.materialise(operator) is operator
    assert lx.linearise(operator) is operator
    np.testing.assert_allclose(lx.diagonal(operator), jnp.diag(dense), rtol=1e-12)
    diagonal, lower, upper = lx.tridiagonal(operator)
    np.testing.assert_allclose(diagonal, jnp.diag(dense), rtol=1e-12)
    np.testing.assert_allclose(lower, jnp.diag(dense, k=-1), rtol=1e-12)
    np.testing.assert_allclose(upper, jnp.diag(dense, k=1), rtol=1e-12)
    assert lx.is_symmetric(operator)
    assert lx.is_positive_semidefinite(operator)
    assert not lx.is_negative_semidefinite(operator)
    assert not lx.is_diagonal(operator)
    assert not lx.is_tridiagonal(operator)
    assert not lx.has_unit_diagonal(operator)
    assert not lx.is_lower_triangular(operator)
    assert not lx.is_upper_triangular(operator)


SOLVERS = {
    "cg": gx.CGSolver(rtol=1e-8, atol=1e-8),
    "minres": gx.MINRESSolver(rtol=1e-8, atol=1e-8),
    "lsmr": gx.LSMRSolver(atol=1e-10, btol=1e-10, max_steps=4000),
    "bbmm": gx.BBMMSolver(rtol=1e-8, atol=1e-8),
    "preconditioned_cg": gx.PreconditionedCGSolver(
        preconditioner_rank=20, shift=0.1, rtol=1e-8, atol=1e-8
    ),
}


@pytest.mark.parametrize("name", list(SOLVERS))
def test_every_solver_runs_matrix_free(name):
    # K + 0.1 I, matrix-free: each solver reaches K̂⁻¹ y to its tolerance.
    # LSMR failed in lx.conj before pyrox#554.
    operator = matrix_free_operator(KERNEL, X, jnp.asarray(0.1))
    rhs = jnp.sin(6.0 * X[:, 0])
    expected = jnp.linalg.solve(operator.as_matrix(), rhs)
    solution = SOLVERS[name].solve(operator, rhs)
    np.testing.assert_allclose(solution, expected, rtol=1e-5, atol=1e-6)
