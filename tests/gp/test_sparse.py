"""Tests for ``pyrox.gp.SparseGPPrior``.

The sparse prior is a thin shell — kernel + inducing inputs + jitter +
helper accessors. Verify shapes, dtype-preservation, jitter regularization,
and that the helpers compose correctly with the kernel.
"""

from __future__ import annotations

import jax.numpy as jnp
from gaussx import CGSolver, DenseSolver

from pyrox.gp import RBF, Matern, SparseGPPrior


def _toy_inducing(M: int = 5, D: int = 1) -> jnp.ndarray:
    return jnp.linspace(-2.0, 2.0, M).reshape(M, D)


def test_num_inducing_property_returns_m():
    Z = _toy_inducing(7)
    prior = SparseGPPrior(kernel=RBF(), Z=Z)
    assert prior.num_inducing == 7


def test_zero_mean_default():
    Z = _toy_inducing()
    prior = SparseGPPrior(kernel=RBF(), Z=Z)
    out = prior.mean(Z)
    assert out.shape == (Z.shape[0],)
    assert jnp.allclose(out, 0.0)


def test_callable_mean_fn_is_applied_to_input():
    Z = _toy_inducing()
    prior = SparseGPPrior(kernel=RBF(), Z=Z, mean_fn=lambda x: x.squeeze(-1) * 2.0)
    X = jnp.array([[0.5], [1.0], [-0.5]])
    assert jnp.allclose(prior.mean(X), jnp.array([1.0, 2.0, -1.0]))


def test_inducing_operator_includes_jitter_regularization():
    """``K_zz + jitter * I`` — diagonal must be lifted by exactly ``jitter``."""
    Z = _toy_inducing(5)
    jitter = 5e-3
    kern = RBF(init_variance=1.5, init_lengthscale=0.4)
    prior = SparseGPPrior(kernel=kern, Z=Z, jitter=jitter)
    K = kern(Z, Z)
    K_reg = prior.inducing_operator().as_matrix()
    assert jnp.allclose(K_reg, K + jitter * jnp.eye(Z.shape[0]), atol=1e-10)


def test_cross_covariance_matches_kernel_call():
    """``cross_covariance(X) == kernel(X, Z)`` — pure delegation."""
    Z = _toy_inducing(4)
    X = jnp.linspace(-3.0, 3.0, 7).reshape(-1, 1)
    kern = Matern(init_variance=0.8, init_lengthscale=0.5)
    prior = SparseGPPrior(kernel=kern, Z=Z)
    assert jnp.allclose(prior.cross_covariance(X), kern(X, Z))


def test_kernel_diag_matches_kernel_diag_call():
    """``kernel_diag(X) == kernel.diag(X)`` — pure delegation; stationary
    kernels override ``diag`` for an O(N) shortcut, which the sparse prior
    must preserve."""
    Z = _toy_inducing(4)
    X = jnp.linspace(-3.0, 3.0, 7).reshape(-1, 1)
    kern = RBF(init_variance=2.0, init_lengthscale=0.7)
    prior = SparseGPPrior(kernel=kern, Z=Z)
    assert jnp.allclose(prior.kernel_diag(X), kern.diag(X))


def test_resolved_solver_defaults_to_dense_solver():
    Z = _toy_inducing()
    prior = SparseGPPrior(kernel=RBF(), Z=Z)
    assert isinstance(prior._resolved_solver(), DenseSolver)


def test_resolved_solver_passes_through_explicit_solver():
    Z = _toy_inducing()
    explicit = CGSolver()
    prior = SparseGPPrior(kernel=RBF(), Z=Z, solver=explicit)
    assert prior._resolved_solver() is explicit


def test_inducing_operator_is_psd_tagged():
    """The PSD tag is what unlocks Cholesky-routed solvers in gaussx."""
    import lineax as lx

    Z = _toy_inducing(3)
    op = SparseGPPrior(kernel=RBF(), Z=Z).inducing_operator()
    assert lx.is_positive_semidefinite(op)
