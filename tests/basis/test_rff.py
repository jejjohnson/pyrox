"""Tests for the shared random-Fourier-feature prior-draw helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pyrox._basis import draw_rff_cosine_basis, evaluate_rff_cosine_paths
from pyrox.gp import RBF, Matern, Periodic


def test_rff_cosine_paths_reconstruct_rbf_kernel():
    kernel = RBF(init_variance=1.3, init_lengthscale=0.4)
    X = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    variance, lengthscale, omega, phase, weights = draw_rff_cosine_basis(
        kernel,
        jax.random.PRNGKey(0),
        n_paths=4096,
        n_features=32,
        in_features=1,
        dtype=jnp.float32,
    )
    paths = evaluate_rff_cosine_paths(
        X,
        variance=variance,
        lengthscale=lengthscale,
        omega=omega,
        phase=phase,
        weights=weights,
    )
    empirical = paths.T @ paths / paths.shape[0]
    exact = kernel(X, X)
    assert jnp.allclose(empirical, exact, atol=0.15)


def test_rff_cosine_paths_reconstruct_matern_1d():
    kernel = Matern(init_variance=1.0, init_lengthscale=0.5, nu=1.5)
    X = jnp.linspace(-1.0, 1.0, 5).reshape(-1, 1)
    variance, lengthscale, omega, phase, weights = draw_rff_cosine_basis(
        kernel,
        jax.random.PRNGKey(1),
        n_paths=4096,
        n_features=64,
        in_features=1,
        dtype=jnp.float32,
    )
    paths = evaluate_rff_cosine_paths(
        X,
        variance=variance,
        lengthscale=lengthscale,
        omega=omega,
        phase=phase,
        weights=weights,
    )
    empirical = paths.T @ paths / paths.shape[0]
    exact = kernel(X, X)
    # Matern RFF with coord-wise t-draws converges more slowly than RBF,
    # so the tolerance is loose; the test exists to catch sign / scale
    # regressions (e.g. wrong spectral density).
    assert jnp.allclose(empirical, exact, atol=0.3)


def test_rff_cosine_paths_rejects_unsupported_kernel():
    kernel = Periodic()
    with pytest.raises(NotImplementedError, match="RBF and Matern"):
        draw_rff_cosine_basis(
            kernel,
            jax.random.PRNGKey(0),
            n_paths=1,
            n_features=4,
            in_features=1,
            dtype=jnp.float32,
        )


@pytest.mark.parametrize("bad_field", ["n_paths", "n_features"])
def test_rff_cosine_paths_rejects_nonpositive_counts(bad_field):
    kwargs = dict(n_paths=2, n_features=4, in_features=1, dtype=jnp.float32)
    kwargs[bad_field] = 0
    with pytest.raises(ValueError):
        draw_rff_cosine_basis(RBF(), jax.random.PRNGKey(0), **kwargs)
