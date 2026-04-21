"""Tests for `pyrox._basis._fourier`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyrox._basis import (
    fourier_basis,
    fourier_basis_1d,
    fourier_eigenvalues,
    fourier_eigenvalues_1d,
)


def test_fourier_basis_1d_shape():
    x = jnp.linspace(-1.0, 1.0, 7)
    Phi = fourier_basis_1d(x, num_basis=4, L=1.0)
    assert Phi.shape == (7, 4)


def test_fourier_basis_1d_orthonormal_on_uniform_grid():
    """Riemann sum of `Phi^T Phi * dx` should approximate `I_M`."""
    L = 5.0
    M = 8
    N = 4001
    x = jnp.linspace(-L, L, N)
    dx = (2 * L) / (N - 1)
    Phi = fourier_basis_1d(x, M, L)
    gram = Phi.T @ Phi * dx
    # Trapezoid is exact for sinusoids on a uniformly-spaced grid that includes
    # the endpoints (where Phi vanishes), so machine precision is fine.
    assert jnp.allclose(gram, jnp.eye(M), atol=1e-5)


def test_fourier_eigenvalues_1d_ascending():
    L = 2.5
    lam = fourier_eigenvalues_1d(num_basis=10, L=L)
    assert jnp.all(jnp.diff(lam) > 0)
    # First eigenvalue lam_1 = (pi / (2L))^2
    expected_first = (jnp.pi / (2 * L)) ** 2
    assert jnp.allclose(lam[0], expected_first, rtol=1e-5)


@pytest.mark.parametrize("num_basis", [0, -1])
def test_fourier_basis_1d_rejects_bad_num_basis(num_basis):
    with pytest.raises(ValueError, match="num_basis"):
        fourier_basis_1d(jnp.zeros(3), num_basis, L=1.0)


def test_fourier_basis_1d_rejects_nonpositive_L():
    with pytest.raises(ValueError, match="L"):
        fourier_basis_1d(jnp.zeros(3), 4, L=0.0)


def test_fourier_basis_2d_shape():
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.uniform(-1, 1, size=(15, 2)))
    Phi, lam = fourier_basis(x, num_basis_per_dim=(3, 4), L=(1.0, 1.0))
    assert Phi.shape == (15, 12)
    assert lam.shape == (12,)


def test_fourier_basis_2d_eigenvalues_are_sums():
    """Tensor-product eigenvalues are sums of per-dim 1D eigenvalues."""
    L = (1.0, 2.0)
    M = (3, 2)
    x = jnp.zeros((1, 2))
    _, lam = fourier_basis(x, num_basis_per_dim=M, L=L)
    lam1d_x = fourier_eigenvalues_1d(M[0], L[0])
    lam1d_y = fourier_eigenvalues_1d(M[1], L[1])
    expected = jnp.asarray(
        [
            float(lam1d_x[i]) + float(lam1d_y[j])
            for i in range(M[0])
            for j in range(M[1])
        ]
    )
    assert jnp.allclose(lam, expected, rtol=1e-5)


def test_fourier_basis_2d_factorizes_as_product():
    """A 2D basis function evaluates as the product of two 1D basis functions."""
    L = (1.5, 2.5)
    M = (4, 3)
    x = jnp.asarray([[0.3, -0.7]])
    Phi, _ = fourier_basis(x, num_basis_per_dim=M, L=L)
    Phi_x = fourier_basis_1d(jnp.asarray([0.3]), M[0], L[0])  # (1, 4)
    Phi_y = fourier_basis_1d(jnp.asarray([-0.7]), M[1], L[1])  # (1, 3)
    expected = jnp.einsum("ni,nj->nij", Phi_x, Phi_y).reshape(1, M[0] * M[1])
    assert jnp.allclose(Phi, expected, rtol=1e-5)


def test_fourier_basis_broadcasts_scalar_inputs():
    rng = np.random.RandomState(1)
    x = jnp.asarray(rng.uniform(-1, 1, size=(8, 3)))
    Phi, lam = fourier_basis(x, num_basis_per_dim=2, L=1.0)  # both broadcast
    assert Phi.shape == (8, 8)  # 2^3
    assert lam.shape == (8,)


def test_fourier_eigenvalues_matches_basis_path():
    """`fourier_eigenvalues` agrees with `fourier_basis` second return value."""
    L = (1.0, 2.0, 0.5)
    M = (3, 2, 4)
    x = jnp.zeros((1, 3))
    _, lam_via_basis = fourier_basis(x, M, L)
    lam_direct = fourier_eigenvalues(M, L, D=3)
    assert jnp.allclose(lam_via_basis, lam_direct)


def test_fourier_basis_jits_under_jit():
    @jax.jit
    def f(x):
        return fourier_basis_1d(x, 5, 1.0).sum()

    x = jnp.linspace(-1, 1, 11)
    out = f(x)
    assert jnp.isfinite(out)


def test_fourier_basis_1d_matches_closed_form():
    """Independent numpy reference: phi_j(x) = sin(j pi (x+L)/(2L)) / sqrt(L)."""
    L = 1.7
    M = 6
    x_np = np.linspace(-L, L, 21)
    j = np.arange(1, M + 1)
    expected = np.sin(np.outer(x_np + L, j) * (np.pi / (2 * L))) / np.sqrt(L)
    Phi = fourier_basis_1d(jnp.asarray(x_np), M, L)
    np.testing.assert_allclose(np.asarray(Phi), expected, atol=1e-5)


def test_fourier_eigenvalues_1d_matches_closed_form():
    """Reference: lambda_j = (j pi / (2L))^2."""
    L = 0.8
    M = 12
    expected = (np.arange(1, M + 1) * np.pi / (2 * L)) ** 2
    lam = fourier_eigenvalues_1d(M, L)
    np.testing.assert_allclose(np.asarray(lam), expected, rtol=1e-5)


def test_fourier_basis_zero_at_boundary():
    """Dirichlet BC: phi_j(+/-L) = 0 for all j."""
    L = 2.3
    M = 7
    x = jnp.asarray([-L, L])
    Phi = fourier_basis_1d(x, M, L)
    assert jnp.allclose(Phi, 0.0, atol=1e-5)


def test_fourier_basis_2d_matches_independent_kron():
    """2D basis equals the Kronecker product of two 1D bases (per row)."""
    L = (1.0, 1.5)
    M = (3, 2)
    x_np = np.array([[0.2, -0.3], [-0.5, 0.7]])
    Phi_x = np.sin(
        np.outer(x_np[:, 0] + L[0], np.arange(1, M[0] + 1)) * (np.pi / (2 * L[0]))
    ) / np.sqrt(L[0])
    Phi_y = np.sin(
        np.outer(x_np[:, 1] + L[1], np.arange(1, M[1] + 1)) * (np.pi / (2 * L[1]))
    ) / np.sqrt(L[1])
    expected = np.einsum("ni,nj->nij", Phi_x, Phi_y).reshape(2, M[0] * M[1])
    Phi, _ = fourier_basis(jnp.asarray(x_np), M, L)
    np.testing.assert_allclose(np.asarray(Phi), expected, atol=1e-5)
