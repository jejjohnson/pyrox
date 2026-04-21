"""Tests for `pyrox._basis._laplacian`."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from pyrox._basis import graph_laplacian_eigpairs


def _path_graph_adjacency(n: int) -> np.ndarray:
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A


def test_path_graph_combinatorial_eigvals():
    """Combinatorial Laplacian of an n-path has known eigenvalues 2(1 - cos(k*pi/n))."""
    n = 8
    A = _path_graph_adjacency(n)
    eigvals, eigvecs = graph_laplacian_eigpairs(jnp.asarray(A), n, normalized=False)
    expected = 2.0 * (1.0 - np.cos(np.pi * np.arange(n) / n))
    assert eigvals.shape == (n,)
    assert eigvecs.shape == (n, n)
    assert jnp.allclose(jnp.sort(eigvals), jnp.sort(jnp.asarray(expected)), atol=1e-5)


def test_eigvecs_are_orthonormal():
    n = 6
    A = _path_graph_adjacency(n)
    _, eigvecs = graph_laplacian_eigpairs(jnp.asarray(A), n)
    gram = eigvecs.T @ eigvecs
    assert jnp.allclose(gram, jnp.eye(n), atol=1e-5)


def test_smallest_normalized_eigval_is_zero():
    """Connected graph: smallest normalized Laplacian eigenvalue is 0."""
    A = jnp.asarray(_path_graph_adjacency(5))
    eigvals, _ = graph_laplacian_eigpairs(A, 3, normalized=True)
    assert eigvals.shape == (3,)
    assert float(eigvals[0]) < 1e-8


def test_eigenvals_sorted_ascending():
    rng = np.random.default_rng(7)
    n = 10
    A = rng.uniform(0, 1, size=(n, n))
    A = 0.5 * (A + A.T)
    eigvals, _ = graph_laplacian_eigpairs(jnp.asarray(A), 5)
    assert jnp.all(jnp.diff(eigvals) >= -1e-10)


def test_rejects_negative_adjacency():
    A = -jnp.ones((3, 3))
    with pytest.raises(ValueError, match="non-negative"):
        graph_laplacian_eigpairs(A, 2)


def test_rejects_non_square():
    A = jnp.zeros((3, 4))
    with pytest.raises(ValueError, match="square"):
        graph_laplacian_eigpairs(A, 2)


def test_rejects_bad_num_basis():
    A = jnp.eye(4)
    with pytest.raises(ValueError, match="num_basis"):
        graph_laplacian_eigpairs(A, 0)
    with pytest.raises(ValueError, match="num_basis"):
        graph_laplacian_eigpairs(A, 5)


def test_matches_scipy_eigh_directly():
    """Cross-check against an independent scipy.linalg.eigh on the same Laplacian."""
    from scipy.linalg import eigh

    rng = np.random.default_rng(101)
    n = 12
    A = rng.uniform(0, 1, size=(n, n))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)

    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, deg ** (-0.5), 0.0)
    L_norm = np.eye(n) - (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]
    L_norm = 0.5 * (L_norm + L_norm.T)
    expected_eigvals, _ = eigh(L_norm)

    eigvals, eigvecs = graph_laplacian_eigpairs(jnp.asarray(A), n, normalized=True)
    np.testing.assert_allclose(np.asarray(eigvals), expected_eigvals[:n], atol=1e-6)
    # Each returned eigenvector satisfies L v = lambda v.
    Lv = L_norm @ np.asarray(eigvecs)
    lam_v = np.asarray(eigvals)[None, :] * np.asarray(eigvecs)
    np.testing.assert_allclose(Lv, lam_v, atol=1e-6)


def test_combinatorial_path_eigvecs_match_dct_modes():
    r"""Path-graph combinatorial Laplacian eigenvectors are the DCT-II basis.

    Eigenvector ``k`` of ``L = D - A`` for the n-path is
    :math:`v_k(j) \propto \cos((j + 1/2) k \pi / n)` (up to an overall sign).
    """
    n = 16
    A = _path_graph_adjacency(n)
    eigvals, eigvecs = graph_laplacian_eigpairs(jnp.asarray(A), n, normalized=False)
    # Build DCT-II reference (cosine modes), normalize columns.
    j = np.arange(n)
    expected = np.stack([np.cos((j + 0.5) * k * np.pi / n) for k in range(n)], axis=-1)
    expected /= np.linalg.norm(expected, axis=0, keepdims=True)
    # Each pyrox eigenvector matches one DCT mode up to sign; pair by eigenvalue.
    sort_pyrox = np.argsort(np.asarray(eigvals))
    eigvecs_sorted = np.asarray(eigvecs)[:, sort_pyrox]
    for k in range(n):
        # Pick the DCT mode whose eigenvalue (2 - 2 cos(k pi / n)) matches.
        cos = np.dot(eigvecs_sorted[:, k], expected[:, k])
        expected_signed = expected[:, k] * np.sign(cos)
        np.testing.assert_allclose(eigvecs_sorted[:, k], expected_signed, atol=1e-5)
