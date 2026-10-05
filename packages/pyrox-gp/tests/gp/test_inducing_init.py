"""Tests for `pyrox_gp.init_inducing` (P3)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from pyrox_gp import RBF, init_inducing


jax.config.update("jax_enable_x64", True)


def _clustered(n=400):
    """Five clusters of very different sizes on [0, 1]^2."""
    centres = jnp.array([[0.1, 0.1], [0.9, 0.2], [0.5, 0.5], [0.2, 0.9], [0.8, 0.8]])
    sizes = [250, 80, 40, 20, 10]
    keys = jr.split(jr.key(0), len(sizes))
    pts = [
        c + 0.04 * jr.normal(k, (m, 2))
        for c, m, k in zip(centres, sizes, keys, strict=True)
    ]
    X = jnp.concatenate(pts)[:n]
    y = jnp.sin(6 * X[:, 0]) + jnp.cos(5 * X[:, 1])
    return X, y + 0.1 * jr.normal(jr.key(1), y.shape)


@pytest.mark.parametrize("method", ["uniform", "rpcholesky", "greedy", "leverage"])
def test_rows_of_x_with_the_right_shape(method):
    X, _ = _clustered()
    Z = init_inducing(
        X, 20, kernel=RBF(init_lengthscale=0.2), method=method, key=jr.key(2)
    )
    assert Z.shape == (20, 2)
    Xn, Zn = np.asarray(X), np.asarray(Z)
    hits = [np.flatnonzero(np.all(Xn == z, axis=1)) for z in Zn]
    assert all(h.size for h in hits)
    assert len({int(h[0]) for h in hits}) == 20  # distinct rows


def test_greedy_is_deterministic():
    X, _ = _clustered()
    kernel = RBF(init_lengthscale=0.2)
    a = init_inducing(X, 15, kernel=kernel, method="greedy", key=jr.key(0))
    b = init_inducing(X, 15, kernel=kernel, method="greedy", key=jr.key(7))
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_kernel_none_only_with_uniform():
    X, _ = _clustered()
    assert init_inducing(X, 5, kernel=None, method="uniform", key=jr.key(0)).shape == (
        5,
        2,
    )
    with pytest.raises(ValueError, match="needs the kernel"):
        init_inducing(X, 5, kernel=None, key=jr.key(0))


def _collapsed_elbo(X, y, Z, lengthscale=0.2, noise=0.01):
    """Titsias' bound: the SVGP ELBO at its optimal q(u), Gaussian likelihood."""

    def k(a, b):
        d2 = jnp.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return jnp.exp(-0.5 * d2 / lengthscale**2)

    Kuu = k(Z, Z) + 1e-8 * jnp.eye(Z.shape[0])
    Kuf = k(Z, X)
    L = jnp.linalg.cholesky(Kuu)
    A = jax.scipy.linalg.solve_triangular(L, Kuf, lower=True)
    Q = A.T @ A
    n = X.shape[0]
    cov = Q + noise * jnp.eye(n)
    logp = jax.scipy.stats.multivariate_normal.logpdf(y, jnp.zeros(n), cov)
    return float(logp - 0.5 * (n - jnp.trace(Q)) / noise)


@pytest.mark.slow
def test_rpcholesky_beats_uniform_on_clustered_data():
    # The collapsed bound is the SVGP ELBO at its optimum, so it measures the
    # inducing set itself, not an optimiser's progress; fixed keys.
    X, y = _clustered()
    kernel = RBF(init_lengthscale=0.2)
    rp = init_inducing(X, 15, kernel=kernel, method="rpcholesky", key=jr.key(3))
    uni = init_inducing(X, 15, kernel=kernel, method="uniform", key=jr.key(3))
    assert _collapsed_elbo(X, y, rp) >= _collapsed_elbo(X, y, uni)
