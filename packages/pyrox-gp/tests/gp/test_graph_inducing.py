"""Graph heat / Matérn inducing features against kernellib's graph kernels (P2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import kernellib as kl
import numpy as np
import pytest
from pyrox_gp import RBF, LaplacianInducingFeatures, Matern


jax.config.update("jax_enable_x64", True)


def _adjacency(n=12, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.uniform(0, 1, size=(n, n)) * (rng.uniform(size=(n, n)) < 0.4)
    A = np.triu(A, 1)
    A = A + A.T + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    return jnp.asarray(A)


def _nystrom(features, kernel):
    nodes = jnp.arange(features.n_nodes)
    kux = features.k_ux(nodes, kernel)
    kuu = features.K_uu(kernel, jitter=0.0).diagonal
    return (kux / kuu) @ kux.T


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_full_basis_recovers_matern_graph_kernel(nu):
    A = _adjacency()
    feats = LaplacianInducingFeatures.fit(A, A.shape[0])
    K = _nystrom(feats, Matern(nu=nu, init_lengthscale=0.7, init_variance=1.3))
    ref = kl.matern_graph_kernel(A, nu=nu, lengthscale=0.7, variance=1.3)
    np.testing.assert_allclose(np.asarray(K), np.asarray(ref), atol=1e-8)


def test_full_basis_recovers_diffusion_kernel():
    A = _adjacency()
    n = A.shape[0]
    feats = LaplacianInducingFeatures.fit(A, n)
    K = _nystrom(feats, RBF(init_lengthscale=0.8, init_variance=2.0))
    ref = kl.diffusion_kernel(A, beta=0.8**2 / 2)
    ref = 2.0 * n * ref / jnp.trace(ref)  # average marginal variance 2.0
    np.testing.assert_allclose(np.asarray(K), np.asarray(ref), atol=1e-8)


@pytest.mark.parametrize(
    "kernel", [RBF(init_lengthscale=0.6, init_variance=1.7), Matern(nu=1.5)]
)
def test_average_marginal_variance_is_sigma2(kernel):
    A = _adjacency()
    feats = LaplacianInducingFeatures.fit(A, A.shape[0])
    var = kernel.init_variance if hasattr(kernel, "init_variance") else 1.0
    np.testing.assert_allclose(
        float(jnp.mean(jnp.diag(_nystrom(feats, kernel)))), var, rtol=1e-8
    )


def test_kronecker_matches_dense_on_grid():
    g = kl.grid_graph((6, 5))
    dense = LaplacianInducingFeatures.fit(
        g, 8, normalization="unnormalized", method="dense"
    )
    kron = LaplacianInducingFeatures.fit(
        g, 8, normalization="unnormalized", method="kronecker"
    )
    np.testing.assert_allclose(dense.eigvals, kron.eigvals, atol=1e-8)
    kernel = Matern(nu=1.5, init_lengthscale=2.0)
    # Eigenvectors of repeated eigenvalues differ by a rotation, so compare
    # the projector-weighted kernel rather than the vectors themselves.
    np.testing.assert_allclose(
        _nystrom(dense, kernel), _nystrom(kron, kernel), atol=1e-8
    )


@pytest.mark.parametrize(
    ("flag", "normalization"), [(True, "symmetric"), (False, "unnormalized")]
)
def test_normalized_alias_warns_and_maps(flag, normalization):
    A = _adjacency()
    with pytest.warns(DeprecationWarning, match="normalized="):
        old = LaplacianInducingFeatures.fit(A, 4, normalized=flag)
    new = LaplacianInducingFeatures.fit(A, 4, normalization=normalization)
    np.testing.assert_allclose(old.eigvals, new.eigvals, atol=1e-10)
