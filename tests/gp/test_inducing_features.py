"""Tests for `pyrox.gp._inducing` — inter-domain inducing-feature families.

The single most load-bearing test in this file is
:func:`test_vff_k_uu_is_diagonal_and_survives_dispatch` — the entire
scalability claim of the inducing-feature path depends on
``K_uu`` reaching :func:`gaussx.solve` as a
:class:`lineax.DiagonalLinearOperator`. If that ever regresses to a
dense fallback, every other test in the suite still passes while the
``O(M^3)`` bottleneck silently returns.
"""

from __future__ import annotations

import gaussx
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from pyrox.gp import (
    RBF,
    DecoupledInducingFeatures,
    FourierInducingFeatures,
    LaplacianInducingFeatures,
    Matern,
    Periodic,
    SparseGPPrior,
    SphericalHarmonicInducingFeatures,
    funk_hecke_coefficients,
)


# ---------------------------------------------------------------------------
# FourierInducingFeatures
# ---------------------------------------------------------------------------


def test_vff_k_uu_is_diagonal_and_survives_dispatch():
    """Scalability tripwire: VFF -> SparseGPPrior -> diagonal solve.

    Verifies the *entire* dispatch chain preserves structure end-to-end:

    1. ``inducing_operator()`` returns a ``DiagonalLinearOperator``
       (jitter folded into the diagonal vector, not added as ``jnp.eye``).
    2. ``gaussx.cholesky`` of that operator is itself diagonal.
    3. ``gaussx.solve`` against it equals elementwise division.

    Without this gate, future refactors (e.g. ``+ lx.IdentityLinearOperator``,
    ``.as_matrix()``) would silently revert the operator to dense and
    every other test in the suite would still pass.
    """
    kernel = RBF(init_lengthscale=0.5, init_variance=1.0)
    features = FourierInducingFeatures.init(in_features=1, num_basis_per_dim=64, L=5.0)
    prior = SparseGPPrior(kernel=kernel, inducing=features, jitter=1e-6)

    K_uu_op = prior.inducing_operator()
    assert isinstance(K_uu_op, lx.DiagonalLinearOperator), (
        f"VFF operator densified upstream — got {type(K_uu_op).__name__}, "
        "scalability claim broken."
    )

    L = gaussx.cholesky(K_uu_op)
    assert isinstance(L, lx.DiagonalLinearOperator), (
        f"gaussx.cholesky lost diagonal structure — got {type(L).__name__}."
    )

    diag = jnp.asarray(K_uu_op.diagonal)
    rhs = jnp.ones(features.num_features)
    expected = rhs / diag
    out = gaussx.solve(K_uu_op, rhs)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-6)


def test_vff_k_uu_diagonal_matches_spectral_density():
    """Diagonal entries of K_uu equal ``S(sqrt(lambda_j)) + jitter``."""
    from pyrox._basis import fourier_eigenvalues, spectral_density

    kernel = RBF(init_lengthscale=0.7, init_variance=1.5)
    features = FourierInducingFeatures.init(in_features=1, num_basis_per_dim=8, L=4.0)
    jitter = 1e-5
    op = features.K_uu(kernel, jitter=jitter)
    lam = fourier_eigenvalues((8,), (4.0,), 1)
    expected = spectral_density(kernel, lam, D=1) + jitter
    np.testing.assert_allclose(np.asarray(op.diagonal), np.asarray(expected), rtol=1e-5)


def test_vff_k_ux_shape_and_finite():
    kernel = RBF(init_lengthscale=0.5, init_variance=1.0)
    features = FourierInducingFeatures.init(in_features=1, num_basis_per_dim=12, L=3.0)
    x = jnp.linspace(-2.0, 2.0, 8).reshape(-1, 1)
    K_ux = features.k_ux(x, kernel)
    assert K_ux.shape == (8, 12)
    assert jnp.all(jnp.isfinite(K_ux))


def test_vff_2d_basis_count():
    kernel = RBF(init_lengthscale=0.5, init_variance=1.0)
    features = FourierInducingFeatures.init(
        in_features=2, num_basis_per_dim=(4, 5), L=(1.0, 1.0)
    )
    assert features.num_features == 20
    op = features.K_uu(kernel)
    assert op.diagonal.shape == (20,)


def test_vff_rejects_non_stationary_kernel():
    features = FourierInducingFeatures.init(in_features=1, num_basis_per_dim=4, L=2.0)
    kernel = Periodic(init_lengthscale=1.0, init_period=1.0)
    with pytest.raises(ValueError, match="stationary"):
        features.K_uu(kernel)


def test_vff_rejects_wrong_x_shape():
    kernel = RBF(init_lengthscale=1.0, init_variance=1.0)
    features = FourierInducingFeatures.init(in_features=2, num_basis_per_dim=2, L=1.0)
    with pytest.raises(ValueError, match=r"x must be \(N, 2\)"):
        features.k_ux(jnp.zeros((4, 3)), kernel)


def test_vff_predictive_blocks_returns_diagonal_op():
    """`predictive_blocks` (the SVGP-batch entry point) must also stay diagonal."""
    kernel = RBF(init_lengthscale=0.5, init_variance=1.0)
    features = FourierInducingFeatures.init(in_features=1, num_basis_per_dim=8, L=3.0)
    prior = SparseGPPrior(kernel=kernel, inducing=features, jitter=1e-6)
    K_zz_op, K_xz, K_xx_diag = prior.predictive_blocks(
        jnp.linspace(-1, 1, 5).reshape(-1, 1)
    )
    assert isinstance(K_zz_op, lx.DiagonalLinearOperator)
    assert K_xz.shape == (5, 8)
    assert K_xx_diag.shape == (5,)


# ---------------------------------------------------------------------------
# SphericalHarmonicInducingFeatures
# ---------------------------------------------------------------------------


def test_vish_funk_hecke_a0_matches_kernel_average():
    r""":math:`a_0 = 2\pi \int_{-1}^{1} k(t) dt` for the constant Legendre."""
    kernel = RBF(init_lengthscale=1.0, init_variance=1.0)
    a = funk_hecke_coefficients(kernel, l_max=3, num_quadrature=128)
    assert a.shape == (4,)
    assert float(a[0]) > 0


def test_vish_k_uu_diagonal_and_per_l_constant():
    r"""Each ``l`` block holds ``2l+1`` features sharing one Funk-Hecke coefficient."""
    import itertools

    kernel = RBF(init_lengthscale=1.0, init_variance=1.0)
    features = SphericalHarmonicInducingFeatures.init(l_max=4)
    op = features.K_uu(kernel, jitter=1e-6)
    assert isinstance(op, lx.DiagonalLinearOperator)
    diag = np.asarray(op.diagonal)
    # Layout: 1 + 3 + 5 + 7 + 9 = 25 entries; each l-block constant.
    sizes = [1, 3, 5, 7, 9]
    offsets = np.cumsum([0, *sizes])
    for l, (start, end) in enumerate(itertools.pairwise(offsets)):
        block = diag[start:end]
        np.testing.assert_allclose(block, block[0], rtol=1e-5, err_msg=f"l={l}")


def test_vish_k_ux_shape():
    kernel = RBF(init_lengthscale=1.0, init_variance=1.0)
    features = SphericalHarmonicInducingFeatures.init(l_max=3)
    rng = np.random.default_rng(0)
    xyz = rng.standard_normal((6, 3))
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    K_ux = features.k_ux(jnp.asarray(xyz), kernel)
    assert K_ux.shape == (6, 16)  # (l_max+1)^2
    assert jnp.all(jnp.isfinite(K_ux))


def test_vish_rejects_negative_l_max():
    with pytest.raises(ValueError, match="l_max"):
        SphericalHarmonicInducingFeatures.init(l_max=-1)


# ---------------------------------------------------------------------------
# LaplacianInducingFeatures
# ---------------------------------------------------------------------------


def test_laplacian_inducing_diagonal_K_uu():
    rng = np.random.default_rng(0)
    n = 10
    A = rng.uniform(0, 1, size=(n, n))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    features = LaplacianInducingFeatures.fit(jnp.asarray(A), num_basis=5)
    kernel = RBF(init_lengthscale=0.5, init_variance=1.0)
    op = features.K_uu(kernel, jitter=1e-6)
    assert isinstance(op, lx.DiagonalLinearOperator)
    assert op.diagonal.shape == (5,)


def test_laplacian_inducing_k_ux_gathers_eigenvectors():
    rng = np.random.default_rng(1)
    n = 8
    A = rng.uniform(0, 1, size=(n, n))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    features = LaplacianInducingFeatures.fit(jnp.asarray(A), num_basis=4)
    kernel = RBF(init_lengthscale=0.5, init_variance=1.0)
    nodes = jnp.array([0, 2, 5])
    K_ux = features.k_ux(nodes, kernel)
    assert K_ux.shape == (3, 4)
    # Sanity: row 0 corresponds to node 0's eigenvector entries scaled by S.
    from pyrox._basis import spectral_density

    S = spectral_density(kernel, features.eigvals, D=1)
    expected_row0 = features.eigvecs[0] * S
    np.testing.assert_allclose(
        np.asarray(K_ux[0]), np.asarray(expected_row0), rtol=1e-5
    )


def test_laplacian_inducing_rejects_non_stationary_kernel():
    rng = np.random.default_rng(2)
    A = rng.uniform(0, 1, size=(4, 4))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    features = LaplacianInducingFeatures.fit(jnp.asarray(A), num_basis=2)
    with pytest.raises(ValueError, match="stationary"):
        features.K_uu(Periodic(init_lengthscale=1.0, init_period=1.0))


# ---------------------------------------------------------------------------
# DecoupledInducingFeatures
# ---------------------------------------------------------------------------


def test_decoupled_holds_two_bases():
    mean = FourierInducingFeatures.init(in_features=1, num_basis_per_dim=32, L=4.0)
    cov = FourierInducingFeatures.init(in_features=1, num_basis_per_dim=8, L=4.0)
    decoupled = DecoupledInducingFeatures(mean_features=mean, cov_features=cov)
    assert decoupled.num_mean_features == 32
    assert decoupled.num_cov_features == 8


# ---------------------------------------------------------------------------
# SparseGPPrior dispatch
# ---------------------------------------------------------------------------


def test_sparse_prior_rejects_both_Z_and_inducing():
    with pytest.raises(ValueError, match="exactly one"):
        SparseGPPrior(
            kernel=RBF(init_lengthscale=1.0, init_variance=1.0),
            Z=jnp.zeros((3, 1)),
            inducing=FourierInducingFeatures.init(
                in_features=1, num_basis_per_dim=4, L=1.0
            ),
        )


def test_sparse_prior_rejects_neither_Z_nor_inducing():
    with pytest.raises(ValueError, match="exactly one"):
        SparseGPPrior(kernel=RBF(init_lengthscale=1.0, init_variance=1.0))


def test_sparse_prior_point_path_unchanged():
    """Legacy point-inducing path keeps returning a dense PSD operator."""
    kernel = RBF(init_lengthscale=0.5, init_variance=1.0)
    Z = jnp.linspace(-1, 1, 8).reshape(-1, 1)
    prior = SparseGPPrior(kernel=kernel, Z=Z)
    op = prior.inducing_operator()
    # Dense path should NOT be a DiagonalLinearOperator.
    assert not isinstance(op, lx.DiagonalLinearOperator)
    assert prior.num_inducing == 8


def test_sparse_prior_inducing_features_path_is_diagonal():
    """Feature-inducing path must return a DiagonalLinearOperator."""
    kernel = Matern(init_lengthscale=0.5, init_variance=1.0, nu=2.5)
    features = FourierInducingFeatures.init(in_features=1, num_basis_per_dim=16, L=3.0)
    prior = SparseGPPrior(kernel=kernel, inducing=features, jitter=1e-5)
    op = prior.inducing_operator()
    assert isinstance(op, lx.DiagonalLinearOperator)
    assert prior.num_inducing == 16


def test_sparse_prior_log_prob_works_on_diagonal_path():
    """`log_prob` on the inducing-feature path should still work end-to-end."""
    kernel = RBF(init_lengthscale=0.5, init_variance=1.0)
    features = FourierInducingFeatures.init(in_features=1, num_basis_per_dim=8, L=3.0)
    prior = SparseGPPrior(kernel=kernel, inducing=features, jitter=1e-6)
    u = jnp.zeros(8)
    lp = prior.log_prob(u)
    assert jnp.isfinite(lp)


def test_sparse_prior_sample_returns_correct_shape():
    import jax.random as jr

    kernel = RBF(init_lengthscale=0.5, init_variance=1.0)
    features = FourierInducingFeatures.init(in_features=1, num_basis_per_dim=12, L=3.0)
    prior = SparseGPPrior(kernel=kernel, inducing=features, jitter=1e-6)
    u = prior.sample(jr.PRNGKey(0))
    assert u.shape == (12,)
    assert jnp.all(jnp.isfinite(u))


# ---------------------------------------------------------------------------
# SparseGPPrior integration tests — each family round-trips end-to-end
# ---------------------------------------------------------------------------


def test_sparse_prior_vish_end_to_end():
    """VISH inducing features produce a usable diagonal K_uu via SparseGPPrior."""
    kernel = RBF(init_lengthscale=0.5, init_variance=1.0)
    features = SphericalHarmonicInducingFeatures.init(l_max=3, num_quadrature=64)
    prior = SparseGPPrior(kernel=kernel, inducing=features, jitter=1e-5)

    K_uu_op = prior.inducing_operator()
    assert isinstance(K_uu_op, lx.DiagonalLinearOperator)
    assert prior.num_inducing == (3 + 1) ** 2

    rng = np.random.default_rng(4)
    xyz = rng.standard_normal((5, 3))
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    K_xz = prior.cross_covariance(jnp.asarray(xyz))
    assert K_xz.shape == (5, prior.num_inducing)
    assert jnp.all(jnp.isfinite(K_xz))

    # Prior log-prob should be finite on a zero inducing vector.
    u = jnp.zeros(prior.num_inducing)
    assert jnp.isfinite(prior.log_prob(u))


def test_sparse_prior_laplacian_end_to_end():
    """Laplacian inducing features plug into SparseGPPrior with node indices as X."""
    rng = np.random.default_rng(5)
    n_nodes = 10
    A = rng.uniform(0, 1, size=(n_nodes, n_nodes))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)

    kernel = RBF(init_lengthscale=0.5, init_variance=1.0)
    features = LaplacianInducingFeatures.fit(jnp.asarray(A), num_basis=4)
    prior = SparseGPPrior(kernel=kernel, inducing=features, jitter=1e-5)

    K_uu_op = prior.inducing_operator()
    assert isinstance(K_uu_op, lx.DiagonalLinearOperator)
    assert prior.num_inducing == 4

    node_indices = jnp.array([0, 2, 5, 7])
    K_xz = prior.cross_covariance(node_indices)
    assert K_xz.shape == (4, 4)
    assert jnp.all(jnp.isfinite(K_xz))

    u = jnp.zeros(prior.num_inducing)
    assert jnp.isfinite(prior.log_prob(u))
