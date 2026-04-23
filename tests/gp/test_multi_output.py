"""Tests for multi-output kernel and inducing helpers."""

from __future__ import annotations

import jax.numpy as jnp

from pyrox.gp import (
    ICMKernel,
    LMCKernel,
    OILMMKernel,
    RBF,
    LMCKernel,
    MultiOutputInducingVariables,
    SharedInducingPoints,
)


def test_lmc_single_latent_matches_kronecker_special_case():
    X = jnp.array([[0.0], [1.0]])
    mixing = jnp.array([[1.0], [2.0]])
    kernel = RBF(init_variance=1.5, init_lengthscale=0.7)
    lmc = LMCKernel(kernels=(kernel,), mixing=mixing)

    K_xx = kernel(X, X)
    B = jnp.outer(mixing[:, 0], mixing[:, 0])

    assert jnp.allclose(lmc.coregionalization_matrix(0), B)
    assert jnp.allclose(lmc.full_covariance(X), jnp.kron(B, K_xx))


def test_lmc_output_covariance_is_symmetric():
    X = jnp.array([[0.0], [0.5], [1.0]])
    lmc = LMCKernel(
        kernels=(
            RBF(init_variance=1.0, init_lengthscale=0.5),
            RBF(init_variance=0.8, init_lengthscale=1.2),
        ),
        mixing=jnp.array([[1.0, 0.5], [0.25, -1.0]]),
    )

    K = lmc.full_covariance(X)
    assert K.shape == (6, 6)
    assert jnp.allclose(K, K.T)


def test_icm_adds_diagonal_kappa_to_coregionalization_matrix():
    X = jnp.array([[0.0], [1.0]])
    kernel = RBF(init_variance=1.2, init_lengthscale=0.4)
    mixing = jnp.array([[1.0, 0.0], [0.5, 1.0]])
    kappa = jnp.array([0.1, 0.2])
    icm = ICMKernel(kernel=kernel, mixing=mixing, kappa=kappa)

    B = mixing @ mixing.T + jnp.diag(kappa)
    assert jnp.allclose(icm.coregionalization_matrix(), B)
    assert jnp.allclose(icm.full_covariance(X), jnp.kron(B, kernel(X, X)))


def test_oilmm_projection_and_noise_behave_as_expected():
    X = jnp.array([[0.0], [1.0]])
    Y = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mixing = jnp.eye(2)
    kernels = (
        RBF(init_variance=1.0, init_lengthscale=0.5),
        RBF(init_variance=2.0, init_lengthscale=1.0),
    )
    oilmm = OILMMKernel(kernels=kernels, mixing=mixing, noise_variance=jnp.array(0.3))

    K1 = kernels[0](X, X)
    K2 = kernels[1](X, X)
    expected_signal = jnp.block([[K1, jnp.zeros_like(K1)], [jnp.zeros_like(K2), K2]])

    assert oilmm.is_orthogonal()
    assert oilmm.independent_gps() == kernels
    assert jnp.allclose(oilmm.project_observations(Y), Y)
    assert jnp.allclose(
        oilmm.full_covariance(X),
        expected_signal + 0.3 * jnp.eye(expected_signal.shape[0]),
    )


def test_shared_inducing_points_build_block_diagonal_covariance():
    Z = jnp.array([[-1.0], [1.0]])
    shared = SharedInducingPoints(locations=Z)
    kernels = (
        RBF(init_variance=1.0, init_lengthscale=0.5),
        RBF(init_variance=2.0, init_lengthscale=1.5),
    )

    K_uu = shared.K_uu(kernels)
    K1 = kernels[0](Z, Z)
    K2 = kernels[1](Z, Z)
    expected = jnp.block([[K1, jnp.zeros_like(K1)], [jnp.zeros_like(K2), K2]])

    assert K_uu.shape == (4, 4)
    assert jnp.allclose(K_uu, expected)


def test_multi_output_inducing_cross_covariance_matches_single_latent_case():
    Z = jnp.array([[-1.0], [1.0]])
    X = jnp.array([[0.0], [0.5], [1.0]])
    kernel = RBF(init_variance=1.0, init_lengthscale=0.8)
    inducing = MultiOutputInducingVariables(
        inducing=SharedInducingPoints(locations=Z),
        mixing=jnp.array([[2.0], [-1.0]]),
    )

    K_zx = kernel(Z, X)
    expected = jnp.concatenate([2.0 * K_zx, -1.0 * K_zx], axis=1)

    assert jnp.allclose(inducing.K_uu((kernel,)), kernel(Z, Z))
    assert jnp.allclose(inducing.K_uf(X, (kernel,)), expected)
