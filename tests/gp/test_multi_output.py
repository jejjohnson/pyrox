"""Tests for multi-output kernel and inducing helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from gaussx import is_block_diagonal, is_kronecker, oilmm_project, solve

from pyrox.gp import (
    RBF,
    ICMKernel,
    LMCKernel,
    MultiOutputInducingVariables,
    OILMMKernel,
    SharedInducingPoints,
)


# ---------------------------------------------------------------------------
# LMCKernel
# ---------------------------------------------------------------------------


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


def test_lmc_handles_d_gt_1():
    X = jax.random.uniform(jax.random.PRNGKey(0), (4, 3))
    lmc = LMCKernel(
        kernels=(
            RBF(init_variance=1.0, init_lengthscale=0.5),
            RBF(init_variance=0.8, init_lengthscale=1.2),
        ),
        mixing=jnp.array([[1.0, 0.5], [0.25, -1.0]]),
    )

    K = lmc.full_covariance(X)
    assert K.shape == (8, 8)
    assert jnp.allclose(K, K.T)


def test_lmc_diag_matches_full_covariance_diagonal():
    X = jnp.array([[0.0], [0.5], [1.0]])
    lmc = LMCKernel(
        kernels=(
            RBF(init_variance=1.0, init_lengthscale=0.5),
            RBF(init_variance=0.8, init_lengthscale=1.2),
        ),
        mixing=jnp.array([[1.0, 0.5], [0.25, -1.0]]),
    )

    dense = lmc.full_covariance(X)
    P, N = lmc.num_outputs, X.shape[0]
    expected = jnp.diag(dense).reshape(P, N).T  # (N, P)
    assert jnp.allclose(lmc.diag(X), expected)


def test_lmc_cross_covariance_operator_is_sum_kronecker():
    X = jnp.array([[0.0], [0.5], [1.0]])
    lmc = LMCKernel(
        kernels=(
            RBF(init_variance=1.0, init_lengthscale=0.5),
            RBF(init_variance=0.8, init_lengthscale=1.2),
        ),
        mixing=jnp.array([[1.0, 0.5], [0.25, -1.0]]),
    )
    op = lmc.cross_covariance_operator(X, X)
    dense = lmc.full_covariance(X)
    assert jnp.allclose(op.as_matrix(), dense)
    v = jax.random.normal(jax.random.PRNGKey(0), (dense.shape[1],))
    assert jnp.allclose(op.mv(v), dense @ v, atol=1e-5)


def test_lmc_single_latent_operator_is_kronecker_tagged():
    X = jnp.array([[0.0], [1.0]])
    lmc = LMCKernel(
        kernels=(RBF(init_variance=1.0, init_lengthscale=0.5),),
        mixing=jnp.array([[1.0], [0.3]]),
    )
    op = lmc.cross_covariance_operator(X, X)
    assert is_kronecker(op)


# ---------------------------------------------------------------------------
# ICMKernel
# ---------------------------------------------------------------------------


def test_icm_adds_diagonal_kappa_to_coregionalization_matrix():
    X = jnp.array([[0.0], [1.0]])
    kernel = RBF(init_variance=1.2, init_lengthscale=0.4)
    mixing = jnp.array([[1.0, 0.0], [0.5, 1.0]])
    kappa = jnp.array([0.1, 0.2])
    icm = ICMKernel(kernel=kernel, mixing=mixing, kappa=kappa)

    B = mixing @ mixing.T + jnp.diag(kappa)
    assert jnp.allclose(icm.coregionalization_matrix(), B)
    assert jnp.allclose(icm.full_covariance(X), jnp.kron(B, kernel(X, X)))


def test_icm_kappa_none_matches_mixing_outer():
    X = jnp.array([[0.0], [0.5]])
    kernel = RBF(init_variance=1.0, init_lengthscale=0.5)
    mixing = jnp.array([[1.0, 0.2], [0.3, 0.8]])
    icm = ICMKernel(kernel=kernel, mixing=mixing)
    B = mixing @ mixing.T
    assert jnp.allclose(icm.coregionalization_matrix(), B)
    assert jnp.allclose(icm.full_covariance(X), jnp.kron(B, kernel(X, X)))


def test_icm_operator_is_kronecker_tagged_and_matches_dense():
    X = jnp.array([[0.0], [0.5], [1.0]])
    icm = ICMKernel(
        kernel=RBF(init_variance=1.0, init_lengthscale=0.4),
        mixing=jnp.array([[1.0, 0.2], [0.3, 0.8]]),
    )
    op = icm.cross_covariance_operator(X, X)
    assert is_kronecker(op)
    dense = icm.full_covariance(X)
    v = jax.random.normal(jax.random.PRNGKey(1), (dense.shape[1],))
    assert jnp.allclose(op.mv(v), dense @ v, atol=1e-5)


# ---------------------------------------------------------------------------
# OILMMKernel
# ---------------------------------------------------------------------------


def test_oilmm_projection_and_back_project_round_trip():
    X = jnp.array([[0.0], [1.0]])
    Y = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mixing = jnp.eye(2)
    kernels = (
        RBF(init_variance=1.0, init_lengthscale=0.5),
        RBF(init_variance=2.0, init_lengthscale=1.0),
    )
    oilmm = OILMMKernel(kernels=kernels, mixing=mixing)

    K1 = kernels[0](X, X)
    K2 = kernels[1](X, X)
    expected_signal = jnp.block([[K1, jnp.zeros_like(K1)], [jnp.zeros_like(K2), K2]])

    assert oilmm.is_orthogonal()
    assert oilmm.independent_gps() == kernels
    Y_latent, noise_latent = oilmm.project(Y, jnp.array([0.1, 0.2]))
    assert jnp.allclose(Y_latent, Y)
    assert jnp.allclose(noise_latent, jnp.array([0.1, 0.2]))
    assert jnp.allclose(oilmm.full_covariance(X), expected_signal)


def test_oilmm_project_routes_to_gaussx_primitive():
    mixing, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(3), (4, 2)))
    # jnp.linalg.qr returns (Q, R); take leading 2 columns as semi-orthogonal
    Y = jax.random.normal(jax.random.PRNGKey(4), (6, 4))
    noise_var = jnp.array([0.1, 0.2, 0.3, 0.4])
    kernels = (
        RBF(init_variance=1.0, init_lengthscale=0.5),
        RBF(init_variance=0.9, init_lengthscale=0.8),
    )
    oilmm = OILMMKernel(kernels=kernels, mixing=mixing)
    expected_Y, expected_noise = oilmm_project(Y, mixing, noise_var)
    Y_latent, noise_latent = oilmm.project(Y, noise_var)
    assert jnp.allclose(Y_latent, expected_Y)
    assert jnp.allclose(noise_latent, expected_noise)


def test_oilmm_back_project_round_trips_through_mixing():
    mixing, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(5), (3, 2)))
    kernels = (
        RBF(init_variance=1.0, init_lengthscale=0.5),
        RBF(init_variance=0.9, init_lengthscale=0.8),
    )
    oilmm = OILMMKernel(kernels=kernels, mixing=mixing)
    f_means = jax.random.normal(jax.random.PRNGKey(6), (5, 2))
    f_vars = jnp.abs(jax.random.normal(jax.random.PRNGKey(7), (5, 2)))
    y_means, y_vars = oilmm.back_project(f_means, f_vars)
    assert jnp.allclose(y_means, f_means @ mixing.T)
    assert jnp.allclose(y_vars, f_vars @ (mixing**2).T)


# ---------------------------------------------------------------------------
# Inducing structures
# ---------------------------------------------------------------------------


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


def test_shared_inducing_K_uu_operator_is_block_diagonal_tagged():
    Z = jnp.array([[-1.0], [0.0], [1.0]])
    shared = SharedInducingPoints(locations=Z)
    kernels = (
        RBF(init_variance=1.0, init_lengthscale=0.5),
        RBF(init_variance=2.0, init_lengthscale=1.5),
    )
    op = shared.K_uu_operator(kernels)
    assert is_block_diagonal(op)
    assert jnp.allclose(op.as_matrix(), shared.K_uu(kernels))


def test_shared_inducing_block_diag_solve_matches_dense():
    """gaussx.solve dispatches on BlockDiag and decomposes into per-block solves."""
    Z = jnp.array([[-1.0], [0.0], [1.0]])
    shared = SharedInducingPoints(locations=Z)
    kernels = (
        RBF(init_variance=1.0, init_lengthscale=0.5),
        RBF(init_variance=2.0, init_lengthscale=1.5),
    )
    op = shared.K_uu_operator(kernels)
    v = jax.random.normal(jax.random.PRNGKey(9), (op.in_size(),))
    solved_op = solve(op, v)
    solved_dense = jnp.linalg.solve(op.as_matrix(), v)
    assert jnp.allclose(solved_op, solved_dense, atol=1e-5)


def test_multi_output_inducing_cross_covariance_matches_single_latent_case():
    Z = jnp.array([[-1.0], [1.0]])
    X = jnp.array([[0.0], [0.5], [1.0]])
    kernel = RBF(init_variance=1.0, init_lengthscale=0.8)
    inducing = MultiOutputInducingVariables(
        inducing=SharedInducingPoints(locations=Z),
        mixing=jnp.array([[2.0], [-1.0]]),
    )

    K_zx = kernel(Z, X)
    K_uf = inducing.K_uf(X, (kernel,))

    assert jnp.allclose(inducing.K_uu((kernel,)), kernel(Z, Z))
    assert K_uf.shape == (2, 6)
    assert jnp.allclose(K_uf[:, :3], 2.0 * K_zx)
    assert jnp.allclose(K_uf[:, 3:], -1.0 * K_zx)


def test_multi_output_inducing_K_uf_multiple_latents_and_outputs():
    Z = jnp.array([[-1.0], [1.0]])
    X = jnp.array([[0.0], [0.5], [1.0]])
    kernels = (
        RBF(init_variance=1.0, init_lengthscale=0.5),
        RBF(init_variance=0.8, init_lengthscale=1.2),
    )
    mixing = jnp.array([[1.0, 0.5], [-0.3, 2.0]])
    inducing = MultiOutputInducingVariables(
        inducing=SharedInducingPoints(locations=Z),
        mixing=mixing,
    )

    K_uf = inducing.K_uf(X, kernels)
    M, N = Z.shape[0], X.shape[0]
    P, Q = mixing.shape
    assert K_uf.shape == (Q * M, P * N)

    K_zx0 = kernels[0](Z, X)
    K_zx1 = kernels[1](Z, X)
    # (q=0, p=0) block = mixing[0, 0] * K_q0
    assert jnp.allclose(K_uf[:M, :N], mixing[0, 0] * K_zx0)
    assert jnp.allclose(K_uf[:M, N : 2 * N], mixing[1, 0] * K_zx0)
    assert jnp.allclose(K_uf[M:, :N], mixing[0, 1] * K_zx1)
    assert jnp.allclose(K_uf[M:, N : 2 * N], mixing[1, 1] * K_zx1)


def test_multi_output_inducing_K_uu_operator_is_block_diagonal():
    Z = jnp.array([[-1.0], [1.0]])
    kernels = (
        RBF(init_variance=1.0, init_lengthscale=0.5),
        RBF(init_variance=0.8, init_lengthscale=1.2),
    )
    inducing = MultiOutputInducingVariables(
        inducing=SharedInducingPoints(locations=Z),
        mixing=jnp.array([[1.0, 0.5], [-0.3, 2.0]]),
    )
    op = inducing.K_uu_operator(kernels)
    assert is_block_diagonal(op)
    assert jnp.allclose(op.as_matrix(), inducing.K_uu(kernels))


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def test_lmc_rejects_mismatched_kernel_count():
    with pytest.raises(ValueError, match="one kernel per latent"):
        LMCKernel(
            kernels=(RBF(),),
            mixing=jnp.array([[1.0, 0.5], [0.25, -1.0]]),
        )


def test_lmc_rejects_empty_kernels():
    with pytest.raises(ValueError, match="at least one"):
        LMCKernel(kernels=(), mixing=jnp.zeros((2, 0)))


def test_icm_rejects_mismatched_kappa_shape():
    with pytest.raises(ValueError, match="kappa must have shape"):
        ICMKernel(
            kernel=RBF(),
            mixing=jnp.array([[1.0], [0.5]]),
            kappa=jnp.array([0.1, 0.2, 0.3]),
        )


def test_oilmm_rejects_more_latents_than_outputs():
    with pytest.raises(ValueError, match="num_latents <= num_outputs"):
        OILMMKernel(
            kernels=(RBF(), RBF(), RBF()),
            mixing=jnp.eye(3)[:2, :],  # (2, 3) — too many latents for 2 outputs
        )


def test_multi_output_inducing_rejects_non_2d_mixing():
    with pytest.raises(ValueError, match="num_outputs, num_latents"):
        MultiOutputInducingVariables(
            inducing=SharedInducingPoints(locations=jnp.zeros((2, 1))),
            mixing=jnp.array([1.0, 0.5]),  # 1D
        )


def test_shared_inducing_rejects_non_2d_locations():
    with pytest.raises(ValueError, match="num_inducing, input_dim"):
        SharedInducingPoints(locations=jnp.array([0.0, 1.0]))


def test_shared_inducing_K_uu_rejects_empty_kernel_tuple():
    shared = SharedInducingPoints(locations=jnp.zeros((2, 1)))
    with pytest.raises(ValueError, match="non-empty"):
        shared.K_uu(())
