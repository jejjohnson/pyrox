"""Tests for multi-output kernel and inducing helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.handlers as handlers
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
            RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
            RBF(pyrox_name="RBF_q1", init_variance=0.8, init_lengthscale=1.2),
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
            RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
            RBF(pyrox_name="RBF_q1", init_variance=0.8, init_lengthscale=1.2),
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
            RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
            RBF(pyrox_name="RBF_q1", init_variance=0.8, init_lengthscale=1.2),
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
            RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
            RBF(pyrox_name="RBF_q1", init_variance=0.8, init_lengthscale=1.2),
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


def test_lmc_cross_covariance_operator_rectangular_X1_X2():
    """Regression: rectangular cross-covariance must not be tagged PSD."""
    X1 = jnp.array([[0.0], [0.5], [1.0]])
    X2 = jnp.array([[0.2], [0.7]])
    lmc = LMCKernel(
        kernels=(
            RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
            RBF(pyrox_name="RBF_q1", init_variance=0.8, init_lengthscale=1.0),
        ),
        mixing=jnp.array([[1.0, 0.5], [0.25, -1.0]]),
    )
    op = lmc.cross_covariance_operator(X1, X2)
    dense = lmc.cross_covariance(X1, X2)
    assert dense.shape == (6, 4)
    assert jnp.allclose(op.as_matrix(), dense)


def test_icm_cross_covariance_operator_rectangular_X1_X2():
    X1 = jnp.array([[0.0], [0.5], [1.0]])
    X2 = jnp.array([[0.2], [0.7]])
    icm = ICMKernel(
        kernel=RBF(init_variance=1.0, init_lengthscale=0.5),
        mixing=jnp.array([[1.0, 0.2], [0.3, 0.8]]),
    )
    op = icm.cross_covariance_operator(X1, X2)
    dense = icm.cross_covariance(X1, X2)
    assert dense.shape == (6, 4)
    assert jnp.allclose(op.as_matrix(), dense)


def test_lmc_cross_covariance_operator_square_non_psd_X1_ne_X2():
    """Regression: N1 == N2 but X1 != X2 yields a square, non-symmetric K_q.

    The old ``_kron_block_op`` inferred PSD from ``K.shape[0] == K.shape[1]``,
    which would mis-tag such a cross-covariance and break structural
    dispatch. The fix uses the caller-provided ``psd_K`` flag, so square
    cross-covariances stay untagged.
    """
    X1 = jnp.array([[0.0], [0.5], [1.0]])
    X2 = jnp.array([[0.2], [0.6], [0.9]])  # same length, different points
    lmc = LMCKernel(
        kernels=(
            RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
            RBF(pyrox_name="RBF_q1", init_variance=0.8, init_lengthscale=1.0),
        ),
        mixing=jnp.array([[1.0, 0.5], [0.25, -1.0]]),
    )
    op = lmc.cross_covariance_operator(X1, X2)
    dense = lmc.cross_covariance(X1, X2)
    assert dense.shape == (6, 6)
    assert jnp.allclose(op.as_matrix(), dense)
    # The dense cross-covariance must not be symmetric — this is exactly
    # the configuration the old shape-based PSD tag would have mis-tagged.
    assert not jnp.allclose(dense, dense.T)


def test_icm_cross_covariance_operator_square_non_psd_X1_ne_X2():
    X1 = jnp.array([[0.0], [0.5], [1.0]])
    X2 = jnp.array([[0.2], [0.6], [0.9]])
    icm = ICMKernel(
        kernel=RBF(init_variance=1.0, init_lengthscale=0.5),
        mixing=jnp.array([[1.0, 0.2], [0.3, 0.8]]),
    )
    op = icm.cross_covariance_operator(X1, X2)
    dense = icm.cross_covariance(X1, X2)
    assert dense.shape == (6, 6)
    assert jnp.allclose(op.as_matrix(), dense)
    assert not jnp.allclose(dense, dense.T)


def test_oilmm_signal_covariance_operator_rectangular_X1_X2():
    X1 = jnp.array([[0.0], [0.5], [1.0]])
    X2 = jnp.array([[0.2], [0.7]])
    mixing = jnp.eye(2)
    oilmm = OILMMKernel(
        kernels=(
            RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
            RBF(pyrox_name="RBF_q1", init_variance=0.8, init_lengthscale=1.0),
        ),
        mixing=mixing,
    )
    op = oilmm.signal_covariance_operator(X1, X2)
    dense = oilmm.signal_covariance(X1, X2)
    assert dense.shape == (6, 4)
    assert jnp.allclose(op.as_matrix(), dense)


def test_oilmm_signal_covariance_operator_square_non_psd_X1_ne_X2():
    X1 = jnp.array([[0.0], [0.5], [1.0]])
    X2 = jnp.array([[0.2], [0.6], [0.9]])
    mixing = jnp.eye(2)
    oilmm = OILMMKernel(
        kernels=(
            RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
            RBF(pyrox_name="RBF_q1", init_variance=0.8, init_lengthscale=1.0),
        ),
        mixing=mixing,
    )
    op = oilmm.signal_covariance_operator(X1, X2)
    dense = oilmm.signal_covariance(X1, X2)
    assert dense.shape == (6, 6)
    assert jnp.allclose(op.as_matrix(), dense)
    assert not jnp.allclose(dense, dense.T)


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
        RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
        RBF(pyrox_name="RBF_q1", init_variance=2.0, init_lengthscale=1.0),
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
        RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
        RBF(pyrox_name="RBF_q1", init_variance=0.9, init_lengthscale=0.8),
    )
    oilmm = OILMMKernel(kernels=kernels, mixing=mixing)
    expected_Y, expected_noise = oilmm_project(Y, mixing, noise_var)
    Y_latent, noise_latent = oilmm.project(Y, noise_var)
    assert jnp.allclose(Y_latent, expected_Y)
    assert jnp.allclose(noise_latent, expected_noise)


def test_oilmm_back_project_round_trips_through_mixing():
    mixing, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(5), (3, 2)))
    kernels = (
        RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
        RBF(pyrox_name="RBF_q1", init_variance=0.9, init_lengthscale=0.8),
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
        RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
        RBF(pyrox_name="RBF_q1", init_variance=2.0, init_lengthscale=1.5),
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
        RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
        RBF(pyrox_name="RBF_q1", init_variance=2.0, init_lengthscale=1.5),
    )
    op = shared.K_uu_operator(kernels)
    assert is_block_diagonal(op)
    assert jnp.allclose(op.as_matrix(), shared.K_uu(kernels))


def test_shared_inducing_block_diag_solve_matches_dense():
    """gaussx.solve dispatches on BlockDiag and decomposes into per-block solves."""
    Z = jnp.array([[-1.0], [0.0], [1.0]])
    shared = SharedInducingPoints(locations=Z)
    kernels = (
        RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
        RBF(pyrox_name="RBF_q1", init_variance=2.0, init_lengthscale=1.5),
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
        RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
        RBF(pyrox_name="RBF_q1", init_variance=0.8, init_lengthscale=1.2),
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
        RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
        RBF(pyrox_name="RBF_q1", init_variance=0.8, init_lengthscale=1.2),
    )
    inducing = MultiOutputInducingVariables(
        inducing=SharedInducingPoints(locations=Z),
        mixing=jnp.array([[1.0, 0.5], [-0.3, 2.0]]),
    )
    op = inducing.K_uu_operator(kernels)
    assert is_block_diagonal(op)
    assert jnp.allclose(op.as_matrix(), inducing.K_uu(kernels))


def test_multi_output_inducing_blocks_matches_separate_K_uu_K_uf():
    Z = jnp.array([[-1.0], [1.0]])
    X = jnp.array([[0.0], [0.5], [1.0]])
    kernels = (
        RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
        RBF(pyrox_name="RBF_q1", init_variance=0.8, init_lengthscale=1.2),
    )
    inducing = MultiOutputInducingVariables(
        inducing=SharedInducingPoints(locations=Z),
        mixing=jnp.array([[1.0, 0.5], [-0.3, 2.0]]),
    )
    K_uu_op, K_uf = inducing.inducing_blocks(X, kernels)
    assert is_block_diagonal(K_uu_op)
    assert jnp.allclose(K_uu_op.as_matrix(), inducing.K_uu(kernels))
    assert jnp.allclose(K_uf, inducing.K_uf(X, kernels))


def test_multi_output_inducing_blocks_shares_context_when_kernel_tied():
    """Regression for the codex P1 review: K_uu and K_uf called
    sequentially via separate ``_kernel_contexts`` would re-register
    sample sites for a tied kernel, tripping a NumPyro trace duplicate-
    site error. ``inducing_blocks`` shares one context across the pair.
    """
    kernel = RBF()
    kernel.set_prior("variance", dist.LogNormal(0.0, 0.3))
    kernel.set_prior("lengthscale", dist.LogNormal(0.0, 0.3))
    Z = jnp.array([[-1.0], [0.0], [1.0]])
    X = jnp.array([[-0.5], [0.5]])
    inducing = MultiOutputInducingVariables(
        inducing=SharedInducingPoints(locations=Z),
        mixing=jnp.array([[1.0, 0.5], [-0.3, 2.0]]),
    )

    def model():
        return inducing.inducing_blocks(X, (kernel, kernel))

    with numpyro.handlers.trace() as tr, handlers.seed(rng_seed=3):
        K_uu_op, K_uf = model()
    assert "RBF.variance" in tr
    assert "RBF.lengthscale" in tr
    # Outputs are still well-formed.
    assert K_uu_op.as_matrix().shape == (6, 6)
    assert K_uf.shape == (6, 4)


def test_shared_inducing_blocks_shares_context_when_kernel_tied():
    kernel = RBF()
    kernel.set_prior("variance", dist.LogNormal(0.0, 0.3))
    Z = jnp.array([[-1.0], [0.0], [1.0]])
    X = jnp.array([[-0.5], [0.5]])
    shared = SharedInducingPoints(locations=Z)

    def model():
        return shared.inducing_blocks(X, (kernel, kernel))

    with numpyro.handlers.trace() as tr, handlers.seed(rng_seed=4):
        K_uu_blocks, K_uf_blocks = model()
    assert "RBF.variance" in tr
    # Blocks for tied kernel must be identical (same hyperparameter draw).
    assert jnp.allclose(K_uu_blocks[0], K_uu_blocks[1])
    assert jnp.allclose(K_uf_blocks[0], K_uf_blocks[1])


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


def test_icm_rejects_negative_kappa():
    """Regression: a negative ``kappa`` entry can pull ``B`` out of PSD,
    but downstream operators tag ``B`` as PSD — silently breaking
    Cholesky-backed solvers. The validator catches this at construction.
    """
    with pytest.raises(ValueError, match=r"ICMKernel\.kappa must be nonnegative"):
        ICMKernel(
            kernel=RBF(),
            mixing=jnp.array([[1.0], [0.5]]),
            kappa=jnp.array([0.1, -0.2]),
        )


def test_icm_accepts_zero_kappa():
    """Zero kappa is on the PSD boundary and must remain accepted."""
    icm = ICMKernel(
        kernel=RBF(),
        mixing=jnp.array([[1.0], [0.5]]),
        kappa=jnp.zeros((2,)),
    )
    assert icm.num_outputs == 2


def test_icm_kappa_check_is_no_op_under_jit():
    """The nonnegativity check is best-effort: under tracing the kappa
    value is not concrete and the check is skipped, leaving construction
    inside ``jax.jit`` valid (the check still runs at the outer
    eager-construction boundary the user controls).
    """
    mixing = jnp.array([[1.0], [0.5]])

    @jax.jit
    def make(kappa):
        icm = ICMKernel(kernel=RBF(), mixing=mixing, kappa=kappa)
        return icm.coregionalization_matrix()

    # Even with negative kappa the jitted construction must not raise —
    # the check has no concrete value to inspect.
    out = make(jnp.array([0.1, -0.2]))
    assert out.shape == (2, 2)


def test_oilmm_rejects_more_latents_than_outputs():
    with pytest.raises(ValueError, match="num_latents <= num_outputs"):
        OILMMKernel(
            kernels=(
                RBF(pyrox_name="RBF_q0"),
                RBF(pyrox_name="RBF_q1"),
                RBF(pyrox_name="RBF_q2"),
            ),
            mixing=jnp.eye(3)[:2, :],  # (2, 3) — too many latents for 2 outputs
        )


def test_multi_output_inducing_from_kernel_rejects_nonzero_kappa():
    """``from_kernel`` drops ``kernel.kappa``, so accepting a non-zero
    ``kappa`` would make K_ff (which keeps ``diag(kappa)``) inconsistent
    with the sparse K_uu / K_uf blocks. Must fail fast instead.
    """
    shared = SharedInducingPoints(locations=jnp.zeros((2, 1)))
    icm = ICMKernel(
        kernel=RBF(),
        mixing=jnp.array([[1.0], [0.5]]),
        kappa=jnp.array([0.1, 0.2]),
    )
    with pytest.raises(ValueError, match=r"ICMKernel\.kappa must be None or all zeros"):
        MultiOutputInducingVariables.from_kernel(icm, shared)


def test_multi_output_inducing_from_kernel_accepts_zero_kappa():
    """All-zero ``kappa`` is equivalent to ``None`` for this construction."""
    shared = SharedInducingPoints(locations=jnp.zeros((2, 1)))
    icm = ICMKernel(
        kernel=RBF(),
        mixing=jnp.array([[1.0], [0.5]]),
        kappa=jnp.zeros((2,)),
    )
    inducing = MultiOutputInducingVariables.from_kernel(icm, shared)
    assert inducing.num_outputs == 2
    assert inducing.num_latents == 1


def test_multi_output_inducing_from_kernel_accepts_none_kappa():
    """``kappa=None`` (the common case) must keep working."""
    shared = SharedInducingPoints(locations=jnp.zeros((2, 1)))
    icm = ICMKernel(kernel=RBF(), mixing=jnp.array([[1.0], [0.5]]))
    inducing = MultiOutputInducingVariables.from_kernel(icm, shared)
    assert inducing.num_outputs == 2


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


# ---------------------------------------------------------------------------
# Shared kernel-context regression (hyperparameter tying across latents)
# ---------------------------------------------------------------------------


def test_lmc_full_covariance_shares_context_when_kernel_tied_across_latents():
    """Reusing one priored kernel instance across latents must not trip
    duplicate sample-site registration.

    Regression for the codex P1 review comment: opening a fresh
    ``_kernel_context`` per kernel call clears the per-call cache between
    calls, so a NumPyro trace sees a second registration of the same
    site name. Sharing one context across the multi-kernel builder loop
    collapses the two calls into a single cached registration.
    """
    kernel = RBF()
    kernel.set_prior("variance", dist.LogNormal(0.0, 0.3))
    kernel.set_prior("lengthscale", dist.LogNormal(0.0, 0.3))
    # Deliberately reuse the same instance for both latents.
    lmc = LMCKernel(
        kernels=(kernel, kernel),
        mixing=jnp.array([[1.0, 0.5], [0.25, -1.0]]),
    )
    X = jnp.array([[0.0], [0.5], [1.0]])

    def model():
        return lmc.full_covariance(X)

    with numpyro.handlers.trace() as tr, handlers.seed(rng_seed=0):
        model()
    # Each priored hyperparameter registered exactly once despite two
    # latents sharing the kernel instance.
    assert "RBF.variance" in tr
    assert "RBF.lengthscale" in tr


def test_oilmm_signal_covariance_shares_context_when_kernel_tied_across_latents():
    kernel = RBF()
    kernel.set_prior("variance", dist.LogNormal(0.0, 0.3))
    mixing = jnp.eye(2)
    oilmm = OILMMKernel(kernels=(kernel, kernel), mixing=mixing)
    X = jnp.array([[0.0], [0.5], [1.0]])

    def model():
        return oilmm.signal_covariance(X, X)

    with numpyro.handlers.trace() as tr, handlers.seed(rng_seed=1):
        model()
    assert "RBF.variance" in tr


def test_shared_inducing_K_uu_shares_context_when_kernel_tied_across_latents():
    kernel = RBF()
    kernel.set_prior("variance", dist.LogNormal(0.0, 0.3))
    shared = SharedInducingPoints(locations=jnp.array([[-1.0], [0.0], [1.0]]))

    def model():
        return shared.K_uu((kernel, kernel))

    with numpyro.handlers.trace() as tr, handlers.seed(rng_seed=2):
        model()
    assert "RBF.variance" in tr


# ---------------------------------------------------------------------------
# Scope-name uniqueness validator (untied priored kernels)
# ---------------------------------------------------------------------------


def test_lmc_rejects_distinct_priored_kernels_sharing_scope():
    """Regression for codex P1: two distinct ``RBF()`` instances both
    default to ``pyrox_name='RBF'``. With priors set, both would
    register identical sample sites in a NumPyro trace, raising the
    duplicate-site error. Catch this at construction.
    """
    k0 = RBF()
    k0.set_prior("variance", dist.LogNormal(0.0, 0.3))
    k1 = RBF()
    k1.set_prior("variance", dist.LogNormal(0.0, 0.3))
    with pytest.raises(ValueError, match="distinct latent kernel instances"):
        LMCKernel(
            kernels=(k0, k1),
            mixing=jnp.array([[1.0, 0.5], [0.25, -1.0]]),
        )


def test_oilmm_rejects_distinct_priored_kernels_sharing_scope():
    k0 = RBF()
    k0.set_prior("variance", dist.LogNormal(0.0, 0.3))
    k1 = RBF()
    k1.set_prior("variance", dist.LogNormal(0.0, 0.3))
    with pytest.raises(ValueError, match="distinct latent kernel instances"):
        OILMMKernel(kernels=(k0, k1), mixing=jnp.eye(2))


def test_lmc_accepts_distinct_priored_kernels_with_explicit_unique_scopes():
    """User-set distinct ``pyrox_name`` values bypass the validator."""
    k0 = RBF(pyrox_name="RBF_q0")
    k0.set_prior("variance", dist.LogNormal(0.0, 0.3))
    k1 = RBF(pyrox_name="RBF_q1")
    k1.set_prior("variance", dist.LogNormal(0.0, 0.3))
    lmc = LMCKernel(
        kernels=(k0, k1),
        mixing=jnp.array([[1.0, 0.5], [0.25, -1.0]]),
    )
    assert lmc.num_latents == 2

    # Trace registers both site names without collision.
    X = jnp.array([[0.0], [0.5], [1.0]])

    def model():
        return lmc.full_covariance(X)

    with numpyro.handlers.trace() as tr, handlers.seed(rng_seed=5):
        model()
    assert "RBF_q0.variance" in tr
    assert "RBF_q1.variance" in tr


def test_lmc_rejects_distinct_non_priored_kernels_sharing_scope():
    """Even without priors set, two distinct ``RBF()`` instances both
    register ``pyrox_param`` sites under the same ``RBF.lengthscale`` /
    ``RBF.variance`` names. NumPyro's param store dedups by name, so
    inside an SVI/MAP trace the two latents would silently share one
    parameter — corrupting fits where the latents are meant to vary
    independently. The validator catches this at construction.
    """
    with pytest.raises(ValueError, match="distinct latent kernel instances"):
        LMCKernel(
            kernels=(
                RBF(init_variance=1.0, init_lengthscale=0.5),
                RBF(init_variance=0.8, init_lengthscale=1.2),
            ),
            mixing=jnp.array([[1.0, 0.5], [0.25, -1.0]]),
        )


def test_lmc_silent_param_tying_regression_documented():
    """Counterpart to the validator: if you bypass the validator by
    using explicit unique scopes, the two latents register *distinct*
    ``numpyro.param`` sites and stay independent under a trace. This
    documents the contract the validator protects.
    """
    lmc = LMCKernel(
        kernels=(
            RBF(pyrox_name="RBF_q0", init_variance=1.0, init_lengthscale=0.5),
            RBF(pyrox_name="RBF_q1", init_variance=0.8, init_lengthscale=1.2),
        ),
        mixing=jnp.array([[1.0, 0.5], [0.25, -1.0]]),
    )
    X = jnp.array([[0.0], [0.5], [1.0]])

    def model():
        return lmc.full_covariance(X)

    with numpyro.handlers.trace() as tr, handlers.seed(rng_seed=7):
        model()
    assert "RBF_q0.variance" in tr
    assert "RBF_q1.variance" in tr
    assert "RBF_q0.lengthscale" in tr
    assert "RBF_q1.lengthscale" in tr


def test_lmc_accepts_tied_priored_kernel_reused_across_latents():
    """The same priored instance reused across latents is intentional
    hyperparameter tying and must remain accepted.
    """
    kernel = RBF()
    kernel.set_prior("variance", dist.LogNormal(0.0, 0.3))
    lmc = LMCKernel(
        kernels=(kernel, kernel),
        mixing=jnp.array([[1.0, 0.5], [0.25, -1.0]]),
    )
    assert lmc.num_latents == 2


def test_shared_inducing_K_uu_rejects_distinct_priored_kernels_sharing_scope():
    """The validator also fires on lower-level ``SharedInducingPoints``
    helpers when the user passes priored colliding kernels directly."""
    k0 = RBF()
    k0.set_prior("variance", dist.LogNormal(0.0, 0.3))
    k1 = RBF()
    k1.set_prior("variance", dist.LogNormal(0.0, 0.3))
    shared = SharedInducingPoints(locations=jnp.array([[-1.0], [1.0]]))
    with pytest.raises(ValueError, match="distinct latent kernel instances"):
        shared.K_uu((k0, k1))
