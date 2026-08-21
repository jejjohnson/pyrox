"""Tests for the collapsed latent-factor likelihood and decoder posterior.

Ground truth throughout is the dense per-column multivariate normal
``N(y_j | 0, Z Z^T + sigma^2 I)`` — the collapsed implementation must agree
with it while never forming an ``(N, N)`` matrix.
"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from pyrox_gp import (
    collapsed_lfr_log_prob,
    decoder_posterior,
    lfr_predictive_moments,
)


jax.config.update("jax_enable_x64", True)


N, P, Q = 7, 11, 3


@pytest.fixture()
def problem() -> tuple[jax.Array, jax.Array, jax.Array]:
    """A small random ``(Y, Z, noise_var)`` triple."""
    k_y, k_z = jr.split(jr.key(0))
    Y = jr.normal(k_y, (N, P))
    Z = jr.normal(k_z, (N, Q))
    return Y, Z, jnp.asarray(0.3)


def _dense_cov(Z: jax.Array, s2: jax.Array) -> jax.Array:
    return Z @ Z.T + s2 * jnp.eye(Z.shape[0])


# --- log-density --------------------------------------------------------


def test_collapsed_matches_dense_per_column_mvn(problem) -> None:
    Y, Z, s2 = problem
    mvn = dist.MultivariateNormal(jnp.zeros(N), covariance_matrix=_dense_cov(Z, s2))
    expected = sum(mvn.log_prob(Y[:, j]) for j in range(P))
    got = collapsed_lfr_log_prob(Y, Z, s2, jitter=0.0)
    assert jnp.allclose(got, expected, atol=1e-8)


def test_woodbury_pieces_separately(problem) -> None:
    """Pin the log-det and inverse identities independently of the density."""
    _, Z, s2 = problem
    C = _dense_cov(Z, s2)
    psi = jnp.eye(Q) + Z.T @ Z / s2

    sign, logdet_c = jnp.linalg.slogdet(C)
    assert sign == 1.0
    logdet_via_psi = N * jnp.log(s2) + jnp.linalg.slogdet(psi)[1]
    assert jnp.allclose(logdet_c, logdet_via_psi, atol=1e-10)

    C_inv = jnp.eye(N) / s2 - Z @ jnp.linalg.inv(psi) @ Z.T / s2**2
    assert jnp.allclose(C_inv, jnp.linalg.inv(C), atol=1e-10)


def test_rotational_invariance(problem) -> None:
    """Only the span of ``Z`` is identified: an orthogonal rotation of its
    columns must leave the likelihood unchanged."""
    Y, Z, s2 = problem
    A, _ = jnp.linalg.qr(jr.normal(jr.key(3), (Q, Q)))
    base = collapsed_lfr_log_prob(Y, Z, s2)
    rotated = collapsed_lfr_log_prob(Y, Z @ A, s2)
    assert jnp.allclose(base, rotated, atol=1e-9)


# --- decoder posterior --------------------------------------------------


def test_decoder_posterior_matches_per_column_blr(problem) -> None:
    Y, Z, s2 = problem
    mean, row_cov = decoder_posterior(Y, Z, s2, jitter=0.0)
    psi = jnp.eye(Q) + Z.T @ Z / s2
    for j in range(P):
        expected_col = jnp.linalg.solve(psi, Z.T @ Y[:, j] / s2)
        assert jnp.allclose(mean[:, j], expected_col, atol=1e-10)
    assert jnp.allclose(row_cov, jnp.linalg.inv(psi), atol=1e-10)


# --- predictive moments -------------------------------------------------


@pytest.mark.slow
def test_predictive_moments_vs_monte_carlo(problem) -> None:
    Y, Z, s2 = problem
    mu_W, Sigma_W = decoder_posterior(Y, Z, s2)
    T = 4
    k_m, k_v, k_z, k_w = jr.split(jr.key(1), 4)
    z_mean = jr.normal(k_m, (T, Q))
    z_var = 0.1 + jr.uniform(k_v, (T, Q))

    n_samples = 200_000
    z = z_mean + jnp.sqrt(z_var) * jr.normal(k_z, (n_samples, T, Q))
    chol = jnp.linalg.cholesky(Sigma_W)
    W = mu_W + jnp.einsum("qr,srp->sqp", chol, jr.normal(k_w, (n_samples, Q, P)))
    f = jnp.einsum("stq,sqp->stp", z, W)

    mean, var = lfr_predictive_moments(z_mean, z_var, mu_W, Sigma_W)
    mc_mean = f.mean(axis=0)
    mc_var = f.var(axis=0)
    assert jnp.max(jnp.abs(mean - mc_mean)) < 0.02 * jnp.max(jnp.abs(mc_mean))
    assert jnp.max(jnp.abs(var - mc_var) / mc_var) < 0.02


def test_predictive_noise_var_adds(problem) -> None:
    Y, Z, s2 = problem
    mu_W, Sigma_W = decoder_posterior(Y, Z, s2)
    z_mean = jnp.ones((2, Q))
    z_var = jnp.full((2, Q), 0.5)
    _, var_signal = lfr_predictive_moments(z_mean, z_var, mu_W, Sigma_W)
    _, var_obs = lfr_predictive_moments(z_mean, z_var, mu_W, Sigma_W, noise_var=s2)
    assert jnp.allclose(var_obs, var_signal + s2)


# --- gradients ----------------------------------------------------------


def test_gradients_finite_and_match_finite_difference(problem) -> None:
    Y, Z, s2 = problem
    g_z, g_s2 = jax.grad(collapsed_lfr_log_prob, argnums=(1, 2))(Y, Z, s2)
    assert jnp.all(jnp.isfinite(g_z))
    assert jnp.isfinite(g_s2)

    eps = 1e-6
    fd = (
        collapsed_lfr_log_prob(Y, Z, s2 + eps) - collapsed_lfr_log_prob(Y, Z, s2 - eps)
    ) / (2 * eps)
    assert jnp.abs(g_s2 - fd) / jnp.abs(fd) < 1e-6


# --- jit ----------------------------------------------------------------


def test_all_functions_run_under_jit(problem) -> None:
    Y, Z, s2 = problem
    lp = jax.jit(collapsed_lfr_log_prob)(Y, Z, s2)
    assert jnp.isfinite(lp)
    mean, row_cov = jax.jit(decoder_posterior)(Y, Z, s2)
    z_mean = jnp.zeros((2, Q))
    z_var = jnp.ones((2, Q))
    m, v = jax.jit(lfr_predictive_moments)(z_mean, z_var, mean, row_cov)
    assert jnp.all(jnp.isfinite(m)) and jnp.all(v > 0)


# --- scaling ------------------------------------------------------------


@pytest.mark.slow
def test_cost_is_linear_in_p() -> None:
    """Wall clock must grow sub-quadratically in the output dimension."""
    n, q = 20, 4
    k_z = jr.key(2)
    Z = jr.normal(k_z, (n, q))
    fn = jax.jit(collapsed_lfr_log_prob)

    def _best_time(p: int) -> float:
        Y = jr.normal(jr.key(p), (n, p))
        fn(Y, Z, jnp.asarray(0.5)).block_until_ready()  # compile
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            fn(Y, Z, jnp.asarray(0.5)).block_until_ready()
            best = min(best, time.perf_counter() - t0)
        return best

    t_small, t_mid, t_big = (_best_time(p) for p in (10**3, 10**4, 10**5))
    # Quadratic growth would give a 100x step; linear gives 10x. Allow
    # generous timer noise while still rejecting a quadratic implementation.
    assert t_big / max(t_mid, 1e-9) < 50.0
    assert t_mid / max(t_small, 1e-9) < 50.0


# --- validation ---------------------------------------------------------


def test_mismatched_rows_raise() -> None:
    Y = jnp.zeros((5, 3))
    Z = jnp.zeros((4, 2))
    with pytest.raises(ValueError, match="leading"):
        collapsed_lfr_log_prob(Y, Z, jnp.asarray(0.1))
    with pytest.raises(ValueError, match="leading"):
        decoder_posterior(Y, Z, jnp.asarray(0.1))


# --- independent oracle -------------------------------------------------


def test_matches_gaussian_pca_oracle(problem) -> None:
    """`gauss_flows.GaussianPCA` is a second, independently written Woodbury
    implementation of the same density: with ``W := Z`` it evaluates one
    column of the collapsed likelihood.

    TODO(gauss_flows#138): pinned at ``s2 = 1.0`` until the sigma-power bug
    in the GaussianPCA Woodbury quadratic is fixed upstream — ``sigma^2 = 1``
    is the one value where the buggy and correct expressions coincide.
    """
    gauss_flows = pytest.importorskip("gauss_flows")

    Y, Z, _ = problem
    s2 = jnp.asarray(1.0)
    pca = gauss_flows.GaussianPCA(jr.key(0), event_shape=(N,), latent_dim=Q)
    pca = eqx.tree_at(lambda m: m.W, pca, Z)
    pca = eqx.tree_at(lambda m: m.log_sigma, pca, jnp.log(jnp.sqrt(s2)))
    oracle = jnp.sum(jax.vmap(pca.log_prob)(Y.T))
    # GaussianPCA floors its variance at exp(2*log_sigma) + eps; match the
    # floor through the jitter argument so the two densities are identical.
    got = collapsed_lfr_log_prob(Y, Z, s2, jitter=pca.eps)
    assert jnp.allclose(oracle, got, atol=1e-8)


# --- numerical robustness -----------------------------------------------


def test_quadratic_is_stable_at_small_noise_in_float32() -> None:
    """With ``Y`` in the span of ``Z`` and small noise, the naive Woodbury
    quadratic differences two ``O(1/s2)`` terms and loses most of its digits.
    The penalized-residual form used here must stay accurate in float32."""
    Z32 = jr.normal(jr.key(1), (7, 3), dtype=jnp.float32)
    W32 = jr.normal(jr.key(2), (3, 11), dtype=jnp.float32)
    Y32 = Z32 @ W32
    s2 = jnp.asarray(1e-4, dtype=jnp.float32)

    got = collapsed_lfr_log_prob(Y32, Z32, s2)
    assert got.dtype == jnp.float32

    ref = collapsed_lfr_log_prob(
        Y32.astype(jnp.float64), Z32.astype(jnp.float64), s2.astype(jnp.float64)
    )
    assert jnp.abs(got - ref) / jnp.abs(ref) < 1e-5


def test_non_scalar_noise_var_raises(problem) -> None:
    """Per-output noise breaks the shared-covariance identity, and with
    ``P == Q`` it would otherwise broadcast silently into a nonsymmetric
    capacitance matrix."""
    Y, Z, _ = problem
    vector_noise = jnp.full((Z.shape[1],), 0.3)
    with pytest.raises(ValueError, match="must be a scalar"):
        collapsed_lfr_log_prob(Y, Z, vector_noise)
    with pytest.raises(ValueError, match="must be a scalar"):
        decoder_posterior(Y, Z, vector_noise)
