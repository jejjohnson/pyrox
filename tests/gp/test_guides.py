"""Tests for the sparse SVGP variational guide families.

Verifies for each of :class:`FullRankGuide`, :class:`MeanFieldGuide`, and
:class:`WhitenedGuide`:

* sample / log_prob shape and value (Gaussian density closed form).
* KL divergence matches the standard Gaussian closed form.
* Predictive mean/variance match a direct numpy implementation.
* The whitened/unwhitened equivalence: a whitened guide ``q(v) = N(m_v,
  L_v L_v^T)`` and its unwhitened equivalent ``q(u) = N(L_zz m_v,
  L_zz L_v L_v^T L_zz^T)`` produce identical KL and predictive moments.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpyro.distributions as nd

from pyrox.gp import (
    RBF,
    FullRankGuide,
    MeanFieldGuide,
    SparseGPPrior,
    WhitenedGuide,
)


# --- Fixtures --------------------------------------------------------------


def _toy_setup(M: int = 5, N: int = 8, seed: int = 0):
    Z = jnp.linspace(-2.0, 2.0, M).reshape(-1, 1)
    X = jnp.linspace(-3.0, 3.0, N).reshape(-1, 1)
    prior = SparseGPPrior(
        kernel=RBF(init_variance=1.2, init_lengthscale=0.6),
        Z=Z,
        jitter=1e-4,
    )
    K_zz_op = prior.inducing_operator()
    K_xz = prior.cross_covariance(X)
    K_xx_diag = prior.kernel_diag(X)
    return Z, X, prior, K_zz_op, K_xz, K_xx_diag


def _direct_sparse_predictive(
    K_xz: jnp.ndarray,
    K_zz: jnp.ndarray,
    K_xx_diag: jnp.ndarray,
    u_mean: jnp.ndarray,
    u_cov: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Closed-form ``mu_*, sigma_*^2`` via numpy linear algebra."""
    K_zx = K_xz.T
    Kzz_inv_Kzx = jnp.linalg.solve(K_zz, K_zx)  # (M, N)
    mu = K_xz @ jnp.linalg.solve(K_zz, u_mean)
    # Prior reduction
    Q = jnp.sum(K_xz * Kzz_inv_Kzx.T, axis=1)
    # Posterior contribution
    S_term = jnp.sum(Kzz_inv_Kzx * (u_cov @ Kzz_inv_Kzx), axis=0)
    sigma2 = K_xx_diag - Q + S_term
    return mu, sigma2


# --- FullRankGuide ---------------------------------------------------------


def test_fullrank_init_returns_zero_mean_identity_scale():
    g = FullRankGuide.init(num_inducing=4)
    assert g.mean.shape == (4,)
    assert jnp.allclose(g.mean, 0.0)
    assert jnp.allclose(g.scale_tril, jnp.eye(4))


def test_fullrank_sample_and_log_prob_match_numpyro_mvn():
    """Sample shape and log-prob match a reference numpyro MVN."""
    M = 5
    mean = jnp.linspace(-1.0, 1.0, M)
    scale = 0.7 * jnp.eye(M) + 0.1 * jnp.tril(jnp.ones((M, M)), -1)
    g = FullRankGuide(mean=mean, scale_tril=scale)
    key = jr.PRNGKey(11)
    u = g.sample(key)
    assert u.shape == (M,)
    ref = nd.MultivariateNormal(mean, scale_tril=scale)
    assert jnp.allclose(g.log_prob(u), ref.log_prob(u), atol=1e-6)


def test_fullrank_kl_matches_numpyro_kl_divergence():
    _, _, _, K_zz_op, _, _ = _toy_setup()
    M = K_zz_op.in_size()
    mean = jnp.linspace(-0.5, 0.5, M)
    scale = 0.6 * jnp.eye(M)
    g = FullRankGuide(mean=mean, scale_tril=scale)

    K_zz = K_zz_op.as_matrix()
    q = nd.MultivariateNormal(mean, scale_tril=scale)
    p = nd.MultivariateNormal(jnp.zeros(M), covariance_matrix=K_zz)
    ref = nd.kl_divergence(q, p)
    got = g.kl_divergence(K_zz_op)
    assert jnp.allclose(got, ref, atol=1e-6)


def test_fullrank_predict_matches_direct_closed_form():
    _, _, _, K_zz_op, K_xz, K_xx_diag = _toy_setup()
    M = K_zz_op.in_size()
    mean = jr.normal(jr.PRNGKey(3), (M,))
    L_raw = jr.uniform(jr.PRNGKey(4), (M, M))
    L = jnp.tril(L_raw) + 0.3 * jnp.eye(M)
    g = FullRankGuide(mean=mean, scale_tril=L)
    mu, var = g.predict(K_xz, K_zz_op, K_xx_diag)
    K_zz = K_zz_op.as_matrix()
    mu_ref, var_ref = _direct_sparse_predictive(K_xz, K_zz, K_xx_diag, mean, L @ L.T)
    assert jnp.allclose(mu, mu_ref, atol=1e-6)
    assert jnp.allclose(var, var_ref, atol=1e-6)


# --- MeanFieldGuide --------------------------------------------------------


def test_meanfield_init_returns_zero_mean_unit_scale():
    g = MeanFieldGuide.init(num_inducing=4, scale=2.0)
    assert g.mean.shape == (4,)
    assert jnp.allclose(g.mean, 0.0)
    assert jnp.allclose(g.scale, 2.0)


def test_meanfield_log_prob_matches_diagonal_normal():
    """Diagonal MVN log-prob via independent normals."""
    M = 6
    mean = jnp.linspace(-1.0, 1.0, M)
    scale = jnp.linspace(0.3, 0.7, M)
    g = MeanFieldGuide(mean=mean, scale=scale)
    u = jr.normal(jr.PRNGKey(0), (M,))
    ref = nd.Normal(mean, scale).log_prob(u).sum()
    assert jnp.allclose(g.log_prob(u), ref, atol=1e-6)


def test_meanfield_kl_matches_numpyro_kl_divergence():
    _, _, _, K_zz_op, _, _ = _toy_setup()
    M = K_zz_op.in_size()
    mean = jnp.linspace(-0.5, 0.5, M)
    scale = jnp.linspace(0.4, 0.8, M)
    g = MeanFieldGuide(mean=mean, scale=scale)

    K_zz = K_zz_op.as_matrix()
    q = nd.MultivariateNormal(mean, covariance_matrix=jnp.diag(scale**2))
    p = nd.MultivariateNormal(jnp.zeros(M), covariance_matrix=K_zz)
    ref = nd.kl_divergence(q, p)
    got = g.kl_divergence(K_zz_op)
    assert jnp.allclose(got, ref, atol=1e-6)


def test_meanfield_predict_matches_direct_closed_form():
    _, _, _, K_zz_op, K_xz, K_xx_diag = _toy_setup()
    M = K_zz_op.in_size()
    mean = jr.normal(jr.PRNGKey(5), (M,))
    scale = 0.3 + jnp.abs(jr.normal(jr.PRNGKey(6), (M,))) * 0.5
    g = MeanFieldGuide(mean=mean, scale=scale)
    mu, var = g.predict(K_xz, K_zz_op, K_xx_diag)
    K_zz = K_zz_op.as_matrix()
    mu_ref, var_ref = _direct_sparse_predictive(
        K_xz, K_zz, K_xx_diag, mean, jnp.diag(scale**2)
    )
    assert jnp.allclose(mu, mu_ref, atol=1e-6)
    assert jnp.allclose(var, var_ref, atol=1e-6)


# --- WhitenedGuide ---------------------------------------------------------


def test_whitened_kl_matches_standard_normal_closed_form_at_init():
    """At ``q(v) = N(0, I)`` the KL against ``N(0, I)`` is zero."""
    g = WhitenedGuide.init(num_inducing=5)
    assert jnp.allclose(g.kl_divergence(), 0.0, atol=1e-12)


def test_whitened_kl_matches_numpyro_kl_against_unit_normal():
    M = 5
    mean = jnp.linspace(-1.0, 1.0, M)
    L_raw = jr.uniform(jr.PRNGKey(2), (M, M))
    L = jnp.tril(L_raw) + 0.3 * jnp.eye(M)
    g = WhitenedGuide(mean=mean, scale_tril=L)
    q = nd.MultivariateNormal(mean, scale_tril=L)
    p = nd.MultivariateNormal(jnp.zeros(M), covariance_matrix=jnp.eye(M))
    ref = nd.kl_divergence(q, p)
    got = g.kl_divergence()
    assert jnp.allclose(got, ref, atol=1e-6)


def test_whitened_kl_ignores_prior_cov_argument():
    """The ``prior_cov`` arg is accepted for signature parity but ignored."""
    M = 5
    mean = jnp.linspace(-1.0, 1.0, M)
    L = 0.5 * jnp.eye(M)
    g = WhitenedGuide(mean=mean, scale_tril=L)
    bogus = lx.MatrixLinearOperator(7.0 * jnp.eye(M), lx.positive_semidefinite_tag)
    assert jnp.allclose(g.kl_divergence(), g.kl_divergence(bogus))


def test_whitened_predict_at_init_matches_prior_diag():
    """At ``q(v) = N(0, I)`` the predictive variance equals ``K_xx_diag``."""
    _, _, _, K_zz_op, K_xz, K_xx_diag = _toy_setup()
    M = K_zz_op.in_size()
    g = WhitenedGuide.init(num_inducing=M)
    mu, var = g.predict(K_xz, K_zz_op, K_xx_diag)
    assert jnp.allclose(mu, jnp.zeros_like(mu))
    assert jnp.allclose(var, K_xx_diag, atol=1e-8)


# --- whitened/unwhitened equivalence --------------------------------------


def test_whitened_and_unwhitened_predict_agree_under_change_of_variables():
    """``u = L_zz v`` change of variables: same predictive both ways."""
    _, _, _, K_zz_op, K_xz, K_xx_diag = _toy_setup()
    M = K_zz_op.in_size()
    L_zz = jnp.linalg.cholesky(K_zz_op.as_matrix())

    m_v = jr.normal(jr.PRNGKey(7), (M,))
    L_v_raw = jr.uniform(jr.PRNGKey(8), (M, M)) * 0.3
    L_v = jnp.tril(L_v_raw) + 0.3 * jnp.eye(M)

    g_w = WhitenedGuide(mean=m_v, scale_tril=L_v)
    mu_w, var_w = g_w.predict(K_xz, K_zz_op, K_xx_diag)

    # Equivalent unwhitened: u_mean = L_zz @ m_v; u_cov = L_zz S_v L_zz^T
    g_full = FullRankGuide(mean=L_zz @ m_v, scale_tril=L_zz @ L_v)
    mu_f, var_f = g_full.predict(K_xz, K_zz_op, K_xx_diag)

    assert jnp.allclose(mu_w, mu_f, atol=1e-8)
    assert jnp.allclose(var_w, var_f, atol=1e-8)


def test_whitened_and_unwhitened_kl_agree_under_change_of_variables():
    """``KL(q(v) || N(0,I)) == KL(q(u) || N(0, K_zz))`` — KL is invariant
    under invertible affine reparameterization."""
    _, _, _, K_zz_op, _, _ = _toy_setup()
    M = K_zz_op.in_size()
    L_zz = jnp.linalg.cholesky(K_zz_op.as_matrix())

    m_v = jr.normal(jr.PRNGKey(13), (M,))
    L_v_raw = jr.uniform(jr.PRNGKey(14), (M, M)) * 0.3
    L_v = jnp.tril(L_v_raw) + 0.3 * jnp.eye(M)

    kl_w = WhitenedGuide(mean=m_v, scale_tril=L_v).kl_divergence(K_zz_op)
    kl_u = FullRankGuide(mean=L_zz @ m_v, scale_tril=L_zz @ L_v).kl_divergence(K_zz_op)
    assert jnp.allclose(kl_w, kl_u, atol=1e-6)
