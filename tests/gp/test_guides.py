"""Tests for the sparse SVGP variational guide families.

Verifies for each of :class:`FullRankGuide`, :class:`MeanFieldGuide`,
:class:`WhitenedGuide`, :class:`NaturalGuide`, and :class:`DeltaGuide`:

* sample / log_prob shape and value (Gaussian density closed form,
  deterministic for ``DeltaGuide``).
* KL divergence matches the standard Gaussian closed form (or the
  loc-dependent ``-log p(loc)`` for ``DeltaGuide``).
* Predictive mean/variance match a direct numpy implementation.
* The whitened/unwhitened equivalence: a whitened guide ``q(v) = N(m_v,
  L_v L_v^T)`` and its unwhitened equivalent ``q(u) = N(L_zz m_v,
  L_zz L_v L_v^T L_zz^T)`` produce identical KL and predictive moments.
* Natural<->moment round-trips and the natural-parameter damped update.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpyro.distributions as nd
from gaussx import (
    damped_natural_update,
    mean_cov_to_natural,
    natural_to_mean_cov,
)

from pyrox.gp import (
    RBF,
    DeltaGuide,
    FullRankGuide,
    MeanFieldGuide,
    NaturalGuide,
    SparseGPPrior,
    WhitenedGuide,
)


def _moments_to_natural_arrays(
    m: jnp.ndarray, S: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Array-form wrapper around :func:`gaussx.mean_cov_to_natural`."""
    S_op = lx.MatrixLinearOperator(S, lx.positive_semidefinite_tag)
    eta1, eta2_op = mean_cov_to_natural(m, S_op)
    return eta1, eta2_op.as_matrix()


def _natural_to_moments_arrays(
    eta1: jnp.ndarray, eta2: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Array-form wrapper around :func:`gaussx.natural_to_mean_cov`."""
    eta2_op = lx.MatrixLinearOperator(eta2, lx.negative_semidefinite_tag)
    m, S_op = natural_to_mean_cov(eta1, eta2_op)
    return m, S_op.as_matrix()


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
    # Loosened to accommodate the dtype-aware Cholesky jitter inside
    # `_svgp_predict_unwhitened` (~10 * eps for float32).
    assert jnp.allclose(var, var_ref, atol=1e-4)


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
    # Loosened to accommodate the dtype-aware Cholesky jitter inside
    # `_svgp_predict_unwhitened` (~10 * eps for float32).
    assert jnp.allclose(var, var_ref, atol=1e-4)


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

    assert jnp.allclose(mu_w, mu_f, atol=1e-6)
    # The unwhitened path adds a dtype-aware Cholesky jitter (~10 * eps
    # for float32) inside `_svgp_predict_unwhitened` that the whitened
    # path does not, so the two predictives agree only up to that jitter.
    assert jnp.allclose(var_w, var_f, atol=1e-4)


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


# --- Natural<->moment conversion utilities (gaussx primitives) ------------


def _toy_moments(M: int = 4, seed: int = 0) -> tuple[jnp.ndarray, jnp.ndarray]:
    """A small mean and SPD covariance for round-trip tests."""
    key_m, key_L = jr.split(jr.PRNGKey(seed))
    m = jr.normal(key_m, (M,))
    L_raw = jr.uniform(key_L, (M, M)) * 0.3
    L = jnp.tril(L_raw) + 0.5 * jnp.eye(M)
    return m, L @ L.T


def test_gaussx_natural_round_trip_through_pyrox_wrappers():
    """``gaussx.natural_to_mean_cov(gaussx.mean_cov_to_natural(m, S)) == (m, S)``.

    Sanity check that the gaussx primitives :class:`NaturalGuide`
    delegates to round-trip cleanly via the operator-form API. The
    pyrox-side guide is the ergonomic shell on top of these primitives;
    the math itself is gaussx's responsibility.
    """
    m, S = _toy_moments(M=5, seed=1)
    eta1, eta2 = _moments_to_natural_arrays(m, S)
    m_back, S_back = _natural_to_moments_arrays(eta1, eta2)
    assert jnp.allclose(m, m_back, atol=1e-5)
    assert jnp.allclose(S, S_back, atol=1e-5)


def test_gaussx_natural_round_trip_starting_from_naturals():
    """The reverse round-trip also recovers the inputs exactly (up to atol)."""
    m, S = _toy_moments(M=4, seed=2)
    eta1, eta2 = _moments_to_natural_arrays(m, S)
    eta1_back, eta2_back = _moments_to_natural_arrays(
        *_natural_to_moments_arrays(eta1, eta2)
    )
    assert jnp.allclose(eta1, eta1_back, atol=1e-5)
    assert jnp.allclose(eta2, eta2_back, atol=1e-5)


def test_gaussx_natural_parameters_match_textbook_formulas():
    """``eta_1 = S^{-1} m`` and ``eta_2 = -1/2 S^{-1}`` exactly."""
    m, S = _toy_moments(M=3, seed=3)
    eta1, eta2 = _moments_to_natural_arrays(m, S)
    S_inv = jnp.linalg.inv(S)
    assert jnp.allclose(eta1, S_inv @ m, atol=1e-5)
    assert jnp.allclose(eta2, -0.5 * S_inv, atol=1e-5)


# --- NaturalGuide ----------------------------------------------------------


def test_natural_init_returns_zero_mean_unit_scale_in_moment_space():
    """Initialized natural parameters map back to ``N(0, scale^2 I)``."""
    g = NaturalGuide.init(num_inducing=5, scale=2.0)
    assert g.nat1.shape == (5,)
    assert jnp.allclose(g.nat1, 0.0)
    assert jnp.allclose(g.mean, 0.0, atol=1e-6)
    assert jnp.allclose(g.covariance, 4.0 * jnp.eye(5), atol=1e-5)


def test_natural_mean_and_covariance_recover_moments():
    """``mean`` / ``covariance`` properties round-trip through
    :func:`gaussx.natural_to_mean_cov`."""
    m, S = _toy_moments(M=4, seed=5)
    eta1, eta2 = _moments_to_natural_arrays(m, S)
    g = NaturalGuide(nat1=eta1, nat2=eta2)
    assert jnp.allclose(g.mean, m, atol=1e-5)
    assert jnp.allclose(g.covariance, S, atol=1e-5)


def test_natural_sample_and_log_prob_match_numpyro_mvn():
    M = 4
    m, S = _toy_moments(M=M, seed=6)
    eta1, eta2 = _moments_to_natural_arrays(m, S)
    g = NaturalGuide(nat1=eta1, nat2=eta2)
    u = g.sample(jr.PRNGKey(11))
    assert u.shape == (M,)
    ref = nd.MultivariateNormal(m, covariance_matrix=S)
    assert jnp.allclose(g.log_prob(u), ref.log_prob(u), atol=1e-5)


def test_natural_kl_matches_numpyro_kl_divergence():
    _, _, _, K_zz_op, _, _ = _toy_setup()
    M = K_zz_op.in_size()
    m, S = _toy_moments(M=M, seed=7)
    eta1, eta2 = _moments_to_natural_arrays(m, S)
    g = NaturalGuide(nat1=eta1, nat2=eta2)

    K_zz = K_zz_op.as_matrix()
    q = nd.MultivariateNormal(m, covariance_matrix=S)
    p = nd.MultivariateNormal(jnp.zeros(M), covariance_matrix=K_zz)
    ref = nd.kl_divergence(q, p)
    got = g.kl_divergence(K_zz_op)
    assert jnp.allclose(got, ref, atol=1e-5)


def test_natural_predict_matches_direct_closed_form():
    _, _, _, K_zz_op, K_xz, K_xx_diag = _toy_setup()
    M = K_zz_op.in_size()
    m, S = _toy_moments(M=M, seed=8)
    eta1, eta2 = _moments_to_natural_arrays(m, S)
    g = NaturalGuide(nat1=eta1, nat2=eta2)
    mu, var = g.predict(K_xz, K_zz_op, K_xx_diag)
    K_zz = K_zz_op.as_matrix()
    mu_ref, var_ref = _direct_sparse_predictive(K_xz, K_zz, K_xx_diag, m, S)
    assert jnp.allclose(mu, mu_ref, atol=1e-5)
    # Loosened to accommodate the dtype-aware Cholesky jitter inside
    # `_svgp_predict_unwhitened` (~10 * eps for float32).
    assert jnp.allclose(var, var_ref, atol=1e-4)


def test_natural_and_fullrank_predict_agree_under_reparameterization():
    """A :class:`NaturalGuide` with parameters from ``(m, S = L L^T)`` and
    a :class:`FullRankGuide` with the same ``(m, L)`` give identical
    predictives — the natural parameterization is just a change of
    variables that leaves the predictive untouched."""
    _, _, _, K_zz_op, K_xz, K_xx_diag = _toy_setup()
    M = K_zz_op.in_size()
    m, S = _toy_moments(M=M, seed=9)
    L = jnp.linalg.cholesky(S)

    eta1, eta2 = _moments_to_natural_arrays(m, S)
    mu_nat, var_nat = NaturalGuide(nat1=eta1, nat2=eta2).predict(
        K_xz, K_zz_op, K_xx_diag
    )
    mu_full, var_full = FullRankGuide(mean=m, scale_tril=L).predict(
        K_xz, K_zz_op, K_xx_diag
    )
    assert jnp.allclose(mu_nat, mu_full, atol=1e-5)
    assert jnp.allclose(var_nat, var_full, atol=1e-5)


def test_natural_update_with_rho_one_replaces_natural_parameters():
    """Damping ``rho=1`` is equivalent to overwriting the natural params."""
    M = 3
    m, S = _toy_moments(M=M, seed=10)
    eta1, eta2 = _moments_to_natural_arrays(m, S)
    g = NaturalGuide.init(num_inducing=M)
    updated = g.natural_update(eta1, eta2, rho=1.0)
    assert jnp.allclose(updated.nat1, eta1, atol=1e-6)
    assert jnp.allclose(updated.nat2, eta2, atol=1e-6)


def test_natural_update_with_rho_zero_is_identity():
    """Damping ``rho=0`` returns a guide with the original natural params."""
    M = 3
    m, S = _toy_moments(M=M, seed=11)
    eta1, eta2 = _moments_to_natural_arrays(m, S)
    g = NaturalGuide(nat1=eta1, nat2=eta2)
    bogus_eta1 = jnp.ones(M)
    bogus_eta2 = -3.0 * jnp.eye(M)
    updated = g.natural_update(bogus_eta1, bogus_eta2, rho=0.0)
    assert jnp.allclose(updated.nat1, eta1, atol=1e-6)
    assert jnp.allclose(updated.nat2, eta2, atol=1e-6)


def test_natural_update_is_convex_combination_in_natural_space():
    """The update interpolates linearly between the two natural-parameter
    endpoints — the property CVI/natural-gradient updates rely on."""
    M = 3
    m_a, S_a = _toy_moments(M=M, seed=12)
    m_b, S_b = _toy_moments(M=M, seed=13)
    eta1_a, eta2_a = _moments_to_natural_arrays(m_a, S_a)
    eta1_b, eta2_b = _moments_to_natural_arrays(m_b, S_b)
    rho = 0.3
    g = NaturalGuide(nat1=eta1_a, nat2=eta2_a)
    updated = g.natural_update(eta1_b, eta2_b, rho=rho)
    expected_nat1 = (1.0 - rho) * eta1_a + rho * eta1_b
    expected_nat2 = (1.0 - rho) * eta2_a + rho * eta2_b
    assert jnp.allclose(updated.nat1, expected_nat1, atol=1e-6)
    assert jnp.allclose(updated.nat2, expected_nat2, atol=1e-6)


def test_natural_update_matches_gaussx_damped_natural_update():
    """``NaturalGuide.natural_update`` is a pyrox-side adapter — its
    output must match :func:`gaussx.damped_natural_update` called
    directly on the same arrays."""
    M = 4
    m_a, S_a = _toy_moments(M=M, seed=14)
    m_b, S_b = _toy_moments(M=M, seed=15)
    eta1_a, eta2_a = _moments_to_natural_arrays(m_a, S_a)
    eta1_b, eta2_b = _moments_to_natural_arrays(m_b, S_b)
    rho = 0.4
    g = NaturalGuide(nat1=eta1_a, nat2=eta2_a)
    updated = g.natural_update(eta1_b, eta2_b, rho=rho)
    ref_nat1, ref_nat2 = damped_natural_update(eta1_a, eta2_a, eta1_b, eta2_b, lr=rho)
    assert jnp.allclose(updated.nat1, ref_nat1, atol=1e-6)
    assert jnp.allclose(updated.nat2, ref_nat2, atol=1e-6)


def test_natural_update_returns_a_new_guide_does_not_mutate():
    """Equinox modules are immutable — ``natural_update`` returns a new
    :class:`NaturalGuide` and leaves the receiver untouched."""
    M = 3
    g = NaturalGuide.init(num_inducing=M)
    updated = g.natural_update(jnp.ones(M), -2.0 * jnp.eye(M), rho=0.5)
    assert updated is not g
    assert jnp.allclose(g.nat1, 0.0)


# --- DeltaGuide ------------------------------------------------------------


def test_delta_init_returns_zero_loc():
    g = DeltaGuide.init(num_inducing=4)
    assert g.loc.shape == (4,)
    assert jnp.allclose(g.loc, 0.0)


def test_delta_sample_is_deterministic_and_returns_loc():
    """All keys produce the same sample — the variational draw is the loc."""
    loc = jnp.array([1.0, -2.0, 0.5])
    g = DeltaGuide(loc=loc)
    s1 = g.sample(jr.PRNGKey(0))
    s2 = g.sample(jr.PRNGKey(123456))
    assert jnp.allclose(s1, loc)
    assert jnp.allclose(s2, loc)


def test_delta_log_prob_returns_zero():
    """Constant log-prob — Pyro/NumPyro ``AutoDelta`` convention."""
    g = DeltaGuide(loc=jnp.array([0.5, -0.5]))
    assert jnp.allclose(g.log_prob(jnp.zeros(2)), 0.0)
    assert jnp.allclose(g.log_prob(jnp.array([0.5, -0.5])), 0.0)


def test_delta_kl_matches_negative_log_prior_density_at_loc():
    """``kl_divergence(prior_cov)`` == ``-log p(loc)`` for ``p = N(0, prior_cov)``."""
    _, _, _, K_zz_op, _, _ = _toy_setup()
    M = K_zz_op.in_size()
    loc = jr.normal(jr.PRNGKey(20), (M,))
    g = DeltaGuide(loc=loc)
    K_zz = K_zz_op.as_matrix()
    p = nd.MultivariateNormal(jnp.zeros(M), covariance_matrix=K_zz)
    expected = -p.log_prob(loc)
    got = g.kl_divergence(K_zz_op)
    assert jnp.allclose(got, expected, atol=1e-5)


def test_delta_predict_matches_prior_conditioning_on_u_equals_loc():
    """With ``u = loc`` deterministically, the predictive variance is the
    prior reduction ``k(x, x) - K_xz K_zz^{-1} K_zx`` — no posterior
    uncertainty contribution."""
    _, _, _, K_zz_op, K_xz, K_xx_diag = _toy_setup()
    M = K_zz_op.in_size()
    loc = jr.normal(jr.PRNGKey(21), (M,))
    g = DeltaGuide(loc=loc)
    mu, var = g.predict(K_xz, K_zz_op, K_xx_diag)
    K_zz = K_zz_op.as_matrix()
    mu_ref, var_ref = _direct_sparse_predictive(
        K_xz, K_zz, K_xx_diag, loc, jnp.zeros((M, M))
    )
    assert jnp.allclose(mu, mu_ref, atol=1e-5)
    # Loosened to accommodate the dtype-aware Cholesky jitter inside
    # `_svgp_predict_unwhitened` (~10 * eps for float32).
    assert jnp.allclose(var, var_ref, atol=1e-4)


def test_delta_predict_variance_is_at_most_prior_diag():
    """Conditioning on ``u = loc`` can only *reduce* the marginal variance
    relative to the unconditional prior."""
    _, _, _, K_zz_op, K_xz, K_xx_diag = _toy_setup()
    M = K_zz_op.in_size()
    g = DeltaGuide(loc=jnp.zeros(M))
    _, var = g.predict(K_xz, K_zz_op, K_xx_diag)
    assert jnp.all(var <= K_xx_diag + 1e-5)


def test_delta_kl_with_zero_loc_is_just_log_normalizer():
    """At ``loc = 0`` the quadratic form vanishes and KL reduces to
    ``1/2 (log|K| + M log(2 pi))`` — the log-partition of the prior."""
    _, _, _, K_zz_op, _, _ = _toy_setup()
    M = K_zz_op.in_size()
    g = DeltaGuide.init(num_inducing=M)
    K_zz = K_zz_op.as_matrix()
    sign, logdet = jnp.linalg.slogdet(K_zz)
    expected = 0.5 * (logdet + M * jnp.log(2.0 * jnp.pi))
    assert sign > 0
    assert jnp.allclose(g.kl_divergence(K_zz_op), expected, atol=1e-5)
