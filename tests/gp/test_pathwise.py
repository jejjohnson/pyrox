"""Tests for pathwise GP posterior samplers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pyrox.gp import (
    RBF,
    ConditionedGP,
    DecoupledPathwiseSampler,
    FourierInducingFeatures,
    FullRankGuide,
    GPPrior,
    Matern,
    MeanFieldGuide,
    PathwiseSampler,
    SparseGPPrior,
    WhitenedGuide,
)


def _toy_dataset(n: int = 6) -> tuple[jnp.ndarray, jnp.ndarray]:
    X = jnp.linspace(-1.0, 1.0, n).reshape(-1, 1)
    y = jnp.sin(2.0 * X).squeeze(-1)
    return X, y


def _exact_posterior_moments(
    posterior: ConditionedGP,
    X_star: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Analytic GP posterior ``(mean, covariance)`` at ``X_star``."""
    prior = posterior.prior
    kernel = prior.kernel
    X = prior.X
    K_xx = kernel(X, X) + (prior.jitter + posterior.noise_var) * jnp.eye(X.shape[0])
    K_star_x = kernel(X_star, X)
    K_star_star = kernel(X_star, X_star)
    mean = posterior.predict_mean(X_star)
    cov = K_star_star - K_star_x @ jnp.linalg.solve(K_xx, K_star_x.T)
    return mean, cov


def test_dense_pathwise_paths_are_batch_consistent():
    X, y = _toy_dataset()
    posterior = GPPrior(
        kernel=RBF(init_variance=1.2, init_lengthscale=0.4),
        X=X,
        jitter=1e-8,
    ).condition(y, jnp.array(0.05))
    sampler = PathwiseSampler(posterior, n_features=512)
    paths = sampler.sample_paths(jax.random.PRNGKey(0), n_paths=4)

    full = paths(X)
    split = jnp.concatenate([paths(X[:3]), paths(X[3:])], axis=1)

    assert full.shape == (4, X.shape[0])
    assert jnp.allclose(full, split, atol=1e-6)


def test_dense_pathwise_empirical_moments_match_posterior_sanity():
    X, y = _toy_dataset(n=5)
    X_star = jnp.array([[-0.75], [0.0], [0.8]])
    posterior = GPPrior(
        kernel=RBF(init_variance=1.0, init_lengthscale=0.35),
        X=X,
        jitter=1e-8,
    ).condition(y, jnp.array(0.02))
    draws = PathwiseSampler(posterior, n_features=2048)(
        jax.random.PRNGKey(1),
        X_star,
        n_paths=512,
    )

    empirical_mean = jnp.mean(draws, axis=0)
    centered = draws - empirical_mean
    empirical_cov = centered.T @ centered / (draws.shape[0] - 1)

    exact_mean, exact_cov = _exact_posterior_moments(posterior, X_star)

    assert jnp.allclose(empirical_mean, exact_mean, atol=0.12)
    assert jnp.allclose(jnp.diag(empirical_cov), jnp.diag(exact_cov), atol=0.12)


def test_dense_pathwise_matern_runs_and_has_sane_moments():
    X, y = _toy_dataset(n=5)
    X_star = jnp.array([[-0.5], [0.3]])
    posterior = GPPrior(
        kernel=Matern(init_variance=1.0, init_lengthscale=0.4, nu=1.5),
        X=X,
        jitter=1e-6,
    ).condition(y, jnp.array(0.05))
    draws = PathwiseSampler(posterior, n_features=4096)(
        jax.random.PRNGKey(11),
        X_star,
        n_paths=512,
    )

    empirical_mean = jnp.mean(draws, axis=0)
    exact_mean, exact_cov = _exact_posterior_moments(posterior, X_star)

    assert draws.shape == (512, X_star.shape[0])
    assert jnp.all(jnp.isfinite(draws))
    # Loose tolerance — Matern RFF converges more slowly than RBF.
    assert jnp.allclose(empirical_mean, exact_mean, atol=0.3)
    empirical_var = jnp.var(draws, axis=0)
    assert jnp.allclose(empirical_var, jnp.diag(exact_cov), atol=0.3)


def test_dense_pathwise_handles_d_gt_1():
    key = jax.random.PRNGKey(17)
    X = jax.random.uniform(key, (8, 2), minval=-1.0, maxval=1.0)
    y = jnp.sin(X[:, 0]) * jnp.cos(X[:, 1])
    posterior = GPPrior(
        kernel=RBF(init_variance=1.0, init_lengthscale=0.6),
        X=X,
        jitter=1e-6,
    ).condition(y, jnp.array(0.05))

    X_star = jax.random.uniform(jax.random.PRNGKey(18), (5, 2))
    paths = PathwiseSampler(posterior, n_features=1024).sample_paths(
        jax.random.PRNGKey(19), n_paths=3
    )
    samples = paths(X_star)

    assert samples.shape == (3, X_star.shape[0])
    assert jnp.all(jnp.isfinite(samples))

    # Batch consistency in D > 1 too.
    split = jnp.concatenate([paths(X_star[:2]), paths(X_star[2:])], axis=1)
    assert jnp.allclose(samples, split, atol=1e-6)


def test_dense_pathwise_noiseless_stays_close_to_training_values():
    X, y = _toy_dataset(n=4)
    posterior = GPPrior(
        kernel=RBF(init_variance=1.0, init_lengthscale=0.5),
        X=X,
        jitter=1e-6,
    ).condition(y, jnp.array(0.0))

    draws = PathwiseSampler(posterior, n_features=2048)(
        jax.random.PRNGKey(3),
        X,
        n_paths=64,
    )

    # With zero observation noise the posterior at training points
    # concentrates tightly around y (slack from jitter + RFF truncation).
    assert jnp.allclose(jnp.mean(draws, axis=0), y, atol=0.05)
    assert jnp.max(jnp.std(draws, axis=0)) < 0.15


def test_dense_pathwise_jit_and_grad_are_finite():
    X, y = _toy_dataset(n=5)
    key = jax.random.PRNGKey(7)

    def loss(log_scale: jnp.ndarray) -> jnp.ndarray:
        posterior = GPPrior(
            kernel=RBF(init_variance=1.0, init_lengthscale=0.5),
            X=X,
            jitter=1e-6,
        ).condition(y, jnp.array(0.05))
        draws = PathwiseSampler(posterior, n_features=64)(key, X, n_paths=1) * jnp.exp(
            log_scale
        )
        return jnp.sum(draws**2)

    value = jax.jit(loss)(jnp.array(0.0))
    grad = jax.jit(jax.grad(loss))(jnp.array(0.0))

    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)


def test_decoupled_pathwise_paths_are_batch_consistent():
    Z = jnp.linspace(-1.5, 1.5, 4).reshape(-1, 1)
    prior = SparseGPPrior(
        kernel=RBF(init_variance=0.9, init_lengthscale=0.6),
        Z=Z,
        jitter=1e-8,
    )
    guide = FullRankGuide.init(Z.shape[0], scale=0.25)
    sampler = DecoupledPathwiseSampler(prior, guide, n_features=512)
    X = jnp.linspace(-2.0, 2.0, 7).reshape(-1, 1)
    paths = sampler.sample_paths(jax.random.PRNGKey(2), n_paths=3)

    full = paths(X)
    split = jnp.concatenate([paths(X[:2]), paths(X[2:5]), paths(X[5:])], axis=1)

    assert full.shape == (3, X.shape[0])
    assert jnp.allclose(full, split, atol=1e-6)


def test_decoupled_pathwise_empirical_moments_match_svgp_predict():
    Z = jnp.linspace(-1.5, 1.5, 5).reshape(-1, 1)
    kernel = RBF(init_variance=1.0, init_lengthscale=0.5)
    prior = SparseGPPrior(kernel=kernel, Z=Z, jitter=1e-8)
    guide = FullRankGuide(
        mean=jnp.linspace(-0.3, 0.3, Z.shape[0]),
        scale_tril=0.2 * jnp.eye(Z.shape[0]),
    )

    X_star = jnp.array([[-0.9], [0.0], [1.1]])
    K_zz_op, K_xz, K_xx_diag = prior.predictive_blocks(X_star)
    svgp_mean, svgp_var = guide.predict(K_xz, K_zz_op, K_xx_diag)

    draws = DecoupledPathwiseSampler(prior, guide, n_features=4096)(
        jax.random.PRNGKey(42),
        X_star,
        n_paths=512,
    )
    empirical_mean = jnp.mean(draws, axis=0)
    empirical_var = jnp.var(draws, axis=0)

    assert jnp.allclose(empirical_mean, svgp_mean, atol=0.15)
    assert jnp.allclose(empirical_var, svgp_var, atol=0.15)


def test_decoupled_pathwise_whitened_guide_matches_equivalent_unwhitened():
    Z = jnp.linspace(-1.0, 1.0, 4).reshape(-1, 1)
    kernel = RBF(init_variance=1.0, init_lengthscale=0.5)
    prior = SparseGPPrior(kernel=kernel, Z=Z, jitter=1e-8)

    # Whitened guide over v; the equivalent unwhitened FullRankGuide
    # over u = L_zz v defines the same q(u), so the two samplers must
    # produce matching empirical moments at X_star.
    whitened = WhitenedGuide(
        mean=jnp.array([0.1, -0.2, 0.05, 0.0]),
        scale_tril=0.25 * jnp.eye(Z.shape[0]),
    )
    L_zz = jnp.linalg.cholesky(prior.inducing_operator().as_matrix())
    unwhitened = FullRankGuide(
        mean=L_zz @ whitened.mean,
        scale_tril=L_zz @ whitened.scale_tril,
    )

    X_star = jnp.array([[-0.5], [0.2], [0.8]])
    draws_w = DecoupledPathwiseSampler(prior, whitened, n_features=4096)(
        jax.random.PRNGKey(100),
        X_star,
        n_paths=512,
    )
    draws_u = DecoupledPathwiseSampler(prior, unwhitened, n_features=4096)(
        jax.random.PRNGKey(100),
        X_star,
        n_paths=512,
    )

    assert jnp.allclose(jnp.mean(draws_w, axis=0), jnp.mean(draws_u, axis=0), atol=0.1)
    assert jnp.allclose(jnp.var(draws_w, axis=0), jnp.var(draws_u, axis=0), atol=0.1)


def test_decoupled_pathwise_rejects_inducing_features_prior():
    features = FourierInducingFeatures.init(in_features=1, num_basis_per_dim=8, L=3.0)
    prior = SparseGPPrior(kernel=RBF(), inducing=features)
    guide = MeanFieldGuide.init(features.num_features, scale=0.2)
    with pytest.raises(ValueError, match="point-inducing"):
        DecoupledPathwiseSampler(prior, guide)


def test_decoupled_pathwise_determinism_under_same_key():
    Z = jnp.linspace(-1.0, 1.0, 3).reshape(-1, 1)
    prior = SparseGPPrior(kernel=RBF(), Z=Z, jitter=1e-8)
    guide = FullRankGuide.init(Z.shape[0], scale=0.1)
    sampler = DecoupledPathwiseSampler(prior, guide, n_features=128)
    key = jax.random.PRNGKey(77)
    X_star = jnp.array([[-0.3], [0.4]])

    first = sampler(key, X_star, n_paths=2)
    second = sampler(key, X_star, n_paths=2)

    assert jnp.allclose(first, second, atol=0.0)
