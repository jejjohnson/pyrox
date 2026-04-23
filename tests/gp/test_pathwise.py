"""Tests for pathwise GP posterior samplers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pyrox.gp import (
    RBF,
    DecoupledPathwiseSampler,
    FullRankGuide,
    GPPrior,
    PathwiseSampler,
    SparseGPPrior,
)


def _toy_dataset(n: int = 6) -> tuple[jnp.ndarray, jnp.ndarray]:
    X = jnp.linspace(-1.0, 1.0, n).reshape(-1, 1)
    y = jnp.sin(2.0 * X).squeeze(-1)
    return X, y


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
    kernel = RBF(init_variance=1.0, init_lengthscale=0.35)
    posterior = GPPrior(kernel=kernel, X=X, jitter=1e-8).condition(y, jnp.array(0.02))
    draws = PathwiseSampler(posterior, n_features=2048)(
        jax.random.PRNGKey(1),
        X_star,
        n_paths=512,
    )

    empirical_mean = jnp.mean(draws, axis=0)
    centered = draws - empirical_mean
    empirical_cov = centered.T @ centered / (draws.shape[0] - 1)

    K_xx = kernel(X, X) + (posterior.prior.jitter + posterior.noise_var) * jnp.eye(
        X.shape[0]
    )
    K_star_x = kernel(X_star, X)
    K_star_star = kernel(X_star, X_star)
    exact_mean = posterior.predict_mean(X_star)
    exact_cov = K_star_star - K_star_x @ jnp.linalg.solve(K_xx, K_star_x.T)

    assert jnp.allclose(empirical_mean, exact_mean, atol=0.12)
    assert jnp.allclose(jnp.diag(empirical_cov), jnp.diag(exact_cov), atol=0.12)


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
