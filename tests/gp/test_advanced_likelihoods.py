"""Tests for the non-Gaussian likelihood family.

Covers shape/finiteness, scalar-vs-multi-latent declarations, and
basic agreement with hand-rolled scipy / numpyro equivalents.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as nd
import pytest

from pyrox.gp import (
    BernoulliLikelihood,
    HeteroscedasticGaussianLikelihood,
    PoissonLikelihood,
    SoftmaxLikelihood,
    StudentTLikelihood,
)


def test_bernoulli_log_prob_matches_numpyro() -> None:
    f = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    y = jnp.array([0.0, 1.0, 1.0, 0.0, 1.0])
    lp_pyrox = BernoulliLikelihood().log_prob(f, y)
    lp_ref = nd.Bernoulli(logits=f).log_prob(y).sum()
    assert jnp.allclose(lp_pyrox, lp_ref)


def test_poisson_log_prob_matches_numpyro() -> None:
    f = jnp.array([-1.0, 0.0, 1.5])
    y = jnp.array([0.0, 2.0, 5.0])
    lp_pyrox = PoissonLikelihood().log_prob(f, y)
    lp_ref = nd.Poisson(rate=jnp.exp(f)).log_prob(y).sum()
    assert jnp.allclose(lp_pyrox, lp_ref)


def test_studentt_log_prob_matches_numpyro() -> None:
    f = jnp.array([-1.0, 0.0, 1.5])
    y = jnp.array([0.5, -0.3, 1.0])
    lik = StudentTLikelihood(df=4.0, scale=0.7)
    lp_pyrox = lik.log_prob(f, y)
    lp_ref = nd.StudentT(df=4.0, loc=f, scale=0.7).log_prob(y).sum()
    assert jnp.allclose(lp_pyrox, lp_ref)


def test_softmax_log_prob_and_latent_dim() -> None:
    lik = SoftmaxLikelihood(num_classes=3)
    assert lik.latent_dim == 3
    assert lik.num_classes == 3

    f = jnp.array([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]])
    y = jnp.array([0, 1])
    lp_pyrox = lik.log_prob(f, y)
    lp_ref = nd.Categorical(logits=f).log_prob(y).sum()
    assert jnp.allclose(lp_pyrox, lp_ref)


def test_softmax_rejects_too_few_classes() -> None:
    with pytest.raises(ValueError, match="num_classes must be >= 2"):
        SoftmaxLikelihood(num_classes=1)


def test_heteroscedastic_log_prob_and_latent_dim() -> None:
    lik = HeteroscedasticGaussianLikelihood()
    assert lik.latent_dim == 2

    f = jnp.array([[0.0, -1.0], [1.0, 0.0]])  # (mean, log_scale)
    y = jnp.array([0.1, 0.9])
    lp_pyrox = lik.log_prob(f, y)
    loc, log_scale = f[..., 0], f[..., 1]
    lp_ref = nd.Normal(loc=loc, scale=jnp.exp(log_scale)).log_prob(y).sum()
    assert jnp.allclose(lp_pyrox, lp_ref)


def test_scalar_likelihoods_default_latent_dim_one() -> None:
    """Scalar likelihoods that don't declare ``latent_dim`` use the
    default of 1 via ``getattr(lik, 'latent_dim', 1)``."""
    assert getattr(BernoulliLikelihood(), "latent_dim", 1) == 1
    assert getattr(PoissonLikelihood(), "latent_dim", 1) == 1
    assert getattr(StudentTLikelihood(df=3.0, scale=1.0), "latent_dim", 1) == 1


def test_likelihoods_are_jit_compatible() -> None:
    f = jnp.array([0.5, -0.3, 1.0])
    y = jnp.array([1.0, 0.0, 1.0])

    @jax.jit
    def go(f: jax.Array, y: jax.Array) -> jax.Array:
        return BernoulliLikelihood().log_prob(f, y)

    out = go(f, y)
    assert jnp.isfinite(out)
