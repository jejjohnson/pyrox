"""Tests for ``pyrox.gp`` likelihood families.

Covers :class:`GaussianLikelihood` and :class:`DistLikelihood`, verifying
that ``log_prob`` matches reference numpyro distributions and that the
``DistLikelihood`` wrapper handles arbitrary observation models.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as nd

from pyrox.gp import DistLikelihood, GaussianLikelihood


# --- GaussianLikelihood ---------------------------------------------------


def test_gaussian_log_prob_matches_numpyro_normal():
    f = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.1, 1.9, 3.2])
    noise_var = 0.25
    lik = GaussianLikelihood(noise_var=noise_var)
    ref = nd.Normal(f, jnp.sqrt(noise_var)).log_prob(y).sum()
    assert jnp.allclose(lik.log_prob(f, y), ref, atol=1e-5)


def test_gaussian_log_prob_zero_residual_is_maximum():
    f = jnp.ones(4)
    lik = GaussianLikelihood(noise_var=0.1)
    lp_zero = lik.log_prob(f, f)
    lp_shift = lik.log_prob(f, f + 0.5)
    assert lp_zero > lp_shift


def test_gaussian_log_prob_dtype_preserved():
    f = jnp.array([1.0, 2.0], dtype=jnp.float32)
    y = jnp.array([1.1, 2.1], dtype=jnp.float32)
    lik = GaussianLikelihood(noise_var=0.1)
    assert lik.log_prob(f, y).dtype == f.dtype


# --- DistLikelihood -------------------------------------------------------


def test_dist_bernoulli_log_prob_matches_numpyro():
    f = jnp.array([0.5, -0.5, 1.0])
    y = jnp.array([1.0, 0.0, 1.0])
    lik = DistLikelihood(dist_fn=lambda f: nd.Bernoulli(logits=f))
    ref = nd.Bernoulli(logits=f).log_prob(y).sum()
    assert jnp.allclose(lik.log_prob(f, y), ref, atol=1e-5)


def test_dist_poisson_log_prob_matches_numpyro():
    f = jnp.array([0.5, 1.0, -0.5])
    y = jnp.array([1, 3, 0], dtype=jnp.int32)
    lik = DistLikelihood(dist_fn=lambda f: nd.Poisson(rate=jnp.exp(f)))
    ref = nd.Poisson(rate=jnp.exp(f)).log_prob(y).sum()
    assert jnp.allclose(lik.log_prob(f, y), ref, atol=1e-5)


def test_dist_student_t_log_prob_matches_numpyro():
    f = jnp.array([0.0, 1.0, -1.0])
    y = jnp.array([0.1, 0.9, -1.2])
    lik = DistLikelihood(dist_fn=lambda f: nd.StudentT(df=3.0, loc=f, scale=0.5))
    ref = nd.StudentT(df=3.0, loc=f, scale=0.5).log_prob(y).sum()
    assert jnp.allclose(lik.log_prob(f, y), ref, atol=1e-5)


def test_dist_likelihood_is_a_pyrox_likelihood():
    from pyrox.gp._protocols import Likelihood

    lik = DistLikelihood(dist_fn=lambda f: nd.Normal(f, 1.0))
    assert isinstance(lik, Likelihood)


def test_gaussian_likelihood_is_a_pyrox_likelihood():
    from pyrox.gp._protocols import Likelihood

    lik = GaussianLikelihood(noise_var=0.1)
    assert isinstance(lik, Likelihood)
