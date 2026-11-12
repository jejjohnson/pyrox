"""``pyrox.likelihoods.AbstractObservation``: the observation-model base."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as nd
import pytest
from pyrox.likelihoods import AbstractObservation


class _Scalar(AbstractObservation):
    name: str = eqx.field(static=True, default="lik")

    def build(self, y, theta, data):
        return ("scalar", y)


class _TwoLatent(AbstractObservation):
    """A heteroscedastic Gaussian: eta[:, 0] the mean, eta[:, 1] the log scale."""

    name: str = eqx.field(static=True, default="lik")

    @property
    def latent_dim(self) -> int:
        return 2

    def theta_spec(self):
        prior = nd.Exponential(1.0)
        return {"shift": (prior, nd.biject_to(prior.support))}

    def build(self, y, theta, data):
        # Any simple object: the core tests need no gaussx.
        return {"y": y, "shift": theta["shift"]}

    def site_distribution(self, eta, theta, data):
        return nd.Normal(eta[:, 0] + theta["shift"], jnp.exp(eta[:, 1]))


class _FieldLatent(AbstractObservation):
    name: str = eqx.field(static=True, default="lik")
    latent_dim: int = eqx.field(static=True, default=3)

    def build(self, y, theta, data):
        return y


def test_the_base_is_abstract():
    with pytest.raises(TypeError):
        AbstractObservation()


def test_latent_dim_defaults_to_one_and_theta_spec_to_empty():
    obs = _Scalar()
    assert obs.latent_dim == 1
    assert obs.theta_spec() == {}


def test_a_static_field_with_a_default_overrides_latent_dim():
    assert _FieldLatent().latent_dim == 3
    assert _FieldLatent(latent_dim=4).latent_dim == 4


def test_site_methods_need_site_distribution():
    with pytest.raises(NotImplementedError, match="site_distribution"):
        _Scalar().site_log_prob(jnp.zeros(2), jnp.zeros(2), {}, {})


def test_a_two_latent_model_builds_and_evaluates_per_site():
    obs = _TwoLatent()
    n = 4
    eta = jax.random.normal(jax.random.key(0), (n, obs.latent_dim))
    y = jax.random.normal(jax.random.key(1), (n,))
    theta = {"shift": jnp.asarray(0.3)}
    assert obs.latent_dim == 2
    assert set(obs.theta_spec()) == {"shift"}

    built = obs.build(y, theta, {})
    assert built["y"] is y

    expected = nd.Normal(eta[:, 0] + 0.3, jnp.exp(eta[:, 1]))
    site_distribution = obs.site_distribution(eta, theta, {})
    assert site_distribution.batch_shape == (n,)
    log_prob = obs.site_log_prob(y, eta, theta, {})
    assert log_prob.shape == (n,)
    assert jnp.allclose(log_prob, expected.log_prob(y), rtol=1e-12)
    cdf = obs.site_cdf(y, eta, theta, {})
    assert cdf.shape == (n,)
    assert jnp.allclose(cdf, expected.cdf(y), rtol=1e-12)


def test_pyrox_lgm_re_exports_the_core_class():
    lgm = pytest.importorskip("pyrox_lgm")
    assert lgm.AbstractObservation is AbstractObservation
    assert isinstance(lgm.Gaussian(), AbstractObservation)
    assert lgm.Gaussian().latent_dim == 1
