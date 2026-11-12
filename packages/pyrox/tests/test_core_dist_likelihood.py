"""``pyrox.likelihoods.DistLikelihood``, an observation model over a distribution.

The core imports no gaussx, so these tests evaluate the ``(log_density,
data)`` pair that `DistLikelihood.build` returns the way
`gaussx.CallableLikelihood` does, with ``jax.vmap`` over the sites. Both
sides evaluate the same NumPyro density, so they agree to float64 round-off
(``rtol=1e-12``).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as nd
import pytest
from pyrox.likelihoods import AbstractObservation, DistLikelihood, Likelihood


RTOL = 1e-12
N = 7
F = jr.normal(jr.key(0), (N,))
Y_COUNT = jr.poisson(jr.key(1), 2.0, (N,)).astype(jnp.float64)


def _zip(eta, theta, data):
    return nd.ZeroInflatedPoisson(gate=theta["pi"], rate=jnp.exp(eta))


def _callable_log_prob(pair, f):
    """``gaussx.CallableLikelihood(*pair).log_prob(f)``, without gaussx."""
    log_density, data = pair
    return jnp.sum(jax.vmap(log_density)(f, data))


def test_it_is_an_observation_model_and_a_likelihood():
    lik = DistLikelihood(_zip, theta={"pi": 0.2})
    assert isinstance(lik, AbstractObservation)
    assert isinstance(lik, Likelihood)
    assert lik.latent_dim == 1
    assert lik.name == "dist"
    assert lik.theta_spec() == {}


def test_build_returns_the_per_site_log_density_and_data():
    lik = DistLikelihood(_zip, theta={"pi": 0.2})
    log_density, data = lik.build(Y_COUNT, {}, {})
    assert set(data) == {"y", "data"}
    assert data["y"].shape == (N,)
    expected = _zip(F, {"pi": 0.2}, {}).log_prob(Y_COUNT)
    per_site = jax.vmap(log_density)(F, data)
    assert jnp.allclose(per_site, expected, rtol=RTOL, atol=0.0)


def test_log_prob_is_the_sum_of_the_per_site_log_densities():
    lik = DistLikelihood(_zip, theta={"pi": 0.3})
    pair = lik.build(Y_COUNT, {}, {})
    assert jnp.allclose(
        lik.log_prob(F, Y_COUNT), _callable_log_prob(pair, F), rtol=RTOL, atol=0.0
    )


def test_the_theta_passed_to_build_overrides_the_field():
    lik = DistLikelihood(_zip, theta={"pi": 0.3})
    pair = lik.build(Y_COUNT, {"pi": jnp.asarray(0.6)}, {})
    expected = _zip(F, {"pi": 0.6}, {}).log_prob(Y_COUNT).sum()
    assert jnp.allclose(_callable_log_prob(pair, F), expected, rtol=RTOL, atol=0.0)


def test_theta_is_a_differentiable_leaf():
    lik = DistLikelihood(_zip, theta={"pi": 0.3})
    grad = eqx.filter_grad(lambda m: m.log_prob(F, Y_COUNT))(lik)
    expected = jax.grad(lambda p: _zip(F, {"pi": p}, {}).log_prob(Y_COUNT).sum())(0.3)
    assert jnp.allclose(grad.theta["pi"], expected, rtol=RTOL, atol=0.0)


def test_each_site_reads_its_slice_of_data():
    def make_dist(eta, theta, data):
        return nd.Normal(eta, 0.2 + jnp.abs(data["X"][..., 0]))

    X = jnp.linspace(-1.0, 1.0, N)[:, None]
    y = jnp.sin(X[:, 0])
    lik = DistLikelihood(make_dist)
    expected = nd.Normal(F, 0.2 + jnp.abs(X[:, 0])).log_prob(y).sum()
    pair = lik.build(y, {}, {"X": X})
    assert jnp.allclose(_callable_log_prob(pair, F), expected, rtol=RTOL, atol=0.0)
    assert jnp.allclose(lik.log_prob(F, y, X), expected, rtol=RTOL, atol=0.0)


def test_the_old_one_argument_link_still_works():
    lik = DistLikelihood(lambda f: nd.Bernoulli(logits=f))
    y = (F > 0).astype(jnp.float64)
    expected = nd.Bernoulli(logits=F).log_prob(y).sum()
    assert jnp.allclose(lik.log_prob(F, y), expected, rtol=RTOL, atol=0.0)
    assert jnp.allclose(
        _callable_log_prob(lik.build(y, {}, {}), F), expected, rtol=RTOL, atol=0.0
    )
    # X is ignored by the old link, as before
    assert jnp.allclose(lik.log_prob(F, y, jnp.ones((N, 2))), expected, rtol=RTOL)


def test_the_dist_fn_keyword_warns_and_still_works():
    with pytest.warns(DeprecationWarning, match="make_dist") as record:
        lik = DistLikelihood(dist_fn=lambda f: nd.Poisson(jnp.exp(f)))
    # The warning points at the caller, not at equinox's metaclass.
    assert record[0].filename == __file__
    expected = nd.Poisson(jnp.exp(F)).log_prob(Y_COUNT).sum()
    assert jnp.allclose(lik.log_prob(F, Y_COUNT), expected, rtol=RTOL, atol=0.0)


def test_dist_fn_and_make_dist_together_raise():
    with pytest.raises(TypeError, match="not both"):
        DistLikelihood(_zip, dist_fn=lambda f: nd.Poisson(jnp.exp(f)))


def test_a_callable_of_the_wrong_arity_raises():
    with pytest.raises(TypeError, match="eta_i, theta, data_i"):
        DistLikelihood(lambda a, b: nd.Normal(a, b))


def test_a_batched_distribution_per_site_raises():
    """Closing over a per-site array gives each site N densities: rejected."""
    scale = jnp.ones(N)
    lik = DistLikelihood(lambda f: nd.Normal(f, scale))
    log_density, data = lik.build(Y_COUNT, {}, {})
    with pytest.raises(ValueError, match="one site's distribution"):
        jax.vmap(log_density)(F, data)


def test_site_cdf_comes_from_make_dist():
    lik = DistLikelihood(lambda eta, theta, data: nd.Normal(eta, 1.0))
    y = jnp.zeros(N)
    expected = nd.Normal(F, 1.0).cdf(y)
    assert jnp.allclose(lik.site_cdf(y, F, {}, {}), expected, rtol=RTOL, atol=0.0)
