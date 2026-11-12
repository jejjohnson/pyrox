"""The Markov strategies on an observation model, through `build`.

`as_gaussx_likelihood` takes an `pyrox.likelihoods.AbstractObservation`
through the one-site gaussx likelihood it builds, and any other
`pyrox.likelihoods.Likelihood` through ``log_prob`` (gh-375). The
observation models here are test-local (pyrox-ssm's tests import nothing
from pyrox-gp); each is compared with the protocol likelihood of the same
family in `ssm_reference`, which is the path the strategies took before.
Both sides evaluate one density through exact formulas, so they agree to
float64 round-off; ``rtol=1e-10`` with a ``1e-12`` floor where a derivative
crosses zero.
"""

from __future__ import annotations

import equinox as eqx
import gaussx
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from pyrox.likelihoods import AbstractObservation
from pyrox_ssm import (
    ExpectationPropagationMarkov,
    LaplaceMarkovInference,
    MarkovGPPrior,
    MaternSDE,
    PosteriorLinearizationMarkov,
)
from pyrox_ssm._markov_sites import _cavity_site_derivatives, as_gaussx_likelihood
from ssm_reference import BernoulliLikelihood, PoissonLikelihood


RTOL = 1e-10
N = 30


class _BernoulliObservation(AbstractObservation):
    name: str = eqx.field(static=True, default="bernoulli")

    def build(self, y, theta, data):
        return gaussx.BernoulliLikelihood(y)


class _PoissonObservation(AbstractObservation):
    name: str = eqx.field(static=True, default="poisson")

    def build(self, y, theta, data):
        return gaussx.PoissonLikelihood(y)


def _cases():
    y_binary = jr.bernoulli(jr.key(0), 0.4, (N,)).astype(jnp.float64)
    y_count = jr.poisson(jr.key(1), 3.0, (N,)).astype(jnp.float64)
    return {
        "bernoulli": (_BernoulliObservation(), BernoulliLikelihood(), y_binary),
        "poisson": (_PoissonObservation(), PoissonLikelihood(), y_count),
    }


NAMES = sorted(_cases())
F = 2.0 * jr.normal(jr.key(2), (N,))
VAR = jnp.exp(jr.normal(jr.key(3), (N,)) - 1.0)
MASK = jr.bernoulli(jr.key(4), 0.8, (N,))


@pytest.mark.parametrize("masked", [False, True], ids=["dense", "masked"])
@pytest.mark.parametrize("name", NAMES)
def test_site_view_of_an_observation_equals_the_protocol_path(name, masked) -> None:
    observation, protocol, y = _cases()[name]
    mask = MASK if masked else None
    new = as_gaussx_likelihood(observation, y, mask=mask)
    old = as_gaussx_likelihood(protocol, y, mask=mask)
    assert jnp.allclose(new.log_prob(F), old.log_prob(F), rtol=RTOL, atol=0.0)
    for a, b in zip(new.site_derivatives(F), old.site_derivatives(F), strict=True):
        assert jnp.allclose(a, b, rtol=RTOL, atol=1e-12)
    rule = gaussx.GaussHermiteIntegrator(order=20)
    for a, b in zip(
        _cavity_site_derivatives(new, F, VAR, rule),
        _cavity_site_derivatives(old, F, VAR, rule),
        strict=True,
    ):
        assert jnp.allclose(a, b, rtol=RTOL, atol=1e-12)


@pytest.mark.parametrize(
    "strategy",
    [
        LaplaceMarkovInference(max_iter=5),
        PosteriorLinearizationMarkov(max_iter=3),
        ExpectationPropagationMarkov(max_iter=3),
    ],
    ids=["laplace", "pl", "ep"],
)
def test_markov_fit_on_an_observation_equals_the_protocol_path(strategy) -> None:
    times = jnp.linspace(0.0, 5.0, 12)
    y = (jnp.sin(2.0 * times) > 0.0).astype(jnp.float64)
    prior = MarkovGPPrior(MaternSDE(variance=1.0, lengthscale=0.7, order=1), times)
    new = strategy.fit(prior, _BernoulliObservation(), y)
    old = strategy.fit(prior, BernoulliLikelihood(), y)
    assert jnp.allclose(new.q_mean, old.q_mean, rtol=RTOL, atol=1e-12)
    assert jnp.allclose(
        new.log_marginal_approx, old.log_marginal_approx, rtol=RTOL, atol=0.0
    )


class _InputNoiseGaussian(AbstractObservation):
    r"""$y_i \sim N(f_i, \sigma(x_i)^2)$, $\sigma(x) = 0.2 + 0.5 \lvert x \rvert$."""

    name: str = eqx.field(static=True, default="input_noise")

    def build(self, y, theta, data):
        std = 0.2 + 0.5 * jnp.abs(data["X"][:, 0])
        return gaussx.CallableLikelihood(
            lambda f_i, d: dist.Normal(f_i, d["std"]).log_prob(d["y"]),
            {"y": y, "std": std},
        )


def test_an_input_dependent_observation_reads_each_site_its_input() -> None:
    X = jnp.linspace(-2.0, 2.0, 7)[:, None]
    y = jnp.sin(X[:, 0])
    f = jnp.zeros_like(y)
    std = 0.2 + 0.5 * jnp.abs(X[:, 0])
    g, h = as_gaussx_likelihood(_InputNoiseGaussian(), y, X).site_derivatives(f)
    # ∂ log N(y; f, s²)/∂f = (y - f) / s², ∂² = -1 / s²
    assert jnp.allclose(g, (y - f) / std**2, rtol=1e-12, atol=0.0)
    assert jnp.allclose(h, -1.0 / std**2, rtol=1e-12, atol=0.0)
