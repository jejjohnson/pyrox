"""The Markov strategies on `pyrox.likelihoods.DistLikelihood` (gh-376).

`as_gaussx_likelihood` wraps the ``(log_density, data)`` pair that
`DistLikelihood.build` returns into a `gaussx.CallableLikelihood`. A
Bernoulli or Poisson written as a NumPyro distribution must give what the
protocol likelihoods of `ssm_reference` give; both sides evaluate one density
through exact formulas, so they agree to float64 round-off (``rtol=1e-10``,
with a ``1e-12`` floor where a derivative crosses zero). A zero-inflated
Poisson, which gaussx has no class for, then runs through the Markov Laplace
evidence inside NUTS. pyrox-ssm's tests import nothing from pyrox-gp, and
`DistLikelihood` is in the core.
"""

from __future__ import annotations

import gaussx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from pyrox.likelihoods import DistLikelihood
from pyrox_ssm import (
    ExpectationPropagationMarkov,
    LaplaceMarkovInference,
    MarkovGPPrior,
    MaternSDE,
    PosteriorLinearizationMarkov,
    markov_laplace_factor,
)
from pyrox_ssm._markov_sites import _cavity_site_derivatives, as_gaussx_likelihood
from ssm_reference import BernoulliLikelihood, PoissonLikelihood


RTOL = 1e-10
N = 30


def _cases():
    y_binary = jr.bernoulli(jr.key(0), 0.4, (N,)).astype(jnp.float64)
    y_count = jr.poisson(jr.key(1), 3.0, (N,)).astype(jnp.float64)
    return {
        "bernoulli": (
            DistLikelihood(lambda eta, th, d: dist.Bernoulli(logits=eta)),
            BernoulliLikelihood(),
            y_binary,
        ),
        "bernoulli_old_form": (
            DistLikelihood(lambda f: dist.Bernoulli(logits=f)),
            BernoulliLikelihood(),
            y_binary,
        ),
        "poisson": (
            DistLikelihood(lambda eta, th, d: dist.Poisson(jnp.exp(eta))),
            PoissonLikelihood(),
            y_count,
        ),
    }


NAMES = sorted(_cases())
F = 2.0 * jr.normal(jr.key(2), (N,))
VAR = jnp.exp(jr.normal(jr.key(3), (N,)) - 1.0)
MASK = jr.bernoulli(jr.key(4), 0.8, (N,))


@pytest.mark.parametrize("masked", [False, True], ids=["dense", "masked"])
@pytest.mark.parametrize("name", NAMES)
def test_site_view_equals_the_protocol_likelihood(name, masked) -> None:
    dist_lik, protocol, y = _cases()[name]
    mask = MASK if masked else None
    new = as_gaussx_likelihood(dist_lik, y, mask=mask)
    old = as_gaussx_likelihood(protocol, y, mask=mask)
    assert isinstance(new, gaussx.CallableLikelihood)
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
        pytest.param(LaplaceMarkovInference(max_iter=5), id="laplace"),
        pytest.param(
            PosteriorLinearizationMarkov(max_iter=3), id="pl", marks=pytest.mark.slow
        ),
        pytest.param(
            ExpectationPropagationMarkov(max_iter=3), id="ep", marks=pytest.mark.slow
        ),
    ],
)
def test_markov_fit_equals_the_protocol_likelihood(strategy) -> None:
    times = jnp.linspace(0.0, 5.0, 12)
    y = (jnp.sin(2.0 * times) > 0.0).astype(jnp.float64)
    prior = MarkovGPPrior(MaternSDE(variance=1.0, lengthscale=0.7, order=1), times)
    lik = DistLikelihood(lambda eta, th, d: dist.Bernoulli(logits=eta))
    new = strategy.fit(prior, lik, y)
    old = strategy.fit(prior, BernoulliLikelihood(), y)
    assert jnp.allclose(new.q_mean, old.q_mean, rtol=RTOL, atol=1e-12)
    assert jnp.allclose(
        new.log_marginal_approx, old.log_marginal_approx, rtol=RTOL, atol=0.0
    )


def test_each_site_reads_its_input_from_data() -> None:
    X = jnp.linspace(-2.0, 2.0, 7)[:, None]
    y = jnp.sin(X[:, 0])
    f = jnp.zeros_like(y)
    std = 0.2 + 0.5 * jnp.abs(X[:, 0])
    lik = DistLikelihood(
        lambda eta, th, d: dist.Normal(eta, 0.2 + 0.5 * jnp.abs(d["X"][..., 0]))
    )
    g, h = as_gaussx_likelihood(lik, y, X).site_derivatives(f)
    # ∂ log N(y; f, s²)/∂f = (y - f) / s², ∂² = -1 / s²
    assert jnp.allclose(g, (y - f) / std**2, rtol=1e-12, atol=0.0)
    assert jnp.allclose(h, -1.0 / std**2, rtol=1e-12, atol=0.0)


# --- a zero-inflated Poisson inside NUTS ---------------------------------------


def _zip(eta, theta, data):
    return dist.ZeroInflatedPoisson(gate=theta["pi"], rate=jnp.exp(eta))


def _zip_series(n: int):
    """Counts from a ZIP with gate 0.3 over a smooth log rate."""
    times = jnp.linspace(0.0, 10.0, n)
    log_rate = 1.0 + jnp.sin(0.8 * times)
    y = _zip(log_rate, {"pi": 0.3}, {}).sample(jr.key(5))
    return times, y.astype(jnp.float64)


def _zip_model(times, y, max_iter):
    variance = numpyro.sample("variance", dist.LogNormal(0.0, 0.5))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(1.0, 0.5))
    pi = numpyro.sample("pi", dist.Beta(2.0, 4.0))
    prior = MarkovGPPrior(MaternSDE(variance, lengthscale, order=1), times)
    markov_laplace_factor(
        "y",
        prior,
        DistLikelihood(_zip, theta={"pi": pi}),
        y,
        strategy=LaplaceMarkovInference(max_iter=max_iter),
    )


def test_zero_inflated_poisson_markov_laplace_smoke() -> None:
    """The ZIP evidence is finite and differentiable in the gate and the kernel."""
    times, y = _zip_series(12)
    params = {"variance": 1.0, "lengthscale": 2.0, "pi": 0.3}

    def evidence(p):
        model = handlers.substitute(handlers.seed(_zip_model, 0), data=p)
        with handlers.trace() as tr:
            model(times, y, 10)
        return tr["y"]["fn"].log_factor

    value, grads = jax.value_and_grad(evidence)(params)
    assert jnp.isfinite(value)
    assert all(jnp.isfinite(g) for g in grads.values())
    assert grads["pi"] != 0.0


@pytest.mark.slow
def test_zero_inflated_poisson_markov_laplace_runs_100_nuts_draws() -> None:
    times, y = _zip_series(40)
    mcmc = MCMC(NUTS(_zip_model), num_warmup=100, num_samples=100, progress_bar=False)
    mcmc.run(jr.key(0), times, y, 20, extra_fields=("diverging",))
    samples = mcmc.get_samples()
    assert samples["pi"].shape == (100,)
    for value in samples.values():
        assert jnp.all(jnp.isfinite(value))
    assert jnp.sum(mcmc.get_extra_fields()["diverging"]) <= 5
    # The gate's posterior mean lies near the true 0.3 (40 counts: loose).
    assert 0.1 < float(jnp.mean(samples["pi"])) < 0.55
