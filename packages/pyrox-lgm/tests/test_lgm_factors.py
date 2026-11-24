"""lgm_factor and lgm_laplace_factor, the collapsed NumPyro factors."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pyrox_lgm as lgm
import pytest
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density


N = 12
T = np.arange(N)


def _prior():
    return lgm.LGMPrior((lgm.RW1(N, name="t"),), lgm.FixedEffects(("intercept",)))


def _data():
    return {"y": np.random.default_rng(0).poisson(3.0, N).astype(float), "t": T}


def _value(model):
    return float(handlers.trace(model).get_trace()["y"]["fn"].log_factor)


def test_the_factors_register_the_evidence_and_nothing_else():
    prior, data, theta = _prior(), _data(), {"t.tau": 1.7}

    def gaussian():
        lgm.lgm_factor("y", prior, theta, data, 0.3)

    def poisson():
        lgm.lgm_laplace_factor("y", prior, theta, lgm.Poisson(), data)

    for model in (gaussian, poisson):
        assert list(handlers.trace(model).get_trace()) == ["y"]
    assert _value(gaussian) == float(prior.log_marginal(theta, data, 0.3))
    evidence = lgm.LGM(prior, likelihood=lgm.Poisson()).log_marginal(theta, data)
    assert _value(poisson) == float(evidence)


def test_the_laplace_factor_is_exact_for_a_gaussian_likelihood():
    prior, data = _prior(), _data()
    theta = {"t.tau": 1.7, "lik.prec": 1.0 / 0.3}
    exact = prior.log_marginal(theta, data, 0.3)

    def model():
        lgm.lgm_laplace_factor("y", prior, theta, lgm.Gaussian(), data)

    # One Newton step lands on the mode either way; the noise is a precision
    # on one side and a variance on the other, hence round-off.
    assert np.isclose(_value(model), float(exact), atol=1e-10)


def test_an_unconverged_fit_can_be_rejected():
    prior, data, theta = _prior(), _data(), {"t.tau": 1.7}

    def model(reject):
        lgm.lgm_laplace_factor(
            "y",
            prior,
            theta,
            lgm.Poisson(),
            data,
            max_newton=1,  # a Poisson mode takes several
            reject_unconverged=reject,
        )

    assert np.isneginf(_value(lambda: model(True)))
    assert np.isfinite(_value(lambda: model(False)))


def test_the_likelihood_may_not_take_a_components_name():
    prior = lgm.LGMPrior((lgm.IID(3, name="lik"),))
    data = {"y": np.ones(3), "lik": np.arange(3)}
    with pytest.raises(ValueError, match="observation model's name"):
        lgm.lgm_laplace_factor("y", prior, {"lik.tau": 1.0}, lgm.Poisson(), data)


def _hybrid(prior, data, projector=None):
    theta = {"t.tau": numpyro.sample("t.tau", dist.Gamma(2.0, 1.0))}
    lgm.lgm_laplace_factor("y", prior, theta, lgm.Poisson(), data, projector=projector)


def test_the_gradient_through_the_factor_matches_central_differences():
    prior, data = _prior(), _data()
    A = prior.projector(data)

    def density(tau):
        return log_density(_hybrid, (prior, data, A), {}, {"t.tau": tau})[0]

    h = 1e-5
    fd = (density(1.3 + h) - density(1.3 - h)) / (2 * h)
    assert np.isclose(float(jax.grad(density)(1.3)), float(fd), rtol=1e-6)


def test_nuts_runs_on_the_laplace_factor():
    prior, data = _prior(), _data()
    mcmc = MCMC(NUTS(_hybrid), num_warmup=5, num_samples=5, progress_bar=False)
    mcmc.run(jax.random.key(0), prior, data, prior.projector(data))
    tau = mcmc.get_samples()["t.tau"]
    assert tau.shape == (5,)
    assert bool(jnp.all(tau > 0))


@pytest.mark.slow
def test_nuts_on_the_gaussian_factor_matches_quadrature():
    # One hyperparameter, so the posterior mean of log tau is a 1-D integral.
    prior = lgm.LGMPrior((lgm.IID(4, name="g"),))
    rng = np.random.default_rng(1)
    group = np.arange(40) % 4
    data = {"y": rng.normal(size=4)[group] + 0.5 * rng.normal(size=40), "g": group}
    A = prior.projector(data)
    hyper = dist.Gamma(2.0, 1.0)

    def model():
        theta = {"g.tau": numpyro.sample("g.tau", hyper)}
        lgm.lgm_factor("y", prior, theta, data, 0.25, projector=A)

    grid = jnp.linspace(1e-3, 12.0, 400)
    logp = jax.lax.map(
        lambda t: prior.log_marginal({"g.tau": t}, data, 0.25, projector=A), grid
    ) + hyper.log_prob(grid)
    w = jnp.exp(logp - logp.max())
    w = w / w.sum()
    mean = float(jnp.sum(w * grid))
    sd = float(jnp.sqrt(jnp.sum(w * (grid - mean) ** 2)))
    mcmc = MCMC(NUTS(model), num_warmup=300, num_samples=1000, progress_bar=False)
    mcmc.run(jax.random.key(0))
    tau = np.asarray(mcmc.get_samples()["g.tau"])
    # At least 200 effective draws from 1000: four standard errors.
    assert abs(tau.mean() - mean) < 4.0 * sd / np.sqrt(200)
