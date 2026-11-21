"""Tests for `gp_laplace_factor` (pyrox#522).

The factor registers the Laplace approximate evidence of a non-Gaussian GP.
It is the fit's ``log_marginal_approx``, equals `gp_factor` for a Gaussian
likelihood (where the Laplace approximation is exact), and differentiates in
the kernel hyperparameters through the implicit mode.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from pyrox_gp import (
    RBF,
    BernoulliLikelihood,
    GaussianLikelihood,
    GaussNewtonInference,
    GPPrior,
    LaplaceInference,
    gp_factor,
    gp_laplace_factor,
)


X = jnp.linspace(-3.0, 3.0, 12)[:, None]
LABELS = (jnp.sin(X[:, 0]) > 0.0).astype(X.dtype)
TARGETS = jnp.sin(X[:, 0]) + 0.1 * jnp.cos(3.0 * X[:, 0])


def _factor_value(model, data=None):
    model = (
        handlers.seed(model, 0)
        if data is None
        else handlers.substitute(handlers.seed(model, 0), data)
    )
    site = handlers.trace(model).get_trace()["y"]
    return site["fn"].log_prob(site["value"])


@pytest.mark.parametrize("strategy", [None, LaplaceInference(), GaussNewtonInference()])
def test_value_is_the_fit_log_marginal(strategy):
    prior = GPPrior(kernel=RBF(), X=X)
    lik = BernoulliLikelihood()

    def model():
        gp_laplace_factor("y", prior, lik, LABELS, strategy=strategy)

    fit = (strategy or LaplaceInference()).fit(prior, lik, LABELS)
    assert jnp.array_equal(_factor_value(model), fit.log_marginal_approx)


def test_gaussian_likelihood_gives_gp_factor():
    prior = GPPrior(kernel=RBF(), X=X)
    noise = jnp.asarray(0.1)

    def laplace_model():
        gp_laplace_factor("y", prior, GaussianLikelihood(noise_var=noise), TARGETS)

    def exact_model():
        gp_factor("y", prior, TARGETS, noise)

    # The Laplace approximation is exact for a Gaussian likelihood; the gap is
    # the round-off of two Cholesky routes under x64.
    assert jnp.allclose(
        _factor_value(laplace_model), _factor_value(exact_model), rtol=1e-9
    )


def _priored_model():
    kernel = RBF()
    kernel.set_prior("lengthscale", dist.LogNormal(0.0, 0.5))
    prior = GPPrior(kernel=kernel, X=X)

    def model():
        gp_laplace_factor("y", prior, BernoulliLikelihood(), LABELS)

    return model


def test_gradient_in_the_lengthscale_matches_finite_differences():
    model = _priored_model()

    def value(lengthscale):
        return _factor_value(model, {"RBF.lengthscale": lengthscale})

    lengthscale, step = jnp.asarray(0.8), 1e-5
    grad = jax.grad(value)(lengthscale)
    central = (value(lengthscale + step) - value(lengthscale - step)) / (2 * step)
    assert bool(jnp.isfinite(grad))
    # Central differences at step 1e-5 under x64 carry ~1e-9 error; the
    # Newton tolerance of 1e-6 on the mode bounds the rest.
    assert jnp.allclose(grad, central, rtol=1e-4)


def test_short_nuts_run_on_a_bernoulli_model():
    mcmc = MCMC(
        NUTS(_priored_model()), num_warmup=30, num_samples=30, progress_bar=False
    )
    mcmc.run(jax.random.key(0))
    lengthscale = mcmc.get_samples()["RBF.lengthscale"]
    assert lengthscale.shape == (30,)
    assert bool(jnp.all(jnp.isfinite(lengthscale)) & jnp.all(lengthscale > 0.0))
