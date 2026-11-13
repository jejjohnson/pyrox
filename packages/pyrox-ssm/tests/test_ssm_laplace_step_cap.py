"""`LaplaceMarkovInference`'s step cap.

A full Newton step on a Bernoulli likelihood at a large prior variance
overshoots and cycles: the toy wet-day record below, a Matérn-3/2 latent on a
seasonal mean, does not converge with ``max_step=None`` at σ_f² = 8 or 16
however many iterations it is given, and its implicit gradient there is wrong.
With the default cap of one prior standard deviation per step it converges,
reaches the same fixed point wherever the full step also converges, and its
evidence gradient matches central differences.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from pyrox_ssm import (
    LaplaceMarkovInference,
    MarkovGPPrior,
    MaternSDE,
    markov_gp_from_innovations,
)
from ssm_reference import BernoulliLikelihood, StudentTLikelihood


jax.config.update("jax_enable_x64", True)

DAYS = jnp.arange(90.0, 180.0)
MASK = ~((DAYS >= 124) & (DAYS <= 133))  # a ten-day logger outage
OMEGA = 2.0 * jnp.pi / 365.25


def _seasonal(t):
    return -1.5 + 2.0 * jnp.cos(OMEGA * t) + 1.1 * jnp.sin(OMEGA * t)


def _wet_days():
    key_f, key_y = jr.split(jr.key(3))
    prior = MarkovGPPrior(MaternSDE(1.0, 8.0, order=1), DAYS)
    f = markov_gp_from_innovations(prior, jr.normal(key_f, (DAYS.shape[0], 2)))
    p = jax.nn.sigmoid(_seasonal(DAYS) + f)
    return jr.bernoulli(key_y, p).astype(jnp.float64)


WET = _wet_days()


def _fit(
    strategy, variance, lengthscale=8.0, likelihood=None, y=WET, mean_fn=_seasonal
):
    sde = MaternSDE(variance, lengthscale, order=1)
    prior = MarkovGPPrior(sde, DAYS, mean_fn=mean_fn)
    lik = BernoulliLikelihood() if likelihood is None else likelihood
    return strategy.fit(prior, lik, y, mask=MASK)


@pytest.mark.parametrize("variance", [8.0, 16.0])
def test_full_newton_step_cycles_at_a_large_prior_variance(variance) -> None:
    """The bug: without the cap the fit never converges here."""
    fit = _fit(LaplaceMarkovInference(max_step=None, max_iter=200), variance)
    assert not bool(fit.converged)


@pytest.mark.parametrize("variance", [8.0, 16.0])
def test_the_default_cap_converges_at_a_large_prior_variance(variance) -> None:
    fit = _fit(LaplaceMarkovInference(), variance)
    assert bool(fit.converged)
    assert jnp.isfinite(fit.log_marginal_approx)


def test_the_cap_keeps_the_fixed_point() -> None:
    """Where the full step converges, the capped fit lands on the same mode."""
    capped = _fit(LaplaceMarkovInference(tol=1e-10), 1.0)
    full = _fit(LaplaceMarkovInference(max_step=None, tol=1e-10), 1.0)
    assert bool(capped.converged) and bool(full.converged)
    np.testing.assert_allclose(capped.q_mean, full.q_mean, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        capped.log_marginal_approx, full.log_marginal_approx, rtol=1e-10
    )


def test_evidence_gradient_matches_central_differences_at_a_large_variance() -> None:
    """At σ_f² = 16, where the uncapped gradient was wrong by orders of magnitude."""
    strategy = LaplaceMarkovInference(tol=1e-10)

    def evidence(log_theta):
        theta = jnp.exp(log_theta)
        return _fit(strategy, theta[0], theta[1]).log_marginal_approx

    x = jnp.log(jnp.array([16.0, 8.0]))
    grad = jax.grad(evidence)(x)
    # central differences, step 1e-5 in log θ: truncation error O(1e-10)
    h = 1e-5
    fd = jnp.array(
        [
            (evidence(x.at[i].add(h)) - evidence(x.at[i].add(-h))) / (2 * h)
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(grad, fd, rtol=1e-5)


def test_the_cap_converges_a_student_t_fit_on_data_far_from_the_prior_mean() -> None:
    """Readings around 20 from a zero prior mean: the full step does not converge."""
    y = 20.0 + 5.0 * jnp.sin(DAYS / 9.0) + jr.t(jr.key(5), 4.0, DAYS.shape)
    lik = StudentTLikelihood(df=4.0, scale=1.0)
    capped = _fit(LaplaceMarkovInference(), 25.0, likelihood=lik, y=y, mean_fn=None)
    full = _fit(
        LaplaceMarkovInference(max_step=None, max_iter=200),
        25.0,
        likelihood=lik,
        y=y,
        mean_fn=None,
    )
    assert bool(capped.converged)
    assert not bool(full.converged)


def test_a_deterministic_initial_level_keeps_the_fit_finite() -> None:
    """σ_n = 0 at a time with a known level: that time is left out of the cap."""
    from pyrox_ssm import IntegratedWienerSDE

    times = jnp.arange(30.0)
    y = (jnp.sin(times / 4.0) > 0).astype(jnp.float64)
    prior = MarkovGPPrior(
        IntegratedWienerSDE(jnp.asarray(0.05), order=1),
        times,
        init_cov=jnp.diag(jnp.array([0.0, 1.0])),
    )
    fit = LaplaceMarkovInference().fit(prior, BernoulliLikelihood(), y)
    assert bool(fit.converged)
    assert jnp.all(jnp.isfinite(fit.q_mean))
    assert jnp.isfinite(fit.log_marginal_approx)
