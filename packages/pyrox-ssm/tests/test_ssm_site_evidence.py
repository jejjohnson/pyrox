"""The evidence estimate ``log_marginal_approx`` of the site-based strategies.

Every strategy fits Gaussian sites and reports the same Laplace-style
correction of the pseudo-data Kalman marginal: swap the pseudo-data fit for
the data fit at the posterior mean. These tests pin that formula against a
dense reference and check the strategies agree on the scale of the result.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax.scipy.stats import norm
from pyrox_ssm import (
    ExpectationPropagationMarkov,
    LaplaceMarkovInference,
    MarkovGPPrior,
    MaternSDE,
    PosteriorLinearizationMarkov,
)
from ssm_reference import BernoulliLikelihood, StudentTLikelihood, matern_kernel


def _bernoulli_problem(n: int = 30):
    times = jnp.linspace(0.0, 5.0, n)
    f = 1.5 * jnp.sin(2.0 * times)
    y = (jr.uniform(jr.key(0), (n,)) < jax.nn.sigmoid(f)).astype(jnp.float64)
    prior = MarkovGPPrior(MaternSDE(variance=1.0, lengthscale=0.6, order=1), times)
    return times, y, prior


def _dense_corrected_evidence(times, y, cond, log_lik_per_point):
    """The corrected evidence built densely in jnp from the fitted sites."""
    K = matern_kernel(times[:, None], times[:, None], 1.0, 0.6, nu=1.5)
    noise = 1.0 / cond.site_nat2
    pseudo = cond.site_nat1 / cond.site_nat2
    cov = K + jnp.diag(noise)
    chol = jnp.linalg.cholesky(cov)
    alpha = jax.scipy.linalg.cho_solve((chol, True), pseudo)
    log_pseudo_marginal = (
        -0.5 * pseudo @ alpha
        - jnp.sum(jnp.log(jnp.diag(chol)))
        - 0.5 * times.shape[0] * jnp.log(2.0 * jnp.pi)
    )
    data_fit = jnp.sum(log_lik_per_point(cond.q_mean, y))
    pseudo_fit = jnp.sum(norm.logpdf(pseudo, cond.q_mean, jnp.sqrt(noise)))
    return log_pseudo_marginal + data_fit - pseudo_fit


def _bernoulli_log_lik(f, y):
    return y * jax.nn.log_sigmoid(f) + (1.0 - y) * jax.nn.log_sigmoid(-f)


@pytest.mark.parametrize(
    "strategy",
    [
        LaplaceMarkovInference(max_iter=50),
        PosteriorLinearizationMarkov(max_iter=60),
        ExpectationPropagationMarkov(max_iter=60),
    ],
    ids=["laplace", "pl", "ep"],
)
def test_evidence_matches_the_dense_corrected_formula(strategy):
    times, y, prior = _bernoulli_problem()
    cond = strategy.fit(prior, BernoulliLikelihood(), y)
    assert bool(cond.converged)
    expected = _dense_corrected_evidence(times, y, cond, _bernoulli_log_lik)
    # Same quantity on the Kalman path and densely, so only round-off differs.
    assert jnp.allclose(cond.log_marginal_approx, expected, rtol=1e-10, atol=1e-8)


def test_pl_and_ep_evidence_sit_beside_laplace_on_bernoulli():
    _, y, prior = _bernoulli_problem()
    lik = BernoulliLikelihood()
    laplace = LaplaceMarkovInference(max_iter=50).fit(prior, lik, y)
    pl = PosteriorLinearizationMarkov(max_iter=60).fit(prior, lik, y)
    ep = ExpectationPropagationMarkov(max_iter=60).fit(prior, lik, y)
    # Before the correction PL and EP reported the pseudo-data marginal, about
    # three times Laplace's value here; corrected, the three differ by the
    # small gap between their site approximations.
    for cond in (pl, ep):
        assert abs(float(cond.log_marginal_approx - laplace.log_marginal_approx)) < 1.0


def test_evidence_stays_on_scale_with_floored_sites():
    # Student-t readings with spikes: sites at the outliers are floored to a
    # tiny precision, whose huge pseudo-targets made the uncorrected marginal
    # of order -1e7 to -1e8.
    times = jnp.linspace(0.0, 5.0, 40)
    y = jnp.sin(2.0 * times) + 0.1 * jr.normal(jr.key(1), (40,))
    y = y.at[jnp.array([5, 17, 30])].add(jnp.array([3.0, -3.5, 4.0]))
    prior = MarkovGPPrior(MaternSDE(variance=1.0, lengthscale=0.5, order=1), times)
    lik = StudentTLikelihood(df=3.0, scale=0.1)
    for strategy in (
        PosteriorLinearizationMarkov(max_iter=100),
        ExpectationPropagationMarkov(max_iter=100),
    ):
        cond = strategy.fit(prior, lik, y)
        value = float(cond.log_marginal_approx)
        assert jnp.isfinite(value)
        # 40 Student-t log densities with scale 0.1 are each a few nats.
        assert -500.0 < value < 100.0
