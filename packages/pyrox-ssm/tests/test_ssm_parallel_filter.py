"""The ``parallel`` flag: `gaussx.parallel_kalman_filter` behind the priors.

The flag only changes how the filter is computed, so every result (filtered
moments, the log marginal, its gradient, the smoothed and predicted moments,
and a non-Gaussian fit run on the same prior) matches the sequential filter
to round-off.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from pyrox_ssm import (
    LaplaceMarkovInference,
    LinearGaussianSSMPrior,
    MarkovGPPrior,
    MaternSDE,
)
from ssm_reference import BernoulliLikelihood


jax.config.update("jax_enable_x64", True)

TIMES = jnp.linspace(0.0, 6.0, 120)
Y = jnp.sin(TIMES) + 0.1 * jr.normal(jr.key(0), TIMES.shape)
MASK = jnp.arange(TIMES.shape[0]) % 7 != 3  # every seventh step unobserved
THETA = jnp.log(jnp.array([1.3, 0.7, 0.05]))

# The two filters compose the same algebra in a different order (an
# associative scan against a sequential one), so they agree to float64
# round-off accumulated over 120 steps.
RTOL, ATOL = 1e-10, 1e-10


def _prior(theta, parallel, order=1):
    sde = MaternSDE(jnp.exp(theta[0]), jnp.exp(theta[1]), order=order)
    return MarkovGPPrior(sde, TIMES, mean_fn=lambda t: 0.2 * t, parallel=parallel)


def test_lgssm_parallel_filter_matches_the_sequential_filter() -> None:
    lgssm = _prior(THETA, False).as_lgssm(jnp.exp(THETA[2]))
    assert not lgssm.parallel
    parallel = LinearGaussianSSMPrior(
        lgssm.transition,
        lgssm.obs_model,
        lgssm.process_noise,
        lgssm.obs_noise,
        lgssm.init_mean,
        lgssm.init_cov,
        lgssm.obs_offset,
        parallel=True,
    )
    seq, par = lgssm.filter(Y, mask=MASK), parallel.filter(Y, mask=MASK)
    for a, b in [
        (seq.filtered_means, par.filtered_means),
        (seq.filtered_covs, par.filtered_covs),
        (seq.predicted_means, par.predicted_means),
        (seq.predicted_covs, par.predicted_covs),
        (seq.log_likelihood, par.log_likelihood),
    ]:
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("order", [1, 2])
def test_markov_log_marginal_and_its_gradient_match(order) -> None:
    def log_marginal(theta, parallel):
        prior = _prior(theta, parallel, order)
        return prior.log_marginal(Y, jnp.exp(theta[2]), mask=MASK)

    assert _prior(THETA, True).as_lgssm(0.1).parallel
    seq = jax.jit(jax.value_and_grad(lambda t: log_marginal(t, False)))(THETA)
    par = jax.jit(jax.value_and_grad(lambda t: log_marginal(t, True)))(THETA)
    np.testing.assert_allclose(par[0], seq[0], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(par[1], seq[1], rtol=RTOL, atol=ATOL)


def test_smooth_and_predict_match() -> None:
    noise = jnp.exp(THETA[2])
    seq, par = _prior(THETA, False), _prior(THETA, True)
    for a, b in zip(
        seq.smooth(Y, noise, mask=MASK), par.smooth(Y, noise, mask=MASK), strict=True
    ):
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)
    t_star = jnp.array([-0.5, 1.234, 3.0, 6.5])
    for a, b in zip(
        seq.condition(Y, noise, mask=MASK).predict(t_star),
        par.condition(Y, noise, mask=MASK).predict(t_star),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)


def test_a_laplace_fit_on_a_parallel_prior_matches() -> None:
    """The non-Gaussian strategies filter through the prior's flag too."""
    wet = (Y > 0.3).astype(jnp.float64)

    def evidence(theta, parallel):
        fit = LaplaceMarkovInference(tol=1e-10).fit(
            _prior(theta, parallel), BernoulliLikelihood(), wet, mask=MASK
        )
        return fit.log_marginal_approx

    seq = jax.jit(jax.value_and_grad(lambda t: evidence(t, False)))(THETA)
    par = jax.jit(jax.value_and_grad(lambda t: evidence(t, True)))(THETA)
    # Agreement to round-off at each of the fit's iterations, compounded.
    np.testing.assert_allclose(par[0], seq[0], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(par[1], seq[1], rtol=1e-8, atol=1e-9)
