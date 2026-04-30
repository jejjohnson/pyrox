"""Tests for the site-based non-Gaussian Markov GP inference strategies.

Coverage strategy:

* shape / finiteness / convergence on a small Bernoulli time-series
  problem, for each of Laplace, GN, PL, EP.
* equivalence at the smoothed posterior: Markov-Laplace on a sorted
  1-D grid using a Matern-1/2 SDE kernel matches dense Laplace on the
  same kernel/data within tolerance — the SDE representation is exact,
  so the two paths must agree.
* prediction at test times: shapes, finiteness, non-negative variance.
* `condition_nongauss` convenience method round-trips correctly.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from pyrox.gp import (
    BernoulliLikelihood,
    ExpectationPropagationMarkov,
    GaussNewtonMarkovInference,
    GPPrior,
    LaplaceInference,
    LaplaceMarkovInference,
    MarkovGPPrior,
    MaternSDE,
    NonGaussConditionedMarkovGP,
    PosteriorLinearizationMarkov,
)


def _make_bernoulli_timeseries(N: int = 16):
    times = jnp.linspace(0.0, 5.0, N)
    y = (jnp.sin(2.0 * times) > 0.0).astype(jnp.float32)
    sde = MaternSDE(variance=1.0, lengthscale=0.7, order=1)
    prior = MarkovGPPrior(sde, times)
    return prior, times, y


@pytest.mark.slow
@pytest.mark.parametrize(
    "strategy_factory",
    [
        lambda: LaplaceMarkovInference(max_iter=15),
        lambda: GaussNewtonMarkovInference(max_iter=15),
        lambda: PosteriorLinearizationMarkov(max_iter=15),
        lambda: ExpectationPropagationMarkov(max_iter=20),
    ],
)
def test_markov_strategy_runs_and_returns_finite_posterior(strategy_factory) -> None:
    prior, _, y = _make_bernoulli_timeseries()
    strategy = strategy_factory()
    cond = strategy.fit(prior, BernoulliLikelihood(), y)
    assert isinstance(cond, NonGaussConditionedMarkovGP)
    assert cond.q_mean.shape == y.shape
    assert cond.q_var.shape == y.shape
    assert jnp.all(jnp.isfinite(cond.q_mean))
    assert jnp.all(jnp.isfinite(cond.q_var))
    assert jnp.all(cond.q_var >= 0.0)
    assert jnp.isfinite(cond.log_marginal_approx)


@pytest.mark.slow
def test_markov_predict_finite_and_psd() -> None:
    prior, times, y = _make_bernoulli_timeseries()
    cond = LaplaceMarkovInference().fit(prior, BernoulliLikelihood(), y)
    t_star = jnp.linspace(times[0] - 1.0, times[-1] + 1.0, 50)
    m, v = cond.predict(t_star)
    assert m.shape == (50,) and v.shape == (50,)
    assert jnp.all(jnp.isfinite(m))
    assert jnp.all(jnp.isfinite(v))
    assert jnp.all(v >= 0.0)


@pytest.mark.slow
def test_markov_laplace_matches_dense_laplace_on_matern12() -> None:
    """Matern-1/2 SDE is the exact state-space form of the Matern-1/2
    kernel, so Markov-Laplace and dense Laplace on the same data must
    produce the same posterior mean / variance up to numerical tolerance."""
    times = jnp.linspace(0.0, 4.0, 14)
    y = (jnp.cos(1.5 * times) > 0.0).astype(jnp.float32)

    # MaternSDE(order=0) is the exact state-space form of Matern with
    # nu=1/2 (exponential covariance). order=1 -> nu=3/2, order=2 -> nu=5/2.
    sde = MaternSDE(variance=1.0, lengthscale=0.6, order=0)
    markov_prior = MarkovGPPrior(sde, times)
    markov_cond = LaplaceMarkovInference(max_iter=30, tol=1e-7).fit(
        markov_prior, BernoulliLikelihood(), y
    )

    from pyrox.gp import Matern

    dense_kernel = Matern(init_lengthscale=0.6, init_variance=1.0, nu=0.5)
    dense_prior = GPPrior(kernel=dense_kernel, X=times[:, None])
    dense_cond = LaplaceInference(max_iter=30, tol=1e-7).fit(
        dense_prior, BernoulliLikelihood(), y
    )

    assert jnp.allclose(markov_cond.q_mean, dense_cond.q_mean, atol=1e-3)
    assert jnp.allclose(markov_cond.q_var, dense_cond.q_var, atol=1e-3)


@pytest.mark.slow
def test_condition_nongauss_method_dispatches_to_strategy() -> None:
    prior, _, y = _make_bernoulli_timeseries()
    cond_method = prior.condition_nongauss(
        BernoulliLikelihood(), y, strategy=LaplaceMarkovInference(max_iter=20)
    )
    cond_direct = LaplaceMarkovInference(max_iter=20).fit(
        prior, BernoulliLikelihood(), y
    )
    assert jnp.allclose(cond_method.q_mean, cond_direct.q_mean)
    assert jnp.allclose(cond_method.q_var, cond_direct.q_var)


@pytest.mark.slow
def test_predict_at_training_times_recovers_q_mean() -> None:
    """Predicting at the training grid should reproduce ``q_mean`` /
    ``q_var`` to within a small tolerance — the merged-grid trick must
    be a no-op when ``t_star`` coincides with ``times``."""
    prior, times, y = _make_bernoulli_timeseries()
    cond = LaplaceMarkovInference(max_iter=20, tol=1e-7).fit(
        prior, BernoulliLikelihood(), y
    )
    m, v = cond.predict(times)
    assert jnp.allclose(m, cond.q_mean, atol=1e-4)
    assert jnp.allclose(v, cond.q_var, atol=1e-4)
