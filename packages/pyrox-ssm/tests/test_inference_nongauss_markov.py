"""Tests for the site-based non-Gaussian Markov GP inference strategies.

Coverage strategy:

* shape / finiteness / convergence on a small Bernoulli time-series
  problem, for each of Laplace, GN, PL, EP.
* equivalence at the smoothed posterior: Markov-Laplace on a sorted
  1-D grid using a Matern-1/2 SDE kernel matches a dense Newton-iteration
  Laplace on the closed-form Gram within tolerance — the SDE
  representation is exact, so the two paths must agree. PL and EP are
  checked against their dense fixed point on the closed-form Gram.
* prediction at test times: shapes, finiteness, non-negative variance.
* `condition_nongauss` convenience method round-trips correctly.
"""

from __future__ import annotations

import dataclasses

import gaussx
import jax
import jax.numpy as jnp
import pytest
from pyrox_ssm import (
    ExpectationPropagationMarkov,
    GaussNewtonMarkovInference,
    IntegratedWienerSDE,
    LaplaceMarkovInference,
    MarkovGPPrior,
    MaternSDE,
    NonGaussConditionedMarkovGP,
    PosteriorLinearizationMarkov,
)
from pyrox_ssm._inference_nongauss_markov import _prior_marginal_variance
from ssm_reference import BernoulliLikelihood, matern_kernel, site_expectation


def _make_bernoulli_timeseries(N: int = 16):
    times = jnp.linspace(0.0, 5.0, N)
    y = (jnp.sin(2.0 * times) > 0.0).astype(jnp.float32)
    sde = MaternSDE(variance=1.0, lengthscale=0.7, order=1)
    prior = MarkovGPPrior(sde, times)
    return prior, times, y


def test_markov_laplace_smoke_tiny() -> None:
    """Fast smoke for ``LaplaceMarkovInference`` fit + predict on a tiny grid.

    Stays un-marked so the default CI run (``pytest -m "not slow"``) still
    exercises the non-Gaussian Markov inference + predict surface on every
    PR; the deeper convergence / dense-equivalence tests are slow.
    """
    times = jnp.linspace(0.0, 1.0, 6)
    y = (jnp.sin(3.0 * times) > 0.0).astype(jnp.float32)
    sde = MaternSDE(variance=1.0, lengthscale=0.5, order=1)
    prior = MarkovGPPrior(sde, times)
    cond = LaplaceMarkovInference(max_iter=2).fit(prior, BernoulliLikelihood(), y)
    assert isinstance(cond, NonGaussConditionedMarkovGP)
    assert cond.q_mean.shape == y.shape
    assert jnp.all(jnp.isfinite(cond.q_mean))
    assert jnp.all(jnp.isfinite(cond.q_var))
    assert jnp.all(cond.q_var >= 0.0)
    m, v = cond.predict(jnp.array([0.25, 0.75]))
    assert m.shape == (2,) and v.shape == (2,)
    assert jnp.all(jnp.isfinite(m)) and jnp.all(v >= 0.0)


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

    # Dense Laplace by Newton's method on the closed-form Matern-1/2 Gram:
    # f ← (K⁻¹ + W)⁻¹ (W f + ∇ log p(y | f)), W = σ(f)(1 − σ(f)).
    K = matern_kernel(times[:, None], times[:, None], 1.0, 0.6, nu=0.5)
    K_inv = jnp.linalg.inv(K)
    f = jnp.zeros_like(times)
    for _ in range(30):
        p = jax.nn.sigmoid(f)
        W = p * (1.0 - p)
        f = jnp.linalg.solve(K_inv + jnp.diag(W), W * f + (y - p))
    p = jax.nn.sigmoid(f)
    dense_var = jnp.diag(jnp.linalg.inv(K_inv + jnp.diag(p * (1.0 - p))))

    assert jnp.allclose(markov_cond.q_mean, f, atol=1e-3)
    assert jnp.allclose(markov_cond.q_var, dense_var, atol=1e-3)


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


# --- non-stationary priors (gh-222) --------------------------------------


def test_prior_marginal_variance_is_constant_for_stationary_kernels() -> None:
    """Unchanged path: the closed form ``H P_inf H^T`` = kernel variance."""
    times = jnp.linspace(0.0, 5.0, 8)
    prior = MarkovGPPrior(MaternSDE(variance=1.7, lengthscale=0.7, order=1), times)

    var = _prior_marginal_variance(prior)

    assert var.shape == times.shape
    assert jnp.allclose(var, 1.7)


def test_prior_marginal_variance_grows_for_a_trend_prior() -> None:
    """A non-stationary prior has no constant marginal to broadcast.

    The variance of an integrated Wiener process grows with ``t``, so the
    seed is propagated through the recursion instead. Against the closed
    form: from ``P_0`` at ``t_0``, ``Var f(t) = P_0[0,0] + t^2 P_0[1,1] +
    q t^3 / 3``.
    """
    times = jnp.linspace(0.0, 4.0, 9)
    q, P_0 = 0.05, jnp.diag(jnp.asarray([0.5, 0.2]))
    prior = MarkovGPPrior(
        IntegratedWienerSDE(diffusion=jnp.asarray(q)), times, init_cov=P_0
    )

    var = _prior_marginal_variance(prior)
    elapsed = times - times[0]
    expected = P_0[0, 0] + elapsed**2 * P_0[1, 1] + q * elapsed**3 / 3.0

    assert var.shape == times.shape
    assert jnp.all(jnp.diff(var) > 0.0)
    assert jnp.allclose(var, expected, rtol=1e-6)


def test_markov_laplace_runs_with_a_trend_prior() -> None:
    """The motivating case: a non-Gaussian likelihood over a trend prior."""
    N = 16
    times = jnp.linspace(0.0, 5.0, N)
    y = (times > 2.5).astype(jnp.float32)
    prior = MarkovGPPrior(
        IntegratedWienerSDE(diffusion=jnp.asarray(0.1)),
        times,
        init_cov=jnp.diag(jnp.asarray([1.0, 0.1])),
    )

    post = prior.condition_nongauss(
        BernoulliLikelihood(), y, strategy=LaplaceMarkovInference(max_iter=10)
    )

    assert post.q_mean.shape == (N,)
    assert jnp.all(jnp.isfinite(post.q_mean))
    assert jnp.all(post.q_var > 0.0)
    # The latent should track the step: negative early, positive late.
    assert post.q_mean[0] < 0.0 < post.q_mean[-1]

    mean, var = post.predict(jnp.asarray([5.5, 6.0]))
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(var > 0.0)


# --- the integrator argument (#335) -----------------------------------------


_MARKOV_STRATEGIES = [PosteriorLinearizationMarkov, ExpectationPropagationMarkov]


def _assert_same_fit(first, second) -> None:
    # Same nodes and weights, so the two runs differ by float64 round-off only.
    assert first.n_iter == second.n_iter
    assert jnp.allclose(first.site_nat1, second.site_nat1, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(first.site_nat2, second.site_nat2, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(
        first.log_marginal_approx, second.log_marginal_approx, rtol=1e-12, atol=0.0
    )


@pytest.mark.parametrize("strategy", _MARKOV_STRATEGIES)
def test_default_integrator_is_gauss_hermite_order_20(strategy) -> None:
    built = strategy()
    assert isinstance(built.integrator, gaussx.GaussHermiteIntegrator)
    assert built.integrator.order == 20
    assert built.deg is None


@pytest.mark.parametrize("strategy", _MARKOV_STRATEGIES)
def test_deg_warns_and_matches_the_integrator(strategy) -> None:
    prior, _, y = _make_bernoulli_timeseries(8)
    with pytest.warns(DeprecationWarning, match=r"GaussHermiteIntegrator\(order=10\)"):
        from_deg = strategy(deg=10, max_iter=3)
    from_integrator = strategy(
        integrator=gaussx.GaussHermiteIntegrator(order=10), max_iter=3
    )
    _assert_same_fit(
        from_deg.fit(prior, BernoulliLikelihood(), y),
        from_integrator.fit(prior, BernoulliLikelihood(), y),
    )


@pytest.mark.parametrize("strategy", _MARKOV_STRATEGIES)
def test_positional_arguments_keep_their_meaning(strategy) -> None:
    built = strategy(10, 1e-3, 0.25, 1e-5)
    assert built.max_iter == 10
    assert built.tol == 1e-3
    assert built.damping == 0.25
    assert built.precision_floor == 1e-5
    assert built.integrator.order == 20
    with pytest.warns(DeprecationWarning, match="deg is deprecated"):
        assert strategy(10, 1e-3, 0.25, 1e-5, 8).integrator.order == 8


@pytest.mark.parametrize("strategy", _MARKOV_STRATEGIES)
def test_deg_and_integrator_together_raise(strategy) -> None:
    with pytest.raises(ValueError, match="not both"):
        strategy(deg=10, integrator=gaussx.GaussHermiteIntegrator(order=10))


@pytest.mark.parametrize("strategy", _MARKOV_STRATEGIES)
def test_a_resolved_strategy_survives_replace(strategy) -> None:
    with pytest.warns(DeprecationWarning):
        built = strategy(deg=10)
    rebuilt = dataclasses.replace(built, max_iter=2)
    assert rebuilt.integrator.order == 10


def test_posterior_linearization_cavity_expectations_match_the_old_grid() -> None:
    """The grid the integrator replaced, built by hand as the reference."""
    _, _, y = _make_bernoulli_timeseries(8)
    likelihood = BernoulliLikelihood()
    lp = lambda f_n, y_n: likelihood.log_prob(f_n[None], y_n[None])
    grad_lp = jax.grad(lp)
    mean = jnp.linspace(-1.0, 1.0, y.shape[0])
    var = jnp.linspace(0.2, 1.5, y.shape[0])
    points, weights = gaussx.gauss_hermite_points(20, dim=1)
    grid = mean[None] + jnp.sqrt(var)[None] * points[:, 0][:, None]
    per_node = jax.vmap(jax.vmap(grad_lp, in_axes=(0, 0)), in_axes=(0, None))(grid, y)
    expected = (weights / jnp.sqrt(2.0 * jnp.pi))[None] @ per_node
    got = site_expectation(
        gaussx.GaussHermiteIntegrator(order=20), grad_lp, mean, var, y
    )
    assert jnp.allclose(got, expected[0], rtol=1e-12, atol=1e-12)


def test_posterior_linearization_converges_with_an_unscented_rule() -> None:
    prior, _, y = _make_bernoulli_timeseries(8)
    tol = 1e-6
    unscented = PosteriorLinearizationMarkov(
        integrator=gaussx.UnscentedIntegrator(alpha=1.0), max_iter=200, tol=tol
    ).fit(prior, BernoulliLikelihood(), y)
    hermite = PosteriorLinearizationMarkov(max_iter=200, tol=tol).fit(
        prior, BernoulliLikelihood(), y
    )
    assert unscented.converged
    # A 3-point and a 20-point rule on the same cavities: close, not identical.
    assert jnp.max(jnp.abs(unscented.q_mean - hermite.q_mean)) < 0.1


def test_expectation_propagation_rejects_other_rules() -> None:
    with pytest.raises(TypeError, match="GaussHermiteIntegrator"):
        ExpectationPropagationMarkov(integrator=gaussx.UnscentedIntegrator())


def _pl_sites(cav_mean, cav_var, y):
    """PL's target sites: the cavity-averaged Newton step, by hand."""
    lp = lambda f_n, y_n: BernoulliLikelihood().log_prob(f_n[None], y_n[None])
    rule = gaussx.GaussHermiteIntegrator(order=20)
    g = site_expectation(rule, jax.grad(lp), cav_mean, cav_var, y)
    h = site_expectation(rule, jax.grad(jax.grad(lp)), cav_mean, cav_var, y)
    # λ = ḡ − h̄ m_c,  Λ = −h̄
    return g - h * cav_mean, -h


def _ep_sites(cav_mean, cav_var, y):
    """EP's target sites: tilted moments minus the cavity, by hand."""
    points, weights = gaussx.gauss_hermite_points(20, dim=1)
    weights = weights / jnp.sqrt(2.0 * jnp.pi)
    f = cav_mean[None] + jnp.sqrt(cav_var)[None] * points[:, 0][:, None]
    # p(yₙ | f) at every node, for a logit link
    lik = jnp.exp(y[None] * f - jnp.logaddexp(0.0, f))
    Z = weights @ lik
    m_t = (weights @ (lik * f)) / Z
    v_t = (weights @ (lik * f**2)) / Z - m_t**2
    return m_t / v_t - cav_mean / cav_var, 1.0 / v_t - 1.0 / cav_var


@pytest.mark.parametrize(
    ("markov", "target_sites"),
    [
        (PosteriorLinearizationMarkov, _pl_sites),
        (ExpectationPropagationMarkov, _ep_sites),
    ],
)
def test_markov_strategies_reach_the_dense_fixed_point_on_matern32(
    markov, target_sites
) -> None:
    """The SDE is exact for this kernel, so the dense algebra on the closed-form
    Gram must reproduce the Markov posterior and leave its sites unchanged."""
    times = jnp.linspace(0.0, 3.0, 10)
    y = (jnp.sin(2.0 * times) > 0.0).astype(jnp.float64)
    fit = markov(max_iter=400, tol=1e-9).fit(
        MarkovGPPrior(MaternSDE(variance=1.0, lengthscale=0.7, order=1), times),
        BernoulliLikelihood(),
        y,
    )
    assert fit.converged
    # Dense posterior from the fitted sites: Σ = (K⁻¹ + diag Λ)⁻¹, m = Σ λ.
    K = matern_kernel(times[:, None], times[:, None], 1.0, 0.7, nu=1.5)
    cov = jnp.linalg.inv(jnp.linalg.inv(K) + jnp.diag(fit.site_nat2))
    mean, var = cov @ fit.site_nat1, jnp.diag(cov)
    # The Kalman smoother and the dense solve agree to round-off.
    assert jnp.allclose(fit.q_mean, mean, rtol=1e-8, atol=1e-10)
    assert jnp.allclose(fit.q_var, var, rtol=1e-8, atol=1e-10)
    # Cavities from the dense marginals; at the fixed point the update returns
    # the same sites. 1e-5 is well above the 1e-9 tol on the posterior mean.
    cav_var = 1.0 / (1.0 / var - fit.site_nat2)
    cav_mean = cav_var * (mean / var - fit.site_nat1)
    nat1, nat2 = target_sites(cav_mean, cav_var, y)
    assert jnp.max(jnp.abs(nat1 - fit.site_nat1)) < 1e-5
    assert jnp.max(jnp.abs(nat2 - fit.site_nat2)) < 1e-5
