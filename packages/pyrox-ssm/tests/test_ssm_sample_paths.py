"""Joint path draws: ``ConditionedMarkovGP.sample``,
``ConditionedLinearGaussianSSM.sample`` and ``LinearGaussianSSMPrior.sample``.

The draws are random, so every comparison is bounded by the estimator's own
sampling distribution: with ``S`` draws, a sample mean sits within
``7 sqrt(Σ_ii / S)`` of the truth and a sample covariance entry within
``7 sqrt((Σ_ii Σ_jj + Σ_ij²) / S)`` (Gaussian draws; 7σ as in gaussx's
``assert_sample_moments``, gh-220). The references are the dense Gaussian
conditional of the closed-form Matérn-3/2 Gram, `predict`, and the smoother.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
from pyrox_ssm import (
    LinearGaussianSSMPrior,
    MarkovGPPrior,
    MaternSDE,
    markov_gp_from_innovations,
)
from ssm_reference import matern_kernel


jax.config.update("jax_enable_x64", True)

S = 20_000
VARIANCE, LENGTHSCALE, NOISE = 1.3, 0.8, 0.05
TIMES = jnp.array([0.0, 0.4, 1.1, 1.5, 2.6, 3.0])
Y = jnp.array([0.2, 0.5, -0.1, jnp.nan, 0.4, 0.1])
MASK = ~jnp.isnan(Y)
T_STAR = jnp.array([0.4, 1.8, 2.2, 3.5])


def _prior(mean_fn=None) -> MarkovGPPrior:
    return MarkovGPPrior(
        MaternSDE(VARIANCE, LENGTHSCALE, order=1), TIMES, mean_fn=mean_fn
    )


def _assert_moments(draws, mean, cov):
    """Sample mean and covariance of ``draws`` (S, K) within 7σ of the truth."""
    n = draws.shape[0]
    centred = draws - draws.mean(axis=0)
    sample_cov = centred.T @ centred / (n - 1)
    diag = jnp.diag(cov)
    mean_band = 7.0 * jnp.sqrt(diag / n)
    cov_band = 7.0 * jnp.sqrt((jnp.outer(diag, diag) + cov**2) / n)
    assert jnp.all(jnp.abs(draws.mean(axis=0) - mean) <= mean_band), (
        draws.mean(axis=0) - mean
    )
    assert jnp.all(jnp.abs(sample_cov - cov) <= cov_band), sample_cov - cov


def _dense_conditional(t_star, mean_fn=None):
    """Mean and covariance of f(t_*) given the observed y, from the dense Gram."""
    mu = (lambda t: jnp.zeros_like(t)) if mean_fn is None else mean_fn
    t_obs, y_obs = TIMES[MASK], Y[MASK]

    def k(a, b):
        return matern_kernel(a[:, None], b[:, None], VARIANCE, LENGTHSCALE, nu=1.5)

    K_oo = k(t_obs, t_obs) + NOISE * jnp.eye(t_obs.shape[0])
    K_so = k(t_star, t_obs)
    gain = jnp.linalg.solve(K_oo, K_so.T).T
    mean = mu(t_star) + gain @ (y_obs - mu(t_obs))
    return mean, k(t_star, t_star) - gain @ K_so.T


def test_markov_gp_draws_have_the_dense_conditional_joint_moments():
    cond = _prior().condition(Y, jnp.asarray(NOISE), mask=MASK)
    draws = cond.sample(jr.key(0), n_paths=S, t_star=T_STAR)
    assert draws.shape == (S, T_STAR.shape[0])
    _assert_moments(draws, *_dense_conditional(T_STAR))


def test_markov_gp_draws_match_predict_at_each_time_with_a_mean_function():
    mean_fn = lambda t: 0.3 * t - 0.2
    cond = _prior(mean_fn).condition(Y, jnp.asarray(NOISE), mask=MASK)
    draws = cond.sample(jr.key(1), n_paths=S, t_star=T_STAR)
    mean, var = cond.predict(T_STAR)
    dense_mean, dense_cov = _dense_conditional(T_STAR, mean_fn)
    # predict's marginals are the dense conditional's (the Markov GP is exact),
    # so the draws' moments are checked against predict's mean and the dense
    # joint covariance, whose diagonal is predict's variance.
    assert jnp.allclose(mean, dense_mean, rtol=1e-8, atol=1e-10)
    assert jnp.allclose(var, jnp.diag(dense_cov), rtol=1e-8, atol=1e-10)
    _assert_moments(draws, mean, dense_cov)


def test_markov_gp_draws_at_the_training_times():
    cond = _prior().condition(Y, jnp.asarray(NOISE), mask=MASK)
    draws = cond.sample(jr.key(2), n_paths=S)
    assert draws.shape == (S, TIMES.shape[0])
    _assert_moments(draws, *_dense_conditional(TIMES))


def test_the_same_key_gives_the_same_draws():
    cond = _prior().condition(Y, jnp.asarray(NOISE), mask=MASK)
    first = cond.sample(jr.key(3), n_paths=4, t_star=T_STAR)
    assert jnp.array_equal(first, cond.sample(jr.key(3), n_paths=4, t_star=T_STAR))
    assert not jnp.allclose(first, cond.sample(jr.key(4), n_paths=4, t_star=T_STAR))


def _two_channel_model() -> LinearGaussianSSMPrior:
    T = 6
    A = jnp.broadcast_to(jnp.array([[0.9, 0.2], [-0.1, 0.8]]), (T, 2, 2))
    H = jnp.array([[1.0, 0.5], [-0.3, 1.0]])
    Q = jnp.broadcast_to(jnp.array([[0.2, 0.05], [0.05, 0.1]]), (T, 2, 2))
    R = jnp.array([[0.3, 0.1], [0.1, 0.4]])
    offset = 0.1 * jnp.arange(2 * T, dtype=float).reshape(T, 2)
    m0, P0 = jnp.array([0.3, -0.2]), jnp.array([[1.0, 0.2], [0.2, 0.5]])
    return LinearGaussianSSMPrior(A, H, Q, R, m0, P0, obs_offset=offset)


def test_lgssm_state_draws_have_the_smoothed_moments_at_each_step():
    prior = _two_channel_model()
    y = jr.normal(jr.key(5), (6, 2))
    mask = jnp.ones((6, 2), dtype=bool).at[2, 1].set(False).at[4, 0].set(False)
    cond = prior.condition(jnp.where(mask, y, jnp.nan), mask=mask)
    draws = cond.sample(jr.key(6), n_paths=S)
    assert draws.shape == (S, 6, 2)
    for t in range(6):
        _assert_moments(draws[:, t], cond.smoothed_means[t], cond.smoothed_covs[t])


def test_lgssm_prior_draws_of_f_have_the_gp_prior_moments():
    """For an ``as_lgssm`` model, H x + b is the GP prior on the times."""
    mean_fn = lambda t: 0.5 - 0.1 * t
    prior = _prior(mean_fn)
    lgssm = prior.as_lgssm(jnp.asarray(NOISE))
    states, y = lgssm.sample(jr.key(7), n=S)
    assert states.shape == (S, TIMES.shape[0], prior.state_dim)
    assert y.shape == (S, TIMES.shape[0], 1)
    gram = matern_kernel(TIMES[:, None], TIMES[:, None], VARIANCE, LENGTHSCALE, nu=1.5)
    f = jnp.einsum("snd,od->sn", states, lgssm.obs_model) + mean_fn(TIMES)
    _assert_moments(f, mean_fn(TIMES), gram)
    # The observations add the noise variance on the diagonal.
    _assert_moments(y[..., 0], mean_fn(TIMES), gram + NOISE * jnp.eye(TIMES.shape[0]))


def test_lgssm_prior_draws_match_markov_gp_from_innovations_in_moments():
    prior = _prior()
    states, _ = prior.as_lgssm(jnp.asarray(NOISE)).sample(jr.key(8), n=S)
    f = jnp.einsum("snd,od->sn", states, prior.as_lgssm(jnp.asarray(NOISE)).obs_model)
    eps = jr.normal(jr.key(9), (S, TIMES.shape[0], prior.state_dim))
    f_ref = markov_gp_from_innovations(prior, eps, sqrt="eigh")
    # Both are draws from the same Gaussian, so their difference in sample
    # means is within the combined band of two independent estimators.
    gram = matern_kernel(TIMES[:, None], TIMES[:, None], VARIANCE, LENGTHSCALE, nu=1.5)
    band = 7.0 * jnp.sqrt(2.0 * jnp.diag(gram) / S)
    assert jnp.all(jnp.abs(f.mean(axis=0) - f_ref.mean(axis=0)) <= band)
    _assert_moments(f_ref, jnp.zeros(TIMES.shape[0]), gram)
    _assert_moments(f, jnp.zeros(TIMES.shape[0]), gram)
