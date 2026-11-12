"""Tests for the whitened innovations path of a Markov GP.

`markov_gp_from_innovations` is the linear map f = μ + M ε built by a scan
over the kernel's discretisation, so the contract is M Mᵀ = K, the dense
prior covariance of the path. The tests check that identity exactly with
`jax.jacfwd`, statistically with Monte Carlo, at a near-zero step and for a
non-stationary kernel, and then the NumPyro sites under `trace`, `jit`,
`vmap`, NUTS, SVI and `Predictive`.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from pyrox_ssm import (
    IntegratedWienerSDE,
    MarkovGPPrior,
    MaternSDE,
    SumSDE,
    markov_gp_from_innovations,
    markov_gp_innovations_sample,
)
from ssm_reference import matern_kernel


jax.config.update("jax_enable_x64", True)

SQRTS = ("cholesky", "eigh")
NU = {0: 0.5, 1: 1.5, 2: 2.5}

# Irregular times over a few lengthscales, so every entry of K is well away
# from zero and an entrywise relative tolerance is meaningful.
TIMES = jnp.sort(jr.uniform(jr.key(3), (12,), minval=0.0, maxval=3.0))


def _matern_gram(times, variance, lengthscale, order):
    X = times[:, None]
    return matern_kernel(
        X, X, jnp.asarray(variance), jnp.asarray(lengthscale), nu=NU[order]
    )


def _kernel_and_gram(name, times):
    """An SDE kernel and its closed-form Gram on ``times``."""
    if name == "sum":
        sde = SumSDE((MaternSDE(1.0, 0.5, order=1), MaternSDE(0.5, 2.0, order=0)))
        K = _matern_gram(times, 1.0, 0.5, 1) + _matern_gram(times, 0.5, 2.0, 0)
        return sde, K
    order = {"matern12": 0, "matern32": 1, "matern52": 2}[name]
    return MaternSDE(1.3, 0.7, order=order), _matern_gram(times, 1.3, 0.7, order)


def _jacobian(prior, **kwargs):
    """M = ∂f/∂ε, flattened to (N, N d)."""
    n, d = prior.times.shape[0], prior.state_dim
    M = jax.jacfwd(lambda e: markov_gp_from_innovations(prior, e, **kwargs))(
        jnp.zeros((n, d))
    )
    return M.reshape(n, n * d)


# --- the identity M Mᵀ = K ------------------------------------------------


@pytest.mark.parametrize("sqrt", SQRTS)
@pytest.mark.parametrize("name", ["matern12", "matern32", "matern52", "sum"])
def test_jacobian_outer_product_is_the_gram(name: str, sqrt: str) -> None:
    sde, K = _kernel_and_gram(name, TIMES)
    M = _jacobian(MarkovGPPrior(sde, TIMES), sqrt=sqrt)
    # x64: the eigh route is exact to ~1e-14; the default Cholesky jitter
    # of 1e-10 per step moves M Mᵀ by under 1e-9 relative on this grid.
    np.testing.assert_allclose(M @ M.T, K, rtol=1e-8)


def test_mean_function_shifts_the_path_and_leaves_the_covariance() -> None:
    sde, K = _kernel_and_gram("matern32", TIMES)
    prior = MarkovGPPrior(sde, TIMES, mean_fn=lambda t: 0.5 + jnp.sin(t))
    d = prior.state_dim
    f0 = markov_gp_from_innovations(prior, jnp.zeros((TIMES.shape[0], d)))
    np.testing.assert_allclose(f0, 0.5 + jnp.sin(TIMES), rtol=1e-14)
    M = _jacobian(prior)
    np.testing.assert_allclose(M @ M.T, K, rtol=1e-8)
    eps = jr.normal(jr.key(0), (TIMES.shape[0], d))
    f = markov_gp_from_innovations(prior, eps)
    np.testing.assert_allclose(f, f0 + M @ eps.reshape(-1), rtol=1e-12)


def test_non_stationary_kernel_matches_the_kalman_prior_marginals() -> None:
    """IntegratedWienerSDE starts from ``initial_covariance``, not P∞.

    Its marginal variances are those of the Kalman filter run with every
    step masked, which propagates the same (P₀, Aₖ, Qₖ).
    """
    sde = IntegratedWienerSDE(diffusion=jnp.asarray(0.3))
    prior = MarkovGPPrior(sde, TIMES, init_cov=jnp.diag(jnp.array([2.0, 0.5])))
    M = _jacobian(prior, sqrt="eigh")
    n = TIMES.shape[0]
    _, P_pred, *_ = prior.filter(
        jnp.zeros(n), jnp.asarray(1.0), mask=jnp.zeros(n, dtype=bool)
    )
    np.testing.assert_allclose(jnp.diag(M @ M.T), P_pred[:, 0, 0], rtol=1e-12)


@pytest.mark.parametrize("name", ["matern12", "matern32", "matern52", "sum"])
def test_monte_carlo_covariance_is_within_three_standard_errors(name: str) -> None:
    times = TIMES[::2]
    sde, K = _kernel_and_gram(name, times)
    prior = MarkovGPPrior(sde, times)
    n_draws = 20_000
    eps = jr.normal(jr.key(1), (n_draws, times.shape[0], prior.state_dim))
    f = jax.vmap(markov_gp_from_innovations, in_axes=(None, 0))(prior, eps)
    C = f.T @ f / n_draws  # the mean is known to be zero
    # Var(fᵢ fⱼ) = Kᵢᵢ Kⱼⱼ + Kᵢⱼ² for a zero-mean Gaussian pair.
    se = jnp.sqrt((jnp.outer(jnp.diag(K), jnp.diag(K)) + K**2) / n_draws)
    assert jnp.all(jnp.abs(C - K) < 3.0 * se), float(jnp.max(jnp.abs(C - K) / se))


# --- near-singular process noise -----------------------------------------


def _near_duplicate_times():
    return jnp.array([0.0, 0.3, 0.3 + 1e-9, 0.8, 1.5])


def _bernoulli_loss(build_sde, sqrt):
    times = _near_duplicate_times()
    y = jnp.array([1.0, 0.0, 0.0, 1.0, 1.0])

    def loss(log_theta, eps):
        prior = MarkovGPPrior(build_sde(jnp.exp(log_theta)), times)
        f = markov_gp_from_innovations(prior, eps, sqrt=sqrt)
        return dist.Bernoulli(logits=f).log_prob(y).sum(), f

    return loss


@pytest.mark.parametrize("sqrt", SQRTS)
@pytest.mark.parametrize(
    "build_sde",
    [
        lambda theta: MaternSDE(theta[0], theta[1], order=2),
        lambda theta: IntegratedWienerSDE(diffusion=theta[0]),
    ],
    ids=["matern52", "integrated_wiener"],
)
def test_near_zero_step_gives_finite_path_and_gradients(build_sde, sqrt) -> None:
    loss = _bernoulli_loss(build_sde, sqrt)
    log_theta = jnp.log(jnp.array([1.0, 0.5]))
    eps = jr.normal(jr.key(2), (5, build_sde(jnp.ones(2)).state_dim))
    (value, f), grads = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(
        log_theta, eps
    )
    assert jnp.all(jnp.isfinite(f)) and jnp.isfinite(value)
    assert all(jnp.all(jnp.isfinite(g)) for g in grads)
    # The two nearly equal times carry nearly the same value.
    assert jnp.abs(f[2] - f[1]) < 1e-3


# --- validation -------------------------------------------------------------


def test_rejects_eps_of_the_wrong_shape() -> None:
    prior = MarkovGPPrior(MaternSDE(1.0, 0.5, order=1), TIMES)
    with pytest.raises(ValueError, match="eps must have shape"):
        markov_gp_from_innovations(prior, jnp.zeros((TIMES.shape[0], 3)))
    with pytest.raises(ValueError, match="eps must have shape"):
        markov_gp_from_innovations(prior, jnp.zeros((2 * TIMES.shape[0],)))


def test_rejects_an_unknown_square_root() -> None:
    prior = MarkovGPPrior(MaternSDE(1.0, 0.5, order=1), TIMES)
    with pytest.raises(ValueError, match="sqrt must be one of"):
        markov_gp_from_innovations(prior, jnp.zeros((TIMES.shape[0], 2)), sqrt="svd")


# --- transforms -----------------------------------------------------------


def test_jit_matches_eager() -> None:
    prior = MarkovGPPrior(MaternSDE(1.0, 0.5, order=1), TIMES)
    eps = jr.normal(jr.key(4), (TIMES.shape[0], 2))
    eager = markov_gp_from_innovations(prior, eps)
    jitted = jax.jit(lambda e: markov_gp_from_innovations(prior, e))(eps)
    np.testing.assert_allclose(jitted, eager, rtol=1e-14)


def test_vmap_over_sites_matches_a_loop() -> None:
    lengthscales = jnp.array([0.3, 0.7, 1.5])
    times = jnp.stack([TIMES, TIMES + 1.0, 2.0 * TIMES])
    eps = jr.normal(jr.key(5), (3, TIMES.shape[0], 2))

    def site(ell, t, e):
        return markov_gp_from_innovations(
            MarkovGPPrior(MaternSDE(1.0, ell, order=1), t), e
        )

    batched = jax.vmap(site)(lengthscales, times, eps)
    looped = jnp.stack(
        [site(*args) for args in zip(lengthscales, times, eps, strict=True)]
    )
    np.testing.assert_allclose(batched, looped, rtol=1e-12)


def test_leading_batch_dimensions_give_one_path_each() -> None:
    prior = MarkovGPPrior(MaternSDE(1.0, 0.5, order=1), TIMES)
    eps = jr.normal(jr.key(8), (2, 3, TIMES.shape[0], 2))
    batched = markov_gp_from_innovations(prior, eps)
    assert batched.shape == (2, 3, TIMES.shape[0])
    looped = jnp.stack(
        [jnp.stack([markov_gp_from_innovations(prior, e) for e in row]) for row in eps]
    )
    np.testing.assert_allclose(batched, looped, rtol=1e-14)


def _bernoulli_series(n):
    times = jnp.linspace(0.0, 10.0, n)
    y = (jr.uniform(jr.key(6), (n,)) < jax.nn.sigmoid(2.0 * jnp.sin(times))).astype(
        jnp.float64
    )
    return times, y


def _model(times, y=None):
    ell = numpyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))
    prior = MarkovGPPrior(MaternSDE(1.0, ell, order=1), times)
    f = markov_gp_innovations_sample("f", prior)
    numpyro.sample("y", dist.Bernoulli(logits=f), obs=y)


def test_sample_registers_one_eps_site_and_a_deterministic_path() -> None:
    times, y = _bernoulli_series(10)
    trace = handlers.trace(handlers.seed(_model, 0)).get_trace(times, y)
    assert set(trace) == {"lengthscale", "f_eps", "f", "y"}
    eps_site = trace["f_eps"]
    assert eps_site["type"] == "sample"
    assert eps_site["value"].shape == (10, 2)
    assert eps_site["fn"].event_shape == (10, 2)
    assert eps_site["fn"].batch_shape == ()
    assert trace["f"]["type"] == "deterministic"
    prior = MarkovGPPrior(MaternSDE(1.0, trace["lengthscale"]["value"], order=1), times)
    expected = markov_gp_from_innovations(prior, eps_site["value"])
    np.testing.assert_allclose(trace["f"]["value"], expected, rtol=1e-14)


def test_sample_under_a_plate_gives_one_independent_path_per_index() -> None:
    times, _ = _bernoulli_series(12)
    prior = MarkovGPPrior(MaternSDE(1.0, 0.5, order=1), times)

    def model():
        with numpyro.plate("series", 3):
            return markov_gp_innovations_sample("f", prior)

    trace = handlers.trace(handlers.seed(model, 0)).get_trace()
    eps = trace["f_eps"]["value"]
    assert eps.shape == (3, 12, 2)
    assert trace["f_eps"]["fn"].batch_shape == (3,)
    assert trace["f"]["value"].shape == (3, 12)
    expected = jnp.stack([markov_gp_from_innovations(prior, e) for e in eps])
    np.testing.assert_allclose(trace["f"]["value"], expected, rtol=1e-14)


def test_runs_under_nuts_smoke() -> None:
    times, y = _bernoulli_series(30)
    mcmc = MCMC(NUTS(_model), num_warmup=20, num_samples=20, progress_bar=False)
    mcmc.run(jr.key(0), times, y)
    samples = mcmc.get_samples()
    assert samples["f_eps"].shape == (20, 30, 2)
    assert samples["f"].shape == (20, 30)
    assert jnp.all(jnp.isfinite(samples["f"]))


def test_runs_under_svi_smoke() -> None:
    times, y = _bernoulli_series(30)
    svi = SVI(_model, AutoNormal(_model), numpyro.optim.Adam(0.05), Trace_ELBO())
    result = svi.run(jr.key(0), 50, times, y, progress_bar=False)
    assert jnp.all(jnp.isfinite(result.losses))
    assert result.losses[-10:].mean() < result.losses[:10].mean()


def test_predictive_returns_the_deterministic_path() -> None:
    times, _ = _bernoulli_series(15)
    draws = Predictive(_model, num_samples=4)(jr.key(7), times)
    assert draws["f"].shape == (4, 15)
    assert draws["f_eps"].shape == (4, 15, 2)
    assert draws["y"].shape == (4, 15)


@pytest.mark.slow
def test_nuts_on_a_2000_point_bernoulli_series() -> None:
    """The issue's integration run: N = 2000, 200 NUTS draws (run time only)."""
    times, y = _bernoulli_series(2000)
    mcmc = MCMC(NUTS(_model), num_warmup=200, num_samples=200, progress_bar=False)
    start = time.perf_counter()
    mcmc.run(jr.key(0), times, y)
    samples = mcmc.get_samples()
    jax.block_until_ready(samples)
    print(f"NUTS N=2000, 200 warmup + 200 draws: {time.perf_counter() - start:.1f} s")
    assert samples["f"].shape == (200, 2000)
    assert jnp.all(jnp.isfinite(samples["f"]))
    assert jnp.std(samples["lengthscale"]) > 0.0
