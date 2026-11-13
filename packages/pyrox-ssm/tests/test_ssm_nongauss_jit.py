"""The non-Gaussian Markov strategies as optimistix fixed points (#367).

* Every strategy reproduces the Python loop it replaced, written out below as
  the reference, to its ``tol``.
* A fit and its ``log_marginal_approx`` trace under ``jax.jit``, compile once,
  and report ``converged`` as a traced boolean.
* The implicit gradient of the Laplace evidence matches central finite
  differences and the gradient through 200 unrolled iterations, and does not
  depend on the damping.
* `markov_laplace_factor` runs under NUTS and SVI.
"""

from __future__ import annotations

import gaussx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import optimistix as optx
import pytest
from gaussx import (
    cavity_distribution,
    damped_natural_update,
    ep_tilted_moments,
    newton_update,
)
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from pyrox_ssm import (
    ExpectationPropagationMarkov,
    GaussNewtonMarkovInference,
    IntegratedWienerSDE,
    LaplaceMarkovInference,
    MarkovGPPrior,
    MaternSDE,
    PosteriorLinearizationMarkov,
    markov_laplace_factor,
)
from pyrox_ssm._inference_nongauss_markov import (
    _markov_smoothed_posterior,
    _mask_sites,
    _prior_marginal_variance,
)
from pyrox_ssm._markov import _observation_mask
from pyrox_ssm._markov_sites import _cavity_site_derivatives, as_gaussx_likelihood
from ssm_reference import BernoulliLikelihood


jax.config.update("jax_enable_x64", True)

SLOW = pytest.mark.slow


# --- the Python loops the fixed points replaced (pyrox 910fafe) -----------


def _python_loop_laplace(strategy, prior, likelihood, y, mask=None):
    """Newton on the smoothed mean, stopping when ``max |Δf| < tol``."""
    site_lik = as_gaussx_likelihood(likelihood, y, mask=mask)
    observed = _observation_mask(mask, prior.times)
    floor = strategy.precision_floor
    f = prior.mean(prior.times)
    nat1, nat2 = _mask_sites(f * floor, jnp.full_like(f, floor), observed, floor)
    for _ in range(strategy.max_iter):
        g, h = site_lik.site_derivatives(f)
        new1, new2 = newton_update(f, g, h, precision_floor=floor)
        nat1, nat2 = damped_natural_update(nat1, nat2, new1, new2, lr=strategy.damping)
        nat1, nat2 = _mask_sites(nat1, nat2, observed, floor)
        f_new, _, _ = _markov_smoothed_posterior(prior, nat1, nat2, observed)
        delta = jnp.max(jnp.where(observed, jnp.abs(f_new - f), 0.0))
        f = f_new
        if delta < strategy.tol:
            break
    g, h = site_lik.site_derivatives(f)
    nat1, nat2 = newton_update(f, g, h, precision_floor=floor)
    nat1, nat2 = _mask_sites(nat1, nat2, observed, floor)
    q_mean, q_var, _ = _markov_smoothed_posterior(prior, nat1, nat2, observed)
    return q_mean, q_var


def _pl_target(strategy, site_lik, cav_mean, cav_var):
    g, h = _cavity_site_derivatives(site_lik, cav_mean, cav_var, strategy.integrator)
    return newton_update(cav_mean, g, h, precision_floor=strategy.precision_floor)


def _ep_target(strategy, site_lik, cav_mean, cav_var):
    order = strategy.integrator.order
    tilted_mean, tilted_var = jax.vmap(
        lambda m, v, d: ep_tilted_moments(
            lambda f: site_lik.log_density(f, d), m, v, order=order
        )
    )(cav_mean, cav_var, site_lik.data)
    prec = jnp.maximum(1.0 / tilted_var - 1.0 / cav_var, strategy.precision_floor)
    return tilted_mean / tilted_var - cav_mean / cav_var, prec


def _python_loop_cavity(target, strategy, prior, likelihood, y, mask=None):
    """PL / EP from the prior marginals, stopping when ``max |Δm| < tol``."""
    site_lik = as_gaussx_likelihood(likelihood, y, mask=mask)
    observed = _observation_mask(mask, prior.times)
    floor = strategy.precision_floor
    q_mean = prior.mean(prior.times)
    q_var = _prior_marginal_variance(prior)
    nat1, nat2 = _mask_sites(
        jnp.zeros_like(q_mean), jnp.full_like(q_mean, floor), observed, floor
    )
    for _ in range(strategy.max_iter):
        cav_mean, cav_var = cavity_distribution(
            q_mean, q_var, nat1, nat2, precision_floor=floor
        )
        new1, new2 = target(strategy, site_lik, cav_mean, cav_var)
        nat1, nat2 = damped_natural_update(nat1, nat2, new1, new2, lr=strategy.damping)
        nat1, nat2 = _mask_sites(nat1, nat2, observed, floor)
        q_new, q_var, _ = _markov_smoothed_posterior(prior, nat1, nat2, observed)
        delta = jnp.max(jnp.where(observed, jnp.abs(q_new - q_mean), 0.0))
        q_mean = q_new
        if delta < strategy.tol:
            break
    return q_mean, q_var


def _bernoulli_series(n: int = 16):
    times = jnp.linspace(0.0, 5.0, n)
    y = (jnp.sin(2.0 * times) > 0.0).astype(jnp.float64)
    return MarkovGPPrior(MaternSDE(1.0, 0.7, order=1), times), y


def _trend_series():
    times = jnp.linspace(0.0, 5.0, 16)
    y = (times > 2.5).astype(jnp.float64)
    sde = IntegratedWienerSDE(diffusion=jnp.asarray(0.1))
    prior = MarkovGPPrior(sde, times, init_cov=jnp.diag(jnp.asarray([1.0, 0.1])))
    return prior, y


def _masked_series():
    times = jnp.sort(jr.uniform(jr.key(0), (12,), maxval=5.0))
    y = (jnp.sin(2.0 * times) > 0).astype(jnp.float64)
    mask = jnp.arange(12) != 5
    return MarkovGPPrior(MaternSDE(1.0, 0.7, order=1), times), y, mask


# The old loop stops within about tol of the fixed point and the new one (a
# tenth of tol on the sites) closer, so the two agree to tol: the largest gap
# measured on these problems is 0.8 tol (EP, tol=1e-9). Laplace converges
# quadratically and agrees to ~1e-15.


@pytest.mark.parametrize(
    ("strategy", "reference", "problem"),
    [
        pytest.param(
            LaplaceMarkovInference(), _python_loop_laplace, "bernoulli", id="laplace"
        ),
        pytest.param(
            LaplaceMarkovInference(max_iter=10),
            _python_loop_laplace,
            "trend",
            id="laplace-trend",
            marks=SLOW,
        ),
        pytest.param(
            GaussNewtonMarkovInference(),
            _python_loop_laplace,
            "masked",
            id="gauss-newton-masked",
            marks=SLOW,
        ),
        pytest.param(
            PosteriorLinearizationMarkov(max_iter=50),
            lambda *a, **k: _python_loop_cavity(_pl_target, *a, **k),
            "bernoulli",
            id="pl",
            marks=SLOW,
        ),
        pytest.param(
            PosteriorLinearizationMarkov(max_iter=100),
            lambda *a, **k: _python_loop_cavity(_pl_target, *a, **k),
            "trend",
            id="pl-trend",
            marks=SLOW,
        ),
        pytest.param(
            ExpectationPropagationMarkov(),
            lambda *a, **k: _python_loop_cavity(_ep_target, *a, **k),
            "masked",
            id="ep-masked",
        ),
        pytest.param(
            ExpectationPropagationMarkov(max_iter=400, tol=1e-9),
            lambda *a, **k: _python_loop_cavity(_ep_target, *a, **k),
            "bernoulli",
            id="ep-tight",
            marks=SLOW,
        ),
    ],
)
def test_strategies_match_the_python_loop(strategy, reference, problem) -> None:
    mask = None
    if problem == "bernoulli":
        prior, y = _bernoulli_series()
    elif problem == "trend":
        prior, y = _trend_series()
    else:
        prior, y, mask = _masked_series()
        y = jnp.where(mask, y, jnp.nan)
    likelihood = BernoulliLikelihood()
    fit = strategy.fit(prior, likelihood, y, mask=mask)
    ref_mean, ref_var = reference(strategy, prior, likelihood, y, mask=mask)
    assert bool(fit.converged)
    assert jnp.max(jnp.abs(fit.q_mean - ref_mean)) < strategy.tol
    assert jnp.max(jnp.abs(fit.q_var - ref_var)) < strategy.tol


# --- jit ------------------------------------------------------------------


def _fit(theta, strategy, mask=None):
    """Fit the Bernoulli series under a Matérn-3/2 with ``(variance, ℓ)``."""
    _, y = _bernoulli_series()
    times = jnp.linspace(0.0, 5.0, y.shape[0])
    prior = MarkovGPPrior(MaternSDE(theta[0], theta[1], order=1), times)
    return strategy.fit(prior, BernoulliLikelihood(), y, mask=mask)


THETA = jnp.asarray([1.3, 0.7])


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param(LaplaceMarkovInference(), id="laplace"),
        pytest.param(GaussNewtonMarkovInference(), id="gauss-newton", marks=SLOW),
        pytest.param(PosteriorLinearizationMarkov(max_iter=100), id="pl", marks=SLOW),
        pytest.param(ExpectationPropagationMarkov(max_iter=100), id="ep"),
    ],
)
def test_the_log_marginal_jits_with_a_traced_converged_flag(strategy) -> None:
    traces = []

    def evidence(theta):
        traces.append(1)  # runs only while jit traces the function
        fit = _fit(theta, strategy)
        # Inside jit the flags are tracers, not Python values.
        assert isinstance(fit.converged, jax.core.Tracer)
        assert isinstance(fit.n_iter, jax.core.Tracer)
        return fit.log_marginal_approx, fit.converged, fit.n_iter

    jitted = jax.jit(evidence)
    value, converged, n_iter = jitted(THETA)
    eager = _fit(THETA, strategy)
    assert converged.dtype == jnp.bool_ and converged.shape == ()
    assert bool(converged) and bool(eager.converged)
    assert int(n_iter) == int(eager.n_iter)
    # jit and eager run the same ops, fused differently.
    assert jnp.allclose(value, eager.log_marginal_approx, rtol=1e-12, atol=0.0)
    # A second call with the same shapes reuses the compiled loop: it does not
    # trace again. (Counting traces, not jit's C++ cache entries, which live in
    # a cache shared by every jitted function in the session.)
    jitted(THETA * 1.1)
    assert len(traces) == 1


def test_converged_is_false_when_max_iter_runs_out() -> None:
    fit = jax.jit(lambda theta: _fit(theta, LaplaceMarkovInference(max_iter=1)))(THETA)
    assert not bool(fit.converged)
    assert int(fit.n_iter) == 1


# --- gradients ------------------------------------------------------------


def _laplace_evidence(strategy):
    return lambda theta: _fit(theta, strategy).log_marginal_approx


def test_laplace_gradient_matches_central_differences() -> None:
    # tol=1e-10 puts the fit at the fixed point to round-off, so the central
    # difference (eps=1e-5, truncation ~eps^2) resolves the gradient to ~1e-9.
    strategy = LaplaceMarkovInference(max_iter=50, tol=1e-10)
    evidence = jax.jit(_laplace_evidence(strategy))
    grad = jax.jit(jax.grad(_laplace_evidence(strategy)))(THETA)
    eps = 1e-5
    fd = jnp.stack(
        [
            (evidence(THETA.at[i].add(eps)) - evidence(THETA.at[i].add(-eps)))
            / (2.0 * eps)
            for i in range(2)
        ]
    )
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.allclose(grad, fd, rtol=1e-4, atol=0.0)


@pytest.mark.parametrize("damping", [1.0, pytest.param(0.3, marks=SLOW)])
def test_implicit_gradient_equals_200_unrolled_iterations(damping) -> None:
    implicit = LaplaceMarkovInference(max_iter=300, tol=1e-10, damping=damping)
    # tol=0 never stops early, so the loop runs all 200 iterations and the
    # gradient differentiates through every one of them.
    unrolled = LaplaceMarkovInference(
        max_iter=200,
        tol=0.0,
        damping=damping,
        adjoint=optx.RecursiveCheckpointAdjoint(),
    )
    assert int(_fit(THETA, unrolled).n_iter) == 200
    g_implicit = jax.jit(jax.grad(_laplace_evidence(implicit)))(THETA)
    g_unrolled = jax.jit(jax.grad(_laplace_evidence(unrolled)))(THETA)
    # 200 iterations converge to round-off even at damping 0.3 (0.7^200).
    assert jnp.allclose(g_implicit, g_unrolled, rtol=1e-8, atol=1e-10)


def test_implicit_gradient_does_not_depend_on_the_damping() -> None:
    full = LaplaceMarkovInference(max_iter=50, tol=1e-10, damping=1.0)
    damped = LaplaceMarkovInference(max_iter=300, tol=1e-10, damping=0.3)
    g_full = jax.jit(jax.grad(_laplace_evidence(full)))(THETA)
    g_damped = jax.jit(jax.grad(_laplace_evidence(damped)))(THETA)
    assert int(_fit(THETA, damped).n_iter) > int(_fit(THETA, full).n_iter)
    assert jnp.allclose(g_full, g_damped, rtol=1e-8, atol=1e-10)


# --- the NumPyro factor ---------------------------------------------------


def _model(times, y, strategy, reject_unconverged=False):
    variance = numpyro.sample("variance", dist.LogNormal(0.0, 0.5))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0.0, 0.5))
    prior = MarkovGPPrior(MaternSDE(variance, lengthscale, order=1), times)
    markov_laplace_factor(
        "y",
        prior,
        BernoulliLikelihood(),
        y,
        strategy=strategy,
        reject_unconverged=reject_unconverged,
    )


def _factor_value(strategy, *, reject_unconverged):
    prior, y = _bernoulli_series()
    model = handlers.substitute(
        handlers.seed(_model, rng_seed=0),
        data={"variance": THETA[0], "lengthscale": THETA[1]},
    )
    trace = handlers.trace(model).get_trace(
        prior.times, y, strategy, reject_unconverged
    )
    return trace["y"]["fn"].log_factor


def test_factor_is_the_log_marginal_and_rejects_on_request() -> None:
    strategy = LaplaceMarkovInference()
    expected = _fit(THETA, strategy).log_marginal_approx
    assert jnp.allclose(
        _factor_value(strategy, reject_unconverged=True), expected, rtol=1e-12
    )
    short = LaplaceMarkovInference(max_iter=1)
    assert jnp.isfinite(_factor_value(short, reject_unconverged=False))
    assert _factor_value(short, reject_unconverged=True) == -jnp.inf


def _run_nuts(num_warmup: int, num_samples: int) -> MCMC:
    prior, y = _bernoulli_series()
    mcmc = MCMC(
        NUTS(_model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=False,
    )
    mcmc.run(
        jr.key(0),
        prior.times,
        y,
        LaplaceMarkovInference(),
        True,
        extra_fields=("diverging",),
    )
    return mcmc


def test_factor_runs_under_nuts_smoke() -> None:
    samples = _run_nuts(num_warmup=5, num_samples=5).get_samples()
    for name in ("variance", "lengthscale"):
        assert samples[name].shape == (5,)
        assert jnp.all(jnp.isfinite(samples[name]))


@SLOW
def test_factor_runs_200_nuts_draws() -> None:
    """Example 4, Route A: NUTS over (σ², ℓ) with the Laplace evidence."""
    mcmc = _run_nuts(num_warmup=200, num_samples=200)
    samples = mcmc.get_samples()
    for name in ("variance", "lengthscale"):
        assert samples[name].shape == (200,)
        assert jnp.all(jnp.isfinite(samples[name]))
        assert jnp.std(samples[name]) > 0.0
    assert not jnp.any(mcmc.get_extra_fields()["diverging"])


def test_factor_runs_under_svi_smoke() -> None:
    prior, y = _bernoulli_series()
    svi = SVI(
        _model,
        AutoDelta(_model),
        numpyro.optim.Adam(0.05),
        Trace_ELBO(),
    )
    result = svi.run(
        jr.key(0), 20, prior.times, y, LaplaceMarkovInference(), progress_bar=False
    )
    assert jnp.all(jnp.isfinite(result.losses))
    # MAP on the Laplace evidence plus the log-normal priors goes downhill.
    assert result.losses[-1] < result.losses[0]


def test_gradient_at_a_non_finite_fit_is_a_value_not_an_error() -> None:
    """Far out in a NUTS step-size search the SDE discretisation overflows.

    At lengthscale e^-10 the Matérn transition is NaN, so the fit is too; the
    implicit solve must then give a non-finite gradient rather than raise.
    """
    theta = jnp.asarray([1.0, jnp.exp(-10.0)])
    strategy = LaplaceMarkovInference()
    value, grad = jax.jit(jax.value_and_grad(_laplace_evidence(strategy)))(theta)
    assert not jnp.isfinite(value)
    assert not bool(_fit(theta, strategy).converged)
    assert grad.shape == (2,)


def test_gauss_hermite_default_survives_the_new_field() -> None:
    """``adjoint`` is appended, so the integrator still resolves as before."""
    assert PosteriorLinearizationMarkov().integrator == gaussx.GaussHermiteIntegrator(
        order=20
    )
    assert isinstance(ExpectationPropagationMarkov().adjoint, optx.ImplicitAdjoint)
