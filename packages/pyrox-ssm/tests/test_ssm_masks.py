"""Observation masks on every Markov entry point.

A masked step runs the Kalman predict only, so the contracts are:

* ``mask=None`` is today's result, bit for bit;
* masking step ``k`` equals dropping step ``k`` from the series (log
  marginal, smoothed marginals and predictions at the kept steps), for the
  Gaussian path, every non-Gaussian strategy and the sparse ELBO;
* a ``NaN`` at a masked step reaches neither the value nor the gradient;
* it all works under ``jit``, ``vmap`` over sites and NUTS.

Masking against dropping is exact in exact arithmetic: two predict steps of
lengths ``a`` and ``b`` compose to one of length ``a + b``. In floating point
the two paths round differently, so the comparisons use ``rtol=1e-10`` under
x64 (the observed gap is ~1e-15).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from gaussx import GaussHermiteIntegrator
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from pyrox.likelihoods import GaussianLikelihood
from pyrox_ssm import (
    ExpectationPropagationMarkov,
    GaussNewtonMarkovInference,
    LaplaceMarkovInference,
    MarkovGPPrior,
    MaternSDE,
    PosteriorLinearizationMarkov,
    SparseMarkovGPPrior,
    markov_gp_factor,
    pad_sites,
    sparse_markov_elbo,
    sparse_markov_factor,
)
from pyrox_ssm._markov import _build_dt_full, _kalman_filter, _rts_smoother
from pyrox_ssm._markov_sites import as_gaussx_likelihood
from ssm_reference import BernoulliLikelihood, FullRankGuide


jax.config.update("jax_enable_x64", True)

N = 12
K = 5  # the masked step
NOISE = jnp.asarray(0.1)
RTOL = 1e-10  # masked vs dropped: two predict steps vs one, ~1e-15 apart
# Laplace and EP run in the fast tier; Gauss-Newton (Laplace with another
# floor) and PL, ~25 s each, run in the slow tier.
SLOW = pytest.mark.slow


def _series(n: int = N, seed: int = 0) -> tuple[jax.Array, jax.Array]:
    times = jnp.sort(jr.uniform(jr.key(seed), (n,), maxval=5.0))
    return times, jnp.sin(2.0 * times)


def _prior(times: jax.Array, *, order: int = 1, lengthscale=0.7) -> MarkovGPPrior:
    return MarkovGPPrior(MaternSDE(1.0, lengthscale, order=order), times)


def _mask_k(n: int = N, k: int = K) -> jax.Array:
    return jnp.arange(n) != k


def _f_marginals(prior: MarkovGPPrior, m, P) -> tuple[jax.Array, jax.Array]:
    _F, _L, H, _Qc, _P = prior.sde_kernel.sde_params()
    return (m @ H.T)[:, 0], jax.vmap(lambda P_k: (H @ P_k @ H.T)[0, 0])(P)


# --- mask=None is today's result -----------------------------------------


def test_mask_none_is_the_unmasked_filter_bit_for_bit() -> None:
    """Every Gaussian entry point with ``mask=None`` equals the all-ones path.

    The reference is the pre-mask construction, spelled out: the private
    filter and smoother over ``jnp.ones_like(times)``.
    """
    times, y = _series()
    prior = _prior(times)
    F, _L, H, _Qc, _P = prior.sde_kernel.sde_params()
    A_seq, Q_seq = prior.sde_kernel.discretise_sequence(_build_dt_full(times))
    R_seq = jnp.broadcast_to(prior._R(NOISE), times.shape)
    reference = _kalman_filter(
        F,
        H,
        prior.initial_covariance(),
        A_seq,
        Q_seq,
        y - prior.mean(times),
        jnp.ones_like(times),
        R_seq,
    )
    m_ref, P_ref = _rts_smoother(*reference[:4], A_seq)

    for out, ref in zip(prior.filter(y, NOISE, mask=None), reference, strict=True):
        assert jnp.array_equal(out, ref)
    assert jnp.array_equal(prior.log_marginal(y, NOISE, mask=None), reference[-1])
    m, P, lm = prior.smooth(y, NOISE, mask=None)
    assert jnp.array_equal(m, m_ref) and jnp.array_equal(P, P_ref)
    assert jnp.array_equal(lm, reference[-1])

    t_star = jnp.linspace(-0.5, 6.0, 7)
    unmasked = prior.condition(y, NOISE).predict(t_star)
    for mask in (None, jnp.ones(N, dtype=bool)):
        cond = prior.condition(y, NOISE, mask=mask)
        for a, b in zip(cond.predict(t_star), unmasked, strict=True):
            assert jnp.array_equal(a, b)


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param(LaplaceMarkovInference(max_iter=5), id="laplace"),
        pytest.param(
            GaussNewtonMarkovInference(max_iter=5), id="gauss-newton", marks=SLOW
        ),
        pytest.param(PosteriorLinearizationMarkov(max_iter=5), id="pl", marks=SLOW),
        pytest.param(ExpectationPropagationMarkov(max_iter=5), id="ep"),
    ],
)
def test_mask_none_and_all_true_match_the_unmasked_strategy(strategy) -> None:
    times, y = _series()
    prior = _prior(times)
    yb = (y > 0).astype(jnp.float64)
    lik = BernoulliLikelihood()
    unmasked = strategy.fit(prior, lik, yb)
    t_star = jnp.linspace(-0.5, 6.0, 7)
    for mask in (None, jnp.ones(N, dtype=bool)):
        cond = strategy.fit(prior, lik, yb, mask=mask)
        for name in (
            "site_nat1",
            "site_nat2",
            "q_mean",
            "q_var",
            "log_marginal_approx",
        ):
            assert jnp.array_equal(getattr(cond, name), getattr(unmasked, name)), name
        assert jnp.array_equal(cond.n_iter, unmasked.n_iter)
        for a, b in zip(cond.predict(t_star), unmasked.predict(t_star), strict=True):
            assert jnp.array_equal(a, b)


def test_mask_none_matches_the_unmasked_sparse_elbo() -> None:
    times, y = _series()
    prior = SparseMarkovGPPrior(MaternSDE(1.0, 0.7, order=1), jnp.linspace(0, 5, 4))
    guide = FullRankGuide.init(4, scale=0.5)
    lik = GaussianLikelihood(NOISE)
    a = sparse_markov_elbo(prior, guide, lik, times, y)
    assert jnp.array_equal(
        a, sparse_markov_elbo(prior, guide, lik, times, y, mask=None)
    )


# --- masking a step equals dropping it -----------------------------------


@pytest.mark.parametrize("order", [0, 1], ids=["matern12", "matern32"])
def test_masking_step_k_equals_dropping_it(order: int) -> None:
    times, y = _series()
    keep = _mask_k()
    prior = _prior(times, order=order)
    dropped = _prior(times[keep], order=order)
    y_nan = y.at[K].set(jnp.nan)

    lm = prior.log_marginal(y_nan, NOISE, mask=keep)
    assert jnp.allclose(lm, dropped.log_marginal(y[keep], NOISE), rtol=RTOL, atol=0)

    m, P, _ = prior.smooth(y_nan, NOISE, mask=keep)
    m_d, P_d, _ = dropped.smooth(y[keep], NOISE)
    mean, var = _f_marginals(prior, m, P)
    mean_d, var_d = _f_marginals(dropped, m_d, P_d)
    assert jnp.allclose(mean[keep], mean_d, rtol=RTOL, atol=1e-13)
    assert jnp.allclose(var[keep], var_d, rtol=RTOL, atol=1e-13)

    t_star = jnp.linspace(-0.5, 6.0, 9)
    pred = prior.condition(y_nan, NOISE, mask=keep).predict(t_star)
    pred_d = dropped.condition(y[keep], NOISE).predict(t_star)
    for a, b in zip(pred, pred_d, strict=True):
        assert jnp.allclose(a, b, rtol=RTOL, atol=1e-13)


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param(LaplaceMarkovInference(), id="laplace"),
        pytest.param(GaussNewtonMarkovInference(), id="gauss-newton", marks=SLOW),
        pytest.param(PosteriorLinearizationMarkov(), id="pl", marks=SLOW),
        pytest.param(ExpectationPropagationMarkov(), id="ep"),
    ],
)
def test_nongauss_masking_step_k_equals_dropping_it(strategy) -> None:
    """A masked site has zero naturals and the kept sites fit as if it were gone."""
    times, y = _series()
    keep = _mask_k()
    yb = (y > 0).astype(jnp.float64)
    lik = BernoulliLikelihood()
    masked = strategy.fit(_prior(times), lik, yb.at[K].set(jnp.nan), mask=keep)
    dropped = strategy.fit(_prior(times[keep]), lik, yb[keep])

    assert masked.site_nat1[K] == 0.0 and masked.site_nat2[K] == 0.0
    assert jnp.array_equal(masked.n_iter, dropped.n_iter)
    for name in ("site_nat1", "site_nat2", "q_mean", "q_var"):
        assert jnp.allclose(
            getattr(masked, name)[keep], getattr(dropped, name), rtol=RTOL, atol=1e-13
        ), name
    assert jnp.allclose(
        masked.log_marginal_approx, dropped.log_marginal_approx, rtol=RTOL, atol=0
    )
    t_star = jnp.linspace(-0.5, 6.0, 9)
    for a, b in zip(masked.predict(t_star), dropped.predict(t_star), strict=True):
        assert jnp.allclose(a, b, rtol=RTOL, atol=1e-13)


def test_condition_nongauss_forwards_the_mask() -> None:
    times, y = _series()
    keep = _mask_k()
    yb = (y > 0).astype(jnp.float64)
    prior = _prior(times)
    strategy = LaplaceMarkovInference(max_iter=5)
    via_method = prior.condition_nongauss(
        BernoulliLikelihood(), yb, strategy=strategy, mask=keep
    )
    direct = strategy.fit(prior, BernoulliLikelihood(), yb, mask=keep)
    assert jnp.array_equal(via_method.q_mean, direct.q_mean)
    assert jnp.array_equal(via_method.mask, keep)


def test_masked_site_likelihood_has_zero_derivatives() -> None:
    y = jnp.array([1.0, jnp.nan, 0.0])
    mask = jnp.array([True, False, True])
    f = jnp.array([0.3, -0.2, 0.4])
    site_lik = as_gaussx_likelihood(BernoulliLikelihood(), y, mask=mask)
    g, h = site_lik.site_derivatives(f)
    assert g[1] == 0.0 and h[1] == 0.0
    kept = jnp.array([0, 2])
    reference = as_gaussx_likelihood(BernoulliLikelihood(), y[kept])
    g_ref, h_ref = reference.site_derivatives(f[kept])
    assert jnp.array_equal(g[kept], g_ref) and jnp.array_equal(h[kept], h_ref)
    assert jnp.array_equal(site_lik.log_prob(f), reference.log_prob(f[kept]))


@pytest.mark.parametrize("gaussian", [True, False], ids=["gaussian", "bernoulli"])
def test_sparse_elbo_masking_equals_dropping(gaussian: bool) -> None:
    times, y = _series()
    keep = _mask_k()
    if gaussian:
        lik, integrator, data = GaussianLikelihood(NOISE), None, y
    else:
        lik = BernoulliLikelihood()
        integrator = GaussHermiteIntegrator(order=20)
        data = (y > 0).astype(jnp.float64)
    prior = SparseMarkovGPPrior(MaternSDE(1.0, 0.7, order=1), jnp.linspace(0, 5, 4))
    guide = FullRankGuide.init(4, scale=0.5)
    masked = sparse_markov_elbo(
        prior,
        guide,
        lik,
        times,
        data.at[K].set(jnp.nan),
        integrator=integrator,
        mask=keep,
    )
    dropped = sparse_markov_elbo(
        prior, guide, lik, times[keep], data[keep], integrator=integrator
    )
    assert jnp.allclose(masked, dropped, rtol=1e-12, atol=0)

    with handlers.trace() as tr:
        sparse_markov_factor(
            "elbo",
            prior,
            guide,
            lik,
            times,
            data.at[K].set(jnp.nan),
            integrator=integrator,
            mask=keep,
        )
    assert jnp.array_equal(tr["elbo"]["fn"].log_factor, masked)


# --- NaN safety ----------------------------------------------------------


def test_nan_at_a_masked_step_gives_a_finite_gradient() -> None:
    times, y = _series()
    keep = _mask_k()
    y_nan = y.at[K].set(jnp.nan)

    def loss(params):
        ell, noise = params
        return _prior(times, lengthscale=ell).log_marginal(y_nan, noise, mask=keep)

    params = (jnp.asarray(0.7), NOISE)
    value, grads = jax.value_and_grad(loss)(params)
    assert jnp.isfinite(value)
    assert all(jnp.isfinite(g) for g in grads)

    def dropped(params):
        ell, noise = params
        return _prior(times[keep], lengthscale=ell).log_marginal(y[keep], noise)

    for g, g_d in zip(grads, jax.grad(dropped)(params), strict=True):
        assert jnp.allclose(g, g_d, rtol=1e-9)

    # Under vmap the Kalman step's branch becomes a select that evaluates the
    # update too, so the NaN must already be gone before the residual.
    batched = jax.vmap(jax.grad(lambda ell: loss((ell, NOISE))))(jnp.array([0.5, 0.9]))
    assert jnp.all(jnp.isfinite(batched))


def test_mask_shape_mismatch_raises() -> None:
    times, y = _series()
    with pytest.raises(ValueError, match="mask has shape"):
        _prior(times).log_marginal(y, NOISE, mask=jnp.ones(N - 1, dtype=bool))


# --- transforms ----------------------------------------------------------


def test_masked_log_marginal_under_jit_and_vmap_over_sites() -> None:
    t1, y1 = _series(12, seed=1)
    t2, y2 = _series(7, seed=2)
    t, y, mask = pad_sites([t1, t2], [y1, y2])

    @jax.jit
    def site_log_marginals(ell, t, y, mask):
        def one(ell_s, t_s, y_s, m_s):
            return _prior(t_s, lengthscale=ell_s).log_marginal(y_s, NOISE, mask=m_s)

        return jax.vmap(one)(ell, t, y, mask)

    ell = jnp.array([0.6, 0.9])
    out = site_log_marginals(ell, t, y, mask)
    expected = jnp.array(
        [
            _prior(t1, lengthscale=0.6).log_marginal(y1, NOISE),
            _prior(t2, lengthscale=0.9).log_marginal(y2, NOISE),
        ]
    )
    # padding adds predict-only steps after the last observation: exact
    assert jnp.allclose(out, expected, rtol=1e-12, atol=0)


def test_markov_gp_factor_registers_the_masked_log_marginal() -> None:
    times, y = _series()
    keep = _mask_k()
    prior = _prior(times)
    with handlers.trace() as tr:
        markov_gp_factor("obs", prior, y.at[K].set(jnp.nan), NOISE, mask=keep)
    expected = prior.log_marginal(y.at[K].set(jnp.nan), NOISE, mask=keep)
    assert jnp.array_equal(tr["obs"]["fn"].log_factor, expected)


def _pooled_model(t, y, mask):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    with numpyro.plate("site", t.shape[0]):
        ell = numpyro.sample("lengthscale", dist.LogNormal(mu, 0.5))

    def site(ell_s, t_s, y_s, m_s):
        return _prior(t_s, lengthscale=ell_s).log_marginal(y_s, NOISE, mask=m_s)

    numpyro.factor("y", jnp.sum(jax.vmap(site)(ell, t, y, mask)))


def _run_pooled_nuts(num_warmup: int, num_samples: int):
    t1, y1 = _series(10, seed=3)
    t2, y2 = _series(6, seed=4)
    t, y, mask = pad_sites([t1, t2], [y1.at[2].set(jnp.nan), y2])
    mask = mask.at[0, 2].set(False)  # a gap inside site 0 as well as padding
    mcmc = MCMC(
        NUTS(_pooled_model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=False,
    )
    mcmc.run(jr.key(0), t, y, mask, extra_fields=("diverging",))
    return mcmc


def test_pooled_two_site_model_runs_under_nuts_smoke() -> None:
    mcmc = _run_pooled_nuts(num_warmup=5, num_samples=5)
    samples = mcmc.get_samples()
    assert samples["lengthscale"].shape == (5, 2)
    assert jnp.all(jnp.isfinite(samples["lengthscale"]))


@pytest.mark.slow
def test_pooled_two_site_model_runs_50_nuts_steps() -> None:
    mcmc = _run_pooled_nuts(num_warmup=50, num_samples=50)
    samples = mcmc.get_samples()
    assert samples["lengthscale"].shape == (50, 2)
    assert jnp.all(jnp.isfinite(samples["lengthscale"]))
    assert not jnp.any(mcmc.get_extra_fields()["diverging"])
