"""Tests for `SharedPlusSiteMarkovGP`, a shared GP plus one GP per site.

The contracts:

* the log marginal equals the dense Gaussian over the observed (t, s) pairs,
  with covariance ``K_g ⊗ 1 1ᵀ + blockdiag_s(K_h,s) + diag(σ²_s)`` built from
  the closed-form Matérn kernel, for pooled, per-site and mixed site
  hyperparameters and for a per-site noise variance;
* with one site whose kernel has zero variance it is `MarkovGPPrior` on g;
* ``decompose`` reads the shared and site blocks of the smoothed state: they
  sum to the smoothed site means and equal the dense conditional means;
* ``from_sites`` merges ragged site times onto one grid with ``NaN`` and a
  mask where a site was not observed, and the ``NaN`` stays out of the value
  and the gradient;
* the log marginal differentiates under ``jit`` (matching the dense gradient)
  and runs inside NUTS.

The dense references are exact in exact arithmetic; under x64 the Kalman
recursion and the dense Cholesky round differently by ~1e-13 on these sizes,
so they are compared at ``rtol=1e-10``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from jax.scipy.stats import multivariate_normal
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from pyrox_ssm import (
    MarkovGPPrior,
    MaternSDE,
    SharedPlusSiteMarkovGP,
    lgssm_factor,
)
from ssm_reference import matern_kernel


jax.config.update("jax_enable_x64", True)

N_DAYS = 40
N_SITES = 3
SHARED = MaternSDE(variance=jnp.asarray(1.0), lengthscale=jnp.asarray(8.0), order=1)
SHARED_NU = 1.5
SITE_NU = 0.5  # the site kernels are Matérn-1/2, order 0
POOLED_SITE = MaternSDE(
    variance=jnp.asarray(0.3), lengthscale=jnp.asarray(2.0), order=0
)
PER_SITE = MaternSDE(
    variance=jnp.array([0.3, 0.5, 0.2]),
    lengthscale=jnp.array([2.0, 1.0, 3.0]),
    order=0,
)
MIXED_SITE = MaternSDE(
    variance=jnp.array([0.3, 0.5, 0.2]), lengthscale=jnp.asarray(2.0), order=0
)
NOISE = jnp.asarray(0.05)
NOISE_PER_SITE = jnp.array([0.05, 0.1, 0.02])


# --- helpers -------------------------------------------------------------


def _ragged_times(key, n_sites: int = N_SITES, n_days: int = N_DAYS):
    """Each site keeps a random ~half of the days, so the sites partly overlap."""
    keys = jr.split(key, n_sites)
    out = []
    for k in keys:
        keep = np.asarray(jr.bernoulli(k, 0.5, (n_days,)))
        out.append(jnp.asarray(np.nonzero(keep)[0], dtype=jnp.float64))
    return out


def _site_params(site: MaternSDE, n_sites: int):
    """``(variance, lengthscale)`` of each site kernel, ``(S,)`` each."""
    return (
        jnp.broadcast_to(site.variance, (n_sites,)),
        jnp.broadcast_to(site.lengthscale, (n_sites,)),
    )


def _dense_cov(shared, site, t_obs, s_obs, noise, n_sites: int):
    """``K_g + [s_i = s_j] K_h,s_i + σ²_{s_i} δ_ij`` over the observed pairs."""
    x = t_obs[:, None]
    K_g = matern_kernel(x, x, shared.variance, shared.lengthscale, nu=SHARED_NU)
    var_h, ell_h = _site_params(site, n_sites)
    K_h_all = jax.vmap(lambda v, ell: matern_kernel(x, x, v, ell, nu=SITE_NU))(
        var_h, ell_h
    )  # (S, N, N), site s's kernel between every pair of observed times
    same_site = s_obs[:, None] == s_obs[None, :]
    idx = jnp.arange(t_obs.shape[0])
    K_h = jnp.where(same_site, K_h_all[s_obs[:, None], idx[:, None], idx[None, :]], 0.0)
    noise = jnp.broadcast_to(noise, (n_sites,))[s_obs]
    return K_g + K_h + jnp.diag(noise)


def _observed_pairs(model: SharedPlusSiteMarkovGP, y_grid):
    """Times, site indices and values of the observed (t, s) pairs."""
    rows, cols = np.nonzero(np.asarray(model.mask))
    return model.times[rows], jnp.asarray(cols), y_grid[rows, cols]


def _dense_log_marginal(model, y_grid, noise):
    t_obs, s_obs, y_obs = _observed_pairs(model, y_grid)
    K = _dense_cov(model.shared, model.site, t_obs, s_obs, noise, model.n_sites)
    return multivariate_normal.logpdf(y_obs, jnp.zeros_like(y_obs), K)


def _dataset(key, site=POOLED_SITE, noise=NOISE):
    """Ragged sites, observations drawn from the dense model itself."""
    k_t, k_y = jr.split(key)
    times = _ragged_times(k_t)
    placeholder = [jnp.zeros_like(t) for t in times]
    model, _ = SharedPlusSiteMarkovGP.from_sites(SHARED, site, times, placeholder)
    t_obs, s_obs, _ = _observed_pairs(model, jnp.zeros(model.mask.shape))
    K = _dense_cov(SHARED, site, t_obs, s_obs, noise, N_SITES)
    y_obs = jnp.linalg.cholesky(K) @ jr.normal(k_y, t_obs.shape)
    ys = [y_obs[np.asarray(s_obs) == s] for s in range(N_SITES)]
    return times, ys


# --- dense reference -----------------------------------------------------


@pytest.mark.parametrize(
    ("site", "noise"),
    [
        (POOLED_SITE, NOISE),
        (PER_SITE, NOISE),
        (MIXED_SITE, NOISE),
        (PER_SITE, NOISE_PER_SITE),
    ],
    ids=["pooled", "per-site", "mixed", "per-site-noise"],
)
def test_log_marginal_matches_dense(site, noise):
    times, ys = _dataset(jr.key(0), site, noise)
    model, y_grid = SharedPlusSiteMarkovGP.from_sites(SHARED, site, times, ys)
    assert model.mask.sum() == sum(t.shape[0] for t in times)
    assert model.times.shape[0] < sum(t.shape[0] for t in times)  # overlapping days

    got = model.log_marginal(y_grid, noise)
    want = _dense_log_marginal(model, y_grid, noise)
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_lgssm_layout():
    model, _ = SharedPlusSiteMarkovGP.from_sites(
        SHARED, PER_SITE, *_dataset(jr.key(1), PER_SITE)
    )
    lgssm = model.as_lgssm(NOISE_PER_SITE)
    d_g, d_h, T = 2, 1, model.times.shape[0]
    D = d_g + N_SITES * d_h
    assert model.state_dim == D
    assert lgssm.transition.shape == (T, D, D)
    assert lgssm.process_noise.shape == (T, D, D)
    assert lgssm.obs_model.shape == (N_SITES, D)
    # the leading zero step: A_1 = I, Q_1 = 0
    np.testing.assert_allclose(lgssm.transition[0], jnp.eye(D), atol=1e-14)
    np.testing.assert_allclose(lgssm.process_noise[0], 0.0, atol=1e-14)
    # H[s] = (H_g, 0, …, H_h at block s, …, 0)
    want_H = jnp.array(
        [
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_array_equal(lgssm.obs_model, want_H)
    np.testing.assert_allclose(lgssm.obs_noise, jnp.diag(NOISE_PER_SITE))
    # P_0 = blockdiag(P∞_g, P∞_h,1, …): the site variances on the diagonal
    np.testing.assert_allclose(jnp.diag(lgssm.init_cov)[d_g:], PER_SITE.variance)
    np.testing.assert_array_equal(lgssm.init_cov[d_g:, :d_g], 0.0)
    np.testing.assert_allclose(lgssm.init_cov[d_g:, d_g:], jnp.diag(PER_SITE.variance))


def test_lgssm_factor_on_as_lgssm_matches_log_marginal():
    model, y_grid = SharedPlusSiteMarkovGP.from_sites(
        SHARED, POOLED_SITE, *_dataset(jr.key(2))
    )

    def numpyro_model():
        lgssm_factor("y", model.as_lgssm(NOISE), y_grid, mask=model.mask)

    trace = handlers.trace(handlers.seed(numpyro_model, 0)).get_trace()
    assert sorted(trace) == ["y"]
    np.testing.assert_allclose(
        trace["y"]["fn"].log_factor, model.log_marginal(y_grid, NOISE), rtol=1e-12
    )


# --- reduction to MarkovGPPrior -------------------------------------------


def test_one_site_with_zero_site_variance_is_markov_gp_on_g():
    """With S = 1 and k_h = 0, y = g + ε: the log marginal of `MarkovGPPrior`.

    The filter never inverts the predicted covariance, so a site kernel of
    exactly zero variance is fine for the log marginal. The RTS smoother
    solves against it, and a zero site block makes it singular, so
    ``decompose`` is checked with a variance of 1e-12 instead: that moves the
    posterior mean of g by O(1e-12 / σ²_n) = O(1e-11).
    """
    times = jnp.sort(_ragged_times(jr.key(3), n_sites=1)[0])
    y = jnp.sin(0.3 * times) + 0.1 * jr.normal(jr.key(4), times.shape)
    reference = MarkovGPPrior(SHARED, times)

    zero_site = MaternSDE(variance=jnp.asarray(0.0), lengthscale=1.0, order=0)
    model, y_grid = SharedPlusSiteMarkovGP.from_sites(SHARED, zero_site, [times], [y])
    np.testing.assert_allclose(
        model.log_marginal(y_grid, NOISE),
        reference.log_marginal(y, NOISE),
        rtol=1e-12,
    )

    tiny_site = MaternSDE(variance=jnp.asarray(1e-12), lengthscale=1.0, order=0)
    model, y_grid = SharedPlusSiteMarkovGP.from_sites(SHARED, tiny_site, [times], [y])
    g_mean, h_mean = model.decompose(y_grid, NOISE)
    m_ref, _, _ = reference.smooth(y, NOISE)
    np.testing.assert_allclose(g_mean, m_ref[:, 0], atol=1e-9)
    np.testing.assert_allclose(h_mean, 0.0, atol=1e-9)


# --- decompose -------------------------------------------------------------


@pytest.mark.parametrize(
    ("site", "noise"),
    [(POOLED_SITE, NOISE), (PER_SITE, NOISE_PER_SITE)],
    ids=["pooled", "per-site-noise"],
)
def test_decompose_matches_smoother_and_dense(site, noise):
    times, ys = _dataset(jr.key(5), site, noise)
    model, y_grid = SharedPlusSiteMarkovGP.from_sites(SHARED, site, times, ys)
    g_mean, h_mean = model.decompose(y_grid, noise)
    assert g_mean.shape == (model.times.shape[0],)
    assert h_mean.shape == (model.times.shape[0], N_SITES)

    # g + h_s is the smoothed mean of site s, H[s] E[x_k | y]
    lgssm = model.as_lgssm(noise)
    means, _, _ = lgssm.smooth(y_grid, mask=model.mask)
    site_means = means @ lgssm.obs_model.T
    np.testing.assert_allclose(g_mean[:, None] + h_mean, site_means, rtol=1e-12)

    # the dense conditional means E[g | y] = K_g(t, o) α, E[h_s | y] = K_h,s(t, o_s) α_s
    t_obs, s_obs, y_obs = _observed_pairs(model, y_grid)
    K = _dense_cov(SHARED, site, t_obs, s_obs, noise, N_SITES)
    alpha = jnp.linalg.solve(K, y_obs)
    grid, obs = model.times[:, None], t_obs[:, None]
    K_g = matern_kernel(grid, obs, SHARED.variance, SHARED.lengthscale, nu=SHARED_NU)
    np.testing.assert_allclose(g_mean, K_g @ alpha, rtol=1e-10, atol=1e-12)
    var_h, ell_h = _site_params(site, N_SITES)
    for s in range(N_SITES):
        K_h = matern_kernel(grid, obs, var_h[s], ell_h[s], nu=SITE_NU)
        want = K_h @ jnp.where(s_obs == s, alpha, 0.0)
        np.testing.assert_allclose(h_mean[:, s], want, rtol=1e-10, atol=1e-12)


# --- from_sites ----------------------------------------------------------


def test_from_sites_merges_grid_mask_and_nan():
    times = [jnp.array([0.0, 2.0, 5.0]), jnp.array([2.0, 3.0]), jnp.array([1.0, 5.0])]
    ys = [jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0]), jnp.array([6.0, 7.0])]
    model, y_grid = SharedPlusSiteMarkovGP.from_sites(SHARED, POOLED_SITE, times, ys)

    assert model.times.tolist() == [0.0, 1.0, 2.0, 3.0, 5.0]
    want_mask = [
        [True, False, False],
        [False, False, True],
        [True, True, False],
        [False, True, False],
        [True, False, True],
    ]
    assert model.mask.tolist() == want_mask
    np.testing.assert_array_equal(np.isnan(y_grid), ~np.asarray(want_mask))
    assert y_grid[np.asarray(want_mask)].tolist() == [1.0, 6.0, 2.0, 4.0, 5.0, 3.0, 7.0]


def test_nan_placeholders_keep_value_and_gradient_finite():
    times, ys = _dataset(jr.key(6))
    model, y_grid = SharedPlusSiteMarkovGP.from_sites(SHARED, POOLED_SITE, times, ys)
    assert jnp.isnan(y_grid).any()

    def loss(log_ell_g, log_var_h, log_noise):
        m = SharedPlusSiteMarkovGP(
            MaternSDE(1.0, jnp.exp(log_ell_g), order=1),
            MaternSDE(jnp.exp(log_var_h), 2.0, order=0),
            model.times,
            model.mask,
        )
        return m.log_marginal(y_grid, jnp.exp(log_noise))

    args = (jnp.log(8.0), jnp.log(jnp.array([0.3, 0.5, 0.2])), jnp.log(0.05))
    value, grads = jax.value_and_grad(loss, argnums=(0, 1, 2))(*args)
    assert jnp.isfinite(value)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)
    # any value at an unobserved entry gives the same result, bit for bit
    filled = jnp.where(model.mask, y_grid, 123.0)
    assert model.log_marginal(filled, NOISE) == model.log_marginal(y_grid, NOISE)


# --- validation ----------------------------------------------------------


def test_validation_errors():
    mask = jnp.ones((4, 3), dtype=bool)
    with pytest.raises(ValueError, match="strictly increasing"):
        SharedPlusSiteMarkovGP(
            SHARED, POOLED_SITE, jnp.array([0.0, 2.0, 1.0, 3.0]), mask
        )
    with pytest.raises(ValueError, match=r"mask must have shape \(T, S\)"):
        SharedPlusSiteMarkovGP(SHARED, POOLED_SITE, jnp.arange(5.0), mask)
    two_sites = MaternSDE(variance=jnp.array([0.3, 0.5]), lengthscale=1.0, order=0)
    with pytest.raises(ValueError, match="leading axis of length S = 3"):
        SharedPlusSiteMarkovGP(SHARED, two_sites, jnp.arange(4.0), mask)
    model = SharedPlusSiteMarkovGP(SHARED, POOLED_SITE, jnp.arange(4.0), mask)
    with pytest.raises(ValueError, match="noise_var must be a scalar or have shape"):
        model.as_lgssm(jnp.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="strictly increasing"):
        SharedPlusSiteMarkovGP.from_sites(
            SHARED, POOLED_SITE, [jnp.array([1.0, 0.0])], [jnp.zeros(2)]
        )
    with pytest.raises(ValueError, match="needs one time"):
        SharedPlusSiteMarkovGP.from_sites(
            SHARED, POOLED_SITE, [jnp.array([0.0, 1.0])], [jnp.zeros(3)]
        )


# --- jit, grad and NUTS --------------------------------------------------


def _loss_and_dense(model, y_grid):
    """The Kalman and the dense log marginal as functions of log hyperparameters."""

    def kernels(log_var_g, log_ell_g, log_var_h, log_ell_h):
        shared = MaternSDE(jnp.exp(log_var_g), jnp.exp(log_ell_g), order=1)
        site = MaternSDE(jnp.exp(log_var_h), jnp.exp(log_ell_h), order=0)
        return shared, site

    def kalman(*log_params):
        shared, site = kernels(*log_params)
        m = SharedPlusSiteMarkovGP(shared, site, model.times, model.mask)
        return m.log_marginal(y_grid, NOISE_PER_SITE)

    def dense(*log_params):
        shared, site = kernels(*log_params)
        m = SharedPlusSiteMarkovGP(shared, site, model.times, model.mask)
        return _dense_log_marginal(m, y_grid, NOISE_PER_SITE)

    return kalman, dense


def test_jit_grad_matches_dense_gradient():
    model, y_grid = SharedPlusSiteMarkovGP.from_sites(
        SHARED, PER_SITE, *_dataset(jr.key(7), PER_SITE, NOISE_PER_SITE)
    )
    kalman, dense = _loss_and_dense(model, y_grid)
    args = (
        jnp.log(1.0),
        jnp.log(8.0),
        jnp.log(PER_SITE.variance),
        jnp.log(PER_SITE.lengthscale),
    )
    argnums = (0, 1, 2, 3)
    got = jax.jit(jax.grad(kalman, argnums=argnums))(*args)
    want = jax.grad(dense, argnums=argnums)(*args)
    for g, w in zip(got, want, strict=True):
        assert jnp.all(jnp.isfinite(g))
        # both are exact gradients of the same function: x64 rounding only
        np.testing.assert_allclose(g, w, rtol=1e-10, atol=1e-12)


def _nuts_model(model, y_grid):
    log_ell_g = numpyro.sample("log_ell_g", dist.Normal(jnp.log(8.0), 0.5))
    with numpyro.plate("site", model.n_sites):
        log_var_h = numpyro.sample("log_var_h", dist.Normal(jnp.log(0.3), 0.5))
    m = SharedPlusSiteMarkovGP(
        MaternSDE(1.0, jnp.exp(log_ell_g), order=1),
        MaternSDE(jnp.exp(log_var_h), 2.0, order=0),
        model.times,
        model.mask,
    )
    numpyro.factor("y", m.log_marginal(y_grid, NOISE))


def _run_nuts(n_warmup: int, n_samples: int, key: int):
    model, y_grid = SharedPlusSiteMarkovGP.from_sites(
        SHARED, POOLED_SITE, *_dataset(jr.key(8))
    )
    mcmc = MCMC(
        NUTS(_nuts_model),
        num_warmup=n_warmup,
        num_samples=n_samples,
        progress_bar=False,
    )
    mcmc.run(jr.key(key), model, y_grid)
    return mcmc.get_samples()


def test_runs_inside_nuts():
    samples = _run_nuts(n_warmup=10, n_samples=10, key=0)
    assert samples["log_ell_g"].shape == (10,)
    assert samples["log_var_h"].shape == (10, N_SITES)
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in samples.values())


@pytest.mark.slow
def test_nuts_longer_chain_is_finite_and_moves():
    samples = _run_nuts(n_warmup=200, n_samples=200, key=1)
    for v in samples.values():
        assert jnp.all(jnp.isfinite(v))
        assert jnp.all(jnp.std(v, axis=0) > 0.0)


@pytest.mark.slow
def test_log_marginal_matches_dense_many_sites():
    """Eight per-site Matérn-1/2 sites over 120 days, against the dense Gaussian."""
    n_sites, n_days = 8, 120
    times = _ragged_times(jr.key(9), n_sites, n_days)
    ys = [
        jnp.cos(0.1 * t) + 0.3 * jr.normal(jr.key(10 + s), t.shape)
        for s, t in enumerate(times)
    ]
    site = MaternSDE(
        variance=jnp.linspace(0.1, 0.8, n_sites),
        lengthscale=jnp.linspace(1.0, 5.0, n_sites),
        order=0,
    )
    noise = jnp.linspace(0.02, 0.1, n_sites)
    model, y_grid = SharedPlusSiteMarkovGP.from_sites(SHARED, site, times, ys)
    np.testing.assert_allclose(
        model.log_marginal(y_grid, noise),
        _dense_log_marginal(model, y_grid, noise),
        rtol=1e-10,
    )
