"""Simplified Laplace (``strategy="sla"``) against a full-Laplace reference.

Full Laplace is a test-only reference (dense, one inner optimisation per
grid value of each latent node): pi_LA(x_i = t) is the joint at the
conditional mode of x_-i, divided by its Gaussian normaliser |H_-i|^(1/2).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pyrox_lgm as lgm
import pytest
from pyrox_lgm._result import mixture_summary, skew_normal_cdf, skew_normal_params
from scipy.special import gammaln
from scipy.stats import skewnorm


jax.config.update("jax_enable_x64", True)


def test_skew_normal_moments_and_cdf_match_scipy():
    for skew in (-0.9, -0.3, 0.0, 0.5, 0.95):
        xi, omega, alpha = skew_normal_params(
            jnp.array(1.0), jnp.array(4.0), jnp.array(skew)
        )
        ref = skewnorm(float(alpha), float(xi), float(omega))
        m, v, s = ref.stats("mvs")
        assert np.allclose([m, v, s], [1.0, 4.0, skew], atol=1e-10)
        x = jnp.linspace(-5.0, 7.0, 9)
        cdf = skew_normal_cdf(x, xi, omega, alpha)
        assert np.allclose(cdf, ref.cdf(np.asarray(x)), atol=1e-12)


def test_skewed_mixture_quantiles_match_scipy():
    s = mixture_summary(
        jnp.array([[0.5]]), jnp.array([[2.0]]), jnp.ones(1), jnp.array([[0.6]])
    )
    xi, omega, alpha = skew_normal_params(0.5, 2.0, 0.6)
    ref = skewnorm(float(alpha), float(xi), float(omega))
    assert np.allclose([s.q025[0], s.q50[0], s.q975[0]], ref.ppf([0.025, 0.5, 0.975]))
    assert np.isclose(float(s.mean[0]), 0.5) and np.isclose(float(s.sd[0]), 2**0.5)


# --- slow: against full Laplace ------------------------------------------------


def _full_laplace(Q, A, y, i, grid):
    """log pi_LA(x_i = t) for a Poisson LGM (dense; test reference only)."""
    rest = np.delete(np.arange(Q.shape[0]), i)
    x = np.zeros(Q.shape[0])
    out = []
    for t in grid:
        x[i] = t
        for _ in range(100):
            mu = np.exp(A @ x)
            g = -Q @ x + A.T @ (y - mu)
            H = Q + A.T @ (mu[:, None] * A)
            step = np.linalg.solve(H[np.ix_(rest, rest)], g[rest])
            x[rest] += step
            if np.max(np.abs(step)) < 1e-12:
                break
        eta = A @ x
        H = Q + A.T @ (np.exp(eta)[:, None] * A)
        joint = -0.5 * x @ Q @ x + np.sum(y * eta - np.exp(eta) - gammaln(y + 1))
        out.append(joint - 0.5 * np.linalg.slogdet(H[np.ix_(rest, rest)])[1])
    return np.asarray(out)


def _grid_summary(grid, logd):
    p = np.exp(logd - logd.max())
    p /= np.trapezoid(p, grid)
    mean = np.trapezoid(grid * p, grid)
    var = np.trapezoid((grid - mean) ** 2 * p, grid)
    skew = np.trapezoid((grid - mean) ** 3 * p, grid) / var**1.5
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(grid))])
    q = np.interp([0.025, 0.975], cdf, grid)
    return mean, np.sqrt(var), skew, q


@pytest.fixture(scope="module")
def poisson_fits():
    # Low counts and tau pinned near 1, so the marginals are visibly skewed.
    n = 8
    rng = np.random.default_rng(1)
    y = rng.poisson(np.exp(-0.5 + 0.8 * rng.normal(size=n))).astype(float)
    iid = lgm.IID(n, name="u", tau_prior=dist.Gamma(400.0, 400.0))
    model = lgm.LGM((iid,), lgm.FixedEffects(("intercept",)), lgm.Poisson())
    data = {"y": y, "u": np.arange(n)}
    fits = {
        s: lgm.inla(model, data, strategy=s, integration="eb")
        for s in ("gaussian", "sla")
    }
    return model, data, fits


@pytest.mark.slow
def test_sla_moves_marginals_to_the_full_laplace_reference(poisson_fits):
    # At the hyperparameter mode (eb), per latent node, against full Laplace:
    # SLA's mean within 0.03 sd and its skewness within 0.05 (measured: 0.012
    # sd, 0.031), always nearer than the Gaussian's zero. SLA keeps the
    # Gaussian variance (up to 5 % narrow here), so its 95 % interval ends
    # are bounded at 0.15 sd (measured 0.107) and, worst case over the
    # nodes, at a quarter of the Gaussian approximation's (measured 0.856).
    model, data, fits = poisson_fits
    gauss, sla = fits["gaussian"], fits["sla"]
    theta = model.unflatten(gauss.theta_points[0])
    Q = np.asarray(model.latent_prior(theta).precision.as_matrix())
    A = np.asarray(model.projector(data).as_matrix())
    y = np.asarray(data["y"])
    sla_q = mixture_summary(
        sla.latent_means, sla.latent_variances, sla.theta_weights, sla.latent_skewness
    )
    gauss_q = mixture_summary(
        gauss.latent_means, gauss.latent_variances, gauss.theta_weights
    )
    worst_gauss = worst_sla = 0.0
    for i in range(Q.shape[0]):
        mg = float(gauss.latent_means[0, i])
        sd = float(gauss.latent_variances[0, i]) ** 0.5
        grid = np.linspace(mg - 8 * sd, mg + 8 * sd, 801)
        m, s, sk, q = _grid_summary(grid, _full_laplace(Q, A, y, i, grid))
        sla_skew = float(sla.latent_skewness[0, i])
        assert abs(float(sla.latent_means[0, i]) - m) < 0.03 * s
        assert abs(sla_skew - sk) < min(0.05, abs(sk))
        sla_err = np.abs(np.array([sla_q.q025[i], sla_q.q975[i]]) - q) / s
        gauss_err = np.abs(np.array([gauss_q.q025[i], gauss_q.q975[i]]) - q) / s
        assert np.all(sla_err < 0.15)
        worst_sla = max(worst_sla, *sla_err)
        worst_gauss = max(worst_gauss, *gauss_err)
    assert worst_sla < 0.25 * worst_gauss


@pytest.mark.slow
def test_sla_leaves_a_gaussian_likelihood_unchanged():
    # log p(y | eta) is quadratic: no third derivative, no correction.
    t = np.arange(12)
    y = np.sin(t / 2.0) + 0.3
    model = lgm.LGM(
        (lgm.RW1(12, name="t"),), lgm.FixedEffects(("intercept",)), lgm.Gaussian()
    )
    data = {"y": y, "t": t}
    gauss = lgm.inla(model, data, strategy="gaussian")
    sla = lgm.inla(model, data, strategy="sla")
    assert np.allclose(sla.latent_means, gauss.latent_means, atol=1e-10)
    assert np.allclose(sla.latent_skewness, 0.0)
    assert np.allclose(sla.random["t"].q975, gauss.random["t"].q975, atol=1e-8)
