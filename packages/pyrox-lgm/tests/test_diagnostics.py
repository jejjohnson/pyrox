"""Diagnostics (CPO, PIT, WAIC, DIC), predictor marginals and f(...).

CPO and PIT remove each site from its Gaussian predictor marginal at the
full-data hyperparameter posterior; at a fixed theta and a Gaussian
likelihood that is exact, which the first tests check against dense
leave-one-out conditionals. A
Poisson model is compared with genuine refits.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import kernellib as kl
import numpy as np
import pyrox_lgm as lgm
import pytest
from scipy.stats import norm, poisson


jax.config.update("jax_enable_x64", True)

N = 15


def _gaussian_rw1():
    t = np.arange(N)
    y = np.sin(t / 3.0) + 0.3 * np.random.default_rng(0).normal(size=N) + 0.5
    rw = lgm.RW1(N, name="t")
    model = lgm.LGM((rw,), lgm.FixedEffects(("intercept",)), lgm.Gaussian())
    return model, {"y": y, "t": t}


def _dense_joint_cov(model, theta):
    """Marginal covariance of y at theta: RW1 (sum-to-zero) + intercept + noise."""
    rw = model.components[0]
    R = np.asarray(rw.structure.as_matrix())
    tau, prec = float(theta["t.tau"]), float(theta["lik.prec"])
    cov_x = np.linalg.pinv(tau * float(rw.scale) * R)
    return cov_x + np.ones((N, N)) / model.fixed.prior_precision + np.eye(
        N
    ) / prec, cov_x


# --- fast ---------------------------------------------------------------------


def test_f_builds_and_binds_components():
    g = kl.grid_graph((3, 3))
    tau = lgm.PCPrecision(0.5, 0.01)
    comp = lgm.f("region", "bym2", graph=g, hyper={"tau": tau})
    assert isinstance(comp, lgm.BYM2) and comp.name == "region"
    assert comp.tau_prior is tau
    assert isinstance(lgm.f("week", "ar1", n=10), lgm.AR1)
    spde = lgm.f("loc", "spde", grid=(4, 4), alpha=2)
    assert isinstance(spde, lgm.SPDE) and spde.name == "loc"
    with pytest.raises(ValueError, match="unknown model"):
        lgm.f("x", "ar2", n=3)
    with pytest.raises(ValueError, match="needs n"):
        lgm.f("x", "rw1")
    with pytest.raises(ValueError, match="needs graph"):
        lgm.f("x", "besag")
    with pytest.raises(ValueError, match="no hyperparameter 'phi'"):
        lgm.f("x", "iid", n=3, hyper={"phi": tau})


@pytest.mark.parametrize(
    ("obs", "y", "extra"),
    [
        (lgm.Poisson(), [0.0, 3.0], {}),
        (lgm.Bernoulli(), [0.0, 1.0], {}),
        (lgm.Binomial(), [2.0, 5.0], {"n_trials": np.array([5.0, 5.0])}),
        (lgm.NegativeBinomial(), [0.0, 4.0], {}),
        (lgm.Gaussian(), [0.3, -1.0], {}),
    ],
    ids=lambda o: type(o).__name__ if isinstance(o, lgm.AbstractObservation) else "",
)
def test_site_log_prob_is_consistent_with_the_gaussx_likelihood(obs, y, extra):
    # sum_i site_log_prob == the gaussx likelihood's log_prob used by Laplace.
    eta = jnp.array([0.4, -0.7])
    theta = {
        k: transform(jnp.asarray(0.3)) for k, (_, transform) in obs.theta_spec().items()
    }
    y = jnp.asarray(y)
    sites = obs.site_log_prob(y, eta, theta, extra)
    total = obs.build(y, theta, extra).log_prob(eta)
    assert np.isclose(float(jnp.sum(sites)), float(total), atol=1e-10)


# --- slow: on fitted models ---------------------------------------------------------


@pytest.fixture(scope="module")
def gaussian_eb():
    model, data = _gaussian_rw1()
    return model, data, lgm.inla(model, data, strategy="gaussian", integration="eb")


@pytest.mark.slow
def test_predictor_variances_are_the_dense_ones(gaussian_eb):
    model, _data, res = gaussian_eb
    theta = model.unflatten(res.theta_points[0])
    _, cov_x = _dense_joint_cov(model, theta)
    # Posterior covariance of eta = x + beta given y: from the joint Gaussian.
    prec = float(theta["lik.prec"])
    prior = cov_x + np.ones((N, N)) / model.fixed.prior_precision
    post = prior - prior @ np.linalg.solve(prior + np.eye(N) / prec, prior)
    assert np.allclose(res.predictor_variances[0], np.diag(post), rtol=1e-6)
    assert res.linear_predictor.mean.shape == (N,)


@pytest.mark.slow
def test_cpo_and_pit_are_the_exact_leave_one_out_at_fixed_theta(gaussian_eb):
    # With one design point (eb) and a Gaussian likelihood, the harmonic
    # identity is exact: CPO_i = N(y_i; m_-i, v_-i) of the joint marginal.
    model, data, res = gaussian_eb
    theta = model.unflatten(res.theta_points[0])
    C, _ = _dense_joint_cov(model, theta)
    P = np.linalg.inv(C)
    y = np.asarray(data["y"])
    v = 1.0 / np.diag(P)
    m = y - v * (P @ y)  # E[y_i | y_-i] = y_i - (P y)_i / P_ii
    # CPO's integrand is constant here, so it is exact at any order (3e-11
    # measured); PIT's Gauss-Hermite error on the cavity is 4e-9 at 80 nodes.
    d = res.diagnostics()
    assert np.allclose(d.cpo, norm.pdf(y, m, np.sqrt(v)), rtol=1e-9)
    assert np.allclose(d.pit, norm.cdf(y, m, np.sqrt(v)), atol=1e-7)


@pytest.mark.slow
def test_waic_and_dic_are_finite_with_sensible_complexities(gaussian_eb):
    _, _, res = gaussian_eb
    d = res.diagnostics()
    for value in (d.waic, d.dic, d.log_score):
        assert np.isfinite(float(value))
    # Effective parameters: positive, below the latent dimension.
    assert 0.0 < float(d.p_d) < N + 1
    assert 0.0 < float(d.p_waic) < N + 1


@pytest.mark.slow
def test_poisson_cpo_agrees_with_refits():
    # Brute force: refit on y_-i and integrate p(y_i | eta_i) against the
    # refit's predictive marginal of eta_i. The harmonic identity keeps the
    # full-data hyperparameter posterior, so the two differ by that alone.
    n = 10
    rng = np.random.default_rng(4)
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(1.0 + 0.5 * x)).astype(float)
    # RW1, not IID: dropping y_i leaves an IID node uncoupled, which gaussx
    # < 0.6.2 cannot differentiate through (jejjohnson/gaussx#521).
    model = lgm.LGM(
        (lgm.RW1(n, name="u"),), lgm.FixedEffects(("intercept", "x")), lgm.Poisson()
    )
    data = {"y": y, "u": np.arange(n), "x": x}
    cpo = np.asarray(lgm.inla(model, data).diagnostics().cpo)

    # Each refit has its own pattern, so compiles afresh: three sites (both
    # ends, where RW1 is weakest, and the middle) keep the test bounded.
    sites = [0, n // 2, n - 1]
    loo = []
    for i in sites:
        keep = np.arange(n) != i
        sub = {"y": y[keep], "u": np.arange(n)[keep], "x": x[keep]}
        res = lgm.inla(model, sub)
        new = {"y": np.zeros(1), "u": np.array([i]), "x": x[i : i + 1]}
        a = np.asarray(model.projector(new).as_matrix())[0]
        # The refit's predictive density of y_i, by Monte Carlo over its
        # posterior (u_i is unobserved in the refit; its neighbours tie it down).
        eta = np.asarray(res.sample_latent(jax.random.key(i), 20_000)) @ a
        loo.append(np.mean(poisson.pmf(y[i], np.exp(eta))))
        jax.clear_caches()  # a refit's executables are never reused
    # Measured agreement is within 0.3 % (Monte Carlo error included); the
    # 5 % bound leaves room for the hyperparameters' full-data posterior.
    assert np.allclose(cpo[sites], np.asarray(loo), rtol=0.05)
