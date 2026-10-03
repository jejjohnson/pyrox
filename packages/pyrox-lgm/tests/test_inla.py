"""LGM and inla() against exact and brute-force references.

R-INLA golden fixtures need R, which this suite does not; the references are
instead exact where the model allows it (a Gaussian likelihood makes the
Laplace step exact), brute-force quadrature over theta, and NUTS on the same
model. Keys and data are pinned.

Every full ``inla()`` run compiles its model structure (~20 s cold on CPU),
so those tests are in the slow tier; they share one compiled structure per
model through a module-scoped fixture.
"""

from __future__ import annotations

import time
import warnings

import jax
import jax.numpy as jnp
import kernellib as kl
import numpy as np
import pyrox_lgm as lgm
import pytest
from pyrox_lgm._inla import _fit_point, _log_post
from pyrox_lgm._result import mixture_summary
from scipy.stats import multivariate_normal, norm


jax.config.update("jax_enable_x64", True)

N = 20


def _gaussian_rw2():
    t = np.arange(N)
    rng = np.random.default_rng(0)
    y = np.sin(t / 3.0) + 0.3 * rng.normal(size=N) + 0.5
    model = lgm.LGM(
        (lgm.RW2(N, name="trend"),), lgm.FixedEffects(("intercept",)), lgm.Gaussian()
    )
    return model, {"y": y, "trend": t}


def _dense_posterior(model, data, u):
    """Exact Gaussian posterior of the latent vector at u (RW2 + intercept).

    Prior precision blockdiag(tau s R, lambda); RW2 constrained to sum to zero
    only (R-INLA's rw2); Gaussian observations with precision prec.
    """
    theta = model.unflatten(jnp.asarray(u))
    tau, prec = float(theta["trend.tau"]), float(theta["lik.prec"])
    rw = model.components[0]
    R = np.asarray(rw.structure.as_matrix())
    n = R.shape[0]
    P0 = np.zeros((n + 1, n + 1))
    P0[:n, :n] = tau * float(rw.scale) * R
    P0[n, n] = model.fixed.prior_precision
    A = np.asarray(model.projector(data).as_matrix())
    H = P0 + prec * A.T @ A
    b = prec * A.T @ np.asarray(data["y"])
    C = np.zeros((n + 1, 1))
    C[:n, 0] = 1.0 / np.sqrt(n)
    Hi = np.linalg.inv(H)
    K = Hi @ C @ np.linalg.inv(C.T @ Hi @ C)
    Sigma = Hi - K @ C.T @ Hi
    return Sigma @ b, np.diag(Sigma)


# --- fast: model-level exactness ---------------------------------------------


def test_log_posterior_theta_is_the_exact_gaussian_marginal():
    t = np.arange(12)
    y = np.sin(t / 2.0) + 0.3 * np.random.default_rng(0).normal(size=12) + 1.0
    rw = lgm.RW1(12, name="t")
    model = lgm.LGM((rw,), lgm.FixedEffects(("intercept",)), lgm.Gaussian())
    data = {"y": y, "t": t}
    u = jnp.array([0.3, 1.1])
    tau, prec = np.exp(0.3), np.exp(1.1)
    R = np.asarray(rw.structure.as_matrix())
    cov = (
        np.linalg.pinv(tau * float(rw.scale) * R)
        + np.ones((12, 12)) / model.fixed.prior_precision
        + np.eye(12) / prec
    )
    ref = multivariate_normal(np.zeros(12), cov).logpdf(y) + float(model.log_prior(u))
    assert np.isclose(float(model.log_posterior_theta(u, data)), ref, atol=1e-9)
    grad = jax.grad(lambda v: model.log_posterior_theta(v, data))(u)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_rw2_constrains_only_the_sum_inside_an_lgm():
    model, _ = _gaussian_rw2()
    prior = model.latent_prior(model.unflatten(jnp.zeros(model.n_theta)))
    assert prior.null_space.shape == (N + 1, 1)  # one constraint, not two
    assert np.allclose(prior.null_space[:N, 0], 1.0 / np.sqrt(N))


def test_theta_bookkeeping():
    model = lgm.LGM(
        (lgm.AR1(5, name="t"), lgm.SPDE(grid=(3, 3), name="s")),
        likelihood=lgm.NegativeBinomial(),
    )
    assert list(model.theta_spec()) == ["t.tau", "t.rho", "s.range_sigma", "lik.size"]
    assert model.n_theta == 5  # range_sigma is a pair
    theta = model.unflatten(jnp.zeros(5))
    assert theta["s.range_sigma"].shape == (2,)
    assert -1.0 < float(theta["t.rho"]) < 1.0


def test_lgm_validation():
    with pytest.raises(ValueError, match="distinct"):
        lgm.LGM((lgm.IID(3, name="a"), lgm.IID(3, name="a")))
    model = lgm.LGM((lgm.IID(3, name="a"),), lgm.FixedEffects(("x",)))
    with pytest.raises(KeyError, match="'a'"):
        model.projector({"y": np.zeros(3)})
    with pytest.raises(KeyError, match="'x'"):
        model.projector({"y": np.zeros(3), "a": np.arange(3)})


def test_mixture_summary_matches_scipy():
    means = jnp.array([[0.0, 1.0], [2.0, -1.0]])
    variances = jnp.array([[1.0, 0.25], [4.0, 1.0]])
    w = jnp.array([0.3, 0.7])
    s = mixture_summary(means, variances, w)
    for j in range(2):

        def cdf(x, j=j):
            return sum(
                float(w[k])
                * norm.cdf(x, float(means[k, j]), np.sqrt(float(variances[k, j])))
                for k in range(2)
            )

        for q, p in zip((s.q025, s.q50, s.q975), (0.025, 0.5, 0.975), strict=True):
            assert np.isclose(cdf(float(q[j])), p, atol=1e-8)
    m = (w[:, None] * means).sum(0)
    v = (w[:, None] * (variances + means**2)).sum(0) - m**2
    assert np.allclose(s.mean, m) and np.allclose(s.sd, np.sqrt(v))


# --- slow: full inla() runs ---------------------------------------------------


@pytest.fixture(scope="module")
def gaussian_fit():
    model, data = _gaussian_rw2()
    return model, data, lgm.inla(model, data, strategy="gaussian")


@pytest.mark.slow
def test_per_point_marginals_are_exact_for_a_gaussian_likelihood(gaussian_fit):
    model, data, res = gaussian_fit
    for k in range(res.theta_points.shape[0]):
        mean, var = _dense_posterior(model, data, res.theta_points[k])
        assert np.allclose(res.latent_means[k], mean, atol=1e-7)
        assert np.allclose(res.latent_variances[k], var, rtol=1e-6)


@pytest.mark.slow
def test_vb_correction_is_zero_for_a_gaussian_likelihood(gaussian_fit):
    model, data, res = gaussian_fit
    A = model.projector(data)
    d = {k: jnp.asarray(v) for k, v in data.items()}
    u = res.theta_points[0]
    plain, _, _ = _fit_point(model, A, d, u, 50, ())
    vb, _, _ = _fit_point(model, A, d, u, 50, (N,))  # the intercept
    assert np.allclose(plain, vb, atol=1e-8)


@pytest.mark.slow
def test_inla_matches_brute_force_quadrature_over_theta(gaussian_fit):
    # Brute force: the exact log-posterior of theta on an 81 x 81 grid of
    # +-4 posterior sds, and the exact per-theta posterior mean.
    model, data, res = gaussian_fit
    A = model.projector(data)
    d = {k: jnp.asarray(v) for k, v in data.items()}
    u_star = np.asarray(res.theta_mode)
    neg_h = -np.asarray(
        jax.jacrev(jax.jacrev(lambda u: _log_post(model, A, d, 50, u)))(res.theta_mode)
    )
    sd = np.sqrt(np.diag(np.linalg.inv(neg_h)))
    g = np.linspace(-4.0, 4.0, 81)
    grid = u_star + np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2) * sd
    lp = np.array([float(_log_post(model, A, d, 50, jnp.asarray(u))) for u in grid])
    w = np.exp(lp - lp.max())
    means = np.array([_fit_point(model, A, d, jnp.asarray(u), 50, ())[0] for u in grid])
    bf_mean = (w / w.sum()) @ means
    bf_logml = lp.max() + np.log(w.sum() * (g[1] - g[0]) ** 2 * np.prod(sd))

    inla_mean = np.r_[res.random["trend"].mean, res.fixed["intercept"].mean]
    inla_sd = np.r_[res.random["trend"].sd, res.fixed["intercept"].sd]
    # The grid design leaves a small integration error: measured 0.4 % of a
    # posterior sd on the means and 0.02 on the log marginal likelihood.
    assert np.max(np.abs(inla_mean - bf_mean) / inla_sd) < 0.02
    assert abs(float(res.log_marginal_likelihood) - bf_logml) < 0.1


@pytest.mark.slow
def test_summaries_are_ordered_and_samples_respect_the_constraint(gaussian_fit):
    _, _, res = gaussian_fit
    for s in [*res.fixed.values(), *res.random.values(), *res.hyperpar.values()]:
        assert bool(jnp.all(s.q025 <= s.q50)) and bool(jnp.all(s.q50 <= s.q975))
        assert bool(jnp.all(s.sd >= 0))
    x = res.sample_latent(jax.random.key(0), 2000)
    assert x.shape == (2000, N + 1)
    assert np.allclose(x[:, :N].sum(axis=1), 0.0, atol=1e-8)  # sum-to-zero
    # Monte Carlo mean within 5 standard errors of the mixture mean.
    mean = np.r_[res.random["trend"].mean, res.fixed["intercept"].mean]
    sd = np.r_[res.random["trend"].sd, res.fixed["intercept"].sd]
    assert np.all(np.abs(x.mean(0) - mean) < 5 * sd / np.sqrt(2000) + 1e-12)


@pytest.mark.slow
def test_predict_and_to_xarray(gaussian_fit):
    _, _, res = gaussian_fit
    new = {"y": np.zeros(3), "trend": np.array([0, 10, 19])}
    pred = res.predict(new, jax.random.key(1), n_samples=500)
    assert pred.mean.shape == (3,)
    assert bool(jnp.all(pred.q025 < pred.q975))
    pytest.importorskip("xarray")
    ds = res.to_xarray()
    assert "fixed.intercept.mean" in ds and "random.trend.mean" in ds


@pytest.mark.slow
def test_non_converged_design_points_are_dropped_with_a_warning(
    gaussian_fit, monkeypatch
):
    import pyrox_lgm._inla as inla_mod

    model, data, _ = gaussian_fit
    real = inla_mod._fit_point
    calls = {"n": 0}

    def flaky(*args):
        mean, var, ok = real(*args)
        calls["n"] += 1
        return mean, var, ok & (calls["n"] > 2)  # first point fails twice

    monkeypatch.setattr(inla_mod, "_fit_point", flaky)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = lgm.inla(model, data, strategy="gaussian")
    assert res.n_dropped == 1
    assert any("dropped 1" in str(w.message) for w in caught)


@pytest.mark.slow
def test_pod_toy_recovers_the_truth_and_runs_under_5_seconds_warm():
    # Bernoulli probability-of-detection toy: an RW2 over 50 size bins plus
    # an intercept and a wind effect, 500 detections (gaussx#155's target).
    rng = np.random.default_rng(0)
    n_obs, n_bins = 500, 50
    size_bin = rng.integers(0, n_bins, n_obs)
    wind = rng.normal(size=n_obs)
    eta = -1.0 + 3.0 * np.sin(np.linspace(-1.5, 1.5, n_bins))[size_bin] - 0.5 * wind
    y = (rng.uniform(size=n_obs) < 1 / (1 + np.exp(-eta))).astype(float)
    model = lgm.LGM(
        (lgm.RW2(n_bins, name="size"),),
        lgm.FixedEffects(("intercept", "wind")),
        lgm.Bernoulli(),
    )
    data = {"y": y, "size": size_bin, "wind": wind}
    lgm.inla(model, data)  # compile
    start = time.perf_counter()
    res = lgm.inla(model, data)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, elapsed
    assert res.n_dropped == 0
    for name, truth in (("intercept", -1.0), ("wind", -0.5)):
        s = res.fixed[name]
        assert abs(float(s.mean) - truth) < 3 * float(s.sd), (name, s)


@pytest.mark.slow
def test_poisson_bym2_agrees_with_nuts():
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    g = kl.grid_graph((8, 8))
    n = 64
    bym2 = lgm.BYM2(g, name="region")
    truth = bym2.prior({"tau": jnp.asarray(2.0), "phi": jnp.asarray(0.7)})
    b = np.asarray(truth.sample(jax.random.key(1))[:n])
    rng = np.random.default_rng(2)
    x = rng.normal(size=n)
    E = rng.uniform(10.0, 30.0, size=n)
    y = rng.poisson(E * np.exp(-0.3 + 0.4 * x + b))
    model = lgm.LGM((bym2,), lgm.FixedEffects(("intercept", "x")), lgm.Poisson())
    data = {"y": y, "offset": np.log(E), "region": np.arange(n), "x": x}
    res = lgm.inla(model, data)

    def nuts_model():
        beta0 = numpyro.sample("intercept", dist.Normal(0.0, 1.0 / np.sqrt(1e-3)))
        beta1 = numpyro.sample("x", dist.Normal(0.0, 1.0 / np.sqrt(1e-3)))
        field = bym2.sample()
        numpyro.sample("y", dist.Poisson(E * jnp.exp(beta0 + beta1 * x + field)), obs=y)

    mcmc = MCMC(NUTS(nuts_model), num_warmup=1000, num_samples=2000, progress_bar=False)
    mcmc.run(jax.random.key(0))
    post = mcmc.get_samples()
    for name in ("intercept", "x"):
        s = res.fixed[name]
        draws = np.asarray(post[name])
        # INLA vs NUTS: means within 0.2 posterior sd, sds within 20 %
        # (Monte Carlo error on 2000 draws is ~0.05 sd).
        assert abs(float(s.mean) - draws.mean()) < 0.2 * draws.std(), name
        assert abs(float(s.sd) / draws.std() - 1.0) < 0.2, name
