"""`MarkovGP`: a gaussx SDE kernel on sorted times as a block-tridiagonal GMRF.

Dense references: the AR(1) precision for Matérn-1/2 on a grid, the closed-form
Matérn-3/2 Gram for the covariance of $f$, and polynomial paths for the null
space of an integrated Wiener process.
"""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import numpy as np
import pyrox_lgm as lgm
import pytest
from numpyro import handlers


jax.config.update("jax_enable_x64", True)


def _irregular_times(n: int, seed: int = 0) -> jnp.ndarray:
    # Spacings in [0.3, 1.0] keep the precision well conditioned (see the
    # MarkovGP docstring), so the dense references hold to 1e-10.
    steps = np.random.default_rng(seed).uniform(0.3, 1.0, n - 1)
    return jnp.asarray(np.concatenate([[0.0], np.cumsum(steps)]))


def _matern32_gram(t, variance, lengthscale):
    r = np.abs(np.asarray(t)[:, None] - np.asarray(t)[None, :])
    z = np.sqrt(3.0) * r / lengthscale
    return variance * (1.0 + z) * np.exp(-z)


def _u(model, theta):
    """The unconstrained vector of ``theta`` (by model key) in ``model``'s order."""
    spec = model.theta_spec()
    blocks = [
        jnp.ravel(spec[key][1].inv(jnp.asarray(theta[key])))
        for key, _, _ in model._sizes()
    ]
    return jnp.concatenate(blocks)


# --- the precision -----------------------------------------------------------


@pytest.mark.parametrize(("range_", "sigma"), [(0.5, 0.3), (3.0, 1.0), (40.0, 2.5)])
def test_matern12_markov_gp_is_ar1(range_, sigma):
    n, step = 25, 0.7
    mgp = lgm.MarkovGP(gx.MaternSDE(1.0, 1.0, order=0), step * jnp.arange(n))
    ours = mgp.prior({"range_sigma": jnp.array([range_, sigma])}).precision
    # practical range 2 ell, so rho = exp(-step / ell) = exp(-2 step / range)
    ar1 = lgm.AR1(n).prior(
        {"tau": jnp.asarray(sigma**-2), "rho": jnp.exp(-2.0 * step / range_)}
    )
    expected = np.asarray(ar1.precision.as_matrix())
    # Round-off only: measured at most 3.1e-15 of the largest entry.
    np.testing.assert_allclose(
        ours.as_matrix(), expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max()
    )


def test_matern32_covariance_of_f_is_the_gram():
    t = _irregular_times(30)
    mgp = lgm.MarkovGP(gx.MaternSDE(1.0, 1.0, order=1), t)
    range_, sigma = 4.0, 1.5
    gmrf = mgp.prior({"range_sigma": jnp.array([range_, sigma])})
    assert isinstance(gmrf.precision, gx.BlockTriDiag)
    cov = np.linalg.inv(np.asarray(gmrf.precision.as_matrix()))[::2, ::2]
    gram = _matern32_gram(t, sigma**2, range_ / 2.0)
    # Measured 1.9e-14 of sigma^2 for the covariance and 5.8e-14 relative for
    # the variances; 1e-10 leaves room for the dense inverse's conditioning.
    np.testing.assert_allclose(cov, gram, rtol=1e-10, atol=1e-10 * sigma**2)
    np.testing.assert_allclose(gmrf.marginal_variances()[::2], sigma**2, rtol=1e-10)


def test_integrated_wiener_null_space_is_the_polynomials():
    t = _irregular_times(20, seed=1)
    order = 2
    mgp = lgm.MarkovGP(gx.IntegratedWienerSDE(1.0, order=order), t)
    d = order + 1
    S = np.asarray(mgp.structure.as_matrix())
    V = np.asarray(mgp.null_space)
    assert V.shape == (20 * d, d)
    # Measured: |V^T V - I| 7.8e-16, |S V| 6.6e-17 of the largest entry, the
    # d null eigenvalues below 2e-17 of the largest and the next at 1e-8.
    np.testing.assert_allclose(V.T @ V, np.eye(d), atol=1e-12)
    assert np.max(np.abs(S @ V)) < 1e-12 * np.abs(S).max()
    ev = np.linalg.eigvalsh(S)
    assert np.sum(ev < 1e-12 * ev.max()) == d  # rank N d - d
    # The f-part of the null space spans 1, t, t^2 exactly.
    poly = np.vander(np.asarray(t), d, increasing=True)
    coef = np.linalg.lstsq(poly, V[::d], rcond=None)[0]
    np.testing.assert_allclose(poly @ coef, V[::d], atol=1e-10)  # measured 3e-15
    # inla() constrains only the level: sum_t f(t) = 0.
    level = np.asarray(mgp.constraint_basis(None))[:, 0]
    np.testing.assert_allclose(level[::d], level[0])
    np.testing.assert_allclose(np.delete(level, np.s_[::d]), 0.0, atol=1e-12)


def test_integrated_wiener_precision_scales_with_tau():
    t = _irregular_times(10)
    mgp = lgm.MarkovGP(gx.IntegratedWienerSDE(1.0, order=1), t)
    gmrf = mgp.prior({"tau": jnp.asarray(4.0)})
    assert isinstance(gmrf, gx.IntrinsicGMRF)
    # Q_k is linear in the diffusion, so the precision is tau times the
    # structure at diffusion 1. Measured exact (0.0).
    np.testing.assert_allclose(
        mgp.assembly_precision({"tau": 4.0}, gmrf).as_matrix(),
        4.0 * mgp.structure.as_matrix(),
        rtol=1e-12,
    )


# --- projector and field ---------------------------------------------------------


def test_projector_reads_f_at_each_time():
    t = _irregular_times(12)
    mgp = lgm.MarkovGP(gx.MaternSDE(1.0, 1.0, order=2), t, name="trend")
    x = jax.random.normal(jax.random.key(0), (mgp.n_nodes,))
    index = np.array([0, 3, 3, 11, 7])
    f = np.asarray(x).reshape(12, 3)[:, 0]
    np.testing.assert_allclose(mgp.projector(index).mv(x), f[index])
    np.testing.assert_allclose(mgp.field_at(x, index), f[index])
    np.testing.assert_allclose(mgp.field_at(x), f)
    with pytest.raises(ValueError, match="out of range"):
        mgp.projector(np.array([12]))


def test_replicate_and_kronecker_read_the_inner_field():
    t = _irregular_times(6)
    mgp = lgm.MarkovGP(gx.MaternSDE(1.0, 1.0, order=1), t, name="trend")
    rep = lgm.Replicate(mgp, 3)
    assert (rep.n_nodes, rep.n_index) == (36, 18)
    x = jax.random.normal(jax.random.key(1), (rep.n_nodes,))
    f = np.asarray(x).reshape(3, 6, 2)[:, :, 0].reshape(-1)  # copy r, time i
    index = np.array([0, 5, 6, 17, 9])
    np.testing.assert_allclose(rep.projector(index).mv(x), f[index])
    np.testing.assert_allclose(rep.field_at(x), f)

    kron = lgm.Kronecker(mgp, lgm.AR1(4, name="g"))
    assert (kron.n_nodes, kron.n_index) == (48, 24)
    x = jax.random.normal(jax.random.key(2), (kron.n_nodes,))
    f = np.asarray(x).reshape(6, 2, 4)[:, 0, :].reshape(-1)  # time i, group j
    index = np.array([0, 3, 4, 23, 13])
    np.testing.assert_allclose(kron.projector(index).mv(x), f[index])
    np.testing.assert_allclose(kron.field_at(x), f)
    gmrf = kron.prior(
        {"trend_range_sigma": jnp.array([2.0, 1.0]), "g_rho": jnp.asarray(0.5)}
    )
    assert gmrf.precision.in_size() == 48


def test_kronecker_with_an_odd_rw2_reads_past_its_padding_node():
    rw2 = lgm.RW2(7, name="t")
    kron = lgm.Kronecker(rw2, lgm.AR1(3, name="g"))
    assert (kron.n_nodes, kron.n_index) == (24, 21)
    x = jax.random.normal(jax.random.key(3), (kron.n_nodes,))
    f = np.asarray(x).reshape(8, 3)[:7].reshape(-1)  # node 7 is the padding
    np.testing.assert_allclose(kron.projector(np.arange(21)).mv(x), f)
    np.testing.assert_allclose(kron.field_at(x), f)


# --- construction ------------------------------------------------------------


def test_markov_gp_validates_its_inputs():
    t = jnp.arange(5.0)
    with pytest.raises(TypeError, match=r"MaternSDE or gaussx\.IntegratedWienerSDE"):
        lgm.MarkovGP(gx.PeriodicSDE(1.0, 1.0, 2.0), t)
    with pytest.raises(ValueError, match="strictly increasing"):
        lgm.MarkovGP(gx.MaternSDE(1.0, 1.0, order=1), jnp.array([0.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="at least 2"):
        lgm.MarkovGP(gx.MaternSDE(1.0, 1.0, order=1), jnp.array([0.0]))
    with pytest.raises(ValueError, match="pass prior="):
        lgm.MarkovGP(
            gx.MaternSDE(1.0, 1.0, order=1), t, tau_prior=lgm.PCPrecision(1.0, 0.01)
        )
    with pytest.raises(ValueError, match="pass tau_prior="):
        lgm.MarkovGP(gx.IntegratedWienerSDE(1.0), t, prior=lgm.PCMatern(d=1))


def test_default_priors_and_formula():
    t = jnp.linspace(0.0, 50.0, 11)
    mgp = lgm.MarkovGP(gx.MaternSDE(1.0, 1.0, order=1), t)
    (key,) = mgp.theta_spec()
    prior = mgp.theta_spec()[key][0]
    assert key == "range_sigma" and isinstance(prior, lgm.PCMatern)
    assert (float(prior.range0), prior.d) == (10.0, 1)  # span / 5, in 1-D
    assert list(lgm.MarkovGP(gx.IntegratedWienerSDE(1.0), t).theta_spec()) == ["tau"]
    pc = lgm.PCMatern(5.0, 0.1, 2.0, 0.05, d=1)
    comp = lgm.f(
        "day",
        "markov",
        sde=gx.MaternSDE(1.0, 1.0, order=1),
        times=t,
        hyper={"range_sigma": pc},
    )
    assert (comp.name, comp.prior_range_sigma) == ("day", pc)


def test_numpyro_face_samples_theta_and_returns_f():
    t = _irregular_times(8)
    for sde, keys in (
        (gx.MaternSDE(1.0, 1.0, order=1), ["trend_range_sigma", "trend"]),
        (gx.IntegratedWienerSDE(1.0, order=1), ["trend_tau", "trend"]),
    ):
        mgp = lgm.MarkovGP(sde, t, name="trend")
        tr = handlers.trace(handlers.seed(mgp.sample, 0)).get_trace()
        assert list(tr) == keys
        x = tr["trend"]["value"]
        assert x.shape == (16,)
        f = handlers.seed(mgp.sample, 0)(np.array([1, 4]))
        np.testing.assert_allclose(f, np.asarray(x)[[2, 8]])


# --- inside an LGM ---------------------------------------------------------------


@pytest.mark.slow  # about 15 s, most of it compiling two Laplace fits
def test_matern12_evidence_equals_ar1_at_matched_theta():
    # The Laplace evidence log p~(y | theta) that inla() integrates, before the
    # hyperprior: the two models give the same latent Gaussian at matched
    # theta, so it agrees to round-off.
    rng = np.random.default_rng(4)
    n = 40
    y = np.sin(np.arange(n) / 5.0) + 0.3 * rng.normal(size=n)
    data = {"y": y, "t": np.arange(n)}
    fixed, lik = lgm.FixedEffects(("intercept",)), lgm.Gaussian()
    ar1 = lgm.LGM((lgm.AR1(n, name="t"),), fixed, lik)
    mgp = lgm.LGM(
        (lgm.MarkovGP(gx.MaternSDE(1.0, 1.0, order=0), jnp.arange(n), name="t"),),
        fixed,
        lik,
    )
    for tau, rho, prec in ((0.8, 0.55, 70.0), (3.0, 0.9, 10.0)):
        u_ar1 = _u(ar1, {"t.tau": tau, "t.rho": rho, "lik.prec": prec})
        range_sigma = jnp.array([-2.0 / np.log(rho), tau**-0.5])
        u_mgp = _u(mgp, {"t.range_sigma": range_sigma, "lik.prec": prec})
        lp_ar1 = float(ar1.laplace(u_ar1, data).log_marginal)
        lp_mgp = float(mgp.laplace(u_mgp, data).log_marginal)
        assert abs(lp_ar1 - lp_mgp) < 1e-8 * abs(lp_ar1), (lp_ar1, lp_mgp)


@pytest.mark.slow
def test_replicated_markov_gp_fits_in_inla():
    t = _irregular_times(40, seed=5)
    mgp = lgm.MarkovGP(gx.MaternSDE(1.0, 1.0, order=1), t, name="trend")
    truth = mgp.prior({"range_sigma": jnp.array([6.0, 1.0])})
    keys = jax.random.split(jax.random.key(6), 3)
    f = np.stack([np.asarray(mgp.field_at(truth.sample(k))) for k in keys])
    rng = np.random.default_rng(7)
    noise_sd = 0.2
    y = (f + noise_sd * rng.normal(size=f.shape)).reshape(-1)
    model = lgm.LGM(
        (lgm.Replicate(mgp, 3),), lgm.FixedEffects(("intercept",)), lgm.Gaussian()
    )
    data = {"y": y, "trend": np.arange(120)}

    # With a Gaussian likelihood the Laplace evidence is exact: it must equal
    # the dense density y ~ N(0, I_3 (x) K + I / prec + 11^T / 1e-3), the last
    # term the intercept's default prior. Measured within 4e-10.
    from scipy.stats import multivariate_normal

    for range_sigma, prec in (([6.0, 1.0], 25.0), ([2.0, 0.5], 39.0)):
        u = _u(model, {"trend.range_sigma": jnp.array(range_sigma), "lik.prec": prec})
        gram = _matern32_gram(t, range_sigma[1] ** 2, range_sigma[0] / 2.0)
        cov = np.kron(np.eye(3), gram) + np.eye(120) / prec + 1e3
        expected = multivariate_normal(np.zeros(120), cov).logpdf(y)
        ours = float(model.laplace(u, data).log_marginal)
        assert abs(ours - expected) < 1e-8, (ours, expected)

    res = lgm.inla(model, data)
    assert res.n_dropped == 0
    mean_f = np.asarray(model.components[0].field_at(res.random["trend"].mean))
    rmse = np.sqrt(
        np.mean((mean_f + float(res.fixed["intercept"].mean) - f.ravel()) ** 2)
    )
    assert rmse < noise_sd, rmse  # the smoother beats the raw readings


@pytest.mark.slow
def test_poisson_markov_gp_agrees_with_nuts():
    import numpyro
    import numpyro.distributions as dist
    from numpyro.diagnostics import effective_sample_size
    from numpyro.infer import MCMC, NUTS

    t = _irregular_times(50, seed=8)
    mgp = lgm.MarkovGP(gx.MaternSDE(1.0, 1.0, order=1), t, name="trend")
    truth = mgp.prior({"range_sigma": jnp.array([8.0, 0.7])})
    f = np.asarray(mgp.field_at(truth.sample(jax.random.key(9))))
    y = np.random.default_rng(10).poisson(np.exp(1.0 + f))
    index = np.arange(50)
    model = lgm.LGM((mgp,), lgm.FixedEffects(("intercept",)), lgm.Poisson())
    res = lgm.inla(model, {"y": y, "trend": index})

    def nuts_model():
        beta0 = numpyro.sample("intercept", dist.Normal(0.0, 1.0 / np.sqrt(1e-3)))
        field = mgp.sample(index)
        numpyro.sample("y", dist.Poisson(jnp.exp(beta0 + field)), obs=y)

    # Measured: no divergences; intercept means 0.072 apart (bound 0.356),
    # sd ratio 0.923; f at most 0.19 posterior sd apart, where the
    # least-mixed node's Monte Carlo error is 0.14 sd (ESS 52).
    mcmc = MCMC(NUTS(nuts_model), num_warmup=500, num_samples=1000, progress_bar=False)
    mcmc.run(jax.random.key(0))
    post = mcmc.get_samples(group_by_chain=True)
    chain = np.asarray(post["intercept"])
    draws = chain.reshape(-1)
    mcse = draws.std() / np.sqrt(float(effective_sample_size(chain)))
    s = res.fixed["intercept"]
    # As test_inla.py's BYM2 check: INLA's error (0.15 sd) plus 4 Monte
    # Carlo standard errors of the NUTS mean.
    assert abs(float(s.mean) - draws.mean()) < 0.15 * draws.std() + 4 * mcse
    assert abs(float(s.sd) / draws.std() - 1.0) < 0.2
    states = np.asarray(post["trend"]).reshape(chain.shape[0], -1, 50, 2)
    f_chain = states[..., 0]
    f_draws = f_chain.reshape(-1, 50)
    f_mcse = f_draws.std(axis=0) / np.sqrt(np.asarray(effective_sample_size(f_chain)))
    f_inla = np.asarray(mgp.field_at(res.random["trend"].mean))
    gap = np.abs(f_inla - f_draws.mean(axis=0))
    assert np.all(gap < 0.15 * f_draws.std(axis=0) + 4 * f_mcse)


def test_float32_times_stay_float32():
    t = jnp.arange(6, dtype=jnp.float32)
    mgp = lgm.MarkovGP(gx.MaternSDE(1.0, 1.0, order=1), t)
    assert mgp.times.dtype == mgp.observation_row.dtype == jnp.float32
    gmrf = mgp.prior({"range_sigma": jnp.array([2.0, 1.0], dtype=jnp.float32)})
    assert gmrf.loc.dtype == gmrf.precision.diagonal.dtype == jnp.float32
    spline = lgm.MarkovGP(gx.IntegratedWienerSDE(1.0, order=1), t)
    assert spline.structure.diagonal.dtype == spline.null_space.dtype == jnp.float32
    # Integer times take the default float dtype.
    assert lgm.MarkovGP(gx.MaternSDE(1.0, 1.0, order=0), jnp.arange(6)).times.dtype == (
        jnp.float64
    )
