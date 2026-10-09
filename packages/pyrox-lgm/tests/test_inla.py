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

import gaussx as gx
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
    assert np.allclose(np.abs(prior.null_space[:N, 0]), 1.0 / np.sqrt(N))


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
    with pytest.raises(ValueError, match="distinct"):  # fixed vs component
        lgm.LGM((lgm.IID(3, name="a"),), lgm.FixedEffects(("a",)))
    with pytest.raises(ValueError, match="distinct"):  # duplicate fixed effects
        lgm.LGM((lgm.IID(3, name="b"),), lgm.FixedEffects(("x", "x")))
    with pytest.raises(ValueError, match="observation data"):  # the response
        lgm.LGM((lgm.IID(3, name="y"),))
    with pytest.raises(ValueError, match="observation data"):
        lgm.LGM((lgm.IID(3, name="b"),), lgm.FixedEffects(("offset",)))
    with pytest.raises(ValueError, match="observation model's name"):
        lgm.LGM((lgm.IID(3, name="lik"),))


def test_batched_hyperparameter_priors_count_every_entry():
    import equinox as eqx
    import gaussx as gx
    import numpyro.distributions as dist

    class Scaled(lgm.AbstractObservation):
        name: str = eqx.field(static=True, default="lik")

        def theta_spec(self):
            prior = dist.Exponential(jnp.ones(3))  # batch (3,), event ()
            return {"s": (prior, dist.biject_to(prior.support))}

        def build(self, y, theta, data):
            return gx.PoissonLikelihood(y)

    model = lgm.LGM((lgm.IID(3, name="a"),), likelihood=Scaled())
    assert model.n_theta == 4
    assert model.unflatten(jnp.zeros(4))["lik.s"].shape == (3,)
    assert jnp.shape(model.log_prior(jnp.zeros(4))) == ()


def test_a_theta_dependent_intrinsic_structure_is_rejected():
    import gaussx as gx
    import lineax as lx
    import numpyro.distributions as dist

    class Moving(lgm.Generic):
        # rho rescales the structure itself, not precision_scale, so the
        # pseudo-determinant (the omitted normaliser) changes with theta.
        def theta_spec(self):
            prior = dist.LogNormal(0.0, 1.0)
            return super().theta_spec() | {
                "rho": (prior, dist.biject_to(prior.support))
            }

        def prior(self, theta, *, constraint="hard", soft_constraint_scale=1e-3):
            R = self.structure.as_matrix() * theta["rho"]
            return gx.IntrinsicGMRF(
                jnp.zeros(self.n_nodes),
                theta["tau"],
                lx.MatrixLinearOperator(R, lx.positive_semidefinite_tag),
                self.null_space,
                constraint=constraint,
                soft_constraint_scale=soft_constraint_scale,
            )

    R = jnp.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    op = lx.MatrixLinearOperator(R, lx.positive_semidefinite_tag)
    with pytest.raises(ValueError, match="include_normalizer=True"):
        lgm.LGM((Moving(op, jnp.ones(3), name="m"),))
    lgm.LGM((lgm.Generic(op, jnp.ones(3), name="g"),))  # theta-free: fine


def test_covariates_can_be_traced_through_the_log_posterior():
    # The MCMC-INLA hybrid: a covariate depending on a parameter phi outside
    # the LGM; d/dphi log pi(y | theta, phi) must flow through the projector.
    rng = np.random.default_rng(3)
    x, y = rng.uniform(size=10), rng.normal(size=10)
    model = lgm.LGM(
        (lgm.IID(5, name="g"),), lgm.FixedEffects(("intercept", "z")), lgm.Gaussian()
    )
    u = jnp.zeros(model.n_theta)

    def lp(phi):
        data = {"y": y, "g": np.arange(10) % 5, "z": jnp.exp(-jnp.asarray(x) / phi)}
        return model.log_posterior_theta(u, data)

    h = 1e-5
    fd = (lp(0.7 + h) - lp(0.7 - h)) / (2 * h)
    assert np.isclose(float(jax.grad(lp)(0.7)), float(fd), rtol=1e-6)


def test_theta_init_must_match_the_hyperparameters():
    model, data = _gaussian_rw2()
    with pytest.raises(ValueError, match=r"theta_init must have shape \(2,\)"):
        lgm.inla(model, data, theta_init=np.zeros(3))


def test_any_generic_structure_assembles():
    # Tridiagonal has a sparse form; any other operator is assembled densely.
    import lineax as lx
    from pyrox_lgm._assembly import full_coo

    d, e = jnp.array([2.0, 3.0, 4.0]), jnp.array([-1.0, -0.5])
    tri = lx.TridiagonalLinearOperator(d, e, e)
    other = lx.AddLinearOperator(tri, lx.IdentityLinearOperator(tri.in_structure()))
    kron = gx.Kronecker(tri, tri, tri)  # more than two factors
    for op in (tri, other, kron):
        r, c, v = full_coo(op)
        n = op.in_size()
        dense = np.zeros((n, n))
        np.add.at(dense, (r, c), np.asarray(v))
        assert np.allclose(dense, op.as_matrix())
        model = lgm.LGM((lgm.Generic(op, name="g"),))
        prior = model.latent_prior(model.unflatten(jnp.zeros(model.n_theta)))
        assert np.allclose(prior.precision.as_matrix()[:n, :n], op.as_matrix())


def test_combinators_lift_the_inner_constraint_basis():
    # A wrapped RW2 keeps R-INLA's sum-only rule: one constraint per copy
    # (Replicate) or per group node (Kronecker), not two.
    rep = lgm.LGM((lgm.Replicate(lgm.RW2(6, name="t"), 3),))
    prior = rep.latent_prior(rep.unflatten(jnp.zeros(rep.n_theta)))
    assert prior.null_space.shape == (18, 3)
    kron = lgm.LGM((lgm.Kronecker(lgm.RW2(6, name="t"), lgm.AR1(4, name="g")),))
    prior = kron.latent_prior(kron.unflatten(jnp.zeros(kron.n_theta)))
    assert prior.null_space.shape == (24, 4)


def test_mesh_points_with_integer_coordinates_are_points():
    xs = np.linspace(0.0, 1.0, 3)
    V = np.stack(np.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    T = np.array(
        [
            [0, 3, 4],
            [0, 4, 1],
            [1, 4, 5],
            [1, 5, 2],
            [3, 6, 7],
            [3, 7, 4],
            [4, 7, 8],
            [4, 8, 5],
        ]
    )
    model = lgm.LGM((lgm.SPDE(mesh=(V, T), name="s"),))
    A = model.projector({"y": np.zeros(2), "s": np.array([[0, 0], [1, 1]])})
    dense = np.asarray(A.as_matrix())
    assert np.allclose(dense[0, 0], 1.0) and np.allclose(dense[1, 8], 1.0)


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
    plain, *_ = _fit_point(model, A, d, u, 50, ())
    vb, *_ = _fit_point(model, A, d, u, 50, (N,))  # the intercept
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
    # The grid design leaves a small integration error: measured 2.3 % of a
    # posterior sd on the means (the RW2's free linear trend is the widest
    # direction) and 0.02 on the log marginal likelihood.
    assert np.max(np.abs(inla_mean - bf_mean) / inla_sd) < 0.05
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
    assert ds["linear_predictor.mean"].shape == (N,)


@pytest.mark.slow
def test_non_converged_design_points_are_dropped_with_a_warning(
    gaussian_fit, monkeypatch
):
    import pyrox_lgm._inla as inla_mod

    model, data, _ = gaussian_fit
    real = inla_mod._fit_point
    calls = {"n": 0}

    def flaky(*args):
        mean, var, ok, lp, *rest = real(*args)
        calls["n"] += 1
        # The second design point (not the mode) fails its fit and its retry.
        return mean, var, ok & (calls["n"] not in (2, 3)), lp, *rest

    monkeypatch.setattr(inla_mod, "_fit_point", flaky)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = lgm.inla(model, data, strategy="gaussian")
    assert res.n_dropped == 1
    assert any("dropped 1" in str(w.message) for w in caught)


@pytest.mark.slow
def test_a_retried_point_is_reweighted_with_its_converged_fit(
    gaussian_fit, monkeypatch
):
    # The first fit of point 1 "fails" with a wrong log-posterior; the retry
    # converges, so weights and evidence must equal those of a clean run.
    import pyrox_lgm._inla as inla_mod

    model, data, clean = gaussian_fit
    real = inla_mod._fit_point
    calls = {"n": 0}

    def flaky(*args):
        mean, var, ok, lp, *rest = real(*args)
        calls["n"] += 1
        if calls["n"] == 2:
            return mean, var, ok & False, lp - 50.0, *rest
        return mean, var, ok, lp, *rest

    monkeypatch.setattr(inla_mod, "_fit_point", flaky)
    res = lgm.inla(model, data, strategy="gaussian")
    assert res.n_dropped == 0
    assert res.newton_iters[1] == 200
    assert {res.newton_iters[0], *res.newton_iters[2:]} == {50}
    assert np.allclose(res.theta_weights, clean.theta_weights)
    assert np.isclose(res.log_marginal_likelihood, clean.log_marginal_likelihood)


@pytest.mark.slow
def test_a_mode_that_needs_the_retry_reruns_the_whole_fit(gaussian_fit, monkeypatch):
    # The mode's fit at the first budget is suspect (the mode search and the
    # Hessian ran on it): inla() re-runs everything at 4x, and raises if the
    # mode still fails there.
    import pyrox_lgm._inla as inla_mod

    model, data, clean = gaussian_fit
    real = inla_mod._fit_point

    def fails_below(budget):
        def fit(model_, A, d, u, n_iter, subspace, sla=False):
            mean, var, ok, lp, *rest = real(model_, A, d, u, n_iter, subspace, sla)
            return mean, var, ok & (n_iter >= budget), lp, *rest

        return fit

    monkeypatch.setattr(inla_mod, "_fit_point", fails_below(200))
    with pytest.warns(UserWarning, match="re-running the whole fit"):
        res = lgm.inla(model, data, strategy="gaussian")
    assert res.newton_iters[0] == 200
    assert np.allclose(res.theta_weights, clean.theta_weights, atol=1e-6)

    monkeypatch.setattr(inla_mod, "_fit_point", fails_below(10_000))
    with pytest.raises(RuntimeError, match="along its search did not converge"):
        lgm.inla(model, data, strategy="gaussian")


@pytest.mark.slow
def test_a_mode_search_that_misses_the_tolerance_raises(gaussian_fit):
    model, data, _ = gaussian_fit
    with pytest.raises(RuntimeError, match="did not reach"):
        lgm.inla(model, data, max_theta_iter=1)


@pytest.mark.slow
def test_the_mode_search_rejects_unconverged_inner_fits():
    # At u = 0 this Poisson model's inner Newton fit needs 21 iterations: with
    # 8 the search must not run on the unconverged log-posterior, so inla()
    # escalates to 32 and lands on the same fit as the default budget.
    import pyrox_lgm._inla as inla_mod

    n = 20
    t = np.arange(n)
    y = np.random.default_rng(0).poisson(np.exp(2.0 + np.sin(t / 3.0)))
    model = lgm.LGM(
        (lgm.RW1(n, name="t"),), lgm.FixedEffects(("intercept",)), lgm.Poisson()
    )
    data = {"y": y.astype(float), "t": t}
    A = model.projector(data)
    jdata = {k: jnp.asarray(v) for k, v in data.items()}
    _, _, ok = inla_mod._theta_mode(
        model, A, jdata, 8, jnp.zeros(1), max_iter=50, tol=1e-5, verbose=False
    )
    assert not ok
    with pytest.warns(UserWarning, match="re-running the whole fit"):
        res = lgm.inla(model, data, max_newton=8, strategy="gaussian")
    clean = lgm.inla(model, data, strategy="gaussian")
    assert np.allclose(res.theta_mode, clean.theta_mode, atol=1e-6)


@pytest.mark.slow
def test_grid_spde_inside_an_lgm_is_the_exact_gaussian_marginal():
    spde = lgm.SPDE(grid=(3, 4), spacing=0.5, name="s")
    model = lgm.LGM((spde,), likelihood=lgm.Gaussian())
    rng = np.random.default_rng(3)
    y = rng.normal(size=12)
    data = {"y": y, "s": np.arange(12)}
    u = jnp.array([0.2, -0.1, 0.5])  # log range, log sigma, log prec
    theta = model.unflatten(u)
    Q = np.asarray(
        spde.prior({"range_sigma": theta["s.range_sigma"]}).precision.as_matrix()
    )
    cov = np.linalg.inv(Q) + np.eye(12) / float(theta["lik.prec"])
    ref = multivariate_normal(np.zeros(12), cov).logpdf(y) + float(model.log_prior(u))
    assert np.isclose(float(model.log_posterior_theta(u, data)), ref, atol=1e-9)


def test_empirical_bayes_moments_are_those_of_the_gaussian_approximation():
    # Both hyperparameters are positive (exp bijection), so under
    # u ~ N(mu, s^2) each is lognormal: the moments are closed form.
    from pyrox_lgm._inla import _hyperpar_summaries

    model, _ = _gaussian_rw2()
    mu = jnp.array([0.3, -0.2])
    cov = jnp.array([[0.5, 0.1], [0.1, 0.2]])
    out = _hyperpar_summaries(model, mu, cov)
    for k, key in enumerate(model.theta_spec()):
        m, v = float(mu[k]), float(cov[k, k])
        assert np.isclose(float(out[key].mean), np.exp(m + v / 2), rtol=1e-8)
        sd = np.sqrt((np.exp(v) - 1) * np.exp(2 * m + v))
        assert np.isclose(float(out[key].sd), sd, rtol=1e-6)
        assert np.isclose(float(out[key].q50), np.exp(m), rtol=1e-12)


@pytest.mark.slow
def test_empirical_bayes_summaries_are_consistent(gaussian_fit):
    model, data, _ = gaussian_fit
    res = lgm.inla(model, data, strategy="gaussian", integration="eb")
    assert res.theta_points.shape[0] == 1
    for s in res.hyperpar.values():
        assert float(s.sd) > 0
        assert float(s.q025) < float(s.mean) < float(s.q975)


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
    from numpyro.diagnostics import effective_sample_size
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
    post = mcmc.get_samples(group_by_chain=True)
    for name in ("intercept", "x"):
        s = res.fixed[name]
        chain = np.asarray(post[name])
        draws = chain.reshape(-1)
        ess = float(effective_sample_size(chain))
        mcse = draws.std() / np.sqrt(ess)
        # INLA's approximation error (0.15 sd) plus 4 Monte Carlo standard
        # errors of the NUTS mean, from the chain's effective sample size.
        assert abs(float(s.mean) - draws.mean()) < 0.15 * draws.std() + 4 * mcse
        assert abs(float(s.sd) / draws.std() - 1.0) < 0.2, name


# ---------------------------------------------------------------------------
# R-INLA's integration over theta (jejjohnson/pyrox#274)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("m", "k"), [(1, 11), (2, 45), (3, 15), (4, 25), (5, 27)])
def test_rinla_design_sizes_and_centre(m, k):
    from pyrox_lgm._inla import _rinla_design

    x, log_w = _rinla_design(m)
    assert x.shape == (k, m) and log_w.shape == (k,)
    assert np.all(x[0] == 0.0)  # the mode first
    if m > 4:  # a resolution-V fraction need not be closed under z -> -z
        return
    # Symmetric under z -> -z, with equal weights.
    for i in range(k):
        j = np.flatnonzero(np.all(np.isclose(x, -x[i]), axis=1))
        assert j.size == 1 and np.isclose(log_w[i], log_w[j[0]])


@pytest.mark.parametrize("m", [3, 4, 5, 6])
def test_rinla_ccd_integrates_a_gaussian_to_second_order(m):
    # The CCD weights (Rue, Martino & Chopin, 2009, sec. 6.5) make the design
    # exact for the mean and covariance of a standard Gaussian posterior.
    from pyrox_lgm._inla import _rinla_design

    x, log_w = _rinla_design(m)
    lw = log_w - 0.5 * np.sum(x**2, axis=1)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    assert np.allclose(w @ x, 0.0, atol=1e-12)
    assert np.allclose(np.einsum("k,ki,kj->ij", w, x, x), np.eye(m), atol=1e-12)
    assert np.allclose(np.linalg.norm(x[1:], axis=1), 1.1 * np.sqrt(m))


def test_rinla_fixed_grids_are_its_tables():
    # R-INLA's m = 1 grid (scaled by f0 = 1.1) gives a standard Gaussian
    # variance 1.004; its m = 2 grid, scaled by f0 sqrt(2) as R-INLA does,
    # gives 0.874 (both measured on R-INLA 26.8.7's tables).
    from pyrox_lgm._inla import _rinla_design

    for m, var in [(1, 1.0041), (2, 0.8738)]:
        x, log_w = _rinla_design(m)
        lw = log_w - 0.5 * np.sum(x**2, axis=1)
        w = np.exp(lw - lw.max())
        w /= w.sum()
        assert np.allclose(np.sum(w[:, None] * x**2, axis=0), var, atol=1e-4)


def test_skewness_corrections_keep_the_scale_and_find_the_asymmetry():
    from pyrox_lgm._inla import _skewness_corrections

    scale = np.diag([2.0, 0.5])
    u_star = np.array([1.0, -1.0])

    def gaussian(u):
        z = np.linalg.solve(scale, np.asarray(u) - u_star)
        return -0.5 * float(z @ z)

    lo, hi = _skewness_corrections(gaussian, u_star, scale, 0.0)
    assert np.allclose(lo, 1.0) and np.allclose(hi, 1.0)

    def split(u):  # half-scales 0.5 below and 2 above on the first axis
        z = np.linalg.solve(scale, np.asarray(u) - u_star)
        s = np.where(z > 0, [2.0, 1.0], [0.5, 1.0])
        return -0.5 * float(np.sum((z / s) ** 2))

    lo, hi = _skewness_corrections(split, u_star, scale, 0.0)
    assert np.allclose(lo * hi, 1.0)  # skewness only (R-INLA's default)
    assert np.allclose(hi / lo, [4.0, 1.0])


def test_theta_marginals_of_a_symmetric_posterior_are_its_gaussian():
    from pyrox_lgm._inla import _hyperpar_summaries, _theta_marginals

    model, _ = _gaussian_rw2()
    mu = np.array([0.3, -0.2])
    cov = np.array([[0.5, 0.1], [0.1, 0.2]])
    evals, evecs = np.linalg.eigh(np.linalg.inv(cov))
    ones = (np.ones(2), np.ones(2))
    out = _hyperpar_summaries(
        model,
        jnp.asarray(mu),
        jnp.asarray(cov),
        _theta_marginals(mu, cov, evecs, evals, ones),
    )
    for k, key in enumerate(model.theta_spec()):
        m, v = mu[k], cov[k, k]
        assert np.isclose(float(out[key].q50), np.exp(m), rtol=1e-4)
        assert np.isclose(
            float(out[key].q975), np.exp(m + 1.959964 * np.sqrt(v)), rtol=1e-3
        )
        assert np.isclose(float(out[key].mean), np.exp(m + v / 2), rtol=1e-4)


def test_vb_subspace_is_r_inlas_node_selection():
    # f.enable.limit = 30: a component of n > 30 nodes contributes 30 nodes
    # j * (n // 30) + max(1, (n // 30) // 2); R-INLA's verbose log of the
    # SPDE fixture (234 nodes) lists 3, 10, ..., 206 and the intercept.
    from pyrox_lgm._inla import _vb_subspace

    for n, nodes in [
        (20, list(range(20))),
        (50, list(range(1, 31))),
        (234, list(range(3, 207, 7))),
    ]:
        model = lgm.LGM(
            (lgm.RW2(n, name="t"),), lgm.FixedEffects(("intercept",)), lgm.Gaussian()
        )
        assert _vb_subspace(model).tolist() == [*nodes, n]


def test_vb_subspace_keeps_the_group_axis_of_a_replicated_kronecker():
    # Replicate(Kronecker(main, group)) is (replicate, main, group) in the flat
    # vector, and the node limit shares 1024 over n_group * n_rep.
    from pyrox_lgm._inla import _vb_subspace

    n, ng, nrep = 100, 4, 2
    comp = lgm.Replicate(
        lgm.Kronecker(lgm.RW2(n, name="t"), lgm.AR1(ng, name="g")), nrep, name="st"
    )
    model = lgm.LGM((comp,), lgm.FixedEffects(("intercept",)), lgm.Gaussian())
    js = (np.arange(30) * (n // 30) + max(1, (n // 30) // 2)) % n
    nodes = [
        r * n * ng + j * ng + g for r in range(nrep) for j in js for g in range(ng)
    ]
    got = _vb_subspace(model).tolist()
    assert got == [*sorted(nodes), n * ng * nrep]
    assert len(got) == 30 * ng * nrep + 1


def test_theta_marginals_with_only_the_mode_fall_back_to_the_hessian():
    from pyrox_lgm._inla import _theta_marginals

    mu, cov = np.array([0.3]), np.array([[0.4]])
    evals, evecs = np.linalg.eigh(np.linalg.inv(cov))
    ones = (np.ones(1), np.ones(1))
    ref = _theta_marginals(mu, cov, evecs, evals, ones)
    only_mode = (np.array([0.3]), np.array([0.0]))
    got = _theta_marginals(mu, cov, evecs, evals, ones, only_mode)
    assert np.allclose(got[0], ref[0]) and np.allclose(got[1], ref[1])


def test_analytic_range_limited_moments_match_numerical_integration():
    # Two-component mixture, per column: the closed-form moments on mean +- 5
    # sds against a fine trapezoid rule, and the memory-bounded skew-normal path
    # (zero skewness is the Gaussian) against the closed form.
    means = jnp.array([[0.0, 1.0, -2.0], [3.0, 1.5, 0.5]])
    variances = jnp.array([[1.0, 0.25, 4.0], [9.0, 0.5, 1.0]])
    w = jnp.array([0.7, 0.3])
    cut = mixture_summary(means, variances, w, limit=5.0)
    mean = jnp.sum(w[:, None] * means, 0)
    sd = jnp.sqrt(jnp.sum(w[:, None] * (variances + means**2), 0) - mean**2)
    for j in range(3):
        a, b = float(mean[j] - 5 * sd[j]), float(mean[j] + 5 * sd[j])
        x = np.linspace(a, b, 400_001)
        f = sum(
            float(w[k])
            * norm.pdf(x, float(means[k, j]), np.sqrt(float(variances[k, j])))
            for k in range(2)
        )
        m0 = np.trapezoid(f, x)
        m = np.trapezoid(f * x, x) / m0
        v = np.trapezoid(f * (x - m) ** 2, x) / m0
        assert np.isclose(float(cut.mean[j]), m, atol=1e-9)
        assert np.isclose(float(cut.sd[j]), np.sqrt(v), atol=1e-9)
    skew0 = mixture_summary(means, variances, w, jnp.zeros((2, 3)), limit=5.0)
    assert np.allclose(skew0.mean, cut.mean, atol=1e-8)
    assert np.allclose(skew0.sd, cut.sd, atol=1e-8)


def test_the_skew_normal_summary_is_chunked_over_columns(monkeypatch):
    import pyrox_lgm._result as result_mod

    rng = np.random.default_rng(1)
    means = jnp.asarray(rng.normal(size=(2, 7)))
    variances = jnp.asarray(rng.uniform(0.5, 2.0, size=(2, 7)))
    skew = jnp.asarray(rng.uniform(-0.5, 0.5, size=(2, 7)))
    w = jnp.array([0.4, 0.6])
    full = mixture_summary(means, variances, w, skew, limit=5.0)
    monkeypatch.setattr(result_mod, "_CHUNK", 3)  # 7 columns -> 3 padded chunks
    chunked = mixture_summary(means, variances, w, skew, limit=5.0)
    for a, b in zip(full, chunked, strict=True):
        assert np.allclose(a, b)


def test_range_limited_mixture_summary():
    # One Gaussian: +-5 sds hold all but 6e-7 of it. A wide component that
    # reaches far beyond the mixture's +-5 sds loses that mass.
    means = jnp.array([[0.0], [0.0]])
    variances = jnp.array([[1.0], [400.0]])
    w = jnp.array([0.99, 0.01])
    one = mixture_summary(means[:1], variances[:1], w[:1] / w[0], limit=5.0)
    assert np.isclose(float(one.sd[0]), 1.0, atol=1e-5)
    assert np.isclose(float(one.q975[0]), 1.959964, atol=1e-4)
    full = mixture_summary(means, variances, w)
    cut = mixture_summary(means, variances, w, limit=5.0)
    sd_full = np.sqrt(0.99 + 0.01 * 400.0)
    assert np.isclose(float(full.sd[0]), sd_full)
    a = 5.0 * sd_full  # the mixture restricted to [-a, a], in closed form
    mass = 0.99 * (2 * norm.cdf(a) - 1) + 0.01 * (2 * norm.cdf(a / 20) - 1)

    def second(s):  # int_{-a}^{a} x^2 N(x; 0, s^2) dx
        return s**2 * (2 * norm.cdf(a / s) - 1) - 2 * a * s * norm.pdf(a / s)

    sd_cut = np.sqrt((0.99 * second(1.0) + 0.01 * second(20.0)) / mass)
    assert np.isclose(float(cut.sd[0]), sd_cut, rtol=1e-6)
    assert float(cut.sd[0]) < 0.9 * sd_full


@pytest.mark.slow
def test_an_unconverged_skewness_probe_asks_for_the_larger_budget(monkeypatch):
    import pyrox_lgm._inla as inla_mod

    n = 12
    t = np.arange(n)
    y = np.sin(t / 3.0) + 0.1 * np.cos(5.0 * t)
    model = lgm.LGM(
        (lgm.RW1(n, name="t"),), lgm.FixedEffects(("intercept",)), lgm.Gaussian()
    )
    data = {"y": y, "t": t}
    real = inla_mod._log_post_converged

    def fails_below(budget):
        def probe(model_, A, d, max_newton, u):
            value, ok = real(model_, A, d, max_newton, u)
            return value, ok & (max_newton >= budget)

        return probe

    clean = lgm.inla(model, data, strategy="gaussian")
    monkeypatch.setattr(inla_mod, "_log_post_converged", fails_below(200))
    with pytest.warns(UserWarning, match="skewness probe"):
        res = lgm.inla(model, data, strategy="gaussian")
    assert np.allclose(res.theta_weights, clean.theta_weights, atol=1e-6)

    monkeypatch.setattr(inla_mod, "_log_post_converged", fails_below(10_000))
    with pytest.raises(RuntimeError, match="skewness probe"):
        lgm.inla(model, data, strategy="gaussian")
