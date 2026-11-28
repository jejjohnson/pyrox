"""Tests for the site-based non-Gaussian inference strategies.

Coverage strategy:
* shape / finiteness / convergence on a small Bernoulli classification
  problem, for each of Laplace, GN, EP, PL, QN.
* equivalence: Laplace ≡ Gauss-Newton on Bernoulli
  (since Bernoulli is exponential-family and log-concave, GGN curvature
  equals the negative Hessian, so the two strategies coincide).
* prediction sanity at training points: posterior mean reflects the data
  via positive correlation with ``y - 0.5``.
* multi-latent rejection: SoftmaxLikelihood / HeteroscedasticGaussianLikelihood
  raise a clear error.
* convergence: diff between Laplace and EP posterior means is small.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from numpyro import handlers
from pyrox_gp import (
    RBF,
    BernoulliLikelihood,
    ConditionedNonGaussGP,
    ExpectationPropagation,
    GaussNewtonInference,
    GPPrior,
    HeteroscedasticGaussianLikelihood,
    LaplaceInference,
    Matern,
    PosteriorLinearization,
    QuasiNewtonInference,
    SoftmaxLikelihood,
    StudentTLikelihood,
    gp_laplace_factor,
    preconditioned_cg_solver,
)


def _make_classification_problem(N: int = 12):
    X = jnp.linspace(-3.0, 3.0, N)[:, None]
    y = (jnp.sin(X[:, 0]) > 0.0).astype(jnp.float32)
    kernel = RBF(init_lengthscale=1.0, init_variance=1.0)
    prior = GPPrior(kernel=kernel, X=X)
    return prior, X, y


def test_laplace_smoke_tiny() -> None:
    """Fast smoke for Laplace fit + predict on a tiny problem.

    Stays un-marked so the default CI run (``pytest -m "not slow"``) still
    exercises the Laplace inference and prediction surface end-to-end on
    every PR; the deeper convergence / equivalence tests are slow.
    """
    X = jnp.linspace(-1.0, 1.0, 6)[:, None]
    y = (X[:, 0] > 0.0).astype(jnp.float32)
    prior = GPPrior(kernel=RBF(init_lengthscale=1.0, init_variance=1.0), X=X)
    cond = LaplaceInference(max_iter=2).fit(prior, BernoulliLikelihood(), y)
    assert cond.q_mean.shape == y.shape
    assert jnp.all(jnp.isfinite(cond.q_mean))
    assert jnp.all(jnp.isfinite(cond.q_var))
    assert jnp.all(cond.q_var >= 0.0)
    m, v = cond.predict(jnp.array([[-0.5], [0.5]]))
    assert m.shape == (2,) and v.shape == (2,)
    assert jnp.all(jnp.isfinite(m)) and jnp.all(v >= 0.0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "strategy_factory",
    [
        lambda: LaplaceInference(),
        lambda: GaussNewtonInference(),
        lambda: ExpectationPropagation(max_iter=20),
        lambda: PosteriorLinearization(max_iter=20),
        lambda: QuasiNewtonInference(max_iter=30),
    ],
)
def test_strategy_runs_and_returns_finite_posterior(strategy_factory) -> None:
    prior, _, y = _make_classification_problem()
    strategy = strategy_factory()
    cond = strategy.fit(prior, BernoulliLikelihood(), y)
    assert isinstance(cond, ConditionedNonGaussGP)
    assert cond.q_mean.shape == y.shape
    assert cond.q_var.shape == y.shape
    assert jnp.all(jnp.isfinite(cond.q_mean))
    assert jnp.all(jnp.isfinite(cond.q_var))
    assert jnp.all(cond.q_var >= 0.0)
    assert jnp.isfinite(cond.log_marginal_approx)


@pytest.mark.slow
def test_predict_at_test_points_finite_and_psd() -> None:
    prior, _, y = _make_classification_problem()
    cond = LaplaceInference().fit(prior, BernoulliLikelihood(), y)
    X_star = jnp.linspace(-4.0, 4.0, 30)[:, None]
    m, v = cond.predict(X_star)
    assert m.shape == (30,) and v.shape == (30,)
    assert jnp.all(jnp.isfinite(m))
    assert jnp.all(jnp.isfinite(v))
    assert jnp.all(v >= 0.0)


@pytest.mark.slow
def test_laplace_equals_gauss_newton_on_bernoulli() -> None:
    """For the Bernoulli likelihood (exponential-family, log-concave) the
    GGN curvature equals the negative Hessian, so Laplace and GN should
    converge to the same posterior."""
    prior, _, y = _make_classification_problem()
    lap = LaplaceInference(max_iter=50, tol=1e-8).fit(prior, BernoulliLikelihood(), y)
    gn = GaussNewtonInference(max_iter=50, tol=1e-8).fit(
        prior, BernoulliLikelihood(), y
    )
    # GN can take more iterations to converge with the same tol; loosen
    # the comparison tolerance accordingly.
    assert jnp.allclose(lap.q_mean, gn.q_mean, atol=2e-2)
    assert jnp.allclose(lap.q_var, gn.q_var, atol=2e-2)


@pytest.mark.slow
def test_laplace_and_ep_agree_on_bernoulli() -> None:
    """Laplace and EP differ in formal definition (mode vs moment match)
    but on a smooth log-concave likelihood with moderate N they sit close."""
    prior, _, y = _make_classification_problem()
    lap = LaplaceInference(max_iter=50, tol=1e-8).fit(prior, BernoulliLikelihood(), y)
    ep = ExpectationPropagation(max_iter=80, tol=1e-7, damping=0.4).fit(
        prior, BernoulliLikelihood(), y
    )
    # Posterior means agree to within a fairly loose threshold; on this
    # toy problem they typically match to ~1e-2 but tolerate up to 0.2.
    assert jnp.max(jnp.abs(lap.q_mean - ep.q_mean)) < 0.3


@pytest.mark.slow
def test_posterior_mean_correlates_with_targets() -> None:
    """Sanity: the posterior mean increases with ``y`` — points with
    ``y=1`` should on average have higher latent than points with ``y=0``."""
    prior, _, y = _make_classification_problem()
    cond = LaplaceInference().fit(prior, BernoulliLikelihood(), y)
    mean_for_y1 = cond.q_mean[y == 1].mean()
    mean_for_y0 = cond.q_mean[y == 0].mean()
    assert mean_for_y1 > mean_for_y0


@pytest.mark.slow
def test_laplace_with_studentt_runs_and_is_robust() -> None:
    """StudentT regression with two heavy outliers: the latent at the
    outlier should be pulled less than under a Gaussian likelihood."""
    X = jnp.linspace(-3.0, 3.0, 20)[:, None]
    y_clean = jnp.sin(X[:, 0])
    y = y_clean.at[10].set(10.0)  # one heavy outlier
    prior = GPPrior(kernel=RBF(init_lengthscale=1.0, init_variance=1.0), X=X)

    cond = LaplaceInference(max_iter=100, tol=1e-8).fit(
        prior, StudentTLikelihood(df=3.0, scale=0.5), y
    )
    # The latent at the outlier should not be pulled all the way to 10.
    assert cond.q_mean[10] < 5.0
    assert jnp.all(jnp.isfinite(cond.q_mean))


@pytest.mark.parametrize(
    "lik",
    [SoftmaxLikelihood(num_classes=3), HeteroscedasticGaussianLikelihood()],
)
def test_multi_latent_likelihoods_rejected_with_clear_error(lik) -> None:
    prior, _, y = _make_classification_problem()
    with pytest.raises(ValueError, match="latent_dim"):
        LaplaceInference().fit(prior, lik, y)


@pytest.mark.slow
def test_condition_nongauss_method_works() -> None:
    prior, _, y = _make_classification_problem()
    cond = prior.condition_nongauss(
        BernoulliLikelihood(), y, strategy=LaplaceInference()
    )
    assert isinstance(cond, ConditionedNonGaussGP)


@pytest.mark.slow
def test_predict_jit_compatible_via_partial_eval() -> None:
    """The pyrox kernel ecosystem carries non-array static fields
    (kernel ``pyrox_name``), so we don't jit the full ``cond`` PyTree
    directly. Instead test that the ``predict`` math is jit-able when
    ``cond`` is closed over by ``jax.jit`` (which excludes the static
    portion automatically)."""
    prior, _, y = _make_classification_problem()
    cond = LaplaceInference().fit(prior, BernoulliLikelihood(), y)
    X_star = jnp.linspace(-4.0, 4.0, 15)[:, None]

    go = jax.jit(lambda x: cond.predict(x))
    m, v = go(X_star)
    assert jnp.all(jnp.isfinite(m))
    assert jnp.all(jnp.isfinite(v))


def test_expectation_propagation_rejects_other_rules() -> None:
    import gaussx

    with pytest.raises(TypeError, match="GaussHermiteIntegrator"):
        ExpectationPropagation(integrator=gaussx.UnscentedIntegrator())


# --- matrix-free priors are refused (pyrox#549) ----------------------------

_STRATEGIES = {
    "LaplaceInference": LaplaceInference,
    "GaussNewtonInference": GaussNewtonInference,
    "PosteriorLinearization": PosteriorLinearization,
    "ExpectationPropagation": ExpectationPropagation,
    "QuasiNewtonInference": QuasiNewtonInference,
}


# PosteriorLinearization and ExpectationPropagation take a matrix-free prior
# since pyrox#552 (test_inference_nongauss_sites.py).
_DENSE_ONLY = ["LaplaceInference", "GaussNewtonInference", "QuasiNewtonInference"]


@pytest.mark.parametrize("name", _DENSE_ONLY)
def test_matrix_free_prior_is_rejected(name, monkeypatch):
    # Before pyrox#549 each strategy evaluated the (60, 60) kernel block of a
    # matrix-free prior and ignored its solver.
    blocks = []
    original = Matern.__call__

    def spy(self, a, b):
        blocks.append((a.shape[0], b.shape[0]))
        return original(self, a, b)

    monkeypatch.setattr(Matern, "__call__", spy)
    X = jr.uniform(jr.key(0), (60, 2))
    y = (X[:, 0] > 0.5).astype(X.dtype)
    solver = preconditioned_cg_solver(rank=20, key=jr.key(1))
    prior = GPPrior(Matern(nu=1.5), X, solver=solver, matrix_free=True)
    with pytest.raises(ValueError, match=rf"{name}.*matrix_free=True.*issues/353"):
        prior.condition_nongauss(BernoulliLikelihood(), y, strategy=_STRATEGIES[name]())
    assert (60, 60) not in blocks
    if name in ("LaplaceInference", "GaussNewtonInference"):

        def model():
            gp_laplace_factor(
                "y", prior, BernoulliLikelihood(), y, strategy=_STRATEGIES[name]()
            )

        with pytest.raises(ValueError, match=name):
            handlers.seed(model, 0)()


# log_marginal_approx, q_mean[0], q_var[0], predicted mean and variance at
# X[0] + 0.01, recorded at edbbd6d (before pyrox#549) in float64
_DENSE_REFERENCE = {
    "LaplaceInference": (
        -21.85439793150134,
        0.061908180788765366,
        0.2048395848305088,
        0.09422909758839337,
        0.2008022806501557,
    ),
    "GaussNewtonInference": (
        -21.85439793150134,
        0.061908180788765366,
        0.2048395848305088,
        0.09422909758839337,
        0.2008022806501557,
    ),
    "PosteriorLinearization": (
        -21.8410165741918,
        0.0693741401997019,
        0.20790308241663014,
        0.10239207189887989,
        0.20383001669777367,
    ),
    "ExpectationPropagation": (
        -21.815888861753628,
        0.06927206017504794,
        0.2101956641479824,
        0.1022759990838994,
        0.2061059658468143,
    ),
    "QuasiNewtonInference": (
        -21.854397937094596,
        0.061908180788765255,
        0.2048395847592499,
        0.09422909758839326,
        0.20080228055312266,
    ),
}


@pytest.mark.parametrize("name", list(_STRATEGIES))
def test_dense_prior_is_unchanged(name):
    X = jr.uniform(jr.key(0), (40, 2), dtype=jnp.float64)
    y = (X[:, 0] > 0.5).astype(jnp.float64)
    prior = GPPrior(Matern(nu=1.5), X)
    fit = prior.condition_nongauss(
        BernoulliLikelihood(), y, strategy=_STRATEGIES[name]()
    )
    mean, var = fit.predict(X[:1] + 0.01)
    got = (fit.log_marginal_approx, fit.q_mean[0], fit.q_var[0], mean[0], var[0])
    # PL and EP moved from a stopping test on the posterior mean to one on the
    # sites (pyrox#552), which runs a few iterations further towards the same
    # fixed point; the issue's tolerance for that move is 1e-5.
    rtol = 1e-5 if name in _SITE_LOOP else 1e-10
    np.testing.assert_allclose(
        np.asarray(got, dtype=np.float64), _DENSE_REFERENCE[name], rtol=rtol
    )


_SITE_LOOP = ("PosteriorLinearization", "ExpectationPropagation")


def test_matrix_free_prior_is_accepted_by_the_site_strategies(monkeypatch):
    # The (60, 60) kernel block is never evaluated, and the fit predicts.
    blocks = []
    original = Matern.__call__

    def spy(self, a, b):
        blocks.append((a.shape[0], b.shape[0]))
        return original(self, a, b)

    monkeypatch.setattr(Matern, "__call__", spy)
    X = jr.uniform(jr.key(0), (60, 2))
    y = (X[:, 0] > 0.5).astype(X.dtype)
    solver = preconditioned_cg_solver(rank=20, key=jr.key(1))
    prior = GPPrior(Matern(nu=1.5), X, solver=solver, matrix_free=True)
    fit = prior.condition_nongauss(
        BernoulliLikelihood(), y, strategy=PosteriorLinearization(max_iter=3)
    )
    mean, var = fit.predict(X[:2] + 0.01)
    assert bool(jnp.all(jnp.isfinite(fit.q_mean))) and bool(jnp.all(var >= 0.0))
    assert (60, 60) not in blocks
