"""Laplace and Gauss-Newton with a solver choice, on a matrix-free prior (#551).

`LaplaceInference` and `GaussNewtonInference` used to form the dense kernel
matrix, ignore the prior's ``solver`` and refuse ``GPPrior(matrix_free=True)``
(#549). They now take ``solver``, ``preconditioner`` and ``logdet`` (falling
back to the prior's), fit through ``argminax.solve.laplace_mode_covariance``
with that solver on the prior's own operator, and keep the fit as a
``gaussx.SitePosterior`` whose weights and hyperparameters `predict` reuses.
The dense default is checked against frozen values in
``test_inference_nongauss.py::test_dense_prior_is_unchanged``.
"""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from pyrox_gp import (
    RBF,
    BernoulliLikelihood,
    GaussNewtonInference,
    GPPrior,
    LaplaceInference,
    Matern,
    PosteriorLinearization,
    QuasiNewtonInference,
    bbmm_solver,
    gp_laplace_factor,
    matfree_solver,
)


# CG well below the Newton tolerance (1e-6), so the loop can resolve its step
_CG = gx.CGSolver(rtol=1e-10, atol=1e-10, max_steps=500)


def _smooth_problem(n: int, seed: int = 0, dtype=jnp.float64):
    """The issue's smooth case: Matérn-3/2, variance 1, lengthscale 0.3."""
    X = jr.uniform(jr.key(seed), (n, 2), dtype=dtype)
    y = (jnp.sin(6.0 * X[:, 0]) + jnp.cos(4.0 * X[:, 1]) > 0.5).astype(dtype)
    kernel = Matern(nu=1.5, init_lengthscale=0.3, init_variance=1.0)
    return X, y, kernel


def _avals(jaxpr):
    """Every abstract value in a jaxpr and the jaxprs nested in its equations."""
    for eqn in jaxpr.eqns:
        for var in (*eqn.invars, *eqn.outvars):
            if hasattr(var, "aval"):
                yield var.aval
        for param in eqn.params.values():
            for sub in param if isinstance(param, list | tuple) else (param,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    yield from _avals(inner)


def _shapes(fn, *args):
    return {tuple(aval.shape) for aval in _avals(jax.make_jaxpr(fn)(*args).jaxpr)}


# --- smoke -------------------------------------------------------------------


def test_matrix_free_cg_smoke() -> None:
    """Laplace with CG on a matrix-free prior fits, predicts and converges."""
    X, y, kernel = _smooth_problem(30)
    prior = GPPrior(kernel, X, solver=_CG, matrix_free=True)
    fit = prior.condition_nongauss(
        BernoulliLikelihood(), y, strategy=LaplaceInference(), key=jr.key(0)
    )
    mean, var = fit.predict(X[:3] + 0.01)
    assert bool(fit.converged)
    assert bool(jnp.all(jnp.isfinite(fit.q_mean)))
    assert bool(jnp.all(fit.q_var > 0.0))
    assert bool(jnp.isfinite(fit.log_marginal_approx))
    assert bool(jnp.all(jnp.isfinite(mean))) and bool(jnp.all(var >= 0.0))


# --- the solvers reach the dense fit -------------------------------------------


def _configurations(key):
    """Strategy and prior settings, each solving by CG in some form."""
    return {
        "cg-dense-prior": (
            {},
            LaplaceInference(solver=_CG, logdet=gx.DenseLogdet()),
        ),
        "cg-nystrom-matrix-free": (
            {
                "solver": gx.CGSolver(rtol=1e-10, atol=1e-10, max_steps=500),
                "preconditioner": gx.NystromPreconditioner.lazy(rank=150, key=key),
                "matrix_free": True,
            },
            LaplaceInference(),
        ),
        "bbmm-matrix-free": (
            {
                "solver": bbmm_solver(rank=150, rtol=1e-10, key=key),
                "matrix_free": True,
            },
            LaplaceInference(),
        ),
        "gauss-newton-cg-matrix-free": (
            {"solver": _CG, "matrix_free": True},
            GaussNewtonInference(),
        ),
    }


_CONFIGURATIONS = list(_configurations(jr.key(0)))


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(name, marks=() if name == "cg-dense-prior" else pytest.mark.slow)
        for name in _CONFIGURATIONS
    ],
)
def test_configurations_match_dense(name) -> None:
    """N = 200 Bernoulli: every iterative configuration reaches the dense fit.

    The issue asks for the posterior mean at rtol 1e-4; CG at 1e-10 lands far
    inside it, and the variances and predictions with it.
    """
    X, y, kernel = _smooth_problem(200)
    prior_fields, strategy = _configurations(jr.key(1))[name]
    reference_strategy = (
        GaussNewtonInference()
        if isinstance(strategy, GaussNewtonInference)
        else LaplaceInference()
    )
    reference = reference_strategy.fit(GPPrior(kernel, X), BernoulliLikelihood(), y)
    fit = strategy.fit(
        GPPrior(kernel, X, **prior_fields), BernoulliLikelihood(), y, key=jr.key(2)
    )
    assert bool(reference.converged) and bool(fit.converged)
    assert fit.solver is not None
    np.testing.assert_allclose(fit.q_mean, reference.q_mean, rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(fit.q_var, reference.q_var, rtol=1e-4)
    X_star = X[:5] + 0.01
    ref_mean, ref_var = reference.predict(X_star)
    mean, var = fit.predict(X_star)
    np.testing.assert_allclose(mean, ref_mean, rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(var, ref_var, rtol=1e-4)
    if name == "cg-dense-prior":
        # an exact log-determinant beside CG: the evidence is the dense one
        np.testing.assert_allclose(
            fit.log_marginal_approx, reference.log_marginal_approx, rtol=1e-8
        )


@pytest.mark.slow
def test_stochastic_evidence_is_within_its_standard_error() -> None:
    """N = 200, smooth kernel, rank 150: the Nyström-SLQ evidence over 20 keys.

    The issue's acceptance numbers: the mean error within three standard
    errors of zero, and an RMS error below 0.05 nats (the issue measured
    0.009 nats for the log-determinant at rank 150, of which the evidence
    carries half).
    """
    X, y, kernel = _smooth_problem(200)
    reference = LaplaceInference().fit(GPPrior(kernel, X), BernoulliLikelihood(), y)
    errors = []
    for seed in range(20):
        solver = matfree_solver(rank=150, rtol=1e-10, key=jr.key(100 + seed))
        prior = GPPrior(kernel, X, solver=solver, matrix_free=True)
        fit = LaplaceInference().fit(prior, BernoulliLikelihood(), y)
        assert bool(fit.converged)
        errors.append(float(fit.log_marginal_approx - reference.log_marginal_approx))
    errors = np.asarray(errors)
    standard_error = errors.std(ddof=1) / np.sqrt(errors.size)
    assert abs(errors.mean()) < 3.0 * standard_error + 1e-12
    assert np.sqrt(np.mean(errors**2)) < 0.05


# --- never N x N ---------------------------------------------------------------


def _matrix_free_fit_shapes(n: int, solver, *, block_size: int = 256):
    X = jnp.linspace(0.0, 1.0, n)[:, None]
    y = (jnp.sin(6.0 * X[:, 0]) > 0.0).astype(X.dtype)
    prior = GPPrior(
        Matern(nu=1.5), X, solver=solver, matrix_free=True, block_size=block_size
    )

    def fit_and_predict(labels):
        fit = LaplaceInference(max_iter=3).fit(
            prior, BernoulliLikelihood(), labels, key=jr.key(0)
        )
        mean, var = fit.predict(X[:3] + 0.01)
        return fit.q_mean, fit.log_marginal_approx, fit.converged, mean, var

    return _shapes(fit_and_predict, y)


def _solvers(name: str):
    if name == "cg":
        return gx.CGSolver(rtol=1e-8, atol=1e-8, max_steps=50)
    return bbmm_solver(rank=20, num_probes=4, lanczos_order=10, key=jr.key(0))


@pytest.mark.parametrize("name", ["cg", "bbmm"])
def test_matrix_free_never_densifies(name) -> None:
    """N = 2100, CG and BBMM: no (N, N) value in the fit, the evidence or predict.

    Before #551 a matrix-free prior raised ``ValueError`` (#549), and before
    #549 the fit evaluated the dense (N, N) kernel block.
    """
    n = 2100
    shapes = _matrix_free_fit_shapes(n, _solvers(name))
    assert (n,) in shapes
    assert not any(len(shape) >= 2 and shape[-2:] == (n, n) for shape in shapes)
    # the kernel is evaluated by row blocks of the matvec
    assert any(shape[-2:] == (256, n) for shape in shapes)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["cg", "bbmm"])
def test_slow_20k_never_densifies(name) -> None:
    """N = 20,000, the issue's scale: the traced fit holds no (N, N) value.

    Only the trace is checked; running the fit at this size is left out of
    the suite.
    """
    n = 20_000
    shapes = _matrix_free_fit_shapes(n, _solvers(name), block_size=512)
    assert not any(len(shape) >= 2 and shape[-2:] == (n, n) for shape in shapes)
    assert any(shape[-2:] == (512, n) for shape in shapes)


# --- fields ----------------------------------------------------------------------


def test_inherits_from_prior_and_overrides() -> None:
    """Each unset field comes from the prior; a set one overrides it."""
    X = jnp.linspace(-3.0, 3.0, 8)[:, None]
    y = (X[:, 0] > 0.0).astype(jnp.float64)
    prior = GPPrior(RBF(), X, solver=_CG)
    inherited = LaplaceInference().fit(prior, BernoulliLikelihood(), y)
    assert inherited.solver is _CG
    dense = LaplaceInference(solver=gx.DenseSolver()).fit(
        prior, BernoulliLikelihood(), y
    )
    # a dense fit keeps its Cholesky factor and predicts by it
    assert dense.solver is None and dense.site_posterior.factor is not None
    exact = gx.DenseLogdet()
    composed = GaussNewtonInference(logdet=exact).fit(prior, BernoulliLikelihood(), y)
    assert isinstance(composed.solver, gx.ComposedSolver)
    assert composed.solver.solve_strategy is _CG
    assert composed.solver.logdet_strategy is exact
    np.testing.assert_allclose(inherited.q_mean, dense.q_mean, rtol=1e-7)
    np.testing.assert_allclose(inherited.q_var, dense.q_var, rtol=1e-7)
    np.testing.assert_allclose(
        composed.log_marginal_approx, dense.log_marginal_approx, rtol=1e-8
    )
    # a preconditioner needs an iterative solver, and the error names the fields
    lazy = gx.NystromPreconditioner.lazy(rank=4, key=jr.key(0))
    with pytest.raises(ValueError, match=r"LaplaceInference\(solver=None"):
        LaplaceInference(preconditioner=lazy).fit(
            GPPrior(RBF(), X), BernoulliLikelihood(), y
        )


def test_inner_solve_failure_is_not_converged() -> None:
    """CG that stops at two steps makes ``converged`` false, without raising."""
    X, y, kernel = _smooth_problem(40)
    solver = gx.CGSolver(rtol=1e-10, atol=1e-10, max_steps=2, throw=False)
    prior = GPPrior(kernel, X, solver=solver, matrix_free=True)
    fit = LaplaceInference().fit(prior, BernoulliLikelihood(), y, key=jr.key(0))
    assert not bool(fit.converged)


def test_key_reaches_the_evidence() -> None:
    """BBMM's evidence depends on the key of condition_nongauss and the factor."""
    X, y, kernel = _smooth_problem(30)
    solver = gx.BBMMSolver(rtol=1e-10, atol=1e-10, num_probes=4)
    prior = GPPrior(kernel, X, solver=solver, matrix_free=True)
    strategy = LaplaceInference()

    def evidence(key):
        fit = prior.condition_nongauss(
            BernoulliLikelihood(), y, strategy=strategy, key=key
        )
        return float(fit.log_marginal_approx)

    first, again, other = evidence(jr.key(0)), evidence(jr.key(0)), evidence(jr.key(1))
    assert first == again and first != other

    def model():
        gp_laplace_factor(
            "y", prior, BernoulliLikelihood(), y, strategy=strategy, key=jr.key(1)
        )

    trace = handlers.trace(handlers.seed(model, 0)).get_trace()
    np.testing.assert_allclose(trace["y"]["fn"].log_factor, other, rtol=1e-12)


# --- one hyperparameter draw per fit ---------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [
        LaplaceInference(max_iter=5),
        QuasiNewtonInference(max_iter=30),
        PosteriorLinearization(max_iter=3),
    ],
    ids=["laplace", "quasi-newton", "posterior-linearisation"],
)
def test_one_hyperparameter_draw(strategy) -> None:
    """A kernel with priors is drawn once for the fit and its predictions.

    Before #551 ``predict`` evaluated the prior's kernel again outside the
    fit's context: under ``trace`` a second ``RBF.lengthscale`` site, under
    ``seed`` a second, independent draw.
    """
    X = jnp.linspace(-2.0, 2.0, 10)[:, None]
    y = (X[:, 0] > 0.0).astype(X.dtype)
    X_star = jnp.array([[-0.5], [0.5]])
    kernel = RBF()
    kernel.set_prior("lengthscale", dist.LogNormal(0.0, 0.5))
    prior = GPPrior(kernel=kernel, X=X)

    def model():
        fit = strategy.fit(prior, BernoulliLikelihood(), y)
        mean, var = fit.predict(X_star)
        numpyro.deterministic("mean", mean)
        numpyro.deterministic("var", var)

    trace = handlers.trace(handlers.seed(model, 3)).get_trace()
    samples = [name for name, site in trace.items() if site["type"] == "sample"]
    assert samples == ["RBF.lengthscale"]
    lengthscale = trace["RBF.lengthscale"]["value"]
    # the same predictions from a kernel fixed at the drawn value
    fixed = GPPrior(kernel=RBF(init_lengthscale=lengthscale), X=X)
    mean, var = strategy.fit(fixed, BernoulliLikelihood(), y).predict(X_star)
    np.testing.assert_allclose(trace["mean"]["value"], mean, rtol=1e-9)
    np.testing.assert_allclose(trace["var"]["value"], var, rtol=1e-9)


# --- transforms ------------------------------------------------------------------


def test_matrix_free_jit_and_vmap_equal_eager() -> None:
    """CG on a matrix-free prior under ``jit`` and ``jit(vmap)`` over datasets."""
    X, y, kernel = _smooth_problem(40)
    prior = GPPrior(kernel, X, solver=_CG, matrix_free=True)
    strategy = LaplaceInference()
    datasets = jnp.stack([y, 1.0 - y, jnp.arange(y.shape[0]) % 2.0])

    # Close over the prior: its kernel registry does not survive a transform.
    def summary(labels):
        fit = strategy.fit(prior, BernoulliLikelihood(), labels, key=jr.key(0))
        return fit.q_mean, fit.log_marginal_approx, fit.converged

    jitted = jax.jit(summary)(y)
    batched = jax.jit(jax.vmap(summary))(datasets)
    for got, want in zip(jitted[:2], summary(y)[:2], strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)
    for row, labels in enumerate(datasets):
        eager = summary(labels)
        assert bool(batched[2][row]) and bool(eager[2])
        # a batched CG keeps iterating rows that already converged, so the
        # rows agree to the loop's tolerance rather than to round-off
        for got, want in zip(batched[:2], eager[:2], strict=True):
            np.testing.assert_allclose(got[row], want, rtol=1e-6, atol=1e-9)


def _evidence_and_convergence(strategy, X, y, *, matrix_free=False, solver=None):
    def evidence(lengthscale):
        kernel = Matern(nu=1.5, init_lengthscale=lengthscale, init_variance=1.0)
        prior = GPPrior(kernel, X, solver=solver, matrix_free=matrix_free)
        fit = strategy.fit(prior, BernoulliLikelihood(), y)
        return fit.log_marginal_approx, fit.converged

    return evidence


@pytest.mark.parametrize(
    "name",
    ["dense", "cg", pytest.param("gauss-newton-cg", marks=pytest.mark.slow)],
)
def test_gradient_matches_finite_differences(name) -> None:
    """d log_marginal_approx / d lengthscale, implicit at the mode.

    CG solves the loop, the tangents and the evidence's quadratic form; an
    exact log-determinant keeps the evidence deterministic.
    """
    X, y, _ = _smooth_problem(40)
    cls = GaussNewtonInference if name.startswith("gauss-newton") else LaplaceInference
    cg = gx.CGSolver(rtol=1e-12, atol=1e-12, max_steps=500)
    if name == "dense":
        strategy = cls(max_iter=50, tol=1e-10)
    else:
        strategy = cls(max_iter=50, tol=1e-10, solver=cg, logdet=gx.DenseLogdet())
    evidence = _evidence_and_convergence(strategy, X, y)
    (_, converged), grad = jax.value_and_grad(evidence, has_aux=True)(0.3)
    assert bool(converged)
    step = 1e-5
    central = (evidence(0.3 + step)[0] - evidence(0.3 - step)[0]) / (2.0 * step)
    # central differences carry O(step²) truncation and ~1e-10 / step round-off
    np.testing.assert_allclose(grad, central, rtol=1e-4)


@pytest.mark.slow
def test_iterative_gradient_matches_dense_at_200() -> None:
    """N = 200: the CG gradient with an exact log-determinant equals the dense one.

    The issue's tolerance is rtol 1e-3; CG at 1e-12 lands far inside it.
    """
    X, y, _ = _smooth_problem(200)
    cg = gx.CGSolver(rtol=1e-12, atol=1e-12, max_steps=1000)
    dense = _evidence_and_convergence(LaplaceInference(max_iter=50, tol=1e-10), X, y)
    iterative = _evidence_and_convergence(
        LaplaceInference(max_iter=50, tol=1e-10, solver=cg, logdet=gx.DenseLogdet()),
        X,
        y,
    )
    (_, dense_ok), dense_grad = jax.value_and_grad(dense, has_aux=True)(0.3)
    (_, cg_ok), cg_grad = jax.value_and_grad(iterative, has_aux=True)(0.3)
    assert bool(dense_ok) and bool(cg_ok)
    np.testing.assert_allclose(cg_grad, dense_grad, rtol=1e-3)


@pytest.mark.parametrize("x64", [True, False])
def test_float32_matrix_free_stays_finite(x64) -> None:
    """The quasi-Newton notebook's float32 problem, by CG on a matrix-free prior."""
    with jax.enable_x64(x64):
        X = jnp.linspace(-3.0, 3.0, 60, dtype=jnp.float32)[:, None]
        y = (jnp.sin(2.0 * X[:, 0]) > 0.0).astype(jnp.float32)
        kernel = RBF(init_lengthscale=0.6, init_variance=1.5)
        solver = gx.CGSolver(rtol=1e-6, atol=1e-6, max_steps=200, throw=False)
        prior = GPPrior(kernel, X, solver=solver, matrix_free=True)
        fit = LaplaceInference(max_iter=50).fit(
            prior, BernoulliLikelihood(), y, key=jr.key(0)
        )
        q_var = fit.q_var
        mean, var = fit.predict(X[:3] + 0.01)
    assert fit.q_mean.dtype == q_var.dtype == jnp.float32
    for value in (fit.q_mean, q_var, fit.log_marginal_approx, mean, var):
        assert bool(jnp.all(jnp.isfinite(value)))
