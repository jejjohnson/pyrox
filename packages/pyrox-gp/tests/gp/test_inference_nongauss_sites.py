"""Posterior linearisation and EP on the site posterior and the site loop (#552).

`PosteriorLinearization` and `ExpectationPropagation` used to be eager Python
loops (``if delta < tol: break``), which ``jax.jit`` rejected with
``TracerBoolConversionError``. They now run on `gaussx.site_posterior` and
`argminax.solve.site_fixed_point`, take ``solver``, ``preconditioner`` and
``logdet`` (falling back to the prior's), and accept a matrix-free prior.
``data/eager_cavity_fits.npz`` froze the eager fits they must still reach.
"""

from __future__ import annotations

import importlib.util
import pathlib

import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from pyrox_gp import (
    RBF,
    BernoulliLikelihood,
    ExpectationPropagation,
    GPPrior,
    Matern,
    PosteriorLinearization,
)


_DATA = pathlib.Path(__file__).parent / "data"
_spec = importlib.util.spec_from_file_location(
    "make_eager_cavity_fixtures", _DATA / "make_eager_cavity_fixtures.py"
)
assert _spec is not None and _spec.loader is not None
_make = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_make)
_FROZEN = np.load(_DATA / "eager_cavity_fits.npz")

_CLASSES = {"pl": PosteriorLinearization, "ep": ExpectationPropagation}


def _classification(n: int = 12):
    X = jnp.linspace(-3.0, 3.0, n)[:, None]
    y = (jnp.sin(X[:, 0]) > 0.0).astype(jnp.float64)
    return GPPrior(kernel=RBF(init_lengthscale=1.0, init_variance=1.0), X=X), y


# --- the eager fixed points ------------------------------------------------


def _fit_case(case, **settings):
    strategy, likelihood = case
    prior, lik, y = _make.problem(likelihood)
    return _CLASSES[strategy](**settings).fit(prior, lik, y)


def _case_params():
    return [
        pytest.param(case, marks=() if case[1] == "bernoulli" else pytest.mark.slow)
        for case in _make.CASES
    ]


def _ids(case) -> str:
    return "-".join(case)


@pytest.mark.parametrize("case", _case_params(), ids=_ids)
def test_same_iterates_as_the_eager_loop(case) -> None:
    """Run as many iterations as the eager loop ran, and land where it landed.

    The site update and its first iteration (cavities from the prior) are
    those of the eager loop, so only the stopping test differs: with the
    budget capped at the eager loop's iteration count both run the same
    iterates, and agree to round-off.
    """
    key = "__".join(case)
    fit = _fit_case(case, max_iter=int(_FROZEN[f"{key}__n_iter"]))
    for field in _make.FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(fit, field)),
            _FROZEN[f"{key}__{field}"],
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"{key}: {field}",
        )


@pytest.mark.parametrize("case", _case_params(), ids=_ids)
def test_defaults_converge_at_least_as_close_as_the_eager_loop(case) -> None:
    """The defaults converge, a few iterations later, nearer the fixed point.

    The eager loops stopped once the posterior mean moved by less than ``tol``
    in one iteration, which a damped map with contraction rate r does while
    still ``tol / (1 - r)`` from its fixed point; on Poisson EP that left
    ``q_mean`` 2.8e-4 away. The site test (every natural within a tenth of
    ``tol``) runs at most 7 more iterations on these problems, and every field
    lands at least as close to the fixed point (here a fit at ``tol=1e-10``).
    """
    key = "__".join(case)
    fit = _fit_case(case)
    fixed_point = _fit_case(case, max_iter=300, tol=1e-10)
    assert bool(fit.converged) and bool(fixed_point.converged)
    if bool(_FROZEN[f"{key}__converged"]):
        extra = int(fit.n_iter) - int(_FROZEN[f"{key}__n_iter"])
        assert 0 <= extra <= 7, extra
    for field in _make.FIELDS:
        target = np.asarray(getattr(fixed_point, field))
        new = np.max(np.abs(np.asarray(getattr(fit, field)) - target))
        old = np.max(np.abs(_FROZEN[f"{key}__{field}"] - target))
        assert new <= old + 1e-12, (field, new, old)


# --- jit and vmap ---------------------------------------------------------


@pytest.mark.parametrize("name", ["pl", pytest.param("ep", marks=pytest.mark.slow)])
def test_jit_and_vmap_equal_eager(name) -> None:
    """``jax.jit(jax.vmap(fit))`` over datasets equals the eager fits.

    Before #552 ``jax.jit`` of either fit raised ``TracerBoolConversionError``.
    """
    strategy = _CLASSES[name](max_iter=60)
    prior, y = _classification()
    likelihood = BernoulliLikelihood()
    datasets = jnp.stack([y, 1.0 - y, jnp.arange(y.shape[0]) % 2.0])

    # Close over the prior: its kernel registry does not survive a transform.
    def summary(labels):
        fit = strategy.fit(prior, likelihood, labels)
        return fit.q_mean, fit.q_var, fit.log_marginal_approx, fit.converged

    jitted = jax.jit(summary)(y)
    batched = jax.jit(jax.vmap(summary))(datasets)
    for row, labels in enumerate(datasets):
        eager = summary(labels)
        assert bool(batched[3][row]) and bool(eager[3])
        for got, want in zip(batched[:3], eager[:3], strict=True):
            assert jnp.allclose(got[row], want, rtol=1e-9, atol=1e-12)
    for got, want in zip(jitted[:3], summary(y)[:3], strict=True):
        assert jnp.allclose(got, want, rtol=1e-9, atol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["pl", "ep"])
def test_gradient_matches_finite_differences(name) -> None:
    """d log_marginal_approx / d lengthscale, implicit at the fixed point."""
    strategy = _CLASSES[name](max_iter=200, tol=1e-10)
    X = jnp.linspace(-3.0, 3.0, 12)[:, None]
    y = (jnp.sin(X[:, 0]) > 0.0).astype(jnp.float64)

    def evidence(lengthscale):
        prior = GPPrior(kernel=RBF(init_lengthscale=lengthscale), X=X)
        return strategy.fit(prior, BernoulliLikelihood(), y).log_marginal_approx

    grad = jax.grad(evidence)(1.0)
    step = 1e-5
    central = (evidence(1.0 + step) - evidence(1.0 - step)) / (2.0 * step)
    # central differences carry O(step²) truncation and ~1e-10 / step round-off
    np.testing.assert_allclose(grad, central, rtol=1e-4)


# --- the solver choice -----------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("name", ["pl", "ep"])
def test_iterative_solver_matches_dense(name) -> None:
    """CG on a dense prior, and on a matrix-free one, reach the dense fit."""
    X = jr.uniform(jr.key(0), (40, 2), dtype=jnp.float64)
    y = (X[:, 0] > 0.5).astype(jnp.float64)
    cg = gx.CGSolver(rtol=1e-10, atol=1e-10, max_steps=400)
    # CG's own log-determinant is stochastic Lanczos; an exact one keeps the
    # evidence comparable to the dense fit
    exact = gx.DenseLogdet()
    dense = GPPrior(Matern(nu=1.5), X)
    reference = _CLASSES[name]().fit(dense, BernoulliLikelihood(), y)
    on_dense = _CLASSES[name](solver=cg, logdet=exact).fit(
        dense, BernoulliLikelihood(), y
    )
    matrix_free = GPPrior(Matern(nu=1.5), X, solver=cg, logdet=exact, matrix_free=True)
    # the strategy's fields are None, so it inherits the prior's
    on_operator = _CLASSES[name]().fit(matrix_free, BernoulliLikelihood(), y)
    assert isinstance(on_operator.solver, gx.ComposedSolver)
    X_star = X[:5] + 0.01
    ref_mean, ref_var = reference.predict(X_star)
    for fit in (on_dense, on_operator):
        assert bool(fit.converged)
        # CG at 1e-10 against a Cholesky: the sites agree far inside 1e-6
        np.testing.assert_allclose(fit.q_mean, reference.q_mean, rtol=1e-6)
        np.testing.assert_allclose(fit.q_var, reference.q_var, rtol=1e-6)
        np.testing.assert_allclose(
            fit.log_marginal_approx, reference.log_marginal_approx, rtol=1e-8
        )
        mean, var = fit.predict(X_star)
        np.testing.assert_allclose(mean, ref_mean, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(var, ref_var, rtol=1e-6)


def test_strategy_fields_override_the_prior() -> None:
    """A strategy's solver overrides the prior's, field by field."""
    X = jnp.linspace(-3.0, 3.0, 8)[:, None]
    y = (X[:, 0] > 0.0).astype(jnp.float64)
    cg = gx.CGSolver(rtol=1e-10, atol=1e-10)
    prior = GPPrior(RBF(), X, solver=cg)
    inherited = PosteriorLinearization(max_iter=3).fit(prior, BernoulliLikelihood(), y)
    assert inherited.solver is cg
    dense = PosteriorLinearization(max_iter=3, solver=gx.DenseSolver()).fit(
        prior, BernoulliLikelihood(), y
    )
    # a dense fit predicts by its Cholesky factor, as before
    assert dense.solver is None
    np.testing.assert_allclose(inherited.q_mean, dense.q_mean, rtol=1e-7)
    np.testing.assert_allclose(inherited.q_var, dense.q_var, rtol=1e-7)


@pytest.mark.parametrize("name", ["pl", "ep"])
def test_stochastic_variance_raises_naming_gaussx_827(name) -> None:
    """The stochastic branch waits on gaussx#827 and says so before any work."""
    X = jnp.linspace(-3.0, 3.0, 8)[:, None]
    y = (X[:, 0] > 0.0).astype(jnp.float64)
    prior = GPPrior(RBF(), X, solver=gx.CGSolver())
    strategy = _CLASSES[name](variance="stochastic")
    with pytest.raises(NotImplementedError, match=r"gaussx#827.*variance='exact'"):
        strategy.fit(prior, BernoulliLikelihood(), y)
    # "auto" turns stochastic above N = 2048 with an iterative solver
    big = jnp.linspace(-3.0, 3.0, 2049)[:, None]
    large = GPPrior(RBF(), big, solver=gx.CGSolver(), matrix_free=True)
    with pytest.raises(NotImplementedError, match="gaussx#827"):
        _CLASSES[name]().fit(large, BernoulliLikelihood(), (big[:, 0] > 0.0) * 1.0)
    with pytest.raises(ValueError, match="variance must be"):
        _CLASSES[name](variance="sampled")


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


def test_matrix_free_exact_variance_never_forms_n_by_n() -> None:
    """N = 2100 > 2048, CG, exact variances in batches: no (N, N) array."""
    n = 2100
    X = jnp.linspace(-3.0, 3.0, n)[:, None]
    y = (X[:, 0] > 0.0).astype(jnp.float64)
    prior = GPPrior(RBF(), X, solver=gx.CGSolver(max_steps=50), matrix_free=True)
    strategy = PosteriorLinearization(max_iter=2, variance="exact", batch_size=16)

    def fit(labels):
        return strategy.fit(prior, BernoulliLikelihood(), labels).q_var

    shapes = {tuple(aval.shape) for aval in _avals(jax.make_jaxpr(fit)(y).jaxpr)}
    assert (n,) in shapes
    assert not any(len(shape) >= 2 and shape[-2:] == (n, n) for shape in shapes)
    # the variance solves run 16 right-hand sides at a time
    assert any(shape[:1] == (16,) and n in shape for shape in shapes)
