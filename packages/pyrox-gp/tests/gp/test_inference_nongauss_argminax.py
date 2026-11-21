"""The Newton strategies on argminax and optimistix (pyrox#346).

`LaplaceInference` and `GaussNewtonInference` run their loop on
``argminax.solve.laplace_mode_covariance`` and `QuasiNewtonInference` on
``optimistix.LBFGS``. The reference below is the eager Python loop they
replaced, kept here so the port is checked against the algorithm itself.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from gaussx import newton_update
from pyrox_gp import (
    RBF,
    BernoulliLikelihood,
    GaussNewtonInference,
    GPPrior,
    LaplaceInference,
    PoissonLikelihood,
    QuasiNewtonInference,
    StudentTLikelihood,
)
from pyrox_gp._inference_nongauss import (
    _laplace_log_marginal,
    _log_prob_per_point_factory,
    _per_point_grad_hess,
    _posterior_from_diag_sites,
    _prior_K,
    _site_data,
)


def _classification(n: int = 12):
    X = jnp.linspace(-3.0, 3.0, n)[:, None]
    y = (jnp.sin(X[:, 0]) > 0.0).astype(jnp.float64)
    return GPPrior(kernel=RBF(), X=X), y


def _counts(n: int = 12):
    X = jnp.linspace(-3.0, 3.0, n)[:, None]
    y = jnp.round(jnp.exp(jnp.sin(X[:, 0]) + 0.5))
    return GPPrior(kernel=RBF(), X=X), y


def _outlier_regression(n: int = 20):
    X = jnp.linspace(-3.0, 3.0, n)[:, None]
    y = jnp.sin(X[:, 0]).at[10].set(10.0)
    return GPPrior(kernel=RBF(), X=X), y


def _eager_newton(prior, likelihood, y, *, max_iter, tol, damping, floor):
    """The eager loop pyrox-gp ran before argminax.

    Returns ``(q_mean, q_var, log_marginal, n_iter)``.
    """
    K = _prior_K(prior)
    prior_mean = prior.mean(prior.X)
    log_prob = _log_prob_per_point_factory(likelihood)
    sites = _site_data(likelihood, y, prior.X)
    f = jnp.asarray(prior_mean)
    n_iter = 0
    for it in range(max_iter):
        g, h = _per_point_grad_hess(log_prob, f, sites)
        nat1, precision = newton_update(f, g, h, precision_floor=floor)
        f_newton, _ = _posterior_from_diag_sites(K, nat1, precision, prior_mean)
        f_new = (1.0 - damping) * f + damping * f_newton
        delta = jnp.max(jnp.abs(f_new - f))
        f = f_new
        n_iter = it + 1
        if delta < tol:
            break
    g, h = _per_point_grad_hess(log_prob, f, sites)
    nat1, precision = newton_update(f, g, h, precision_floor=floor)
    q_mean, q_var = _posterior_from_diag_sites(K, nat1, precision, prior_mean)
    log_marginal = _laplace_log_marginal(log_prob, f, sites, prior_mean, K, precision)
    return q_mean, q_var, log_marginal, n_iter


_PORT_CASES = {
    "laplace-bernoulli": (LaplaceInference, _classification, BernoulliLikelihood, 1e-6),
    "laplace-poisson": (LaplaceInference, _counts, PoissonLikelihood, 1e-6),
    "gauss-newton-bernoulli": (
        GaussNewtonInference,
        _classification,
        BernoulliLikelihood,
        1e-3,
    ),
    "gauss-newton-studentt": (
        GaussNewtonInference,
        _outlier_regression,
        lambda: StudentTLikelihood(df=3.0, scale=0.5),
        1e-3,
    ),
}


# A full Newton step oscillates on the Student-t outlier in both loops (neither
# converges in 200 steps at damping 1.0 or 0.8); at 0.5 both converge in 31.
_PORT_PARAMS = [
    (case, damping)
    for case in _PORT_CASES
    for damping in ((0.5,) if case.endswith("studentt") else (1.0, 0.8))
]


@pytest.mark.parametrize(("case", "damping"), _PORT_PARAMS)
def test_port_matches_the_eager_loop(case, damping):
    strategy_cls, problem, likelihood_factory, floor = _PORT_CASES[case]
    prior, y = problem()
    likelihood = likelihood_factory()
    fit = strategy_cls(max_iter=50, damping=damping).fit(prior, likelihood, y)
    q_mean, q_var, log_marginal, n_iter = _eager_newton(
        prior, likelihood, y, max_iter=50, tol=1e-6, damping=damping, floor=floor
    )
    assert bool(fit.converged)
    # Both stop on max |Δf| < 1e-6, so they agree far inside the issue's rtol 1e-5.
    assert jnp.allclose(fit.q_mean, q_mean, rtol=1e-5, atol=1e-10)
    assert jnp.allclose(fit.q_var, q_var, rtol=1e-5, atol=1e-10)
    assert jnp.allclose(fit.log_marginal_approx, log_marginal, rtol=1e-5)
    assert abs(int(fit.n_iter) - n_iter) <= 2


def test_laplace_equals_gauss_newton_on_bernoulli():
    prior, y = _classification()
    laplace = LaplaceInference().fit(prior, BernoulliLikelihood(), y)
    gauss_newton = GaussNewtonInference().fit(prior, BernoulliLikelihood(), y)
    assert jnp.allclose(laplace.q_mean, gauss_newton.q_mean, rtol=1e-6)
    assert jnp.allclose(laplace.q_var, gauss_newton.q_var, rtol=1e-6)


def test_gauss_newton_on_studentt_stays_finite():
    prior, y = _outlier_regression()
    fit = GaussNewtonInference(max_iter=30).fit(
        prior, StudentTLikelihood(df=3.0, scale=0.5), y
    )
    assert bool(jnp.all(jnp.isfinite(fit.q_mean)))
    assert bool(jnp.all(fit.q_var > 0.0))
    # The heavy tail discounts the outlier at index 10.
    assert fit.q_mean[10] < 5.0


def test_quasi_newton_reaches_the_laplace_mode():
    prior, y = _classification()
    likelihood = BernoulliLikelihood()
    laplace = LaplaceInference(max_iter=50).fit(prior, likelihood, y)
    quasi = QuasiNewtonInference(max_iter=200, tol=1e-10).fit(prior, likelihood, y)
    assert bool(quasi.converged)
    assert jnp.allclose(quasi.q_mean, laplace.q_mean, rtol=1e-4, atol=1e-8)
    assert jnp.allclose(
        quasi.log_marginal_approx, laplace.log_marginal_approx, rtol=1e-4
    )


def test_result_counters_are_array_leaves():
    prior, y = _classification()
    fit = LaplaceInference().fit(prior, BernoulliLikelihood(), y)
    leaves = jax.tree.leaves(fit)
    assert any(leaf is fit.n_iter for leaf in leaves)
    assert any(leaf is fit.converged for leaf in leaves)
    assert fit.n_iter.shape == () and fit.converged.dtype == jnp.bool_


_STRATEGIES = {
    "laplace": LaplaceInference(max_iter=30),
    "gauss-newton": GaussNewtonInference(max_iter=30),
    "quasi-newton": QuasiNewtonInference(max_iter=100),
}


@pytest.mark.parametrize(
    "name",
    [
        "laplace",
        pytest.param("gauss-newton", marks=pytest.mark.slow),
        pytest.param("quasi-newton", marks=pytest.mark.slow),
    ],
)
def test_fit_jits_and_vmaps_over_datasets(name):
    strategy = _STRATEGIES[name]
    prior, y = _classification()
    likelihood = BernoulliLikelihood()
    # Three label sets on the same inputs: the first, flipped, and alternating.
    datasets = jnp.stack([y, 1.0 - y, jnp.arange(y.shape[0]) % 2.0])

    # Close over the prior: its kernel registry does not survive a transform.
    def summary(labels):
        fit = strategy.fit(prior, likelihood, labels)
        return fit.q_mean, fit.q_var, fit.log_marginal_approx, fit.converged

    batched = jax.jit(jax.vmap(summary))(datasets)
    for row, labels in enumerate(datasets):
        eager = summary(labels)
        assert bool(batched[3][row]) and bool(eager[3])
        for got, want in zip(batched[:3], eager[:3], strict=True):
            assert jnp.allclose(got[row], want, rtol=1e-6, atol=1e-9)
