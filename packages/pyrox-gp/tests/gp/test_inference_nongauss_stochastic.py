"""Stochastic marginal variances of the non-Gaussian fits (gaussx#827).

Until gaussx v0.8.4, ``variance="stochastic"`` (and ``"auto"`` with an
iterative solver above N = 2048) raised ``NotImplementedError`` in
`PosteriorLinearization` and `ExpectationPropagation`, and `LaplaceInference`
and `GaussNewtonInference` had only exact variances. Now all four take the
estimate of `gaussx.SitePosterior.variance_and_error`: SBPV on a dense prior,
SPV (products with K only) on a matrix-free one, with the cavity clamps of
pyrox#552 in the two cavity loops.
"""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.scipy.special import ndtri
from pyrox_gp import (
    BernoulliLikelihood,
    ExpectationPropagation,
    GaussNewtonInference,
    GPPrior,
    LaplaceInference,
    Matern,
    PosteriorLinearization,
)
from pyrox_gp._inference_nongauss import _clamp_cavity_variance, _MarginalVariance


_CLASSES = {
    "pl": PosteriorLinearization,
    "ep": ExpectationPropagation,
    "laplace": LaplaceInference,
    "gn": GaussNewtonInference,
}
_CG = gx.CGSolver(rtol=1e-10, atol=1e-10, max_steps=500)


def _problem(n: int = 200, *, matrix_free: bool = False):
    X = jr.uniform(jr.key(0), (n, 2), dtype=jnp.float64)
    y = (jnp.sin(4.0 * X[:, 0]) + X[:, 1] > 0.6).astype(jnp.float64)
    return GPPrior(Matern(nu=1.5), X, matrix_free=matrix_free), y


def _sem_bound(standard_error, size: int):
    """z SE, with z from a Bonferroni family-wise level of 1e-3 over the N points.

    Each estimate is a mean over ``num_probes`` independent draws, so its
    error is about normal with the returned standard error; the probability
    that any of the N points leaves the band is then at most 1e-3. The
    absolute 1e-8 covers the CG solves at 1e-10, far below the sampling
    error.
    """
    z = float(ndtri(1.0 - 1e-3 / (2.0 * size)))
    return z * standard_error + 1e-8


def _fit(name, prior, y, **settings):
    strategy = _CLASSES[name](solver=_CG, variance="stochastic", **settings)
    return strategy.fit(prior, BernoulliLikelihood(), y, key=jr.key(7))


@pytest.mark.parametrize(
    ("name", "matrix_free"),
    [
        ("pl", False),
        ("laplace", True),
        pytest.param("ep", False, marks=pytest.mark.slow),
        pytest.param("gn", False, marks=pytest.mark.slow),
        pytest.param("pl", True, marks=pytest.mark.slow),
        pytest.param("ep", True, marks=pytest.mark.slow),
    ],
)
def test_stochastic_variances_within_their_standard_error(name, matrix_free) -> None:
    """q_var is the estimate of gaussx, and within z SE of the exact variances."""
    prior, y = _problem(matrix_free=matrix_free)
    fit = _fit(name, prior, y, num_probes=200)
    marginal = fit.marginal_variance
    assert marginal.method == "stochastic"
    # SBPV on a dense prior, SPV (no prior draws) on a matrix-free one
    assert marginal.estimator == ("spv" if matrix_free else "sbpv")
    posterior = fit.site_posterior
    estimate, error = posterior.variance_and_error(
        num_probes=200, estimator=marginal.estimator, key=marginal.key
    )
    exact = posterior.variance(method="exact")
    assert bool(jnp.all(jnp.abs(estimate - exact) <= _sem_bound(error, y.shape[0])))
    np.testing.assert_array_equal(fit.q_var, jnp.maximum(estimate, 0.0))


@pytest.mark.parametrize("name", ["pl", pytest.param("ep", marks=pytest.mark.slow)])
def test_stochastic_fit_mean_within_two_percent(name) -> None:
    """pyrox#552's bar: N = 200 Bernoulli, 200 probes, mean within 2 percent.

    The sites see clamped SBPV variances at every iteration, from one key,
    so the loop still converges, and the posterior mean is that of the
    exact-variance fit to 2 percent of its largest entry (0.23 percent for
    both measured on this problem).
    """
    prior, y = _problem()
    exact = _CLASSES[name](solver=_CG, variance="exact").fit(
        prior, BernoulliLikelihood(), y
    )
    stochastic = _fit(name, prior, y, num_probes=200)
    assert bool(exact.converged) and bool(stochastic.converged)
    gap = jnp.max(jnp.abs(stochastic.q_mean - exact.q_mean))
    assert float(gap) <= 0.02 * float(jnp.max(jnp.abs(exact.q_mean)))


def test_cavity_variance_is_clamped() -> None:
    """Both clamps of a stochastic cavity variance (pyrox#552).

    Above: at most (1 - 1e-3) / lambda, so the cavity precision stays
    positive. Below: at least 1e-6 [K]_nn. Where they cross, the upper wins.
    """
    precision = jnp.array([2.0, 2.0, 2.0, 1e8])
    prior_variance = jnp.ones(4)
    estimate = jnp.array([0.9, -0.3, 0.2, 0.1])
    clamped = _clamp_cavity_variance(estimate, precision, prior_variance)
    np.testing.assert_allclose(clamped, [(1.0 - 1e-3) / 2.0, 1e-6, 0.2, 0.999e-8])
    cavity_mean, cavity_variance = gx.cavity_distribution(
        jnp.zeros(4), clamped, jnp.zeros(4), precision
    )
    assert bool(jnp.all(jnp.isfinite(cavity_mean)))
    assert bool(jnp.all(cavity_variance > 0.0))


def test_the_cavity_route_clamps_a_poor_estimate() -> None:
    """Two SPV probes on strong sites: raw estimates leave the band, cavities do not."""
    prior, _ = _problem(60)
    operator = prior._prior_operator()
    precision = jnp.full(60, 50.0)
    posterior = gx.site_posterior(
        jnp.zeros(60), operator, jnp.zeros(60), precision, solver=_CG
    )
    marginal = _MarginalVariance(
        method="stochastic", estimator="spv", num_probes=2, key=jr.key(0)
    )
    prior_variance = jnp.diag(operator.as_matrix())
    raw, _ = posterior.variance_and_error(num_probes=2, estimator="spv", key=jr.key(0))
    assert bool(jnp.any((raw >= 1.0 / precision) | (raw <= 0.0)))
    cavity = marginal.cavity(posterior, prior_variance)
    assert bool(jnp.all(cavity < 1.0 / precision))
    assert bool(jnp.all(cavity >= 1e-6 * prior_variance))


@pytest.mark.parametrize("name", ["pl", "ep"])
def test_the_cavity_loops_need_a_key(name) -> None:
    """Without a key the probes cannot be drawn: raised before any work."""
    prior, y = _problem(8)
    strategy = _CLASSES[name](solver=_CG, variance="stochastic")
    with pytest.raises(ValueError, match=r"PRNG key.*variance='exact'"):
        strategy.fit(prior, BernoulliLikelihood(), y)
    with pytest.raises(ValueError, match="estimator must be"):
        _CLASSES[name](estimator="love")


def test_laplace_reads_the_key_only_for_q_var() -> None:
    """The Newton loop needs no variances, so only reading q_var needs a key."""
    prior, y = _problem(8)
    fit = LaplaceInference(solver=_CG, variance="stochastic").fit(
        prior, BernoulliLikelihood(), y
    )
    assert bool(jnp.all(jnp.isfinite(fit.q_mean)))
    with pytest.raises(ValueError, match="PRNG key"):
        fit.q_var
    with pytest.raises(ValueError, match="variance must be"):
        GaussNewtonInference(variance="sampled")


def test_defaults_stay_exact() -> None:
    """A dense solver, or N <= 2048, keeps the exact variances and needs no key."""
    prior, y = _problem(30)
    for name in _CLASSES:
        for solver in (None, _CG):
            fit = _CLASSES[name](solver=solver).fit(prior, BernoulliLikelihood(), y)
            assert fit.marginal_variance.method == "exact"
            assert fit.marginal_variance.key is None


def test_jit_equals_eager() -> None:
    """The stochastic PL fit traces, with the key as an argument."""
    prior, y = _problem(40)
    strategy = PosteriorLinearization(solver=_CG, variance="stochastic", num_probes=50)

    def summary(labels, key):
        fit = strategy.fit(prior, BernoulliLikelihood(), labels, key=key)
        return fit.q_mean, fit.q_var, fit.log_marginal_approx

    eager = summary(y, jr.key(1))
    jitted = jax.jit(summary)(y, jr.key(1))
    for got, want in zip(jitted, eager, strict=True):
        # the same probes and solves, compiled: round-off only
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


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


@pytest.mark.parametrize("name", ["pl", "ep", "laplace"])
def test_matrix_free_auto_turns_stochastic_without_n_by_n(name) -> None:
    """N = 2100 > 2048, CG, variance="auto": SPV, and no (N, N) array anywhere."""
    n = 2100
    X = jnp.linspace(-3.0, 3.0, n)[:, None]
    y = (X[:, 0] > 0.0).astype(jnp.float64)
    prior = GPPrior(
        Matern(nu=1.5), X, solver=gx.CGSolver(max_steps=50), matrix_free=True
    )
    strategy = _CLASSES[name](num_probes=8, max_iter=2)

    def fit(labels):
        result = strategy.fit(prior, BernoulliLikelihood(), labels, key=jr.key(0))
        assert result.marginal_variance.method == "stochastic"
        assert result.marginal_variance.estimator == "spv"
        return result.q_mean, result.q_var

    shapes = {tuple(aval.shape) for aval in _avals(jax.make_jaxpr(fit)(y).jaxpr)}
    assert (n,) in shapes
    assert not any(len(shape) >= 2 and shape[-2:] == (n, n) for shape in shapes)
