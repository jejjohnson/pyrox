"""`DistLikelihood` against the dedicated likelihood classes (gh-376).

`pyrox_gp.DistLikelihood` is the core `pyrox.likelihoods.DistLikelihood`;
its `build` returns the arguments of `gaussx.CallableLikelihood`, which
pyrox-gp's inference wraps (`_gaussx_likelihood`). A Bernoulli or Poisson
written as a NumPyro distribution, in the new three-argument form and the
old one-argument form, must give what `BernoulliLikelihood` and
`PoissonLikelihood` give: log densities, site derivatives, the SVGP ELL and
the dense fits. The two sides evaluate the same density through different
exact formulas (numpyro's against gaussx's), so they agree to float64
round-off: ``rtol=1e-10``, with a ``1e-12`` floor where a derivative crosses
zero.
"""

from __future__ import annotations

import gaussx as gx
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as nd
import pyrox.likelihoods
import pytest
from pyrox_gp import (
    RBF,
    BernoulliLikelihood,
    ConjugateVI,
    DistLikelihood,
    ExpectationPropagation,
    FullRankGuide,
    GaussNewtonInference,
    GPPrior,
    LaplaceInference,
    PoissonLikelihood,
    PosteriorLinearization,
    SparseGPPrior,
    svgp_elbo,
)
from pyrox_gp._inference import _ell_numerical
from pyrox_gp._inference_nongauss import (
    _log_prob_per_point_factory,
    _per_point_grad_hess,
    _site_data,
)
from pyrox_gp._likelihoods import _gaussx_likelihood


RTOL = 1e-10
N = 40


def _bernoulli(eta, theta, data):
    return nd.Bernoulli(logits=eta)


def _poisson(eta, theta, data):
    return nd.Poisson(rate=jnp.exp(eta))


def _cases():
    y_binary = jr.bernoulli(jr.key(0), 0.4, (N,)).astype(jnp.float64)
    y_count = jr.poisson(jr.key(1), 2.0, (N,)).astype(jnp.float64)
    return {
        "bernoulli": (DistLikelihood(_bernoulli), BernoulliLikelihood(), y_binary),
        "bernoulli_old_form": (
            DistLikelihood(lambda f: nd.Bernoulli(logits=f)),
            BernoulliLikelihood(),
            y_binary,
        ),
        "poisson": (DistLikelihood(_poisson), PoissonLikelihood(), y_count),
        "poisson_old_form": (
            DistLikelihood(lambda f: nd.Poisson(rate=jnp.exp(f))),
            PoissonLikelihood(),
            y_count,
        ),
    }


NAMES = sorted(_cases())
F = 2.0 * jr.normal(jr.key(2), (N,))
VAR = jnp.exp(jr.normal(jr.key(3), (N,)) - 1.0)


def test_pyrox_gp_name_is_the_core_object() -> None:
    assert DistLikelihood is pyrox.likelihoods.DistLikelihood


@pytest.mark.parametrize("name", NAMES)
def test_log_prob_equals_the_dedicated_class(name) -> None:
    dist_lik, dedicated, y = _cases()[name]
    assert jnp.allclose(
        dist_lik.log_prob(F, y), dedicated.log_prob(F, y), rtol=RTOL, atol=0.0
    )


@pytest.mark.parametrize("name", NAMES)
def test_the_wrapped_gaussx_likelihood_equals_the_dedicated_one(name) -> None:
    dist_lik, dedicated, y = _cases()[name]
    got = _gaussx_likelihood(dist_lik, y, {}, {})
    expected = dedicated.build(y, {}, {})
    assert isinstance(got, gx.CallableLikelihood)
    assert jnp.allclose(got.log_prob(F), expected.log_prob(F), rtol=RTOL, atol=0.0)
    for a, b in zip(got.site_derivatives(F), expected.site_derivatives(F), strict=True):
        assert jnp.allclose(a, b, rtol=RTOL, atol=1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_strategy_site_derivatives_equal_the_dedicated_class(name) -> None:
    """Laplace / GN / QN's per-site gradient and curvature."""
    dist_lik, dedicated, y = _cases()[name]
    X = jnp.zeros((N, 1))
    got = _per_point_grad_hess(
        _log_prob_per_point_factory(dist_lik), F, _site_data(dist_lik, y, X)
    )
    expected = _per_point_grad_hess(
        _log_prob_per_point_factory(dedicated), F, _site_data(dedicated, y, X)
    )
    for a, b in zip(got, expected, strict=True):
        assert jnp.allclose(a, b, rtol=RTOL, atol=1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_expected_log_likelihood_equals_the_dedicated_class(name) -> None:
    dist_lik, dedicated, y = _cases()[name]
    X = jr.normal(jr.key(4), (N, 2))
    rule = gx.GaussHermiteIntegrator(order=20)
    got = _ell_numerical(dist_lik, y, F, VAR, rule, X)
    expected = _ell_numerical(dedicated, y, F, VAR, rule, X)
    assert jnp.allclose(got, expected, rtol=RTOL, atol=0.0)


@pytest.mark.parametrize("name", NAMES)
def test_cvi_site_gradients_equal_the_dedicated_class(name) -> None:
    dist_lik, dedicated, y = _cases()[name]
    cvi = ConjugateVI(integrator=gx.GaussHermiteIntegrator(order=20))
    X = jnp.zeros((N, 1))
    got = cvi._site_gradients(dist_lik, y, F, VAR, X)
    expected = cvi._site_gradients(dedicated, y, F, VAR, X)
    for a, b in zip(got, expected, strict=True):
        assert jnp.allclose(a, b, rtol=RTOL, atol=1e-12)


def test_svgp_elbo_equals_the_dedicated_class() -> None:
    Z = jnp.linspace(-2.0, 2.0, 5)[:, None]
    X = jnp.linspace(-3.0, 3.0, N)[:, None]
    prior = SparseGPPrior(
        kernel=RBF(init_variance=1.0, init_lengthscale=0.5), Z=Z, jitter=1e-4
    )
    guide = FullRankGuide.init(num_inducing=5)
    dist_lik, dedicated, y = _cases()["poisson"]
    rule = gx.GaussHermiteIntegrator(order=20)
    got = svgp_elbo(prior, guide, dist_lik, X, y, integrator=rule)
    expected = svgp_elbo(prior, guide, dedicated, X, y, integrator=rule)
    assert jnp.allclose(got, expected, rtol=RTOL, atol=0.0)


def _classification_prior():
    X = jnp.linspace(-3.0, 3.0, 12)[:, None]
    y = (jnp.sin(X[:, 0]) > 0.0).astype(jnp.float64)
    return GPPrior(kernel=RBF(init_lengthscale=1.0, init_variance=1.0), X=X), y


# The Laplace fit is the issue's check and stays in the fast tier; the other
# dense strategies run eager loops of a few seconds each, so they are slow.
@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param(LaplaceInference(max_iter=20), id="laplace"),
        pytest.param(
            GaussNewtonInference(max_iter=5), id="gauss_newton", marks=pytest.mark.slow
        ),
        pytest.param(
            PosteriorLinearization(max_iter=3), id="pl", marks=pytest.mark.slow
        ),
        pytest.param(
            ExpectationPropagation(max_iter=3), id="ep", marks=pytest.mark.slow
        ),
    ],
)
@pytest.mark.parametrize("form", ["new", "old"])
def test_dense_fit_equals_bernoulli_likelihood(strategy, form) -> None:
    prior, y = _classification_prior()
    dist_lik = (
        DistLikelihood(_bernoulli)
        if form == "new"
        else DistLikelihood(lambda f: nd.Bernoulli(logits=f))
    )
    got = strategy.fit(prior, dist_lik, y)
    expected = strategy.fit(prior, BernoulliLikelihood(), y)
    assert jnp.allclose(got.q_mean, expected.q_mean, rtol=RTOL, atol=1e-12)
    assert jnp.allclose(got.q_var, expected.q_var, rtol=RTOL, atol=1e-12)
    assert jnp.allclose(
        got.log_marginal_approx, expected.log_marginal_approx, rtol=RTOL, atol=0.0
    )


def test_each_site_reads_its_input_from_data() -> None:
    """$y_i \\sim N(f_i, s_i^2)$, $s_i = 0.2 + 0.5 |x_i|$, from ``data["X"]``."""
    X = jnp.linspace(-2.0, 2.0, 9)[:, None]
    y = jnp.sin(X[:, 0])
    f = jnp.zeros_like(y)
    lik = DistLikelihood(
        lambda eta, theta, d: nd.Normal(eta, 0.2 + 0.5 * jnp.abs(d["X"][..., 0]))
    )
    std = 0.2 + 0.5 * jnp.abs(X[:, 0])
    g, h = _per_point_grad_hess(
        _log_prob_per_point_factory(lik), f, _site_data(lik, y, X)
    )
    # ∂ log N(y; f, s²)/∂f = (y - f) / s², ∂² = -1 / s²
    assert jnp.allclose(g, (y - f) / std**2, rtol=1e-12, atol=0.0)
    assert jnp.allclose(h, -1.0 / std**2, rtol=1e-12, atol=0.0)
    assert jnp.allclose(
        lik.log_prob(f, y, X), nd.Normal(f, std).log_prob(y).sum(), rtol=1e-12
    )
