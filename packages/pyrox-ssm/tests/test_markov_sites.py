"""The Markov track's per-site numerics on gaussx.

`pyrox_ssm._markov_sites` adapts a pyrox likelihood to gaussx's per-site
interface. Each test checks it against a closed form or a hand-built
Gauss-Hermite grid; the import boundary itself is `test_ssm_import_guard.py`.
Where the reference is the same arithmetic in a different order, the
tolerances are float64 round-off.
"""

from __future__ import annotations

import gaussx
import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from pyrox.likelihoods import GaussianLikelihood, Likelihood
from pyrox_ssm._markov_sites import (
    _cavity_site_derivatives,
    _pushforward_moments,
    as_gaussx_likelihood,
)
from ssm_reference import (
    BernoulliLikelihood,
    MultiLatentLikelihood,
    PoissonLikelihood,
    StudentTLikelihood,
    site_expectation,
)


# --- the adapter against closed forms ----------------------------------------


SCALAR_LIKELIHOODS = {
    "bernoulli": (BernoulliLikelihood(), jnp.array([1.0, 0.0, 1.0, 1.0, 0.0])),
    "poisson": (PoissonLikelihood(), jnp.array([0.0, 2.0, 1.0, 5.0, 3.0])),
    "student_t": (
        StudentTLikelihood(df=3.0, scale=0.5),
        jnp.array([0.3, -1.2, 2.0, 0.0, -0.4]),
    ),
    "gaussian": (
        GaussianLikelihood(noise_var=0.3),
        jnp.array([0.3, -1.2, 2.0, 0.0, -0.4]),
    ),
}
F = jnp.array([-0.8, 0.1, 0.6, 1.3, -0.2])
VAR = jnp.array([0.2, 0.5, 1.0, 1.5, 0.05])


def _closed_form_grad_hess(name, y):
    """Per-site d/df and d²/df² of log p(yᵢ | fᵢ) at F, by hand."""
    if name == "bernoulli":
        p = jax.nn.sigmoid(F)
        return y - p, -p * (1.0 - p)
    if name == "poisson":
        return y - jnp.exp(F), -jnp.exp(F)
    if name == "student_t":
        nu, s2, r = 3.0, 0.25, y - F
        return (nu + 1.0) * r / (nu * s2 + r**2), (nu + 1.0) * (r**2 - nu * s2) / (
            nu * s2 + r**2
        ) ** 2
    return (y - F) / 0.3, jnp.full_like(F, -1.0 / 0.3)


def _hermite_ell(log_density, y):
    """∑ᵢ 𝔼[ℓ(fᵢ, yᵢ)] for fᵢ ~ N(Fᵢ, VARᵢ) on a 20-node grid built by hand."""
    points, weights = gaussx.gauss_hermite_points(20, dim=1)
    nodes = F[None] + jnp.sqrt(VAR)[None] * points[:, 0][:, None]
    per_site = jax.vmap(jax.vmap(log_density))(nodes, jnp.broadcast_to(y, nodes.shape))
    return jnp.sum((weights / jnp.sqrt(2.0 * jnp.pi)) @ per_site)


@pytest.mark.parametrize("name", sorted(SCALAR_LIKELIHOODS))
def test_site_derivatives_match_the_closed_form(name) -> None:
    lik, y = SCALAR_LIKELIHOODS[name]
    g_ref, h_ref = _closed_form_grad_hess(name, y)
    g, h = as_gaussx_likelihood(lik, y).site_derivatives(F)
    assert jnp.allclose(g, g_ref, rtol=1e-6, atol=1e-12)
    assert jnp.allclose(h, h_ref, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("name", sorted(SCALAR_LIKELIHOODS))
def test_log_prob_matches_the_summed_likelihood(name) -> None:
    lik, y = SCALAR_LIKELIHOODS[name]
    got = as_gaussx_likelihood(lik, y).log_prob(F)
    assert jnp.allclose(got, lik.log_prob(F, y), rtol=1e-12)


@pytest.mark.parametrize("name", sorted(SCALAR_LIKELIHOODS))
def test_expected_log_likelihood_matches_a_hermite_grid(name) -> None:
    lik, y = SCALAR_LIKELIHOODS[name]
    rule = gaussx.GaussHermiteIntegrator(order=20)
    state = gaussx.GaussianState(F, lx.DiagonalLinearOperator(VAR))
    got = gaussx.expected_log_likelihood(as_gaussx_likelihood(lik, y), state, rule)
    expected = _hermite_ell(lambda f, y_i: lik.log_prob(f[None], y_i[None]), y)
    assert jnp.allclose(got, expected, rtol=1e-6)


class _InputScaledGaussian(Likelihood):
    """A Gaussian whose noise variance is the input's squared norm."""

    def log_prob(self, f, y, X=None):
        var = jnp.sum(X**2) if X is not None else 1.0
        return jnp.sum(-0.5 * (y - f) ** 2 / var - 0.5 * jnp.log(2 * jnp.pi * var))


class _LegacyGaussian(Likelihood):
    """A likelihood written against the pre-gh-204 ``log_prob(f, y)``."""

    def log_prob(self, f, y):
        return jnp.sum(-0.5 * (y - f) ** 2 - 0.5 * jnp.log(2 * jnp.pi))


@pytest.mark.parametrize("lik", [_InputScaledGaussian(), _LegacyGaussian()])
def test_inputs_reach_only_a_likelihood_that_takes_them(lik) -> None:
    y = jnp.array([0.3, -1.2, 2.0, 0.0, -0.4])
    X = jnp.linspace(0.5, 2.0, 10).reshape(5, 2)
    rule = gaussx.GaussHermiteIntegrator(order=20)
    state = gaussx.GaussianState(F, lx.DiagonalLinearOperator(VAR))
    got = gaussx.expected_log_likelihood(as_gaussx_likelihood(lik, y, X), state, rule)
    # Closed form: 𝔼[log N(y; f, s²)] = −½ log(2π s²) − ((y − m)² + v) / 2s²,
    # with s² = ‖Xᵢ‖² when the likelihood takes inputs and 1 otherwise.
    s2 = jnp.sum(X**2, axis=1) if isinstance(lik, _InputScaledGaussian) else 1.0
    expected = jnp.sum(
        -0.5 * jnp.log(2 * jnp.pi * s2) - ((y - F) ** 2 + VAR) / (2 * s2)
    )
    assert jnp.allclose(got, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "lik", [MultiLatentLikelihood(latent_dim=3), MultiLatentLikelihood(latent_dim=2)]
)
def test_adapter_rejects_a_multi_latent_likelihood(lik) -> None:
    with pytest.raises(ValueError, match="latent_dim"):
        as_gaussx_likelihood(lik, jnp.zeros(4))


@pytest.mark.parametrize(
    "rule",
    [gaussx.GaussHermiteIntegrator(order=20), gaussx.UnscentedIntegrator(alpha=1.0)],
    ids=["hermite", "unscented"],
)
@pytest.mark.parametrize("name", ["bernoulli", "poisson", "student_t"])
def test_cavity_site_derivatives_match_the_per_site_expectations(name, rule) -> None:
    """Bonnet and Price on the ELL equal integrating the site derivatives directly."""
    lik, y = SCALAR_LIKELIHOODS[name]

    def lp(f_n, y_n):
        return lik.log_prob(f_n[None], y_n[None])

    grad_lp = jax.grad(lp)
    g_ref = site_expectation(rule, grad_lp, F, VAR, y)
    h_ref = site_expectation(rule, jax.grad(grad_lp), F, VAR, y)
    g, h = _cavity_site_derivatives(as_gaussx_likelihood(lik, y), F, VAR, rule)
    assert jnp.allclose(g, g_ref, rtol=1e-6, atol=1e-12)
    assert jnp.allclose(h, h_ref, rtol=1e-6, atol=1e-12)


def test_pushforward_moments_match_the_lognormal() -> None:
    mean = jnp.array([[-0.5, 0.2], [1.0, 3.0]])
    var = jnp.array([[0.3, 1.0], [0.01, 2.0]])
    got = _pushforward_moments(jnp.exp, mean, var, order=30)
    # exp(f) for f ~ N(μ, σ²) is lognormal: mean e^{μ+σ²/2}, var (e^{σ²}−1) e^{2μ+σ²}
    expected_mean = jnp.exp(mean + var / 2)
    expected_var = jnp.expm1(var) * jnp.exp(2 * mean + var)
    assert jnp.allclose(got[0], expected_mean, rtol=1e-10)
    assert jnp.allclose(got[1], expected_var, rtol=1e-10)
