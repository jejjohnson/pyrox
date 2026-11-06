"""Tests for ``pyrox_gp`` SVGP inference entry points.

Covers :func:`svgp_elbo`, :func:`svgp_factor`, and :class:`ConjugateVI`.
"""

from __future__ import annotations

import gaussx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpyro.distributions as nd
import pytest
from gaussx import GaussHermiteIntegrator, variational_elbo_gaussian
from numpyro import handlers
from pyrox_gp import (
    RBF,
    ConjugateVI,
    DistLikelihood,
    FullRankGuide,
    GaussianLikelihood,
    NaturalGuide,
    PoissonLikelihood,
    SparseGPPrior,
    svgp_elbo,
    svgp_factor,
)
from pyrox_gp._inference import _ell_numerical, _per_site_expectation
from pyrox_gp._protocols import Likelihood


def _toy(M: int = 5, N: int = 10, seed: int = 0):
    """Return a toy sparse GP setup."""
    Z = jnp.linspace(-2.0, 2.0, M).reshape(-1, 1)
    X = jnp.linspace(-3.0, 3.0, N).reshape(-1, 1)
    y = jnp.sin(X.squeeze(-1)) + 0.1 * jr.normal(jr.PRNGKey(seed), (N,))
    prior = SparseGPPrior(
        kernel=RBF(init_variance=1.0, init_lengthscale=0.5),
        Z=Z,
        jitter=1e-4,
    )
    return prior, X, y


# --- svgp_elbo: Gaussian path ----------------------------------------------


def test_svgp_elbo_gaussian_returns_finite_scalar():
    prior, X, y = _toy()
    guide = FullRankGuide.init(num_inducing=prior.num_inducing)
    lik = GaussianLikelihood(noise_var=0.1)
    elbo = svgp_elbo(prior, guide, lik, X, y)
    assert elbo.shape == ()
    assert jnp.isfinite(elbo)


def test_svgp_elbo_gaussian_matches_gaussx_direct():
    """The Gaussian fast path delegates to
    :func:`gaussx.variational_elbo_gaussian` — verify the result matches
    a manual call with the same predictive moments and KL."""
    prior, X, y = _toy()
    guide = FullRankGuide.init(num_inducing=prior.num_inducing)
    lik = GaussianLikelihood(noise_var=0.1)
    elbo_pyrox = svgp_elbo(prior, guide, lik, X, y)

    K_zz_op, K_xz, K_xx_diag = prior.predictive_blocks(X)
    f_loc, f_var = guide.predict(K_xz, K_zz_op, K_xx_diag)
    f_loc = f_loc + prior.mean(X)
    kl = guide.kl_divergence(K_zz_op)
    elbo_ref = variational_elbo_gaussian(y, f_loc, f_var, 0.1, kl)
    assert jnp.allclose(elbo_pyrox, elbo_ref, atol=1e-5)


def test_svgp_elbo_gaussian_increases_with_better_guide():
    """A guide whose mean is closer to the data should produce a higher
    ELBO than the default zero-mean initialization."""
    prior, X, y = _toy()
    lik = GaussianLikelihood(noise_var=0.1)
    guide_bad = FullRankGuide.init(num_inducing=prior.num_inducing)
    elbo_bad = svgp_elbo(prior, guide_bad, lik, X, y)

    K_zz_op = prior.inducing_operator()
    alpha = jnp.linalg.solve(K_zz_op.as_matrix(), jnp.sin(prior.Z.squeeze(-1)))
    guide_good = FullRankGuide(mean=alpha, scale_tril=guide_bad.scale_tril)
    elbo_good = svgp_elbo(prior, guide_good, lik, X, y)
    assert elbo_good > elbo_bad


# --- svgp_elbo: non-conjugate path -----------------------------------------


def test_svgp_elbo_bernoulli_returns_finite_scalar():
    prior, X, _ = _toy()
    y = jnp.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    guide = FullRankGuide.init(num_inducing=prior.num_inducing)
    lik = DistLikelihood(dist_fn=lambda f: nd.Bernoulli(logits=f))
    elbo = svgp_elbo(
        prior,
        guide,
        lik,
        X,
        y,
        integrator=GaussHermiteIntegrator(order=20),
    )
    assert elbo.shape == ()
    assert jnp.isfinite(elbo)


def test_svgp_elbo_nonconjugate_requires_integrator():
    prior, X, _ = _toy()
    y = jnp.ones(10)
    guide = FullRankGuide.init(num_inducing=prior.num_inducing)
    lik = DistLikelihood(dist_fn=lambda f: nd.Bernoulli(logits=f))
    with pytest.raises(ValueError, match="integrator"):
        svgp_elbo(prior, guide, lik, X, y)


# --- svgp_factor -----------------------------------------------------------


def test_svgp_factor_registers_in_numpyro_trace():
    prior, X, y = _toy()
    guide = FullRankGuide.init(num_inducing=prior.num_inducing)
    lik = GaussianLikelihood(noise_var=0.1)

    def model():
        svgp_factor("elbo", prior, guide, lik, X, y)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model()

    assert "elbo" in tr


# --- ConjugateVI -----------------------------------------------------------


def test_cvi_step_returns_natural_guide():
    prior, X, y = _toy()
    guide = NaturalGuide.init(num_inducing=prior.num_inducing)
    lik = GaussianLikelihood(noise_var=0.1)
    cvi = ConjugateVI(damping=0.5)
    new_guide = cvi.step(prior, guide, lik, X, y)
    assert isinstance(new_guide, NaturalGuide)
    assert new_guide.nat1.shape == guide.nat1.shape


def test_cvi_step_changes_natural_parameters():
    prior, X, y = _toy()
    guide = NaturalGuide.init(num_inducing=prior.num_inducing)
    lik = GaussianLikelihood(noise_var=0.1)
    cvi = ConjugateVI(damping=1.0)
    new_guide = cvi.step(prior, guide, lik, X, y)
    assert not jnp.allclose(new_guide.nat1, guide.nat1)


def test_cvi_step_damping_zero_is_identity():
    prior, X, y = _toy()
    guide = NaturalGuide.init(num_inducing=prior.num_inducing)
    lik = GaussianLikelihood(noise_var=0.1)
    cvi = ConjugateVI(damping=0.0)
    new_guide = cvi.step(prior, guide, lik, X, y)
    assert jnp.allclose(new_guide.nat1, guide.nat1, atol=1e-6)
    assert jnp.allclose(new_guide.nat2, guide.nat2, atol=1e-6)


def test_cvi_gaussian_elbo_converges():
    """Repeated CVI steps on a Gaussian regression toy should converge
    to an ELBO substantially better than the initial value. The
    trajectory may oscillate early (natural-gradient methods can
    overshoot), so we check convergence, not strict monotonicity."""
    prior, X, y = _toy()
    guide = NaturalGuide.init(num_inducing=prior.num_inducing)
    lik = GaussianLikelihood(noise_var=0.1)
    cvi = ConjugateVI(damping=0.5)

    elbo_init = float(svgp_elbo(prior, guide, lik, X, y))
    for _ in range(10):
        guide = cvi.step(prior, guide, lik, X, y)
    elbo_final = float(svgp_elbo(prior, guide, lik, X, y))

    assert elbo_final > elbo_init + 10.0


def test_cvi_nonconjugate_requires_integrator():
    prior, X, _ = _toy()
    y = jnp.ones(10)
    guide = NaturalGuide.init(num_inducing=prior.num_inducing)
    lik = DistLikelihood(dist_fn=lambda f: nd.Bernoulli(logits=f))
    cvi = ConjugateVI(damping=0.5)
    with pytest.raises(ValueError, match="integrator"):
        cvi.step(prior, guide, lik, X, y)


# --- one per-site expectation (#334) ---------------------------------------


def _integrate_per_site(integrator, fn, mean, var, y):
    """The old path: the full ``integrate`` on a lifted 1-D Gaussian per site."""

    def per_site(mean_n, var_n, y_n):
        cov = lx.MatrixLinearOperator(var_n[None, None], lx.positive_semidefinite_tag)
        state = gaussx.GaussianState(mean=mean_n[None], cov=cov)
        lifted = lambda f: fn(f[0], y_n)[None]
        return integrator.integrate(lifted, state).state.mean[0]

    return jax.vmap(per_site)(mean, var, y)


@pytest.mark.parametrize(
    "integrator",
    [
        gaussx.GaussHermiteIntegrator(order=20),
        gaussx.UnscentedIntegrator(alpha=1.0),
        gaussx.TaylorIntegrator(),
    ],
    ids=["gauss_hermite", "unscented", "taylor"],
)
def test_per_site_expectation_matches_integrate(integrator):
    mean = jnp.array([0.2, -0.4, 1.1])
    var = jnp.array([0.5, 0.1, 0.9])
    y = jnp.array([1.0, 0.0, 2.0])
    fn = lambda f, y_n: y_n * f - jnp.exp(f)
    got = _per_site_expectation(integrator, fn, mean, var, y)
    expected = _integrate_per_site(integrator, fn, mean, var, y)
    assert jnp.allclose(got, expected, rtol=1e-12, atol=0.0)


def test_ell_numerical_matches_a_direct_sum_of_expectations():
    lik = PoissonLikelihood()
    integrator = gaussx.GaussHermiteIntegrator(order=20)
    mean = jnp.array([0.2, -0.4])
    var = jnp.array([0.5, 0.1])
    y = jnp.array([1.0, 3.0])
    per_site = _integrate_per_site(
        integrator, lambda f, y_n: lik.log_prob(f[None], y_n[None]), mean, var, y
    )
    got = _ell_numerical(lik, y, mean, var, integrator)
    assert jnp.allclose(got, per_site.sum(), rtol=1e-12, atol=0.0)


class _RecordsInputs(Likelihood):
    """A conditional likelihood whose log-density reads ``X``."""

    def log_prob(self, f, y, X=None):
        return -jnp.sum((y - f - X[0]) ** 2)


class _OldSignature(Likelihood):
    """A likelihood from before ``X``: a third argument would raise."""

    def log_prob(self, f, y):
        return -jnp.sum((y - f) ** 2)


def test_ell_numerical_passes_each_site_its_input():
    integrator = gaussx.GaussHermiteIntegrator(order=8)
    mean = jnp.zeros(3)
    var = jnp.zeros(3) + 1e-12
    y = jnp.array([1.0, 2.0, 3.0])
    X = jnp.array([[1.0], [2.0], [3.0]])
    # y - f - X = 0 at every site when each site sees its own input.
    got = _ell_numerical(_RecordsInputs(), y, mean, var, integrator, X)
    assert jnp.allclose(got, 0.0, atol=1e-10)


def test_ell_numerical_gives_an_old_signature_no_third_argument():
    integrator = gaussx.GaussHermiteIntegrator(order=8)
    y = jnp.array([1.0, 2.0])
    X = jnp.array([[1.0], [2.0]])
    got = _ell_numerical(_OldSignature(), y, jnp.zeros(2), jnp.ones(2), integrator, X)
    # E[-(y - f)²] = -(y² + 1) for f ~ N(0, 1)
    assert jnp.allclose(got, -(y**2 + 1.0).sum(), rtol=1e-12, atol=0.0)
