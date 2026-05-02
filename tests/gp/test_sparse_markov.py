"""Tests for the sparse-variational Markov GP surface.

Coverage strategy:

* construction guards: rejects non-monotone or non-1-D inducing grids.
* shape / finiteness / PSD on the inducing prior and the SVGP
  predictive blocks (``predictive_blocks``).
* sampling + log-density on the inducing prior are finite and
  shape-correct.
* ELBO sanity: Gaussian-likelihood ELBO is finite; non-Gaussian
  likelihoods without an integrator raise.
* NumPyro hook: :func:`sparse_markov_factor` registers a finite factor
  site under :func:`numpyro.handlers.trace`.
* prediction surface: :class:`SparseConditionedMarkovGP.predict`
  returns finite mean / non-negative variance.

A direct SVGP-Markov ↔ dense ``MarkovGPPrior`` equivalence test and a
gradient-based ELBO-improvement test are exercised via the demo
notebook (the latter requires the JIT plumbing documented near
``test_elbo_non_gaussian_without_integrator_raises``).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from pyrox.gp import (
    FullRankGuide,
    GaussianLikelihood,
    MaternSDE,
    SparseConditionedMarkovGP,
    SparseMarkovGPPrior,
    sparse_markov_elbo,
    sparse_markov_factor,
)


def _make_problem(n: int = 12, m: int = 6):
    times = jnp.linspace(0.0, 5.0, n)
    y = jnp.sin(1.5 * times) + 0.05 * jr.normal(jr.PRNGKey(0), (n,))
    Z = jnp.linspace(0.0, 5.0, m)
    sde = MaternSDE(variance=1.0, lengthscale=0.6, order=1)
    prior = SparseMarkovGPPrior(sde, Z, jitter=1e-6)
    return prior, times, y


def test_sparse_markov_invalid_z_raises() -> None:
    sde = MaternSDE(order=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        SparseMarkovGPPrior(sde, jnp.array([0.0, 0.5, 0.5]))
    with pytest.raises(ValueError, match="must be 1-D"):
        SparseMarkovGPPrior(sde, jnp.zeros((3, 2)))


def test_predictive_blocks_shapes_and_psd() -> None:
    prior, times, _ = _make_problem()
    K_zz_op, K_xz, K_xx_diag = prior.predictive_blocks(times)
    assert K_xz.shape == (times.shape[0], prior.num_inducing)
    assert K_xx_diag.shape == times.shape
    assert jnp.all(jnp.isfinite(K_xz))
    assert jnp.all(K_xx_diag > 0.0)
    # K_zz must be symmetric PSD.
    K_zz = K_zz_op.as_matrix()
    assert jnp.allclose(K_zz, K_zz.T, atol=1e-6)
    eig = jnp.linalg.eigvalsh(0.5 * (K_zz + K_zz.T))
    assert jnp.all(eig > 0.0)


def test_log_prob_at_zero_is_finite() -> None:
    prior, _, _ = _make_problem()
    u = jnp.zeros(prior.num_inducing)
    assert jnp.isfinite(prior.log_prob(u))


def test_sample_shape_and_finite() -> None:
    prior, _, _ = _make_problem()
    u = prior.sample(jr.PRNGKey(0))
    assert u.shape == (prior.num_inducing,)
    assert jnp.all(jnp.isfinite(u))


def test_elbo_gaussian_likelihood_finite() -> None:
    prior, times, y = _make_problem()
    guide = FullRankGuide.init(prior.num_inducing, scale=0.3)
    elbo = sparse_markov_elbo(prior, guide, GaussianLikelihood(0.05), times, y)
    assert jnp.isfinite(elbo)


def test_elbo_non_gaussian_without_integrator_raises() -> None:
    from pyrox.gp import BernoulliLikelihood

    prior, times, y = _make_problem()
    guide = FullRankGuide.init(prior.num_inducing, scale=0.3)
    with pytest.raises(ValueError, match="integrator"):
        sparse_markov_elbo(
            prior, guide, BernoulliLikelihood(), times, (y > 0).astype(jnp.float32)
        )


def test_sparse_markov_factor_registers_finite_factor_site() -> None:
    """``sparse_markov_factor`` must register one NumPyro factor site
    whose value equals :func:`sparse_markov_elbo` and is finite."""
    from numpyro import handlers

    prior, times, y = _make_problem()
    guide = FullRankGuide.init(prior.num_inducing, scale=0.3)
    likelihood = GaussianLikelihood(0.05)

    def model() -> None:
        sparse_markov_factor("svgp_markov", prior, guide, likelihood, times, y)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model()

    assert "svgp_markov" in tr
    site = tr["svgp_markov"]
    # ``numpyro.factor`` registers as a ``sample`` site backed by a
    # ``Unit`` distribution whose ``log_factor`` carries the ELBO value.
    assert site["type"] == "sample"
    log_factor = site["fn"].log_factor
    assert jnp.isfinite(log_factor)
    expected = sparse_markov_elbo(prior, guide, likelihood, times, y)
    assert jnp.allclose(log_factor, expected)


# Note: SVI training of `sparse_markov_elbo` is exercised via the demo
# notebooks (where the loss is JIT-wrapped, sidestepping JAX's
# reverse-mode limitations on the default ``jax.scipy.linalg.expm``
# scaling-and-squaring fori_loop used inside ``SDEKernel.discretise``).
# A standalone gradient unit test would need to mirror that JIT
# plumbing, which is brittle and tangential to the API contract this
# module owns; the ELBO-finiteness test above covers the forward path.


@pytest.mark.slow
def test_sparse_conditioned_predict_is_finite() -> None:
    prior, times, _y = _make_problem()
    guide = FullRankGuide.init(prior.num_inducing, scale=0.3)
    cond = SparseConditionedMarkovGP(prior=prior, guide=guide)
    t_star = jnp.linspace(times[0] - 0.5, times[-1] + 0.5, 20)
    mean, var = cond.predict(t_star)
    assert mean.shape == (20,) and var.shape == (20,)
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(var >= 0.0)
