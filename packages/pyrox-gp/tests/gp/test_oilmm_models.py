"""Tests for the OILMM projected multi-output GP model.

Covers :class:`OILMMGPPrior` / :class:`ConditionedOILMMGP` — the
orthogonal-projection exact workflow on top of :class:`OILMMKernel`.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from pyrox_gp import (
    RBF,
    MultiOutputGPPrior,
    OILMMGPPrior,
    OILMMKernel,
)


def _oilmm(P: int = 3, Q: int = 2, seed: int = 3) -> OILMMKernel:
    W, _ = jnp.linalg.qr(jr.normal(jr.PRNGKey(seed), (P, Q)))
    kernels = tuple(
        RBF(
            pyrox_name=f"RBF_q{q}",
            init_variance=1.0 - 0.2 * q,
            init_lengthscale=0.5 + 0.5 * q,
        )
        for q in range(Q)
    )
    return OILMMKernel(kernels=kernels, mixing=W, check_orthogonal=True)


def _data(N: int = 8, D: int = 2, P: int = 3, seed: int = 0):
    X = jr.uniform(jr.PRNGKey(seed), (N, D))
    Y = jr.normal(jr.PRNGKey(seed + 1), (N, P))
    return X, Y


def test_oilmm_prior_shapes():
    X, _ = _data()
    prior = OILMMGPPrior(kernel=_oilmm(), X=X)
    assert prior.num_outputs == 3
    assert prior.num_latents == 2
    assert prior.mean(X).shape == (8, 3)
    assert prior.sample(jr.PRNGKey(0)).shape == (8, 3)
    assert len(prior.latent_priors()) == 2


def test_oilmm_condition_predict_shapes_and_positive_variance():
    X, Y = _data()
    cond = OILMMGPPrior(kernel=_oilmm(), X=X).condition(Y, noise_var=0.1)
    mean, var = cond.predict(X[:5])
    assert mean.shape == (5, 3)
    assert var.shape == (5, 3)
    assert bool((var > 0.0).all())


@pytest.mark.parametrize("shape", [(2, 2), (3, 2)])
def test_oilmm_matches_dense_exact_model_isotropic_noise(shape):
    """Orthonormal mixing + scalar noise: the projected posterior equals
    the exact dense multi-output GP, for both square (P=Q) and
    semi-orthogonal (P>Q) mixings."""
    P, Q = shape
    X, Y = _data(P=P)
    kernel = _oilmm(P=P, Q=Q)
    jitter, noise = 1e-5, 0.1
    X_star = jr.uniform(jr.PRNGKey(2), (5, X.shape[1]))

    oilmm = OILMMGPPrior(kernel=kernel, X=X, jitter=jitter).condition(Y, noise)
    dense = MultiOutputGPPrior(kernel=kernel, X=X, jitter=jitter).condition(Y, noise)

    mean_o, var_o = oilmm.predict(X_star)
    mean_d, var_d = dense.predict(X_star)
    assert jnp.allclose(mean_o, mean_d, atol=1e-4)
    assert jnp.allclose(var_o, var_d, atol=1e-4)


def test_oilmm_predict_paths_consistent():
    """predict_mean / predict_var must agree with the joint predict()."""
    X, Y = _data()
    cond = OILMMGPPrior(kernel=_oilmm(), X=X).condition(Y, 0.1)
    X_star = X[:4]
    mean, var = cond.predict(X_star)
    assert jnp.allclose(cond.predict_mean(X_star), mean, atol=1e-6)
    assert jnp.allclose(cond.predict_var(X_star), var, atol=1e-6)


def test_oilmm_per_output_noise_runs():
    """Per-output noise uses the projected-noise approximation; it must
    produce finite moments of the right shape."""
    X, Y = _data()
    cond = OILMMGPPrior(kernel=_oilmm(), X=X).condition(
        Y, noise_var=jnp.array([0.05, 0.1, 0.2])
    )
    mean, var = cond.predict(X[:4])
    assert mean.shape == (4, 3)
    assert bool(jnp.isfinite(var).all())
    assert bool((var > 0.0).all())


def test_oilmm_mean_fn_added_to_predictions():
    X, Y = _data()
    offset = 2.0

    def mean_fn(X):
        return jnp.full((X.shape[0], 3), offset, dtype=X.dtype)

    base = OILMMGPPrior(kernel=_oilmm(), X=X).condition(Y, 0.1)
    shifted = OILMMGPPrior(kernel=_oilmm(), X=X, mean_fn=mean_fn).condition(
        Y + offset, 0.1
    )
    mean_base, var_base = base.predict(X[:4])
    mean_shifted, var_shifted = shifted.predict(X[:4])
    assert jnp.allclose(mean_shifted, mean_base + offset, atol=1e-4)
    assert jnp.allclose(var_shifted, var_base, atol=1e-5)


def test_oilmm_conditioned_sample_shape():
    X, Y = _data()
    cond = OILMMGPPrior(kernel=_oilmm(), X=X).condition(Y, 0.1)
    samples = cond.sample(jr.PRNGKey(0), X[:4], n_samples=5)
    assert samples.shape == (5, 4, 3)
    assert bool(jnp.isfinite(samples).all())


def test_oilmm_condition_rejects_wrong_target_shape():
    X, Y = _data()
    prior = OILMMGPPrior(kernel=_oilmm(), X=X)
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        prior.condition(Y[:, :2], 0.1)


def test_oilmm_condition_rejects_bad_noise_shape():
    X, Y = _data()
    prior = OILMMGPPrior(kernel=_oilmm(), X=X)
    with pytest.raises(ValueError, match="noise_var"):
        prior.condition(Y, noise_var=jnp.ones(2))


def test_oilmm_condition_shares_context_for_priored_tied_kernel():
    """A priored kernel reused across latents registers each site once
    across the whole condition sweep."""
    kernel = RBF()
    kernel.set_prior("variance", dist.LogNormal(0.0, 0.3))
    kernel.set_prior("lengthscale", dist.LogNormal(0.0, 0.3))
    W, _ = jnp.linalg.qr(jr.normal(jr.PRNGKey(0), (2, 2)))
    oilmm = OILMMKernel(kernels=(kernel, kernel), mixing=W)
    X = jnp.array([[0.0], [0.5], [1.0]])
    Y = jnp.zeros((3, 2))
    prior = OILMMGPPrior(kernel=oilmm, X=X)

    def model():
        return prior.condition(Y, noise_var=0.1)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model()
    assert "RBF.variance" in tr
    assert "RBF.lengthscale" in tr
