"""Tests for ``predict(X, include_noise=True)`` on the exact GP results (pyrox#521).

The keyword adds exactly the noise variance the posterior was conditioned
on (scalar, or per output) to the variance and leaves the mean unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest
from pyrox_gp import (
    RBF,
    GPPrior,
    ICMKernel,
    MultiOutputGPPrior,
    OILMMGPPrior,
    OILMMKernel,
)


N, P, Q = 7, 3, 2
X = jr.uniform(jr.key(0), (N, 1))
X_STAR = jr.uniform(jr.key(1), (4, 1))
Y = jr.normal(jr.key(2), (N, P))


def _assert_noise_added(cond, noise):
    mean, var = cond.predict(X_STAR)
    mean_n, var_n = cond.predict(X_STAR, include_noise=True)
    assert jnp.array_equal(mean_n, mean)
    # One addition, so the difference is the noise to round-off of var.
    assert jnp.allclose(
        var_n - var, jnp.broadcast_to(noise, var.shape), rtol=0, atol=1e-14
    )


def test_conditioned_gp_adds_scalar_noise():
    noise = jnp.asarray(0.2)
    cond = GPPrior(kernel=RBF(), X=X, jitter=1e-6).condition(Y[:, 0], noise)
    _assert_noise_added(cond, noise)


@pytest.mark.parametrize("noise", [jnp.asarray(0.2), jnp.array([0.1, 0.2, 0.4])])
def test_conditioned_multi_output_gp_adds_noise(noise):
    kernel = ICMKernel(kernel=RBF(), mixing=jnp.array([[1.0], [0.5], [-0.3]]))
    cond = MultiOutputGPPrior(kernel=kernel, X=X, jitter=1e-6).condition(Y, noise)
    _assert_noise_added(cond, noise)


@pytest.mark.parametrize("noise", [jnp.asarray(0.2), jnp.array([0.1, 0.2, 0.4])])
def test_conditioned_oilmm_gp_adds_per_output_noise(noise):
    W, _ = jnp.linalg.qr(jr.normal(jr.key(3), (P, Q)))
    kernels = tuple(RBF(pyrox_name=f"RBF_q{q}") for q in range(Q))
    prior = OILMMGPPrior(
        kernel=OILMMKernel(kernels=kernels, mixing=W), X=X, jitter=1e-6
    )
    _assert_noise_added(prior.condition(Y, noise), noise)


def test_include_noise_is_keyword_only():
    cond = GPPrior(kernel=RBF(), X=X).condition(Y[:, 0], jnp.asarray(0.2))
    with pytest.raises(TypeError):
        cond.predict(X_STAR, True)
