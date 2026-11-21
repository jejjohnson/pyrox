"""Tests for `OILMMGPPrior.log_marginal` and `oilmm_factor` (pyrox#519).

The projected log marginal must equal the dense `MultiOutputGPPrior` log
marginal of the same `OILMMKernel` (Bruinsma et al., 2020): both are exact
under orthonormal mixing and isotropic noise, so they agree to round-off.
"""

from __future__ import annotations

import jax
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
    oilmm_factor,
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


def _mean_fn(X):
    return jnp.stack([X[:, 0], -X[:, 1], 0.5 + 0.0 * X[:, 0]], axis=-1)


@pytest.mark.parametrize("shape", [(3, 2), (3, 1), (3, 3)])
@pytest.mark.parametrize("use_mean", [False, True])
def test_log_marginal_matches_dense_model(shape, use_mean):
    P, Q = shape
    X, Y = _data(P=P)
    kernel = _oilmm(P=P, Q=Q)
    mean_fn = _mean_fn if use_mean else None
    jitter, noise = 1e-5, jnp.asarray(0.1)
    oilmm = OILMMGPPrior(kernel=kernel, X=X, jitter=jitter, mean_fn=mean_fn)
    dense = MultiOutputGPPrior(kernel=kernel, X=X, jitter=jitter, mean_fn=mean_fn)
    # Both are exact: the gap is round-off of an O(PN) sum under x64.
    assert jnp.allclose(
        oilmm.log_marginal(Y, noise), dense.log_marginal(Y, noise), rtol=1e-10
    )


def test_log_marginal_rejects_per_output_noise():
    X, Y = _data()
    prior = OILMMGPPrior(kernel=_oilmm(), X=X)
    with pytest.raises(ValueError, match="mo_gp_factor"):
        prior.log_marginal(Y, jnp.full(3, 0.1))


def test_log_marginal_rejects_wrong_target_shape():
    X, Y = _data()
    prior = OILMMGPPrior(kernel=_oilmm(), X=X)
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        prior.log_marginal(Y[:, :2], jnp.asarray(0.1))


def test_log_marginal_jits_and_differentiates_in_noise():
    X, Y = _data()
    prior = OILMMGPPrior(kernel=_oilmm(), X=X)
    value, grad = jax.jit(jax.value_and_grad(lambda s: prior.log_marginal(Y, s)))(
        jnp.asarray(0.1)
    )
    assert jnp.allclose(value, prior.log_marginal(Y, jnp.asarray(0.1)), rtol=1e-12)
    assert bool(jnp.isfinite(grad))


def test_oilmm_factor_registers_log_marginal():
    X, Y = _data()
    prior = OILMMGPPrior(kernel=_oilmm(), X=X)
    noise = jnp.asarray(0.1)

    def model():
        oilmm_factor("y", prior, Y, noise)

    tr = handlers.trace(handlers.seed(model, 0)).get_trace()
    assert tr["y"]["type"] == "sample"
    assert jnp.array_equal(
        tr["y"]["fn"].log_prob(tr["y"]["value"]), prior.log_marginal(Y, noise)
    )


def test_oilmm_factor_draws_each_priored_kernel_site_once():
    """A priored kernel reused across latents registers its sites once."""
    kernel = RBF()
    kernel.set_prior("variance", dist.LogNormal(0.0, 0.3))
    kernel.set_prior("lengthscale", dist.LogNormal(0.0, 0.3))
    W, _ = jnp.linalg.qr(jr.normal(jr.PRNGKey(0), (3, 2)))
    prior = OILMMGPPrior(
        kernel=OILMMKernel(kernels=(kernel, kernel), mixing=W), X=_data()[0]
    )

    def model():
        oilmm_factor("y", prior, _data()[1], jnp.asarray(0.1))

    tr = handlers.trace(handlers.seed(model, 0)).get_trace()
    assert sorted(tr) == ["RBF.lengthscale", "RBF.variance", "y"]
