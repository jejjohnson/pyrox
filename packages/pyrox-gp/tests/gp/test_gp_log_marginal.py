"""`GPPrior.log_marginal` and `MultiOutputGPPrior.log_marginal` (pyrox#517).

The reference is the dense Gaussian log density of the observations, from
the materialised Gram, at ``rtol=1e-10`` under x64; the factors register the
same value bit for bit.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax.scipy.stats import multivariate_normal
from numpyro import handlers
from pyrox_gp import (
    RBF,
    GPPrior,
    ICMKernel,
    LMCKernel,
    MultiOutputGPPrior,
    gp_factor,
    mo_gp_factor,
)


N, P = 7, 3
X = jnp.linspace(-1.0, 1.0, N)[:, None]
Y1 = jnp.sin(3.0 * X[:, 0]) + 0.1 * jr.normal(jr.key(0), (N,))
Y = jr.normal(jr.key(1), (N, P))
MIXING = jnp.array([[1.0], [0.5], [-0.3]])


def _factor_value(model):
    trace = handlers.trace(handlers.seed(model, 0)).get_trace()
    return trace["y"]["fn"].log_factor


@pytest.mark.parametrize("with_mean", [False, True], ids=["zero-mean", "mean-fn"])
def test_gp_log_marginal_is_the_dense_gaussian(with_mean):
    mean_fn = (lambda x: 0.4 * x[:, 0] - 0.1) if with_mean else None
    prior = GPPrior(RBF(init_lengthscale=0.4), X, mean_fn=mean_fn, jitter=1e-6)
    noise = jnp.asarray(0.05)
    K = prior.kernel(X, X) + (1e-6 + noise) * jnp.eye(N)
    expected = multivariate_normal.logpdf(Y1, prior.mean(X), K)
    assert jnp.allclose(prior.log_marginal(Y1, noise), expected, rtol=1e-10)
    assert jnp.array_equal(
        _factor_value(lambda: gp_factor("y", prior, Y1, noise)),
        prior.log_marginal(Y1, noise),
    )


@pytest.mark.parametrize(
    "kernel",
    [
        ICMKernel(kernel=RBF(init_lengthscale=0.4), mixing=MIXING),
        LMCKernel(
            kernels=(
                RBF(pyrox_name="RBF_q0"),
                RBF(pyrox_name="RBF_q1", init_lengthscale=0.3),
            ),
            mixing=jnp.array([[1.0, 0.2], [0.5, -0.4], [-0.3, 0.8]]),
        ),
    ],
    ids=["icm-kronecker", "lmc-dense"],
)
@pytest.mark.parametrize(
    "per_output", [False, True], ids=["scalar-noise", "per-output"]
)
def test_multi_output_log_marginal_is_the_dense_gaussian(kernel, per_output):
    prior = MultiOutputGPPrior(kernel, X, jitter=1e-6)
    noise = jnp.array([0.05, 0.1, 0.2]) if per_output else jnp.asarray(0.07)
    # Output-major (p n) layout, as the operators use.
    noise_diag = jnp.repeat(jnp.broadcast_to(noise, (P,)), N)
    K = kernel.full_covariance(X) + jnp.diag(1e-6 + noise_diag)
    expected = multivariate_normal.logpdf(Y.T.reshape(-1), jnp.zeros(N * P), K)
    assert jnp.allclose(prior.log_marginal(Y, noise), expected, rtol=1e-10)
    assert jnp.array_equal(
        _factor_value(lambda: mo_gp_factor("y", prior, Y, noise)),
        prior.log_marginal(Y, noise),
    )


def test_multi_output_log_marginal_rejects_misshapen_targets():
    prior = MultiOutputGPPrior(ICMKernel(kernel=RBF(), mixing=MIXING), X)
    with pytest.raises(ValueError):
        prior.log_marginal(Y[:, :2], jnp.asarray(0.1))
