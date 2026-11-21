"""`SparseGPPrior.condition` and `MultiOutputSparseGPPrior.condition` (pyrox#518).

The conditioned results compute exactly what the hand-written prediction
did, so the comparisons are bit for bit.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest
from pyrox_gp import (
    RBF,
    ConditionedMultiOutputSparseGP,
    ConditionedSparseGP,
    FullRankGuide,
    LMCKernel,
    MultiOutputSparseGPPrior,
    SharedInducingPoints,
    SparseGPPrior,
)


X_STAR = jnp.linspace(-1.2, 1.2, 9)[:, None]


def _fitted_guide(num_inducing, seed=0):
    """A guide with a non-trivial mean and covariance, as after a fit."""
    k1, k2 = jr.split(jr.key(seed))
    scale_tril = jnp.tril(0.3 * jr.normal(k2, (num_inducing, num_inducing)))
    return FullRankGuide(
        mean=jr.normal(k1, (num_inducing,)),
        scale_tril=scale_tril + 0.5 * jnp.eye(num_inducing),
    )


@pytest.mark.parametrize("with_mean", [False, True], ids=["zero-mean", "mean-fn"])
def test_sparse_condition_equals_the_hand_written_prediction(with_mean):
    mean_fn = (lambda x: 0.3 * x[:, 0] + 1.0) if with_mean else None
    prior = SparseGPPrior(
        RBF(init_lengthscale=0.5),
        Z=jnp.linspace(-1.0, 1.0, 5)[:, None],
        mean_fn=mean_fn,
    )
    guide = _fitted_guide(prior.num_inducing)
    cond = prior.condition(guide)
    assert isinstance(cond, ConditionedSparseGP)
    assert cond.prior is prior and cond.guide is guide
    mean, var = cond.predict(X_STAR)
    K_zz, K_xz, K_xx_diag = prior.predictive_blocks(X_STAR)
    mean_ref, var_ref = guide.predict(K_xz, K_zz, K_xx_diag)
    assert jnp.array_equal(mean, mean_ref + prior.mean(X_STAR))
    assert jnp.array_equal(var, var_ref)


def _mo_prior(with_mean=False):
    kernel = LMCKernel(
        kernels=(
            RBF(pyrox_name="RBF_q0"),
            RBF(pyrox_name="RBF_q1", init_lengthscale=0.4),
        ),
        mixing=jnp.array([[1.0, 0.2], [0.5, -0.4], [-0.3, 0.8]]),
    )
    mean_fn = (
        (lambda x: jnp.stack([x[:, 0], -x[:, 0], 0.5 + 0 * x[:, 0]], -1))
        if with_mean
        else None
    )
    return MultiOutputSparseGPPrior(
        kernel,
        SharedInducingPoints(locations=jnp.linspace(-1.0, 1.0, 4)[:, None]),
        mean_fn=mean_fn,
    )


@pytest.mark.parametrize("with_mean", [False, True], ids=["zero-mean", "mean-fn"])
def test_multi_output_condition_equals_the_deprecated_predict(with_mean):
    prior = _mo_prior(with_mean)
    guide = _fitted_guide(prior.num_inducing, seed=1)
    cond = prior.condition(guide)
    assert isinstance(cond, ConditionedMultiOutputSparseGP)
    mean, var = cond.predict(X_STAR)
    assert mean.shape == var.shape == (X_STAR.shape[0], 3)
    with pytest.warns(DeprecationWarning, match=r"condition\(guide\)\.predict"):
        mean_old, var_old = prior.predict(guide, X_STAR)
    assert jnp.array_equal(mean, mean_old) and jnp.array_equal(var, var_old)
