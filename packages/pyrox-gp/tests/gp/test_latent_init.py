"""Tests for `pyrox_gp.latent_init` (P1)."""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import jax.random as jr
import kernellib as kl
import numpy as np
import optax
import pytest
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.initialization import init_to_median, init_to_sample, init_to_value
from pyrox_gp import RBF, LatentFactorGPPrior, latent_init, lfr_model


def _linear_gaussian(n=200, q=2, d=6, noise=0.05):
    X = jr.normal(jr.PRNGKey(0), (n, q))
    W = jr.normal(jr.PRNGKey(1), (d, q))
    return X, X @ W.T + noise * jr.normal(jr.PRNGKey(2), (n, d))


def _cosines(A, B):
    """Cosines of the principal angles between the column spans of A and B."""
    qa, _ = np.linalg.qr(np.asarray(A) - np.asarray(A).mean(0))
    qb, _ = np.linalg.qr(np.asarray(B) - np.asarray(B).mean(0))
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


def test_pca_spans_the_true_latent_subspace():
    # PCA is the ML subspace of the linear-Gaussian model (Tipping & Bishop).
    X, Y = _linear_gaussian()
    Z = latent_init(Y, 2)
    assert np.all(_cosines(Z, X) > 0.999)


@pytest.mark.parametrize("method", ["pca", "kernel_pca", "laplacian_eigenmaps"])
def test_every_method_is_n_by_q_with_unit_column_variance(method):
    _, Y = _linear_gaussian(n=80)
    kwargs = {"kernel": kl.RBF()} if method == "kernel_pca" else {}
    Z = latent_init(Y, 2, method=method, **kwargs)
    assert Z.shape == (80, 2)
    assert np.allclose(np.std(np.asarray(Z), axis=0), 1.0, atol=1e-5)
    assert np.allclose(np.mean(np.asarray(Z), axis=0), 0.0, atol=1e-5)


def test_signs_are_deterministic_and_unstandardized_keeps_scale():
    _, Y = _linear_gaussian(n=50)
    Z = np.asarray(latent_init(Y, 2, standardize=False))
    peaks = Z[np.argmax(np.abs(Z), axis=0), [0, 1]]
    assert np.all(peaks > 0)
    assert np.allclose(Z, np.asarray(latent_init(Y + 0.0, 2, standardize=False)))
    assert np.std(Z[:, 0]) > np.std(Z[:, 1])  # PCA scores, by variance


def test_validation():
    _, Y = _linear_gaussian(n=20)
    with pytest.raises(ValueError, match="needs a kernel"):
        latent_init(Y, 2, method="kernel_pca")
    with pytest.raises(ValueError, match="method must be"):
        latent_init(Y, 2, method="isomap")  # ty: ignore[invalid-argument-type]


def _lfr_final_loss(init_loc_fn, steps=2000):
    n_all, p, q = 40, 6, 2
    X = jnp.linspace(0.0, 1.0, n_all)[:, None]
    Z_true = jnp.stack([jnp.sin(4.0 * X[:, 0]), jnp.cos(7.0 * X[:, 0])], axis=-1)
    Y = Z_true @ jr.normal(jr.PRNGKey(0), (q, p))
    Y = Y + 0.05 * jr.normal(jr.PRNGKey(1), (n_all, p))
    prior = LatentFactorGPPrior(
        kernels=tuple(
            RBF(pyrox_name=f"RBF_q{i}", init_lengthscale=0.2) for i in range(q)
        ),
        X=X,
    )
    guide = AutoDelta(lfr_model, init_loc_fn=init_loc_fn(Y, q))
    svi = SVI(lfr_model, guide, optax.adam(1e-2), loss=Trace_ELBO())
    result = svi.run(jr.PRNGKey(2), steps, X, Y, prior, progress_bar=False)
    return float(jnp.mean(result.losses[-50:]))


def _sampling(Y, q):
    def init(site):
        return init_to_sample(site) if site["name"] == "Z_T" else init_to_median(site)

    return partial(init)


def _from_latent_init(Y, q):
    Z0 = latent_init(Y, q)

    def init(site):
        if site["name"] == "Z_T":
            return init_to_value(site, values={"Z_T": Z0.T})  # stored (Q, N)
        return init_to_median(site)

    return partial(init)


@pytest.mark.slow
def test_lfr_started_from_latent_init_reaches_the_sampled_start_elbo():
    # Same step budget and key: the PCA start is at least as good. At 2000
    # steps both reach the same optimum (-275.95 measured); at 600 the
    # sampled start was not even reproducible (-205 to -276 across runs)
    # while the PCA start stayed within -258..-266, so the budget is the
    # converged one and the margin is 0.5 nats.
    assert _lfr_final_loss(_from_latent_init) <= _lfr_final_loss(_sampling) + 0.5
