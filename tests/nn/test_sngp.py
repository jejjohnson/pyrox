"""Tests for ``pyrox.nn._sngp`` SNGP output head."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro import handlers

from pyrox._core.pyrox_module import PyroxModule
from pyrox.nn import LaplaceRandomFeatureCovariance, RandomFeatureGaussianProcess


# --- LaplaceRandomFeatureCovariance ---------------------------------------


def test_lrfc_init_validates():
    with pytest.raises(ValueError, match="num_features must be > 0"):
        LaplaceRandomFeatureCovariance.init(0)
    with pytest.raises(ValueError, match="momentum"):
        LaplaceRandomFeatureCovariance.init(4, momentum=-0.1)
    with pytest.raises(ValueError, match="momentum"):
        LaplaceRandomFeatureCovariance.init(4, momentum=1.1)
    with pytest.raises(ValueError, match="ridge"):
        LaplaceRandomFeatureCovariance.init(4, ridge=0.0)


def test_lrfc_init_starts_at_ridge_identity():
    cov = LaplaceRandomFeatureCovariance.init(4, ridge=2.5)
    assert jnp.allclose(cov.precision, 2.5 * jnp.eye(4))


def test_lrfc_update_is_pure_functional_and_ema():
    cov = LaplaceRandomFeatureCovariance.init(3, momentum=0.9, ridge=1.0)
    features = jnp.eye(3) * 2.0  # rows are 2 * e_i; outer/3 = (4/3) I_3 ... no
    # Actually with rows (2,0,0),(0,2,0),(0,0,2): outer = features.T @ features / 3
    # = diag(4,4,4)/3 = (4/3) I_3
    new_cov = cov.update(features)
    # Original is unchanged.
    assert jnp.allclose(cov.precision, jnp.eye(3))
    # EMA: 0.9 * I + 0.1 * (4/3) I = (0.9 + 4/30) I.
    expected_diag = 0.9 + (1.0 - 0.9) * 4.0 / 3.0
    assert jnp.allclose(new_cov.precision, expected_diag * jnp.eye(3), atol=1e-6)


def test_lrfc_covariance_inverts_precision():
    """covariance() ≈ inv(precision) up to symmetrisation."""
    key = jr.PRNGKey(0)
    A = jr.normal(key, (5, 5))
    sym = A @ A.T + jnp.eye(5)  # symmetric positive definite
    cov = LaplaceRandomFeatureCovariance(precision=sym, momentum=0.999, ridge=1.0)
    Sigma = cov.covariance()
    assert jnp.allclose(sym @ Sigma, jnp.eye(5), atol=1e-4)


def test_lrfc_variance_at_matches_explicit_quadratic_form():
    key = jr.PRNGKey(1)
    A = jr.normal(key, (4, 4))
    sym = A @ A.T + jnp.eye(4)
    cov = LaplaceRandomFeatureCovariance(precision=sym, momentum=0.999, ridge=1.0)

    features = jr.normal(jr.PRNGKey(2), (3, 4))
    var = cov.variance_at(features)

    Sigma = jnp.linalg.inv(sym)
    expected = jnp.einsum("nd,de,ne->n", features, Sigma, features)
    assert jnp.allclose(var, expected, atol=1e-4)
    # Quadratic form on a positive-definite matrix is non-negative.
    assert jnp.all(var >= 0.0)


# --- RandomFeatureGaussianProcess -----------------------------------------


def test_sngp_output_shape_mean_only():
    layer = RandomFeatureGaussianProcess.init(
        jr.PRNGKey(0),
        in_features=5,
        num_features=16,
        out_features=2,
    )
    x = jnp.ones((6, 5))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert y.shape == (6, 2)


def test_sngp_output_shape_with_cov():
    layer = RandomFeatureGaussianProcess.init(
        jr.PRNGKey(0),
        in_features=5,
        num_features=16,
        out_features=2,
    )
    x = jnp.ones((6, 5))
    with handlers.seed(rng_seed=0):
        mean, var = layer(x, return_cov=True)
    assert mean.shape == (6, 2)
    assert var.shape == (6,)
    assert jnp.all(var >= 0.0)


def test_sngp_supports_arbitrary_batch_dims():
    layer = RandomFeatureGaussianProcess.init(
        jr.PRNGKey(0),
        in_features=4,
        num_features=12,
        out_features=2,
    )
    x = jnp.ones((3, 5, 4))  # (batch, time, D_in)
    with handlers.seed(rng_seed=0):
        mean = layer(x)
        mean2, var = layer(x, return_cov=True)
    assert mean.shape == (3, 5, 2)
    assert mean2.shape == (3, 5, 2)
    assert var.shape == (3, 5)


def test_sngp_init_validates_positive_dims_and_lengthscale():
    with pytest.raises(ValueError, match="must all be > 0"):
        RandomFeatureGaussianProcess.init(
            jr.PRNGKey(0), in_features=0, num_features=4, out_features=2
        )
    with pytest.raises(ValueError, match="init_lengthscale must be > 0"):
        RandomFeatureGaussianProcess.init(
            jr.PRNGKey(0),
            in_features=4,
            num_features=8,
            out_features=2,
            init_lengthscale=0.0,
        )


def test_sngp_validates_init_arrays_shape():
    with pytest.raises(ValueError, match="W_init shape"):
        RandomFeatureGaussianProcess(
            in_features=4,
            num_features=8,
            out_features=2,
            W_init=jnp.zeros((3, 8)),  # wrong: should be (4, 8)
            bias_init=jnp.zeros(8),
            output_linear_init=jnp.zeros((8, 2)),
            covariance=LaplaceRandomFeatureCovariance.init(8),
        )


def test_sngp_registers_param_sites():
    layer = RandomFeatureGaussianProcess.init(
        jr.PRNGKey(0),
        in_features=4,
        num_features=8,
        out_features=2,
        pyrox_name="sngp",
    )
    x = jnp.ones((1, 4))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(x)
    for k in (
        "sngp.W",
        "sngp.bias",
        "sngp.lengthscale",
        "sngp.output_linear",
        "sngp.output_bias",
    ):
        assert tr[k]["type"] == "param", f"{k} should be a param site"


def test_sngp_is_deterministic_across_seeds():
    """Mean prediction is deterministic — RFF freqs are frozen, head is linear."""
    layer = RandomFeatureGaussianProcess.init(
        jr.PRNGKey(0),
        in_features=4,
        num_features=8,
        out_features=2,
    )
    x = jr.normal(jr.PRNGKey(7), (5, 4))
    with handlers.seed(rng_seed=0):
        y1 = layer(x)
    with handlers.seed(rng_seed=1):
        y2 = layer(x)
    assert jnp.allclose(y1, y2)


def test_sngp_update_precision_is_pure_functional():
    """update_precision returns a new layer with EMA-updated covariance."""
    layer = RandomFeatureGaussianProcess.init(
        jr.PRNGKey(0),
        in_features=4,
        num_features=8,
        out_features=2,
        momentum=0.9,
        ridge=1.0,
    )
    x = jr.normal(jr.PRNGKey(7), (16, 4))
    with handlers.seed(rng_seed=0):
        features = layer.feature_map(x)
    new_layer = layer.update_precision(features)

    # Original layer is unchanged.
    assert jnp.allclose(layer.covariance.precision, jnp.eye(8))
    # New layer's precision has shifted away from the ridge identity.
    assert not jnp.allclose(new_layer.covariance.precision, jnp.eye(8))
    # And exactly matches the manual EMA.
    expected = 0.9 * jnp.eye(8) + 0.1 * (features.T @ features) / 16
    assert jnp.allclose(new_layer.covariance.precision, expected, atol=1e-5)


def test_sngp_variance_changes_after_precision_updates():
    """Repeated precision updates change the per-input variance."""
    layer = RandomFeatureGaussianProcess.init(
        jr.PRNGKey(0),
        in_features=4,
        num_features=12,
        out_features=1,
        momentum=0.5,
        ridge=1.0,
    )
    x = jr.normal(jr.PRNGKey(7), (20, 4))

    with handlers.seed(rng_seed=0):
        train_features = layer.feature_map(x)
    new_layer = layer
    for _ in range(8):
        new_layer = new_layer.update_precision(train_features)

    with handlers.seed(rng_seed=0):
        _, var0 = layer(x, return_cov=True)
        _, var1 = new_layer(x, return_cov=True)
    # Before updates, precision is `ridge * I` so var = ||phi||^2 / ridge.
    # After updates, precision concentrates on directions populated by
    # the training features, so the variance distribution shifts —
    # we only assert it actually moved (the direction/sign depends on
    # how features happen to align, which the unit test cannot fix).
    assert jnp.all(jnp.isfinite(var1))
    assert jnp.all(var1 >= 0.0)
    assert not jnp.allclose(var0, var1)


def test_sngp_is_pyrox_module():
    layer = RandomFeatureGaussianProcess.init(
        jr.PRNGKey(0),
        in_features=2,
        num_features=4,
        out_features=2,
    )
    assert isinstance(layer, PyroxModule)


def test_sngp_layer_is_jax_pytree_with_covariance_leaf():
    """The covariance container must be a leaf in the JAX PyTree so
    `update_precision`'s `eqx.tree_at` works correctly."""
    layer = RandomFeatureGaussianProcess.init(
        jr.PRNGKey(0),
        in_features=2,
        num_features=4,
        out_features=1,
    )
    leaves = eqx.filter(layer, eqx.is_array)
    # At least precision is in there.
    assert any(
        leaf.shape == (4, 4) and jnp.allclose(leaf, jnp.eye(4))
        for leaf in jax_tree_leaves(leaves)
    )


def jax_tree_leaves(tree):
    import jax

    return jax.tree_util.tree_leaves(tree)
