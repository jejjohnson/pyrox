"""GMRFDistribution: a gaussx GMRF field as a NumPyro distribution."""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import numpyro
import pyrox_gauss as pg
import pytest


def _fields():
    n = 6
    # Path-graph Laplacian R and the proper precision R + I
    rows = np.r_[np.arange(n), np.arange(1, n)]
    cols = np.r_[np.arange(n), np.arange(n - 1)]
    degree = jnp.r_[1.0, 2.0 * jnp.ones(n - 2), 1.0]
    structure = gx.SparseOperator.from_coo(
        rows, cols, jnp.r_[degree, -jnp.ones(n - 1)], (n, n), symmetric=True
    )
    precision = gx.SparseOperator.from_coo(
        rows, cols, jnp.r_[degree + 1.0, -jnp.ones(n - 1)], (n, n), symmetric=True
    )
    null = jnp.ones(n) / jnp.sqrt(n)
    proper = gx.GaussianMRF(jnp.zeros(n), precision)
    return {
        "proper": proper,
        "constrained": proper.condition_on_constraints(jnp.ones(n), 0.0),
        "intrinsic": gx.IntrinsicGMRF(jnp.zeros(n), 1.5, structure, null),
        "soft": gx.IntrinsicGMRF(jnp.zeros(n), 1.5, structure, null, constraint="soft"),
        "dense": gx.GaussianMRF(
            jnp.ones(3),
            lx.MatrixLinearOperator(2.0 * jnp.eye(3), lx.positive_semidefinite_tag),
        ),
    }


@pytest.mark.parametrize(
    "name", ["proper", "constrained", "intrinsic", "soft", "dense"]
)
def test_seed_and_trace_reproduce_the_field(name):
    field = _fields()[name]

    def model():
        numpyro.sample("x", pg.GMRFDistribution(field))

    trace = numpyro.handlers.trace(numpyro.handlers.seed(model, 0)).get_trace()
    site = trace["x"]
    assert site["value"].shape == field.event_shape
    np.testing.assert_allclose(
        site["fn"].log_prob(site["value"]), field.log_prob(site["value"])
    )
    # A re-seeded run reproduces the draw.
    again = numpyro.handlers.trace(numpyro.handlers.seed(model, 0)).get_trace()
    np.testing.assert_array_equal(again["x"]["value"], site["value"])


def test_sample_mean_and_variance_delegate_to_the_field():
    field = _fields()["dense"]
    wrapped = pg.GMRFDistribution(field)
    key = jr.key(0)
    np.testing.assert_array_equal(wrapped.sample(key, (4,)), field.sample(key, (4,)))
    np.testing.assert_allclose(wrapped.mean, field.mean)
    np.testing.assert_allclose(wrapped.variance, jnp.full(3, 0.5))


def test_pytree_round_trip_under_jit():
    wrapped = pg.GMRFDistribution(_fields()["proper"])
    x = jr.normal(jr.key(1), wrapped.event_shape)
    leaves, treedef = jax.tree_util.tree_flatten(wrapped)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_allclose(
        jax.jit(lambda d, v: d.log_prob(v))(rebuilt, x), wrapped.log_prob(x)
    )
