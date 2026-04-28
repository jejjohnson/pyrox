"""Tests for ``pyrox.nn.DeepVSSGP``.

Covers forward shape, NumPyro site registration (3L sites for L layers),
init validation, and the ``depth=1`` reduction to a single VSSGP layer.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro import handlers

from pyrox.nn import DeepVSSGP


def test_forward_shape_default():
    """Forward pass returns ``(N, out_features)``."""
    net = DeepVSSGP.init(
        in_features=2,
        hidden_features=4,
        out_features=1,
        depth=3,
        n_features=8,
    )
    with handlers.seed(rng_seed=0):
        y = net(jnp.ones((5, 2)))
    assert y.shape == (5, 1)


def test_forward_shape_multi_output():
    """Multi-output works."""
    net = DeepVSSGP.init(
        in_features=3,
        hidden_features=8,
        out_features=2,
        depth=2,
        n_features=16,
    )
    with handlers.seed(rng_seed=1):
        y = net(jnp.ones((7, 3)))
    assert y.shape == (7, 2)


def test_depth_one_reduces_to_single_vssgp():
    """At ``depth=1`` we have exactly one (RFF + projection) block.

    Sample sites: ``layer_0.W_freq``, ``layer_0.lengthscale``,
    ``layer_0.W_proj``. Output dim = out_features directly.
    """
    net = DeepVSSGP.init(
        in_features=2,
        hidden_features=999,  # unused at depth=1
        out_features=3,
        depth=1,
        n_features=4,
        pyrox_name="dgp",
    )
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        y = net(jnp.zeros((6, 2)))
    assert y.shape == (6, 3)
    sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
    assert sample_names == {
        "dgp.layer_0.W_freq",
        "dgp.layer_0.lengthscale",
        "dgp.layer_0.W_proj",
    }
    assert tr["dgp.layer_0.W_freq"]["value"].shape == (2, 4)  # (D_in, M)
    assert tr["dgp.layer_0.W_proj"]["value"].shape == (8, 3)  # (2M, D_out)


def test_site_count_scales_with_depth():
    """A depth-L network registers exactly 3L sample sites."""
    for depth in (1, 2, 4):
        net = DeepVSSGP.init(
            in_features=2,
            hidden_features=4,
            out_features=1,
            depth=depth,
            n_features=8,
        )
        with handlers.trace() as tr, handlers.seed(rng_seed=0):
            net(jnp.ones((3, 2)))
        sample_sites = [k for k, v in tr.items() if v["type"] == "sample"]
        assert len(sample_sites) == 3 * depth, (
            f"depth={depth}: expected {3 * depth} sites, got {len(sample_sites)}: "
            f"{sample_sites}"
        )


def test_intermediate_layer_dims():
    """Hidden layers project to ``hidden_features``; readout to ``out_features``."""
    net = DeepVSSGP.init(
        in_features=2,
        hidden_features=5,
        out_features=1,
        depth=3,
        n_features=4,
        pyrox_name="dgp",
    )
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        net(jnp.ones((4, 2)))
    # layer 0: in=2, out=hidden=5
    assert tr["dgp.layer_0.W_freq"]["value"].shape == (2, 4)
    assert tr["dgp.layer_0.W_proj"]["value"].shape == (8, 5)
    # layer 1: in=hidden=5, out=hidden=5
    assert tr["dgp.layer_1.W_freq"]["value"].shape == (5, 4)
    assert tr["dgp.layer_1.W_proj"]["value"].shape == (8, 5)
    # layer 2 (readout): in=hidden=5, out=out_features=1
    assert tr["dgp.layer_2.W_freq"]["value"].shape == (5, 4)
    assert tr["dgp.layer_2.W_proj"]["value"].shape == (8, 1)


def test_stochastic_under_seed():
    """Different seeds give different outputs (samples are non-degenerate)."""
    net = DeepVSSGP.init(
        in_features=2,
        hidden_features=4,
        out_features=1,
        depth=2,
        n_features=8,
    )
    with handlers.seed(rng_seed=0):
        y0 = net(jnp.ones((3, 2)))
    with handlers.seed(rng_seed=1):
        y1 = net(jnp.ones((3, 2)))
    assert not jnp.allclose(y0, y1)


def test_substitute_freezes_output():
    """With all sites substituted, output is deterministic."""
    net = DeepVSSGP.init(
        in_features=2,
        hidden_features=4,
        out_features=1,
        depth=2,
        n_features=4,
    )
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        net(jnp.zeros((1, 2)))
    sub = {k: v["value"] for k, v in tr.items() if v["type"] == "sample"}
    x = jnp.linspace(-1.0, 1.0, 8).reshape(-1, 2)
    with handlers.substitute(data=sub), handlers.seed(rng_seed=42):
        y_a = net(x)
    with handlers.substitute(data=sub), handlers.seed(rng_seed=99):
        y_b = net(x)
    assert jnp.allclose(y_a, y_b)


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"depth": 0}, "depth must be"),
        ({"n_features": 0}, "n_features must be"),
        ({"lengthscale": 0.0}, "lengthscale must be"),
        ({"prior_std": -1.0}, "prior_std must be"),
        ({"in_features": 0}, "in_features must be"),
        ({"hidden_features": 0}, "hidden_features must be"),
        ({"out_features": 0}, "out_features must be"),
    ],
)
def test_init_rejects_bad_args(kwargs, msg):
    base = dict(
        in_features=2,
        hidden_features=4,
        out_features=1,
        depth=2,
        n_features=4,
        lengthscale=1.0,
        prior_std=1.0,
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        DeepVSSGP.init(**base)


def test_pyrox_name_scoping():
    """Setting ``pyrox_name`` prefixes all sample sites."""
    net = DeepVSSGP.init(
        in_features=2,
        hidden_features=4,
        out_features=1,
        depth=2,
        n_features=4,
        pyrox_name="dgp",
    )
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        net(jnp.ones((2, 2)))
    sample_names = {k for k, v in tr.items() if v["type"] == "sample"}
    expected = {
        f"dgp.layer_{i}.{nm}"
        for i in range(2)
        for nm in ("W_freq", "lengthscale", "W_proj")
    }
    assert sample_names == expected


def test_determinism_under_reseeding():
    """Two independent forward passes under the same ``rng_seed`` produce
    identical outputs — i.e., randomness comes solely from the seeded handler,
    with no hidden global counter."""
    net = DeepVSSGP.init(
        in_features=2,
        hidden_features=4,
        out_features=1,
        depth=2,
        n_features=4,
    )
    x = jr.normal(jr.PRNGKey(7), (3, 2))
    with handlers.seed(rng_seed=42):
        y_seed42_a = net(x)
    with handlers.seed(rng_seed=42):
        y_seed42_b = net(x)
    assert jnp.allclose(y_seed42_a, y_seed42_b)
