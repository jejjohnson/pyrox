"""Tests for ``pyrox.nn._heteroscedastic`` FA-noise output layers."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta

from pyrox._core.pyrox_module import PyroxModule
from pyrox.nn import MCSigmoidDenseFA, MCSoftmaxDenseFA


# --- shape & init validation ------------------------------------------------


def test_softmax_fa_output_shape_and_normalisation():
    layer = MCSoftmaxDenseFA.init(
        jr.PRNGKey(0), in_features=4, num_classes=3, rank=2, num_mc_samples=5
    )
    x = jnp.ones((6, 4))
    with handlers.seed(rng_seed=0):
        probs = layer(x)
    assert probs.shape == (6, 3)
    # MC-averaged softmax probabilities still sum to 1 along the class axis.
    assert jnp.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)


def test_sigmoid_fa_output_shape_and_range():
    layer = MCSigmoidDenseFA.init(
        jr.PRNGKey(0), in_features=4, num_classes=3, rank=2, num_mc_samples=5
    )
    x = jnp.ones((6, 4))
    with handlers.seed(rng_seed=0):
        probs = layer(x)
    assert probs.shape == (6, 3)
    assert jnp.all(probs >= 0.0)
    assert jnp.all(probs <= 1.0)


def test_softmax_fa_init_validates_positive_dims():
    with pytest.raises(ValueError, match="must all be > 0"):
        MCSoftmaxDenseFA.init(jr.PRNGKey(0), in_features=0, num_classes=3, rank=2)
    with pytest.raises(ValueError, match="must all be > 0"):
        MCSoftmaxDenseFA.init(jr.PRNGKey(0), in_features=4, num_classes=3, rank=0)
    with pytest.raises(ValueError, match="num_mc_samples"):
        MCSoftmaxDenseFA.init(
            jr.PRNGKey(0),
            in_features=4,
            num_classes=3,
            rank=2,
            num_mc_samples=0,
        )


def test_softmax_fa_validates_init_arrays_shape():
    with pytest.raises(ValueError, match="W_loc_init shape"):
        MCSoftmaxDenseFA(
            in_features=4,
            num_classes=3,
            rank=2,
            W_loc_init=jnp.zeros((3, 3)),  # wrong: should be (4, 3)
            W_scale_init=jnp.zeros((4, 6)),
            W_diag_init=jnp.zeros((4, 3)),
        )


def test_softmax_fa_requires_init_arrays():
    with pytest.raises(ValueError, match=r"\.init\(key"):
        MCSoftmaxDenseFA(in_features=4, num_classes=3, rank=2)


# --- site registration ------------------------------------------------------


def test_softmax_fa_registers_param_sites():
    layer = MCSoftmaxDenseFA.init(
        jr.PRNGKey(0),
        in_features=4,
        num_classes=3,
        rank=2,
        num_mc_samples=3,
        pyrox_name="hs",
    )
    x = jnp.ones((1, 4))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(x)
    for k in (
        "hs.W_loc",
        "hs.b_loc",
        "hs.W_scale",
        "hs.b_scale",
        "hs.W_diag",
        "hs.b_diag",
    ):
        assert tr[k]["type"] == "param", f"{k} should be a param site"


def test_sigmoid_fa_registers_param_sites():
    layer = MCSigmoidDenseFA.init(
        jr.PRNGKey(0),
        in_features=4,
        num_classes=3,
        rank=2,
        num_mc_samples=3,
        pyrox_name="hs",
    )
    x = jnp.ones((1, 4))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(x)
    assert tr["hs.W_loc"]["type"] == "param"
    assert tr["hs.W_scale"]["type"] == "param"
    assert tr["hs.W_diag"]["type"] == "param"


# --- stochasticity ----------------------------------------------------------


def test_softmax_fa_stochastic_across_seeds():
    layer = MCSoftmaxDenseFA.init(
        jr.PRNGKey(0),
        in_features=4,
        num_classes=3,
        rank=2,
        num_mc_samples=3,
        diag_init_bias=0.0,  # nontrivial init noise so MC sampling matters
    )
    x = jr.normal(jr.PRNGKey(7), (5, 4))
    with handlers.seed(rng_seed=0):
        y1 = layer(x)
    with handlers.seed(rng_seed=1):
        y2 = layer(x)
    assert not jnp.allclose(y1, y2)


def test_softmax_fa_low_noise_limit_approaches_softmax_of_mu():
    """log_sigma << 0 + zero scale ⇒ output ≈ softmax(W_loc x + b_loc)."""
    layer = MCSoftmaxDenseFA.init(
        jr.PRNGKey(0),
        in_features=4,
        num_classes=3,
        rank=2,
        num_mc_samples=5,
        diag_init_bias=-10.0,
        scale_init_factor=0.0,
    )
    x = jnp.ones((1, 4))
    with handlers.seed(rng_seed=0):
        probs = layer(x)
    # Reproduce mean logits from the same init values.
    mu = x @ layer.W_loc_init + jnp.zeros(3)
    expected = jnp.exp(mu) / jnp.exp(mu).sum(axis=-1, keepdims=True)
    assert jnp.allclose(probs, expected, atol=1e-3)


# --- SVI smoke test ---------------------------------------------------------


def test_softmax_fa_svi_loss_decreases_on_classification():
    """Canonical SVI pattern: forward outside the data plate, obs inside."""
    rng = jr.PRNGKey(0)
    x = jr.normal(jr.PRNGKey(11), (32, 3))
    # Two well-separated classes from a linear boundary.
    y = (x[:, 0] > 0).astype(jnp.int32)

    def model(x, y=None):
        layer = MCSoftmaxDenseFA.init(
            jr.PRNGKey(1),
            in_features=3,
            num_classes=2,
            rank=2,
            num_mc_samples=3,
            pyrox_name="hs",
        )
        probs = layer(x)
        log_probs = jnp.log(probs + 1e-12)
        with numpyro.plate("data", x.shape[0]):
            numpyro.sample("obs", dist.Categorical(logits=log_probs), obs=y)

    guide = AutoDelta(model)
    svi = SVI(model, guide, numpyro.optim.Adam(5e-2), Trace_ELBO())
    state = svi.init(rng, x, y)
    losses = []
    for _ in range(40):
        state, loss = svi.update(state, x, y)
        losses.append(float(loss))
    assert all(jnp.isfinite(jnp.asarray(losses)))
    assert losses[-1] < losses[0] - 1.0


# --- type checks ------------------------------------------------------------


def test_layers_are_pyrox_modules():
    soft = MCSoftmaxDenseFA.init(jr.PRNGKey(0), in_features=2, num_classes=2, rank=1)
    sig = MCSigmoidDenseFA.init(jr.PRNGKey(0), in_features=2, num_classes=2, rank=1)
    assert isinstance(soft, PyroxModule)
    assert isinstance(sig, PyroxModule)
