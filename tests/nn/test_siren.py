"""Tests for pyrox.nn SIREN layers.

Covers :class:`pyrox.nn.SirenDense`, :class:`pyrox.nn.SIREN`, and
:class:`pyrox.nn.BayesianSIREN`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest
from numpyro import handlers

from pyrox.nn import SIREN, BayesianSIREN, SirenDense
from pyrox.nn._layers import _siren_W_limit


# ---------------------------------------------------------------------------
# 1. SirenDense output shape
# ---------------------------------------------------------------------------


def test_siren_dense_output_shape():
    layer = SirenDense.init(3, 16, key=jr.PRNGKey(0), layer_type="hidden")
    x = jnp.ones((5, 3))
    y = layer(x)
    assert y.shape == (5, 16)


# ---------------------------------------------------------------------------
# 2. Activation regimes — first/hidden apply sin; last does not
# ---------------------------------------------------------------------------


def test_siren_dense_activation_regimes():
    key = jr.PRNGKey(1)
    in_f, out_f = 4, 8
    omega = 30.0

    # Build three layers that share the same W and b but differ by layer_type.
    k_w, k_b, _k_rest = jr.split(key, 3)
    w_limit = _siren_W_limit("hidden", in_f, omega)
    b_limit = 1.0 / jnp.sqrt(in_f)
    W = jr.uniform(k_w, (in_f, out_f), minval=-w_limit, maxval=w_limit)
    b = jr.uniform(k_b, (out_f,), minval=-b_limit, maxval=b_limit)

    x = jr.normal(jr.PRNGKey(99), (6, in_f))

    def make_layer(lt: str) -> SirenDense:
        # Override W and b with the shared values so layers are comparable.
        return SirenDense(
            W=W,
            b=b,
            omega=omega,
            in_features=in_f,
            out_features=out_f,
            layer_type=lt,
            c=6.0,
            pyrox_name=None,
        )

    layer_first = make_layer("first")
    layer_hidden = make_layer("hidden")
    layer_last = make_layer("last")

    pre = x @ W + b

    # first and hidden should apply sin(omega * pre)
    np.testing.assert_allclose(
        np.asarray(layer_first(x)),
        np.asarray(jnp.sin(omega * pre)),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(layer_hidden(x)),
        np.asarray(jnp.sin(omega * pre)),
        atol=1e-5,
    )
    # last should be a plain linear
    np.testing.assert_allclose(
        np.asarray(layer_last(x)),
        np.asarray(pre),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# 3. SIREN composite output shape
# ---------------------------------------------------------------------------


def test_siren_composite_output_shape():
    net = SIREN.init(2, 32, 4, depth=5, key=jr.PRNGKey(2))
    x = jnp.ones((10, 2))
    y = net(x)
    assert y.shape == (10, 4)


# ---------------------------------------------------------------------------
# 4. Activation variance preserved (Sitzmann Theorem 1 smoke test)
# ---------------------------------------------------------------------------


def test_siren_activation_variance_preserved():
    """Hidden-layer activations should have variance in [0.3, 3.0] at every layer."""
    depth = 5
    hidden = 64
    net = SIREN.init(64, hidden, 1, depth=depth, key=jr.PRNGKey(3))

    x = jr.normal(jr.PRNGKey(42), (4096, 64))
    z = x
    # Check each non-readout layer.
    # Sitzmann Theorem 1 (§3.2) shows that with the prescribed init the
    # post-sin activations are approximately N(0,1) at every layer.
    # We use a relaxed window [0.3, 3.0] — roughly ±1.5 orders of magnitude
    # from the target variance of 1.0 — to avoid test flakiness while still
    # catching any init regime that causes explosive or collapsing activations.
    for layer in net.layers[:-1]:
        z = layer(z)
        var = float(jnp.var(z))
        assert 0.3 <= var <= 3.0, (
            f"Hidden-layer variance {var:.4f} outside [0.3, 3.0] — "
            "SIREN init may be incorrect."
        )


# ---------------------------------------------------------------------------
# 5. JIT
# ---------------------------------------------------------------------------


def test_siren_jits():
    net = SIREN.init(2, 32, 1, depth=4, key=jr.PRNGKey(4))

    @jax.jit
    def fwd(x: jax.Array) -> jax.Array:
        return net(x)

    y = fwd(jnp.ones((8, 2)))
    assert jnp.all(jnp.isfinite(y))


# ---------------------------------------------------------------------------
# 6. Gradient flows through every W and b leaf
# ---------------------------------------------------------------------------


def test_siren_gradient_flows():
    net = SIREN.init(2, 32, 1, depth=4, key=jr.PRNGKey(5))
    x = jr.normal(jr.PRNGKey(55), (16, 2))

    def loss(params: SIREN) -> jax.Array:
        return jnp.mean(params(x) ** 2)

    grads = jax.grad(loss)(net)
    for i, layer in enumerate(grads.layers):
        assert jnp.all(jnp.isfinite(layer.W)), f"layer {i} W grad not finite"
        assert jnp.all(jnp.isfinite(layer.b)), f"layer {i} b grad not finite"
        assert jnp.any(layer.W != 0.0), f"layer {i} W grad is all zero"
        assert jnp.any(layer.b != 0.0), f"layer {i} b grad is all zero"


# ---------------------------------------------------------------------------
# 7. Fits a 1-D sinusoid (regression sanity, MSE < 1e-3 in 500 steps)
# ---------------------------------------------------------------------------


def test_siren_fits_1d_sinusoid():
    n_points = 256
    x = jnp.linspace(-jnp.pi, jnp.pi, n_points).reshape(-1, 1)
    y_true = jnp.sin(2.0 * x)

    net = SIREN.init(1, 64, 1, depth=3, key=jr.PRNGKey(6))
    optim = optax.adam(1e-3)
    opt_state = optim.init(net)

    @jax.jit
    def step(
        params: SIREN, state: optax.OptState
    ) -> tuple[SIREN, optax.OptState, jax.Array]:
        def loss_fn(p: SIREN) -> jax.Array:
            return jnp.mean((p(x) - y_true) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_state = optim.update(grads, state)
        new_params = optax.apply_updates(params, updates)  # type: ignore[arg-type]
        return new_params, new_state, loss

    for _ in range(500):
        net, opt_state, loss = step(net, opt_state)

    assert float(loss) < 1e-3, f"SIREN fit MSE {float(loss):.6f} >= 1e-3"


# ---------------------------------------------------------------------------
# 8. BayesianSIREN registers exactly 2·depth sample sites
# ---------------------------------------------------------------------------


def test_bayesian_siren_registers_sites():
    depth = 4
    net = BayesianSIREN.init(2, 16, 1, depth=depth, key=jr.PRNGKey(7), pyrox_name="bs")
    x = jnp.ones((3, 2))

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        net(x)

    sample_sites = {n for n, s in tr.items() if s["type"] == "sample"}
    expected_W = {f"bs.layer_{i}.W" for i in range(depth)}
    expected_b = {f"bs.layer_{i}.b" for i in range(depth)}
    assert expected_W <= sample_sites, f"Missing W sites: {expected_W - sample_sites}"
    assert expected_b <= sample_sites, f"Missing b sites: {expected_b - sample_sites}"
    assert len(sample_sites) == 2 * depth, (
        f"Expected {2 * depth} sample sites, got {len(sample_sites)}: {sample_sites}"
    )


# ---------------------------------------------------------------------------
# 9. BayesianSIREN prior stddev matches prior_std · init_scale per layer
# ---------------------------------------------------------------------------


def test_bayesian_siren_prior_respects_init_scale():
    depth = 3
    hidden = 512
    prior_std = 2.0
    net = BayesianSIREN.init(
        hidden,
        hidden,
        1,
        depth=depth,
        key=jr.PRNGKey(8),
        prior_std=prior_std,
        pyrox_name="bsp",
    )
    x = jnp.ones((1, hidden))

    n_samples = 32
    w_samples: dict[int, list[np.ndarray]] = {i: [] for i in range(depth)}

    for s in range(n_samples):
        with handlers.trace() as tr, handlers.seed(rng_seed=s):
            net(x)
        for i in range(depth):
            w_samples[i].append(np.asarray(tr[f"bsp.layer_{i}.W"]["value"]))

    for i, layer in enumerate(net.layers):
        expected_std = prior_std * _siren_W_limit(
            layer.layer_type, layer.in_features, layer.omega, layer.c
        )
        all_w = np.concatenate([w.ravel() for w in w_samples[i]])
        empirical_std = float(np.std(all_w))
        assert abs(empirical_std - expected_std) / expected_std < 0.20, (
            f"Layer {i} W prior std {empirical_std:.6f} deviates "
            f">20% from expected {expected_std:.6f}"
        )


# ---------------------------------------------------------------------------
# 10. SIREN rejects invalid depth
# ---------------------------------------------------------------------------


def test_siren_rejects_invalid_depth():
    for bad_depth in (0, 1):
        with pytest.raises(ValueError, match="depth"):
            SIREN.init(2, 16, 1, depth=bad_depth, key=jr.PRNGKey(0))


# ---------------------------------------------------------------------------
# 11. SirenDense rejects invalid layer_type
# ---------------------------------------------------------------------------


def test_siren_rejects_invalid_layer_type():
    with pytest.raises(ValueError, match="layer_type"):
        SirenDense.init(4, 8, key=jr.PRNGKey(0), layer_type="mystery")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 12. SIREN wires first_omega and hidden_omega to the correct layers
# ---------------------------------------------------------------------------


def test_siren_first_omega_differs_from_hidden():
    net = SIREN.init(
        2,
        32,
        1,
        depth=4,
        key=jr.PRNGKey(9),
        first_omega=60.0,
        hidden_omega=30.0,
    )
    assert net.layers[0].omega == 60.0, "first layer should have first_omega=60"
    assert net.layers[1].omega == 30.0, "second layer should have hidden_omega=30"
