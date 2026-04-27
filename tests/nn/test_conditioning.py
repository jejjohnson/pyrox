"""Tests for ``pyrox.nn._conditioning``.

Covers the unified conditioning API: :class:`AbstractConditioner` plus
the three concrete conditioners (Concat, Affine, Hyper), their Bayesian
variants, the :class:`ConditionedINR` composite, and the
:func:`HyperSIREN` constructor.
"""

from __future__ import annotations

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest
from numpyro import handlers

from pyrox.nn import (
    SIREN,
    AffineModulation,
    BayesianAffineModulation,
    BayesianConcatConditioner,
    BayesianHyperLinear,
    ConcatConditioner,
    ConditionedINR,
    ConditionedRFFNet,
    FiLM,
    HyperFourierFeatures,
    HyperLinear,
    HyperSIREN,
)


# ---------------------------------------------------------------------------
# 1. Per-conditioner shape contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda key: ConcatConditioner.init(num_features=8, cond_dim=3, key=key),
        lambda key: AffineModulation.init(num_features=8, cond_dim=3, key=key),
    ],
    ids=["concat", "affine"],
)
def test_conditioner_output_shape_2d(factory):
    cond = factory(jr.key(0))
    h = jnp.ones((6, 8))
    z = jnp.ones((6, 3))
    assert cond(h, z).shape == (6, 8)


def test_hyper_linear_output_shape_2d():
    hyper = HyperLinear.init(target_in=4, target_out=8, cond_dim=3, key=jr.key(0))
    x = jnp.ones((6, 4))
    z = jnp.ones((6, 3))
    assert hyper(x, z).shape == (6, 8)


# ---------------------------------------------------------------------------
# 2. Broadcast: z=(K,) is reused across all rows of h
# ---------------------------------------------------------------------------


def test_affine_modulation_broadcasts_scalar_z():
    cond = AffineModulation.init(num_features=8, cond_dim=3, key=jr.key(0))
    h = jr.normal(jr.key(1), (6, 8))
    z = jr.normal(jr.key(2), (3,))
    z_expanded = einops.repeat(z, "k -> n k", n=6)
    np.testing.assert_allclose(
        np.asarray(cond(h, z)), np.asarray(cond(h, z_expanded)), atol=1e-6
    )


def test_concat_conditioner_broadcasts_scalar_z():
    cond = ConcatConditioner.init(num_features=8, cond_dim=3, key=jr.key(0))
    h = jr.normal(jr.key(1), (6, 8))
    z = jr.normal(jr.key(2), (3,))
    z_expanded = einops.repeat(z, "k -> n k", n=6)
    np.testing.assert_allclose(
        np.asarray(cond(h, z)), np.asarray(cond(h, z_expanded)), atol=1e-6
    )


# ---------------------------------------------------------------------------
# 3. AffineModulation: identity-at-init when bias=0 and z=0
# ---------------------------------------------------------------------------


def test_affine_modulation_identity_at_init_with_zero_z():
    """``one_plus_tanh`` with bias=0 and z=0 ⇒ γ=1, β=0 ⇒ y=h exactly."""
    cond = AffineModulation.init(num_features=8, cond_dim=3, key=jr.key(0))
    h = jr.normal(jr.key(1), (6, 8))
    z = jnp.zeros((6, 3))
    np.testing.assert_allclose(np.asarray(cond(h, z)), np.asarray(h), atol=1e-6)


def test_film_alias_is_affine_modulation():
    """``FiLM`` is a backwards-compat alias for ``AffineModulation``."""
    assert FiLM is AffineModulation


# ---------------------------------------------------------------------------
# 4. AffineModulation.log_det only valid for gamma_activation='exp'
# ---------------------------------------------------------------------------


def test_affine_modulation_log_det_exp_matches_sum_raw_gamma():
    cond = AffineModulation.init(
        num_features=4, cond_dim=2, key=jr.key(0), gamma_activation="exp"
    )
    z = jr.normal(jr.key(1), (3, 2))
    raw = jax.vmap(cond.generator)(z)
    raw_gamma = raw[:, 4:]  # second half on the feature axis (β | raw_γ)
    expected = jnp.sum(raw_gamma, axis=-1)
    np.testing.assert_allclose(
        np.asarray(cond.log_det(z)), np.asarray(expected), atol=1e-6
    )


def test_affine_modulation_log_det_raises_when_not_exp():
    cond = AffineModulation.init(num_features=4, cond_dim=2, key=jr.key(0))
    with pytest.raises(NotImplementedError, match="exp"):
        cond.log_det(jnp.zeros((2,)))


# ---------------------------------------------------------------------------
# 5. HyperLinear: shared vs per-sample dispatch agrees on broadcast z
# ---------------------------------------------------------------------------


def test_hyper_linear_shared_vs_per_sample_agree_on_broadcast():
    """Load-bearing: einops dispatch on z.ndim must agree when z is broadcast."""
    hyper = HyperLinear.init(target_in=4, target_out=6, cond_dim=3, key=jr.key(0))
    x = jr.normal(jr.key(1), (5, 4))
    z = jr.normal(jr.key(2), (3,))
    z_expanded = einops.repeat(z, "k -> n k", n=5)
    np.testing.assert_allclose(
        np.asarray(hyper(x, z)), np.asarray(hyper(x, z_expanded)), atol=1e-6
    )


def test_hyper_linear_persample_uses_distinct_weights():
    """Different rows of z should produce different generated W."""
    hyper = HyperLinear.init(target_in=4, target_out=6, cond_dim=3, key=jr.key(0))
    x = jnp.ones((1, 4))
    z = jr.normal(jr.key(1), (2, 3)) * 5.0  # large magnitude → distinct outputs
    z_a = z[0:1]
    z_b = z[1:2]
    y_a = hyper(x, z_a)
    y_b = hyper(x, z_b)
    assert not jnp.allclose(y_a, y_b, atol=1e-3)


# ---------------------------------------------------------------------------
# 6. JIT smoke
# ---------------------------------------------------------------------------


def test_conditioners_jit():
    h = jnp.ones((4, 8))
    z = jnp.ones((4, 3))
    for cond in (
        ConcatConditioner.init(num_features=8, cond_dim=3, key=jr.key(0)),
        AffineModulation.init(num_features=8, cond_dim=3, key=jr.key(1)),
    ):
        y = jax.jit(lambda c=cond: c(h, z))()
        assert jnp.all(jnp.isfinite(y))

    hyper = HyperLinear.init(target_in=4, target_out=8, cond_dim=3, key=jr.key(2))
    y = jax.jit(lambda: hyper(jnp.ones((4, 4)), z))()
    assert jnp.all(jnp.isfinite(y))


# ---------------------------------------------------------------------------
# 7. Bayesian variants only register generator sites
# ---------------------------------------------------------------------------


def test_bayesian_affine_modulation_registers_two_generator_sites():
    cond = BayesianAffineModulation.init(num_features=8, cond_dim=3)
    with handlers.seed(rng_seed=0), handlers.trace() as tr:
        cond(jnp.ones((4, 8)), jnp.ones((4, 3)))
    sites = sorted(tr.keys())
    assert len(sites) == 2
    assert all(s.endswith(".gen_W") or s.endswith(".gen_b") for s in sites)


def test_bayesian_hyper_linear_registers_only_generator_sites():
    """Load-bearing: target weights are *generated*, not sampled."""
    cond = BayesianHyperLinear.init(target_in=4, target_out=6, cond_dim=3)
    with handlers.seed(rng_seed=0), handlers.trace() as tr:
        cond(jnp.ones((4, 4)), jnp.ones((4, 3)))
    sites = sorted(tr.keys())
    assert len(sites) == 2
    assert all(s.endswith(".gen_W") or s.endswith(".gen_b") for s in sites)
    # No site mentions "target" or anything inner-network-y.
    assert not any("target" in s or "inner" in s for s in sites)


def test_bayesian_concat_conditioner_registers_two_sites():
    cond = BayesianConcatConditioner.init(num_features=8, cond_dim=3)
    with handlers.seed(rng_seed=0), handlers.trace() as tr:
        cond(jnp.ones((4, 8)), jnp.ones((4, 3)))
    assert len(tr) == 2


# ---------------------------------------------------------------------------
# 8. ConditionedINR composite — wraps SIREN end-to-end
# ---------------------------------------------------------------------------


def test_conditioned_siren_output_shape():
    inner = SIREN.init(2, 16, 1, depth=4, key=jr.key(0))
    wrapped = ConditionedINR.init(
        inner, conditioner_cls=AffineModulation, cond_dim=4, key=jr.key(1)
    )
    y = wrapped(jnp.ones((10, 2)), jnp.ones((10, 4)))
    assert y.shape == (10, 1)


def test_conditioned_siren_gradient_flows_to_inner_and_conditioners():
    inner = SIREN.init(2, 16, 1, depth=3, key=jr.key(0))
    wrapped = ConditionedINR.init(
        inner, conditioner_cls=AffineModulation, cond_dim=3, key=jr.key(1)
    )
    x = jnp.ones((4, 2))
    z = jnp.ones((4, 3))

    def loss(model):
        return jnp.sum(model(x, z) ** 2)

    grads = eqx.filter_grad(loss)(wrapped)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
    # At least one conditioner leaf must have non-zero gradient.
    cond_grads = jax.tree_util.tree_leaves(eqx.filter(grads.conditioners, eqx.is_array))
    assert any(jnp.any(jnp.abs(g) > 0) for g in cond_grads)


def test_conditioned_inr_input_mode():
    """``mode='input'`` runs a single concat head before inner."""
    inner = SIREN.init(5, 16, 1, depth=3, key=jr.key(0))  # in_features = 2 + 3
    wrapped = ConditionedINR.init(
        inner,
        conditioner_cls=ConcatConditioner,
        cond_dim=3,
        key=jr.key(1),
        mode="input",
    )
    # The head projects (in_features=5, cond_dim=3) → (5,) per row, then runs SIREN.
    # We fake-feed `x` of size 5 (matching inner.in_features) since input mode
    # uses h = x of size num_features = inner.in_features.
    y = wrapped(jnp.ones((4, 5)), jnp.ones((4, 3)))
    assert y.shape == (4, 1)


# ---------------------------------------------------------------------------
# 9. HyperSIREN — NIF composite
# ---------------------------------------------------------------------------


class _Identity(eqx.Module):
    """Test-only parameter net that just passes mu through."""

    def __call__(self, mu):
        return mu


def test_hyper_siren_output_shape():
    nif = HyperSIREN(
        in_features=2,
        hidden_features=16,
        out_features=1,
        depth=3,
        cond_dim=3,
        parameter_net=_Identity(),
        key=jr.key(0),
    )
    y = nif(jnp.ones((8, 2)), jnp.ones((3,)))
    assert y.shape == (8, 1)


def test_hyper_siren_per_layer_calibration_uses_siren_W_limit():
    """Each per-layer init_scale should depend on the SIREN regime half-width."""
    nif = HyperSIREN(
        in_features=2,
        hidden_features=16,
        out_features=1,
        depth=4,
        cond_dim=3,
        parameter_net=_Identity(),
        key=jr.key(0),
    )
    # First layer (Sitzmann "first" regime) has half-width 1/in_features ≈ 0.5;
    # hidden layers have a much smaller half-width because of the omega divisor.
    # Therefore the first hyper layer's effective generator weight magnitude
    # should be larger than a hidden hyper layer's.
    first_max = float(jnp.max(jnp.abs(nif.hyper_layers[0].generator.weight)))
    hidden_max = float(jnp.max(jnp.abs(nif.hyper_layers[1].generator.weight)))
    assert first_max > hidden_max


def test_hyper_siren_fits_parametric_family():
    """Load-bearing behavioural anchor: HyperSIREN fits ``u(x; a) = sin(a x)``."""
    nif = HyperSIREN(
        in_features=1,
        hidden_features=32,
        out_features=1,
        depth=3,
        cond_dim=1,
        parameter_net=_Identity(),
        key=jr.key(0),
    )

    rng = jr.key(42)
    k_a, k_x = jr.split(rng)
    n = 256
    a = jr.uniform(k_a, (n, 1), minval=1.0, maxval=3.0)
    x = jr.uniform(k_x, (n, 1), minval=-1.0, maxval=1.0)
    y = jnp.sin(a * x)

    @eqx.filter_jit
    def step(model, opt_state):
        def loss(m):
            preds = jax.vmap(m, in_axes=(0, 0))(x, a)
            return jnp.mean((preds - y) ** 2)

        loss_v, grads = eqx.filter_value_and_grad(loss)(model)
        updates, new_state = opt.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_state, loss_v

    opt = optax.adam(3e-3)
    state = opt.init(eqx.filter(nif, eqx.is_array))

    final_loss = jnp.array(jnp.inf)
    for _ in range(200):
        nif, state, final_loss = step(nif, state)

    assert float(final_loss) < 0.05, f"final MSE was {float(final_loss)}"


# ---------------------------------------------------------------------------
# 10. Validation: clear error messages on shape mismatch
# ---------------------------------------------------------------------------


def test_affine_modulation_rejects_mismatched_num_features():
    cond = AffineModulation.init(num_features=8, cond_dim=3, key=jr.key(0))
    with pytest.raises(ValueError, match="num_features"):
        cond(jnp.ones((4, 7)), jnp.ones((4, 3)))


def test_affine_modulation_rejects_mismatched_cond_dim():
    cond = AffineModulation.init(num_features=8, cond_dim=3, key=jr.key(0))
    with pytest.raises(ValueError, match="cond_dim"):
        cond(jnp.ones((4, 8)), jnp.ones((4, 4)))


def test_hyper_linear_rejects_mismatched_target_in():
    hyper = HyperLinear.init(target_in=4, target_out=6, cond_dim=3, key=jr.key(0))
    with pytest.raises(ValueError, match="target_in"):
        hyper(jnp.ones((4, 5)), jnp.ones((4, 3)))


def test_hyper_siren_rejects_invalid_depth():
    with pytest.raises(ValueError, match="depth"):
        HyperSIREN(
            in_features=2,
            hidden_features=8,
            out_features=1,
            depth=1,
            cond_dim=3,
            parameter_net=_Identity(),
            key=jr.key(0),
        )


# ---------------------------------------------------------------------------
# 11. 1-D input handling — squeeze convention
# ---------------------------------------------------------------------------


def test_affine_modulation_handles_1d_input():
    cond = AffineModulation.init(num_features=4, cond_dim=2, key=jr.key(0))
    y = cond(jnp.ones((4,)), jnp.ones((2,)))
    assert y.shape == (4,)


def test_hyper_linear_handles_1d_input():
    hyper = HyperLinear.init(target_in=3, target_out=4, cond_dim=2, key=jr.key(0))
    y = hyper(jnp.ones((3,)), jnp.ones((2,)))
    assert y.shape == (4,)


# ---------------------------------------------------------------------------
# 12. HyperFourierFeatures — parameter-net-generated (W, b, lengthscale)
# ---------------------------------------------------------------------------


def _flat_size(in_features: int, n_features: int) -> int:
    return in_features * n_features + n_features + 1


def _make_pnet(in_size: int, out_size: int, *, key) -> eqx.nn.MLP:
    return eqx.nn.MLP(
        in_size=in_size, out_size=out_size, width_size=16, depth=2, key=key
    )


def test_hyper_fourier_features_output_shape_shared_z():
    in_f, n_f, K = 1, 8, 3
    pnet = _make_pnet(K, _flat_size(in_f, n_f), key=jr.key(0))
    hff = HyperFourierFeatures.init(
        parameter_net=pnet, in_features=in_f, n_features=n_f, cond_dim=K
    )
    out = hff(jnp.ones((6, in_f)), jnp.ones((K,)))
    assert out.shape == (6, 2 * n_f)


def test_hyper_fourier_features_output_shape_per_sample_z():
    in_f, n_f, K = 2, 8, 3
    pnet = _make_pnet(K, _flat_size(in_f, n_f), key=jr.key(0))
    hff = HyperFourierFeatures.init(
        parameter_net=pnet, in_features=in_f, n_features=n_f, cond_dim=K
    )
    out = hff(jnp.ones((6, in_f)), jnp.ones((6, K)))
    assert out.shape == (6, 2 * n_f)


def test_hyper_fourier_features_shared_vs_per_sample_agree_on_broadcast():
    """Shared and per-sample paths must agree when ``z`` is broadcast."""
    in_f, n_f, K = 2, 8, 3
    pnet = _make_pnet(K, _flat_size(in_f, n_f), key=jr.key(0))
    hff = HyperFourierFeatures.init(
        parameter_net=pnet, in_features=in_f, n_features=n_f, cond_dim=K
    )
    x = jr.normal(jr.key(1), (5, in_f))
    z = jr.normal(jr.key(2), (K,))
    z_expanded = einops.repeat(z, "k -> n k", n=5)
    np.testing.assert_allclose(
        np.asarray(hff(x, z)), np.asarray(hff(x, z_expanded)), atol=1e-5
    )


def test_hyper_fourier_features_validates_pnet_output_shape():
    """``init`` must reject a parameter_net whose output shape is wrong."""
    in_f, n_f, K = 1, 8, 3
    bad = _make_pnet(K, _flat_size(in_f, n_f) + 1, key=jr.key(0))  # off by 1
    with pytest.raises(ValueError, match="parameter_net"):
        HyperFourierFeatures.init(
            parameter_net=bad, in_features=in_f, n_features=n_f, cond_dim=K
        )


def test_hyper_fourier_features_rejects_mismatched_input():
    in_f, n_f, K = 1, 8, 3
    pnet = _make_pnet(K, _flat_size(in_f, n_f), key=jr.key(0))
    hff = HyperFourierFeatures.init(
        parameter_net=pnet, in_features=in_f, n_features=n_f, cond_dim=K
    )
    with pytest.raises(ValueError, match="in_features"):
        hff(jnp.ones((4, 2)), jnp.ones((K,)))


def test_conditioned_rff_net_output_shape():
    in_f, n_f, K, out_f = 1, 16, 4, 1
    pnet = _make_pnet(K, _flat_size(in_f, n_f), key=jr.key(0))
    feat = HyperFourierFeatures.init(
        parameter_net=pnet, in_features=in_f, n_features=n_f, cond_dim=K
    )
    net = ConditionedRFFNet.init(feat=feat, out_features=out_f, key=jr.key(1))
    y = net(jnp.zeros((10, in_f)), jnp.zeros((10, K)))
    assert y.shape == (10, out_f)


def test_conditioned_rff_net_fits_parametric_family():
    """Behavioural anchor: Hyper-RFF fits ``u(x; a) = sin(a x)``."""
    in_f, n_f, K, out_f = 1, 32, 4, 1
    keys = jr.split(jr.key(0), 3)
    pnet = _make_pnet(K, _flat_size(in_f, n_f), key=keys[0])
    feat = HyperFourierFeatures.init(
        parameter_net=pnet, in_features=in_f, n_features=n_f, cond_dim=K
    )
    model = ConditionedRFFNet.init(feat=feat, out_features=out_f, key=keys[1])

    rng = keys[2]
    k_a, k_x = jr.split(rng)
    n = 256
    a = jr.uniform(k_a, (n, 1), minval=1.0, maxval=3.0)
    x = jr.uniform(k_x, (n, 1), minval=-3.0, maxval=3.0)
    z = jnp.broadcast_to(a, (n, K))
    y = jnp.sin(a * x)

    @eqx.filter_jit
    def step(model, opt_state):
        def loss(m):
            preds = jax.vmap(m, in_axes=(0, 0))(x, z)
            return jnp.mean((preds - y) ** 2)

        loss_v, grads = eqx.filter_value_and_grad(loss)(model)
        updates, new_state = opt.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_state, loss_v

    opt = optax.adam(3e-3)
    state = opt.init(eqx.filter(model, eqx.is_array))

    final_loss = jnp.array(jnp.inf)
    for _ in range(400):
        model, state, final_loss = step(model, state)

    assert float(final_loss) < 0.05, f"final MSE was {float(final_loss)}"


def test_hyper_fourier_features_handles_1d_input():
    in_f, n_f, K = 1, 8, 2
    pnet = _make_pnet(K, _flat_size(in_f, n_f), key=jr.key(0))
    hff = HyperFourierFeatures.init(
        parameter_net=pnet, in_features=in_f, n_features=n_f, cond_dim=K
    )
    out = hff(jnp.ones((in_f,)), jnp.ones((K,)))
    assert out.shape == (2 * n_f,)
