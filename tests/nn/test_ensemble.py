"""Tests for ``pyrox.nn._ensemble`` rank-1 ensemble layers."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from pyrox._core.pyrox_module import PyroxModule
from pyrox.nn import DenseRank1, LayerNormEnsemble, MultiHeadAttentionBE


# --- forward shape & init ---------------------------------------------------


def test_dense_rank1_output_shape():
    layer = DenseRank1.init(
        jr.PRNGKey(0), in_features=4, out_features=2, ensemble_size=3
    )
    x = jnp.ones((5, 4))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert y.shape == (3, 5, 2)


def test_dense_rank1_no_bias_output_shape():
    layer = DenseRank1.init(
        jr.PRNGKey(0), in_features=4, out_features=2, ensemble_size=3, bias=False
    )
    x = jnp.ones((5, 4))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert y.shape == (3, 5, 2)


def test_dense_rank1_init_validates_positive_dims():
    with pytest.raises(ValueError, match="must all be > 0"):
        DenseRank1.init(jr.PRNGKey(0), in_features=0, out_features=2, ensemble_size=3)
    with pytest.raises(ValueError, match="must all be > 0"):
        DenseRank1.init(jr.PRNGKey(0), in_features=4, out_features=2, ensemble_size=0)


def test_dense_rank1_prior_scale_only_validated_in_bayesian_mode():
    """Deterministic configs may carry a sentinel ``prior_scale`` that's unused."""
    # bayesian=False ⇒ prior_scale=0.0 is fine (just unused).
    layer = DenseRank1.init(
        jr.PRNGKey(0),
        in_features=3,
        out_features=2,
        ensemble_size=2,
        bayesian=False,
        prior_scale=0.0,
    )
    assert layer.bayesian is False
    # bayesian=True ⇒ prior_scale must be positive.
    with pytest.raises(ValueError, match="prior_scale must be > 0"):
        DenseRank1.init(
            jr.PRNGKey(0),
            in_features=3,
            out_features=2,
            ensemble_size=2,
            bayesian=True,
            prior_scale=0.0,
        )


def test_dense_rank1_supports_arbitrary_batch_dims():
    """Forward accepts ``(*batch, D_in)`` and returns ``(M, *batch, D_out)``."""
    layer = DenseRank1.init(
        jr.PRNGKey(0), in_features=4, out_features=2, ensemble_size=3
    )
    x_3d = jnp.ones((6, 5, 4))  # e.g. (batch, time, D_in)
    with handlers.seed(rng_seed=0):
        y_3d = layer(x_3d)
    assert y_3d.shape == (3, 6, 5, 2)

    # Unbatched single example: (D_in,) → (M, D_out).
    x_1d = jnp.ones((4,))
    with handlers.seed(rng_seed=0):
        y_1d = layer(x_1d)
    assert y_1d.shape == (3, 2)


def test_dense_rank1_validates_init_arrays_shape():
    """Bypassing ``init`` with mis-sized arrays must fail loudly."""
    with pytest.raises(ValueError, match="W_init shape"):
        DenseRank1(
            in_features=4,
            out_features=2,
            ensemble_size=3,
            W_init=jnp.zeros((3, 2)),  # wrong: should be (4, 2)
            r_init=jnp.ones((3, 2)),
            s_init=jnp.ones((3, 4)),
        )


def test_dense_rank1_requires_init_arrays():
    with pytest.raises(ValueError, match=r"DenseRank1\.init"):
        DenseRank1(in_features=4, out_features=2, ensemble_size=3)


# --- deterministic mode (BatchEnsemble) ------------------------------------


def test_dense_rank1_deterministic_registers_param_sites():
    layer = DenseRank1.init(
        jr.PRNGKey(0),
        in_features=4,
        out_features=2,
        ensemble_size=3,
        pyrox_name="be",
    )
    x = jnp.ones((5, 4))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(x)
    for k in ("be.W", "be.r", "be.s", "be.b"):
        assert tr[k]["type"] == "param", f"{k} should be a param site"


def test_dense_rank1_deterministic_is_deterministic_across_seeds():
    """In BatchEnsemble mode the forward is purely deterministic."""
    layer = DenseRank1.init(
        jr.PRNGKey(0), in_features=4, out_features=2, ensemble_size=3
    )
    x = jr.normal(jr.PRNGKey(7), (5, 4))
    with handlers.seed(rng_seed=0):
        y1 = layer(x)
    with handlers.seed(rng_seed=1):
        y2 = layer(x)
    assert jnp.allclose(y1, y2)


def test_dense_rank1_members_are_distinct():
    """Random per-member init (init_scale > 0) gives distinct outputs."""
    layer = DenseRank1.init(
        jr.PRNGKey(0),
        in_features=4,
        out_features=2,
        ensemble_size=3,
        init_scale=0.5,
    )
    x = jr.normal(jr.PRNGKey(7), (5, 4))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    # No two members produce identical outputs.
    assert not jnp.allclose(y[0], y[1])
    assert not jnp.allclose(y[1], y[2])


def test_dense_rank1_zero_init_scale_collapses_members():
    """init_scale=0 ⇒ all r_i = s_i = 1 ⇒ every member is identical."""
    layer = DenseRank1.init(
        jr.PRNGKey(0),
        in_features=4,
        out_features=2,
        ensemble_size=3,
        init_scale=0.0,
    )
    x = jr.normal(jr.PRNGKey(7), (5, 4))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert jnp.allclose(y[0], y[1])
    assert jnp.allclose(y[1], y[2])


def test_dense_rank1_matches_explicit_per_member_kernel():
    """The efficient einsum path equals materialising W_i = (s_i⊗r_i)∘W."""
    layer = DenseRank1.init(
        jr.PRNGKey(0), in_features=3, out_features=2, ensemble_size=4
    )
    x = jr.normal(jr.PRNGKey(11), (6, 3))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    W = layer.W_init
    r = layer.r_init
    s = layer.s_init
    b = jnp.zeros((4, 2))
    expected = jnp.stack(
        [(x * s[i]) @ W * r[i] + b[i] for i in range(4)],
        axis=0,
    )
    assert jnp.allclose(y, expected, atol=1e-5)


# --- Bayesian mode (rank-1 BNN) --------------------------------------------


def test_dense_rank1_bayesian_registers_sample_sites():
    layer = DenseRank1.init(
        jr.PRNGKey(0),
        in_features=4,
        out_features=2,
        ensemble_size=3,
        bayesian=True,
        pyrox_name="rb",
    )
    x = jnp.ones((5, 4))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(x)
    assert tr["rb.W"]["type"] == "param"
    assert tr["rb.r"]["type"] == "sample"
    assert tr["rb.s"]["type"] == "sample"
    assert tr["rb.b"]["type"] == "param"


def test_dense_rank1_bayesian_stochastic_across_seeds():
    layer = DenseRank1.init(
        jr.PRNGKey(0),
        in_features=4,
        out_features=2,
        ensemble_size=3,
        bayesian=True,
    )
    x = jr.normal(jr.PRNGKey(7), (5, 4))
    with handlers.seed(rng_seed=0):
        y1 = layer(x)
    with handlers.seed(rng_seed=1):
        y2 = layer(x)
    assert not jnp.allclose(y1, y2)


def test_dense_rank1_bayesian_priors_centred_on_inits():
    """Each per-member prior is Normal(r_init[i], prior_scale)."""
    layer = DenseRank1.init(
        jr.PRNGKey(0),
        in_features=4,
        out_features=2,
        ensemble_size=3,
        bayesian=True,
        prior_scale=0.25,
        pyrox_name="rb",
    )
    x = jnp.ones((1, 4))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(x)
    r_fn = tr["rb.r"]["fn"]
    assert jnp.allclose(r_fn.base_dist.loc, layer.r_init)
    assert jnp.allclose(r_fn.base_dist.scale, 0.25)


def test_dense_rank1_bayesian_svi_elbo_decreases_with_observation_plate():
    """Canonical SVI pattern: forward outside the data plate, obs inside."""
    rng = jr.PRNGKey(0)
    x = jnp.linspace(-1.0, 1.0, 16)[:, None]
    y = 2.0 * x.squeeze(-1) + 0.5

    def model(x, y=None):
        layer = DenseRank1.init(
            jr.PRNGKey(1),
            in_features=1,
            out_features=1,
            ensemble_size=4,
            bayesian=True,
            pyrox_name="rb",
        )
        # Average over ensemble members to get a single regression output.
        f = layer(x).mean(axis=0).squeeze(-1)
        sigma = numpyro.param(
            "sigma", jnp.array(0.5), constraint=dist.constraints.positive
        )
        with numpyro.plate("data", x.shape[0]):
            numpyro.sample("obs", dist.Normal(f, sigma), obs=y)

    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(1e-2), Trace_ELBO())
    state = svi.init(rng, x, y)
    losses = []
    for _ in range(40):
        state, loss = svi.update(state, x, y)
        losses.append(float(loss))
    assert all(jnp.isfinite(jnp.asarray(losses)))
    assert losses[-1] < losses[0] - 1.0


def test_dense_rank1_is_pyrox_module():
    layer = DenseRank1.init(
        jr.PRNGKey(0), in_features=2, out_features=2, ensemble_size=2
    )
    assert isinstance(layer, PyroxModule)


# --- LayerNormEnsemble ----------------------------------------------------


def test_layer_norm_ensemble_output_shape_3d():
    ln = LayerNormEnsemble(ensemble_size=3, feature_dim=4)
    x = jnp.ones((3, 5, 4))
    with handlers.seed(rng_seed=0):
        y = ln(x)
    assert y.shape == (3, 5, 4)


def test_layer_norm_ensemble_supports_4d_input():
    """Arbitrary intermediate batch / time axes pass through."""
    ln = LayerNormEnsemble(ensemble_size=3, feature_dim=4)
    x = jnp.ones((3, 6, 5, 4))  # (M, batch, time, D)
    with handlers.seed(rng_seed=0):
        y = ln(x)
    assert y.shape == (3, 6, 5, 4)


def test_layer_norm_ensemble_default_init_normalises_to_zero_mean_unit_var():
    """With default scale=1, bias=0 the output has zero mean and unit
    variance per (M, *batch) slice along the feature axis."""
    ln = LayerNormEnsemble(ensemble_size=2, feature_dim=8)
    x = jr.normal(jr.PRNGKey(7), (2, 4, 8)) * 3.0 + 1.5
    with handlers.seed(rng_seed=0):
        y = ln(x)
    means = jnp.mean(y, axis=-1)
    variances = jnp.var(y, axis=-1)
    assert jnp.allclose(means, 0.0, atol=1e-5)
    assert jnp.allclose(variances, 1.0, atol=1e-3)


def test_layer_norm_ensemble_registers_param_sites():
    ln = LayerNormEnsemble(ensemble_size=3, feature_dim=4, pyrox_name="ln")
    x = jnp.ones((3, 2, 4))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        ln(x)
    assert tr["ln.scales"]["type"] == "param"
    assert tr["ln.biases"]["type"] == "param"
    assert tr["ln.scales"]["value"].shape == (3, 4)
    assert tr["ln.biases"]["value"].shape == (3, 4)


def test_layer_norm_ensemble_per_member_scale_and_bias():
    """Substituting different scales/biases per member produces different outputs."""
    ln = LayerNormEnsemble(ensemble_size=2, feature_dim=4, pyrox_name="ln")
    scales = jnp.array([[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0]])
    biases = jnp.array([[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]])
    x = jnp.broadcast_to(jr.normal(jr.PRNGKey(7), (4, 4)), (2, 4, 4))
    with (
        handlers.substitute(data={"ln.scales": scales, "ln.biases": biases}),
        handlers.seed(rng_seed=0),
    ):
        y = ln(x)
    # Member 0: vanilla layer-norm; member 1: 3 * x_hat + 10.
    assert not jnp.allclose(y[0], y[1])
    # Member 1 is the affine transform of member 0's vanilla normalisation.
    assert jnp.allclose(y[1], 3.0 * y[0] + 10.0, atol=1e-5)


def test_layer_norm_ensemble_validates_input_shape():
    ln = LayerNormEnsemble(ensemble_size=3, feature_dim=4)
    with (
        pytest.raises(ValueError, match="ensemble_size"),
        handlers.seed(rng_seed=0),
    ):
        ln(jnp.ones((2, 5, 4)))  # ensemble_size mismatch
    with (
        pytest.raises(ValueError, match="feature_dim"),
        handlers.seed(rng_seed=0),
    ):
        ln(jnp.ones((3, 5, 8)))  # feature_dim mismatch
    with (
        pytest.raises(ValueError, match="at least 2 dims"),
        handlers.seed(rng_seed=0),
    ):
        ln(jnp.ones((4,)))


def test_layer_norm_ensemble_validates_constructor_args():
    with pytest.raises(ValueError, match="ensemble_size"):
        LayerNormEnsemble(ensemble_size=0, feature_dim=4)
    with pytest.raises(ValueError, match="feature_dim"):
        LayerNormEnsemble(ensemble_size=3, feature_dim=0)
    with pytest.raises(ValueError, match="eps"):
        LayerNormEnsemble(ensemble_size=3, feature_dim=4, eps=0.0)


def test_layer_norm_ensemble_composes_with_dense_rank1():
    """LayerNormEnsemble accepts DenseRank1 output without reshape."""
    rank1 = DenseRank1.init(
        jr.PRNGKey(0), in_features=4, out_features=6, ensemble_size=3
    )
    ln = LayerNormEnsemble(ensemble_size=3, feature_dim=6)
    x = jnp.ones((5, 4))
    with handlers.seed(rng_seed=0):
        h = rank1(x)  # shape (3, 5, 6)
        y = ln(h)
    assert y.shape == (3, 5, 6)


def test_layer_norm_ensemble_is_pyrox_module():
    ln = LayerNormEnsemble(ensemble_size=2, feature_dim=4)
    assert isinstance(ln, PyroxModule)


# --- MultiHeadAttentionBE -------------------------------------------------


def test_mha_be_self_attention_output_shape():
    mha = MultiHeadAttentionBE.init(
        jr.PRNGKey(0), embed_dim=8, num_heads=2, ensemble_size=3
    )
    x = jnp.ones((5, 8))
    with handlers.seed(rng_seed=0):
        y = mha(x, x, x)
    assert y.shape == (3, 5, 8)


def test_mha_be_handles_seq_len_equal_to_ensemble_size():
    """Regression: a shape-based ``has_ensemble`` heuristic would
    mis-classify the un-ensembled query / key / value when the
    sequence length happens to equal the ensemble size, breaking
    self-attention for short sequences.
    """
    M = 4
    mha = MultiHeadAttentionBE.init(
        jr.PRNGKey(0), embed_dim=8, num_heads=2, ensemble_size=M
    )
    # T = M, S = M — the failure mode Codex flagged.
    x = jnp.ones((M, 8))
    with handlers.seed(rng_seed=0):
        y = mha(x, x, x)
    assert y.shape == (M, M, 8)
    # Cross-attention with S = M but T ≠ M.
    q = jnp.ones((3, 8))
    kv = jnp.ones((M, 8))
    with handlers.seed(rng_seed=0):
        y_cross = mha(q, kv, kv)
    assert y_cross.shape == (M, 3, 8)


def test_mha_be_cross_attention_output_shape():
    """T (query) ≠ S (key/value) is supported for cross-attention."""
    mha = MultiHeadAttentionBE.init(
        jr.PRNGKey(0), embed_dim=8, num_heads=2, ensemble_size=3
    )
    q = jnp.ones((4, 8))
    kv = jnp.ones((6, 8))
    with handlers.seed(rng_seed=0):
        y = mha(q, kv, kv)
    assert y.shape == (3, 4, 8)


def test_mha_be_init_validates_embed_dim_divisibility():
    with pytest.raises(ValueError, match=r"divisible by num_heads"):
        MultiHeadAttentionBE.init(
            jr.PRNGKey(0), embed_dim=7, num_heads=2, ensemble_size=2
        )


def test_mha_be_init_validates_positive_dims():
    with pytest.raises(ValueError, match=r"must all be > 0"):
        MultiHeadAttentionBE.init(
            jr.PRNGKey(0), embed_dim=0, num_heads=2, ensemble_size=2
        )
    with pytest.raises(ValueError, match=r"must all be > 0"):
        MultiHeadAttentionBE.init(
            jr.PRNGKey(0), embed_dim=8, num_heads=2, ensemble_size=0
        )


def test_mha_be_requires_inits():
    with pytest.raises(ValueError, match=r"\.init\(key"):
        MultiHeadAttentionBE(embed_dim=4, num_heads=2, ensemble_size=2)


def test_mha_be_registers_all_four_projection_param_sites():
    mha = MultiHeadAttentionBE.init(
        jr.PRNGKey(0),
        embed_dim=4,
        num_heads=2,
        ensemble_size=3,
        pyrox_name="mha",
    )
    x = jnp.ones((2, 4))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        mha(x, x, x)
    # With bias=True (default): 4 projections × {W, r, s, b} = 16 sites.
    for proj in ("q", "k", "v", "o"):
        for name in ("W", "r", "s", "b"):
            key = f"mha.{proj}_{name}"
            assert tr[key]["type"] == "param", f"{key} should be a param site"


def test_mha_be_no_bias_skips_bias_param_sites():
    """``bias=False`` should not leak unused ``_b`` params into the store."""
    mha = MultiHeadAttentionBE.init(
        jr.PRNGKey(0),
        embed_dim=4,
        num_heads=2,
        ensemble_size=3,
        bias=False,
        pyrox_name="mha_nobias",
    )
    x = jnp.ones((2, 4))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        y = mha(x, x, x)
    assert y.shape == (3, 2, 4)
    for proj in ("q", "k", "v", "o"):
        for name in ("W", "r", "s"):
            assert f"mha_nobias.{proj}_{name}" in tr
        # No bias site emitted for any projection.
        assert f"mha_nobias.{proj}_b" not in tr


def test_mha_be_post_init_validates_positive_dims():
    """Direct construction must reject non-positive dims with a clear error,
    not raise ``ZeroDivisionError`` from the embed_dim % num_heads check.
    """
    from pyrox.nn._ensemble import _Rank1ProjInit

    good = _Rank1ProjInit(
        W=jnp.zeros((4, 4)),
        r=jnp.ones((2, 4)),
        s=jnp.ones((2, 4)),
        b=jnp.zeros((2, 4)),
    )
    with pytest.raises(ValueError, match=r"must all be > 0"):
        MultiHeadAttentionBE(
            embed_dim=4,
            num_heads=0,  # would crash modulo before this fix
            ensemble_size=2,
            q_init=good,
            k_init=good,
            v_init=good,
            o_init=good,
        )
    with pytest.raises(ValueError, match=r"must all be > 0"):
        MultiHeadAttentionBE(
            embed_dim=0,
            num_heads=2,
            ensemble_size=2,
            q_init=good,
            k_init=good,
            v_init=good,
            o_init=good,
        )


def test_mha_be_post_init_validates_init_array_shapes():
    """Bypassing ``.init`` with mis-sized projection arrays must fail loudly."""
    from pyrox.nn._ensemble import _Rank1ProjInit

    good = _Rank1ProjInit(
        W=jnp.zeros((4, 4)),
        r=jnp.ones((2, 4)),
        s=jnp.ones((2, 4)),
        b=jnp.zeros((2, 4)),
    )
    bad_W = _Rank1ProjInit(
        W=jnp.zeros((3, 4)),  # wrong: should be (4, 4)
        r=jnp.ones((2, 4)),
        s=jnp.ones((2, 4)),
        b=jnp.zeros((2, 4)),
    )
    with pytest.raises(ValueError, match=r"q_init\.W shape"):
        MultiHeadAttentionBE(
            embed_dim=4,
            num_heads=2,
            ensemble_size=2,
            q_init=bad_W,
            k_init=good,
            v_init=good,
            o_init=good,
        )


def test_mha_be_members_are_distinct_with_nonzero_init_scale():
    mha = MultiHeadAttentionBE.init(
        jr.PRNGKey(0),
        embed_dim=8,
        num_heads=2,
        ensemble_size=3,
        init_scale=0.5,
    )
    x = jr.normal(jr.PRNGKey(7), (5, 8))
    with handlers.seed(rng_seed=0):
        y = mha(x, x, x)
    assert not jnp.allclose(y[0], y[1])
    assert not jnp.allclose(y[1], y[2])


def test_mha_be_members_collapse_at_zero_init_scale():
    """init_scale=0 ⇒ all per-member r_i = s_i = 1, all biases = 0,
    so every member is identical to the shared kernel."""
    mha = MultiHeadAttentionBE.init(
        jr.PRNGKey(0),
        embed_dim=8,
        num_heads=2,
        ensemble_size=3,
        init_scale=0.0,
        bias=False,
    )
    x = jr.normal(jr.PRNGKey(7), (5, 8))
    with handlers.seed(rng_seed=0):
        y = mha(x, x, x)
    assert jnp.allclose(y[0], y[1], atol=1e-5)
    assert jnp.allclose(y[1], y[2], atol=1e-5)


def test_mha_be_attention_reduces_to_value_at_uniform_keys():
    """When all keys are zero, softmax gives uniform weights — the
    per-position output is the mean over the key/value sequence."""
    mha = MultiHeadAttentionBE.init(
        jr.PRNGKey(0),
        embed_dim=4,
        num_heads=2,
        ensemble_size=2,
        bias=False,
        init_scale=0.0,  # collapse members so output is deterministic
    )
    # Substitute K and V projection kernels to identity so we get exactly
    # the input keys/values back out, and Q to zero so all attention
    # weights are uniform.
    M = 2
    D = 4
    eye = jnp.eye(D, dtype=jnp.float32)
    ones_M_D = jnp.ones((M, D), dtype=jnp.float32)
    zeros_M_D = jnp.zeros((M, D), dtype=jnp.float32)
    x = jr.normal(jr.PRNGKey(7), (5, D))
    subst = {}
    for proj in ("q", "k", "v", "o"):
        # Zero out Q (so scores=0 → uniform softmax).
        # Identity for K/V/O.
        subst[f"MultiHeadAttentionBE_{id(mha):x}.{proj}_W"] = (
            jnp.zeros((D, D)) if proj == "q" else eye
        )
        subst[f"MultiHeadAttentionBE_{id(mha):x}.{proj}_r"] = ones_M_D
        subst[f"MultiHeadAttentionBE_{id(mha):x}.{proj}_s"] = ones_M_D
        subst[f"MultiHeadAttentionBE_{id(mha):x}.{proj}_b"] = zeros_M_D
    with handlers.substitute(data=subst), handlers.seed(rng_seed=0):
        y = mha(x, x, x)
    expected_per_position = jnp.mean(x, axis=0)
    # Output is (M, T, D); each position equals the mean of x.
    for m in range(M):
        for t in range(x.shape[0]):
            assert jnp.allclose(y[m, t], expected_per_position, atol=1e-5)


def test_mha_be_validates_input_shapes():
    mha = MultiHeadAttentionBE.init(
        jr.PRNGKey(0), embed_dim=8, num_heads=2, ensemble_size=2
    )
    with handlers.seed(rng_seed=0), pytest.raises(ValueError, match=r"query must be"):
        mha(jnp.ones((4, 5)), jnp.ones((4, 8)), jnp.ones((4, 8)))
    with handlers.seed(rng_seed=0), pytest.raises(ValueError, match=r"key must be"):
        mha(jnp.ones((4, 8)), jnp.ones((4, 5)), jnp.ones((4, 5)))
    with (
        handlers.seed(rng_seed=0),
        pytest.raises(ValueError, match=r"key and value must share shape"),
    ):
        mha(jnp.ones((4, 8)), jnp.ones((6, 8)), jnp.ones((5, 8)))


def test_mha_be_is_pyrox_module():
    mha = MultiHeadAttentionBE.init(
        jr.PRNGKey(0), embed_dim=4, num_heads=2, ensemble_size=2
    )
    assert isinstance(mha, PyroxModule)
