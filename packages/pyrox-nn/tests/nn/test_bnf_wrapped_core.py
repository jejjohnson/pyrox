"""`BayesianNeuralField` as a wrapper of `geonnax.NeuralFieldMLP`.

The reference is the inline implementation it replaced, kept verbatim below
as `_InlineBNF` (pyrox-nn before the rewrite): with ``encoder=None`` the
wrapper must register the same sites, with the same priors and shapes, in
the same order, and return the same field for identical site values.
"""

from __future__ import annotations

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from geonnax import SphericalHarmonicEncoder
from geonnax.basis import fourier_features, interaction_features, seasonal_features
from jaxtyping import Array, Float
from numpyro import handlers
from numpyro.infer import Predictive
from pyrox._core.pyrox_module import PyroxModule, pyrox_method
from pyrox_nn import BayesianNeuralField


class _InlineBNF(PyroxModule):
    """The pre-geonnax `BayesianNeuralField`, verbatim (docstring dropped)."""

    input_scales: tuple[float, ...] = eqx.field(static=True)
    fourier_degrees: tuple[int, ...] = eqx.field(static=True)
    interactions: tuple[tuple[int, int], ...] = eqx.field(static=True)
    seasonality_periods: tuple[float, ...] = eqx.field(static=True)
    num_seasonal_harmonics: tuple[int, ...] = eqx.field(static=True)
    width: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    time_col: int = eqx.field(static=True, default=0)
    pyrox_name: str | None = None

    @staticmethod
    def _logistic_prior(shape: tuple[int, ...]) -> dist.Distribution:
        if not shape:
            return dist.Logistic(0.0, 1.0)
        return dist.Logistic(jnp.zeros(shape), 1.0).to_event(len(shape))

    @pyrox_method
    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, " N"]:
        d_in = len(self.input_scales)
        input_scales = jnp.asarray(self.input_scales, dtype=jnp.float32)
        log_scale_adjustment = self.pyrox_sample(
            "log_scale_adjustment",
            self._logistic_prior((d_in,)),
        )
        scaled_x = x / (input_scales * jnp.exp(log_scale_adjustment))
        feature_blocks: list[Float[Array, "N F_block"]] = [scaled_x]
        for col_idx, d in enumerate(self.fourier_degrees):
            if d > 0:
                feature_blocks.append(
                    fourier_features(scaled_x[:, col_idx], d, rescale=True)
                )
        if self.seasonality_periods and any(self.num_seasonal_harmonics):
            feature_blocks.append(
                seasonal_features(
                    x[:, self.time_col],
                    self.seasonality_periods,
                    self.num_seasonal_harmonics,
                    rescale=True,
                )
            )
        if self.interactions:
            feature_blocks.append(
                interaction_features(
                    scaled_x, jnp.asarray(self.interactions, dtype=jnp.int32)
                )
            )
        gated_blocks: list[Float[Array, "N F_block"]] = []
        for b_idx, block in enumerate(feature_blocks):
            if block.shape[-1] == 0:
                continue
            gain = self.pyrox_sample(
                f"feature_gain_{b_idx}",
                self._logistic_prior(()),
            )
            gated_blocks.append(block * jax.nn.softplus(gain))
        h = jnp.concatenate(gated_blocks, axis=-1)
        logit_activation_weight = self.pyrox_sample(
            "logit_activation_weight",
            self._logistic_prior(()),
        )
        alpha = jax.nn.sigmoid(logit_activation_weight)

        def activation(z: Float[Array, ...]) -> Float[Array, ...]:
            return alpha * jax.nn.elu(z) + (1.0 - alpha) * jnp.tanh(z)

        for layer_idx in range(self.depth):
            fan_in = h.shape[-1]
            W = self.pyrox_sample(
                f"layer_{layer_idx}_W",
                self._logistic_prior((fan_in, self.width)),
            )
            b = self.pyrox_sample(
                f"layer_{layer_idx}_b",
                self._logistic_prior((self.width,)),
            )
            layer_gain = self.pyrox_sample(
                f"layer_{layer_idx}_gain",
                self._logistic_prior(()),
            )
            h = h / jnp.sqrt(fan_in)
            h = activation(
                jax.nn.softplus(layer_gain)
                * (einx.dot("... i, i o -> ... o", h, W) + b)
            )
        fan_in = h.shape[-1]
        W_out = self.pyrox_sample(
            "output_W",
            self._logistic_prior((fan_in, 1)),
        )
        b_out = self.pyrox_sample(
            "output_b",
            self._logistic_prior((1,)),
        )
        output_gain = self.pyrox_sample(
            "output_gain",
            self._logistic_prior(()),
        )
        h = h / jnp.sqrt(fan_in)
        out = einx.dot("... i, i o -> ... o", h, W_out) + b_out
        return (jax.nn.softplus(output_gain) * out).squeeze(-1)


# Input scales exactly representable in float32: the inline implementation
# cast them to float32, the wrapper keeps the input's dtype (float64 here,
# the root conftest enables x64), so only such scales give bitwise-equal
# rescaled inputs.
_CONFIGS = {
    "full": dict(
        input_scales=(2.0, 0.5, 4.0),
        fourier_degrees=(2, 0, 3),
        interactions=((0, 1), (1, 2)),
        seasonality_periods=(7.0, 1.5),
        num_seasonal_harmonics=(3, 1),
        width=8,
        depth=2,
        time_col=0,
    ),
    "plain": dict(
        input_scales=(1.0, 1.0),
        fourier_degrees=(),
        interactions=(),
        seasonality_periods=(),
        num_seasonal_harmonics=(),
        width=4,
        depth=1,
        time_col=1,
    ),
    "no_hidden": dict(
        input_scales=(1.0, 2.0, 0.25),
        fourier_degrees=(1, 1, 1),
        interactions=(),
        seasonality_periods=(3.0,),
        num_seasonal_harmonics=(2,),
        width=4,
        depth=0,
        time_col=2,
    ),
}


def _trace(model, x, seed):
    with handlers.trace() as tr, handlers.seed(rng_seed=seed):
        out = model(x)
    return out, tr


def _site_values(tr) -> dict[str, Array]:
    return {
        name: site["value"]
        for name, site in tr.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }


@pytest.mark.parametrize("name", list(_CONFIGS))
def test_sites_match_inline_implementation(name):
    """Same site names, order, priors, shapes and seeded draws as before."""
    cfg = _CONFIGS[name]
    x = jr.normal(jr.PRNGKey(1), (6, len(cfg["input_scales"])))
    _, tr_old = _trace(_InlineBNF(**cfg, pyrox_name="bnf"), x, seed=0)
    _, tr_new = _trace(BayesianNeuralField(**cfg, pyrox_name="bnf"), x, seed=0)
    assert list(tr_new) == list(tr_old)
    for site_name, old in tr_old.items():
        new = tr_new[site_name]
        assert new["fn"].batch_shape == old["fn"].batch_shape
        assert new["fn"].event_shape == old["fn"].event_shape
        base_old = getattr(old["fn"], "base_dist", old["fn"])
        base_new = getattr(new["fn"], "base_dist", new["fn"])
        assert type(base_new) is type(base_old) is dist.Logistic
        assert jnp.all(base_new.loc == 0.0) and jnp.all(base_new.scale == 1.0)
        # Same registration order under one seed ⇒ the same PRNG stream.
        assert jnp.array_equal(new["value"], old["value"])


@pytest.mark.parametrize("name", list(_CONFIGS))
def test_forward_equals_inline_implementation_on_five_draws(name):
    """Old == new for identical site values, on 5 prior draws (substitute)."""
    cfg = _CONFIGS[name]
    old = _InlineBNF(**cfg, pyrox_name="bnf")
    new = BayesianNeuralField(**cfg, pyrox_name="bnf")
    x = 3.0 * jr.normal(jr.PRNGKey(2), (11, len(cfg["input_scales"])))
    for seed in range(5):
        # The stored trace: the inline implementation's own draws.
        _, tr = _trace(old, x, seed=seed)
        params = _site_values(tr)
        f_old = handlers.substitute(old, data=params)(x)
        f_new = handlers.substitute(new, data=params)(x)
        assert f_new.shape == f_old.shape == (11,)
        # Float64 round-off: the two differ only in summation order (a
        # batched matmul against per-row dot products), ~1e-16 relative
        # per term over fan-ins below 40.
        assert jnp.allclose(f_new, f_old, rtol=1e-12, atol=1e-12)


def test_arbitrary_batch_shape():
    cfg = _CONFIGS["full"]
    bnf = BayesianNeuralField(**cfg)
    x = jr.normal(jr.PRNGKey(3), (2, 5, 3))
    with handlers.seed(rng_seed=0):
        f_batched = bnf(x)
    with handlers.seed(rng_seed=0):
        f_flat = bnf(x.reshape(10, 3))
    assert f_batched.shape == (2, 5)
    assert jnp.array_equal(f_batched.reshape(10), f_flat)


def test_core_is_a_neural_field_mlp():
    from geonnax import NeuralFieldMLP

    bnf = BayesianNeuralField(**_CONFIGS["full"])
    assert isinstance(bnf.core, NeuralFieldMLP)
    assert bnf.core.extra_features == 0


# -- encoder ------------------------------------------------------------------


def _encoded_reference(params, x, cfg, encoder, cols):
    """The inline forward, encoder columns moved to a block appended last."""
    keep = [c for c in range(x.shape[-1]) if c not in cols]
    scales = jnp.asarray(cfg["input_scales"])[jnp.asarray(keep)]
    scaled = x[:, jnp.asarray(keep)] / (
        scales * jnp.exp(params["log_scale_adjustment"])
    )
    blocks = [scaled]
    if cfg["seasonality_periods"]:
        blocks.append(
            seasonal_features(
                x[:, cfg["time_col"]],
                cfg["seasonality_periods"],
                cfg["num_seasonal_harmonics"],
                rescale=True,
            )
        )
    blocks.append(jax.vmap(encoder)(x[:, jnp.asarray(cols)]))
    h = jnp.concatenate(
        [
            jax.nn.softplus(params[f"feature_gain_{b}"]) * blk
            for b, blk in enumerate(blocks)
        ],
        axis=-1,
    )
    alpha = jax.nn.sigmoid(params["logit_activation_weight"])
    for layer in range(cfg["depth"]):
        z = jax.nn.softplus(params[f"layer_{layer}_gain"]) * (
            h / jnp.sqrt(h.shape[-1]) @ params[f"layer_{layer}_W"]
            + params[f"layer_{layer}_b"]
        )
        h = alpha * jax.nn.elu(z) + (1 - alpha) * jnp.tanh(z)
    out = h / jnp.sqrt(h.shape[-1]) @ params["output_W"] + params["output_b"]
    return jax.nn.softplus(params["output_gain"]) * out[:, 0]


_ENC_CFG = dict(
    input_scales=(1.0, 2.0, 1.0),
    fourier_degrees=(),
    interactions=(),
    seasonality_periods=(365.25,),
    num_seasonal_harmonics=(2,),
    width=8,
    depth=2,
    time_col=0,
)


def _lonlat_inputs(key, n):
    k1, k2, k3 = jr.split(key, 3)
    t = jr.uniform(k1, (n,), maxval=365.0)
    lon = jr.uniform(k2, (n,), minval=-jnp.pi, maxval=jnp.pi)
    lat = jr.uniform(k3, (n,), minval=-jnp.pi / 2, maxval=jnp.pi / 2)
    return jnp.stack([t, lon, lat], axis=-1)


def test_encoder_adds_one_gated_block_last():
    enc = SphericalHarmonicEncoder(l_max=3, input_mode="lonlat")
    bnf = BayesianNeuralField(
        **_ENC_CFG, encoder=enc, encoder_cols=(1, 2), pyrox_name="bnf"
    )
    plain = BayesianNeuralField(**_ENC_CFG, pyrox_name="bnf")
    x = _lonlat_inputs(jr.PRNGKey(4), 7)
    f, tr = _trace(bnf, x, seed=0)
    _, tr_plain = _trace(plain, x, seed=0)
    assert f.shape == (7,)
    assert bnf.core.extra_features == enc.num_features == 16
    gains = sorted(n for n in tr if n.startswith("bnf.feature_gain_"))
    plain_gains = sorted(n for n in tr_plain if n.startswith("bnf.feature_gain_"))
    # Raw inputs + seasonal, then the encoder block.
    assert plain_gains == ["bnf.feature_gain_0", "bnf.feature_gain_1"]
    assert gains == [*plain_gains, "bnf.feature_gain_2"]
    # The encoded columns leave the raw block (3 -> 1 column) and enter
    # through the encoder block only.
    assert tr["bnf.log_scale_adjustment"]["value"].shape == (1,)
    assert tr["bnf.layer_0_W"]["value"].shape == (1 + 4 + 16, 8)
    assert tr_plain["bnf.layer_0_W"]["value"].shape == (3 + 4, 8)


def test_encoder_forward_matches_batched_reference():
    """The encoder sees x[encoder_cols] raw, its block gated by the last gain."""
    enc = SphericalHarmonicEncoder(l_max=4, input_mode="lonlat")
    cols = (1, 2)
    bnf = BayesianNeuralField(
        **_ENC_CFG, encoder=enc, encoder_cols=cols, pyrox_name="bnf"
    )
    x = _lonlat_inputs(jr.PRNGKey(5), 9)
    for seed in range(5):
        f, tr = _trace(bnf, x, seed=seed)
        params = {k.removeprefix("bnf."): v for k, v in _site_values(tr).items()}
        ref = _encoded_reference(params, x, _ENC_CFG, enc, cols)
        # Float64 round-off, as above.
        assert jnp.allclose(f, ref, rtol=1e-12, atol=1e-12)


def test_spherical_harmonic_encoder_field_has_no_dateline_seam():
    """Lon/lat enter only through the SH encoder: f(t, π, φ) = f(t, −π, φ)."""
    enc = SphericalHarmonicEncoder(l_max=6, input_mode="lonlat")
    bnf = BayesianNeuralField(**_ENC_CFG, encoder=enc, encoder_cols=(1, 2))
    lat = jnp.linspace(-1.5, 1.5, 25)
    t = jnp.full_like(lat, 40.0)
    east = jnp.stack([t, jnp.full_like(lat, jnp.pi), lat], axis=-1)
    west = jnp.stack([t, jnp.full_like(lat, -jnp.pi), lat], axis=-1)
    for seed in range(5):
        with handlers.seed(rng_seed=seed):
            f = bnf(jnp.concatenate([east, west]))
        # Float64 round-off: sin(±π) = ±1.2e-16 differ only in sign.
        assert jnp.allclose(f[:25], f[25:], rtol=0.0, atol=1e-10)


def test_encoder_column_order_matters():
    """encoder_cols=(2, 1) feeds (lat, lon): a different field from (1, 2)."""
    enc = SphericalHarmonicEncoder(l_max=2, input_mode="lonlat")
    x = _lonlat_inputs(jr.PRNGKey(6), 5)
    with handlers.seed(rng_seed=0):
        f12 = BayesianNeuralField(**_ENC_CFG, encoder=enc, encoder_cols=(1, 2))(x)
    with handlers.seed(rng_seed=0):
        f21 = BayesianNeuralField(**_ENC_CFG, encoder=enc, encoder_cols=(2, 1))(x)
    assert not jnp.allclose(f12, f21)


def test_encoder_jit_grad_and_predictive():
    enc = SphericalHarmonicEncoder(l_max=2, input_mode="lonlat")
    bnf = BayesianNeuralField(**_ENC_CFG, encoder=enc, encoder_cols=(1, 2))
    x = _lonlat_inputs(jr.PRNGKey(7), 4)

    seeded = jax.jit(handlers.seed(bnf, rng_seed=0))
    assert seeded(x).shape == (4,)
    g = jax.grad(lambda z: handlers.seed(bnf, rng_seed=0)(z).sum())(x)
    assert g.shape == x.shape and jnp.all(jnp.isfinite(g))

    samples = Predictive(bnf, num_samples=3)(jr.PRNGKey(0), x)
    assert samples["BayesianNeuralField.feature_gain_2"].shape == (3,)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(encoder=SphericalHarmonicEncoder(l_max=1)), "given together"),
        (dict(encoder_cols=(1, 2)), "given together"),
        (
            dict(
                encoder=SphericalHarmonicEncoder(l_max=1, input_mode="lonlat"),
                encoder_cols=(1, 3),
            ),
            "encoder_cols",
        ),
        (
            dict(
                encoder=SphericalHarmonicEncoder(l_max=1, input_mode="lonlat"),
                encoder_cols=(),
            ),
            "encoder_cols",
        ),
        (
            dict(encoder=lambda z: jnp.outer(z, z), encoder_cols=(1, 2)),
            "vector",
        ),
        (
            dict(
                encoder=SphericalHarmonicEncoder(l_max=1, input_mode="lonlat"),
                encoder_cols=(0, 1),
            ),
            r"clashes on columns \[0\]",  # the seasonal time column
        ),
        (
            dict(
                encoder=SphericalHarmonicEncoder(l_max=1, input_mode="lonlat"),
                encoder_cols=(1, 2),
                fourier_degrees=(0, 2, 0),
            ),
            r"clashes on columns \[1\]",
        ),
        (
            dict(
                encoder=SphericalHarmonicEncoder(l_max=1, input_mode="lonlat"),
                encoder_cols=(1, 2),
                interactions=((0, 2),),
            ),
            r"clashes on columns \[2\]",
        ),
        (
            dict(
                encoder=lambda z: z,
                encoder_cols=(0, 1, 2),
                seasonality_periods=(),
                num_seasonal_harmonics=(),
            ),
            "at least one input column",
        ),
    ],
)
def test_encoder_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BayesianNeuralField(**{**_ENC_CFG, **kwargs})


def test_invalid_config_raises():
    """The core validates the configuration at construction now."""
    with pytest.raises(ValueError, match="fourier_degrees"):
        BayesianNeuralField(**{**_CONFIGS["full"], "fourier_degrees": (1, 2)})
