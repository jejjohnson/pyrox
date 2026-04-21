"""Tests for the BNF layer family in `pyrox.nn._bnf`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
from numpyro import handlers

from pyrox.nn import (
    BayesianNeuralField,
    FourierFeatures,
    InteractionFeatures,
    SeasonalFeatures,
    Standardization,
)


# -- Standardization ---------------------------------------------------------


def test_standardization_zero_mean_unit_std_identity():
    layer = Standardization(mu=jnp.zeros(3), std=jnp.ones(3))
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = layer(x)
    assert jnp.allclose(out, x)


def test_standardization_subtracts_mu_divides_std():
    layer = Standardization(mu=jnp.array([2.0, 4.0]), std=jnp.array([1.0, 2.0]))
    x = jnp.array([[2.0, 4.0], [3.0, 6.0]])
    out = layer(x)
    assert jnp.allclose(out, jnp.array([[0.0, 0.0], [1.0, 1.0]]))


# -- FourierFeatures ---------------------------------------------------------


def test_fourier_features_shape_with_zero_skips():
    """A column with degree=0 contributes no features."""
    layer = FourierFeatures(degrees=(2, 0, 3))
    x = jnp.ones((5, 3))
    with handlers.seed(rng_seed=0):
        out = layer(x)
    # 2 * (2 + 0 + 3) = 10 columns
    assert out.shape == (5, 10)


def test_fourier_features_all_zero_degrees_returns_empty():
    layer = FourierFeatures(degrees=(0, 0))
    out = layer(jnp.ones((3, 2)))
    assert out.shape == (3, 0)


# -- SeasonalFeatures --------------------------------------------------------


def test_seasonal_features_shape():
    layer = SeasonalFeatures(periods=(7.0, 365.25), harmonics=(2, 3))
    out = layer(jnp.arange(10.0))
    assert out.shape == (10, 10)  # 2 * (2 + 3)


# -- InteractionFeatures ----------------------------------------------------


def test_interaction_features_pair_products():
    layer = InteractionFeatures(pairs=((0, 1), (1, 2)))
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = layer(x)
    assert jnp.allclose(out, jnp.array([[2.0, 6.0], [20.0, 30.0]]))


def test_interaction_features_empty_pairs_yields_empty_output():
    layer = InteractionFeatures(pairs=())
    out = layer(jnp.ones((4, 3)))
    assert out.shape == (4, 0)


# -- BayesianNeuralField ----------------------------------------------------


def _make_bnf(width: int = 8, depth: int = 2) -> BayesianNeuralField:
    return BayesianNeuralField(
        input_scales=(1.0, 1.0, 1.0),
        fourier_degrees=(2, 0, 2),
        interactions=((0, 1),),
        seasonality_periods=(7.0,),
        num_seasonal_harmonics=(3,),
        width=width,
        depth=depth,
        time_col=0,
        pyrox_name="bnf",
    )


def test_bnf_output_shape():
    bnf = _make_bnf()
    x = jnp.ones((5, 3))
    with handlers.seed(rng_seed=0):
        out = bnf(x)
    assert out.shape == (5,)


def test_bnf_registers_all_logistic_priors():
    """Every learnable leaf must be a sample site with a Logistic prior."""
    import numpyro.distributions as dist

    bnf = _make_bnf()
    x = jnp.ones((3, 3))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        bnf(x)
    sample_sites = [
        (name, site)
        for name, site in tr.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    ]
    assert len(sample_sites) > 0
    expected_prefixes = {
        "bnf.log_scale_adjustment",
        "bnf.feature_gain_",
        "bnf.layer_",
        "bnf.output_W",
        "bnf.output_b",
        "bnf.output_gain",
        "bnf.logit_activation_weight",
    }
    for name, site in sample_sites:
        # Find the matching prefix.
        assert any(name.startswith(p) for p in expected_prefixes), name
        # Underlying base distribution must be Logistic (allow .to_event wrap).
        base_fn = site["fn"]
        while hasattr(base_fn, "base_dist"):
            base_fn = base_fn.base_dist
        assert isinstance(base_fn, dist.Logistic), (name, type(base_fn).__name__)


def test_bnf_deterministic_under_substitute():
    """Substitute fixed values for every site -> output is reproducible."""
    from numpyro.handlers import substitute, trace

    bnf = _make_bnf()
    x = jnp.ones((4, 3))
    # Run once with seed to discover sites + sample values.
    with handlers.seed(rng_seed=0):
        out_seeded = bnf(x)
    tr = trace(handlers.seed(bnf, jr.PRNGKey(0))).get_trace(x)
    params = {n: s["value"] for n, s in tr.items() if s["type"] == "sample"}
    # Re-run with substitute => must reproduce out_seeded exactly.
    out_subbed = substitute(bnf, params)(x)
    assert jnp.allclose(out_seeded, out_subbed)


def test_bnf_jit_compatible():
    """The BNF call should be jittable inside a numpyro substitute handler."""
    from numpyro.handlers import substitute, trace

    bnf = _make_bnf()
    x = jnp.ones((4, 3))
    tr = trace(handlers.seed(bnf, jr.PRNGKey(0))).get_trace(x)
    params = {n: s["value"] for n, s in tr.items() if s["type"] == "sample"}

    @jax.jit
    def predict(p, x):
        return substitute(bnf, p)(x)

    out = predict(params, x)
    assert out.shape == (4,)
