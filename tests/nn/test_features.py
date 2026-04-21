"""Tests for `pyrox.nn._features` pure-JAX feature helpers."""

from __future__ import annotations

import jax.numpy as jnp

from pyrox.nn._features import (
    fourier_features,
    interaction_features,
    seasonal_features,
    seasonal_frequencies,
    standardize,
    unstandardize,
)


# -- fourier_features --------------------------------------------------------


def test_fourier_features_at_zero():
    """At x=0, cos(0)=1 and sin(0)=0 so the row should be [1,...,1,0,...,0]."""
    out = fourier_features(jnp.zeros(1), max_degree=3)
    assert out.shape == (1, 6)
    assert jnp.allclose(out[0], jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))


def test_fourier_features_shape():
    out = fourier_features(jnp.arange(7.0), max_degree=4)
    assert out.shape == (7, 8)


def test_fourier_features_rescale_divides_by_degree_plus_one():
    out = fourier_features(jnp.array([1.0]), max_degree=3, rescale=True)
    raw = fourier_features(jnp.array([1.0]), max_degree=3, rescale=False)
    expected = raw / jnp.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    assert jnp.allclose(out, expected)


# -- seasonal_frequencies / seasonal_features --------------------------------


def test_seasonal_frequencies_layout():
    period_idx, freqs = seasonal_frequencies((7.0, 365.25), (2, 3))
    assert len(period_idx) == 5
    assert len(freqs) == 5
    # First two come from period 7 (h=1, h=2), next three from 365.25.
    assert period_idx == [0, 0, 1, 1, 1]
    assert jnp.allclose(
        jnp.array(freqs),
        jnp.array([1 / 7, 2 / 7, 1 / 365.25, 2 / 365.25, 3 / 365.25]),
    )


def test_seasonal_features_shape():
    out = seasonal_features(jnp.arange(10.0), (7.0,), (3,))
    assert out.shape == (10, 6)  # 2 * (3 harmonics) = 6 cols


def test_seasonal_features_empty_when_no_harmonics():
    out = seasonal_features(jnp.arange(5.0), (), ())
    assert out.shape == (5, 0)


# -- interaction_features ----------------------------------------------------


def test_interaction_features_known_value():
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pairs = jnp.array([[0, 1], [1, 2]])
    out = interaction_features(x, pairs)
    assert out.shape == (2, 2)
    assert jnp.allclose(out, jnp.array([[2.0, 6.0], [20.0, 30.0]]))


def test_interaction_features_empty_pairs():
    out = interaction_features(jnp.ones((4, 3)), jnp.zeros((0, 2), dtype=jnp.int32))
    assert out.shape == (4, 0)


# -- standardize / unstandardize --------------------------------------------


def test_standardize_unstandardize_roundtrip():
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    mu, std = jnp.float32(2.5), jnp.float32(1.0)
    z = standardize(x, mu, std)
    back = unstandardize(z, mu, std)
    assert jnp.allclose(back, x)


def test_standardize_zero_mean_unit_std():
    x = jnp.array([10.0, 20.0, 30.0])
    z = standardize(x, jnp.float32(20.0), jnp.float32(10.0))
    assert jnp.allclose(z, jnp.array([-1.0, 0.0, 1.0]))
