"""Tests for ``pyrox.nn`` geographic and spherical encoders."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyrox._basis import real_spherical_harmonics
from pyrox.gp import RBF, SphericalHarmonicInducingFeatures
from pyrox.nn import (
    Cartesian3DEncoder,
    CyclicEncoder,
    Deg2Rad,
    LonLatScale,
    SphericalHarmonicEncoder,
    cyclic_encode,
    deg2rad,
    lonlat_scale,
)


def _random_lonlat(n: int) -> jnp.ndarray:
    rng = np.random.default_rng(0)
    lon = rng.uniform(-jnp.pi, jnp.pi, size=n)
    lat = rng.uniform(-0.5 * jnp.pi, 0.5 * jnp.pi, size=n)
    return jnp.asarray(np.stack([lon, lat], axis=-1), dtype=jnp.float32)


def _random_unit_xyz(n: int) -> jnp.ndarray:
    rng = np.random.default_rng(1)
    xyz = rng.standard_normal((n, 3))
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    return jnp.asarray(xyz, dtype=jnp.float32)


def _array_leaves(tree: object) -> list[jax.Array]:
    return [
        leaf
        for leaf in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))
        if leaf is not None
    ]


def test_deg2rad_matches_numpy():
    x1 = jnp.array([0.0, 90.0, 180.0], dtype=jnp.float32)
    x2 = jnp.array([[45.0, 180.0], [-90.0, 270.0]], dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(deg2rad(x1)), np.deg2rad(np.asarray(x1)))
    np.testing.assert_allclose(np.asarray(deg2rad(x2)), np.deg2rad(np.asarray(x2)))


def test_lonlat_scale_maps_to_unit_interval():
    lonlat = jnp.array(
        [
            [-180.0, -90.0],
            [0.0, 0.0],
            [180.0, 90.0],
        ],
        dtype=jnp.float32,
    )
    expected = jnp.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]], dtype=jnp.float32)
    np.testing.assert_allclose(lonlat_scale(lonlat), expected)


def test_cartesian3d_unit_norm():
    lonlat = _random_lonlat(512)
    xyz = Cartesian3DEncoder()(lonlat)
    norms = jnp.linalg.norm(xyz, axis=-1)
    np.testing.assert_allclose(np.asarray(norms), np.ones(512), atol=1e-6)


def test_cartesian3d_matches_hand_computed_axes():
    lonlat = jnp.array(
        [
            [0.0, 0.0],
            [0.5 * jnp.pi, 0.0],
            [0.0, 0.5 * jnp.pi],
        ],
        dtype=jnp.float32,
    )
    expected = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    np.testing.assert_allclose(Cartesian3DEncoder()(lonlat), expected, atol=1e-6)


def test_cartesian3d_convention_matches_sh_inducing_features():
    lonlat = jnp.array(
        [
            [0.0, 0.0],
            [0.5 * jnp.pi, 0.0],
            [0.0, 0.5 * jnp.pi],
        ],
        dtype=jnp.float32,
    )
    expected_xyz = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    kernel = RBF(init_lengthscale=1.0, init_variance=1.0)
    features = SphericalHarmonicInducingFeatures.init(l_max=3)
    encoded_xyz = Cartesian3DEncoder()(lonlat)
    np.testing.assert_allclose(encoded_xyz, expected_xyz, atol=1e-6)
    np.testing.assert_allclose(
        features.k_ux(encoded_xyz, kernel),
        features.k_ux(expected_xyz, kernel),
        atol=1e-6,
    )


def test_cyclic_encode_shape_and_range():
    x1 = jnp.linspace(-jnp.pi, jnp.pi, 7, dtype=jnp.float32)
    x2 = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    y1 = cyclic_encode(x1)
    y2 = cyclic_encode(x2)
    assert y1.shape == (7, 2)
    assert y2.shape == (4, 6)
    assert jnp.all(y1 <= 1.0) and jnp.all(y1 >= -1.0)
    np.testing.assert_allclose(y2[:, :3], jnp.cos(x2), atol=1e-6)
    np.testing.assert_allclose(y2[:, 3:], jnp.sin(x2), atol=1e-6)


def test_sh_encoder_matches_basis_function():
    xyz = _random_unit_xyz(64)
    layer = SphericalHarmonicEncoder(l_max=5, input_mode="cartesian")
    expected = real_spherical_harmonics(xyz, l_max=5)
    np.testing.assert_allclose(layer(xyz), expected, atol=1e-6)


@pytest.mark.parametrize("l_max", [0, 1, 3, 8])
def test_sh_encoder_num_features(l_max: int):
    assert SphericalHarmonicEncoder(l_max=l_max).num_features == (l_max + 1) ** 2


def test_sh_encoder_lonlat_input_mode():
    lonlat = _random_lonlat(32)
    direct = SphericalHarmonicEncoder(l_max=4, input_mode="lonlat")(lonlat)
    via_cartesian = SphericalHarmonicEncoder(l_max=4, input_mode="cartesian")(
        Cartesian3DEncoder()(lonlat)
    )
    np.testing.assert_allclose(direct, via_cartesian, atol=1e-6)


@pytest.mark.parametrize(
    ("layer", "x"),
    [
        (Deg2Rad(), jnp.array([[0.0, 90.0]], dtype=jnp.float32)),
        (LonLatScale(), jnp.array([[0.0, 0.0]], dtype=jnp.float32)),
        (
            Cartesian3DEncoder(input_unit="degrees"),
            jnp.array([[0.0, 0.0]], dtype=jnp.float32),
        ),
        (CyclicEncoder(), jnp.array([[0.0, jnp.pi]], dtype=jnp.float32)),
        (
            SphericalHarmonicEncoder(l_max=3, input_mode="lonlat"),
            jnp.array([[0.0, 0.0]], dtype=jnp.float32),
        ),
    ],
)
def test_encoders_jit(layer: object, x: jnp.ndarray):
    y = jax.jit(layer)(x)
    assert jnp.all(jnp.isfinite(y))


@pytest.mark.parametrize(
    "layer",
    [
        Deg2Rad(),
        LonLatScale(),
        Cartesian3DEncoder(),
        CyclicEncoder(),
        SphericalHarmonicEncoder(l_max=3),
    ],
)
def test_encoders_no_trainable_params(layer: object):
    assert _array_leaves(layer) == []


def test_encoders_composition_end_to_end():
    lonlat_deg = jnp.array(
        [
            [-180.0, -45.0],
            [0.0, 0.0],
            [120.0, 30.0],
        ],
        dtype=jnp.float32,
    )
    radians = Deg2Rad()(lonlat_deg)
    scaled = LonLatScale()(lonlat_deg)
    xyz = Cartesian3DEncoder(input_unit="radians")(radians)
    sh = SphericalHarmonicEncoder(l_max=3, input_mode="cartesian")(xyz)
    assert scaled.shape == (3, 2)
    assert sh.shape == (3, 16)
    assert jnp.all(jnp.isfinite(scaled))
    assert jnp.all(jnp.isfinite(sh))


def test_encoders_reject_bad_shapes():
    with pytest.raises(ValueError, match="lonlat"):
        LonLatScale()(jnp.zeros((3, 3), dtype=jnp.float32))
    with pytest.raises(ValueError, match="lonlat"):
        Cartesian3DEncoder()(jnp.zeros((3, 3), dtype=jnp.float32))
    with pytest.raises(ValueError, match="input_mode"):
        SphericalHarmonicEncoder(l_max=2, input_mode="lonlat")(
            jnp.zeros((3, 3), dtype=jnp.float32)
        )


def test_lonlat_scale_does_not_clip_out_of_range():
    """Out-of-range lon/lat values map linearly outside ``[-1, 1]``;
    ``lonlat_scale`` is a pure affine transform by design.
    """
    lonlat = jnp.array([[-270.0, -135.0], [360.0, 135.0]], dtype=jnp.float32)
    scaled = lonlat_scale(lonlat)
    # 2 * (lonlat - lower) / (upper - lower) - 1, no clipping.
    np.testing.assert_allclose(np.asarray(scaled[:, 0]), np.array([-1.5, 2.0]))
    np.testing.assert_allclose(np.asarray(scaled[:, 1]), np.array([-1.5, 1.5]))


def test_lonlat_scale_promotes_integer_input_to_float():
    """Integer inputs are promoted to ``float32`` so the affine op does
    not get truncated to ``-1 / 0 / 1`` by integer division.
    """
    lonlat_int = jnp.array([[-90, -45], [45, 30]], dtype=jnp.int32)
    scaled = lonlat_scale(lonlat_int)
    assert scaled.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(scaled), np.array([[-0.5, -0.5], [0.25, 1.0 / 3.0]]), atol=1e-6
    )
