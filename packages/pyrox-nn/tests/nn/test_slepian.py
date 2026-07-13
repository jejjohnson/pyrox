"""Tests for Slepian positional encoders."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.handlers as handlers
from pyrox_gp._basis import slepian_cap_basis
from pyrox_nn import (
    BayesianSlepianEncoder,
    HybridSphericalSlepianEncoder,
    SlepianEncoder,
)


def test_slepian_encoder_matches_basis_direct_call():
    basis = slepian_cap_basis(4, jnp.deg2rad(45.0), n_modes=5)
    encoder = SlepianEncoder(
        basis=basis, input_mode="cartesian", weight_by_eigenvalue=False
    )
    xyz = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(
        np.asarray(jax.vmap(encoder)(xyz)),
        np.asarray(basis.evaluate(xyz)),
        atol=1e-6,
    )


def test_slepian_encoder_lonlat_shape_and_eigenvalue_weighting():
    basis = slepian_cap_basis(3, jnp.deg2rad(60.0), n_modes=4)
    weighted = SlepianEncoder(basis=basis, input_mode="lonlat")
    unweighted = SlepianEncoder(
        basis=basis, input_mode="lonlat", weight_by_eigenvalue=False
    )
    lonlat = jnp.array([[0.0, 0.0], [0.5 * jnp.pi, 0.0]])

    assert jax.vmap(weighted)(lonlat).shape == (2, 4)
    np.testing.assert_allclose(
        np.asarray(jax.vmap(weighted)(lonlat)),
        np.asarray(jax.vmap(unweighted)(lonlat) * jnp.sqrt(basis.eigenvalues)[None, :]),
        atol=1e-6,
    )


def test_hybrid_spherical_slepian_encoder_concatenates_features():
    encoder = HybridSphericalSlepianEncoder.from_cap(
        sh_l_max=2,
        slepian_l_max=3,
        cap_radius_deg=50.0,
        cap_centre_lonlat_deg=(0.0, 0.0),
        n_modes=4,
        input_mode="lonlat",
    )
    lonlat = jnp.array([[0.0, 0.0], [1.0, 0.2]])

    assert jax.vmap(encoder)(lonlat).shape == (2, 13)
    assert encoder.num_features == 13


def test_bayesian_slepian_encoder_registers_cap_sites():
    encoder = BayesianSlepianEncoder(
        l_max=2,
        init_cap_radius_deg=45.0,
        init_cap_centre_lonlat_deg=(0.0, 0.0),
        n_modes=3,
    )
    lonlat = jnp.array([[0.0, 0.0]])
    trace = handlers.trace(encoder).get_trace(lonlat)

    assert any(name.endswith(".cap_radius") for name in trace)
    assert any(name.endswith(".cap_centre") for name in trace)
    assert encoder(lonlat).shape == (1, 3)


def test_slepian_encoder_filter_jit():
    basis = slepian_cap_basis(3, jnp.deg2rad(45.0), n_modes=4)
    encoder = SlepianEncoder(basis=basis, input_mode="cartesian")
    xyz = jnp.array([[1.0, 0.0, 0.0]])

    import equinox as eqx

    encoded = eqx.filter_jit(lambda x: jax.vmap(encoder)(x))(xyz)

    assert encoded.shape == (1, 4)
    assert jnp.all(jnp.isfinite(encoded))
