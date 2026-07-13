"""Tests for Slepian cap basis primitives."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from pyrox_gp._basis import (
    shannon_number,
    slepian_cap_basis,
    slepian_cap_eigh_per_m,
    slepian_concentration_matrix,
)


def test_slepian_eigenvalues_are_sorted_concentration_ratios():
    vals, coeffs = slepian_cap_eigh_per_m(6, jnp.deg2rad(40.0))

    assert vals.shape == (49,)
    assert coeffs.shape == (49, 49)
    assert jnp.all(vals >= 0.0)
    assert jnp.all(vals <= 1.0)
    assert jnp.all(vals[:-1] >= vals[1:])


def test_slepian_concentration_matrix_is_symmetric():
    matrix = slepian_concentration_matrix(4, jnp.deg2rad(35.0))

    np.testing.assert_allclose(np.asarray(matrix), np.asarray(matrix.T), atol=1e-6)


def test_slepian_shannon_number_matches_well_concentrated_count():
    l_max = 10
    radius = jnp.deg2rad(40.0)
    vals, _ = slepian_cap_eigh_per_m(l_max, radius)
    area = 2.0 * jnp.pi * (1.0 - jnp.cos(radius))
    expected = shannon_number(l_max, area)

    assert abs(int((vals > 0.5).sum()) - float(expected)) <= 2.0


def test_slepian_coefficients_are_orthonormal_in_sh_space():
    basis = slepian_cap_basis(5, jnp.deg2rad(45.0), n_modes=8)

    gram = basis.coeffs.T @ basis.coeffs
    np.testing.assert_allclose(np.asarray(gram), np.eye(8), atol=1e-5)


def test_slepian_basis_evaluate_and_rotate_to_cap_centre():
    north_basis = slepian_cap_basis(4, jnp.deg2rad(50.0), n_modes=6)
    rotated = north_basis.rotate_to(jnp.array([0.25 * jnp.pi, 0.0]))
    north_pole = jnp.array([[0.0, 0.0, 1.0]])
    cap_centre = jnp.array([[2**-0.5, 2**-0.5, 0.0]])

    assert rotated.evaluate(cap_centre).shape == (1, 6)
    np.testing.assert_allclose(
        np.asarray(rotated.evaluate(cap_centre)),
        np.asarray(north_basis.evaluate(north_pole)),
        atol=1e-5,
    )
