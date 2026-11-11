"""functax's bases and sphere lift return geonnax's numbers at pyrox-nn's call sites.

pyrox-nn takes its bases from the ``functax.basis`` family modules and its
lon/lat lift from ``functax.geometry.lonlat_to_unit`` instead of geonnax. Each
case calls the geonnax function pyrox-nn used before and its functax
replacement with the arguments pyrox-nn passes, on fixed inputs. The tolerance
is float64 round-off, since the implementations are copies.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from functax.basis.features import (
    fourier_features,
    seasonal_features,
    seasonal_frequencies,
)
from functax.basis.fourier import fourier_basis
from functax.basis.slepian import slepian_cap_basis
from functax.basis.sphere import harmonic_degrees, real_spherical_harmonics
from functax.geometry import lonlat_to_unit


geonnax_basis = pytest.importorskip("geonnax.basis")
geonnax_geo = pytest.importorskip("geonnax.geo")

_ATOL = 1e-12
_RNG = np.random.default_rng(0)
_X1 = jnp.asarray(_RNG.uniform(-1.0, 1.0, size=17))
_X2 = jnp.asarray(_RNG.uniform(-0.9, 0.9, size=(11, 2)))
_LONLAT_RAD = jnp.asarray(
    np.stack(
        [_RNG.uniform(-np.pi, np.pi, 13), _RNG.uniform(-np.pi / 2, np.pi / 2, 13)], -1
    )
)
_XYZ = lonlat_to_unit(_LONLAT_RAD)


def _close(got, expected):
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(expected), rtol=0.0, atol=_ATOL
    )


@pytest.mark.parametrize("rescale", [False, True])
def test_fourier_features(rescale):
    _close(
        fourier_features(_X1, 5, rescale=rescale),
        geonnax_basis.fourier_features(_X1, 5, rescale=rescale),
    )


@pytest.mark.parametrize("rescale", [False, True])
def test_seasonal_features_and_frequencies(rescale):
    periods, harmonics = (7.0, 365.25), (2, 3)
    t = jnp.linspace(0.0, 400.0, 23)
    _close(
        seasonal_features(t, periods, harmonics, rescale=rescale),
        geonnax_basis.seasonal_features(t, periods, harmonics, rescale=rescale),
    )
    assert seasonal_frequencies(periods, harmonics) == (
        geonnax_basis.seasonal_frequencies(periods, harmonics)
    )


def test_fourier_basis():
    got_phi, got_lam = fourier_basis(_X2, (4, 5), (1.0, 1.5))
    old_phi, old_lam = geonnax_basis.fourier_basis(_X2, (4, 5), (1.0, 1.5))
    _close(got_phi, old_phi)
    _close(got_lam, old_lam)


def test_spherical_harmonics_and_degrees():
    _close(
        real_spherical_harmonics(_XYZ, 7),
        geonnax_basis.real_spherical_harmonics(_XYZ, 7),
    )
    np.testing.assert_array_equal(
        np.asarray(harmonic_degrees(7)), np.asarray(geonnax_basis.harmonic_degrees(7))
    )


@pytest.mark.parametrize(
    ("unit", "scale"), [("radians", 1.0), ("degrees", 180 / np.pi)]
)
def test_lonlat_to_unit_is_lonlat_to_cartesian3d(unit, scale):
    lonlat = _LONLAT_RAD * scale
    _close(
        lonlat_to_unit(lonlat, unit=unit),
        geonnax_geo.lonlat_to_cartesian3d(lonlat, input_unit=unit),
    )


def test_slepian_cap_basis():
    args = (8, jnp.deg2rad(30.0))
    kwargs = {
        "n_modes": 6,
        "eig_threshold": None,
        "lonlat_centre": jnp.deg2rad(jnp.array([20.0, 45.0])),
    }
    new = slepian_cap_basis(*args, **kwargs)
    old = geonnax_basis.slepian_cap_basis(*args, **kwargs)
    _close(new.eigenvalues, old.eigenvalues)
    _close(new.evaluate(_XYZ), old.evaluate(_XYZ))
