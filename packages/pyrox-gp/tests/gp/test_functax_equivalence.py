"""The inducing features give the same numbers on functax's bases as on geonnax's.

pyrox-gp's bases moved from ``geonnax.basis`` to the ``functax.basis`` family
modules unchanged. Each case builds the features and evaluates ``K_uu`` and
``k_ux`` twice: once as the module stands, and once with the six basis names
``pyrox_gp._inducing`` imports patched back to their geonnax objects (the code
path before the move). The tolerance is float64 round-off, since the
implementations are copies.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pyrox_gp._inducing as inducing
import pytest
from pyrox_gp import (
    RBF,
    FourierInducingFeatures,
    Matern,
    SlepianInducingFeatures,
    SphericalHarmonicInducingFeatures,
)


geonnax_basis = pytest.importorskip("geonnax.basis")

_NAMES = (
    "SlepianCapBasis",
    "fourier_basis",
    "fourier_eigenvalues",
    "harmonic_degrees",
    "real_spherical_harmonics",
    "slepian_cap_basis",
)


def _unit_xyz(n: int) -> jnp.ndarray:
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n, 3))
    return jnp.asarray(v / np.linalg.norm(v, axis=1, keepdims=True))


def _fourier_1d():
    f = FourierInducingFeatures.init(in_features=1, num_basis_per_dim=12, L=3.0)
    return f, jnp.linspace(-2.0, 2.0, 9).reshape(-1, 1), RBF(init_lengthscale=0.5)


def _fourier_2d():
    f = FourierInducingFeatures.init(
        in_features=2, num_basis_per_dim=(4, 5), L=(1.0, 1.5)
    )
    x = jnp.asarray(np.random.default_rng(1).uniform(-0.9, 0.9, size=(7, 2)))
    return f, x, Matern(nu=1.5, init_lengthscale=0.4)


def _spherical():
    f = SphericalHarmonicInducingFeatures.init(l_max=6)
    return f, _unit_xyz(11), RBF(init_lengthscale=0.7, init_variance=1.3)


def _slepian():
    f = SlepianInducingFeatures.init(
        l_max=8, cap_radius_deg=30.0, cap_centre_lonlat_deg=(20.0, 45.0)
    )
    return f, _unit_xyz(13), Matern(nu=2.5, init_lengthscale=0.6)


def _evaluate(build):
    features, x, kernel = build()
    return (
        np.asarray(features.K_uu(kernel).as_matrix()),
        np.asarray(features.k_ux(x, kernel)),
    )


@pytest.mark.parametrize("build", [_fourier_1d, _fourier_2d, _spherical, _slepian])
def test_inducing_features_match_the_geonnax_code_path(build, monkeypatch):
    new = _evaluate(build)
    with monkeypatch.context() as patch:
        for name in _NAMES:
            patch.setattr(inducing, name, getattr(geonnax_basis, name))
        old = _evaluate(build)
    for got, expected in zip(new, old, strict=True):
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)
