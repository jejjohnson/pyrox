"""Tests for `pyrox._basis._spectral_density`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyrox._basis import spectral_density
from pyrox.gp import RBF, Matern, Periodic


jax.config.update("jax_enable_x64", True)


def _integrate_to_kzero(spectral, omega_max=300.0, n=200_001):
    r"""Numerically integrate ``S(omega) / (2*pi) d omega`` to recover ``k(0)``."""
    omega = jnp.linspace(-omega_max, omega_max, n)
    S = spectral(omega**2)  # eigvals = omega^2
    return float(jnp.trapezoid(S, omega) / (2 * jnp.pi))


@pytest.mark.parametrize(
    ("variance", "lengthscale"),
    [(1.0, 1.0), (2.5, 0.5), (0.3, 2.0)],
)
def test_rbf_spectral_density_integrates_to_variance(variance, lengthscale):
    kernel = RBF(init_variance=variance, init_lengthscale=lengthscale)
    k0 = _integrate_to_kzero(lambda lam: spectral_density(kernel, lam, D=1))
    assert k0 == pytest.approx(variance, rel=1e-5)


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_matern_spectral_density_integrates_to_variance(nu):
    variance = 1.7
    lengthscale = 0.5
    kernel = Matern(init_variance=variance, init_lengthscale=lengthscale, nu=nu)
    k0 = _integrate_to_kzero(lambda lam: spectral_density(kernel, lam, D=1))
    # nu=0.5 has heavier tails so the truncated integral converges more slowly;
    # widen the tolerance there.
    rel = 1e-2 if nu == 0.5 else 1e-3
    assert k0 == pytest.approx(variance, rel=rel)


def test_rbf_spectral_density_shape():
    kernel = RBF(init_variance=1.0, init_lengthscale=1.0)
    eigvals = jnp.linspace(0.0, 10.0, 32)
    S = spectral_density(kernel, eigvals, D=1)
    assert S.shape == (32,)
    assert jnp.all(S > 0)


def test_unsupported_kernel_raises():
    kernel = Periodic(init_lengthscale=1.0, init_period=1.0)
    with pytest.raises(NotImplementedError, match="Periodic"):
        spectral_density(kernel, jnp.zeros(3), D=1)


def test_rbf_higher_D_lengthscale_exponent():
    """``D`` parameter must scale the RBF prefactor as ``ell^D``."""
    variance = 1.0
    lengthscale = 0.5
    kernel = RBF(init_variance=variance, init_lengthscale=lengthscale)
    eigvals = jnp.zeros(1)  # at omega = 0
    S_D1 = spectral_density(kernel, eigvals, D=1)
    S_D2 = spectral_density(kernel, eigvals, D=2)
    # S(0) = sigma^2 * ell^D * (2 pi)^{D/2}
    expected_D1 = variance * lengthscale * np.sqrt(2 * np.pi)
    expected_D2 = variance * (lengthscale**2) * (2 * np.pi)
    assert jnp.allclose(S_D1[0], expected_D1, rtol=1e-5)
    assert jnp.allclose(S_D2[0], expected_D2, rtol=1e-5)
