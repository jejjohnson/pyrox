"""Tests for the scalar Gaussian-expectation integrators."""

from __future__ import annotations

import jax.numpy as jnp

from pyrox.gp import GaussHermite, MonteCarlo


def test_gauss_hermite_integrates_polynomial_exactly() -> None:
    """Gauss-Hermite of order ``deg`` is exact for polynomials of
    degree ``2*deg - 1``. With ``deg = 20`` we exactly integrate
    ``f^4`` against ``N(m, v)`` and recover ``m^4 + 6 m^2 v + 3 v^2``.
    """
    integ = GaussHermite(deg=20)
    m = jnp.array([0.5, -1.0])
    v = jnp.array([0.3, 1.2])
    expected = m**4 + 6.0 * m**2 * v + 3.0 * v**2

    out = integ.integrate(lambda f: f**4, m, v)
    assert jnp.allclose(out, expected, atol=1e-8)


def test_gauss_hermite_recovers_mean() -> None:
    """``E[f] = m`` should be exact (degree-1 polynomial)."""
    integ = GaussHermite(deg=20)
    m = jnp.array([2.0, -0.5])
    v = jnp.array([1.0, 0.5])
    out = integ.integrate(lambda f: f, m, v)
    assert jnp.allclose(out, m, atol=1e-10)


def test_gauss_hermite_recovers_second_moment() -> None:
    """``E[f^2] = m^2 + v``."""
    integ = GaussHermite(deg=20)
    m = jnp.array([0.0, 1.0])
    v = jnp.array([1.0, 0.4])
    out = integ.integrate(lambda f: f**2, m, v)
    assert jnp.allclose(out, m**2 + v, atol=1e-10)


def test_monte_carlo_close_to_truth_with_enough_samples() -> None:
    """With 4096 samples, MC should hit ``E[f^2] = v + m^2`` to
    moderate precision."""
    integ = MonteCarlo(n=4096)
    m = jnp.array([0.0, 1.0])
    v = jnp.array([1.0, 0.4])
    out = integ.integrate(lambda f: f**2, m, v)
    assert jnp.allclose(out, m**2 + v, atol=0.2)
