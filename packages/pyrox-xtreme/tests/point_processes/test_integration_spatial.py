"""Tests for :mod:`xtremax.point_processes._integration_spatial`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from xtremax.point_processes import (
    RectangularDomain,
    integrate_log_intensity_spatial,
)


class TestTrapezoidIntegration:
    def test_constant_intensity_d2(self) -> None:
        # ∫_[0,L]² λ ds = λ L².
        domain = RectangularDomain.from_size(jnp.array([3.0, 3.0]))

        def log_lam(s):
            return jnp.full(s.shape[:-1], jnp.log(2.5))

        result = integrate_log_intensity_spatial(
            log_lam, domain, n_points=1024, method="trapezoid"
        )
        assert jnp.allclose(result, 2.5 * 9.0, atol=1e-3)

    def test_separable_exponential_d2(self) -> None:
        # ∫_[0,L]² exp(c1 x + c2 y) ds = (e^{c1 L}-1)/c1 · (e^{c2 L}-1)/c2.
        L = 2.0
        c1, c2 = 0.5, -0.3
        domain = RectangularDomain.from_size(jnp.array([L, L]))

        def log_lam(s):
            return c1 * s[..., 0] + c2 * s[..., 1]

        exact = (jnp.exp(c1 * L) - 1) / c1 * (jnp.exp(c2 * L) - 1) / c2
        result = integrate_log_intensity_spatial(
            log_lam, domain, n_points=2500, method="trapezoid"
        )
        assert jnp.allclose(result, exact, atol=1e-2)


class TestQmcIntegration:
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_constant_intensity(self, d: int) -> None:
        domain = RectangularDomain.from_size(jnp.full((d,), 2.0))
        rate = 1.5

        def log_lam(s):
            return jnp.full(s.shape[:-1], jnp.log(rate))

        result = integrate_log_intensity_spatial(
            log_lam, domain, n_points=4096, method="qmc"
        )
        assert jnp.allclose(result, rate * 2.0**d, rtol=1e-3)

    def test_d3_quadratic(self) -> None:
        # ∫_[0,1]³ exp(-r²) ds = ∏_axis ∫_0^1 e^{-x²} dx ≈ 0.7468² × ... .
        domain = RectangularDomain.from_size(jnp.array([1.0, 1.0, 1.0]))

        def log_lam(s):
            return -jnp.sum(s**2, axis=-1)

        # Reference via 1-D erf.
        from jax.scipy.special import erf

        per_axis = (jnp.sqrt(jnp.pi) / 2.0) * erf(1.0)
        exact = per_axis**3

        result = integrate_log_intensity_spatial(
            log_lam, domain, n_points=8192, method="qmc"
        )
        # QMC convergence is sub-linear; 1% is comfortable at 8192 nodes.
        assert jnp.allclose(result, exact, rtol=1e-2)
