"""Tests for the spatiotemporal Lebesgue integrator."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from xtremax.point_processes import (
    RectangularDomain,
    TemporalDomain,
    integrate_log_intensity_spatiotemporal,
)


class TestSpatiotemporalIntegration:
    def test_constant_intensity_qmc(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([3.0, 3.0]))
        temporal = TemporalDomain.from_duration(2.0)

        def log_lam(s, t):
            return jnp.full(s.shape[:-1], jnp.log(0.5))

        Lambda = integrate_log_intensity_spatiotemporal(
            log_lam, spatial, temporal, n_points=2048, method="qmc"
        )
        assert jnp.allclose(Lambda, 0.5 * 9.0 * 2.0, rtol=1e-3)

    def test_constant_intensity_trapezoid(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([3.0, 3.0]))
        temporal = TemporalDomain.from_duration(2.0)

        def log_lam(s, t):
            return jnp.full(s.shape[:-1], jnp.log(0.5))

        Lambda = integrate_log_intensity_spatiotemporal(
            log_lam, spatial, temporal, n_points=512, method="trapezoid"
        )
        # Trapezoid is exact for constants.
        assert jnp.allclose(Lambda, 9.0, rtol=1e-5)

    def test_separable_product(self) -> None:
        # λ(s, t) = exp(-||s||²/2) · exp(t/3) — verify separable
        # integration matches trapezoid in 1D × QMC in space.
        spatial = RectangularDomain(
            lo=jnp.array([-2.0, -2.0]), hi=jnp.array([2.0, 2.0])
        )
        temporal = TemporalDomain.from_duration(3.0)

        def log_lam(s, t):
            return -0.5 * jnp.sum(s * s, axis=-1) + t / 3.0

        # Reference: spatial integral via QMC × temporal integral closed-form.
        from xtremax.point_processes import integrate_log_intensity_spatial

        spatial_int = integrate_log_intensity_spatial(
            lambda s: -0.5 * jnp.sum(s * s, axis=-1), spatial, n_points=4096
        )
        temporal_int = 3.0 * (jnp.exp(1.0) - 1.0)  # ∫_0^3 e^(t/3) dt
        ref = spatial_int * temporal_int

        Lambda = integrate_log_intensity_spatiotemporal(
            log_lam, spatial, temporal, n_points=4096, method="qmc"
        )
        assert jnp.allclose(Lambda, ref, rtol=2e-2)

    def test_unknown_method_raises(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([1.0]))
        temporal = TemporalDomain.from_duration(1.0)
        with pytest.raises(ValueError, match="Unknown method"):
            integrate_log_intensity_spatiotemporal(
                lambda s, t: jnp.zeros(s.shape[:-1]),
                spatial,
                temporal,
                method="bogus",
            )
