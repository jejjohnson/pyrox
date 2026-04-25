"""Tests for :mod:`xtremax.point_processes.primitives.ipp_spatiotemporal`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

from xtremax.point_processes import (
    RectangularDomain,
    TemporalDomain,
    integrate_log_intensity_spatiotemporal,
)
from xtremax.point_processes.primitives import (
    ipp_spatiotemporal_chi_square_gof,
    ipp_spatiotemporal_intensity_surface_at_time,
    ipp_spatiotemporal_log_prob,
    ipp_spatiotemporal_marginal_spatial_intensity,
    ipp_spatiotemporal_marginal_temporal_intensity,
    ipp_spatiotemporal_pearson_residuals,
    ipp_spatiotemporal_predict_count,
    ipp_spatiotemporal_sample_thinning,
)


class TestIppSpatiotemporalLogProb:
    def test_separable_constant(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([3.0, 3.0]))
        temporal = TemporalDomain.from_duration(2.0)
        rate = 0.6

        def log_lam(s, t):
            return jnp.full(s.shape[:-1], jnp.log(rate))

        Lambda = integrate_log_intensity_spatiotemporal(log_lam, spatial, temporal)
        assert jnp.allclose(Lambda, rate * 9.0 * 2.0, rtol=1e-3)

        locs = jnp.array([[0.5, 0.5], [1.0, 1.5], [2.0, 0.5], [0.0, 0.0], [0.0, 0.0]])
        times = jnp.array([0.2, 0.5, 1.5, 0.0, 0.0])
        mask = jnp.array([True, True, True, False, False])
        lp = ipp_spatiotemporal_log_prob(locs, times, mask, log_lam, Lambda)
        # 3 log(λ) - λ|D|T
        expected = 3.0 * jnp.log(rate) - rate * 9.0 * 2.0
        assert jnp.allclose(lp, expected, rtol=1e-3)

    def test_grad_through_intensity(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        temporal = TemporalDomain.from_duration(2.0)
        locs = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        times = jnp.array([0.2, 0.4, 0.6])
        mask = jnp.array([True, True, True])

        def loss(c):
            def log_lam(s, t):
                return c + jnp.zeros(s.shape[:-1])

            Lambda = integrate_log_intensity_spatiotemporal(
                log_lam, spatial, temporal, n_points=2048
            )
            return ipp_spatiotemporal_log_prob(locs, times, mask, log_lam, Lambda)

        c = 0.0
        grad = jax.grad(loss)(c)
        # log L = ∑ c − exp(c) · |D| · T = 3c − exp(c)·32
        expected = 3.0 - jnp.exp(c) * 32.0
        assert jnp.allclose(grad, expected, atol=1e-1)


class TestIppSpatiotemporalSampleThinning:
    def test_acceptance_under_constant(self) -> None:
        # Constant intensity matches HPP-ST counts at λ = lambda_max.
        spatial = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        temporal = TemporalDomain.from_duration(2.0)

        def log_lam(s, t):
            return jnp.full(s.shape[:-1], jnp.log(0.5))

        keys = random.split(random.PRNGKey(0), 200)

        def draw(k):
            return ipp_spatiotemporal_sample_thinning(
                k, log_lam, spatial, temporal, lambda_max=0.5, max_candidates=128
            )[3]

        counts = jax.vmap(draw)(keys)
        # E[N] = 0.5 * 16 * 2 = 16
        assert jnp.abs(jnp.mean(counts.astype(jnp.float32)) - 16.0) < 1.0

    def test_padding_invariants(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([2.0, 2.0]))
        temporal = TemporalDomain.from_duration(1.0)

        def log_lam(s, t):
            return jnp.full(s.shape[:-1], jnp.log(0.05))

        locs, times, mask, _ = ipp_spatiotemporal_sample_thinning(
            random.PRNGKey(0),
            log_lam,
            spatial,
            temporal,
            lambda_max=0.05,
            max_candidates=64,
        )
        # All padding rows should be at domain.lo / temporal.t1.
        assert jnp.all(locs[~mask] == spatial.lo)
        assert jnp.all(times[~mask] == temporal.t1)


class TestMarginalIntensities:
    def test_marginal_spatial_constant(self) -> None:
        temporal = TemporalDomain.from_duration(4.0)

        def log_lam(s, t):
            return jnp.full(s.shape[:-1], jnp.log(0.7))

        locs = jnp.array([[0.5, 0.5], [2.0, 2.0]])
        marginal = ipp_spatiotemporal_marginal_spatial_intensity(
            locs, log_lam, temporal, n_time_points=32
        )
        # ∫_0^4 0.7 dt = 2.8
        assert jnp.allclose(marginal, jnp.array([2.8, 2.8]), rtol=1e-4)

    def test_marginal_temporal_constant(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([3.0, 3.0]))

        def log_lam(s, t):
            return jnp.full(s.shape[:-1], jnp.log(0.7))

        times = jnp.array([0.5, 1.0, 3.0])
        marginal = ipp_spatiotemporal_marginal_temporal_intensity(
            times, log_lam, spatial, n_spatial_points=512
        )
        # ∫_D 0.7 ds = 0.7 * 9 = 6.3
        assert jnp.allclose(marginal, 6.3 * jnp.ones(3), rtol=1e-2)

    def test_intensity_surface_shapes(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([2.0, 2.0]))

        def log_lam(s, t):
            return jnp.full(s.shape[:-1], jnp.log(1.0)) + 0.0 * t

        grid, intensity = ipp_spatiotemporal_intensity_surface_at_time(
            jnp.asarray(0.5), log_lam, spatial, grid_size=10
        )
        assert grid.shape == (10, 10, 2)
        assert intensity.shape == (10, 10)
        assert jnp.allclose(intensity, jnp.ones((10, 10)))


class TestPredictCount:
    def test_mean_eq_var(self) -> None:
        mean, var = ipp_spatiotemporal_predict_count(jnp.asarray(7.0))
        assert jnp.allclose(mean, 7.0)
        assert jnp.allclose(var, 7.0)


class TestPearsonResiduals:
    def test_residuals_under_truth_are_small(self) -> None:
        # Sample under λ = 0.5 then evaluate residuals against the same
        # log_lam. Residuals should be O(1) in scale.
        spatial = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        temporal = TemporalDomain.from_duration(2.0)

        def log_lam(s, t):
            return jnp.full(s.shape[:-1], jnp.log(0.5))

        locs, times, mask, _ = ipp_spatiotemporal_sample_thinning(
            random.PRNGKey(0),
            log_lam,
            spatial,
            temporal,
            lambda_max=0.5,
            max_candidates=128,
        )
        residuals = ipp_spatiotemporal_pearson_residuals(
            locs,
            times,
            mask,
            log_lam,
            spatial,
            temporal,
            n_spatial_bins=2,
            n_temporal_bins=2,
            n_integration_points=8,
        )
        # 2*2*2 = 8 cells; residuals finite.
        assert residuals.shape == (8,)
        assert jnp.all(jnp.isfinite(residuals))
        # Average per-cell expected count is ~16/8 = 2; residuals should
        # be O(1) — sanity check via |r| ≤ 5 across all cells.
        assert jnp.all(jnp.abs(residuals) < 5.0)

    def test_chi_square_finite(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        temporal = TemporalDomain.from_duration(2.0)

        def log_lam(s, t):
            return jnp.full(s.shape[:-1], jnp.log(0.5))

        locs, times, mask, _ = ipp_spatiotemporal_sample_thinning(
            random.PRNGKey(1),
            log_lam,
            spatial,
            temporal,
            lambda_max=0.5,
            max_candidates=128,
        )
        chi2 = ipp_spatiotemporal_chi_square_gof(
            locs,
            times,
            mask,
            log_lam,
            spatial,
            temporal,
            n_spatial_bins=2,
            n_temporal_bins=2,
            n_integration_points=8,
        )
        assert jnp.isfinite(chi2)
        assert chi2 >= 0.0
