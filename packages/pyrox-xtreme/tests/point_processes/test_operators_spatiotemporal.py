"""Tests for the spatiotemporal PP operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from jax import random

from xtremax.point_processes import RectangularDomain, TemporalDomain
from xtremax.point_processes.operators import (
    HomogeneousSpatioTemporalPP,
    InhomogeneousSpatioTemporalPP,
    MarkedSpatioTemporalPP,
    SpatioTemporalHawkes,
)


class TestHomogeneousSpatioTemporalPP:
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_round_trip(self, d: int) -> None:
        spatial = RectangularDomain.from_size(jnp.full((d,), 3.0))
        temporal = TemporalDomain.from_duration(2.0)
        op = HomogeneousSpatioTemporalPP(rate=0.4, spatial=spatial, temporal=temporal)
        locs, times, mask, _ = op.sample(random.PRNGKey(0), max_events=128)
        assert locs.shape == (128, d)
        lp = op.log_prob(locs, times, mask)
        assert jnp.isfinite(lp)
        # Count log-prob normalises.
        ns = jnp.arange(60)
        log_pmf = jax.vmap(op.count_log_prob)(ns)
        assert jnp.isclose(jnp.sum(jnp.exp(log_pmf)), 1.0, atol=1e-3)

    def test_grad_through_rate(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        temporal = TemporalDomain.from_duration(2.0)

        def loss(rate):
            op = HomogeneousSpatioTemporalPP(
                rate=rate, spatial=spatial, temporal=temporal
            )
            return op.count_log_prob(jnp.asarray(20))

        grad = jax.grad(loss)(0.5)
        # d/dλ [n log(λ|D|T) − λ|D|T − log n!] = n/λ − |D|T.
        expected = 20 / 0.5 - 16.0 * 2.0
        assert jnp.allclose(grad, expected)


class TestInhomogeneousSpatioTemporalPP:
    def test_sample_log_prob_finite(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([5.0, 5.0]))
        temporal = TemporalDomain.from_duration(3.0)

        def log_lam(s, t):
            return -0.05 * jnp.sum((s - 2.5) ** 2, axis=-1) - 0.1 * t + jnp.log(2.0)

        op = InhomogeneousSpatioTemporalPP(
            log_lam,
            spatial,
            temporal,
            lambda_max=2.5,
            n_integration_points=2048,
        )
        locs, times, mask, _ = op.sample(random.PRNGKey(0), max_candidates=512)
        lp = op.log_prob(locs, times, mask)
        assert jnp.isfinite(lp)

    def test_pinned_integrated_intensity(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([3.0, 3.0]))
        temporal = TemporalDomain.from_duration(2.0)

        def log_lam(s, t):
            return jnp.full(s.shape[:-1], jnp.log(0.5))

        pinned = jnp.asarray(9.0)  # 0.5 * 9 * 2 = 9.0
        op = InhomogeneousSpatioTemporalPP(
            log_lam, spatial, temporal, integrated_intensity=pinned
        )
        # Default slab returns the pinned value verbatim.
        assert jnp.allclose(op.effective_integrated_intensity(), pinned)
        # Sub-spatial query triggers re-integration.
        sub_sp = RectangularDomain.from_size(jnp.array([1.0, 1.0]))
        sub_lambda = op.effective_integrated_intensity(sub_spatial=sub_sp)
        assert jnp.allclose(sub_lambda, 0.5 * 1.0 * 2.0, rtol=1e-2)

    def test_grad_through_intensity_params(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        temporal = TemporalDomain.from_duration(2.0)

        def loss(c):
            def log_lam(s, t):
                return c + jnp.zeros(s.shape[:-1])

            op = InhomogeneousSpatioTemporalPP(
                log_lam, spatial, temporal, n_integration_points=1024
            )
            locs = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
            times = jnp.array([0.2, 0.4, 0.6])
            mask = jnp.array([True, True, True])
            return op.log_prob(locs, times, mask)

        c = 0.0
        grad = jax.grad(loss)(c)
        # log L = ∑ c − exp(c) · 16 · 2 = 3c − exp(c)·32.
        expected = 3.0 - jnp.exp(c) * 32.0
        assert jnp.allclose(grad, expected, atol=1e-1)

    def test_chi_square_finite(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        temporal = TemporalDomain.from_duration(2.0)

        def log_lam(s, t):
            return jnp.full(s.shape[:-1], jnp.log(0.5))

        op = InhomogeneousSpatioTemporalPP(
            log_lam,
            spatial,
            temporal,
            lambda_max=0.5,
        )
        locs, times, mask, _ = op.sample(random.PRNGKey(0), max_candidates=128)
        chi2 = op.chi_square_gof(
            locs,
            times,
            mask,
            n_spatial_bins=2,
            n_temporal_bins=2,
            n_integration_points=8,
        )
        assert jnp.isfinite(chi2)


class TestMarkedSpatioTemporalPP:
    def test_log_prob_decomposes(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        temporal = TemporalDomain.from_duration(2.0)
        ground = HomogeneousSpatioTemporalPP(
            rate=0.5, spatial=spatial, temporal=temporal
        )
        mark_dist = dist.Gamma(concentration=5.0, rate=2.0)
        mpp = MarkedSpatioTemporalPP(
            ground=ground, mark_distribution_fn=lambda s, t: mark_dist
        )

        locs = jnp.array([[1.0, 1.0], [2.0, 3.0], [3.5, 0.5], [0.0, 0.0]])
        times = jnp.array([0.1, 0.5, 1.5, 0.0])
        mask = jnp.array([True, True, True, False])
        marks = jnp.array([1.0, 2.0, 0.5, 0.0])
        joint = mpp.log_prob(locs, times, mask, marks)
        ground_lp = ground.log_prob(locs, times, mask)
        marks_lp = jnp.sum(mark_dist.log_prob(marks[:3]))
        assert jnp.allclose(joint, ground_lp + marks_lp, rtol=1e-5)

    def test_sample_returns_marks(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([5.0, 5.0]))
        temporal = TemporalDomain.from_duration(3.0)
        ground = HomogeneousSpatioTemporalPP(
            rate=0.5, spatial=spatial, temporal=temporal
        )
        mark_dist = dist.Exponential(rate=1.0)
        mpp = MarkedSpatioTemporalPP(
            ground=ground, mark_distribution_fn=lambda s, t: mark_dist
        )
        locs, times, mask, marks = mpp.sample(random.PRNGKey(0), max_events=128)
        assert locs.shape == (128, 2)
        assert times.shape == (128,)
        assert mask.shape == (128,)
        assert marks.shape == (128,)
        assert jnp.all(marks[mask] >= 0.0)

    def test_time_dependent_marks(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        temporal = TemporalDomain.from_duration(2.0)
        ground = HomogeneousSpatioTemporalPP(
            rate=0.5, spatial=spatial, temporal=temporal
        )

        def fn(s, t):
            return dist.Exponential(rate=1.0 / (1.0 + t))

        mpp = MarkedSpatioTemporalPP(ground=ground, mark_distribution_fn=fn)
        locs = jnp.array([[1.0, 1.0], [3.0, 1.0]])
        times = jnp.array([0.2, 1.5])
        mask = jnp.array([True, True])
        marks = jnp.array([0.2, 1.5])
        lp = mpp.log_prob(locs, times, mask, marks)
        # Swapping times should change the joint log-prob because
        # mark rates are time-dependent.
        lp_swapped = mpp.log_prob(locs, times[::-1], mask, marks)
        assert not jnp.allclose(lp, lp_swapped)


class TestSpatioTemporalHawkes:
    def test_intensity_baseline_only(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([3.0, 3.0]))
        temporal = TemporalDomain.from_duration(2.0)
        op = SpatioTemporalHawkes(
            mu=0.4,
            alpha=0.5,
            beta=1.0,
            sigma=0.5,
            spatial=spatial,
            temporal=temporal,
        )
        # Empty history → intensity = mu.
        s = jnp.array([1.0, 1.0])
        empty_locs = jnp.zeros((4, 2))
        empty_times = jnp.zeros((4,))
        empty_mask = jnp.zeros((4,), dtype=jnp.bool_)
        lam = op.intensity(s, jnp.asarray(0.5), empty_locs, empty_times, empty_mask)
        assert jnp.allclose(lam, 0.4)

    def test_log_prob_grad_finite(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([3.0, 3.0]))
        temporal = TemporalDomain.from_duration(2.0)
        locs = jnp.array([[1.0, 1.0], [2.0, 2.0]])
        times = jnp.array([0.5, 1.0])
        mask = jnp.array([True, True])

        def loss(params):
            mu, alpha = params
            op = SpatioTemporalHawkes(
                mu=mu,
                alpha=alpha,
                beta=1.0,
                sigma=0.5,
                spatial=spatial,
                temporal=temporal,
            )
            return op.log_prob(locs, times, mask)

        grad = jax.grad(loss)(jnp.array([0.3, 0.2]))
        assert jnp.all(jnp.isfinite(grad))

    def test_sample_subcritical(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        temporal = TemporalDomain.from_duration(2.0)
        op = SpatioTemporalHawkes(
            mu=0.2,
            alpha=0.3,
            beta=1.0,
            sigma=0.5,
            spatial=spatial,
            temporal=temporal,
        )
        _locs, times, mask, n = op.sample(random.PRNGKey(0), max_events=64)
        assert n >= 0
        assert jnp.all(times[mask] >= 0.0)
        assert jnp.all(times[mask] < 2.0 + 1e-6)
