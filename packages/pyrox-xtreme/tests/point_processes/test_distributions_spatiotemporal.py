"""Tests for the NumPyro spatiotemporal PP distribution wrappers."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from jax import random

from xtremax.point_processes import RectangularDomain, TemporalDomain
from xtremax.point_processes.distributions import (
    HomogeneousSpatioTemporalPP,
    InhomogeneousSpatioTemporalPP,
    MarkedSpatioTemporalPP,
    SpatioTemporalHawkes,
)
from xtremax.point_processes.operators import (
    HomogeneousSpatioTemporalPP as HppStOp,
    InhomogeneousSpatioTemporalPP as IppStOp,
)


class TestHomogeneousSpatioTemporalPPDistribution:
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_round_trip(self, d: int) -> None:
        spatial = RectangularDomain.from_size(jnp.full((d,), 4.0))
        temporal = TemporalDomain.from_duration(2.0)
        d_pp = HomogeneousSpatioTemporalPP(
            rate=0.4, spatial=spatial, temporal=temporal, max_events=128
        )
        locs, times, mask = d_pp.sample(random.PRNGKey(0))
        lp = d_pp.log_prob((locs, times, mask))
        assert jnp.isfinite(lp)
        assert locs.shape == (128, d)
        assert times.shape == (128,)

    def test_log_prob_with_count(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        temporal = TemporalDomain.from_duration(2.0)
        d_pp = HomogeneousSpatioTemporalPP(rate=0.5, spatial=spatial, temporal=temporal)
        locs, times, mask = d_pp.sample(random.PRNGKey(0))
        n = jnp.sum(mask).astype(jnp.int32)
        lp_count = d_pp.log_prob((locs, times, n))
        lp_mask = d_pp.log_prob((locs, times, mask))
        assert jnp.allclose(lp_count, lp_mask)

    def test_rejects_invalid_key(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([2.0, 2.0]))
        temporal = TemporalDomain.from_duration(1.0)
        d_pp = HomogeneousSpatioTemporalPP(rate=1.0, spatial=spatial, temporal=temporal)
        with pytest.raises(TypeError):
            d_pp.sample(jnp.array([1, 2, 3]))

    def test_rejects_batched_rate(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([2.0, 2.0]))
        temporal = TemporalDomain.from_duration(1.0)
        with pytest.raises(ValueError, match="scalar `rate`"):
            HomogeneousSpatioTemporalPP(
                rate=jnp.array([0.5, 1.0]), spatial=spatial, temporal=temporal
            )


class TestInhomogeneousSpatioTemporalPPDistribution:
    def test_round_trip(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([5.0, 5.0]))
        temporal = TemporalDomain.from_duration(3.0)

        def log_lam(s, t):
            return -0.05 * jnp.sum((s - 2.5) ** 2, axis=-1) - 0.05 * t + jnp.log(2.0)

        d_pp = InhomogeneousSpatioTemporalPP(
            log_lam, spatial, temporal, lambda_max=2.0, max_candidates=256
        )
        locs, times, mask = d_pp.sample(random.PRNGKey(0))
        lp = d_pp.log_prob((locs, times, mask))
        assert jnp.isfinite(lp)


class TestMarkedSpatioTemporalPPDistribution:
    def test_round_trip_independent_marks(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([5.0, 5.0]))
        temporal = TemporalDomain.from_duration(2.0)
        ground = HppStOp(rate=0.5, spatial=spatial, temporal=temporal)
        mark_dist = dist.Gamma(concentration=5.0, rate=2.0)
        d_pp = MarkedSpatioTemporalPP(
            ground=ground,
            mark_distribution_fn=lambda s, t: mark_dist,
            max_events=128,
        )
        locs, times, mask, marks = d_pp.sample(random.PRNGKey(0))
        assert locs.shape[0] == 128
        assert times.shape[0] == 128
        assert marks.shape[0] == 128
        lp = d_pp.log_prob((locs, times, mask, marks))
        assert jnp.isfinite(lp)

    def test_ipp_ground(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([5.0, 5.0]))
        temporal = TemporalDomain.from_duration(2.0)

        def log_lam(s, t):
            return jnp.full(s.shape[:-1], jnp.log(0.3)) + 0.0 * t

        ipp_op = IppStOp(log_lam, spatial, temporal, lambda_max=0.3)
        mark_dist = dist.Exponential(rate=1.0)
        d_pp = MarkedSpatioTemporalPP(
            ground=ipp_op,
            mark_distribution_fn=lambda s, t: mark_dist,
            max_events=256,
        )
        locs, times, mask, marks = d_pp.sample(random.PRNGKey(0))
        assert locs.shape[0] == 256
        lp = d_pp.log_prob((locs, times, mask, marks))
        assert jnp.isfinite(lp)


class TestSpatioTemporalHawkesDistribution:
    def test_round_trip(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        temporal = TemporalDomain.from_duration(2.0)
        d_pp = SpatioTemporalHawkes(
            mu=0.2,
            alpha=0.3,
            beta=1.0,
            sigma=0.5,
            spatial=spatial,
            temporal=temporal,
            max_events=64,
        )
        locs, times, mask = d_pp.sample(random.PRNGKey(0))
        lp = d_pp.log_prob((locs, times, mask))
        assert jnp.isfinite(lp)

    def test_rejects_batched_param(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([2.0, 2.0]))
        temporal = TemporalDomain.from_duration(1.0)
        with pytest.raises(ValueError, match="scalar"):
            SpatioTemporalHawkes(
                mu=jnp.array([0.2, 0.3]),
                alpha=0.3,
                beta=1.0,
                sigma=0.5,
                spatial=spatial,
                temporal=temporal,
            )
