"""Tests for the NumPyro spatial PP distribution wrappers."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from jax import random

from xtremax.point_processes import RectangularDomain
from xtremax.point_processes.distributions import (
    HomogeneousSpatialPP,
    InhomogeneousSpatialPP,
    MarkedSpatialPP,
)
from xtremax.point_processes.operators import (
    HomogeneousSpatialPP as HppOp,
    InhomogeneousSpatialPP as IppOp,
)


class TestHomogeneousSpatialPPDistribution:
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_round_trip(self, d: int) -> None:
        domain = RectangularDomain.from_size(jnp.full((d,), 5.0))
        d_pp = HomogeneousSpatialPP(rate=0.4, domain=domain, max_events=128)
        locs, mask = d_pp.sample(random.PRNGKey(0))
        lp = d_pp.log_prob((locs, mask))
        assert jnp.isfinite(lp)
        assert locs.shape == (128, d)

    def test_log_prob_with_count(self) -> None:
        domain = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        d_pp = HomogeneousSpatialPP(rate=0.5, domain=domain)
        locs, mask = d_pp.sample(random.PRNGKey(0))
        n = jnp.sum(mask).astype(jnp.int32)
        # Pass an integer count instead of the bool mask.
        lp_count = d_pp.log_prob((locs, n))
        lp_mask = d_pp.log_prob((locs, mask))
        assert jnp.allclose(lp_count, lp_mask)

    def test_rejects_invalid_key(self) -> None:
        domain = RectangularDomain.from_size(jnp.array([2.0, 2.0]))
        d_pp = HomogeneousSpatialPP(rate=1.0, domain=domain)
        with pytest.raises(TypeError):
            d_pp.sample(jnp.array([1, 2, 3]))


class TestInhomogeneousSpatialPPDistribution:
    def test_round_trip(self) -> None:
        domain = RectangularDomain.from_size(jnp.array([10.0, 10.0]))

        def log_lam(s):
            return -0.1 * jnp.sum((s - 5.0) ** 2, axis=-1) + jnp.log(2.0)

        d_pp = InhomogeneousSpatialPP(
            log_lam, domain, lambda_max=2.0, max_candidates=256
        )
        locs, mask = d_pp.sample(random.PRNGKey(0))
        lp = d_pp.log_prob((locs, mask))
        assert jnp.isfinite(lp)


class TestMarkedSpatialPPDistribution:
    def test_round_trip_independent_marks(self) -> None:
        domain = RectangularDomain.from_size(jnp.array([5.0, 5.0]))
        ground_op = HppOp(rate=0.5, domain=domain)
        mark_dist = dist.Gamma(concentration=5.0, rate=2.0)
        d_pp = MarkedSpatialPP(
            ground=ground_op,
            mark_distribution_fn=lambda s: mark_dist,
            max_events=128,
        )
        locs, mask, marks = d_pp.sample(random.PRNGKey(0))
        assert locs.shape[0] == 128
        assert marks.shape[0] == 128
        lp = d_pp.log_prob((locs, mask, marks))
        assert jnp.isfinite(lp)

    def test_ipp_ground(self) -> None:
        domain = RectangularDomain.from_size(jnp.array([5.0, 5.0]))

        def log_lam(s):
            return jnp.full(s.shape[:-1], jnp.log(0.3))

        # Constant log-intensity of 0.3 — pass a tight upper bound.
        ipp_op = IppOp(log_lam, domain, lambda_max=0.3)
        mark_dist = dist.Exponential(rate=1.0)
        d_pp = MarkedSpatialPP(
            ground=ipp_op,
            mark_distribution_fn=lambda s: mark_dist,
            max_events=256,
        )
        locs, mask, marks = d_pp.sample(random.PRNGKey(0))
        assert locs.shape[0] == 256
        # Round-trip log_prob.
        lp = d_pp.log_prob((locs, mask, marks))
        assert jnp.isfinite(lp)
