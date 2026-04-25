"""Tests for the spatial PP operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from jax import random

from xtremax.point_processes import RectangularDomain
from xtremax.point_processes.operators import (
    HomogeneousSpatialPP,
    InhomogeneousSpatialPP,
    MarkedSpatialPP,
)


class TestHomogeneousSpatialPP:
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_round_trip(self, d: int) -> None:
        domain = RectangularDomain.from_size(jnp.full((d,), 3.0))
        op = HomogeneousSpatialPP(rate=0.5, domain=domain)
        locs, mask, _n = op.sample(random.PRNGKey(0), max_events=128)
        assert locs.shape == (128, d)
        # Round-trip log_prob should be finite.
        lp = op.log_prob(locs, mask)
        assert jnp.isfinite(lp)
        # Count log_prob normalises across a wide enough range.
        ns = jnp.arange(60)
        log_pmf = jax.vmap(op.count_log_prob)(ns)
        assert jnp.isclose(jnp.sum(jnp.exp(log_pmf)), 1.0, atol=1e-4)

    def test_grad_through_rate(self) -> None:
        domain = RectangularDomain.from_size(jnp.array([4.0, 4.0]))

        def loss(rate):
            op = HomogeneousSpatialPP(rate=rate, domain=domain)
            n = jnp.asarray(20)
            return op.count_log_prob(n)

        grad = jax.grad(loss)(0.5)
        # d/dλ [n log(λ|D|) − λ|D| − log n!] = n/λ − |D|.
        expected = 20 / 0.5 - 16.0
        assert jnp.allclose(grad, expected)

    def test_diagnostics(self) -> None:
        domain = RectangularDomain.from_size(jnp.array([10.0, 10.0]))
        op = HomogeneousSpatialPP(rate=0.5, domain=domain)
        # K(2) = π · 4 = 4π.
        assert jnp.allclose(op.ripleys_k(2.0), 4.0 * jnp.pi)
        # L(r) = r identically.
        assert jnp.allclose(
            op.l_function(jnp.linspace(0.5, 2.0, 5)),
            jnp.linspace(0.5, 2.0, 5),
            atol=1e-5,
        )


class TestInhomogeneousSpatialPP:
    def test_sample_log_prob_finite(self) -> None:
        domain = RectangularDomain.from_size(jnp.array([10.0, 10.0]))

        def log_lam(s):
            return -0.1 * jnp.sum((s - 5.0) ** 2, axis=-1) + jnp.log(2.0)

        # Peak intensity is at the centre with value ``2``; pass a true
        # upper bound (``lambda_max=2.0``) so the thinning sampler can
        # construct an exact envelope.
        op = InhomogeneousSpatialPP(log_lam, domain, lambda_max=2.0)
        locs, mask, _ = op.sample(random.PRNGKey(0), max_candidates=256)
        lp = op.log_prob(locs, mask)
        assert jnp.isfinite(lp)

    def test_pinned_integrated_intensity(self) -> None:
        # Pinned value short-circuits quadrature on the default domain.
        domain = RectangularDomain.from_size(jnp.array([4.0, 4.0]))

        def log_lam(s):
            return jnp.full(s.shape[:-1], jnp.log(0.5))

        pinned = jnp.asarray(8.0)
        op = InhomogeneousSpatialPP(log_lam, domain, integrated_intensity=pinned)
        # Default domain returns the pinned value verbatim.
        assert jnp.allclose(op.effective_integrated_intensity(), pinned)
        # A subdomain triggers a fresh quadrature.
        sub = RectangularDomain.from_size(jnp.array([2.0, 2.0]))
        sub_lambda = op.effective_integrated_intensity(sub)
        assert jnp.allclose(sub_lambda, 0.5 * 4.0, rtol=1e-3)

    def test_grad_through_intensity_params(self) -> None:
        domain = RectangularDomain.from_size(jnp.array([4.0, 4.0]))

        def loss(c):
            def log_lam(s):
                return c + jnp.zeros(s.shape[:-1])

            op = InhomogeneousSpatialPP(log_lam, domain, n_integration_points=512)
            locs = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
            mask = jnp.array([True, True, True])
            return op.log_prob(locs, mask)

        # log L = ∑ c − exp(c) · |D|. d/dc = n − exp(c)·|D|.
        c = 0.0
        grad = jax.grad(loss)(c)
        expected = 3.0 - jnp.exp(c) * 16.0
        assert jnp.allclose(grad, expected, atol=1e-2)


class TestMarkedSpatialPP:
    def test_log_prob_decomposes(self) -> None:
        domain = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        ground = HomogeneousSpatialPP(rate=0.5, domain=domain)
        mark_dist = dist.Gamma(concentration=5.0, rate=2.0)
        mpp = MarkedSpatialPP(ground=ground, mark_distribution_fn=lambda s: mark_dist)

        # Hand-build (locs, mask, marks).
        locs = jnp.array([[1.0, 1.0], [2.0, 3.0], [3.5, 0.5], [0.0, 0.0]])
        mask = jnp.array([True, True, True, False])
        marks = jnp.array([1.0, 2.0, 0.5, 0.0])
        joint_lp = mpp.log_prob(locs, mask, marks)
        ground_lp = ground.log_prob(locs, mask)
        mark_lp = jnp.sum(mark_dist.log_prob(marks[:3]))
        assert jnp.allclose(joint_lp, ground_lp + mark_lp)

    def test_sample_returns_marks(self) -> None:
        domain = RectangularDomain.from_size(jnp.array([5.0, 5.0]))
        ground = HomogeneousSpatialPP(rate=0.5, domain=domain)
        mark_dist = dist.Exponential(rate=1.0)
        mpp = MarkedSpatialPP(ground=ground, mark_distribution_fn=lambda s: mark_dist)
        locs, mask, marks = mpp.sample(random.PRNGKey(0), max_events=128)
        assert locs.shape == (128, 2)
        assert mask.shape == (128,)
        assert marks.shape == (128,)
        # Marks at real events are positive Exp draws.
        assert jnp.all(marks[mask] >= 0.0)

    def test_location_dependent_marks(self) -> None:
        # Marks pair with location-dependent rates; swapping locations
        # without swapping marks must change the joint log-prob.
        domain = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        ground = HomogeneousSpatialPP(rate=0.5, domain=domain)

        def fn(s):
            return dist.Exponential(rate=1.0 / (1.0 + s[0]))

        mpp = MarkedSpatialPP(ground=ground, mark_distribution_fn=fn)
        locs = jnp.array([[1.0, 1.0], [3.0, 1.0]])
        mask = jnp.array([True, True])
        # Distinct marks paired with distinct locations break the
        # symmetry: ∑(log λ_i − λ_i m_i) is order-dependent unless
        # marks happen to be equal.
        marks = jnp.array([0.2, 1.5])
        lp = mpp.log_prob(locs, mask, marks)
        lp_swapped = mpp.log_prob(locs[::-1], mask, marks)
        assert not jnp.allclose(lp, lp_swapped)
