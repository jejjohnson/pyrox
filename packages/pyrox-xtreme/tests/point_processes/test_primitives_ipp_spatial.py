"""Tests for :mod:`xtremax.point_processes.primitives.ipp_spatial`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import random

from xtremax.point_processes import RectangularDomain
from xtremax.point_processes.primitives import (
    ipp_spatial_intensity,
    ipp_spatial_log_prob,
    ipp_spatial_predict_count,
    ipp_spatial_sample_thinning,
)


def _gaussian_log_intensity(s):
    # Smooth bump centred at (5, 5) with peak λ = 2.
    return -0.1 * jnp.sum((s - 5.0) ** 2, axis=-1) + jnp.log(2.0)


class TestIppSpatialLogProb:
    def test_log_prob_decomposition(self) -> None:
        # log L = ∑ log λ(sᵢ) − Λ. Choose easy points + an analytic Λ.
        locations = jnp.array([[1.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        # Pad to length 5; mask drops the padding.
        locations = jnp.pad(locations, ((0, 2), (0, 0)))
        mask = jnp.array([True, True, True, False, False])
        Lambda = jnp.asarray(7.5)
        log_lams = _gaussian_log_intensity(locations)
        expected = jnp.sum(jnp.where(mask, log_lams, 0.0)) - Lambda
        result = ipp_spatial_log_prob(locations, mask, _gaussian_log_intensity, Lambda)
        assert jnp.allclose(result, expected)

    def test_grad_through_intensity_params(self) -> None:
        # Gradient w.r.t. a closure parameter must flow through the sum.
        locations = jnp.array([[1.0, 1.0], [2.0, 3.0]])
        mask = jnp.array([True, True])

        def loss(c):
            def lam(s):
                return c * jnp.sum((s - 5.0) ** 2, axis=-1)

            return ipp_spatial_log_prob(locations, mask, lam, 1.0)

        # d/dc [c·||s1-5||² + c·||s2-5||² − 1] = sum of squared distances.
        s2 = jnp.sum((locations - 5.0) ** 2, axis=-1)
        grad = jax.grad(loss)(0.0)
        assert jnp.allclose(grad, jnp.sum(s2))


class TestIppSpatialSampleThinning:
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_shapes(self, d: int) -> None:
        domain = RectangularDomain.from_size(jnp.full((d,), 5.0))

        def log_lam(s):
            return jnp.full(s.shape[:-1], jnp.log(0.5))

        locs, mask, _ = ipp_spatial_sample_thinning(
            random.PRNGKey(0), log_lam, domain, lambda_max=0.5, max_candidates=128
        )
        assert locs.shape == (128, d)
        assert mask.shape == (128,)

    def test_constant_intensity_recovers_hpp_count(self) -> None:
        # Under constant log_intensity_fn = log λ_max, thinning accepts
        # everything and the empirical count should be ≈ λ |D|.
        domain = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        rate = 0.5

        def log_lam(s):
            return jnp.full(s.shape[:-1], jnp.log(rate))

        keys = random.split(random.PRNGKey(0), 1000)

        def count(k):
            _, mask, _ = ipp_spatial_sample_thinning(
                k, log_lam, domain, lambda_max=rate, max_candidates=256
            )
            return jnp.sum(mask)

        counts = jax.vmap(count)(keys)
        # E[N] = λ |D| = 8.
        assert jnp.abs(jnp.mean(counts.astype(jnp.float32)) - 8.0) < 0.4

    def test_intensity_recovery_by_subdomain_count(self) -> None:
        # Drawn intensity matches the analytical one: in a small box
        # centred on the peak we should get higher density than at the
        # edge.
        domain = RectangularDomain.from_size(jnp.array([10.0, 10.0]))

        def log_lam(s):
            return -0.5 * jnp.sum((s - 5.0) ** 2, axis=-1) + jnp.log(5.0)

        # ``λ_max ≈ 5`` at the peak.
        keys = random.split(random.PRNGKey(0), 200)

        def counts(k):
            locs, mask, _ = ipp_spatial_sample_thinning(
                k, log_lam, domain, lambda_max=5.0, max_candidates=512
            )
            in_centre = jnp.all(jnp.abs(locs - 5.0) < 1.0, axis=-1) & mask
            in_edge = jnp.all(jnp.abs(locs - 1.0) < 1.0, axis=-1) & mask
            return jnp.sum(in_centre), jnp.sum(in_edge)

        c_centre, c_edge = jax.vmap(counts)(keys)
        # Centre should be much denser than edge (peak intensity vs tail).
        assert (
            jnp.mean(c_centre.astype(jnp.float32))
            > jnp.mean(c_edge.astype(jnp.float32)) * 5.0
        )


class TestIppSpatialPredictCount:
    def test_subdomain_recovers_constant_rate(self) -> None:
        domain = RectangularDomain.from_size(jnp.array([4.0, 4.0]))

        def log_lam(s):
            return jnp.full(s.shape[:-1], jnp.log(0.5))

        result = ipp_spatial_predict_count(log_lam, domain, n_integration_points=2048)
        assert jnp.allclose(result, 0.5 * 16.0, rtol=1e-3)


class TestIppSpatialIntensity:
    def test_evaluates_pointwise(self) -> None:
        s = jnp.array([[1.0, 1.0], [5.0, 5.0]])
        result = ipp_spatial_intensity(s, _gaussian_log_intensity)
        expected = jnp.exp(_gaussian_log_intensity(s))
        assert jnp.allclose(result, expected)
