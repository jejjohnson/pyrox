"""Tests for :mod:`xtremax.point_processes.primitives.hpp_spatiotemporal`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import random

from xtremax.point_processes import RectangularDomain, TemporalDomain
from xtremax.point_processes.primitives import (
    hpp_spatiotemporal_count_log_prob,
    hpp_spatiotemporal_log_prob,
    hpp_spatiotemporal_predict_count,
    hpp_spatiotemporal_sample,
    ipp_spatiotemporal_log_prob,
)


class TestHppSpatiotemporalLogProb:
    def test_matches_closed_form(self) -> None:
        # Janossy form: log L = n log λ - λ |D| T.
        rate = 0.5
        vol = 100.0
        T = 10.0
        n = jnp.asarray(20)
        expected = 20 * jnp.log(rate) - rate * vol * T
        assert jnp.allclose(hpp_spatiotemporal_log_prob(n, rate, vol, T), expected)

    def test_matches_ipp_for_constant_intensity(self) -> None:
        rate = 0.7
        vol = 9.0
        T = 4.0
        locs = jnp.array([[1.0, 1.0], [1.5, 2.0], [2.5, 0.5], [0.0, 0.0], [0.0, 0.0]])
        times = jnp.array([0.5, 1.0, 1.5, 0.0, 0.0])
        mask = jnp.array([True, True, True, False, False])

        def log_lam(s, t):
            return jnp.full(s.shape[:-1], jnp.log(rate))

        ipp_lp = ipp_spatiotemporal_log_prob(locs, times, mask, log_lam, rate * vol * T)
        hpp_lp = hpp_spatiotemporal_log_prob(jnp.asarray(3), rate, vol, T)
        assert jnp.allclose(ipp_lp, hpp_lp)

    def test_grad_through_rate(self) -> None:
        # d(log L)/dλ = n/λ − |D|·T.
        n = jnp.asarray(15.0)
        vol = jnp.asarray(50.0)
        T = jnp.asarray(2.0)
        rate = 0.4
        grad = jax.grad(lambda r: hpp_spatiotemporal_log_prob(n, r, vol, T))(rate)
        expected = 15.0 / rate - 50.0 * 2.0
        assert jnp.allclose(grad, expected)

    def test_count_log_prob_normalises(self) -> None:
        rate = 0.8
        vol = 4.0
        T = 2.0
        ns = jnp.arange(40)
        log_pmf = jax.vmap(
            hpp_spatiotemporal_count_log_prob, in_axes=(0, None, None, None)
        )(ns, rate, vol, T)
        assert jnp.isclose(jnp.sum(jnp.exp(log_pmf)), 1.0, atol=1e-5)


class TestHppSpatiotemporalSample:
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_shapes_and_mask(self, d: int) -> None:
        spatial = RectangularDomain.from_size(jnp.full((d,), 5.0))
        temporal = TemporalDomain.from_duration(3.0)
        locs, times, mask, n = hpp_spatiotemporal_sample(
            random.PRNGKey(0),
            rate=0.3,
            spatial=spatial,
            temporal=temporal,
            max_events=64,
        )
        assert locs.shape == (64, d)
        assert times.shape == (64,)
        assert mask.shape == (64,)
        assert mask.sum() == jnp.minimum(n, 64)
        # Real events are inside the slab.
        assert jnp.all(times[mask] >= temporal.t0)
        assert jnp.all(times[mask] < temporal.t1 + 1e-6)
        for i in range(d):
            assert jnp.all(locs[mask, i] >= spatial.lo[i])
            assert jnp.all(locs[mask, i] < spatial.hi[i] + 1e-6)
        # Times are sorted at real events.
        sorted_times = jnp.sort(jnp.where(mask, times, jnp.inf))
        assert jnp.allclose(jnp.where(mask, times, jnp.inf), sorted_times)

    def test_empirical_count_matches_lambda_volume_T(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        temporal = TemporalDomain.from_duration(2.0)
        rate = 0.5  # E[N] = 0.5 * 16 * 2 = 16.
        keys = random.split(random.PRNGKey(42), 2000)

        def draw(k):
            return hpp_spatiotemporal_sample(
                k, rate, spatial, temporal, max_events=128
            )[3]

        counts = jax.vmap(draw)(keys)
        assert jnp.abs(jnp.mean(counts.astype(jnp.float32)) - 16.0) < 0.5

    def test_padding_locations_in_domain(self) -> None:
        spatial = RectangularDomain.from_size(jnp.full((2,), 5.0))
        temporal = TemporalDomain.from_duration(3.0)
        # Tiny rate → most rows are padding.
        locs, times, mask, _ = hpp_spatiotemporal_sample(
            random.PRNGKey(0),
            rate=0.01,
            spatial=spatial,
            temporal=temporal,
            max_events=32,
        )
        padding_locs = locs[~mask]
        assert jnp.all(jnp.isfinite(padding_locs))
        assert jnp.all(padding_locs == spatial.lo)
        assert jnp.all(times[~mask] == temporal.t1)


class TestHppSpatiotemporalPredictCount:
    def test_mean_eq_var(self) -> None:
        mean, var = hpp_spatiotemporal_predict_count(
            rate=0.5, sub_volume=8.0, sub_duration=2.0
        )
        assert jnp.allclose(mean, 8.0)
        assert jnp.allclose(var, 8.0)
