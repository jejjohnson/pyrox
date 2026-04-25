"""Tests for :mod:`xtremax.point_processes.primitives.hawkes_spatiotemporal`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

from xtremax.point_processes import RectangularDomain, TemporalDomain
from xtremax.point_processes.primitives import (
    stpp_hawkes_compensator,
    stpp_hawkes_intensity,
    stpp_hawkes_lambda_max,
    stpp_hawkes_log_prob,
    stpp_hawkes_sample,
)


class TestStppHawkesIntensity:
    def test_no_history_returns_baseline(self) -> None:
        # With an empty mask, intensity == mu everywhere.
        s = jnp.array([1.0, 2.0])
        t = jnp.asarray(1.5)
        locs = jnp.zeros((4, 2))
        times = jnp.zeros((4,))
        mask = jnp.zeros((4,), dtype=jnp.bool_)
        lam = stpp_hawkes_intensity(
            s, t, locs, times, mask, mu=0.3, alpha=0.5, beta=2.0, sigma=0.5
        )
        assert jnp.allclose(lam, 0.3)

    def test_excitation_decays_in_time(self) -> None:
        # Same spatial position; later query time → lower contribution.
        s = jnp.array([1.0, 1.0])
        locs = jnp.array([[1.0, 1.0], [0.0, 0.0]])
        times = jnp.array([0.5, 0.0])
        mask = jnp.array([True, False])
        lam_close = stpp_hawkes_intensity(
            s,
            jnp.asarray(0.6),
            locs,
            times,
            mask,
            mu=0.0,
            alpha=1.0,
            beta=2.0,
            sigma=0.5,
        )
        lam_far = stpp_hawkes_intensity(
            s,
            jnp.asarray(2.0),
            locs,
            times,
            mask,
            mu=0.0,
            alpha=1.0,
            beta=2.0,
            sigma=0.5,
        )
        assert lam_close > lam_far


class TestStppHawkesCompensator:
    def test_zero_alpha_recovers_homogeneous(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([3.0, 3.0]))
        temporal = TemporalDomain.from_duration(2.0)
        locs = jnp.zeros((4, 2))
        times = jnp.zeros((4,))
        mask = jnp.zeros((4,), dtype=jnp.bool_)
        Lambda = stpp_hawkes_compensator(
            locs,
            times,
            mask,
            mu=0.4,
            alpha=0.0,
            beta=1.0,
            sigma=0.5,
            spatial=spatial,
            temporal=temporal,
        )
        # μ |D| T = 0.4 * 9 * 2 = 7.2
        assert jnp.allclose(Lambda, 7.2)

    def test_excitation_adds_alpha_for_old_events(self) -> None:
        # An event at t=0 with β·(T-t)=β·T → temporal factor → 1 in the
        # limit; with boundary_correction=False the spatial factor is 1
        # too. Then excitation term = α·n.
        spatial = RectangularDomain.from_size(jnp.array([3.0, 3.0]))
        temporal = TemporalDomain.from_duration(100.0)
        locs = jnp.array([[1.0, 1.0]])
        times = jnp.array([0.0])
        mask = jnp.array([True])
        Lambda_zero = stpp_hawkes_compensator(
            locs,
            times,
            mask,
            mu=0.0,
            alpha=0.5,
            beta=1.0,
            sigma=0.5,
            spatial=spatial,
            temporal=temporal,
            boundary_correction=False,
        )
        assert jnp.allclose(Lambda_zero, 0.5, rtol=1e-5)


class TestStppHawkesLogProb:
    def test_zero_alpha_matches_hpp_log_prob(self) -> None:
        # alpha → 0 collapses to homogeneous: ∑ log μ - μ |D| T.
        from xtremax.point_processes.primitives import hpp_spatiotemporal_log_prob

        spatial = RectangularDomain.from_size(jnp.array([3.0, 3.0]))
        temporal = TemporalDomain.from_duration(2.0)
        locs = jnp.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]])
        times = jnp.array([0.2, 1.0, 0.0])
        mask = jnp.array([True, True, False])

        hawkes_lp = stpp_hawkes_log_prob(
            locs,
            times,
            mask,
            mu=0.5,
            alpha=0.0,
            beta=1.0,
            sigma=0.5,
            spatial=spatial,
            temporal=temporal,
            boundary_correction=False,
        )
        hpp_lp = hpp_spatiotemporal_log_prob(
            jnp.sum(mask), 0.5, spatial.volume(), temporal.duration
        )
        assert jnp.allclose(hawkes_lp, hpp_lp, rtol=1e-5)

    def test_grad_through_mu(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([3.0, 3.0]))
        temporal = TemporalDomain.from_duration(2.0)
        locs = jnp.array([[1.0, 1.0], [2.0, 2.0], [1.5, 0.5]])
        times = jnp.array([0.2, 1.0, 1.5])
        mask = jnp.array([True, True, True])

        def loss(mu):
            return stpp_hawkes_log_prob(
                locs,
                times,
                mask,
                mu=mu,
                alpha=0.1,
                beta=1.0,
                sigma=0.5,
                spatial=spatial,
                temporal=temporal,
            )

        grad = jax.grad(loss)(0.3)
        assert jnp.isfinite(grad)


class TestStppHawkesSample:
    def test_alpha_zero_matches_hpp_count(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        temporal = TemporalDomain.from_duration(5.0)
        keys = random.split(random.PRNGKey(42), 200)

        def draw(k):
            return stpp_hawkes_sample(
                k,
                mu=0.5,
                alpha=1e-6,
                beta=1.0,
                sigma=0.5,
                spatial=spatial,
                temporal=temporal,
                max_events=256,
            )[3]

        counts = jax.vmap(draw)(keys)
        # E[N] = μ |D| T = 0.5 * 16 * 5 = 40
        assert jnp.abs(jnp.mean(counts.astype(jnp.float32)) - 40.0) < 1.5

    def test_subcritical_run_finite(self) -> None:
        spatial = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        temporal = TemporalDomain.from_duration(2.0)
        locs, times, mask, n = stpp_hawkes_sample(
            random.PRNGKey(0),
            mu=0.2,
            alpha=0.3,
            beta=1.0,
            sigma=0.5,
            spatial=spatial,
            temporal=temporal,
            max_events=128,
        )
        assert n >= 0
        assert jnp.all(times[mask] >= temporal.t0)
        assert jnp.all(times[mask] < temporal.t1 + 1e-6)
        assert jnp.all(jnp.isfinite(locs))


class TestLambdaMax:
    def test_monotone_in_n(self) -> None:
        a = stpp_hawkes_lambda_max(0, mu=0.5, alpha=0.5, beta=1.0, sigma=0.5, n_dims=2)
        b = stpp_hawkes_lambda_max(10, mu=0.5, alpha=0.5, beta=1.0, sigma=0.5, n_dims=2)
        assert b > a
