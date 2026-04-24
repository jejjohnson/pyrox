"""Tests for renewal-process primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from xtremax.point_processes.primitives.hpp import hpp_log_prob
from xtremax.point_processes.primitives.renewal import (
    renewal_cumulative_hazard,
    renewal_hazard,
    renewal_log_prob,
    renewal_sample,
    renewal_survival,
)


class TestRenewalLogProb:
    def test_exponential_matches_hpp_log_prob(self):
        """Renewal(Exponential(λ)) coincides with HPP(λ) on the same events.

        The renewal log-likelihood is Σ log f(Δt) + log S(T - t_n); for an
        Exponential inter-event law this simplifies to n log λ − λ T,
        which is exactly `hpp_log_prob`.
        """
        rate = 2.0
        T = 5.0
        event_times = jnp.array([0.5, 1.8, 2.7, 3.9, 4.5])
        mask = jnp.array([True, True, True, True, True])
        renewal_ll = renewal_log_prob(event_times, mask, T, dist.Exponential(rate))
        hpp_ll = hpp_log_prob(jnp.sum(mask), rate, T)
        assert float(renewal_ll) == pytest.approx(float(hpp_ll), rel=1e-5)

    def test_handles_empty_sequence(self):
        T = 5.0
        mask = jnp.zeros(4, dtype=jnp.bool_)
        event_times = jnp.full((4,), T)
        ll = renewal_log_prob(event_times, mask, T, dist.Exponential(2.0))
        # Empty sequence: log L = -λ T. Float32 loses precision when
        # cdf(T) is close to 1, so allow a mild absolute tolerance.
        assert float(ll) == pytest.approx(-2.0 * T, abs=1e-2)

    def test_grad_through_inter_event_params(self):
        """Gradient through the inter-event Exponential rate should flow cleanly."""
        event_times = jnp.array([0.3, 1.0, 1.9])
        mask = jnp.ones(3, dtype=jnp.bool_)
        T = 3.0

        def loss(rate):
            d = dist.Exponential(rate)
            return -renewal_log_prob(event_times, mask, T, d)

        g = jax.grad(loss)(jnp.asarray(2.0))
        # d/dλ of -(n log λ - λ T) = -n/λ + T; at λ=2, n=3, T=3 → -1.5 + 3 = 1.5
        assert float(g) == pytest.approx(1.5, rel=1e-4)


class TestRenewalSample:
    def test_sampled_count_is_Poisson_for_exponential(self):
        """Exponential inter-events → count is Poisson(λ T); check mean."""
        key = jax.random.PRNGKey(0)
        rate = 3.0
        T = 5.0
        keys = jax.random.split(key, 2000)
        counts = jax.vmap(
            lambda k: jnp.sum(renewal_sample(k, dist.Exponential(rate), T, 128)[1])
        )(keys)
        assert float(jnp.mean(counts)) == pytest.approx(rate * T, abs=0.3)

    def test_returned_times_are_sorted(self):
        key = jax.random.PRNGKey(0)
        t, m, _ = renewal_sample(key, dist.Exponential(2.0), 10.0, 64)
        real = t[m]
        diffs = jnp.diff(real)
        assert jnp.all(diffs >= 0)


class TestRenewalHazard:
    def test_exponential_hazard_is_constant(self):
        d = dist.Exponential(2.5)
        h1 = renewal_hazard(jnp.asarray(0.5), d)
        h2 = renewal_hazard(jnp.asarray(3.7), d)
        # Hazard of Exponential is exactly λ; float32 subtraction
        # of 1 − cdf near the tail loses a few ULPs, hence the mild tol.
        assert float(h1) == pytest.approx(2.5, rel=1e-3)
        assert float(h2) == pytest.approx(2.5, rel=1e-3)

    def test_weibull_cumulative_hazard_closed_form(self):
        """Λ(τ) = (τ/scale)^k for Weibull(scale, k)."""
        scale, k = 2.0, 1.5
        tau = jnp.asarray(3.0)
        d = dist.Weibull(scale=scale, concentration=k)
        cum = renewal_cumulative_hazard(tau, d)
        expected = (3.0 / scale) ** k
        assert float(cum) == pytest.approx(expected, rel=1e-4)

    def test_survival_in_unit_interval(self):
        d = dist.Gamma(concentration=2.0, rate=1.0)
        for tau in (0.0, 0.5, 1.0, 5.0, 10.0):
            s = float(renewal_survival(jnp.asarray(tau), d))
            assert 0.0 <= s <= 1.0
