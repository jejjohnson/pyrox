"""Tests for :mod:`xtremax.point_processes.primitives.hpp`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import random

from xtremax.point_processes.primitives import (
    hpp_cumulative_intensity,
    hpp_exceedance_log_prob,
    hpp_hazard,
    hpp_intensity,
    hpp_inter_event_log_prob,
    hpp_log_prob,
    hpp_mean_residual_life,
    hpp_predict_count,
    hpp_return_period,
    hpp_sample,
    hpp_survival,
)


RATE = 2.5
T = 4.0


class TestHppLogProb:
    def test_matches_closed_form(self):
        # log L = n log λ - λ T
        n = jnp.asarray(7)
        expected = 7 * jnp.log(RATE) - RATE * T
        assert jnp.allclose(hpp_log_prob(n, RATE, T), expected)

    def test_vmap_over_counts(self):
        ns = jnp.arange(10)
        result = jax.vmap(hpp_log_prob, in_axes=(0, None, None))(ns, RATE, T)
        expected = ns * jnp.log(RATE) - RATE * T
        assert jnp.allclose(result, expected)

    def test_grad_through_rate(self):
        # d(log L)/dλ = n/λ - T
        n = jnp.asarray(5.0)
        grad = jax.grad(lambda r: hpp_log_prob(n, r, T))(RATE)
        expected = 5.0 / RATE - T
        assert jnp.allclose(grad, expected)


class TestHppSample:
    def test_sample_shapes_and_mask_consistency(self):
        key = random.PRNGKey(0)
        times, mask, n = hpp_sample(key, RATE, T, max_events=64)
        assert times.shape == (64,)
        assert mask.shape == (64,)
        assert mask.sum() == jnp.minimum(n, 64)
        # Events sorted and within [0, T].
        valid = times[mask]
        assert jnp.all(jnp.diff(valid) >= -1e-7)
        assert jnp.all((valid >= 0.0) & (valid <= T))

    def test_empirical_count_matches_rate_T(self):
        # Average count across many seeds should be close to λ T.
        keys = random.split(random.PRNGKey(42), 2000)
        counts = jax.vmap(lambda k: hpp_sample(k, RATE, T, max_events=128)[2])(keys)
        mean_count = jnp.mean(counts.astype(jnp.float32))
        # Monte Carlo tolerance: sem ~ sqrt(λT/N) = sqrt(10/2000) ≈ 0.07.
        assert jnp.abs(mean_count - RATE * T) < 0.2

    def test_batched_rate(self):
        key = random.PRNGKey(7)
        rates = jnp.array([1.0, 3.0, 5.0])
        times, _mask, n = hpp_sample(key, rates, T, max_events=64)
        assert times.shape == (3, 64)
        assert n.shape == (3,)

    def test_sample_shape_leads_batch(self):
        key = random.PRNGKey(8)
        times, _mask, n = hpp_sample(key, RATE, T, max_events=32, sample_shape=(5, 2))
        assert times.shape == (5, 2, 32)
        assert n.shape == (5, 2)

    def test_sample_jits(self):
        jitted = jax.jit(
            lambda k: hpp_sample(k, RATE, T, max_events=32),
        )
        key = random.PRNGKey(0)
        times, _mask, _n = jitted(key)
        assert times.shape == (32,)


class TestHppIntensityHazardSurvival:
    def test_intensity_constant(self):
        t = jnp.linspace(0.0, T, 10)
        assert jnp.allclose(hpp_intensity(t, RATE), RATE)

    def test_hazard_constant(self):
        t = jnp.linspace(0.0, T, 10)
        assert jnp.allclose(hpp_hazard(t, RATE), RATE)

    def test_cumulative_intensity_linear(self):
        t = jnp.array([0.0, 1.0, 2.0, T])
        assert jnp.allclose(hpp_cumulative_intensity(t, RATE), RATE * t)

    def test_survival_matches_exp(self):
        t = jnp.array(3.0)
        s = jnp.array(1.0)
        assert jnp.allclose(hpp_survival(t, s, RATE), jnp.exp(-RATE * (t - s)))

    def test_survival_s_equals_t_is_one(self):
        t = jnp.asarray(2.0)
        assert jnp.allclose(hpp_survival(t, t, RATE), 1.0)


class TestHppInterEvent:
    def test_inter_event_log_prob_matches_exponential(self):
        tau = jnp.array([0.0, 0.5, 1.0, 2.0])
        expected = jnp.log(RATE) - RATE * tau
        assert jnp.allclose(hpp_inter_event_log_prob(tau, RATE), expected)

    def test_inter_event_negative_tau_is_neg_inf(self):
        tau = jnp.asarray(-0.5)
        assert jnp.isneginf(hpp_inter_event_log_prob(tau, RATE))


class TestHppPredictions:
    def test_mean_residual_life(self):
        assert jnp.allclose(hpp_mean_residual_life(RATE), 1.0 / RATE)
        # Given time is ignored for HPP.
        assert jnp.allclose(hpp_mean_residual_life(RATE, given_time=7.3), 1.0 / RATE)

    def test_return_period_inverse(self):
        # τ = -log(1-p)/λ
        p = 0.9
        tau = hpp_return_period(p, RATE)
        # Probability of at least one event in τ under HPP is 1 - exp(-λτ) = p.
        assert jnp.allclose(1.0 - jnp.exp(-RATE * tau), p)

    def test_predict_count_linear(self):
        count = hpp_predict_count(1.0, 4.0, RATE)
        assert jnp.allclose(count, RATE * 3.0)

    def test_exceedance_log_prob_zero_rate_window(self):
        # No events expected ⇒ P(N > 0) = 0.
        log_p = hpp_exceedance_log_prob(0, 1.0, 1.0, RATE)
        assert jnp.isneginf(log_p) or jnp.allclose(jnp.exp(log_p), 0.0, atol=1e-6)


def test_sample_truncation_cap_is_exact():
    # With λT very small and small max_events, cap should almost never bite.
    key = random.PRNGKey(0)
    _times, mask, n = hpp_sample(key, rate=0.1, T=1.0, max_events=64)
    assert mask.sum() == jnp.minimum(n, 64)


@pytest.mark.parametrize("rate", [0.5, 1.0, 10.0])
def test_empirical_intensity_matches_parameter(rate):
    key = random.PRNGKey(2024)
    keys = random.split(key, 5000)
    counts = jax.vmap(lambda k: hpp_sample(k, rate, 2.0, max_events=64)[2])(keys)
    mean_rate = jnp.mean(counts.astype(jnp.float32)) / 2.0
    assert jnp.abs(mean_rate - rate) < 0.15 * rate + 0.05


def test_hpp_intensity_broadcasts_scalar_t_with_batched_rate():
    """Regression (PR #10 review): scalar t + batched rate must broadcast."""
    from xtremax.point_processes.primitives import hpp_intensity

    rates = jnp.array([1.0, 2.0, 3.0])
    out = hpp_intensity(jnp.asarray(0.5), rates)
    assert out.shape == rates.shape
    assert jnp.allclose(out, rates)

    # Reverse direction: batched t, scalar rate.
    t = jnp.array([0.0, 1.0, 2.0, 3.0])
    out2 = hpp_intensity(t, jnp.asarray(4.0))
    assert out2.shape == t.shape
    assert jnp.all(out2 == 4.0)
