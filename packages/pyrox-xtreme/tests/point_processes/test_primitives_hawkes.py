"""Tests for Hawkes primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtremax.point_processes.primitives.hawkes import (
    exp_hawkes_cumulative_intensity,
    exp_hawkes_intensity,
    exp_hawkes_log_prob,
    exp_hawkes_sample,
    general_hawkes_intensity,
    general_hawkes_log_prob,
)


class TestExpHawkesIntensity:
    def test_matches_closed_form(self):
        """λ*(t) = μ + α Σ_{t_i<t} exp(−β(t−t_i))."""
        event_times = jnp.array([1.0, 2.0, 3.0])
        mask = jnp.array([True, True, True])
        mu, alpha, beta = 0.5, 0.3, 1.0
        t = 3.5
        lam = float(
            exp_hawkes_intensity(jnp.asarray(t), event_times, mask, mu, alpha, beta)
        )
        expected = mu + alpha * (
            jnp.exp(-beta * 2.5) + jnp.exp(-beta * 1.5) + jnp.exp(-beta * 0.5)
        )
        assert lam == pytest.approx(float(expected), rel=1e-5)

    def test_matches_general_hawkes_intensity(self):
        """General kernel with φ(τ) = α·exp(−β·τ) must match the exp form."""
        event_times = jnp.array([0.5, 1.2, 2.8, 3.5])
        mask = jnp.array([True, True, True, True])
        mu, alpha, beta = 0.7, 0.4, 1.5
        t = jnp.asarray(4.0)

        def kernel_fn(dt):
            return alpha * jnp.exp(-beta * jnp.clip(dt, 0.0, jnp.inf))

        lam_exp = exp_hawkes_intensity(t, event_times, mask, mu, alpha, beta)
        lam_gen = general_hawkes_intensity(t, event_times, mask, mu, kernel_fn)
        assert float(abs(lam_exp - lam_gen)) < 1e-5


class TestExpHawkesCompensator:
    def test_closed_form_matches_manual_sum(self):
        event_times = jnp.array([1.0, 2.0, 3.0])
        mask = jnp.array([True, True, True])
        mu, alpha, beta = 0.5, 0.3, 1.0
        T = 5.0
        Lambda = float(
            exp_hawkes_cumulative_intensity(
                jnp.asarray(T), event_times, mask, mu, alpha, beta
            )
        )
        expected = mu * T + (alpha / beta) * (
            (1 - jnp.exp(-beta * 4))
            + (1 - jnp.exp(-beta * 3))
            + (1 - jnp.exp(-beta * 2))
        )
        assert Lambda == pytest.approx(float(expected), rel=1e-5)


class TestExpHawkesLogProb:
    def test_alpha_zero_matches_hpp(self):
        """For α=0 the Hawkes process degenerates to an HPP with rate μ.

        Critical invariant — if the Ozaki recursion leaks the `1` self-term
        this test fails because log L then picks up spurious events.
        """
        from xtremax.point_processes.primitives.hpp import hpp_log_prob

        event_times = jnp.array([0.5, 1.2, 2.7, 3.9])
        mask = jnp.array([True, True, True, True])
        T = 5.0
        mu = 0.6

        hawkes_ll = exp_hawkes_log_prob(event_times, mask, T, mu, 0.0, 1.0)
        hpp_ll = hpp_log_prob(jnp.sum(mask), mu, T)
        assert float(hawkes_ll) == pytest.approx(float(hpp_ll), rel=1e-5)

    def test_matches_general_hawkes_log_prob(self):
        event_times = jnp.array([0.3, 1.1, 2.0, 2.8])
        mask = jnp.array([True, True, True, True])
        T = 4.0
        mu, alpha, beta = 0.5, 0.3, 1.0

        def kernel_fn(dt):
            return alpha * jnp.exp(-beta * jnp.clip(dt, 0.0, jnp.inf))

        def kernel_integral_fn(a, b):
            a_c = jnp.clip(a, 0.0, jnp.inf)
            b_c = jnp.clip(b, 0.0, jnp.inf)
            return (alpha / beta) * (jnp.exp(-beta * a_c) - jnp.exp(-beta * b_c))

        ll_exp = float(exp_hawkes_log_prob(event_times, mask, T, mu, alpha, beta))
        ll_gen = float(
            general_hawkes_log_prob(
                event_times, mask, T, mu, kernel_fn, kernel_integral_fn
            )
        )
        assert ll_exp == pytest.approx(ll_gen, rel=1e-4)

    def test_log_prob_grad_flows(self):
        """Gradients through μ, α, β are finite and non-zero."""
        event_times = jnp.array([0.3, 1.1, 2.0, 2.8])
        mask = jnp.array([True, True, True, True])
        T = 4.0

        def loss(params):
            mu, alpha, beta = params
            return -exp_hawkes_log_prob(event_times, mask, T, mu, alpha, beta)

        g = jax.grad(loss)(jnp.array([0.5, 0.3, 1.0]))
        assert jnp.all(jnp.isfinite(g))
        assert jnp.any(jnp.abs(g) > 1e-6)

    def test_padding_positions_ignored(self):
        """Extending with padding must not change the log-prob."""
        real_times = jnp.array([0.3, 1.1, 2.0])
        mask_short = jnp.array([True, True, True])
        mask_long = jnp.array([True, True, True, False, False])
        padded = jnp.array([0.3, 1.1, 2.0, 4.0, 4.0])

        ll1 = exp_hawkes_log_prob(real_times, mask_short, 4.0, 0.5, 0.3, 1.0)
        ll2 = exp_hawkes_log_prob(padded, mask_long, 4.0, 0.5, 0.3, 1.0)
        assert float(ll1) == pytest.approx(float(ll2), rel=1e-5)


class TestExpHawkesSample:
    def test_expected_count_matches_closed_form(self):
        """Stationary expected count: μT/(1−α/β)."""
        key = jax.random.PRNGKey(0)
        mu, alpha, beta = 0.5, 0.3, 1.0
        T = 50.0
        keys = jax.random.split(key, 200)
        counts = jax.vmap(
            lambda k: jnp.sum(exp_hawkes_sample(k, T, mu, alpha, beta, 1024)[1])
        )(keys)
        expected = mu * T / (1 - alpha / beta)
        assert float(jnp.mean(counts)) == pytest.approx(expected, rel=0.2)

    def test_sampled_times_are_sorted_and_in_window(self):
        key = jax.random.PRNGKey(0)
        t, m, _ = exp_hawkes_sample(key, jnp.asarray(10.0), 0.5, 0.3, 1.0, 128)
        real = t[m]
        assert jnp.all(jnp.diff(real) >= 0)
        assert jnp.all(real <= 10.0)
