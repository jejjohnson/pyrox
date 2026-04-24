"""Tests for the generic thinning primitive."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtremax.point_processes import EventHistory
from xtremax.point_processes.primitives.thinning import (
    retention_compensator,
    thinning_retention_log_prob,
    thinning_sample,
)


class TestThinningSample:
    def test_reproduces_hpp_for_constant_intensity(self):
        """Constant λ: the generic thinning sampler must agree with HPP.

        Using λ*(t) = c and λ_max = c, the sampler reduces to an HPP
        with rate c. The mean count should therefore be c·T.
        """

        def intensity(t, history):
            return jnp.asarray(3.0)

        def lambda_max(history):
            return jnp.asarray(3.0)

        keys = jax.random.split(jax.random.PRNGKey(0), 500)
        counts = jax.vmap(
            lambda k: jnp.sum(thinning_sample(k, intensity, lambda_max, 5.0, 64)[1])
        )(keys)
        # HPP(rate=3, T=5) → expected count 15.
        assert float(jnp.mean(counts)) == pytest.approx(15.0, abs=0.5)

    def test_half_retention_halves_expected_count(self):
        """λ_max=c, λ=c/2 → acceptance 0.5 → HPP(c/2)."""

        def intensity(t, history):
            return jnp.asarray(1.5)

        def lambda_max(history):
            return jnp.asarray(3.0)

        keys = jax.random.split(jax.random.PRNGKey(1), 500)
        counts = jax.vmap(
            lambda k: jnp.sum(thinning_sample(k, intensity, lambda_max, 10.0, 128)[1])
        )(keys)
        assert float(jnp.mean(counts)) == pytest.approx(15.0, abs=0.8)

    def test_sampled_times_in_window(self):
        def intensity(t, history):
            return jnp.asarray(1.0)

        def lambda_max(history):
            return jnp.asarray(1.0)

        t, m, _ = thinning_sample(
            jax.random.PRNGKey(42), intensity, lambda_max, 10.0, 128
        )
        real = t[m]
        assert jnp.all(real >= 0.0)
        assert jnp.all(real <= 10.0)
        assert jnp.all(jnp.diff(real) >= 0)


class TestRetentionCompensator:
    def test_constant_retention_gives_expected_integral(self):
        """∫(1-p)λ dt = (1-p)·rate·T for constants."""

        def retention(t, history, mark=None):
            return jnp.asarray(0.3)

        def intensity(t, history):
            return jnp.asarray(2.0)

        history = EventHistory.empty(max_events=4)
        val = float(
            retention_compensator(retention, intensity, history, 5.0, n_points=100)
        )
        expected = (1 - 0.3) * 2.0 * 5.0
        assert val == pytest.approx(expected, rel=1e-3)


class TestThinningRetentionLogProb:
    def test_constant_p_gives_n_log_p(self):
        """Σ log p = n log p for constant retention."""

        def retention(t, history, mark=None):
            return jnp.asarray(0.4)

        event_times = jnp.array([0.1, 0.5, 1.0, 2.0])
        mask = jnp.array([True, True, True, False])
        history = EventHistory(times=event_times, mask=mask, marks=None)
        val = float(thinning_retention_log_prob(retention, event_times, mask, history))
        expected = 3 * float(jnp.log(0.4))
        assert val == pytest.approx(expected, rel=1e-4)
