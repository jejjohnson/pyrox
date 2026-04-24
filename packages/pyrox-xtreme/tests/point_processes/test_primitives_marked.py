"""Tests for marked-process primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from xtremax.point_processes import constant_mark_distribution
from xtremax.point_processes.primitives.marked import (
    marks_log_prob,
    sample_marks_at_times,
)


class TestMarksLogProb:
    def test_constant_gaussian_marks_match_sum_of_densities(self):
        """Σ log N(m; 0, 1) with independent marks."""
        event_times = jnp.array([0.1, 0.5, 1.0, 2.0])
        mask = jnp.array([True, True, True, False])
        marks = jnp.array([0.3, -0.7, 1.2, 0.0])
        fn = constant_mark_distribution(dist.Normal(0.0, 1.0))
        ll = float(
            marks_log_prob(event_times, marks, mask, fn, history_at_each_event=False)
        )
        expected = float(jnp.sum(dist.Normal(0.0, 1.0).log_prob(marks[mask])))
        assert ll == pytest.approx(expected, rel=1e-4)

    def test_per_event_history_matches_full_history_for_time_only_marks(self):
        """When marks depend only on time, both modes must agree."""
        event_times = jnp.array([0.1, 0.5, 1.0, 2.0])
        mask = jnp.array([True, True, True, False])
        marks = jnp.array([0.3, -0.7, 1.2, 0.0])

        def marks_fn(t, history):
            return dist.Normal(t, 1.0)

        ll_full = float(
            marks_log_prob(
                event_times, marks, mask, marks_fn, history_at_each_event=False
            )
        )
        ll_perevent = float(
            marks_log_prob(
                event_times, marks, mask, marks_fn, history_at_each_event=True
            )
        )
        assert ll_full == pytest.approx(ll_perevent, rel=1e-4)


class TestSampleMarksAtTimes:
    def test_shapes_are_correct(self):
        event_times = jnp.array([0.1, 0.5, 1.0, 2.0])
        mask = jnp.array([True, True, True, False])
        fn = constant_mark_distribution(dist.Normal(0.0, 1.0))
        marks = sample_marks_at_times(jax.random.PRNGKey(0), event_times, mask, fn)
        assert marks.shape == event_times.shape

    def test_padding_marks_are_zeroed(self):
        event_times = jnp.array([0.1, 0.5, 1.0, 2.0])
        mask = jnp.array([True, True, True, False])
        fn = constant_mark_distribution(dist.Normal(10.0, 1.0))
        marks = sample_marks_at_times(
            jax.random.PRNGKey(0), event_times, mask, fn, history_at_each_event=False
        )
        assert float(marks[3]) == 0.0
