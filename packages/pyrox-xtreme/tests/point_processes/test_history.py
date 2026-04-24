"""Tests for :class:`EventHistory` — the shared PyTree used by all families."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtremax.point_processes import EventHistory


class TestEventHistory:
    def test_empty_has_no_events(self):
        h = EventHistory.empty(max_events=8)
        assert int(h.n_events()) == 0
        assert jnp.allclose(h.last_time(default=-1.0), -1.0)

    def test_append_sets_mask_and_time(self):
        h = EventHistory.empty(max_events=4)
        h1 = h.append(jnp.asarray(0.5))
        assert int(h1.n_events()) == 1
        assert bool(h1.mask[0])
        assert float(h1.times[0]) == pytest.approx(0.5)

    def test_append_with_accepted_false_is_noop(self):
        h = EventHistory.empty(max_events=4)
        h1 = h.append(jnp.asarray(0.5), accepted=False)
        assert int(h1.n_events()) == 0
        assert not any(h1.mask)

    def test_append_buffer_overflow_is_noop(self):
        h = EventHistory.empty(max_events=2)
        h1 = h.append(jnp.asarray(0.1)).append(jnp.asarray(0.3))
        h2 = h1.append(jnp.asarray(0.8))  # buffer full
        assert int(h2.n_events()) == 2
        assert float(h2.times[0]) == pytest.approx(0.1)
        assert float(h2.times[1]) == pytest.approx(0.3)

    def test_time_since_last(self):
        h = EventHistory.empty(max_events=4)
        h1 = h.append(jnp.asarray(1.0)).append(jnp.asarray(2.5))
        assert float(h1.time_since_last(jnp.asarray(3.0))) == pytest.approx(0.5)

    def test_append_with_marks(self):
        h = EventHistory.empty(max_events=3, mark_dim=2)
        h1 = h.append(jnp.asarray(0.5), mark=jnp.asarray([1.0, 2.0]))
        assert float(h1.marks[0, 0]) == pytest.approx(1.0)
        assert float(h1.marks[0, 1]) == pytest.approx(2.0)

    def test_append_missing_mark_raises(self):
        h = EventHistory.empty(max_events=3, mark_dim=2)
        with pytest.raises(ValueError):
            h.append(jnp.asarray(0.5))  # mark required

    def test_jit_compatible(self):
        h = EventHistory.empty(max_events=4)

        @jax.jit
        def two_events(h):
            return h.append(jnp.asarray(0.5)).append(jnp.asarray(1.5))

        h2 = two_events(h)
        assert int(h2.n_events()) == 2
