"""Padded event-history PyTree shared by all TPP families.

The thinning, Hawkes, renewal, and marked operators all need a common
representation of "events seen so far" so that user-supplied retention
and mark callables can inspect history without caring which family
generated it. Keeping everything in a fixed-size buffer with a boolean
mask is what lets the surrounding scan / while loop stay ``jax.jit`` /
``jax.vmap`` friendly.

:class:`EventHistory` is an ``equinox.Module`` — PyTree leaves only,
no static Python scalars — so sequences flow through ``scan`` carries
without re-tracing. Helper methods (:meth:`n_events`,
:meth:`last_time`, :meth:`time_since_last`) cover the needs of most
user callables so they rarely have to manipulate the buffer directly.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


class EventHistory(eqx.Module):
    """Fixed-size buffer of events + optional marks.

    Attributes:
        times: Sorted event times with padding positions (typically) at
            the right edge of the observation window.
        mask: ``True`` where ``times`` is a real event, ``False`` at
            padding positions.
        marks: Optional marks aligned with ``times``. ``None`` for
            unmarked processes. For marked processes this is a 2-D
            ``(max_events, mark_dim)`` array with padding positions
            filled with zeros.
    """

    times: Float[Array, ...]
    mask: Bool[Array, ...]
    marks: Float[Array, ...] | None = None

    @classmethod
    def empty(
        cls,
        max_events: int,
        mark_dim: int | None = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> EventHistory:
        """Buffer with no events yet: all times ``0``, mask all ``False``."""
        times = jnp.zeros(max_events, dtype=dtype)
        mask = jnp.zeros(max_events, dtype=jnp.bool_)
        marks = (
            None if mark_dim is None else jnp.zeros((max_events, mark_dim), dtype=dtype)
        )
        return cls(times=times, mask=mask, marks=marks)

    def n_events(self) -> Int[Array, ...]:
        """Number of real events currently in the buffer."""
        return jnp.sum(self.mask).astype(jnp.int32)

    def last_time(
        self,
        default: float | Float[Array, ...] = 0.0,
    ) -> Float[Array, ...]:
        """Most recent real event time, or ``default`` if the history is empty.

        Uses the mask directly rather than indexing by count, so it is
        safe under ``jit`` without triggering a dynamic gather.
        """
        default_arr = jnp.asarray(default, dtype=self.times.dtype)
        safe_times = jnp.where(self.mask, self.times, -jnp.inf)
        last = jnp.max(safe_times)
        return jnp.where(jnp.any(self.mask), last, default_arr)

    def time_since_last(
        self,
        t: Float[Array, ...],
        default: float | Float[Array, ...] = 0.0,
    ) -> Float[Array, ...]:
        """``t - last_time`` with ``default`` returned for empty history."""
        return jnp.asarray(t) - self.last_time(default=default)

    def append(
        self,
        t: Float[Array, ...],
        mark: Float[Array, ...] | None = None,
        *,
        accepted: Bool[Array, ...] | bool = True,
    ) -> EventHistory:
        """Return a new history with ``t`` appended if ``accepted``.

        Writes at position ``n_events()``; a ``False`` ``accepted`` or an
        overrun past the buffer is a no-op so the routine stays shape-
        and jit-stable. Callers that care about overruns can compare
        ``n_events()`` before/after.
        """
        accepted_arr = jnp.asarray(accepted, dtype=jnp.bool_)
        max_events = self.times.shape[-1]
        idx = self.n_events()
        can_write = accepted_arr & (idx < max_events)
        safe_idx = jnp.clip(idx, 0, max_events - 1)

        current_time = self.times[safe_idx]
        current_mask = self.mask[safe_idx]
        new_time_at_idx = jnp.where(can_write, t, current_time)
        new_mask_at_idx = jnp.where(can_write, jnp.bool_(True), current_mask)
        new_times = self.times.at[safe_idx].set(new_time_at_idx)
        new_mask = self.mask.at[safe_idx].set(new_mask_at_idx)

        if self.marks is None:
            return EventHistory(times=new_times, mask=new_mask, marks=None)

        if mark is None:
            raise ValueError(
                "EventHistory has a marks buffer but append() was called "
                "without a mark; supply mark=<value> or use an unmarked history."
            )
        current_mark_row = self.marks[safe_idx]
        new_mark_row = jnp.where(can_write, jnp.asarray(mark), current_mark_row)
        new_marks = self.marks.at[safe_idx].set(new_mark_row)
        return EventHistory(times=new_times, mask=new_mask, marks=new_marks)
