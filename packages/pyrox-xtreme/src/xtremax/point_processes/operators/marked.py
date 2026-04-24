"""Equinox operator for separable marked temporal point processes.

A marked TPP pairs a ground process — any
:mod:`~xtremax.point_processes.operators` operator exposing
``log_prob(times, mask)`` / ``sample(key, max_events)`` — with a
user-supplied mark distribution callable
``mark_distribution_fn(t, history) -> Distribution``. Under the
separable factorisation
:math:`\\lambda(t, m) = \\lambda_g(t) \\cdot f(m | t, H)` the joint
log-likelihood decomposes as

.. math::
    \\log L = \\log L_\\text{ground}(\\{t_i\\}) + \\sum_i \\log f(m_i | t_i, H_i),

so most of the work is delegated to the ground operator and the
primitives in :mod:`~xtremax.point_processes.primitives.marked`.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from xtremax.point_processes._history import EventHistory
from xtremax.point_processes.primitives.marked import (
    marks_log_prob,
    sample_marks_at_times,
)


class MarkedTemporalPointProcess(eqx.Module):
    """Separable marked TPP: ground process + time-/history-conditioned marks.

    Args:
        ground: Any temporal point process operator exposing
            ``log_prob(times, mask)`` and
            ``sample(key, max_events) -> (times, mask, n)``.
        mark_distribution_fn: Callable ``(t, history) -> Distribution``.
            History is an :class:`EventHistory` containing events
            *before* ``t`` (empty at the first event). Any NumPyro
            ``Distribution`` works; scalar, multivariate, and discrete
            marks all follow the same protocol.
        mark_dim: Dimensionality of each mark. Use ``None`` for scalar
            marks (mark shape ``()``) and an ``int`` for vector marks
            (mark shape ``(mark_dim,)``).
        history_at_each_event: Pass per-event histories to
            ``mark_distribution_fn``. When ``False`` the full observed
            history is used instead — cheaper but only correct for
            history-independent mark distributions.
    """

    ground: eqx.Module
    mark_distribution_fn: Callable[[Array, EventHistory], dist.Distribution]
    mark_dim: int | None = eqx.field(static=True, default=None)
    history_at_each_event: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        ground: eqx.Module,
        mark_distribution_fn: Callable[[Array, EventHistory], dist.Distribution],
        mark_dim: int | None = None,
        history_at_each_event: bool = True,
    ) -> None:
        self.ground = ground
        self.mark_distribution_fn = mark_distribution_fn
        self.mark_dim = mark_dim
        self.history_at_each_event = history_at_each_event

    @property
    def observation_window(self) -> Float[Array, ...]:
        """Forward to the ground operator."""
        return self.ground.observation_window

    # ------------------------------------------------------------
    # Core distribution API
    # ------------------------------------------------------------

    def log_prob(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
        marks: Float[Array, ...],
    ) -> Float[Array, ...]:
        """Joint log-likelihood: ground + marks.

        The ground-process log-prob signature varies across families —
        Hawkes needs ``(times, mask)``, HPP needs count, etc. This
        wrapper passes ``(times, mask)`` when the ground operator
        accepts that signature and falls back to just the mask-derived
        count for HPP-like families.
        """
        ground_log_prob = self._call_ground_log_prob(event_times, mask)
        mark_log_prob = marks_log_prob(
            event_times,
            marks,
            mask,
            self.mark_distribution_fn,
            history_at_each_event=self.history_at_each_event,
        )
        return ground_log_prob + mark_log_prob

    def _call_ground_log_prob(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Forward to the ground's ``log_prob`` matching its signature."""
        # All IPP/Hawkes/Renewal operators in this package take (times, mask);
        # HPP takes n_events. Try (times, mask) first, fall back to the count.
        try:
            return self.ground.log_prob(event_times, mask)
        except TypeError:
            n = jnp.sum(mask, axis=-1)
            return self.ground.log_prob(n)

    def sample(
        self,
        key: PRNGKeyArray,
        max_events: int,
        **ground_kwargs,
    ) -> tuple[Float[Array, ...], Bool[Array, ...], Float[Array, ...]]:
        """Sample ground event times, then draw marks at each time.

        Returns:
            Tuple ``(times, mask, marks)``. ``marks`` has shape
            ``(max_events,)`` for scalar marks or
            ``(max_events, mark_dim)`` for vector marks; padding
            positions are filled with zeros.
        """
        key_ground, key_marks = jax.random.split(key)
        # Forward any operator-specific sample kwargs (e.g. ``max_candidates``
        # for a thinning-based ground sampler).
        ground_result = self.ground.sample(key_ground, max_events, **ground_kwargs)
        times, mask, _ = ground_result
        marks = sample_marks_at_times(
            key_marks,
            times,
            mask,
            self.mark_distribution_fn,
            history_at_each_event=self.history_at_each_event,
        )
        return times, mask, marks

    # ------------------------------------------------------------
    # Ground-process pass-throughs
    # ------------------------------------------------------------

    def ground_log_prob(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Log-likelihood of the ground events only."""
        return self._call_ground_log_prob(event_times, mask)

    def marks_log_prob(
        self,
        event_times: Float[Array, ...],
        marks: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Log-likelihood of the marks only, conditional on observed times."""
        return marks_log_prob(
            event_times,
            marks,
            mask,
            self.mark_distribution_fn,
            history_at_each_event=self.history_at_each_event,
        )
