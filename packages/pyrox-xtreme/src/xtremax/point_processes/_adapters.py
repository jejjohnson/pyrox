"""Ergonomic adapters wrapping plain values as retention / mark callables.

Users rarely need the full ``(t, history, proposed_mark) -> Array``
signature. These helpers build the common cases (constant probability,
time-varying function, pre-built NumPyro distribution) so that calling
code stays readable while the downstream operators only see the full
protocol.

The canonical path is still "write your own callable"; these are
three-line conveniences that save typing on the boring cases. They are
deliberately tiny — not abstractions, just names.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float

from xtremax.point_processes._history import EventHistory


def constant_retention(
    p: float | Float[Array, ...],
) -> Callable[[Float[Array, ...], EventHistory, Float[Array, ...] | None], Array]:
    """Retention that always returns the same probability ``p``."""
    p_arr = jnp.asarray(p)

    def _fn(
        t: Float[Array, ...],
        history: EventHistory,
        proposed_mark: Float[Array, ...] | None = None,
    ) -> Float[Array, ...]:
        del t, history, proposed_mark
        return p_arr

    return _fn


def time_varying_retention(
    fn: Callable[[Float[Array, ...]], Float[Array, ...]],
) -> Callable[[Float[Array, ...], EventHistory, Float[Array, ...] | None], Array]:
    """Wrap a pure function of time into the full retention signature."""

    def _fn(
        t: Float[Array, ...],
        history: EventHistory,
        proposed_mark: Float[Array, ...] | None = None,
    ) -> Float[Array, ...]:
        del history, proposed_mark
        return fn(t)

    return _fn


def constant_mark_distribution(
    d: dist.Distribution,
) -> Callable[[Float[Array, ...], EventHistory], dist.Distribution]:
    """Mark distribution that ignores time and history and returns ``d``."""

    def _fn(
        t: Float[Array, ...],
        history: EventHistory,
    ) -> dist.Distribution:
        del t, history
        return d

    return _fn


def time_varying_marks(
    fn: Callable[[Float[Array, ...]], dist.Distribution],
) -> Callable[[Float[Array, ...], EventHistory], dist.Distribution]:
    """Wrap a time → Distribution callable into the full signature."""

    def _fn(
        t: Float[Array, ...],
        history: EventHistory,
    ) -> dist.Distribution:
        del history
        return fn(t)

    return _fn
