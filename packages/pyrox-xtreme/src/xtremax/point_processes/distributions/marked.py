"""NumPyro ``Distribution`` wrapper for a separable marked TPP."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Bool, Float, PRNGKeyArray
from numpyro.distributions import constraints

from xtremax._rng import check_prng_key
from xtremax.point_processes._history import EventHistory
from xtremax.point_processes.operators.marked import (
    MarkedTemporalPointProcess as _MarkedOp,
)


class MarkedTemporalPointProcess(dist.Distribution):
    """NumPyro wrapper for a separable marked temporal point process.

    Samples and log-probs consume / return the triple
    ``(times, mask, marks)``.

    Args:
        ground: Ground-process operator.
        mark_distribution_fn: ``(t, history) -> Distribution`` callable.
        mark_dim: Dimensionality of each mark (``None`` for scalar).
        history_at_each_event: Whether marks see per-event histories.
        max_events: Static buffer size.
    """

    arg_constraints: dict = {}
    support = constraints.dependent
    reparametrized_params: list[str] = []

    def __init__(
        self,
        ground: eqx.Module,
        mark_distribution_fn: Callable[[Array, EventHistory], dist.Distribution],
        mark_dim: int | None = None,
        history_at_each_event: bool = True,
        max_events: int = 1024,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self._op = _MarkedOp(
            ground,
            mark_distribution_fn,
            mark_dim=mark_dim,
            history_at_each_event=history_at_each_event,
        )
        self.observation_window = jnp.asarray(self._op.observation_window)
        self._max_events = int(max_events)
        super().__init__(
            batch_shape=self.observation_window.shape, validate_args=validate_args
        )

    @property
    def max_events(self) -> int:
        return self._max_events

    def sample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[Float[Array, ...], Bool[Array, ...], Float[Array, ...]]:
        check_prng_key(key)
        if sample_shape != ():
            raise NotImplementedError(
                "MarkedTemporalPointProcess supports sample_shape=() only."
            )
        return self._op.sample(key, self._max_events)

    def log_prob(
        self,
        value: tuple[Float[Array, ...], Bool[Array, ...], Float[Array, ...]],
    ) -> Float[Array, ...]:
        event_times, mask, marks = value
        return self._op.log_prob(event_times, mask, marks)

    def _validate_sample(self, value) -> None:
        return None
