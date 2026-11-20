"""NumPyro ``Distribution`` wrapper for a renewal temporal point process."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, PRNGKeyArray
from numpyro.distributions import constraints

from xtremax._rng import check_prng_key
from xtremax.point_processes.operators.renewal import RenewalProcess as _RenewalOp


class RenewalProcess(dist.Distribution):
    """NumPyro wrapper for a renewal temporal point process.

    Args:
        inter_event_dist: NumPyro Distribution for inter-event times.
        observation_window: Window length ``T > 0``.
        max_events: Static buffer size for ``sample``.
        validate_args: Forwarded to the ``Distribution`` base.
    """

    arg_constraints = {"observation_window": constraints.positive}
    support = constraints.dependent
    reparametrized_params: list[str] = []

    def __init__(
        self,
        inter_event_dist: dist.Distribution,
        observation_window: ArrayLike,
        max_events: int = 1024,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.observation_window = jnp.asarray(observation_window)
        self._max_events = int(max_events)
        self._op = _RenewalOp(inter_event_dist, self.observation_window)
        # A batched inter-event distribution (e.g. vmapped-over rates)
        # adds batch dims to the per-sequence log-likelihood; the
        # window length can also be batched independently. Broadcast
        # both into the NumPyro batch_shape so plated models don't see
        # a phantom "scalar" shape that disagrees with the actual
        # output shape of ``log_prob``.
        batch_shape = jnp.broadcast_shapes(
            self.observation_window.shape, inter_event_dist.batch_shape
        )
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @property
    def max_events(self) -> int:
        return self._max_events

    def sample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[Float[Array, ...], Bool[Array, ...]]:
        check_prng_key(key)
        if sample_shape != ():
            raise NotImplementedError(
                "RenewalProcess distribution supports sample_shape=() only; "
                "use vmap over PRNG keys at the operator layer for batches."
            )
        times, mask, _ = self._op.sample(key, self._max_events)
        return times, mask

    def log_prob(
        self,
        value: tuple[Float[Array, ...], Bool[Array, ...]],
    ) -> Float[Array, ...]:
        event_times, mask = value
        return self._op.log_prob(event_times, mask)

    def _validate_sample(self, value) -> None:
        return None
