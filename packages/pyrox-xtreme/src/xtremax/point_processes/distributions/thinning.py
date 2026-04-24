"""NumPyro ``Distribution`` wrapper for a thinned temporal point process."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, PRNGKeyArray
from numpyro.distributions import constraints

from xtremax._rng import check_prng_key
from xtremax.point_processes.operators.thinning import ThinningProcess as _ThinningOp


class ThinningProcess(dist.Distribution):
    """NumPyro wrapper for a thinned TPP.

    Args:
        base: Base temporal operator.
        retention_fn: ``(t, history, proposed_mark=None) -> p`` callable.
        observation_window: Window length. Defaults to the base's.
        max_events: Static buffer size.
        n_integration_points: Trapezoid grid size for the retention
            compensator.
    """

    arg_constraints: dict = {}
    support = constraints.dependent
    reparametrized_params: list[str] = []

    def __init__(
        self,
        base: eqx.Module,
        retention_fn: Callable[..., Array],
        observation_window: ArrayLike | None = None,
        max_events: int = 1024,
        n_integration_points: int = 100,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self._op = _ThinningOp(
            base,
            retention_fn,
            observation_window=observation_window,
            n_integration_points=n_integration_points,
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
    ) -> (
        tuple[Float[Array, ...], Bool[Array, ...]]
        | tuple[Float[Array, ...], Bool[Array, ...], Float[Array, ...]]
    ):
        """Return ``(times, mask)`` for an unmarked base, or
        ``(times, mask, marks)`` when the base is a marked process.

        The triple matches what :meth:`log_prob` accepts so that
        ``d.log_prob(d.sample(key))`` round-trips for both cases.
        """
        check_prng_key(key)
        if sample_shape != ():
            raise NotImplementedError(
                "ThinningProcess distribution supports sample_shape=() only."
            )
        times, mask, third = self._op.sample(key, self._max_events)
        # Marked base: ``third`` is a mark array. Otherwise a scalar
        # integer count that we drop (users who want the count can
        # read ``mask.sum()``).
        third_arr = jnp.asarray(third)
        if third_arr.ndim == 0 and jnp.issubdtype(third_arr.dtype, jnp.integer):
            return times, mask
        return times, mask, third

    def log_prob(
        self,
        value: tuple[Float[Array, ...], Bool[Array, ...]]
        | tuple[Float[Array, ...], Bool[Array, ...], Float[Array, ...]],
    ) -> Float[Array, ...]:
        if len(value) == 3:
            event_times, mask, marks = value
            return self._op.log_prob(event_times, mask, marks=marks)
        event_times, mask = value
        return self._op.log_prob(event_times, mask)

    def _validate_sample(self, value) -> None:
        return None
