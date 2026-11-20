"""NumPyro ``Distribution`` wrappers for Hawkes temporal point processes."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, PRNGKeyArray
from numpyro.distributions import constraints

from xtremax._rng import check_prng_key
from xtremax.point_processes.operators.hawkes import (
    ExponentialHawkes as _ExpOp,
    GeneralHawkesProcess as _GeneralOp,
)


class ExponentialHawkes(dist.Distribution):
    """NumPyro wrapper for an exponential-kernel Hawkes process.

    Args:
        mu: Baseline rate.
        alpha: Excitation amplitude.
        beta: Excitation decay.
        observation_window: Window length.
        max_events: Static buffer for sampling.
    """

    arg_constraints = {
        "mu": constraints.positive,
        "alpha": constraints.nonnegative,
        "beta": constraints.positive,
        "observation_window": constraints.positive,
    }
    support = constraints.dependent
    reparametrized_params: list[str] = []

    def __init__(
        self,
        mu: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        observation_window: ArrayLike,
        max_events: int = 1024,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.mu = jnp.asarray(mu)
        self.alpha = jnp.asarray(alpha)
        self.beta = jnp.asarray(beta)
        self.observation_window = jnp.asarray(observation_window)
        self._max_events = int(max_events)
        self._op = _ExpOp(self.mu, self.alpha, self.beta, self.observation_window)
        batch_shape = jnp.broadcast_shapes(
            self.mu.shape,
            self.alpha.shape,
            self.beta.shape,
            self.observation_window.shape,
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
                "ExponentialHawkes distribution supports sample_shape=() only; "
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


class GeneralHawkesProcess(dist.Distribution):
    """NumPyro wrapper for a general-kernel Hawkes process.

    See :class:`~xtremax.point_processes.operators.GeneralHawkesProcess`
    for the kernel protocol and sampling details.
    """

    arg_constraints = {
        "mu": constraints.positive,
        "observation_window": constraints.positive,
    }
    support = constraints.dependent
    reparametrized_params: list[str] = []

    def __init__(
        self,
        mu: ArrayLike,
        kernel: eqx.Module,
        observation_window: ArrayLike,
        max_events: int = 1024,
        n_integration_points: int = 100,
        max_kernel_value: ArrayLike | None = None,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.mu = jnp.asarray(mu)
        self.observation_window = jnp.asarray(observation_window)
        self._max_events = int(max_events)
        self._op = _GeneralOp(
            self.mu,
            kernel,
            self.observation_window,
            n_integration_points=n_integration_points,
            max_kernel_value=max_kernel_value,
        )
        batch_shape = jnp.broadcast_shapes(self.mu.shape, self.observation_window.shape)
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
                "GeneralHawkesProcess distribution supports sample_shape=() only."
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
