"""NumPyro ``Distribution`` wrappers around the temporal PP operators.

These are thin shims: every meaningful computation lives in the
operator layer (``xtremax.point_processes.operators.temporal``). The
wrappers exist so these processes can appear inside a NumPyro model as
first-class priors / likelihoods without the user having to touch
``factor`` sites by hand.

Shape conventions:
    * ``batch_shape`` matches the broadcast shape of the scalar
      parameters (``rate``, ``observation_window``).
    * ``event_shape`` is empty — samples are variable-length
      sequences, and we deliberately return them as a ``(times, mask)``
      PyTree. NumPyro's validate_sample machinery does not support
      variable-length events, so :meth:`_validate_sample` is a no-op.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from numpyro.distributions import constraints

from xtremax._rng import check_prng_key
from xtremax.point_processes.operators.temporal import (
    HomogeneousPoissonProcess as _HppOp,
    InhomogeneousPoissonProcess as _IppOp,
)


class HomogeneousPoissonProcess(dist.Distribution):
    """NumPyro wrapper for a homogeneous temporal Poisson process.

    Args:
        rate: Intensity ``λ > 0``.
        observation_window: Window length ``T > 0``.
        max_events: Static buffer size for ``sample``.
        validate_args: Forwarded to ``numpyro.distributions.Distribution``.
    """

    arg_constraints = {
        "rate": constraints.positive,
        "observation_window": constraints.positive,
    }
    # Samples are a ``(times, mask)`` PyTree rather than a vector in ℝⁿ,
    # so ``dependent`` is the correct constraint to register with NumPyro.
    support = constraints.dependent
    reparametrized_params: list[str] = []

    def __init__(
        self,
        rate: ArrayLike,
        observation_window: ArrayLike,
        max_events: int = 512,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.rate = jnp.asarray(rate)
        self.observation_window = jnp.asarray(observation_window)
        self._max_events = int(max_events)
        self._op = _HppOp(self.rate, self.observation_window)
        batch_shape = jnp.broadcast_shapes(
            self.rate.shape, self.observation_window.shape
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
        """Return a ``(times, mask)`` PyTree.

        This matches the format :meth:`log_prob` consumes, so
        ``dist.log_prob(dist.sample(key))`` is a valid round-trip.
        If you need the uncapped Poisson count, call the operator
        layer directly or use ``mask.sum(axis=-1)``.
        """
        check_prng_key(key)
        times, mask, _ = self._op.sample(key, self._max_events, sample_shape)
        return times, mask

    def log_prob(
        self,
        value: tuple[Float[Array, ...], Bool[Array, ...]]
        | tuple[Float[Array, ...], Int[Array, ...]],
    ) -> Float[Array, ...]:
        """Log-likelihood. ``value`` may be ``(times, mask)`` or
        ``(times, n_events)``.

        The event *times* of an HPP drop out of the likelihood, so only
        the count is used.
        """
        _, counts_or_mask = value
        counts_or_mask = jnp.asarray(counts_or_mask)
        if counts_or_mask.dtype == jnp.bool_:
            n_events = jnp.sum(counts_or_mask, axis=-1)
        else:
            n_events = counts_or_mask
        return self._op.log_prob(n_events)

    def _validate_sample(self, value) -> None:
        # Variable-length events — skip NumPyro's per-element validation.
        return None


class InhomogeneousPoissonProcess(dist.Distribution):
    """NumPyro wrapper for an inhomogeneous temporal Poisson process.

    Args:
        log_intensity_fn: Callable returning ``log λ(t)``.
        observation_window: Window length.
        integrated_intensity: :math:`\\Lambda(T)`. If ``None``, computed
            by quadrature at every ``log_prob`` call.
        lambda_max: Upper bound on :math:`\\lambda(t)`. Required for
            ``sample``.
        max_candidates: Static buffer for thinning.
        n_integration_points: Quadrature nodes.
        validate_args: Forwarded to ``numpyro.distributions.Distribution``.
    """

    arg_constraints = {
        "observation_window": constraints.positive,
    }
    # Samples are a ``(times, mask)`` PyTree rather than a vector in ℝⁿ.
    support = constraints.dependent
    reparametrized_params: list[str] = []

    def __init__(
        self,
        log_intensity_fn: Callable[[Array], Array],
        observation_window: ArrayLike,
        integrated_intensity: ArrayLike | None = None,
        lambda_max: ArrayLike | None = None,
        max_candidates: int = 1024,
        n_integration_points: int = 100,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.observation_window = jnp.asarray(observation_window)
        self._max_candidates = int(max_candidates)
        self._op = _IppOp(
            log_intensity_fn,
            self.observation_window,
            integrated_intensity=integrated_intensity,
            lambda_max=lambda_max,
            n_integration_points=n_integration_points,
        )
        batch_shape = self.observation_window.shape
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @property
    def max_candidates(self) -> int:
        return self._max_candidates

    def sample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[Float[Array, ...], Bool[Array, ...]]:
        """Return a ``(times, mask)`` PyTree (matching :meth:`log_prob`).

        ``sample_shape`` must be empty for now — batched IPP sampling
        over an upper-bound ``λ_max`` requires a shared buffer size,
        which we leave to the operator layer where the user can
        ``vmap`` explicitly. Drops the uncapped candidate count
        returned by the operator; that is available via
        ``mask.sum()`` for accepted events, or from the operator API
        if the raw count matters.
        """
        check_prng_key(key)
        if sample_shape != ():
            raise NotImplementedError(
                "IPP distribution sample_shape=() only; use vmap over "
                "PRNG keys at the operator layer for batched draws."
            )
        times, mask, _ = self._op.sample(key, self._max_candidates)
        return times, mask

    def log_prob(
        self,
        value: tuple[Float[Array, ...], Bool[Array, ...]],
    ) -> Float[Array, ...]:
        """Log-likelihood ``value = (event_times, mask)``."""
        event_times, mask = value
        return self._op.log_prob(event_times, mask)

    def _validate_sample(self, value) -> None:
        return None
