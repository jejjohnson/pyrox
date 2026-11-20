"""NumPyro ``Distribution`` wrapper for the homogeneous spatiotemporal PP."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from numpyro.distributions import constraints

from xtremax._rng import check_prng_key
from xtremax.point_processes._domain import RectangularDomain, TemporalDomain
from xtremax.point_processes.operators.hpp_spatiotemporal import (
    HomogeneousSpatioTemporalPP as _HppStOp,
)
from xtremax.point_processes.primitives.hpp_spatiotemporal import (
    hpp_spatiotemporal_log_prob,
)


class HomogeneousSpatioTemporalPP(dist.Distribution):
    """NumPyro wrapper for a homogeneous spatiotemporal Poisson process.

    Args:
        rate: Intensity ``λ > 0`` (scalar — vmap at the operator layer
            for batched rates).
        spatial: Spatial domain.
        temporal: Temporal interval.
        max_events: Static buffer size for ``sample``.
        validate_args: Forwarded to ``numpyro``.
    """

    arg_constraints = {"rate": constraints.positive}
    support = constraints.dependent
    reparametrized_params: list[str] = []

    def __init__(
        self,
        rate: ArrayLike,
        spatial: RectangularDomain,
        temporal: TemporalDomain,
        max_events: int = 512,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.rate = jnp.asarray(rate)
        if self.rate.shape != ():
            raise ValueError(
                "HomogeneousSpatioTemporalPP distribution requires a scalar "
                f"`rate`; got shape {self.rate.shape}. Spatiotemporal sampling "
                "produces variable-length point patterns and is not batch-safe "
                "at the distribution layer — vmap over `rate` at the operator "
                "layer "
                "(`xtremax.point_processes.operators.HomogeneousSpatioTemporalPP`) "
                "if you need batched draws."
            )
        self.spatial = spatial
        self.temporal = temporal
        self._max_events = int(max_events)
        self._op = _HppStOp(self.rate, self.spatial, self.temporal)
        super().__init__(batch_shape=(), validate_args=validate_args)

    @property
    def max_events(self) -> int:
        return self._max_events

    @property
    def n_dims(self) -> int:
        return self.spatial.n_dims

    def sample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[Float[Array, ...], Float[Array, ...], Bool[Array, ...]]:
        """Return ``(locations, times, mask)``."""
        check_prng_key(key)
        if sample_shape != ():
            raise NotImplementedError(
                "HomogeneousSpatioTemporalPP supports sample_shape=() only; "
                "vmap over PRNG keys at the operator layer for batched draws."
            )
        locations, times, mask, _ = self._op.sample(key, self._max_events)
        return locations, times, mask

    def log_prob(
        self,
        value: tuple[Float[Array, ...], Float[Array, ...], Bool[Array, ...]]
        | tuple[Float[Array, ...], Float[Array, ...], Int[Array, ...]],
    ) -> Float[Array, ...]:
        """Joint log-likelihood. ``value`` may be ``(locations, times, mask)``
        or ``(locations, times, n_events)``.
        """
        locations, times, counts_or_mask = value
        del locations, times
        counts_or_mask = jnp.asarray(counts_or_mask)
        if counts_or_mask.dtype == jnp.bool_:
            n_events = jnp.sum(counts_or_mask, axis=-1)
        else:
            n_events = counts_or_mask
        return hpp_spatiotemporal_log_prob(
            n_events,
            self._op.rate,
            self._op.spatial.volume(),
            self._op.temporal.duration,
        )

    def _validate_sample(self, value) -> None:
        return None
