"""NumPyro ``Distribution`` wrapper for the homogeneous spatial PP."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from numpyro.distributions import constraints

from xtremax._rng import check_prng_key
from xtremax.point_processes._domain import RectangularDomain
from xtremax.point_processes.operators.hpp_spatial import (
    HomogeneousSpatialPP as _HppSpatialOp,
)
from xtremax.point_processes.primitives.hpp_spatial import hpp_spatial_log_prob


class HomogeneousSpatialPP(dist.Distribution):
    """NumPyro wrapper for a homogeneous spatial Poisson process.

    Args:
        rate: Intensity ``λ > 0``.
        domain: Rectangular domain.
        max_events: Static buffer size for ``sample``.
        validate_args: Forwarded to ``numpyro.distributions.Distribution``.
    """

    arg_constraints = {"rate": constraints.positive}
    # Samples are a ``(locations, mask)`` PyTree, not a vector in ℝⁿ.
    support = constraints.dependent
    reparametrized_params: list[str] = []

    def __init__(
        self,
        rate: ArrayLike,
        domain: RectangularDomain,
        max_events: int = 512,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.rate = jnp.asarray(rate)
        if self.rate.shape != ():
            raise ValueError(
                "HomogeneousSpatialPP distribution requires a scalar `rate`; "
                f"got shape {self.rate.shape}. Spatial sampling produces "
                "variable-length point patterns and is not batch-safe at the "
                "distribution layer — vmap over `rate` at the operator layer "
                "(`xtremax.point_processes.operators.HomogeneousSpatialPP`) "
                "if you need batched draws."
            )
        self.domain = domain
        self._max_events = int(max_events)
        self._op = _HppSpatialOp(self.rate, self.domain)
        # Empty batch_shape — `rate` is required scalar so the
        # distribution carries no batched axes.
        super().__init__(batch_shape=(), validate_args=validate_args)

    @property
    def max_events(self) -> int:
        return self._max_events

    @property
    def n_dims(self) -> int:
        return self.domain.n_dims

    def sample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[Float[Array, ...], Bool[Array, ...]]:
        """Return ``(locations, mask)``."""
        check_prng_key(key)
        if sample_shape != ():
            raise NotImplementedError(
                "HomogeneousSpatialPP distribution supports sample_shape=() "
                "only; vmap over PRNG keys at the operator layer for batched draws."
            )
        locations, mask, _ = self._op.sample(key, self._max_events)
        return locations, mask

    def log_prob(
        self,
        value: tuple[Float[Array, ...], Bool[Array, ...]]
        | tuple[Float[Array, ...], Int[Array, ...]],
    ) -> Float[Array, ...]:
        """Joint log-likelihood. ``value`` may be ``(locations, mask)`` or
        ``(locations, n_events)``.
        """
        locations, counts_or_mask = value
        del locations  # the HPP log-likelihood depends only on the count
        counts_or_mask = jnp.asarray(counts_or_mask)
        if counts_or_mask.dtype == jnp.bool_:
            n_events = jnp.sum(counts_or_mask, axis=-1)
        else:
            # Use the raw count directly. Fabricating a mask from
            # ``ranks < count`` would silently clip ``count > max_events``
            # to ``max_events`` and return the wrong likelihood for any
            # caller scoring observed counts above the buffer size
            # (Codex P1 in PR #12 review).
            n_events = counts_or_mask
        return hpp_spatial_log_prob(n_events, self._op.rate, self._op.domain.volume())

    def _validate_sample(self, value) -> None:
        return None
