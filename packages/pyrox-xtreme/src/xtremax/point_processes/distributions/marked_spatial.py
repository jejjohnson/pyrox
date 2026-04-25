"""NumPyro ``Distribution`` wrapper for the marked spatial PP."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import numpyro.distributions as dist
from jaxtyping import Array, Bool, Float, PRNGKeyArray
from numpyro.distributions import constraints

from xtremax._rng import check_prng_key
from xtremax.point_processes.operators.marked_spatial import (
    MarkedSpatialPP as _MarkedSpatialOp,
)


class MarkedSpatialPP(dist.Distribution):
    """NumPyro wrapper for a separable marked spatial Poisson process.

    Args:
        ground: Spatial PP operator (``HomogeneousSpatialPP`` or
            ``InhomogeneousSpatialPP``) — *the operator-layer object*,
            not its distribution wrapper. The buffer size lives on the
            wrapper, not the operator, so we accept it explicitly.
        mark_distribution_fn: Callable ``s -> Distribution`` for marks.
        max_events: Static buffer size for ``sample`` — forwarded to
            the ground operator's sampler. Choose comfortably above
            the expected count.
        mark_dim: Static mark dimensionality (``None`` for scalar).
        validate_args: Forwarded to ``numpyro.distributions.Distribution``.
    """

    support = constraints.dependent
    reparametrized_params: list[str] = []

    def __init__(
        self,
        ground: eqx.Module,
        mark_distribution_fn: Callable[[Array], dist.Distribution],
        max_events: int = 1024,
        mark_dim: int | None = None,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self._op = _MarkedSpatialOp(
            ground=ground,
            mark_distribution_fn=mark_distribution_fn,
            mark_dim=mark_dim,
        )
        self._max_events = int(max_events)
        super().__init__(batch_shape=(), validate_args=validate_args)

    @property
    def max_events(self) -> int:
        return self._max_events

    @property
    def n_dims(self) -> int:
        return self._op.n_dims

    @property
    def mark_dim(self) -> int | None:
        return self._op.mark_dim

    def sample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[Float[Array, ...], Bool[Array, ...], Float[Array, ...]]:
        """Return ``(locations, mask, marks)``.

        Sample shapes other than ``()`` are not supported here — vmap
        over PRNG keys at the operator layer for batched draws.
        """
        check_prng_key(key)
        if sample_shape != ():
            raise NotImplementedError(
                "MarkedSpatialPP distribution supports sample_shape=() only; "
                "vmap over PRNG keys at the operator layer for batched draws."
            )
        return self._op.sample(key, self._max_events)

    def log_prob(
        self,
        value: tuple[Float[Array, ...], Bool[Array, ...], Float[Array, ...]],
    ) -> Float[Array, ...]:
        """Joint log-likelihood ``log L_ground + Σ log f(mᵢ | sᵢ)``."""
        locations, mask, marks = value
        return self._op.log_prob(locations, mask, marks)

    def _validate_sample(self, value) -> None:
        return None
