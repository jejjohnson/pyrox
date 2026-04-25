"""Equinox operator for the homogeneous spatiotemporal Poisson process."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._domain import RectangularDomain, TemporalDomain
from xtremax.point_processes.primitives import (
    hpp_spatiotemporal_count_log_prob,
    hpp_spatiotemporal_intensity,
    hpp_spatiotemporal_log_prob,
    hpp_spatiotemporal_predict_count,
    hpp_spatiotemporal_sample,
)


class HomogeneousSpatioTemporalPP(eqx.Module):
    """Homogeneous Poisson process on ``D × [t0, t1)``.

    Args:
        rate: Intensity ``λ > 0`` (points per unit space-time volume).
        spatial: Spatial domain.
        temporal: Temporal interval.
    """

    rate: Float[Array, ...]
    spatial: RectangularDomain
    temporal: TemporalDomain

    def __init__(
        self,
        rate: ArrayLike,
        spatial: RectangularDomain,
        temporal: TemporalDomain,
    ) -> None:
        self.rate = jnp.asarray(rate)
        self.spatial = spatial
        self.temporal = temporal

    @property
    def n_dims(self) -> int:
        """Spatial dimension ``d`` (forwarded from the spatial domain)."""
        return self.spatial.n_dims

    def log_prob(
        self,
        locations: Float[Array, ...],
        times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Janossy log-likelihood ``n log λ − λ |D| (t1 − t0)``."""
        del locations, times  # the homogeneous likelihood depends only on the count
        n_events = jnp.sum(mask, axis=-1)
        return hpp_spatiotemporal_log_prob(
            n_events, self.rate, self.spatial.volume(), self.temporal.duration
        )

    def count_log_prob(self, n_events: Int[Array, ...]) -> Float[Array, ...]:
        """Marginal Poisson-count log-pmf with mean ``λ |D| T``."""
        return hpp_spatiotemporal_count_log_prob(
            n_events, self.rate, self.spatial.volume(), self.temporal.duration
        )

    def sample(
        self,
        key: PRNGKeyArray,
        max_events: int,
    ) -> tuple[Float[Array, ...], Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
        """Draw ``(locations, times, mask, n_events)``; times sorted ascending."""
        return hpp_spatiotemporal_sample(
            key, self.rate, self.spatial, self.temporal, max_events
        )

    def intensity(
        self,
        locations: Float[Array, ...] | None = None,
        times: Float[Array, ...] | None = None,
    ) -> Float[Array, ...]:
        """Pointwise intensity (constant)."""
        if locations is None or times is None:
            return self.rate
        return hpp_spatiotemporal_intensity(locations, times, self.rate)

    def predict_count(
        self,
        sub_spatial: RectangularDomain | None = None,
        sub_temporal: TemporalDomain | None = None,
    ) -> tuple[Float[Array, ...], Float[Array, ...]]:
        """Mean and variance of the count over a sub-slab (defaults to full slab)."""
        vol = (sub_spatial or self.spatial).volume()
        dur = (sub_temporal or self.temporal).duration
        return hpp_spatiotemporal_predict_count(self.rate, vol, dur)
