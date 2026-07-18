"""Equinox operator for separable marked spatiotemporal PPs.

Pairs a ground spatiotemporal process — :class:`HomogeneousSpatioTemporalPP`
or :class:`InhomogeneousSpatioTemporalPP` — with a user-supplied mark
distribution callable ``mark_distribution_fn(s, t) -> Distribution``.
Under the separable factorisation
:math:`\\lambda(s, t, m) = \\lambda(s, t) \\, f(m \\mid s, t)` the joint
log-likelihood splits into ground + marks; the operator delegates to the
primitives in
:mod:`~xtremax.point_processes.primitives.marked_spatiotemporal`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpyro.distributions as dist
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from xtremax.point_processes._domain import RectangularDomain, TemporalDomain
from xtremax.point_processes._results import MarkedSpatiotemporalSampleResult
from xtremax.point_processes.operators._base import SeparableMarkedPP
from xtremax.point_processes.primitives.marked_spatiotemporal import (
    sample_spatiotemporal_marks,
    spatiotemporal_marks_log_prob,
)


class MarkedSpatioTemporalPP(SeparableMarkedPP):
    """Separable marked spatiotemporal PP: ground × mark distribution.

    Args:
        ground: Spatiotemporal PP operator exposing
            ``log_prob(locations, times, mask)`` and
            ``sample(key, max_events) -> (locs, times, mask, n)``.
        mark_distribution_fn: Callable ``(s, t) -> Distribution``. Used
            in both ``log_prob`` (per real event) and ``sample`` (per
            slot — padding slot draws are masked out by
            :func:`sample_spatiotemporal_marks`).
        mark_dim: Static dimensionality of the marks; ``None`` for
            scalar marks. Same role as in the marked spatial operator.
    """

    def __init__(
        self,
        ground: object,
        mark_distribution_fn: Callable[[Array, Array], dist.Distribution],
        mark_dim: int | None = None,
    ) -> None:
        self.ground = ground
        self.mark_distribution_fn = mark_distribution_fn
        self.mark_dim = mark_dim

    @property
    def spatial(self) -> RectangularDomain:
        """Forward to the ground operator's ``spatial`` domain."""
        return self.ground.spatial

    @property
    def temporal(self) -> TemporalDomain:
        """Forward to the ground operator's ``temporal`` interval."""
        return self.ground.temporal

    @property
    def n_dims(self) -> int:
        return self.ground.n_dims

    def log_prob(
        self,
        locations: Float[Array, ...],
        times: Float[Array, ...],
        mask: Bool[Array, ...],
        marks: Float[Array, ...],
    ) -> Float[Array, ...]:
        """Joint log-likelihood: ``log L_ground + Σ log f(mᵢ | sᵢ, tᵢ)``."""
        ground_lp = self.ground.log_prob(locations, times, mask)
        mark_lp = spatiotemporal_marks_log_prob(
            locations, times, marks, mask, self.mark_distribution_fn
        )
        return ground_lp + mark_lp

    def _sample_marks(
        self,
        key: PRNGKeyArray,
        *event_coords: Array,
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        locations, times = event_coords
        return sample_spatiotemporal_marks(
            key, locations, times, mask, self.mark_distribution_fn
        )

    def sample(
        self,
        key: PRNGKeyArray,
        max_events: int,
        **ground_kwargs,
    ) -> MarkedSpatiotemporalSampleResult:
        """Sample ground (locations, times, mask), then marks at each event.

        Returns :class:`MarkedSpatiotemporalSampleResult` — unpacks like
        the ``(locations, times, mask, marks)`` tuple it replaces.
        """
        locations, times, mask, marks = self._sample_joint(
            key, max_events, **ground_kwargs
        )
        return MarkedSpatiotemporalSampleResult(locations, times, mask, marks)

    def ground_log_prob(
        self,
        locations: Float[Array, ...],
        times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Log-likelihood of the ground process only."""
        return self.ground.log_prob(locations, times, mask)

    def marks_log_prob(
        self,
        locations: Float[Array, ...],
        times: Float[Array, ...],
        marks: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Log-likelihood of the marks only, conditional on (s, t)."""
        return spatiotemporal_marks_log_prob(
            locations, times, marks, mask, self.mark_distribution_fn
        )

    def intensity(
        self,
        locations: Float[Array, ...],
        times: Float[Array, ...],
    ) -> Float[Array, ...]:
        """Forward to the ground operator's ``intensity``."""
        return self.ground.intensity(locations, times)

    # ``mark_intensity(locations, times, marks)`` — the joint intensity
    # λ(s, t, m) = λ(s, t)·f(m | s, t) — comes from SeparableMarkedPP.
