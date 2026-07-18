"""Equinox operator for separable marked spatial point processes.

Pairs a ground spatial process — :class:`HomogeneousSpatialPP` or
:class:`InhomogeneousSpatialPP` — with a user-supplied mark
distribution callable ``mark_distribution_fn(s) -> Distribution``.
Under the separable factorisation
:math:`\\lambda(s, m) = \\lambda(s) \\, f(m \\mid s)` the joint
log-likelihood splits into ground + marks, so most of the work is
delegated and the mark term is computed with the primitives in
:mod:`~xtremax.point_processes.primitives.marked_spatial`.

Marks may be scalar (``event_shape = ()``), vector
(``event_shape = (mark_dim,)``), or discrete — anything a
``numpyro.distributions.Distribution`` can express. Use a closure or a
small ``eqx.Module`` to make the mark law location-dependent.
"""

from __future__ import annotations

from collections.abc import Callable

import numpyro.distributions as dist
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from xtremax.point_processes._domain import RectangularDomain
from xtremax.point_processes._results import MarkedSpatialSampleResult
from xtremax.point_processes.operators._base import SeparableMarkedPP
from xtremax.point_processes.primitives.marked_spatial import (
    sample_spatial_marks_at_locations,
    spatial_marks_log_prob,
)


class MarkedSpatialPP(SeparableMarkedPP):
    """Separable marked spatial PP: ground process × mark distribution.

    Args:
        ground: Spatial PP operator exposing
            ``log_prob(locations, mask)`` and
            ``sample(key, max_events) -> (locations, mask, n)``.
        mark_distribution_fn: Callable ``s -> Distribution``. Called
            once per real event during ``log_prob`` and once per
            slot during ``sample`` (padding slot draws are masked
            out). Distributions with ``event_shape == ()`` give
            scalar marks; non-trivial ``event_shape`` gives vector
            marks.
        mark_dim: Static dimensionality of each mark; ``None`` for
            scalar marks. Used only by callers that need the mark
            shape up front (e.g. to preallocate buffers); the
            operator itself reads the shape from
            ``mark_distribution_fn``.
    """

    def __init__(
        self,
        ground: object,
        mark_distribution_fn: Callable[[Array], dist.Distribution],
        mark_dim: int | None = None,
    ) -> None:
        self.ground = ground
        self.mark_distribution_fn = mark_distribution_fn
        self.mark_dim = mark_dim

    @property
    def domain(self) -> RectangularDomain:
        """Forward to the ground operator's ``domain``."""
        return self.ground.domain

    @property
    def n_dims(self) -> int:
        """Spatial dimension ``d``."""
        return self.ground.n_dims

    # ------------------------------------------------------------
    # Core distribution API
    # ------------------------------------------------------------

    def log_prob(
        self,
        locations: Float[Array, ...],
        mask: Bool[Array, ...],
        marks: Float[Array, ...],
    ) -> Float[Array, ...]:
        """Joint log-likelihood: ``log L_ground(s) + Σ log f(mᵢ | sᵢ)``."""
        ground_lp = self.ground.log_prob(locations, mask)
        mark_lp = spatial_marks_log_prob(
            locations, marks, mask, self.mark_distribution_fn
        )
        return ground_lp + mark_lp

    def _sample_marks(
        self,
        key: PRNGKeyArray,
        *event_coords: Array,
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        (locations,) = event_coords
        return sample_spatial_marks_at_locations(
            key, locations, mask, self.mark_distribution_fn
        )

    def sample(
        self,
        key: PRNGKeyArray,
        max_events: int,
        **ground_kwargs,
    ) -> MarkedSpatialSampleResult:
        """Sample ground locations, then draw marks at each location.

        Returns :class:`MarkedSpatialSampleResult`
        ``(locations, mask, marks)`` — unpacks like the strict
        three-tuple it replaces. The ground's event count is dropped;
        ``mask`` already encodes which slots are real.
        """
        locations, mask, marks = self._sample_joint(key, max_events, **ground_kwargs)
        return MarkedSpatialSampleResult(locations, mask, marks)

    # ------------------------------------------------------------
    # Component pass-throughs
    # ------------------------------------------------------------

    def ground_log_prob(
        self,
        locations: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Log-likelihood of the ground locations only."""
        return self.ground.log_prob(locations, mask)

    def marks_log_prob(
        self,
        locations: Float[Array, ...],
        marks: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Log-likelihood of the marks only, conditional on locations."""
        return spatial_marks_log_prob(locations, marks, mask, self.mark_distribution_fn)

    def intensity(self, locations: Float[Array, ...]) -> Float[Array, ...]:
        """Forward to the ground operator's ``intensity``."""
        # Some operators (HPP) accept ``locations`` optionally; the
        # spatial PP operators all accept it as a positional argument.
        return self.ground.intensity(locations)

    # ``mark_intensity(locations, marks)`` — the joint intensity
    # λ(s, m) = λ(s)·f(m | s) — comes from SeparableMarkedPP.
