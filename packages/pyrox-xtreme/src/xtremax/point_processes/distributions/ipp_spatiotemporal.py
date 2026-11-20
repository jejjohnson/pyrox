"""NumPyro ``Distribution`` wrapper for the inhomogeneous spatiotemporal PP."""

from __future__ import annotations

from collections.abc import Callable

import numpyro.distributions as dist
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, PRNGKeyArray
from numpyro.distributions import constraints

from xtremax._rng import check_prng_key
from xtremax.point_processes._domain import RectangularDomain, TemporalDomain
from xtremax.point_processes._integration_spatiotemporal import SpatiotemporalMethod
from xtremax.point_processes.operators.ipp_spatiotemporal import (
    InhomogeneousSpatioTemporalPP as _IppStOp,
)


class InhomogeneousSpatioTemporalPP(dist.Distribution):
    """NumPyro wrapper for an inhomogeneous spatiotemporal Poisson process.

    Args:
        log_intensity_fn: Callable ``(s, t) → log λ(s, t)``.
        spatial: Spatial domain.
        temporal: Temporal interval.
        integrated_intensity: Optional pinned :math:`\\Lambda`.
        lambda_max: Optional thinning bound (must be a *true* upper
            bound; the operator raises if neither this nor a
            ``log_intensity_fn.max_intensity()`` is available).
        max_candidates: Static thinning buffer.
        n_integration_points: Quadrature node count.
        integration_method: ``"qmc"`` (default) or ``"trapezoid"``.
        validate_args: Forwarded to ``numpyro``.
    """

    support = constraints.dependent
    reparametrized_params: list[str] = []

    def __init__(
        self,
        log_intensity_fn: Callable[[Array, Array], Array],
        spatial: RectangularDomain,
        temporal: TemporalDomain,
        integrated_intensity: ArrayLike | None = None,
        lambda_max: ArrayLike | None = None,
        max_candidates: int = 1024,
        n_integration_points: int = 4096,
        integration_method: SpatiotemporalMethod = "qmc",
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.spatial = spatial
        self.temporal = temporal
        self._max_candidates = int(max_candidates)
        self._op = _IppStOp(
            log_intensity_fn,
            spatial,
            temporal,
            integrated_intensity=integrated_intensity,
            lambda_max=lambda_max,
            n_integration_points=n_integration_points,
            integration_method=integration_method,
        )
        super().__init__(batch_shape=(), validate_args=validate_args)

    @property
    def max_candidates(self) -> int:
        return self._max_candidates

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
                "InhomogeneousSpatioTemporalPP supports sample_shape=() only; "
                "vmap over PRNG keys at the operator layer for batched draws."
            )
        locations, times, mask, _ = self._op.sample(key, self._max_candidates)
        return locations, times, mask

    def log_prob(
        self,
        value: tuple[Float[Array, ...], Float[Array, ...], Bool[Array, ...]],
    ) -> Float[Array, ...]:
        """Log-likelihood :math:`\\sum_i \\log \\lambda(s_i, t_i) - \\Lambda`."""
        locations, times, mask = value
        return self._op.log_prob(locations, times, mask)

    def _validate_sample(self, value) -> None:
        return None
