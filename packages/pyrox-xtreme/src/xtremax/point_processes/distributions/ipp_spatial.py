"""NumPyro ``Distribution`` wrapper for the inhomogeneous spatial PP."""

from __future__ import annotations

from collections.abc import Callable

import numpyro.distributions as dist
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, PRNGKeyArray
from numpyro.distributions import constraints

from xtremax._rng import check_prng_key
from xtremax.point_processes._domain import RectangularDomain
from xtremax.point_processes._integration_spatial import SpatialMethod
from xtremax.point_processes.operators.ipp_spatial import (
    InhomogeneousSpatialPP as _IppSpatialOp,
)


class InhomogeneousSpatialPP(dist.Distribution):
    """NumPyro wrapper for an inhomogeneous spatial Poisson process.

    Args:
        log_intensity_fn: Log-intensity callable ``(..., n, d) -> (..., n)``.
        domain: Rectangular domain.
        integrated_intensity: Optional pinned ``Λ(D)``; ``None``
            triggers quadrature.
        lambda_max: Optional thinning bound; ``None`` falls back to the
            conservative ``2 Λ(D) / |D|``.
        max_candidates: Static buffer for thinning.
        n_integration_points: Quadrature node count.
        integration_method: ``"qmc"`` (default) or ``"trapezoid"``.
        validate_args: Forwarded to ``numpyro.distributions.Distribution``.
    """

    support = constraints.dependent
    reparametrized_params: list[str] = []

    def __init__(
        self,
        log_intensity_fn: Callable[[Array], Array],
        domain: RectangularDomain,
        integrated_intensity: ArrayLike | None = None,
        lambda_max: ArrayLike | None = None,
        max_candidates: int = 1024,
        n_integration_points: int = 2048,
        integration_method: SpatialMethod = "qmc",
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.domain = domain
        self._max_candidates = int(max_candidates)
        self._op = _IppSpatialOp(
            log_intensity_fn,
            domain,
            integrated_intensity=integrated_intensity,
            lambda_max=lambda_max,
            n_integration_points=n_integration_points,
            integration_method=integration_method,
        )
        # No batched parameters here — the domain is a structural
        # element. Use an empty batch shape; vmap externally for
        # multiple intensities.
        super().__init__(batch_shape=(), validate_args=validate_args)

    @property
    def max_candidates(self) -> int:
        return self._max_candidates

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
                "InhomogeneousSpatialPP distribution supports sample_shape=() "
                "only; vmap over PRNG keys at the operator layer for batched draws."
            )
        locations, mask, _ = self._op.sample(key, self._max_candidates)
        return locations, mask

    def log_prob(
        self,
        value: tuple[Float[Array, ...], Bool[Array, ...]],
    ) -> Float[Array, ...]:
        """Log-likelihood ``Σ log λ(sᵢ) − Λ(D)``."""
        locations, mask = value
        return self._op.log_prob(locations, mask)

    def _validate_sample(self, value) -> None:
        return None
