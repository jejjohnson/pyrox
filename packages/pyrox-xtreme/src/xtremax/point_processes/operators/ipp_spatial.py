"""Equinox operator for the inhomogeneous spatial Poisson process.

The operator stores a log-intensity callable, a rectangular domain,
and numerical defaults (quadrature node count, optional pinned
``λ_max``). Sampling uses Lewis–Shedler thinning and ``log_prob`` uses
the same closed-form
``∑ log λ(sᵢ) − Λ(D)`` decomposition as the temporal IPP.

If the user pins ``integrated_intensity`` and/or ``lambda_max`` the
operator trusts those values; otherwise it computes them on every
call from the live ``log_intensity_fn``. This matches the temporal IPP
contract — an optimiser updating intensity parameters can keep the
quadrature-based estimates fresh without having to re-instantiate.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._domain import RectangularDomain
from xtremax.point_processes._integration_spatial import (
    SpatialMethod,
    integrate_log_intensity_spatial,
)
from xtremax.point_processes.primitives import (
    ipp_spatial_intensity,
    ipp_spatial_log_prob,
    ipp_spatial_predict_count,
    ipp_spatial_sample_thinning,
)


class InhomogeneousSpatialPP(eqx.Module):
    """Inhomogeneous spatial Poisson process on a rectangular domain.

    Args:
        log_intensity_fn: Callable mapping ``(..., n, d)`` locations to
            ``(..., n)`` log-intensities. If this is itself an
            ``eqx.Module`` (e.g. a neural intensity), its parameters
            are real PyTree leaves and gradients flow through
            :meth:`log_prob`.
        domain: Rectangular domain ``D ⊂ ℝᵈ``.
        integrated_intensity: Optional fixed value of ``Λ(D)``. Pin
            **only** when the intensity is static; during training,
            stale values silently bias the likelihood. ``None``
            triggers quadrature on every call.
        lambda_max: Optional upper bound on ``λ(s)`` over ``D``,
            required by :meth:`sample`. Must be a *true* upper bound
            for thinning to be exact — passing too small a value
            silently biases the sampled process low in peaks. If
            ``None``, the operator queries
            ``log_intensity_fn.max_intensity()`` when the intensity
            module exposes it and otherwise raises at sample time.
            There is no automatic fallback (e.g. ``2 Λ / |D|``)
            because no estimator can guarantee an upper bound for an
            arbitrary user-supplied intensity.
        n_integration_points: Static node count for quadrature
            (trapezoid total nodes or QMC samples).
        integration_method: ``"qmc"`` (default) or ``"trapezoid"``.
            QMC is preferred for ``d ≥ 3``; trapezoid is exact for
            multilinear log-intensities.
    """

    log_intensity_fn: Callable[[Array], Array]
    domain: RectangularDomain
    integrated_intensity: Float[Array, ...] | None
    lambda_max: Float[Array, ...] | None
    n_integration_points: int = eqx.field(static=True, default=2048)
    integration_method: SpatialMethod = eqx.field(static=True, default="qmc")

    def __init__(
        self,
        log_intensity_fn: Callable[[Array], Array],
        domain: RectangularDomain,
        integrated_intensity: ArrayLike | None = None,
        lambda_max: ArrayLike | None = None,
        n_integration_points: int = 2048,
        integration_method: SpatialMethod = "qmc",
    ) -> None:
        self.log_intensity_fn = log_intensity_fn
        self.domain = domain
        self.integrated_intensity = (
            None if integrated_intensity is None else jnp.asarray(integrated_intensity)
        )
        self.lambda_max = None if lambda_max is None else jnp.asarray(lambda_max)
        self.n_integration_points = n_integration_points
        self.integration_method = integration_method

    @property
    def n_dims(self) -> int:
        """Spatial dimension ``d``."""
        return self.domain.n_dims

    # ------------------------------------------------------------
    # Live-value accessors
    # ------------------------------------------------------------

    def effective_integrated_intensity(
        self,
        subdomain: RectangularDomain | None = None,
    ) -> Float[Array, ...]:
        """Compute :math:`\\Lambda(A)` for ``A = subdomain`` (default ``D``).

        Precedence: (1) pinned :attr:`integrated_intensity` for the
        default domain only; (2) quadrature against the live
        ``log_intensity_fn`` otherwise. Sub-domain queries always
        re-integrate so they stay consistent with the live intensity.
        """
        target = self.domain if subdomain is None else subdomain
        if subdomain is None and self.integrated_intensity is not None:
            return self.integrated_intensity
        return integrate_log_intensity_spatial(
            self.log_intensity_fn,
            target,
            n_points=self.n_integration_points,
            method=self.integration_method,
        )

    def effective_lambda_max(self) -> Float[Array, ...]:
        """Return the thinning bound used by :meth:`sample`.

        Prefers the pinned :attr:`lambda_max`; otherwise queries the
        intensity module's ``.max_intensity()`` if it has one (e.g. a
        :class:`~xtremax.point_processes.operators.temporal.PiecewiseConstantLogIntensity`-style
        helper that tracks ``log_rates`` and reports
        ``exp(max(log_rates))`` live). Raises when neither is
        available — Lewis–Shedler thinning requires a *true* upper
        bound, and no quadrature-derived estimate (e.g. ``2 Λ / |D|``,
        twice the mean intensity) can guarantee that for an arbitrary
        intensity surface.
        """
        if self.lambda_max is not None:
            return self.lambda_max
        max_intensity = getattr(self.log_intensity_fn, "max_intensity", None)
        if max_intensity is not None:
            return max_intensity()
        raise ValueError(
            "Cannot sample via thinning: no `lambda_max` pinned on the "
            "operator and `log_intensity_fn` has no `.max_intensity()` "
            "method. Pass an upper bound at construction (it must be a "
            "true upper bound on `λ(s)` over the domain — too small a "
            "value silently biases the sampler low in intensity peaks)."
        )

    # ------------------------------------------------------------
    # Core distribution API
    # ------------------------------------------------------------

    def log_prob(
        self,
        locations: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Log-likelihood :math:`\\sum \\log \\lambda(s_i) - \\Lambda(D)`."""
        Lambda_D = self.effective_integrated_intensity()
        return ipp_spatial_log_prob(locations, mask, self.log_intensity_fn, Lambda_D)

    def sample(
        self,
        key: PRNGKeyArray,
        max_candidates: int,
    ) -> tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
        """Lewis–Shedler thinning sampler with a static buffer."""
        lam_max = self.effective_lambda_max()
        return ipp_spatial_sample_thinning(
            key, self.log_intensity_fn, self.domain, lam_max, max_candidates
        )

    # ------------------------------------------------------------
    # Intensity and predictions
    # ------------------------------------------------------------

    def intensity(self, locations: Float[Array, ...]) -> Float[Array, ...]:
        """``λ(s) = exp(log_intensity_fn(s))`` evaluated at ``locations``."""
        return ipp_spatial_intensity(locations, self.log_intensity_fn)

    def predict_count(
        self,
        subdomain: RectangularDomain | None = None,
    ) -> Float[Array, ...]:
        """Expected count over a sub-domain (defaults to the full ``D``)."""
        target = self.domain if subdomain is None else subdomain
        return ipp_spatial_predict_count(
            self.log_intensity_fn,
            target,
            n_integration_points=self.n_integration_points,
            method=self.integration_method,
        )
