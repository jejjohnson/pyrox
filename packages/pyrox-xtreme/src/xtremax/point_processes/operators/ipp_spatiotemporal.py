"""Equinox operator for the inhomogeneous spatiotemporal Poisson process.

Bundles a log-intensity callable ``log_intensity_fn(s, t)`` with a
spatial domain, a temporal interval, and numerical defaults
(quadrature node count, optional pinned ``Λ`` and ``λ_max``). Sampling
uses Lewis–Shedler thinning on ``D × [t0, t1)``.

The same no-fallback rule as the spatial IPP applies to ``λ_max``:
``effective_lambda_max`` returns the pinned bound or queries
``log_intensity_fn.max_intensity()``, and otherwise raises. There is
no automatic ``2 Λ/(|D| T)`` heuristic — that produces silently biased
samples on peaked intensities.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._domain import RectangularDomain, TemporalDomain
from xtremax.point_processes._integration_spatiotemporal import (
    SpatiotemporalMethod,
    integrate_log_intensity_spatiotemporal,
)
from xtremax.point_processes.primitives import (
    ipp_spatiotemporal_chi_square_gof,
    ipp_spatiotemporal_intensity,
    ipp_spatiotemporal_intensity_surface_at_time,
    ipp_spatiotemporal_log_prob,
    ipp_spatiotemporal_marginal_spatial_intensity,
    ipp_spatiotemporal_marginal_temporal_intensity,
    ipp_spatiotemporal_pearson_residuals,
    ipp_spatiotemporal_predict_count,
    ipp_spatiotemporal_sample_thinning,
)


class InhomogeneousSpatioTemporalPP(eqx.Module):
    """Inhomogeneous Poisson process on ``D × [t0, t1)``.

    Args:
        log_intensity_fn: Callable ``(s, t) → log λ(s, t)`` accepting
            ``s: (N, d)`` and ``t: (N,)``. If this is itself an
            ``eqx.Module``, its parameters are real PyTree leaves and
            gradients flow through :meth:`log_prob`.
        spatial: Spatial domain.
        temporal: Temporal interval.
        integrated_intensity: Optional fixed value of :math:`\\Lambda`.
            Pin only when the intensity is static; during training,
            stale values silently bias the likelihood.
        lambda_max: Optional upper bound on ``λ(s, t)`` over the slab,
            required by :meth:`sample`. Must be a true upper bound.
        n_integration_points: Static node count for quadrature.
        integration_method: ``"qmc"`` (default) or ``"trapezoid"``.
    """

    log_intensity_fn: Callable[[Array, Array], Array]
    spatial: RectangularDomain
    temporal: TemporalDomain
    integrated_intensity: Float[Array, ...] | None
    lambda_max: Float[Array, ...] | None
    n_integration_points: int = eqx.field(static=True, default=4096)
    integration_method: SpatiotemporalMethod = eqx.field(static=True, default="qmc")

    def __init__(
        self,
        log_intensity_fn: Callable[[Array, Array], Array],
        spatial: RectangularDomain,
        temporal: TemporalDomain,
        integrated_intensity: ArrayLike | None = None,
        lambda_max: ArrayLike | None = None,
        n_integration_points: int = 4096,
        integration_method: SpatiotemporalMethod = "qmc",
    ) -> None:
        self.log_intensity_fn = log_intensity_fn
        self.spatial = spatial
        self.temporal = temporal
        self.integrated_intensity = (
            None if integrated_intensity is None else jnp.asarray(integrated_intensity)
        )
        self.lambda_max = None if lambda_max is None else jnp.asarray(lambda_max)
        self.n_integration_points = n_integration_points
        self.integration_method = integration_method

    @property
    def n_dims(self) -> int:
        return self.spatial.n_dims

    def effective_integrated_intensity(
        self,
        sub_spatial: RectangularDomain | None = None,
        sub_temporal: TemporalDomain | None = None,
    ) -> Float[Array, ...]:
        """Compute :math:`\\Lambda` over a sub-slab (defaults to full slab).

        Pinned :attr:`integrated_intensity` is honoured **only** when
        both ``sub_spatial`` and ``sub_temporal`` are ``None`` (i.e. the
        caller is asking about the originally-pinned slab); subdomain
        queries always re-integrate against the live intensity.
        """
        sp = self.spatial if sub_spatial is None else sub_spatial
        tp = self.temporal if sub_temporal is None else sub_temporal
        if (
            sub_spatial is None
            and sub_temporal is None
            and self.integrated_intensity is not None
        ):
            return self.integrated_intensity
        return integrate_log_intensity_spatiotemporal(
            self.log_intensity_fn,
            sp,
            tp,
            n_points=self.n_integration_points,
            method=self.integration_method,
        )

    def effective_lambda_max(self) -> Float[Array, ...]:
        """Return the thinning bound used by :meth:`sample`."""
        if self.lambda_max is not None:
            return self.lambda_max
        max_intensity = getattr(self.log_intensity_fn, "max_intensity", None)
        if max_intensity is not None:
            return max_intensity()
        raise ValueError(
            "Cannot sample via thinning: no `lambda_max` pinned on the "
            "operator and `log_intensity_fn` has no `.max_intensity()` "
            "method. Pass an upper bound at construction (it must be a "
            "true upper bound on `λ(s, t)` over the slab — too small a "
            "value silently biases the sampler low in intensity peaks)."
        )

    def log_prob(
        self,
        locations: Float[Array, ...],
        times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Janossy log-likelihood ``∑_i log λ(s_i, t_i) − Λ``."""
        Lambda = self.effective_integrated_intensity()
        return ipp_spatiotemporal_log_prob(
            locations, times, mask, self.log_intensity_fn, Lambda
        )

    def sample(
        self,
        key: PRNGKeyArray,
        max_candidates: int,
    ) -> tuple[Float[Array, ...], Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
        """Lewis–Shedler thinning sampler on the full slab."""
        lam_max = self.effective_lambda_max()
        return ipp_spatiotemporal_sample_thinning(
            key,
            self.log_intensity_fn,
            self.spatial,
            self.temporal,
            lam_max,
            max_candidates,
        )

    def intensity(
        self,
        locations: Float[Array, ...],
        times: Float[Array, ...],
    ) -> Float[Array, ...]:
        """``λ(s, t) = exp(log_intensity_fn(s, t))`` evaluated on a batch."""
        return ipp_spatiotemporal_intensity(locations, times, self.log_intensity_fn)

    def predict_count(
        self,
        sub_spatial: RectangularDomain | None = None,
        sub_temporal: TemporalDomain | None = None,
    ) -> tuple[Float[Array, ...], Float[Array, ...]]:
        """Mean and variance of the count over a sub-slab."""
        Lambda = self.effective_integrated_intensity(sub_spatial, sub_temporal)
        return ipp_spatiotemporal_predict_count(Lambda)

    def marginal_spatial_intensity(
        self,
        locations: Float[Array, ...],
        n_time_points: int = 100,
    ) -> Float[Array, ...]:
        """``λ_S(s) = ∫_{t0}^{t1} λ(s, t) dt`` via trapezoid in time."""
        return ipp_spatiotemporal_marginal_spatial_intensity(
            locations, self.log_intensity_fn, self.temporal, n_time_points=n_time_points
        )

    def marginal_temporal_intensity(
        self,
        times: Float[Array, ...],
        n_spatial_points: int = 256,
    ) -> Float[Array, ...]:
        """``λ_T(t) = ∫_D λ(s, t) ds`` via Halton QMC in space."""
        return ipp_spatiotemporal_marginal_temporal_intensity(
            times,
            self.log_intensity_fn,
            self.spatial,
            n_spatial_points=n_spatial_points,
        )

    def intensity_surface_at_time(
        self,
        t: Float[Array, ...],
        grid_size: int = 50,
    ) -> tuple[Float[Array, ...], Float[Array, ...]]:
        """Evaluate ``λ(s, t)`` on a tensor-product spatial grid at fixed ``t``."""
        return ipp_spatiotemporal_intensity_surface_at_time(
            t, self.log_intensity_fn, self.spatial, grid_size=grid_size
        )

    def pearson_residuals(
        self,
        locations: Float[Array, ...],
        times: Float[Array, ...],
        mask: Bool[Array, ...],
        n_spatial_bins: int = 5,
        n_temporal_bins: int = 5,
        n_integration_points: int = 16,
    ) -> Float[Array, ...]:
        """Cell-wise Pearson residuals on a space-time grid."""
        return ipp_spatiotemporal_pearson_residuals(
            locations,
            times,
            mask,
            self.log_intensity_fn,
            self.spatial,
            self.temporal,
            n_spatial_bins=n_spatial_bins,
            n_temporal_bins=n_temporal_bins,
            n_integration_points=n_integration_points,
        )

    def chi_square_gof(
        self,
        locations: Float[Array, ...],
        times: Float[Array, ...],
        mask: Bool[Array, ...],
        n_spatial_bins: int = 5,
        n_temporal_bins: int = 5,
        n_integration_points: int = 16,
    ) -> Float[Array, ...]:
        """Pearson :math:`\\chi^2` statistic from binned residuals."""
        return ipp_spatiotemporal_chi_square_gof(
            locations,
            times,
            mask,
            self.log_intensity_fn,
            self.spatial,
            self.temporal,
            n_spatial_bins=n_spatial_bins,
            n_temporal_bins=n_temporal_bins,
            n_integration_points=n_integration_points,
        )
