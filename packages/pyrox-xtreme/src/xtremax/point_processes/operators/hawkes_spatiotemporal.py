r"""Equinox operator for the spatiotemporal Hawkes process.

ETAS-style separable exp-Gaussian model:

.. math::
    \lambda^*(s, t) = \mu + \sum_{t_i < t} \alpha\,
                     \beta e^{-\beta (t - t_i)}\,
                     \mathcal{N}(s; s_i, \sigma^2 I).

Parameters live as positive PyTree leaves; pass them through a softplus
or ``jnp.exp`` reparameterisation in user code if doing unconstrained
optimisation.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._domain import RectangularDomain, TemporalDomain
from xtremax.point_processes.primitives import (
    stpp_hawkes_compensator,
    stpp_hawkes_intensity,
    stpp_hawkes_log_prob,
    stpp_hawkes_sample,
)


class SpatioTemporalHawkes(eqx.Module):
    """ETAS-style separable exp-Gaussian Hawkes process.

    Args:
        mu: Baseline rate ``μ > 0`` (constant in space and time).
        alpha: Excitation amplitude (``α < 1`` for sub-criticality).
        beta: Temporal decay rate ``β > 0``.
        sigma: Spatial bandwidth ``σ > 0``.
        spatial: Spatial domain.
        temporal: Temporal interval.
        boundary_correction: When ``True`` (default) the compensator
            uses the axis-wise erf-CDF to compute the in-domain mass
            of each event's spatial kernel. When ``False`` it assumes
            the whole spatial kernel mass falls in ``D`` (cheap, exact
            when ``σ`` is small relative to the box and events are
            interior).
    """

    mu: Float[Array, ...]
    alpha: Float[Array, ...]
    beta: Float[Array, ...]
    sigma: Float[Array, ...]
    spatial: RectangularDomain
    temporal: TemporalDomain
    boundary_correction: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        mu: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        sigma: ArrayLike,
        spatial: RectangularDomain,
        temporal: TemporalDomain,
        boundary_correction: bool = True,
    ) -> None:
        self.mu = jnp.asarray(mu)
        self.alpha = jnp.asarray(alpha)
        self.beta = jnp.asarray(beta)
        self.sigma = jnp.asarray(sigma)
        self.spatial = spatial
        self.temporal = temporal
        self.boundary_correction = boundary_correction

    @property
    def n_dims(self) -> int:
        return self.spatial.n_dims

    def intensity(
        self,
        s: Float[Array, ...],
        t: Float[Array, ...],
        history_locations: Float[Array, ...],
        history_times: Float[Array, ...],
        history_mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Conditional intensity :math:`\\lambda^*(s, t \\mid \\mathcal H_t)`."""
        return stpp_hawkes_intensity(
            s,
            t,
            history_locations,
            history_times,
            history_mask,
            self.mu,
            self.alpha,
            self.beta,
            self.sigma,
        )

    def compensator(
        self,
        locations: Float[Array, ...],
        times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Closed-form compensator :math:`\\Lambda` over the slab."""
        return stpp_hawkes_compensator(
            locations,
            times,
            mask,
            self.mu,
            self.alpha,
            self.beta,
            self.sigma,
            self.spatial,
            self.temporal,
            boundary_correction=self.boundary_correction,
        )

    def log_prob(
        self,
        locations: Float[Array, ...],
        times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Log-likelihood :math:`\\sum_i \\log \\lambda^*(s_i, t_i) - \\Lambda`."""
        return stpp_hawkes_log_prob(
            locations,
            times,
            mask,
            self.mu,
            self.alpha,
            self.beta,
            self.sigma,
            self.spatial,
            self.temporal,
            boundary_correction=self.boundary_correction,
        )

    def sample(
        self,
        key: PRNGKeyArray,
        max_events: int,
    ) -> tuple[Float[Array, ...], Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
        """Ogata-style spatiotemporal thinning sample."""
        return stpp_hawkes_sample(
            key,
            self.mu,
            self.alpha,
            self.beta,
            self.sigma,
            self.spatial,
            self.temporal,
            max_events,
        )
