"""Equinox operator for the homogeneous spatial Poisson process.

Bundles a positive intensity ``rate`` with a
:class:`~xtremax.point_processes._domain.RectangularDomain` and exposes
the same ``log_prob`` / ``sample`` / second-order summary surface as
the temporal HPP — extended with a spatial dimension axis. The operator
itself is a pure ``equinox.Module``: the rate is a real PyTree leaf and
the domain is a nested PyTree, so the whole thing differentiates and
vmaps without ceremony.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._domain import RectangularDomain
from xtremax.point_processes.primitives import (
    csr_l_function,
    csr_pair_correlation,
    csr_ripleys_k,
    hpp_spatial_count_log_prob,
    hpp_spatial_intensity,
    hpp_spatial_log_prob,
    hpp_spatial_nearest_neighbor_distance,
    hpp_spatial_predict_count,
    hpp_spatial_sample,
)


class HomogeneousSpatialPP(eqx.Module):
    """Homogeneous spatial Poisson process on a rectangular domain.

    Args:
        rate: Intensity ``λ > 0`` (points per unit volume). Stored as a
            JAX array.
        domain: Rectangular domain ``D ⊂ ℝᵈ``.
    """

    rate: Float[Array, ...]
    domain: RectangularDomain

    def __init__(self, rate: ArrayLike, domain: RectangularDomain) -> None:
        self.rate = jnp.asarray(rate)
        self.domain = domain

    @property
    def n_dims(self) -> int:
        """Spatial dimension ``d`` (forwarded from the domain)."""
        return self.domain.n_dims

    # ------------------------------------------------------------
    # Core distribution API
    # ------------------------------------------------------------

    def log_prob(
        self,
        locations: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Joint log-likelihood of an observed CSR pattern.

        Uses the iid-uniform-locations form
        ``n log λ − λ|D| − n log |D|``. ``locations`` is consumed only
        through its event count via ``mask`` — under CSR the location
        density factors through and contributes ``-n log |D|``.
        """
        del locations  # CSR locations only enter via the count
        n_events = jnp.sum(mask, axis=-1)
        return hpp_spatial_log_prob(n_events, self.rate, self.domain.volume())

    def count_log_prob(self, n_events: Int[Array, ...]) -> Float[Array, ...]:
        """Marginal Poisson-count log-pmf ``log P(N = n)``."""
        return hpp_spatial_count_log_prob(n_events, self.rate, self.domain.volume())

    def sample(
        self,
        key: PRNGKeyArray,
        max_events: int,
    ) -> tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
        """Draw a padded ``(locations, mask, n_events)`` realisation."""
        return hpp_spatial_sample(key, self.rate, self.domain, max_events)

    # ------------------------------------------------------------
    # Intensity and second-order summaries
    # ------------------------------------------------------------

    def intensity(
        self,
        locations: Float[Array, ...] | None = None,
    ) -> Float[Array, ...]:
        """Pointwise intensity (constant). ``locations`` is optional."""
        if locations is None:
            return self.rate
        return hpp_spatial_intensity(locations, self.rate)

    def ripleys_k(self, r: Float[Array, ...]) -> Float[Array, ...]:
        """Theoretical Ripley's K under CSR: :math:`V_d r^d`."""
        return csr_ripleys_k(r, self.n_dims)

    def l_function(self, r: Float[Array, ...]) -> Float[Array, ...]:
        """Besag's L-function (linearised K)."""
        return csr_l_function(r, self.n_dims)

    def pair_correlation(self, r: Float[Array, ...]) -> Float[Array, ...]:
        """Pair correlation function under CSR (constant 1)."""
        return csr_pair_correlation(r, self.n_dims)

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------

    def predict_count(
        self,
        subdomain: RectangularDomain | None = None,
    ) -> tuple[Float[Array, ...], Float[Array, ...]]:
        """Mean and variance of the count over a sub-domain (defaults to ``D``)."""
        vol = self.domain.volume() if subdomain is None else subdomain.volume()
        return hpp_spatial_predict_count(self.rate, vol)

    def predict_nearest_neighbor_distance(self) -> Float[Array, ...]:
        """Expected nearest-neighbour distance under CSR."""
        return hpp_spatial_nearest_neighbor_distance(self.rate, self.n_dims)
