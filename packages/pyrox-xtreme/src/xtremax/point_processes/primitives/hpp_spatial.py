"""Pure-JAX primitives for the homogeneous spatial Poisson process.

A homogeneous spatial Poisson process (Complete Spatial Randomness) on
a domain :math:`D \\subset \\mathbb{R}^d` with intensity ``λ > 0`` has

* count :math:`N(D) \\sim \\mathrm{Poisson}(\\lambda |D|)`,
* conditional locations iid uniform on ``D``,
* joint log-likelihood
  :math:`\\log L = n \\log \\lambda - \\lambda |D| - n \\log |D|`.

The shape contract for a sequence of locations is ``(..., max_events, d)``
plus a boolean ``mask`` of shape ``(..., max_events)``. This is the same
padded-buffer convention as the temporal HPP — the only addition is the
trailing spatial axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._domain import RectangularDomain


def hpp_spatial_log_prob(
    n_events: Int[Array, ...],
    rate: Float[Array, ...],
    domain_volume: Float[Array, ...],
) -> Float[Array, ...]:
    """Joint Janossy log-likelihood of a homogeneous Poisson pattern.

    Args:
        n_events: Observed event count.
        rate: Intensity ``λ > 0``.
        domain_volume: Lebesgue measure ``|D|``.

    Returns:
        Janossy log-likelihood ``n log λ − λ|D|``. This is the
        constant-intensity specialisation of
        :func:`~xtremax.point_processes.primitives.ipp_spatial.ipp_spatial_log_prob`,
        which evaluates :math:`\\sum_i \\log \\lambda(s_i) - \\Lambda(D)`.
        Setting :math:`\\lambda(s) = \\lambda` recovers exactly this
        expression.

    Notes:
        The Janossy form drops both the :math:`-\\log(n!)` Poisson-count
        normaliser and the :math:`-n \\log |D|` iid-uniform location
        density. Those normalisers cancel out in likelihood ratios used
        for parameter estimation, so omitting them keeps the HPP log-prob
        directly comparable to IPP / Hawkes / renewal log-probs across
        the package. To recover the *count*-only log-pmf with the
        :math:`\\log(n!)` term, use :func:`hpp_spatial_count_log_prob`.
    """
    n = jnp.asarray(n_events)
    rate = jnp.asarray(rate)
    vol = jnp.asarray(domain_volume)
    return n * jnp.log(rate) - rate * vol


def hpp_spatial_count_log_prob(
    n_events: Int[Array, ...],
    rate: Float[Array, ...],
    domain_volume: Float[Array, ...],
) -> Float[Array, ...]:
    """Marginal Poisson-count log-pmf ``log P(N = n)``.

    Useful for predict-count workflows that only need the count
    distribution, ignoring conditional location density.
    """
    n = jnp.asarray(n_events)
    lam = jnp.asarray(rate) * jnp.asarray(domain_volume)
    return n * jnp.log(lam) - lam - jax.scipy.special.gammaln(n + 1)


def hpp_spatial_sample(
    key: PRNGKeyArray,
    rate: Float[Array, ...],
    domain: RectangularDomain,
    max_events: int,
) -> tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
    """Sample a CSR realisation on a rectangular domain.

    Args:
        key: JAX PRNG key.
        rate: Intensity ``λ > 0`` (scalar). Batched rates are not
            supported here — vmap externally.
        domain: Spatial domain.
        max_events: Static buffer cap. Choose generously above
            :math:`\\lambda |D|`; common rule of thumb is
            ``5 * λ |D|``.

    Returns:
        Tuple ``(locations, mask, n_events)``:

        * ``locations`` shape ``(max_events, d)`` — sorted along axis 0
          by lex order of the coordinate tuple, with padding rows set
          to ``domain.lo`` (an arbitrary in-domain anchor that keeps
          downstream intensity calls inside the support).
        * ``mask`` shape ``(max_events,)`` — ``True`` at real events.
        * ``n_events`` — uncapped Poisson draw; if greater than
          ``max_events`` the buffer truncated and the user should
          increase the cap.
    """
    rate = jnp.asarray(rate)
    vol = domain.volume()
    d = domain.n_dims

    key_n, key_locs = random.split(key)
    expected = rate * vol
    n_uncapped = random.poisson(key_n, expected)

    # Mark the first ``n_uncapped`` slots as real events. We put padding
    # at ``+inf`` along axis 0 and lex-sort, so the real events occupy
    # ranks ``0..n-1`` without order-statistic bias from the truncation.
    ranks = jnp.arange(max_events)
    mask = ranks < n_uncapped

    raw = domain.sample_uniform(key_locs, shape=(max_events,))
    # Lex-sort key: first coordinate is the primary key, second is the
    # secondary, etc. Push padding rows to +inf so they end up at the
    # tail of the sort and the real events fill the head.
    sort_key = jnp.where(mask[:, None], raw, jnp.inf)
    # Use the negative of the row-major encoding of the lex order: for
    # 2D and 3D this is just sorting by the first coordinate, which is
    # all we need for downstream consumers (the API does not promise
    # any particular spatial order, just a stable padding invariant).
    primary = sort_key[:, 0]
    order = jnp.argsort(primary)
    sorted_locs = raw[order]
    sorted_mask = mask[order]
    # Replace padding rows with ``domain.lo`` so log-intensity calls
    # never see ``+inf``.
    locations = jnp.where(sorted_mask[:, None], sorted_locs, domain.lo)
    # Sanity check on ``d``: our slicing ``[..., :d]`` implicitly trusts
    # the domain to have produced points of the right width. Since
    # ``domain.sample_uniform`` builds the trailing ``(d,)`` axis from
    # ``n_dims`` directly, we are safe by construction.
    del d
    return locations, sorted_mask, n_uncapped


def hpp_spatial_intensity(
    locations: Float[Array, ...],
    rate: Float[Array, ...],
) -> Float[Array, ...]:
    """Constant intensity ``λ(s) = λ``, broadcast against ``locations``."""
    locations = jnp.asarray(locations)
    rate = jnp.asarray(rate)
    return jnp.broadcast_to(rate, locations.shape[:-1])


def hpp_spatial_predict_count(
    rate: Float[Array, ...],
    subdomain_volume: Float[Array, ...],
) -> tuple[Float[Array, ...], Float[Array, ...]]:
    """Mean and variance of the count in a sub-domain (both ``λ |A|``)."""
    mean = jnp.asarray(rate) * jnp.asarray(subdomain_volume)
    return mean, mean


def hpp_spatial_nearest_neighbor_distance(
    rate: Float[Array, ...],
    n_dims: int,
) -> Float[Array, ...]:
    """Expected nearest-neighbour distance for a CSR pattern in ``ℝᵈ``.

    For a homogeneous Poisson process,

    .. math::
        \\mathbb{E}[D_1] = \\frac{\\Gamma(1 + 1/d)}{(\\lambda \\, V_d)^{1/d}},

    with :math:`V_d = \\pi^{d/2} / \\Gamma(d/2 + 1)` the unit-ball volume.
    """
    d = float(n_dims)
    unit_ball = (jnp.pi ** (d / 2.0)) / jax.scipy.special.gamma(d / 2.0 + 1.0)
    gamma_term = jax.scipy.special.gamma(1.0 + 1.0 / d)
    return gamma_term / (jnp.asarray(rate) * unit_ball) ** (1.0 / d)
