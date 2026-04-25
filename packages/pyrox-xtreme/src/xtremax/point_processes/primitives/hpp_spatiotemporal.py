"""Pure-JAX primitives for the homogeneous spatiotemporal Poisson process.

A homogeneous spatiotemporal Poisson process on
:math:`D \\subset \\mathbb{R}^d` and ``[t_0, t_1)`` with intensity
``λ > 0`` has

* count :math:`N \\sim \\mathrm{Poisson}(\\lambda \\, |D| \\, (t_1 - t_0))`,
* conditional locations and times iid uniform on the slab,
* joint Janossy log-likelihood
  :math:`\\log L = n \\log \\lambda - \\lambda |D| (t_1 - t_0)`.

This module mirrors :mod:`xtremax.point_processes.primitives.hpp_spatial`
extended with one extra axis (time). Returned buffers are sorted in
time so downstream consumers can scan through events causally without
re-sorting.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._domain import RectangularDomain, TemporalDomain


def hpp_spatiotemporal_log_prob(
    n_events: Int[Array, ...],
    rate: Float[Array, ...],
    spatial_volume: Float[Array, ...],
    temporal_duration: Float[Array, ...],
) -> Float[Array, ...]:
    """Janossy log-likelihood ``n log λ − λ |D| (t1 − t0)``.

    Same convention as :func:`hpp_spatial_log_prob`: drops the
    :math:`-\\log(n!)` count normaliser and the iid-uniform location
    density so this stays directly comparable to the IPP-spatiotemporal
    log-prob (which uses the matching Janossy form).
    """
    n = jnp.asarray(n_events)
    rate = jnp.asarray(rate)
    vol = jnp.asarray(spatial_volume)
    dur = jnp.asarray(temporal_duration)
    return n * jnp.log(rate) - rate * vol * dur


def hpp_spatiotemporal_count_log_prob(
    n_events: Int[Array, ...],
    rate: Float[Array, ...],
    spatial_volume: Float[Array, ...],
    temporal_duration: Float[Array, ...],
) -> Float[Array, ...]:
    """Marginal Poisson-count log-pmf ``log P(N = n)`` with mean ``λ |D| T``."""
    n = jnp.asarray(n_events)
    lam = (
        jnp.asarray(rate) * jnp.asarray(spatial_volume) * jnp.asarray(temporal_duration)
    )
    return n * jnp.log(lam) - lam - jax.scipy.special.gammaln(n + 1)


def hpp_spatiotemporal_sample(
    key: PRNGKeyArray,
    rate: Float[Array, ...],
    spatial: RectangularDomain,
    temporal: TemporalDomain,
    max_events: int,
) -> tuple[Float[Array, ...], Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
    """Sample a homogeneous spatiotemporal Poisson realisation.

    Args:
        key: JAX PRNG key.
        rate: Intensity ``λ > 0`` (scalar). Vmap externally for batched
            rates.
        spatial: Spatial domain.
        temporal: Temporal interval.
        max_events: Static buffer cap; choose generously above
            :math:`\\lambda |D| T` (rule of thumb ``5·λ|D|T``).

    Returns:
        Tuple ``(locations, times, mask, n_events)``:

        * ``locations`` shape ``(max_events, d)`` — sorted by time,
          padding rows set to ``spatial.lo``.
        * ``times`` shape ``(max_events,)`` — sorted ascending, padding
          rows set to ``temporal.t1`` (so they live just outside the
          half-open interval but inside any quadrature support).
        * ``mask`` shape ``(max_events,)`` — ``True`` at real events.
        * ``n_events`` — uncapped Poisson draw; if greater than
          ``max_events`` the buffer is truncated.
    """
    rate = jnp.asarray(rate)
    vol = spatial.volume()
    dur = temporal.duration

    key_n, key_locs, key_times = random.split(key, 3)
    expected = rate * vol * dur
    n_uncapped = random.poisson(key_n, expected)

    ranks = jnp.arange(max_events)
    mask = ranks < n_uncapped

    raw_locs = spatial.sample_uniform(key_locs, shape=(max_events,))
    raw_times = temporal.sample_uniform(key_times, shape=(max_events,))

    # Sort by time. Push padding rows to +inf so the real events occupy
    # the head of the sort.
    sort_key = jnp.where(mask, raw_times, jnp.inf)
    order = jnp.argsort(sort_key)
    sorted_times = raw_times[order]
    sorted_locs = raw_locs[order]
    sorted_mask = mask[order]

    times = jnp.where(sorted_mask, sorted_times, temporal.t1)
    locations = jnp.where(sorted_mask[:, None], sorted_locs, spatial.lo)
    return locations, times, sorted_mask, n_uncapped


def hpp_spatiotemporal_intensity(
    locations: Float[Array, ...],
    times: Float[Array, ...],
    rate: Float[Array, ...],
) -> Float[Array, ...]:
    """Constant intensity ``λ(s, t) = λ``, broadcast against ``times``."""
    times = jnp.asarray(times)
    rate = jnp.asarray(rate)
    del locations  # the homogeneous intensity does not depend on locations
    return jnp.broadcast_to(rate, times.shape)


def hpp_spatiotemporal_predict_count(
    rate: Float[Array, ...],
    sub_volume: Float[Array, ...],
    sub_duration: Float[Array, ...],
) -> tuple[Float[Array, ...], Float[Array, ...]]:
    """Mean and variance of the count in a sub-slab (both ``λ |A| τ``)."""
    mean = jnp.asarray(rate) * jnp.asarray(sub_volume) * jnp.asarray(sub_duration)
    return mean, mean
