"""Primitives for marked spatiotemporal point processes.

Under the separable factorisation
:math:`\\lambda(s, t, m) = \\lambda(s, t)\\, f(m \\mid s, t)` the joint
log-likelihood decomposes into a ground (location, time) component and
a mark component:

.. math::
    \\log L = \\sum_i [\\log \\lambda(s_i, t_i) + \\log f(m_i \\mid s_i, t_i)]
              - \\Lambda.

These primitives focus on the mark contribution conditional on
``(s, t)`` and delegate the ground term to
:func:`~xtremax.point_processes.primitives.ipp_spatiotemporal.ipp_spatiotemporal_log_prob`.
The mark distribution is supplied as a callable
``mark_distribution_fn(s, t) -> numpyro.Distribution`` so users can
condition on either or both axes (e.g. earthquake magnitudes that
follow Gutenberg–Richter with epicenter-dependent ``b``-values, or
disease severity that scales with calendar time).

The shape contract matches :mod:`marked_spatial`: a single event
sequence per call. Vmap at the operator/distribution layer for
batched usage.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from xtremax.point_processes.primitives.marked_spatial import _safe_padding_mark


def spatiotemporal_marks_log_prob(
    locations: Float[Array, ...],
    times: Float[Array, ...],
    marks: Float[Array, ...],
    mask: Bool[Array, ...],
    mark_distribution_fn: Callable[[Array, Array], dist.Distribution],
) -> Float[Array, ...]:
    """Sum of :math:`\\log f(m_i \\mid s_i, t_i)` over real events.

    Args:
        locations: ``(max_events, d_space)`` event locations.
        times: ``(max_events,)`` event times.
        marks: ``(max_events,)`` or ``(max_events, mark_dim)`` marks.
        mask: ``(max_events,)`` boolean event mask.
        mark_distribution_fn: Callable ``(s, t) -> Distribution``.

    Returns:
        Scalar log-likelihood contribution from marks. Padding rows are
        substituted with a value inside the distribution's support
        before ``log_prob`` is called, so distributions with bounded
        support do not produce NaN gradients.
    """
    locations = jnp.asarray(locations)
    times = jnp.asarray(times)
    marks = jnp.asarray(marks)

    def per_event(s_i: Array, t_i: Array, m_i: Array, valid: Array) -> Array:
        d = mark_distribution_fn(s_i, t_i)
        safe_m = jnp.where(valid, m_i, _safe_padding_mark(d, m_i))
        return d.log_prob(safe_m)

    per_event_log_prob = jax.vmap(per_event)(locations, times, marks, mask)
    while per_event_log_prob.ndim > mask.ndim:
        per_event_log_prob = jnp.sum(per_event_log_prob, axis=-1)
    return jnp.sum(jnp.where(mask, per_event_log_prob, 0.0), axis=-1)


def sample_spatiotemporal_marks(
    key: PRNGKeyArray,
    locations: Float[Array, ...],
    times: Float[Array, ...],
    mask: Bool[Array, ...],
    mark_distribution_fn: Callable[[Array, Array], dist.Distribution],
) -> Float[Array, ...]:
    """Draw marks :math:`m_i \\sim f(\\cdot \\mid s_i, t_i)` at every event.

    Independent across events, so a single ``vmap`` over PRNG splits
    suffices.

    Args:
        key: JAX PRNG key.
        locations: ``(max_events, d_space)`` locations.
        times: ``(max_events,)`` times.
        mask: ``(max_events,)`` event mask.
        mark_distribution_fn: Callable ``(s, t) -> Distribution``.

    Returns:
        Marks with the same leading shape as ``times``. Padding rows
        are filled with ``zeros_like(marks)`` to preserve the mark
        dtype (important for discrete distributions).
    """
    n_max = times.shape[-1]
    keys = random.split(key, n_max)

    def per_event(s_i: Array, t_i: Array, k: PRNGKeyArray) -> Array:
        d = mark_distribution_fn(s_i, t_i)
        return d.sample(k)

    marks = jax.vmap(per_event)(locations, times, keys)
    zero = jnp.zeros_like(marks)
    if marks.ndim == mask.ndim:
        return jnp.where(mask, marks, zero)
    return jnp.where(mask[..., None], marks, zero)
