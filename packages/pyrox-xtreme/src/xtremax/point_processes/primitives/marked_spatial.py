"""Primitives for marked spatial point processes.

Under the separable factorisation
:math:`\\lambda(s, m) = \\lambda(s) \\, f(m \\mid s)` the joint
log-likelihood decomposes into a ground (location) component and a
mark component:

.. math::
    \\log L = \\sum_i [\\log \\lambda(s_i) + \\log f(m_i \\mid s_i)]
              - \\Lambda(D).

These primitives focus on the mark contribution conditional on
locations and delegate the ground term to
:func:`~xtremax.point_processes.primitives.ipp_spatial.ipp_spatial_log_prob`.
The mark distribution is supplied as a callable
``mark_distribution_fn(s) -> numpyro.Distribution`` so users can
condition on location (e.g. magnitude that scales with risk gradient).
For location-independent marks pass a function that ignores its input.

Shape contract:
    The primitives are written for **a single event sequence** —
    ``locations`` shape ``(max_events, d)``, ``marks`` shape
    ``(max_events,)`` or ``(max_events, mark_dim)``, ``mask`` shape
    ``(max_events,)``. For batched usage (multiple sequences) ``vmap``
    at the operator or distribution layer rather than passing batched
    arrays here directly — ``mark_distribution_fn`` typically expects
    a single location vector and would receive a batched one if we
    vmapped over a leading axis.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from jaxtyping import Array, Bool, Float, PRNGKeyArray


def _safe_padding_mark(d: dist.Distribution, m_i: Array) -> Array:
    """Return a value inside ``d``'s support, dtype-matching ``m_i``.

    Uses ``d.support.feasible_like(m_i)`` when available — NumPyro's
    ``Constraint`` API ships this for every standard support
    (positive, unit interval, real, simplex, integer interval, ...) —
    and produces a value that is *strictly* inside the support and
    of the same dtype as the prototype, so it works for both
    continuous and discrete marks.

    Falls back to ``ones_like(m_i)`` when the constraint lacks
    ``feasible_like``. Custom user constraints that don't implement
    it must therefore have ``1`` (cast to the mark dtype) inside
    their support, or the caller must supply marks with safe padding
    already in place.
    """
    fl = getattr(d.support, "feasible_like", None)
    if fl is None:
        return jnp.ones_like(m_i)
    return fl(m_i)


def spatial_marks_log_prob(
    locations: Float[Array, ...],
    marks: Float[Array, ...],
    mask: Bool[Array, ...],
    mark_distribution_fn: Callable[[Array], dist.Distribution],
) -> Float[Array, ...]:
    """Sum of ``log f(mᵢ | sᵢ)`` over real events.

    Args:
        locations: Padded locations ``(max_events, d)``.
        marks: Padded marks. Shape ``(max_events,)`` for scalar
            marks or ``(max_events, mark_dim)`` for vector marks.
        mask: Event mask shape ``(max_events,)``.
        mark_distribution_fn: Callable ``s -> Distribution``. The
            distribution must have ``event_shape == ()`` for scalar
            marks or ``event_shape == (mark_dim,)`` for vector marks.

    Returns:
        Scalar log-likelihood contribution from marks.

    Notes:
        ``log_prob`` is still called at padding positions for shape
        stability, but padding marks are first replaced with a value
        inside the distribution's support
        (:func:`_safe_padding_mark`). Without this, distributions with
        bounded support produce ``-inf`` log-probs and NaN gradients
        even though the result is masked out — JAX's autodiff threads
        through both branches of ``jnp.where``.
    """
    locations = jnp.asarray(locations)
    marks = jnp.asarray(marks)

    def per_event(s_i: Array, m_i: Array, valid: Array) -> Array:
        d = mark_distribution_fn(s_i)
        # Replace padding mark with a value inside ``d``'s support so
        # ``log_prob`` returns finite values (and finite gradients);
        # the result is then masked out below.
        safe_m = jnp.where(valid, m_i, _safe_padding_mark(d, m_i))
        return d.log_prob(safe_m)

    # Vmap over the leading event axis (the only batched axis of these
    # 1-D / 2-D arrays per the shape contract documented at the
    # module level).
    per_event_log_prob = jax.vmap(per_event)(locations, marks, mask)
    while per_event_log_prob.ndim > mask.ndim:
        per_event_log_prob = jnp.sum(per_event_log_prob, axis=-1)
    return jnp.sum(jnp.where(mask, per_event_log_prob, 0.0), axis=-1)


def sample_spatial_marks_at_locations(
    key: PRNGKeyArray,
    locations: Float[Array, ...],
    mask: Bool[Array, ...],
    mark_distribution_fn: Callable[[Array], dist.Distribution],
) -> Float[Array, ...]:
    """Draw marks ``mᵢ ~ f(· | sᵢ)`` at every location.

    Independent across events, so a single ``vmap`` over PRNG splits is
    sufficient — no scan is needed (unlike the temporal marked PP,
    which threads history through each step).

    Args:
        key: JAX PRNG key.
        locations: Padded locations ``(max_events, d)``.
        mask: Event mask ``(max_events,)``.
        mark_distribution_fn: Mark distribution callable.

    Returns:
        Marks aligned with ``locations``. Padding positions are filled
        with a dtype-preserving zero (`zeros_like(marks)`) so discrete
        / integer marks keep their dtype. Scalar marks have shape
        ``(max_events,)``; vector marks have shape ``(max_events, mark_dim)``.
    """
    n_max = locations.shape[-2]
    keys = random.split(key, n_max)

    def per_event(s_i: Array, k: PRNGKeyArray) -> Array:
        d = mark_distribution_fn(s_i)
        return d.sample(k)

    marks = jax.vmap(per_event)(locations, keys)
    # ``zeros_like(marks)`` preserves the mark dtype (important for
    # discrete distributions like Categorical / Poisson that produce
    # integer arrays, which a literal ``0.0`` would silently upcast).
    zero = jnp.zeros_like(marks)
    if marks.ndim == mask.ndim:
        return jnp.where(mask, marks, zero)
    return jnp.where(mask[..., None], marks, zero)
