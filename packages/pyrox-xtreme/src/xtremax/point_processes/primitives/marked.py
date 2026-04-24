"""Primitives for marked temporal point processes.

Separable factorisation
:math:`\\lambda(t, m) = \\lambda_g(t) \\, f(m \\mid t, \\mathcal{H})`
with a user-supplied mark distribution ``mark_distribution_fn`` that
returns a NumPyro ``Distribution`` conditioned on time and (optionally)
history. Under separability the joint log-likelihood decomposes into
the ground-process log-likelihood plus a sum of mark log-densities,
so these primitives focus on the mark contribution and leave the
ground-process term to whichever family (IPP, Hawkes, renewal) the
caller is using.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._history import EventHistory


def marks_log_prob(
    event_times: Float[Array, ...],
    marks: Float[Array, ...],
    mask: Bool[Array, ...],
    mark_distribution_fn: Callable[[Array, EventHistory], dist.Distribution],
    history_at_each_event: bool = True,
) -> Float[Array, ...]:
    r"""Sum of ``log f(mᵢ | tᵢ, Hᵢ)`` over real events.

    Args:
        event_times: Padded event times ``(max_events,)``.
        marks: Padded marks aligned with ``event_times``. Shape
            ``(max_events,)`` for scalar marks, ``(max_events, d)`` for
            ``d``-dimensional marks.
        mask: Event mask.
        mark_distribution_fn: Callable ``(t, history) -> Distribution``.
        history_at_each_event: When ``True`` the history passed to
            ``mark_distribution_fn(tᵢ, Hᵢ)`` contains events
            ``{t_j : j < i}``; when ``False`` (the default used by
            callers that do not need prior-event conditioning) it is
            the full observed history — this saves the per-event
            vmap but is only correct for history-independent mark
            distributions.

    Returns:
        Scalar log-likelihood contribution from the marks.

    Notes:
        At padding positions ``mark_distribution_fn`` is still called
        (to keep shapes static) but the log-prob is masked out.
    """
    event_times = jnp.asarray(event_times)
    marks = jnp.asarray(marks)

    if not history_at_each_event:
        full_history = EventHistory(
            times=event_times, mask=mask, marks=marks if marks.ndim == 2 else None
        )

        def per_event(i: Int[Array, ...]) -> Float[Array, ...]:
            t_i = event_times[i]
            d = mark_distribution_fn(t_i, full_history)
            mark_i = marks[i]
            return d.log_prob(mark_i)

    else:
        n_max = event_times.shape[-1]
        idx_grid = jnp.arange(n_max)
        # A per-event history uses mask[..., j] & (j < i).
        marks_2d = marks if marks.ndim == 2 else marks[..., None]

        def per_event(i: Int[Array, ...]) -> Float[Array, ...]:
            before_i = (idx_grid < i) & mask
            h_i = EventHistory(
                times=jnp.where(before_i, event_times, 0.0),
                mask=before_i,
                marks=jnp.where(before_i[..., None], marks_2d, 0.0),
            )
            d = mark_distribution_fn(event_times[i], h_i)
            mark_i = marks[i]
            return d.log_prob(mark_i)

    idx = jnp.arange(event_times.shape[-1])
    per_event_log_prob = jax.vmap(per_event)(idx)
    # Mark distributions on multi-dim marks return per-draw scalars, on
    # batched marks they return per-component values — sum any trailing dims.
    while per_event_log_prob.ndim > mask.ndim:
        per_event_log_prob = jnp.sum(per_event_log_prob, axis=-1)
    return jnp.sum(jnp.where(mask, per_event_log_prob, 0.0), axis=-1)


def sample_marks_at_times(
    key: PRNGKeyArray,
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    mark_distribution_fn: Callable[[Array, EventHistory], dist.Distribution],
    history_at_each_event: bool = True,
) -> Float[Array, ...]:
    """Draw marks ``mᵢ ~ f(· | tᵢ, Hᵢ)`` at each observed time.

    When ``history_at_each_event=True`` sampling is sequential: each
    draw's history pytree includes the previously drawn marks, so mark
    distributions can legitimately depend on both past times and past
    marks. When ``False`` a single ``vmap`` fans the draws out in
    parallel and the mark distribution is evaluated against the full
    observed history with ``marks=None`` — cheaper, but only correct
    for mark laws that do not look at prior marks.

    Args:
        key: JAX PRNG key.
        event_times: Padded event times.
        mask: Event mask.
        mark_distribution_fn: Mark distribution callable.
        history_at_each_event: See above.

    Returns:
        Array of marks shaped ``event_times.shape`` for scalar marks or
        ``event_times.shape + (mark_dim,)`` for vector marks, with
        padding positions zeroed out.
    """
    n_max = event_times.shape[-1]
    keys = random.split(key, n_max)

    if not history_at_each_event:
        full_history = EventHistory(times=event_times, mask=mask, marks=None)

        def per_event(i: Int[Array, ...], k: PRNGKeyArray) -> Array:
            d = mark_distribution_fn(event_times[i], full_history)
            return d.sample(k)

        idx = jnp.arange(n_max)
        marks = jax.vmap(per_event)(idx, keys)
        if marks.ndim == mask.ndim:
            return jnp.where(mask, marks, 0.0)
        return jnp.where(mask[..., None], marks, 0.0)

    # Sequential path: probe the mark dist once on an empty history so
    # we know the mark shape/dtype, preallocate a 2-D marks buffer on
    # the carry (scalar marks get a trailing ``1`` to keep the buffer
    # uniformly 2-D), and scan. Each step appends the drawn mark to the
    # carry so subsequent steps can condition on ``history.marks``.
    empty_history = EventHistory.empty(max_events=n_max, dtype=event_times.dtype)
    probe_spec = jax.eval_shape(
        lambda et, k: mark_distribution_fn(et, empty_history).sample(k),
        event_times[0],
        keys[0],
    )
    mark_event_shape = probe_spec.shape
    mark_dtype = probe_spec.dtype
    is_scalar_mark = len(mark_event_shape) == 0
    mark_dim = 1 if is_scalar_mark else mark_event_shape[-1]
    initial_carry = EventHistory(
        times=jnp.zeros_like(event_times),
        mask=jnp.zeros_like(mask),
        marks=jnp.zeros((n_max, mark_dim), dtype=mark_dtype),
    )

    def step(
        carry: EventHistory, inp: tuple[Int[Array, ...], PRNGKeyArray]
    ) -> tuple[EventHistory, Array]:
        i, k = inp
        d = mark_distribution_fn(event_times[i], carry)
        m = d.sample(k)
        m_row = jnp.expand_dims(m, axis=-1) if is_scalar_mark else m

        valid_i = mask[i]
        new_times = carry.times.at[i].set(
            jnp.where(valid_i, event_times[i], carry.times[i])
        )
        new_mask = carry.mask.at[i].set(
            jnp.where(valid_i, jnp.bool_(True), carry.mask[i])
        )
        zero_row = jnp.zeros_like(m_row)
        new_marks_row = jnp.where(valid_i, m_row, zero_row)
        new_marks = carry.marks.at[i].set(new_marks_row)
        new_carry = EventHistory(times=new_times, mask=new_mask, marks=new_marks)

        # Emitted per-step mark (in the user's original scalar/vector shape).
        emitted = jnp.where(valid_i, m, jnp.zeros_like(m))
        return new_carry, emitted

    idx = jnp.arange(n_max)
    _, marks = jax.lax.scan(step, initial_carry, (idx, keys))
    return marks
