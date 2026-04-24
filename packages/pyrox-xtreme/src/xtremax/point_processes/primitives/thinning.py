"""Generic Ogata thinning sampler and retention primitives.

The Lewis–Shedler sampler already in :mod:`~xtremax.point_processes.primitives.ipp`
is specialised to the Inhomogeneous Poisson case with a *global* upper
bound :math:`\\lambda_{\\max}` over the whole window. For self-exciting
processes (Hawkes) and for thinning-of-base constructions, the upper
bound depends on the history accumulated so far. Ogata (1981) handles
this with an adaptive-envelope variant:

1. Start with an empty history and :math:`t = 0`.
2. Compute a history-dependent upper bound :math:`\\bar\\lambda`.
3. Draw an inter-arrival :math:`\\Delta t \\sim \\mathrm{Exp}(\\bar\\lambda)`;
   set :math:`t \\leftarrow t + \\Delta t`.
4. Accept :math:`t` with probability
   :math:`\\lambda^*(t \\mid H) / \\bar\\lambda`; append to history if so.
5. Repeat until :math:`t \\geq T` or the proposal buffer is exhausted.

:func:`thinning_sample` is a generic implementation: it takes a
``conditional_intensity_fn`` and a ``lambda_max_fn`` that both receive
the current :class:`EventHistory`. Hawkes, ThinningProcess, and any
future history-dependent family can delegate to it.

:func:`retention_compensator` computes
:math:`\\int_0^T (1 - p(t|H(t)))\\lambda(t|H(t))\\, dt` used by
thinning-of-base log-likelihoods.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._history import EventHistory


def thinning_sample(
    key: PRNGKeyArray,
    conditional_intensity_fn: Callable[[Array, EventHistory], Array],
    lambda_max_fn: Callable[[EventHistory], Array],
    T: Float[Array, ...],
    max_events: int,
    max_candidates: int | None = None,
    initial_history: EventHistory | None = None,
) -> tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
    """Sample a history-dependent temporal point process via Ogata thinning.

    Args:
        key: JAX PRNG key.
        conditional_intensity_fn: Callable ``(t, history) -> λ*(t | H)``.
            ``history`` is the :class:`EventHistory` accumulated *before*
            ``t``; callers must not peek at anything at or after ``t``.
        lambda_max_fn: Callable ``history -> λ_bar``. Must return a
            scalar ``>=`` the supremum of ``conditional_intensity_fn``
            over the remaining window ``[last_time, T]`` given the
            current history. Over-estimates are safe (lose efficiency);
            under-estimates break the sampler.
        T: Observation window length (scalar).
        max_events: Static cap on the number of accepted events.
        max_candidates: Static cap on the number of proposals. Defaults
            to ``4 * max_events`` if ``None``. Choose comfortably above
            the expected count to avoid truncation under low acceptance.
        initial_history: Optional starting history. Defaults to an
            empty buffer of size ``max_events``.

    Returns:
        Tuple ``(times, mask, n_proposals)``:

        * ``times`` with shape ``(max_events,)`` — sorted, padding
          positions filled with ``T``.
        * ``mask`` — ``True`` at accepted positions.
        * ``n_proposals`` — number of thinning proposals actually
          consumed (useful for diagnosing buffer over-runs).
    """
    T_arr = jnp.asarray(T)
    if max_candidates is None:
        max_candidates = 4 * max_events

    if initial_history is None:
        initial_history = EventHistory.empty(max_events=max_events, dtype=T_arr.dtype)

    keys = random.split(key, max_candidates)

    def step(
        carry: tuple[Float[Array, ...], EventHistory, Int[Array, ...]],
        key_i: PRNGKeyArray,
    ) -> tuple[tuple[Float[Array, ...], EventHistory, Int[Array, ...]], None]:
        t_prev, history, consumed = carry
        key_dt, key_u = random.split(key_i)

        # Guard λ_bar against 0 so ``dt`` can never be ``+inf``; a zero
        # upper bound is only reachable when the user supplies an
        # identically-zero process, in which case a tiny effective rate
        # still overshoots ``T`` on the first step and terminates.
        lambda_bar = jnp.maximum(lambda_max_fn(history), 1e-30)
        # Exp(λ_bar) draw with a log1p-safe form (``u`` is in (0, 1)).
        # Clip both ends away from 0 / 1: ``u == 0`` would give
        # ``dt == 0`` (duplicate times), and ``u == 1`` gives ``-inf``.
        u1 = random.uniform(key_dt, dtype=T_arr.dtype)
        u1 = jnp.clip(u1, 1e-7, 1.0 - 1e-7)
        dt = -jnp.log1p(-u1) / lambda_bar
        t_new = t_prev + dt

        in_window = t_new < T_arr
        lambda_new = conditional_intensity_fn(t_new, history)
        u2 = random.uniform(key_u, dtype=T_arr.dtype)
        accepted = in_window & (u2 * lambda_bar < lambda_new)

        new_history = history.append(t_new, accepted=accepted)
        # Always advance to the new proposal; once past ``T`` subsequent
        # proposals stay outside the window and ``in_window`` stays
        # ``False``, naturally stopping acceptance. Resetting to
        # ``t_prev`` on overshoot would let the sampler re-propose into
        # the window and double-count events near the right edge.
        t_out = t_new
        # Consumed counts proposals actually processed (i.e. within [0, T]).
        new_consumed = consumed + in_window.astype(consumed.dtype)
        return (t_out, new_history, new_consumed), None

    initial = (
        jnp.asarray(0.0, dtype=T_arr.dtype),
        initial_history,
        jnp.asarray(0, dtype=jnp.int32),
    )
    (_, final_history, n_proposals), _ = jax.lax.scan(step, initial, keys)

    # Final pass: pad padding positions with T so downstream intensity
    # evaluations stay inside the window.
    times = jnp.where(final_history.mask, final_history.times, T_arr)
    return times, final_history.mask, n_proposals


def _prefix_history(history: EventHistory, keep: Bool[Array, ...]) -> EventHistory:
    """Return a copy of ``history`` with ``mask`` restricted to ``keep``.

    Used by the retention primitives to evaluate callables against a
    "history up to t" view without copying the underlying arrays. The
    times and marks buffers themselves are zeroed at filtered-out
    positions so a retention callable that accidentally looks past the
    mask does not read stale data.
    """
    new_mask = history.mask & keep
    new_times = jnp.where(new_mask, history.times, jnp.zeros_like(history.times))
    if history.marks is None:
        new_marks = None
    else:
        mask_broadcast = new_mask[..., None] if history.marks.ndim == 2 else new_mask
        new_marks = jnp.where(
            mask_broadcast, history.marks, jnp.zeros_like(history.marks)
        )
    return EventHistory(times=new_times, mask=new_mask, marks=new_marks)


def retention_compensator(
    retention_fn: Callable[[Array, EventHistory, Array | None], Array],
    conditional_intensity_fn: Callable[[Array, EventHistory], Array],
    history: EventHistory,
    T: Float[Array, ...],
    n_points: int = 100,
    mark_sampler_fn: Callable[[Array, EventHistory, PRNGKeyArray], Array] | None = None,
    mark_sample_key: PRNGKeyArray | None = None,
    n_mark_samples: int = 8,
) -> Float[Array, ...]:
    r"""Compute the retention compensator via trapezoid quadrature.

    :math:`\int_0^T (1 - \bar p(t|H(t)))\,\lambda_g(t|H(t))\, dt`.

    Used by :class:`~xtremax.point_processes.operators.ThinningProcess`
    to correct the thinned-process log-likelihood by the expected
    compensator of the *discarded* events. At each quadrature node
    ``t`` the retention and intensity callables see only the observed
    events with ``time < t``, so a history-dependent retention never
    "peeks" at events in the future of its query point.

    For **mark-independent** retention (the common case), ``retention_fn``
    is called as ``retention_fn(t, prefix, None)`` and :math:`\bar p = p`.
    For **mark-dependent** retention, pass a ``mark_sampler_fn`` that
    draws one mark from the base's mark distribution given
    ``(t, prefix, key)``. The retention is then averaged over
    ``n_mark_samples`` such marks per grid point, yielding a Monte-Carlo
    estimate of
    :math:`\bar p(t|H) = \mathbb{E}_{m \sim f(\cdot|t,H)}[p(t|H,m)]`.
    Without ``mark_sampler_fn``, mark-dependent retention silently
    receives ``None``; callers who need the mark-averaged correction
    must supply the sampler explicitly.

    Args:
        retention_fn: Retention callable.
        conditional_intensity_fn: Intensity callable (the **ground**
            intensity when the base is a marked process — see the
            operator layer).
        history: Observed (retained) history. Filtered per-``t`` to the
            prefix ``{i : times[i] < t}`` inside this routine.
        T: Window upper limit.
        n_points: Trapezoid grid size (static Python int).
        mark_sampler_fn: Optional ``(t, prefix, key) -> mark`` for MC
            marginalisation over marks in mark-dependent retention.
        mark_sample_key: JAX key used for mark MC; required when
            ``mark_sampler_fn`` is set.
        n_mark_samples: Number of mark MC samples per grid point.

    Returns:
        Scalar integral.
    """
    T_arr = jnp.asarray(T)
    grid = jnp.linspace(jnp.zeros_like(T_arr), T_arr, n_points)

    if mark_sampler_fn is None:

        def integrand(t: Array) -> Array:
            prefix = _prefix_history(history, history.times < t)
            # Clip ``p`` into [0, 1] so a badly-behaved retention (e.g.
            # numerical over/undershoot at grid points) cannot flip the
            # compensator sign.
            p = jnp.clip(retention_fn(t, prefix, None), 0.0, 1.0)
            lam = conditional_intensity_fn(t, prefix)
            return (1.0 - p) * lam

        values = jax.vmap(integrand)(grid)
        return jnp.trapezoid(values, grid)

    if mark_sample_key is None:
        raise ValueError(
            "retention_compensator: mark_sampler_fn requires mark_sample_key."
        )
    per_t_keys = random.split(mark_sample_key, n_points)

    def integrand_mc(t: Array, key_t: PRNGKeyArray) -> Array:
        prefix = _prefix_history(history, history.times < t)
        mark_keys = random.split(key_t, n_mark_samples)

        def one_sample(k: PRNGKeyArray) -> Array:
            m = mark_sampler_fn(t, prefix, k)
            return jnp.clip(retention_fn(t, prefix, m), 0.0, 1.0)

        p_bar = jnp.mean(jax.vmap(one_sample)(mark_keys))
        lam = conditional_intensity_fn(t, prefix)
        return (1.0 - p_bar) * lam

    values = jax.vmap(integrand_mc)(grid, per_t_keys)
    return jnp.trapezoid(values, grid)


def thinning_retention_log_prob(
    retention_fn: Callable[[Array, EventHistory, Array | None], Array],
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    history: EventHistory,
    marks: Float[Array, ...] | None = None,
) -> Float[Array, ...]:
    """Sum of ``log p(tᵢ | Hᵢ, mᵢ)`` over retained events.

    At each event index ``i`` the retention callable sees only events
    ``{j : j < i}`` in the history — so the contract
    :math:`p(t_i \\mid \\mathcal{H}_i, m_i)` is honoured exactly, even
    for history-dependent retention.

    Args:
        retention_fn: Retention callable.
        event_times: Observed times (padded).
        mask: Observed mask.
        history: The full observed history. Restricted per-event to the
            prefix ``{j : j < i}`` inside this routine.
        marks: Optional marks aligned with ``event_times``; passed to
            ``retention_fn`` as ``proposed_mark`` at each position.

    Returns:
        Scalar sum; padding positions contribute ``0`` via the mask.
    """
    n = event_times.shape[-1]
    index_grid = jnp.arange(n)

    def eval_at(i: Int[Array, ...]) -> Float[Array, ...]:
        prefix = _prefix_history(history, index_grid < i)
        t_i = event_times[i]
        mark_i = None if marks is None else marks[i]
        return retention_fn(t_i, prefix, mark_i)

    probs = jax.vmap(eval_at)(index_grid)
    log_p = jnp.log(jnp.clip(probs, 1e-30, 1.0))
    return jnp.sum(jnp.where(mask, log_p, 0.0), axis=-1)
