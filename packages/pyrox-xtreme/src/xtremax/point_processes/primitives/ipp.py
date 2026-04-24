"""Pure-JAX primitives for the Inhomogeneous Poisson Process on ``[0, T]``.

An IPP with intensity function :math:`\\lambda(t) \\geq 0` on ``[0, T]``
has compensator :math:`\\Lambda(t) = \\int_0^t \\lambda(s) ds`, count
:math:`N \\sim \\mathrm{Poisson}(\\Lambda(T))`, and log-likelihood

.. math::
    \\log L = \\sum_{i=1}^n \\log \\lambda(t_i) - \\Lambda(T).

The intensity is always provided in log-space (``log_intensity_fn``) —
neural intensities are naturally parameterised as log-intensities, and
:math:`\\log \\lambda(t_i)` is what the likelihood actually needs.

All functions here are pure. Variable-length event sequences are
represented as padded ``times`` plus a boolean ``mask`` to keep shapes
static under ``jax.jit``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._integration import (
    cumulative_log_intensity,
    integrate_log_intensity,
)


def ipp_log_prob(
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    log_intensity_fn: Callable[[Array], Array],
    integrated_intensity: Float[Array, ...],
) -> Float[Array, ...]:
    """Log-likelihood of an observed IPP event sequence.

    Args:
        event_times: Padded array of event times. Padding positions are
            evaluated by ``log_intensity_fn`` but masked out of the sum
            — choose padding values inside the support of the
            log-intensity so evaluation does not raise.
        mask: Boolean mask with the same shape as ``event_times``;
            ``True`` marks real events.
        log_intensity_fn: Callable returning ``log λ(t)`` for an array
            of times. Must broadcast element-wise.
        integrated_intensity: :math:`\\Lambda(T)`. The caller is
            responsible for computing this (via
            :func:`integrate_log_intensity` for numerical quadrature,
            or analytically when a closed form is available).

    Returns:
        Log-likelihood with shape equal to the leading (batch) dims of
        ``event_times`` (i.e. ``mask.shape[:-1]``).

    Notes:
        Padding positions contribute ``0`` to the sum because ``mask``
        zeros them out. Keeping the log-intensity call unmasked (rather
        than gathering only real events) preserves ``jit`` / ``vmap``
        compatibility with static shapes.
    """
    log_intensities = log_intensity_fn(event_times)
    masked = jnp.where(mask, log_intensities, 0.0)
    return jnp.sum(masked, axis=-1) - jnp.asarray(integrated_intensity)


def ipp_sample_thinning(
    key: PRNGKeyArray,
    log_intensity_fn: Callable[[Array], Array],
    T: Float[Array, ...],
    lambda_max: Float[Array, ...],
    max_candidates: int,
) -> tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
    """Sample via Lewis–Shedler thinning with a static buffer.

    Algorithm (Lewis & Shedler, 1979):

    1. Draw ``n`` candidates from a homogeneous Poisson process with
       rate ``λ_max`` on ``[0, T]``.
    2. Accept candidate at time ``t`` with probability
       :math:`\\lambda(t) / \\lambda_{\\max}`.

    For JAX-friendliness the candidate buffer is of static size
    ``max_candidates``: the Poisson draw is capped to this cap via a
    rank mask so the returned shape is deterministic.

    Args:
        key: JAX PRNG key.
        log_intensity_fn: Log-intensity callable.
        T: Observation window length (scalar).
        lambda_max: Upper bound on :math:`\\lambda(t)` over ``[0, T]``.
            Tighter bounds give higher acceptance rates; too loose a
            bound wastes work but is still exact.
        max_candidates: Static cap on the candidate pool.

    Returns:
        Tuple ``(times, accepted_mask, n_candidates_uncapped)``:

        * ``times`` with shape ``(max_candidates,)`` — sorted;
          padding positions (rank ``>= n_candidates``) are set to ``T``.
        * ``accepted_mask`` with shape ``(max_candidates,)`` — ``True``
          at positions that are (i) real candidates and (ii) accepted
          by thinning.
        * ``n_candidates_uncapped`` — the Poisson draw before capping
          at ``max_candidates``; useful for diagnosing buffer
          over-runs.
    """
    key_n, key_times, key_thin = random.split(key, 3)

    T = jnp.asarray(T)
    lambda_max = jnp.asarray(lambda_max)

    expected = lambda_max * T
    n_uncapped = random.poisson(key_n, expected)

    # Mark the first n_uncapped buffer positions as real candidates and
    # push padding to +inf before sorting so the real candidates occupy
    # sorted ranks 0..n-1 without order-statistic bias. (Sorting first
    # and then truncating to the first n would draw from the n smallest
    # of ``max_candidates`` uniforms, not from an actual HPP.)
    ranks = jnp.arange(max_candidates)
    candidate_mask = ranks < n_uncapped
    raw = random.uniform(key_times, shape=(max_candidates,)) * T
    raw_sortable = jnp.where(candidate_mask, raw, jnp.inf)
    candidate_times_sorted = jnp.sort(raw_sortable)
    # Replace padding positions with T so downstream log-intensity
    # calls evaluate inside the intended window.
    candidate_times = jnp.where(candidate_mask, candidate_times_sorted, T)

    log_intensities = log_intensity_fn(candidate_times)
    log_accept = log_intensities - jnp.log(lambda_max)
    u = random.uniform(key_thin, shape=(max_candidates,))
    thinning_accept = jnp.log(u) < log_accept
    accepted_mask = candidate_mask & thinning_accept

    times = jnp.where(candidate_mask, candidate_times, T)
    return times, accepted_mask, n_uncapped


def ipp_sample_inversion(
    key: PRNGKeyArray,
    inverse_cumulative_intensity_fn: Callable[[Array], Array],
    Lambda_T: Float[Array, ...],
    max_events: int,
) -> tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
    """Exact inversion sampling when :math:`\\Lambda^{-1}` is available.

    When the inverse compensator is known in closed form, sampling is
    exact and does not require an upper bound:

    1. Draw ``n`` from :math:`\\mathrm{Poisson}(\\Lambda(T))`.
    2. Draw ``n`` iid points :math:`u_i \\sim \\mathrm{Uniform}(0, \\Lambda(T))`.
    3. Map through :math:`\\Lambda^{-1}` and sort.

    Args:
        key: JAX PRNG key.
        inverse_cumulative_intensity_fn: Callable computing
            :math:`\\Lambda^{-1}(y)` for ``y`` in ``[0, Λ(T)]``. Must
            be vectorised over its input.
        Lambda_T: :math:`\\Lambda(T)` (scalar).
        max_events: Static buffer size.

    Returns:
        Tuple ``(times, mask, n_events)`` with the same semantics as
        :func:`ipp_sample_thinning` (no separate candidate/accepted
        split — every position is either a real event or padding).
    """
    key_n, key_u = random.split(key)
    Lambda_T = jnp.asarray(Lambda_T)

    n_events = random.poisson(key_n, Lambda_T)

    # Mark the first ``n_events`` buffer slots as real and push padding
    # uniforms to ``+inf`` before sorting, so the real draws occupy
    # sorted ranks ``0..n-1`` without being biased to small values.
    # (Sorting all ``max_events`` uniforms and then truncating would
    # pick the ``n`` smallest — not ``n`` iid uniforms on ``[0, Λ(T)]``.)
    ranks = jnp.arange(max_events)
    mask = ranks < n_events
    u = random.uniform(key_u, shape=(max_events,)) * Lambda_T
    u_sortable = jnp.where(mask, u, jnp.inf)
    u_sorted = jnp.sort(u_sortable)
    # Replace padding positions with ``Λ(T)`` before applying the
    # inverse — keeps every buffer slot inside a representable domain
    # so ``Λ⁻¹`` never sees ``+inf``.
    u_clamped = jnp.where(mask, u_sorted, Lambda_T)
    times = inverse_cumulative_intensity_fn(u_clamped)

    # Final padding-position value: ``Λ⁻¹(Λ(T))`` = right edge of window.
    t_right = inverse_cumulative_intensity_fn(Lambda_T)
    times = jnp.where(mask, times, t_right)
    return times, mask, n_events


def ipp_intensity(
    t: Float[Array, ...],
    log_intensity_fn: Callable[[Array], Array],
) -> Float[Array, ...]:
    """Pointwise intensity ``λ(t) = exp(log_intensity_fn(t))``."""
    return jnp.exp(log_intensity_fn(jnp.asarray(t)))


def ipp_cumulative_intensity(
    t: Float[Array, ...],
    log_intensity_fn: Callable[[Array], Array],
    n_points: int = 100,
) -> Float[Array, ...]:
    """Compensator :math:`\\Lambda(t)` via composite trapezoid."""
    return cumulative_log_intensity(log_intensity_fn, t, n_points=n_points)


def ipp_survival(
    t: Float[Array, ...],
    given_time: Float[Array, ...],
    log_intensity_fn: Callable[[Array], Array],
    n_points: int = 100,
) -> Float[Array, ...]:
    """:math:`S(t|s) = \\exp(-[\\Lambda(t) - \\Lambda(s)])` for ``t >= s``."""
    integrated = integrate_log_intensity(
        log_intensity_fn,
        jnp.asarray(given_time),
        jnp.asarray(t),
        n_points=n_points,
    )
    return jnp.exp(-integrated)


def ipp_hazard(
    t: Float[Array, ...],
    log_intensity_fn: Callable[[Array], Array],
) -> Float[Array, ...]:
    """Hazard equals the intensity for an IPP."""
    return ipp_intensity(t, log_intensity_fn)


def ipp_cumulative_hazard(
    t: Float[Array, ...],
    given_time: Float[Array, ...],
    log_intensity_fn: Callable[[Array], Array],
    n_points: int = 100,
) -> Float[Array, ...]:
    """:math:`\\Lambda(t) - \\Lambda(s)` via quadrature."""
    return integrate_log_intensity(
        log_intensity_fn,
        jnp.asarray(given_time),
        jnp.asarray(t),
        n_points=n_points,
    )


def ipp_inter_event_log_prob(
    tau: Float[Array, ...],
    current_time: Float[Array, ...],
    log_intensity_fn: Callable[[Array], Array],
    n_points: int = 100,
) -> Float[Array, ...]:
    """Log density of the next-event delay ``τ`` given last event at ``s``.

    For an IPP,

    .. math::
        f(\\tau \\mid s) = \\lambda(s + \\tau)
        \\exp(-[\\Lambda(s + \\tau) - \\Lambda(s)]).
    """
    tau = jnp.asarray(tau)
    s = jnp.asarray(current_time)
    t = s + tau
    log_lambda = log_intensity_fn(t)
    cum = integrate_log_intensity(log_intensity_fn, s, t, n_points=n_points)
    return jnp.where(tau >= 0.0, log_lambda - cum, -jnp.inf)


def ipp_predict_count(
    start_time: Float[Array, ...],
    end_time: Float[Array, ...],
    log_intensity_fn: Callable[[Array], Array],
    n_points: int = 100,
) -> Float[Array, ...]:
    r"""Expected count :math:`\Lambda(end) - \Lambda(start)` on ``[start, end]``."""
    return integrate_log_intensity(
        log_intensity_fn,
        jnp.asarray(start_time),
        jnp.asarray(end_time),
        n_points=n_points,
    )
