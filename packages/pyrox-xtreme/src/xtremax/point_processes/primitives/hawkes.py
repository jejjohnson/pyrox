r"""Pure primitives for self-exciting (Hawkes) temporal point processes.

Given a baseline :math:`\mu > 0` and an excitation kernel
:math:`\phi(\tau) \geq 0` supported on :math:`\tau \geq 0`, the
conditional intensity given history :math:`\mathcal{H}_t` is

.. math::
    \lambda^*(t) = \mu + \sum_{t_i < t} \phi(t - t_i).

The compensator on ``[0, T]`` is

.. math::
    \Lambda(T) = \mu T + \sum_{i} \int_{t_i}^{T} \phi(s - t_i)\, ds
              = \mu T + \sum_{i} \Phi(T - t_i),

with :math:`\Phi` the integral of :math:`\phi`. The log-likelihood is
:math:`\sum_i \log \lambda^*(t_i) - \Lambda(T)`.

This module specialises the common **exponential kernel**
:math:`\phi(\tau) = \alpha e^{-\beta \tau}` (closed-form compensator,
O(n) recursion for intensities via Ozaki 1979) and provides the
general-kernel machinery that routes through
:mod:`~xtremax.point_processes.primitives.thinning` for sampling.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._history import EventHistory
from xtremax.point_processes.primitives.thinning import thinning_sample


def exp_hawkes_intensity(
    t: Float[Array, ...],
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    mu: Float[Array, ...],
    alpha: Float[Array, ...],
    beta: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Conditional intensity of the exponential-kernel Hawkes process.

    :math:`\lambda^*(t) = \mu + \alpha \sum_{t_i < t} e^{-\beta (t - t_i)}`.

    Args:
        t: Evaluation time(s).
        event_times: Padded, sorted past events.
        mask: Event mask.
        mu: Baseline rate (positive).
        alpha: Excitation amplitude (positive).
        beta: Excitation decay rate (positive).

    Returns:
        :math:`\lambda^*(t)` with shape equal to the broadcast of
        ``t`` and the leading event-time batch dims.
    """
    t_arr = jnp.asarray(t)
    past = mask & (
        event_times < t_arr[..., None] if t_arr.ndim > 0 else event_times < t_arr
    )
    # Avoid overflow from negative exponents at padding positions.
    if t_arr.ndim == 0:
        dt = t_arr - event_times
        contributions = jnp.where(past, alpha * jnp.exp(-beta * dt), 0.0)
        return mu + jnp.sum(contributions, axis=-1)
    dt = t_arr[..., None] - event_times
    contributions = jnp.where(past, alpha * jnp.exp(-beta * dt), 0.0)
    return mu + jnp.sum(contributions, axis=-1)


def exp_hawkes_cumulative_intensity(
    t: Float[Array, ...],
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    mu: Float[Array, ...],
    alpha: Float[Array, ...],
    beta: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Exponential-kernel Hawkes compensator.

    :math:`\Lambda(t) = \mu t + (\alpha / \beta) \sum_{t_i \leq t}
    (1 - e^{-\beta(t - t_i)})`.
    """
    t_arr = jnp.asarray(t)
    if t_arr.ndim == 0:
        past = mask & (event_times <= t_arr)
        dt = t_arr - event_times
        per_event = (alpha / beta) * (1.0 - jnp.exp(-beta * dt))
        return mu * t_arr + jnp.sum(jnp.where(past, per_event, 0.0), axis=-1)
    past = mask & (event_times <= t_arr[..., None])
    dt = t_arr[..., None] - event_times
    per_event = (alpha / beta) * (1.0 - jnp.exp(-beta * dt))
    return mu * t_arr + jnp.sum(jnp.where(past, per_event, 0.0), axis=-1)


def exp_hawkes_log_prob(
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    T: Float[Array, ...],
    mu: Float[Array, ...],
    alpha: Float[Array, ...],
    beta: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Log-likelihood :math:`\sum_i \log \lambda^*(t_i) - \Lambda(T)` for exp Hawkes.

    Uses the Ozaki recursion :math:`R_i = e^{-\beta (t_i - t_{i-1})}(1 + R_{i-1})`
    so intensity evaluation at each observed event is O(1) amortised,
    giving O(n) total cost.

    Args:
        event_times: Padded, sorted event times.
        mask: Event mask.
        T: Observation window length.
        mu: Baseline.
        alpha: Excitation amplitude.
        beta: Excitation decay.

    Returns:
        Scalar log-likelihood.
    """
    event_times = jnp.asarray(event_times)
    T_arr = jnp.asarray(T)

    # Ozaki recursion for intensities at observed event times.
    # Using the A-variable A_i = 1 + Σ_{j<i} exp(-β(t_i - t_j)) = 1 + R_i
    # which satisfies A_i = 1 + exp(-β·gap_i) · A_{i-1}, A_{-1} := 0,
    # so R_i = A_i − 1 and the recursion is numerically clean.
    zero = jnp.zeros_like(event_times[..., :1])
    prev_times = jnp.concatenate([zero, event_times[..., :-1]], axis=-1)
    gaps = event_times - prev_times

    def step(A_prev: Array, inp: tuple[Array, Array]) -> tuple[Array, Array]:
        gap, valid = inp
        A_new = jnp.where(
            valid,
            1.0 + jnp.exp(-beta * gap) * A_prev,
            A_prev,
        )
        # Output R_i = A_new - 1 when the current event is valid; at
        # padding positions emit 0 (masked out of the sum anyway).
        R_i = jnp.where(valid, A_new - 1.0, 0.0)
        return A_new, R_i

    _, R_strict = jax.lax.scan(
        step,
        jnp.zeros(event_times.shape[:-1], dtype=event_times.dtype),
        (jnp.moveaxis(gaps, -1, 0), jnp.moveaxis(mask, -1, 0)),
    )
    R_strict = jnp.moveaxis(R_strict, 0, -1)
    lam_at_events = mu + alpha * R_strict
    # Guard the log at padding positions (lam_at_events = μ > 0 there anyway).
    log_lam = jnp.log(jnp.clip(lam_at_events, 1e-30, jnp.inf))
    sum_log_lam = jnp.sum(jnp.where(mask, log_lam, 0.0), axis=-1)

    Lambda_T = exp_hawkes_cumulative_intensity(
        T_arr, event_times, mask, mu, alpha, beta
    )
    return sum_log_lam - Lambda_T


def exp_hawkes_lambda_max(
    history: EventHistory,
    mu: Float[Array, ...],
    alpha: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Upper bound :math:`\mu + \alpha \cdot N(t)` valid from the last event onward."""
    n = jnp.sum(history.mask).astype(mu.dtype)
    return mu + alpha * n


def exp_hawkes_sample(
    key: PRNGKeyArray,
    T: Float[Array, ...],
    mu: Float[Array, ...],
    alpha: Float[Array, ...],
    beta: Float[Array, ...],
    max_events: int,
    max_candidates: int | None = None,
) -> tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
    """Ogata thinning sample of an exponential-kernel Hawkes process.

    Uses the adaptive envelope :math:`\\bar\\lambda_k = \\mu + \\alpha k`
    valid for :math:`t \\geq t_k`. With a decaying excitation this
    bound is tight just after each event and loosens in between —
    efficient enough for :math:`\\alpha / \\beta < 1` (sub-critical).
    """
    mu_arr = jnp.asarray(mu)
    alpha_arr = jnp.asarray(alpha)
    beta_arr = jnp.asarray(beta)

    def intensity(t: Array, history: EventHistory) -> Array:
        return exp_hawkes_intensity(
            t, history.times, history.mask, mu_arr, alpha_arr, beta_arr
        )

    def lambda_max(history: EventHistory) -> Array:
        return exp_hawkes_lambda_max(history, mu_arr, alpha_arr)

    return thinning_sample(
        key=key,
        conditional_intensity_fn=intensity,
        lambda_max_fn=lambda_max,
        T=T,
        max_events=max_events,
        max_candidates=max_candidates,
    )


def general_hawkes_intensity(
    t: Float[Array, ...],
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    mu: Float[Array, ...],
    kernel_fn: Callable[[Array], Array],
) -> Float[Array, ...]:
    r"""Conditional intensity for a general kernel.

    :math:`\lambda^*(t) = \mu + \sum_{t_i < t} \phi(t - t_i)`.
    """
    t_arr = jnp.asarray(t)
    if t_arr.ndim == 0:
        past = mask & (event_times < t_arr)
        dt = t_arr - event_times
        kernel_vals = kernel_fn(dt)
        return mu + jnp.sum(jnp.where(past, kernel_vals, 0.0), axis=-1)
    past = mask & (event_times < t_arr[..., None])
    dt = t_arr[..., None] - event_times
    kernel_vals = kernel_fn(dt)
    return mu + jnp.sum(jnp.where(past, kernel_vals, 0.0), axis=-1)


def general_hawkes_cumulative_intensity(
    T: Float[Array, ...],
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    mu: Float[Array, ...],
    kernel_integral_fn: Callable[[Array, Array], Array],
) -> Float[Array, ...]:
    r"""Compensator :math:`\mu T + \sum_i \int_{t_i}^{T} \phi(s - t_i) ds`."""
    T_arr = jnp.asarray(T)
    lower = jnp.zeros_like(event_times)
    upper = jnp.clip(T_arr - event_times, 0.0, jnp.inf)
    per_event = kernel_integral_fn(lower, upper)
    return mu * T_arr + jnp.sum(jnp.where(mask, per_event, 0.0), axis=-1)


def general_hawkes_log_prob(
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    T: Float[Array, ...],
    mu: Float[Array, ...],
    kernel_fn: Callable[[Array], Array],
    kernel_integral_fn: Callable[[Array, Array], Array],
) -> Float[Array, ...]:
    r"""Log-likelihood for a general-kernel Hawkes process.

    Args:
        event_times: Padded event times.
        mask: Event mask.
        T: Window length.
        mu: Baseline rate.
        kernel_fn: ``φ(τ) -> Array`` broadcast-friendly.
        kernel_integral_fn: ``(a, b) -> ∫_a^b φ(s) ds`` broadcast-friendly.
            Used for the compensator term; closed-form forms give exact
            gradients, quadrature fallbacks (see the operator layer)
            trade some accuracy for generality.

    Returns:
        Scalar log-likelihood.
    """
    # λ*(t_i) per event via O(n²) pairwise construction (padded-mask-safe).
    n_max = event_times.shape[-1]
    dt = event_times[..., :, None] - event_times[..., None, :]
    kernel_vals = kernel_fn(dt)
    idx = jnp.arange(n_max)
    strictly_before = idx[None, :] < idx[:, None]  # j strictly before i
    pair_valid = strictly_before & mask[..., None, :]
    per_i = jnp.sum(jnp.where(pair_valid, kernel_vals, 0.0), axis=-1)
    lam_at_events = mu + per_i
    log_lam = jnp.log(jnp.clip(lam_at_events, 1e-30, jnp.inf))
    sum_log_lam = jnp.sum(jnp.where(mask, log_lam, 0.0), axis=-1)

    Lambda_T = general_hawkes_cumulative_intensity(
        T, event_times, mask, mu, kernel_integral_fn
    )
    return sum_log_lam - Lambda_T


def general_hawkes_sample(
    key: PRNGKeyArray,
    T: Float[Array, ...],
    mu: Float[Array, ...],
    kernel_fn: Callable[[Array], Array],
    kernel_max_fn: Callable[[EventHistory], Array],
    max_events: int,
    max_candidates: int | None = None,
) -> tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
    r"""Ogata thinning for a general-kernel Hawkes process.

    The caller supplies ``kernel_max_fn(history) -> λ_bar`` — the
    history-dependent upper bound. For monotone-decreasing kernels
    with maximum at :math:`\tau = 0`, a safe choice is
    :math:`\mu + \phi(0) \cdot N(t)`; tighter bounds give higher
    acceptance.
    """
    mu_arr = jnp.asarray(mu)

    def intensity(t: Array, history: EventHistory) -> Array:
        return general_hawkes_intensity(
            t, history.times, history.mask, mu_arr, kernel_fn
        )

    return thinning_sample(
        key=key,
        conditional_intensity_fn=intensity,
        lambda_max_fn=kernel_max_fn,
        T=T,
        max_events=max_events,
        max_candidates=max_candidates,
    )
