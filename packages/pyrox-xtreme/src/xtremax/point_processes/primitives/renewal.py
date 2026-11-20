"""Pure primitives for 1-D renewal temporal point processes.

A renewal process on ``[0, T]`` has iid inter-event times
:math:`\\Delta t_i \\sim F` with density :math:`f` and survival
:math:`S = 1 - F`. Given events :math:`\\{t_1, \\dots, t_n\\}`,

.. math::
    \\log L = \\sum_{i=1}^{n} \\log f(\\Delta t_i) + \\log S(T - t_n).

The interface accepts any NumPyro ``Distribution`` for the inter-event
law — ``Exponential``, ``Gamma``, ``Weibull``, ``LogNormal``, or
custom. Survival is computed via ``log1p(-cdf(·))`` since NumPyro
Distributions don't uniformly expose a log-survival method.

All functions are ``jit`` / ``vmap`` / ``grad`` safe; variable-length
sequences use the padded-times + mask representation established in
the rest of the module.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._results import SampleResult


def renewal_log_prob(
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    T: Float[Array, ...],
    inter_event_dist: dist.Distribution,
) -> Float[Array, ...]:
    """Log-likelihood of an observed renewal-process sequence.

    Args:
        event_times: Padded, sorted event times.
        mask: Boolean event mask.
        T: Observation window length.
        inter_event_dist: NumPyro Distribution for inter-event times.
            Must expose ``log_prob`` and ``cdf``.

    Returns:
        Scalar log-likelihood.

    Notes:
        The final survival term uses ``log1p(-cdf(T - t_n))`` with
        clipping to avoid ``log(0)`` when the observed gap has CDF
        very close to 1. The inter-event densities at padding positions
        are zeroed out by the mask.

        Requires the package padded-buffer invariant (contiguous-prefix
        mask): gaps are taken between consecutive *buffer slots*, so a
        hole mask (a masked-out slot between two real events) silently
        corrupts the gap terms. Package samplers always emit
        contiguous-prefix masks.
    """
    event_times = jnp.asarray(event_times)
    T_arr = jnp.asarray(T)

    # Inter-event gaps: Δt_i = t_i - t_{i-1} (with t_0 = 0).
    zero = jnp.zeros_like(event_times[..., :1])
    prev_times = jnp.concatenate([zero, event_times[..., :-1]], axis=-1)
    gaps = event_times - prev_times
    # Guard padding positions from producing NaN-producing log_probs.
    safe_gaps = jnp.where(mask, gaps, jnp.ones_like(gaps))
    log_f = inter_event_dist.log_prob(safe_gaps)
    log_density_sum = jnp.sum(jnp.where(mask, log_f, 0.0), axis=-1)

    # Tail survival from the last real event to the window edge.
    n_real = jnp.sum(mask, axis=-1)
    last_time = jnp.max(
        jnp.where(mask, event_times, jnp.zeros_like(event_times)), axis=-1
    )
    gap_after_last = T_arr - last_time
    # With no observed events the tail gap is the whole window.
    no_events = n_real == 0
    gap_after_last = jnp.where(no_events, T_arr, gap_after_last)
    cdf_tail = inter_event_dist.cdf(gap_after_last)
    log_survival = jnp.log(jnp.clip(1.0 - cdf_tail, 1e-30, 1.0))

    return log_density_sum + log_survival


def renewal_sample(
    key: PRNGKeyArray,
    inter_event_dist: dist.Distribution,
    T: Float[Array, ...],
    max_events: int,
) -> tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
    """Sample a renewal-process event sequence by accumulating gaps.

    Args:
        key: JAX PRNG key.
        inter_event_dist: Inter-event NumPyro Distribution. Must have
            a reparameterised ``sample`` that is ``jit`` friendly.
        T: Observation window length.
        max_events: Static buffer size.

    Returns:
        Tuple ``(times, mask, n_events)``.

    Notes:
        Draws ``max_events`` iid gaps unconditionally and cumulative-
        sums them; events past ``T`` are masked out. This keeps shape
        static while preserving the exact renewal distribution on
        ``[0, T]``. If the buffer fills (sum of gaps < ``T``) the
        returned sequence is truncated — choose ``max_events`` so that
        ``max_events * E[Δt] >> T``.
    """
    T_arr = jnp.asarray(T)
    gaps = inter_event_dist.sample(key, sample_shape=(max_events,))
    times = jnp.cumsum(gaps, axis=-1)
    mask = times < T_arr
    n_events = jnp.sum(mask, axis=-1).astype(jnp.int32)
    # Pad out-of-window positions to T so downstream intensity evaluations stay safe.
    times = jnp.where(mask, times, T_arr)
    return SampleResult(times, mask, n_events)


def renewal_hazard(
    tau: Float[Array, ...],
    inter_event_dist: dist.Distribution,
) -> Float[Array, ...]:
    r"""Hazard of the inter-event distribution: :math:`h(\tau) = f(\tau)/S(\tau)`.

    Implemented as ``exp(log_prob(τ) - log(1 - cdf(τ)))`` to stay
    numerically safe.
    """
    tau = jnp.asarray(tau)
    log_f = inter_event_dist.log_prob(tau)
    log_s = jnp.log(jnp.clip(1.0 - inter_event_dist.cdf(tau), 1e-30, 1.0))
    return jnp.exp(log_f - log_s)


def renewal_cumulative_hazard(
    tau: Float[Array, ...],
    inter_event_dist: dist.Distribution,
) -> Float[Array, ...]:
    r""":math:`\Lambda(\tau) = -\log S(\tau) = -\log(1 - F(\tau))`."""
    tau = jnp.asarray(tau)
    return -jnp.log(jnp.clip(1.0 - inter_event_dist.cdf(tau), 1e-30, 1.0))


def renewal_intensity(
    t: Float[Array, ...],
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    inter_event_dist: dist.Distribution,
) -> Float[Array, ...]:
    r"""Conditional intensity :math:`\lambda^*(t) = h(t - t_{N(t)})`.

    Args:
        t: Evaluation time.
        event_times: Padded history.
        mask: Event mask.
        inter_event_dist: Inter-event distribution.

    Returns:
        ``h(τ)`` where ``τ`` is the time since the most recent event
        strictly before ``t``.
    """
    t = jnp.asarray(t)
    # Most recent event strictly before t (0 if none).
    before_t = mask & (event_times < t)
    last_time = jnp.max(
        jnp.where(before_t, event_times, jnp.zeros_like(event_times)), axis=-1
    )
    last_time = jnp.where(jnp.any(before_t, axis=-1), last_time, 0.0)
    tau = t - last_time
    return renewal_hazard(tau, inter_event_dist)


def renewal_inter_event_log_prob(
    tau: Float[Array, ...],
    inter_event_dist: dist.Distribution,
) -> Float[Array, ...]:
    """Log density of an inter-event gap: ``log f(τ)``."""
    return inter_event_dist.log_prob(jnp.asarray(tau))


def renewal_survival(
    tau: Float[Array, ...],
    inter_event_dist: dist.Distribution,
) -> Float[Array, ...]:
    """``S(τ) = 1 − F(τ)`` clipped to be non-negative."""
    return jnp.clip(1.0 - inter_event_dist.cdf(jnp.asarray(tau)), 0.0, 1.0)


def renewal_expected_count(
    T: Float[Array, ...],
    inter_event_dist: dist.Distribution,
    n_points: int = 200,
) -> Float[Array, ...]:
    r"""Expected number of events on ``[0, T]`` via the renewal equation.

    Solves :math:`m(T) = F(T) + \\int_0^T m(T - s)\\, f(s)\\, ds` on an
    ``n_points`` trapezoid grid by **forward substitution**: the
    discretized equation is lower-triangular in the grid values of
    ``m``, so a single sweep over grid rows solves it exactly (up to
    quadrature error) at *any* horizon. The previous fixed 8-iteration
    Neumann series added one convolution power per iteration and
    therefore saturated near ``E[N] ≈ 9`` regardless of ``T``. For
    ``Exponential`` inter-events this recovers ``λT`` to quadrature
    tolerance.

    Args:
        T: Window length (scalar).
        inter_event_dist: Inter-event NumPyro Distribution.
        n_points: Trapezoid grid size.

    Returns:
        Scalar expected count.
    """
    T_arr = jnp.asarray(T)
    grid = jnp.linspace(jnp.zeros_like(T_arr), T_arr, n_points)
    f_grid = jnp.exp(inter_event_dist.log_prob(grid))
    F_grid = inter_event_dist.cdf(grid)
    dt = grid[1] - grid[0]

    # Row i of the trapezoid-discretized equation:
    #   m_i = F_i + dt·[ f_0 m_i / 2 + Σ_{j=1}^{i-1} f_j m_{i-j} + f_i m_0 / 2 ]
    # The f_0 m_i / 2 term moves to the left-hand side; everything else
    # involves only already-computed rows, so a forward scan solves the
    # lower-triangular system in one pass.
    j = jnp.arange(n_points)
    denom = 1.0 - 0.5 * dt * f_grid[0]

    def row(m_cur: Array, i: Array) -> tuple[Array, None]:
        m_rev = m_cur[jnp.clip(i - j, 0, n_points - 1)]
        weights = jnp.where((j >= 1) & (j <= i - 1), 1.0, 0.0) + jnp.where(
            (j == i) & (i > 0), 0.5, 0.0
        )
        conv = jnp.sum(f_grid * m_rev * weights) * dt
        m_i = (F_grid[i] + conv) / denom
        return m_cur.at[i].set(m_i), None

    m_init = jnp.zeros_like(F_grid)
    m_final, _ = jax.lax.scan(row, m_init, jnp.arange(n_points))
    return m_final[-1]


def renewal_ogata_intensity_fn(
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    inter_event_dist: dist.Distribution,
):
    """Build a ``(t, history) -> λ*(t)`` closure usable by thinning samplers.

    Convenience constructor for callers that want to mix renewal with
    other history-dependent constructions (e.g. a ThinningProcess
    wrapping a renewal base). The ``event_times`` / ``mask`` arguments
    are **unused** — they exist only for signature symmetry with the
    other ``*_intensity_fn`` builders; the returned closure evaluates
    against the *live* ``history`` the sampler passes in, so
    self-generated chains see their own events.
    """
    del event_times, mask  # unused; kept for signature symmetry

    def _fn(t: Array, history) -> Array:
        # Prefer the live history (passed in by the sampler) over the
        # captured times so self-generated chains see their own events.
        return renewal_intensity(t, history.times, history.mask, inter_event_dist)

    return _fn


def renewal_log_prob_counts(
    n_events: Int[Array, ...],
    T: Float[Array, ...],
    inter_event_dist: dist.Distribution,
) -> Float[Array, ...]:
    """Count marginal :math:`P(N(T) = n)` — **not implemented**.

    The count marginal of a general renewal process,
    :math:`F_n(T) - F_{n+1}(T)` with :math:`F_k` the CDF of the sum of
    ``k`` iid gaps, has no simple closed form (only Exponential gaps
    recover the Poisson pmf). This function exists to reserve the name
    and unconditionally raises ``NotImplementedError`` — use Monte
    Carlo over :func:`renewal_sample`, or the event-time likelihood
    :func:`renewal_log_prob`, instead.
    """
    raise NotImplementedError(
        "Count marginal P(N(T) = n) has no simple closed form for "
        "general renewal; use Monte Carlo from `renewal_sample` or "
        "work with the event-time likelihood `renewal_log_prob` instead."
    )
