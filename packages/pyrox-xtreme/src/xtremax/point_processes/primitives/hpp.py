"""Pure-JAX primitives for the Homogeneous Poisson Process on ``[0, T]``.

A homogeneous Poisson process with intensity :math:`\\lambda > 0` on
:math:`[0, T]` has

* count :math:`N([0, T]) \\sim \\mathrm{Poisson}(\\lambda T)`
* conditional event times iid uniform on :math:`[0, T]`
* joint log-likelihood :math:`\\log L = n \\log \\lambda - \\lambda T`

All functions here are stateless and ``jax.jit`` / ``jax.vmap`` /
``jax.grad`` safe. Variable-length event sequences are represented as
*padded* arrays plus a boolean mask so the output shape is static — a
concession to JAX that the ``(locations, n_events)`` tuple pattern in
the original snippet cannot make.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray


def hpp_log_prob(
    n_events: Int[Array, ...],
    rate: Float[Array, ...],
    T: Float[Array, ...],
) -> Float[Array, ...]:
    """Log-likelihood :math:`\\log L = n \\log \\lambda - \\lambda T`.

    Args:
        n_events: Observed event count.
        rate: Intensity ``λ > 0``.
        T: Length of the observation window.

    Returns:
        Log-likelihood, broadcast over inputs.

    Notes:
        The event times drop out because, given ``n``, they are uniform
        on ``[0, T]`` and their density cancels with the Poisson count
        factors :math:`n!` and :math:`T^n`. What remains is
        :math:`\\lambda^n \\exp(-\\lambda T)`, which is what this
        returns in log-space. If you instead want the full
        Poisson-count log-probability, add
        :math:`-\\log(n!) - n \\log T`.
    """
    n = jnp.asarray(n_events)
    rate = jnp.asarray(rate)
    T = jnp.asarray(T)
    return n * jnp.log(rate) - rate * T


def hpp_sample(
    key: PRNGKeyArray,
    rate: Float[Array, ...],
    T: Float[Array, ...],
    max_events: int,
    sample_shape: tuple[int, ...] = (),
) -> tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
    """Sample ordered event times from a homogeneous Poisson process.

    Args:
        key: JAX PRNG key.
        rate: Intensity ``λ > 0`` (scalar or batched).
        T: Observation window length (broadcasts with ``rate``).
        max_events: Static buffer size. The sampler truncates at this
            cap; choose comfortably above the expected count to avoid
            loss of realism (``5 * λ T`` is a reasonable starting
            point).
        sample_shape: Optional leading sample dimensions.

    Returns:
        Tuple ``(times, mask, n_events)``:

        * ``times`` with shape ``sample_shape + batch_shape + (max_events,)``
          — sorted, with padding positions set to ``T``.
        * ``mask`` with the same shape as ``times`` — ``True`` where
          ``times`` is a real event, ``False`` for padding.
        * ``n_events`` with shape ``sample_shape + batch_shape`` — the
          uncapped Poisson draw. If ``n_events > max_events`` the
          returned events have been truncated to the first ``max_events``.

    Notes:
        The algorithm draws ``n`` from :math:`\\mathrm{Poisson}(\\lambda T)`
        and then, unconditionally, generates ``max_events`` uniform
        candidates on ``[0, T]``. Sorting and masking by rank lets the
        same padded buffer serve any batch element regardless of its
        count — and keeps the whole routine jit-compatible.
    """
    rate = jnp.asarray(rate)
    T = jnp.asarray(T)
    batch_shape = jnp.broadcast_shapes(rate.shape, T.shape)
    shape = tuple(sample_shape) + batch_shape

    key_n, key_times = random.split(key)

    expected = jnp.broadcast_to(rate * T, shape)
    n_events = random.poisson(key_n, expected, shape=shape)

    # Draw one uniform per buffer slot; mark the first ``n_events`` as
    # real events via ``mask`` and push padding positions to ``+inf``
    # before sorting so the real events occupy ranks ``0..n-1`` in the
    # sorted order without being biased to small values (which would
    # happen if we sorted first and then truncated to the first n).
    T_b = jnp.broadcast_to(T, shape)[..., None]
    ranks = jnp.arange(max_events)
    mask = ranks < n_events[..., None]
    raw = random.uniform(key_times, shape=(*shape, max_events)) * T_b
    raw_sortable = jnp.where(mask, raw, jnp.inf)
    times_sorted = jnp.sort(raw_sortable, axis=-1)
    # Real events are finite; padding positions are set to ``T`` (a
    # safe value inside the window) so downstream intensity evaluations
    # don't trip on ``+inf``.
    times = jnp.where(mask, times_sorted, T_b)
    return times, mask, n_events


def hpp_intensity(
    t: Float[Array, ...],
    rate: Float[Array, ...],
) -> Float[Array, ...]:
    """Constant intensity :math:`\\lambda(t) = \\lambda`.

    Broadcasts to the joint shape of ``t`` and ``rate`` so scalar ``t``
    with batched ``rate`` (and vice versa) both work, matching the
    broadcasting semantics of the other HPP primitives.
    """
    t = jnp.asarray(t)
    rate = jnp.asarray(rate)
    out_shape = jnp.broadcast_shapes(t.shape, rate.shape)
    return jnp.broadcast_to(rate, out_shape)


def hpp_cumulative_intensity(
    t: Float[Array, ...],
    rate: Float[Array, ...],
) -> Float[Array, ...]:
    """Compensator :math:`\\Lambda(t) = \\lambda t`."""
    return jnp.asarray(rate) * jnp.asarray(t)


def hpp_survival(
    t: Float[Array, ...],
    given_time: Float[Array, ...],
    rate: Float[Array, ...],
) -> Float[Array, ...]:
    """:math:`S(t|s) = \\exp(-\\lambda (t - s))` for ``t >= s``."""
    rate = jnp.asarray(rate)
    return jnp.exp(-rate * (jnp.asarray(t) - jnp.asarray(given_time)))


def hpp_hazard(
    t: Float[Array, ...],
    rate: Float[Array, ...],
) -> Float[Array, ...]:
    """Constant hazard equal to ``rate``."""
    return hpp_intensity(t, rate)


def hpp_inter_event_log_prob(
    tau: Float[Array, ...],
    rate: Float[Array, ...],
) -> Float[Array, ...]:
    """Log density of ``τ ~ Exp(λ)``: :math:`\\log \\lambda - \\lambda \\tau`."""
    tau = jnp.asarray(tau)
    rate = jnp.asarray(rate)
    return jnp.where(tau >= 0.0, jnp.log(rate) - rate * tau, -jnp.inf)


def hpp_mean_residual_life(
    rate: Float[Array, ...],
    given_time: Float[Array, ...] | None = None,
) -> Float[Array, ...]:
    """Memorylessness gives :math:`\\mathbb{E}[T - s \\mid T > s] = 1/\\lambda`.

    Args:
        rate: Intensity.
        given_time: Ignored — present for API symmetry with IPP, where
            mean residual life does depend on the conditioning time.

    Returns:
        ``1 / rate`` broadcast to the shape of ``rate``.
    """
    del given_time
    return 1.0 / jnp.asarray(rate)


def hpp_return_period(
    probability: Float[Array, ...],
    rate: Float[Array, ...],
) -> Float[Array, ...]:
    """Time ``τ`` such that :math:`P(\\text{no event in } \\tau) = 1 - p`.

    Equivalently :math:`\\tau = -\\log(1 - p) / \\lambda`. The ``log1p``
    form stays accurate as ``p → 0``.
    """
    p = jnp.asarray(probability)
    return -jnp.log1p(-p) / jnp.asarray(rate)


def hpp_predict_count(
    start_time: Float[Array, ...],
    end_time: Float[Array, ...],
    rate: Float[Array, ...],
) -> Float[Array, ...]:
    """Expected event count on ``[start, end]``: ``λ (end - start)``."""
    return jnp.asarray(rate) * (jnp.asarray(end_time) - jnp.asarray(start_time))


def hpp_exceedance_log_prob(
    k: Int[Array, ...],
    start_time: Float[Array, ...],
    end_time: Float[Array, ...],
    rate: Float[Array, ...],
) -> Float[Array, ...]:
    """log P(N > k) over ``[start, end]``, computed via the Poisson CDF.

    Returns ``log(1 - F_Poisson(k; λ (end - start)))``. Uses
    :func:`jax.scipy.stats.poisson.cdf` under the hood; for large ``k``
    or small tails this can underflow — callers that care about very
    rare events should work in log-space end to end.
    """
    lam = jnp.asarray(rate) * (jnp.asarray(end_time) - jnp.asarray(start_time))
    cdf_k = jax.scipy.stats.poisson.cdf(jnp.asarray(k), lam)
    return jnp.log(jnp.clip(1.0 - cdf_k, 0.0, 1.0))
