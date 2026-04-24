"""Goodness-of-fit diagnostics for temporal point processes.

Under the time-rescaling theorem, if :math:`\\{t_i\\}` are the events of
a process with compensator :math:`\\Lambda`, then the rescaled inter-event
times

.. math::
    \\tau_i = \\Lambda(t_i) - \\Lambda(t_{i-1})

are iid :math:`\\mathrm{Exp}(1)`. This module packages that null
distribution: residuals, Kolmogorov–Smirnov statistic, QQ-plot data,
and the compensator curve. Everything is a pure JAX function that
takes a ``cumulative_intensity_fn`` — so HPP, IPP, and (later) Hawkes
and renewal processes can all reuse the same diagnostics by plugging
in their own compensator.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


def time_rescaling_residuals(
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    cumulative_intensity_fn: Callable[[Array], Array],
) -> tuple[Float[Array, ...], Bool[Array, ...]]:
    """Rescaled inter-event times under the compensator.

    Args:
        event_times: Padded, sorted event times.
        mask: ``True`` for real events.
        cumulative_intensity_fn: Callable computing :math:`\\Lambda(t)`
            pointwise. For IPP this will typically be a closure around
            :func:`cumulative_log_intensity`; for HPP it is simply
            ``λ t``.

    Returns:
        Tuple ``(residuals, residual_mask)``:

        * ``residuals`` with shape equal to ``event_times`` — where
          position ``i`` holds :math:`\\Lambda(t_i) - \\Lambda(t_{i-1})`
          with :math:`t_0 = 0`. Padding positions are ``0``.
        * ``residual_mask`` — ``True`` for positions that correspond to
          real events. Same shape as ``mask``.

    Notes:
        Residuals at padding positions are deliberately returned as
        ``0`` (rather than ``NaN``) so downstream reductions that
        forget to mask do not propagate NaNs — they will instead see a
        biased statistic. All consumers in this module apply the mask.
    """
    Lambda_t = cumulative_intensity_fn(event_times)
    # Prepend Λ(0) = 0 so that residuals[0] = Λ(t_0) - 0 = Λ(t_0).
    zero_padding = jnp.zeros_like(Lambda_t[..., :1])
    Lambda_prev = jnp.concatenate([zero_padding, Lambda_t[..., :-1]], axis=-1)
    residuals = Lambda_t - Lambda_prev
    residuals = jnp.where(mask, residuals, 0.0)
    return residuals, mask


def ks_statistic_exp1(
    residuals: Float[Array, ...],
    mask: Bool[Array, ...],
) -> Float[Array, ...]:
    """Two-sided Kolmogorov–Smirnov statistic against ``Exp(1)``.

    Args:
        residuals: Time-rescaled residuals.
        mask: ``True`` at real-event positions.

    Returns:
        :math:`\\max_i |F_n(\\tau_{(i)}) - (1 - e^{-\\tau_{(i)}})|` where
        :math:`F_n` is the empirical CDF computed from the unmasked
        residuals only.

    Notes:
        With padding present this requires a little care: we sort
        residuals with padding positions pushed to ``+inf`` so they
        land at the top of the sorted order, and compute empirical
        ranks using the per-batch real count. Padding contributes no
        mass to the empirical CDF.
    """
    n_real = jnp.sum(mask, axis=-1)
    # Push padding to +inf so real residuals occupy ranks 0..n_real-1.
    sort_keys = jnp.where(mask, residuals, jnp.inf)
    sorted_res = jnp.sort(sort_keys, axis=-1)

    max_len = residuals.shape[-1]
    ranks = jnp.arange(1, max_len + 1, dtype=residuals.dtype)
    n_broadcast = jnp.expand_dims(n_real.astype(residuals.dtype), axis=-1)
    inv_n = 1.0 / jnp.maximum(n_broadcast, 1.0)

    theoretical = 1.0 - jnp.exp(-sorted_res)
    # Two-sided one-sample KS: at rank i the empirical CDF jumps from
    # (i-1)/n to i/n, so the supremum of |F_n(x) - F(x)| can land on
    # either side of the jump. Taking only ``|i/n - F|`` (as the first
    # version did) misses the ``F - (i-1)/n`` branch and under-reports
    # the distance when the theoretical CDF lands just below a step.
    upper = ranks * inv_n - theoretical
    lower = theoretical - (ranks - 1.0) * inv_n
    diff = jnp.maximum(upper, lower)
    valid = ranks <= n_broadcast
    return jnp.max(jnp.where(valid, diff, 0.0), axis=-1)


def qq_exp1_quantiles(
    residuals: Float[Array, ...],
    mask: Bool[Array, ...],
) -> tuple[Float[Array, ...], Float[Array, ...]]:
    """Theoretical vs empirical quantiles against ``Exp(1)`` for a QQ plot.

    Args:
        residuals: Time-rescaling residuals.
        mask: ``True`` at real events.

    Returns:
        Tuple ``(theoretical, empirical)`` of shape equal to
        ``residuals``. Positions with ``rank > n_real`` are filled
        with ``NaN`` so a plotter can detect and drop them.
    """
    n_real = jnp.sum(mask, axis=-1)
    sort_keys = jnp.where(mask, residuals, jnp.inf)
    sorted_res = jnp.sort(sort_keys, axis=-1)

    max_len = residuals.shape[-1]
    ranks = jnp.arange(1, max_len + 1, dtype=residuals.dtype)
    n_broadcast = jnp.expand_dims(n_real.astype(residuals.dtype), axis=-1)
    # Plotting positions (i - 0.5) / n avoid the F⁻¹(1) = ∞ edge case.
    plot_positions = (ranks - 0.5) / jnp.maximum(n_broadcast, 1.0)
    theoretical = -jnp.log1p(-plot_positions)

    valid = ranks <= n_broadcast
    theoretical = jnp.where(valid, theoretical, jnp.nan)
    empirical = jnp.where(valid, sorted_res, jnp.nan)
    return theoretical, empirical


def compensator_curve(
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    cumulative_intensity_fn: Callable[[Array], Array],
) -> tuple[Float[Array, ...], Float[Array, ...]]:
    """Pairs ``(t_i, Λ(t_i))`` for a compensator plot.

    Under the null the points lie along the line
    :math:`y = \\Lambda(t_i)` with slope 1 when plotted against the
    event index ``i`` — the purpose of the plot is to see deviations
    from that line.

    Args:
        event_times: Padded event times.
        mask: Event mask.
        cumulative_intensity_fn: Compensator callable.

    Returns:
        Tuple ``(times, compensator)`` where padding positions are
        replaced with ``NaN`` for plot-friendliness.
    """
    Lambda_t = cumulative_intensity_fn(event_times)
    times = jnp.where(mask, event_times, jnp.nan)
    cum = jnp.where(mask, Lambda_t, jnp.nan)
    return times, cum
