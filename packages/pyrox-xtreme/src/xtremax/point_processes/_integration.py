"""Numerical integration utilities for temporal point processes.

These helpers compute definite integrals of intensity functions
:math:`\\Lambda(t_1) - \\Lambda(t_0) = \\int_{t_0}^{t_1} \\lambda(s)\\, ds`
using composite quadrature on a fixed-size grid. The grid size is a
static Python ``int`` so the whole pipeline stays ``jax.jit`` friendly.

The intensity function is always passed in log-space
(``log_intensity_fn``) — integrating ``exp(log_intensity_fn(grid))`` is
more numerically stable than asking users to build a positive
``intensity_fn`` by hand, especially when the log-intensity is the
output of a neural network.

All functions are pure and vmap-safe: ``t0`` and ``t1`` can be
batched together.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


Method = Literal["trapezoid", "simpson"]


def integrate_log_intensity(
    log_intensity_fn: Callable[[Float[Array, ...]], Float[Array, ...]],
    t0: Float[Array, ...],
    t1: Float[Array, ...],
    n_points: int = 100,
    method: Method = "trapezoid",
) -> Float[Array, ...]:
    """Compute :math:`\\int_{t_0}^{t_1} \\exp(\\log\\lambda(s))\\, ds`.

    Args:
        log_intensity_fn: Callable mapping a 1-D ``(n_points,)`` array of
            times to an array of log-intensities with the same shape.
            Must support vectorised evaluation along the grid axis.
        t0: Lower limit (scalar or batched). Broadcast against ``t1``.
        t1: Upper limit (scalar or batched). Broadcast against ``t0``.
        n_points: Number of quadrature nodes (static Python int). Must
            be ``>= 2`` for trapezoid and ``>= 3`` (odd) for Simpson.
        method: Quadrature rule — ``"trapezoid"`` (default) or
            ``"simpson"``.

    Returns:
        Array with shape ``broadcast(t0, t1)`` giving the integral.

    Notes:
        For batched limits, each pair :math:`(t_0^{(i)}, t_1^{(i)})`
        gets its own linspace grid via ``vmap``; the user's
        ``log_intensity_fn`` only ever sees a 1-D grid of times.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if method == "simpson" and n_points % 2 == 0:
        # Composite Simpson's rule requires an odd number of nodes
        # (equivalently, an even number of sub-intervals).
        n_points = n_points + 1

    t0 = jnp.asarray(t0)
    t1 = jnp.asarray(t1)

    def _single(a: Array, b: Array) -> Array:
        grid = jnp.linspace(a, b, n_points)
        intensities = jnp.exp(log_intensity_fn(grid))
        if method == "trapezoid":
            return jnp.trapezoid(intensities, grid)
        # Simpson's composite rule on uniform grid.
        h = (b - a) / (n_points - 1)
        # Weight pattern: 1, 4, 2, 4, 2, ..., 4, 1.
        idx = jnp.arange(n_points)
        weights = jnp.where(
            (idx == 0) | (idx == n_points - 1),
            1.0,
            jnp.where(idx % 2 == 1, 4.0, 2.0),
        )
        return (h / 3.0) * jnp.sum(weights * intensities)

    broadcast_shape = jnp.broadcast_shapes(t0.shape, t1.shape)
    t0_b = jnp.broadcast_to(t0, broadcast_shape)
    t1_b = jnp.broadcast_to(t1, broadcast_shape)

    flat0 = t0_b.reshape(-1)
    flat1 = t1_b.reshape(-1)
    flat_result = jax.vmap(_single)(flat0, flat1)
    return flat_result.reshape(broadcast_shape)


def cumulative_log_intensity(
    log_intensity_fn: Callable[[Float[Array, ...]], Float[Array, ...]],
    t: Float[Array, ...],
    n_points: int = 100,
    method: Method = "trapezoid",
) -> Float[Array, ...]:
    """Shorthand for :math:`\\Lambda(t) = \\int_0^t \\lambda(s) ds`.

    Args:
        log_intensity_fn: Log-intensity callable (see
            :func:`integrate_log_intensity`).
        t: Upper limit(s). Must be ``>= 0``.
        n_points: Quadrature nodes.
        method: Quadrature rule.

    Returns:
        Cumulative intensity with the same shape as ``t``.
    """
    t = jnp.asarray(t)
    return integrate_log_intensity(
        log_intensity_fn, jnp.zeros_like(t), t, n_points=n_points, method=method
    )
