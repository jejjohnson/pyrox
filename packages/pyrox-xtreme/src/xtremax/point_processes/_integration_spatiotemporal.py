"""Quadrature over a space-time slab ``D × [t0, t1)``.

A spatiotemporal point process needs
:math:`\\Lambda = \\int_{t_0}^{t_1} \\int_D \\lambda(s, t)\\, ds\\, dt`
for the log-likelihood and for sizing thinning candidate buffers. We
support the same two strategies as the spatial integrator — tensor-
product trapezoid and Halton QMC — extended to ``d_space + 1``
dimensions by treating time as one extra axis. This keeps the
analytical guarantees (closed-form trapezoid weights, deterministic
QMC) without introducing a separate temporal quadrature path.

Both routines accept a callable ``log_intensity_fn(s, t)`` with
``s`` of shape ``(N, d_space)`` and ``t`` of shape ``(N,)``, returning
``(N,)`` log-intensities. This matches the source-snippet convention.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float

from xtremax.point_processes._domain import RectangularDomain, TemporalDomain
from xtremax.point_processes._integration_spatial import _halton_sequence


SpatiotemporalMethod = Literal["trapezoid", "qmc"]


def integrate_log_intensity_spatiotemporal(
    log_intensity_fn: Callable[
        [Float[Array, ...], Float[Array, ...]], Float[Array, ...]
    ],
    spatial: RectangularDomain,
    temporal: TemporalDomain,
    n_points: int = 4096,
    method: SpatiotemporalMethod = "qmc",
) -> Float[Array, ...]:
    """Compute :math:`\\int_{t_0}^{t_1} \\int_D \\exp(\\log\\lambda(s, t))\\, ds\\, dt`.

    Args:
        log_intensity_fn: Callable ``(s, t) → log λ(s, t)`` where ``s``
            has shape ``(N, d_space)`` and ``t`` has shape ``(N,)``.
        spatial: Spatial domain.
        temporal: Temporal interval.
        n_points: Static node budget across all ``d_space + 1`` axes.
            For ``trapezoid`` this is the total cell count
            (per-axis nodes are :math:`n^{1/(d+1)}` rounded up).
        method: ``"trapezoid"`` (closed form, best for ``d_space ≤ 2``)
            or ``"qmc"`` (Halton sequence, default).

    Returns:
        Scalar :math:`\\Lambda`, differentiable through
        ``log_intensity_fn``'s parameters.
    """
    if method == "trapezoid":
        return _integrate_trapezoid(log_intensity_fn, spatial, temporal, n_points)
    if method == "qmc":
        return _integrate_qmc(log_intensity_fn, spatial, temporal, n_points)
    raise ValueError(f"Unknown method {method!r}; expected 'trapezoid' or 'qmc'.")


def _integrate_trapezoid(
    log_intensity_fn: Callable[
        [Float[Array, ...], Float[Array, ...]], Float[Array, ...]
    ],
    spatial: RectangularDomain,
    temporal: TemporalDomain,
    n_points: int,
) -> Float[Array, ...]:
    """Tensor-product composite trapezoid on ``D × [t0, t1)``."""
    d_space = spatial.n_dims
    d_total = d_space + 1
    n_per_axis = max(2, math.ceil(n_points ** (1.0 / d_total)))

    spatial_grids = [
        jnp.linspace(spatial.lo[i], spatial.hi[i], n_per_axis) for i in range(d_space)
    ]
    time_grid = jnp.linspace(temporal.t0, temporal.t1, n_per_axis)

    h_spatial = jnp.array(
        [(spatial.hi[i] - spatial.lo[i]) / (n_per_axis - 1) for i in range(d_space)]
    )
    h_time = (temporal.t1 - temporal.t0) / (n_per_axis - 1)

    base_weight = jnp.ones(n_per_axis)
    base_weight = base_weight.at[0].set(0.5)
    base_weight = base_weight.at[-1].set(0.5)

    mesh = jnp.meshgrid(*spatial_grids, time_grid, indexing="ij")
    flat_s = jnp.stack([axis.reshape(-1) for axis in mesh[:d_space]], axis=-1)
    flat_t = mesh[-1].reshape(-1)

    flat_log_intensity = log_intensity_fn(flat_s, flat_t)
    intensities = jnp.exp(flat_log_intensity).reshape((n_per_axis,) * d_total)

    weight = base_weight
    for _ in range(d_total - 1):
        weight = jnp.tensordot(weight, base_weight, axes=0)

    cell_volume = jnp.prod(h_spatial) * h_time
    return jnp.sum(weight * intensities) * cell_volume


def _integrate_qmc(
    log_intensity_fn: Callable[
        [Float[Array, ...], Float[Array, ...]], Float[Array, ...]
    ],
    spatial: RectangularDomain,
    temporal: TemporalDomain,
    n_points: int,
) -> Float[Array, ...]:
    """Halton QMC over ``[lo, hi)^{d_space} × [t0, t1)``.

    Builds a single ``(d_space + 1)``-dimensional Halton sequence so
    spatial and temporal axes share a common quasi-random fill, instead
    of treating them as independent QMC streams (which would correlate
    poorly across the slab boundary).
    """
    d_space = spatial.n_dims
    d_total = d_space + 1
    unit_cube = _halton_sequence(n_points, d_total)
    s = spatial.lo + unit_cube[:, :d_space] * spatial.side_lengths
    t = temporal.t0 + unit_cube[:, -1] * temporal.duration
    log_intensities = log_intensity_fn(s, t)
    mean_intensity = jnp.mean(jnp.exp(log_intensities))
    return spatial.volume() * temporal.duration * mean_intensity
