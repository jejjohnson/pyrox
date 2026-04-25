"""Pure-JAX primitives for the inhomogeneous spatiotemporal Poisson process.

For an intensity :math:`\\lambda(s, t) > 0` on
:math:`D \\times [t_0, t_1)`, define
:math:`\\Lambda = \\int_{t_0}^{t_1}\\!\\!\\int_D \\lambda(s, t)\\, ds\\, dt`.
Then

* :math:`N \\sim \\mathrm{Poisson}(\\Lambda)`,
* conditional on ``n``, the sequence :math:`\\{(s_i, t_i)\\}` has joint
  density :math:`\\prod_i \\lambda(s_i, t_i) / \\Lambda`,
* Janossy log-likelihood
  :math:`\\sum_i \\log \\lambda(s_i, t_i) - \\Lambda`.

Sampling uses Lewis–Shedler thinning extended to ``d_space + 1`` axes:
draw a homogeneous candidate set on ``D × [t_0, t_1)`` at rate
:math:`\\lambda_{\\max}`, then thin by :math:`\\lambda(s, t)/\\lambda_{\\max}`.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._domain import RectangularDomain, TemporalDomain


def ipp_spatiotemporal_log_prob(
    locations: Float[Array, ...],
    times: Float[Array, ...],
    mask: Bool[Array, ...],
    log_intensity_fn: Callable[
        [Float[Array, ...], Float[Array, ...]], Float[Array, ...]
    ],
    integrated_intensity: Float[Array, ...],
) -> Float[Array, ...]:
    """Janossy log-likelihood :math:`\\sum_i \\log \\lambda(s_i, t_i) - \\Lambda`.

    Args:
        locations: ``(max_events, d_space)`` event locations.
        times: ``(max_events,)`` event times.
        mask: ``(max_events,)`` boolean mask of real events.
        log_intensity_fn: Callable ``(s, t) → log λ(s, t)``.
        integrated_intensity: Pre-computed
            :math:`\\Lambda = \\int_{t_0}^{t_1}\\!\\!\\int_D \\lambda\\, ds\\, dt`,
            typically from
            :func:`xtremax.point_processes.integrate_log_intensity_spatiotemporal`.

    Returns:
        Scalar log-likelihood.
    """
    log_lam = log_intensity_fn(locations, times)
    masked = jnp.where(mask, log_lam, 0.0)
    return jnp.sum(masked) - jnp.asarray(integrated_intensity)


def ipp_spatiotemporal_sample_thinning(
    key: PRNGKeyArray,
    log_intensity_fn: Callable[
        [Float[Array, ...], Float[Array, ...]], Float[Array, ...]
    ],
    spatial: RectangularDomain,
    temporal: TemporalDomain,
    lambda_max: Float[Array, ...],
    max_candidates: int,
) -> tuple[Float[Array, ...], Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
    """Lewis–Shedler thinning on ``D × [t_0, t_1)``.

    Args:
        key: PRNG key.
        log_intensity_fn: Callable ``(s, t) → log λ(s, t)``.
        spatial: Spatial domain.
        temporal: Temporal interval.
        lambda_max: Upper bound on :math:`\\lambda(s, t)` over the slab.
            **Must** be a true bound: a too-low ``lambda_max`` silently
            biases the sampler low in the intensity peaks.
        max_candidates: Static cap on candidate-buffer size; choose
            generously above :math:`\\lambda_{\\max} |D| T`.

    Returns:
        ``(locations, times, mask, n_events)`` with the same shape
        contract as :func:`hpp_spatiotemporal_sample`. Times are sorted
        ascending; locations follow the time order. ``n_events`` is the
        post-thinning count clipped to ``max_candidates``.
    """
    lambda_max = jnp.asarray(lambda_max)
    vol = spatial.volume()
    dur = temporal.duration

    key_n, key_locs, key_times, key_thin = random.split(key, 4)
    expected = lambda_max * vol * dur
    n_candidates = random.poisson(key_n, expected)

    ranks = jnp.arange(max_candidates)
    candidate_mask = ranks < n_candidates

    raw_locs = spatial.sample_uniform(key_locs, shape=(max_candidates,))
    raw_times = temporal.sample_uniform(key_times, shape=(max_candidates,))
    log_lam = log_intensity_fn(raw_locs, raw_times)

    # Acceptance probability λ(s, t) / λ_max.
    log_accept = log_lam - jnp.log(lambda_max)
    u = random.uniform(key_thin, shape=(max_candidates,))
    accepted = jnp.log(u) < log_accept
    keep = candidate_mask & accepted

    n_kept = jnp.sum(keep)

    # Sort kept events by time; push the rest to the tail.
    sort_key = jnp.where(keep, raw_times, jnp.inf)
    order = jnp.argsort(sort_key)
    sorted_times = raw_times[order]
    sorted_locs = raw_locs[order]
    sorted_keep = keep[order]

    # The post-thinning real count is at most ``max_candidates``; the
    # output mask records exactly which slots are real.
    final_mask = sorted_keep
    times = jnp.where(final_mask, sorted_times, temporal.t1)
    locations = jnp.where(final_mask[:, None], sorted_locs, spatial.lo)
    return locations, times, final_mask, n_kept


def ipp_spatiotemporal_intensity(
    locations: Float[Array, ...],
    times: Float[Array, ...],
    log_intensity_fn: Callable[
        [Float[Array, ...], Float[Array, ...]], Float[Array, ...]
    ],
) -> Float[Array, ...]:
    """Evaluate ``λ(s, t) = exp(log λ(s, t))`` on a batch of events."""
    return jnp.exp(log_intensity_fn(locations, times))


def ipp_spatiotemporal_predict_count(
    integrated_intensity: Float[Array, ...],
) -> tuple[Float[Array, ...], Float[Array, ...]]:
    """Mean and variance of the Poisson count over the integration domain."""
    lam = jnp.asarray(integrated_intensity)
    return lam, lam


def ipp_spatiotemporal_marginal_spatial_intensity(
    locations: Float[Array, ...],
    log_intensity_fn: Callable[
        [Float[Array, ...], Float[Array, ...]], Float[Array, ...]
    ],
    temporal: TemporalDomain,
    n_time_points: int = 100,
) -> Float[Array, ...]:
    """Compute :math:`\\lambda_S(s) = \\int_{t_0}^{t_1} \\lambda(s, t)\\, dt`.

    Trapezoid quadrature on a ``n_time_points``-uniform grid in time;
    vectorised over an arbitrary leading event axis.

    Args:
        locations: ``(N, d_space)`` query locations.
        log_intensity_fn: Callable ``(s, t) → log λ(s, t)`` accepting a
            shared spatial input broadcast against a time grid.
        temporal: Temporal interval.
        n_time_points: Static number of trapezoid nodes in time.

    Returns:
        ``(N,)`` array of marginal spatial intensities.
    """
    locations = jnp.asarray(locations)
    n_query = locations.shape[0]
    t_grid = jnp.linspace(temporal.t0, temporal.t1, n_time_points)
    # Tile locations across the time axis: (N, T, d_space) and (N, T).
    s_tiled = jnp.repeat(locations[:, None, :], n_time_points, axis=1)
    t_tiled = jnp.broadcast_to(t_grid, (n_query, n_time_points))
    s_flat = s_tiled.reshape(-1, locations.shape[-1])
    t_flat = t_tiled.reshape(-1)
    intensities = jnp.exp(log_intensity_fn(s_flat, t_flat)).reshape(
        n_query, n_time_points
    )
    h = (temporal.t1 - temporal.t0) / (n_time_points - 1)
    weights = jnp.ones(n_time_points).at[0].set(0.5).at[-1].set(0.5)
    return jnp.sum(intensities * weights, axis=-1) * h


def ipp_spatiotemporal_marginal_temporal_intensity(
    times: Float[Array, ...],
    log_intensity_fn: Callable[
        [Float[Array, ...], Float[Array, ...]], Float[Array, ...]
    ],
    spatial: RectangularDomain,
    n_spatial_points: int = 256,
) -> Float[Array, ...]:
    """Compute :math:`\\lambda_T(t) = \\int_D \\lambda(s, t)\\, ds`.

    Halton-QMC over the spatial domain at each query time; cheap and
    deterministic, and converges as :math:`O(\\log^d N / N)`.

    Args:
        times: ``(M,)`` query times.
        log_intensity_fn: Callable ``(s, t) → log λ(s, t)``.
        spatial: Spatial domain.
        n_spatial_points: Static number of QMC nodes.

    Returns:
        ``(M,)`` array of marginal temporal intensities.
    """
    from xtremax.point_processes._integration_spatial import _halton_sequence

    times = jnp.asarray(times)
    n_query = times.shape[0]
    unit_cube = _halton_sequence(n_spatial_points, spatial.n_dims)
    spatial_grid = spatial.lo + unit_cube * spatial.side_lengths

    # (M, n_spatial_points, d_space) and (M, n_spatial_points)
    s_tiled = jnp.broadcast_to(
        spatial_grid[None], (n_query, n_spatial_points, spatial.n_dims)
    )
    t_tiled = jnp.repeat(times[:, None], n_spatial_points, axis=1)
    s_flat = s_tiled.reshape(-1, spatial.n_dims)
    t_flat = t_tiled.reshape(-1)
    intensities = jnp.exp(log_intensity_fn(s_flat, t_flat)).reshape(
        n_query, n_spatial_points
    )
    return jnp.mean(intensities, axis=-1) * spatial.volume()


def ipp_spatiotemporal_intensity_surface_at_time(
    t: Float[Array, ...],
    log_intensity_fn: Callable[
        [Float[Array, ...], Float[Array, ...]], Float[Array, ...]
    ],
    spatial: RectangularDomain,
    grid_size: int = 50,
) -> tuple[Float[Array, ...], Float[Array, ...]]:
    """Evaluate ``λ(s, t)`` on a tensor-product spatial grid at fixed ``t``.

    Args:
        t: Scalar query time.
        log_intensity_fn: Callable ``(s, t) → log λ(s, t)``.
        spatial: Spatial domain.
        grid_size: Static per-axis grid size.

    Returns:
        ``(grid, intensity)`` where ``grid`` has shape
        ``(grid_size, ..., grid_size, d_space)`` and ``intensity`` has
        shape ``(grid_size, ..., grid_size)``.
    """
    d = spatial.n_dims
    axis_grids = [
        jnp.linspace(spatial.lo[i], spatial.hi[i], grid_size) for i in range(d)
    ]
    mesh = jnp.stack(jnp.meshgrid(*axis_grids, indexing="ij"), axis=-1)
    flat_s = mesh.reshape(-1, d)
    flat_t = jnp.full((flat_s.shape[0],), t)
    log_lam = log_intensity_fn(flat_s, flat_t)
    intensity = jnp.exp(log_lam).reshape((grid_size,) * d)
    return mesh, intensity


def ipp_spatiotemporal_pearson_residuals(
    locations: Float[Array, ...],
    times: Float[Array, ...],
    mask: Bool[Array, ...],
    log_intensity_fn: Callable[
        [Float[Array, ...], Float[Array, ...]], Float[Array, ...]
    ],
    spatial: RectangularDomain,
    temporal: TemporalDomain,
    n_spatial_bins: int = 5,
    n_temporal_bins: int = 5,
    n_integration_points: int = 16,
) -> Float[Array, ...]:
    """Pearson residuals on a tensor-product space-time grid.

    For each cell ``A_k = R_k × [τ_k, τ_{k+1})``,

    .. math::
        r_k = \\frac{N(A_k) - \\Lambda(A_k)}{\\sqrt{\\Lambda(A_k)}}.

    Cells with :math:`\\Lambda(A_k) = 0` return ``0`` to keep the
    residual array finite.

    Args:
        locations: ``(max_events, d_space)`` event locations.
        times: ``(max_events,)`` event times.
        mask: ``(max_events,)`` boolean mask of real events.
        log_intensity_fn: Callable ``(s, t) → log λ(s, t)``.
        spatial: Spatial domain.
        temporal: Temporal interval.
        n_spatial_bins: Bins per spatial axis (so ``n_spatial_bins ** d``
            spatial cells total).
        n_temporal_bins: Bins along the time axis.
        n_integration_points: Per-cell trapezoid nodes per axis when
            estimating :math:`\\Lambda(A_k)`.

    Returns:
        Flattened residual array of shape
        ``(n_temporal_bins * n_spatial_bins ** d,)``.
    """
    d = spatial.n_dims
    s_edges = [
        jnp.linspace(spatial.lo[i], spatial.hi[i], n_spatial_bins + 1) for i in range(d)
    ]
    t_edges = jnp.linspace(temporal.t0, temporal.t1, n_temporal_bins + 1)

    # Bucket the events. Use jnp.digitize per axis with right-open bins.
    spatial_idx = jnp.stack(
        [
            jnp.clip(
                jnp.digitize(locations[:, i], s_edges[i]) - 1, 0, n_spatial_bins - 1
            )
            for i in range(d)
        ],
        axis=-1,
    )  # (N, d)
    temporal_idx = jnp.clip(jnp.digitize(times, t_edges) - 1, 0, n_temporal_bins - 1)

    # Encode (spatial_idx, temporal_idx) → flat cell index
    # row-major (temporal slowest, spatial axes packed in C order).
    spatial_flat = jnp.zeros_like(temporal_idx)
    stride = 1
    for axis in range(d - 1, -1, -1):
        spatial_flat = spatial_flat + spatial_idx[:, axis] * stride
        stride = stride * n_spatial_bins
    cell_idx = temporal_idx * (n_spatial_bins**d) + spatial_flat

    n_cells = n_temporal_bins * (n_spatial_bins**d)
    counts = jnp.zeros(n_cells)
    # Scatter-add a per-event weight: 1 for real events, 0 for padding.
    # Padding rows still scatter into a real cell (the digitize/clip
    # path collapses them to the boundary cell), but the 0 weight
    # leaves it untouched. The earlier sentinel-bucket trick was a
    # no-op under JAX's negative-index wrap, so drop it (Codex P1 in
    # PR #13 review).
    counts = counts.at[cell_idx].add(jnp.where(mask, 1.0, 0.0))

    # Per-cell integrated intensity by trapezoid quadrature on a small
    # tensor-product grid inside each cell.
    def cell_lambda(temporal_bin: int, spatial_bins: tuple[int, ...]) -> Array:
        sub_lo = jnp.array([s_edges[i][spatial_bins[i]] for i in range(d)])
        sub_hi = jnp.array([s_edges[i][spatial_bins[i] + 1] for i in range(d)])
        sub_t0 = t_edges[temporal_bin]
        sub_t1 = t_edges[temporal_bin + 1]

        n = n_integration_points
        spatial_grids = [jnp.linspace(sub_lo[i], sub_hi[i], n) for i in range(d)]
        time_grid = jnp.linspace(sub_t0, sub_t1, n)
        h_spatial = jnp.array([(sub_hi[i] - sub_lo[i]) / (n - 1) for i in range(d)])
        h_time = (sub_t1 - sub_t0) / (n - 1)

        base_w = jnp.ones(n).at[0].set(0.5).at[-1].set(0.5)
        mesh = jnp.meshgrid(*spatial_grids, time_grid, indexing="ij")
        flat_s = jnp.stack([axis.reshape(-1) for axis in mesh[:d]], axis=-1)
        flat_t = mesh[-1].reshape(-1)
        intensities = jnp.exp(log_intensity_fn(flat_s, flat_t)).reshape((n,) * (d + 1))

        weight = base_w
        for _ in range(d):  # d + 1 axes total → d more outer products
            weight = jnp.tensordot(weight, base_w, axes=0)
        cell_volume = jnp.prod(h_spatial) * h_time
        return jnp.sum(weight * intensities) * cell_volume

    # Build the per-cell λ list eagerly. The bin counts are static
    # Python ints so this is unrolled at trace time and fine for the
    # typical small ``n_*_bins`` choices we expect for residuals.
    spatial_index_grid = jnp.stack(
        jnp.meshgrid(*[jnp.arange(n_spatial_bins) for _ in range(d)], indexing="ij"),
        axis=-1,
    ).reshape(-1, d)
    expected = []
    for tb in range(n_temporal_bins):
        for sb in spatial_index_grid:
            expected.append(cell_lambda(tb, tuple(int(x) for x in sb)))
    expected_arr = jnp.stack(expected)

    safe_expected = jnp.where(expected_arr > 0.0, expected_arr, 1.0)
    residuals = (counts - expected_arr) / jnp.sqrt(safe_expected)
    return jnp.where(expected_arr > 0.0, residuals, 0.0)


def ipp_spatiotemporal_chi_square_gof(
    locations: Float[Array, ...],
    times: Float[Array, ...],
    mask: Bool[Array, ...],
    log_intensity_fn: Callable[
        [Float[Array, ...], Float[Array, ...]], Float[Array, ...]
    ],
    spatial: RectangularDomain,
    temporal: TemporalDomain,
    n_spatial_bins: int = 5,
    n_temporal_bins: int = 5,
    n_integration_points: int = 16,
) -> Float[Array, ...]:
    """Pearson :math:`\\chi^2` statistic from binned residuals.

    Returns :math:`\\sum_k r_k^2`. Compare to a :math:`\\chi^2_{K - p}`
    distribution with degrees of freedom equal to the number of cells
    minus the number of fitted parameters.
    """
    residuals = ipp_spatiotemporal_pearson_residuals(
        locations,
        times,
        mask,
        log_intensity_fn,
        spatial,
        temporal,
        n_spatial_bins=n_spatial_bins,
        n_temporal_bins=n_temporal_bins,
        n_integration_points=n_integration_points,
    )
    return jnp.sum(residuals**2)
