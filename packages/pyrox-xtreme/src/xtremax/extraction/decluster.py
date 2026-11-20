from __future__ import annotations

from collections.abc import Callable

import numpy as np
import xarray as xr


_REDUCERS: dict[str, Callable[[np.ndarray], np.floating]] = {
    "max": np.max,
    "mean": np.mean,
    "sum": np.sum,
    "min": np.min,
}


def decluster_runs(
    da: xr.DataArray,
    threshold: float | xr.DataArray,
    dim: str = "time",
    reduction: str = "max",
    min_separation: int = 1,
) -> xr.DataArray:
    """
    Decluster exceedances using the runs method.

    Identifies runs of exceedances and applies a reduction function to
    each run. This ensures temporal independence by treating each
    cluster as a single event.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    threshold : float or xr.DataArray
        Threshold value(s)
    dim : str, default 'time'
        Dimension along which to identify runs
    reduction : str, default 'max'
        Reduction function to apply to each run: 'max', 'mean', 'sum', 'min'
    min_separation : int, default 1
        Runs-declustering separation parameter *r*: two exceedances
        belong to the same cluster when fewer than ``min_separation``
        consecutive below-threshold steps separate them. The default 1
        reproduces classical contiguous runs (any single dip below the
        threshold ends the run).

    Returns
    -------
    xr.DataArray
        Array with the same shape as ``da``. Run representatives are placed
        at the last exceedance position of each run along ``dim`` and all
        other positions are NaN. For multi-dimensional inputs, each
        non-``dim`` slice is declustered independently (so run IDs never
        collide across slices).

    Examples
    --------
    >>> # Maximum value from each exceedance cluster
    >>> declustered = decluster_runs(da, threshold=100, reduction='max')

    >>> # Merge exceedances separated by gaps of fewer than 3 steps
    >>> declustered = decluster_runs(da, threshold=100, min_separation=3)

    Notes
    -----
    The runs method identifies periods where values exceed the
    threshold, merging exceedances whose below-threshold gap is shorter
    than ``min_separation`` into one cluster, and extracts one
    representative value per cluster. The reduction is applied to the
    cluster's *exceedance* values only — intra-cluster below-threshold
    samples do not contribute.
    """
    if reduction not in _REDUCERS:
        raise ValueError(f"Unknown reduction: {reduction}")
    if min_separation < 1:
        raise ValueError(f"min_separation must be >= 1, got {min_separation}")
    reducer = _REDUCERS[reduction]

    exceedances = da > threshold

    def _decluster_runs_1d(values: np.ndarray, exc: np.ndarray) -> np.ndarray:
        """Reduce each run of exceedances to a single value.

        The representative is stored at the last exceedance position of
        its run; all other entries are NaN. Operating per 1-D slice keeps
        run IDs from different batch rows (e.g. different ``site``s) from
        colliding.
        """
        out = np.full(values.shape[0], np.nan, dtype=float)
        positions = np.flatnonzero(exc)
        if positions.size == 0:
            return out
        # A gap of >= min_separation below-threshold steps between
        # consecutive exceedances starts a new cluster.
        breaks = np.flatnonzero(np.diff(positions) - 1 >= min_separation) + 1
        for cluster in np.split(positions, breaks):
            out[cluster[-1]] = reducer(values[cluster])
        return out

    return xr.apply_ufunc(
        _decluster_runs_1d,
        da,
        exceedances,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        output_dtypes=[float],
    )


def decluster_separation(
    da: xr.DataArray,
    threshold: float | xr.DataArray,
    min_separation: int,
    dim: str = "time",
) -> xr.DataArray:
    """
    Decluster by enforcing minimum separation between peaks.

    Identifies peaks above threshold and ensures they are separated by
    at least min_separation time steps. When peaks are too close, keeps
    the larger one.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    threshold : float or xr.DataArray
        Threshold value(s)
    min_separation : int
        Minimum separation (in time steps) between independent peaks
    dim : str, default 'time'
        Dimension along which to enforce separation

    Returns
    -------
    xr.DataArray
        Declustered peak values

    Examples
    --------
    >>> # Peaks separated by at least 3 days
    >>> declustered = decluster_separation(da, threshold=100, min_separation=3)

    >>> # Storm events with 7-day minimum separation
    >>> storms = decluster_separation(da, threshold=95, min_separation=7)

    Notes
    -----
    This method first identifies all local maxima above threshold,
    then iteratively removes peaks that are too close to larger peaks.

    Plateaus (runs of equal consecutive values) count as a single peak
    when the values adjacent to the entire run are strictly lower — the
    peak is stamped at the run's first sample. Quantized observational
    data (rounded gauges, integer records) hit this case routinely;
    requiring strict inequality between adjacent samples used to drop
    such events entirely, while a naive adjacent-sample tie-break would
    also mark rising shelves (e.g. ``[0, 5, 5, 6]``) as peaks.
    """
    # Identify local maxima above threshold.
    #
    # `da.shift` fills the edge with NaN, so comparisons at the first /
    # last positions along `dim` silently evaluate to False. Previously
    # this meant boundary elements could NEVER be marked as peaks — a
    # silent event drop for short windows or chunked workflows. Treat a
    # missing neighbour as "no constraint from that side": a boundary
    # position is a peak if it is strictly greater than its single
    # existing neighbour (and above threshold).
    # Plateau-aware peak detection (equal consecutive maxima): group the
    # series into maximal runs of equal values; a run is a peak iff the
    # values adjacent to the *entire* run are strictly lower (a missing
    # neighbour at the series boundary is no constraint). The peak is
    # stamped at the run's first sample. Comparing only adjacent samples
    # either dropped flat-topped exceedances entirely (strict on both
    # sides) or marked rising shelves like [0, 5, 5, 6] as peaks
    # (>= on one side).
    exceedances = da > threshold

    def _plateau_peaks_1d(values: np.ndarray, exc: np.ndarray) -> np.ndarray:
        n = values.shape[0]
        peaks = np.zeros(n, dtype=bool)
        if n == 0:
            return peaks
        # Run starts: first index, then every value change. NaN != NaN,
        # so NaN samples form their own (never-exceeding) runs.
        change = np.concatenate(([True], values[1:] != values[:-1]))
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:] - 1, n - 1)
        left_ok = (starts == 0) | (values[np.maximum(starts - 1, 0)] < values[starts])
        right_ok = (ends == n - 1) | (
            values[np.minimum(ends + 1, n - 1)] < values[starts]
        )
        peak_starts = starts[left_ok & right_ok & exc[starts]]
        peaks[peak_starts] = True
        return peaks

    is_peak = xr.apply_ufunc(
        _plateau_peaks_1d,
        da,
        exceedances,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        output_dtypes=[bool],
    )

    def _select_separated_peaks_1d(
        values: np.ndarray, peak_mask: np.ndarray
    ) -> np.ndarray:
        """Greedily pick peaks so retained maxima are ≥ min_separation apart.

        Works purely in positional indices along the core axis, which is
        what `min_separation` is defined against (time steps, not
        elements of the filtered peak list).
        """
        peak_positions = np.flatnonzero(peak_mask)
        selected = np.zeros_like(peak_mask, dtype=bool)
        if peak_positions.size == 0:
            return selected
        peak_values = values[peak_positions]
        order = np.argsort(peak_values)[::-1]
        chosen: list[int] = []
        for idx in peak_positions[order]:
            pos = int(idx)
            if all(abs(pos - s) >= min_separation for s in chosen):
                chosen.append(pos)
                selected[pos] = True
        return selected

    # `apply_ufunc` with core dim == `dim` vectorises the 1-D selector
    # over any remaining batch dims (e.g. `site`), so each batch row
    # gets its own separation-filtered peak set and peak indices stay
    # valid positions along `dim`.
    selected_mask = xr.apply_ufunc(
        _select_separated_peaks_1d,
        da,
        is_peak,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        output_dtypes=[bool],
    )

    return da.where(selected_mask, drop=True)


def estimate_extremal_index(
    da: xr.DataArray,
    threshold: float | xr.DataArray,
    dim: str = "time",
    method: str = "runs",
    min_separation: int = 1,
) -> float | xr.DataArray:
    """
    Estimate the extremal index (theta), a measure of clustering.

    The extremal index θ ∈ [0, 1] measures the degree of clustering in extremes:
    - θ = 1: no clustering (independent extremes)
    - θ < 1: clustering present (dependent extremes)
    - θ = 0: complete clustering (perfect dependence)

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    threshold : float or xr.DataArray
        Threshold value for identifying extremes
    dim : str, default 'time'
        Dimension along which to estimate clustering
    method : str, default 'runs'
        Estimation method: 'runs' or 'intervals'
    min_separation : int, default 1
        Runs parameter *r*: exceedances separated by fewer than
        ``min_separation`` consecutive below-threshold steps count as
        the same cluster. The default 1 reproduces the classical
        contiguous-runs estimator.

    Returns
    -------
    float or xr.DataArray
        Extremal index estimate (between 0 and 1). For 1-D input a scalar
        ``float``. For inputs with additional dimensions, a DataArray of
        per-slice estimates (one θ per batch row, e.g. per site).

    Examples
    --------
    >>> # Estimate clustering strength
    >>> theta = estimate_extremal_index(da, threshold=100)
    >>> print(f"Extremal index: {theta:.3f}")
    >>> # theta ≈ 0.7 suggests moderate clustering

    >>> # Use high threshold (99th percentile)
    >>> from xtremax.extraction.threshold import quantile_threshold
    >>> thresh = quantile_threshold(da, 0.99)
    >>> theta = estimate_extremal_index(da, thresh)

    Notes
    -----
    The runs estimator uses: θ = (number of clusters) / (number of exceedances)
    A smaller θ indicates more clustering. Counting is performed
    independently per non-``dim`` slice so cluster labels from different
    slices cannot collide.
    """
    if method == "intervals":
        raise NotImplementedError("Intervals method not yet implemented")
    if method != "runs":
        raise ValueError(f"Unknown method: {method}")
    if min_separation < 1:
        raise ValueError(f"min_separation must be >= 1, got {min_separation}")

    exceedances = da > threshold

    def _theta_runs_1d(exc: np.ndarray) -> float:
        positions = np.flatnonzero(exc)
        if positions.size == 0:
            return np.nan
        # Count clusters: a new cluster starts whenever the gap of
        # consecutive below-threshold steps reaches min_separation.
        n_runs = 1 + int(np.sum(np.diff(positions) - 1 >= min_separation))
        return n_runs / positions.size

    theta = xr.apply_ufunc(
        _theta_runs_1d,
        exceedances,
        input_core_dims=[[dim]],
        output_core_dims=[[]],
        vectorize=True,
        output_dtypes=[float],
    )
    # Preserve the historical 1-D scalar return type.
    if theta.ndim == 0:
        return float(theta.values)
    return theta
