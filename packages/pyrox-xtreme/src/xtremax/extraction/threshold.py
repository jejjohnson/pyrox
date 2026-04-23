"""Threshold selection for peaks-over-threshold (POT) analysis.

Simple threshold selectors: constant, empirical quantile, temporal
groupby quantile (annual / seasonal / monthly), rolling quantile, and
seasonal quantile. Quantile-regression thresholds (linear trend and
covariate-driven) live in :mod:`xtremax.extraction.quantile_regression`.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import xarray as xr


def constant_threshold(da: xr.DataArray, threshold: float) -> float:
    """
    Return a constant threshold value.

    This is the simplest threshold selection method, using a fixed value
    across all times and locations.

    Parameters
    ----------
    da : xr.DataArray
        Input data array (included for API consistency)
    threshold : float
        Constant threshold value

    Returns
    -------
    float
        The threshold value

    Examples
    --------
    >>> # Use a constant threshold of 100
    >>> thresh = constant_threshold(da, 100)
    """
    return threshold


def quantile_threshold(
    da: xr.DataArray,
    quantile: float,
    dim: Hashable | Sequence[Hashable] | None = None,
) -> float | xr.DataArray:
    """
    Calculate threshold based on a percentile of the data.

    Common choices are 0.95 (95th percentile) or 0.99 (99th percentile).
    Higher quantiles result in fewer but more extreme exceedances.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    quantile : float
        Quantile value between 0 and 1 (e.g., 0.95 for 95th percentile)
    dim : Hashable or Sequence[Hashable], optional
        Dimension(s) over which to compute quantile. If None, computes
        over all dimensions returning a scalar.

    Returns
    -------
    float or xr.DataArray
        Threshold value(s)

    Examples
    --------
    >>> # 95th percentile threshold
    >>> thresh = quantile_threshold(da, 0.95)

    >>> # 99th percentile, computed separately for each location
    >>> thresh = quantile_threshold(da, 0.99, dim='time')
    """
    return da.quantile(quantile, dim=dim)


def temporal_threshold(
    da: xr.DataArray,
    quantile: float,
    groupby: str = "time.season",
    time_dim: str = "time",
    method: str = "quantile",
) -> xr.DataArray:
    """
    Calculate time-varying threshold based on temporal grouping.

    Computes threshold separately for each temporal group (season, month,
    year, week, dayofyear, etc.) using groupby operations. More flexible
    than rolling window approach.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with time dimension
    quantile : float
        Quantile value between 0 and 1 (e.g., 0.95 for 95th percentile)
    groupby : str, default 'time.season'
        Grouping specification. Common options:
        - 'time.season' : seasonal (DJF, MAM, JJA, SON)
        - 'time.month' : monthly (1-12)
        - 'time.year' : annual
        - 'time.dayofyear' : day of year (1-365/366)
        - 'time.week' : week of year
        - 'time.hour' : hour of day
    time_dim : str, default 'time'
        Name of the time dimension
    method : str, default 'quantile'
        Aggregation method: 'quantile', 'mean', 'median', 'max', 'min'

    Returns
    -------
    xr.DataArray
        Time-varying threshold values, aligned with original time coordinate

    Examples
    --------
    >>> # Seasonal 95th percentile thresholds
    >>> thresh = temporal_threshold(da, 0.95, groupby='time.season')

    >>> # Monthly 99th percentile thresholds
    >>> thresh = temporal_threshold(da, 0.99, groupby='time.month')

    >>> # Day-of-year thresholds (smooth seasonal cycle)
    >>> thresh = temporal_threshold(da, 0.95, groupby='time.dayofyear')

    >>> # Annual maximum
    >>> thresh = temporal_threshold(da, 0.95, groupby='time.year', method='max')

    >>> # Weekly mean thresholds
    >>> thresh = temporal_threshold(da, 0.9, groupby='time.week', method='mean')

    Notes
    -----
    The function groups data by the specified temporal accessor (e.g., month),
    computes the threshold for each group, then broadcasts back to the original
    time coordinate. This is useful for capturing regular temporal patterns in
    the tail distribution.

    For day-of-year grouping, this gives a smooth seasonal cycle of thresholds.
    For month/season, this gives discrete jumps between periods.
    """
    # Group by temporal accessor
    grouped = da.groupby(groupby)

    # Compute threshold for each group
    if method == "quantile":
        threshold_by_group = grouped.quantile(quantile, dim=time_dim)
    elif method == "mean":
        threshold_by_group = grouped.mean(dim=time_dim)
    elif method == "median":
        threshold_by_group = grouped.median(dim=time_dim)
    elif method == "max":
        threshold_by_group = grouped.max(dim=time_dim)
    elif method == "min":
        threshold_by_group = grouped.min(dim=time_dim)
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Choose from: 'quantile', 'mean', 'median', 'max', 'min'"
        )

    # Broadcast threshold back to original time coordinate
    # This ensures threshold has same shape as input
    threshold_expanded = threshold_by_group.sel({groupby.split(".")[-1]: da[groupby]})

    return threshold_expanded


def rolling_threshold(
    da: xr.DataArray,
    quantile: float,
    window_size: int = 15,
    time_dim: str = "time",
    center: bool = True,
    min_periods: int | None = None,
) -> xr.DataArray:
    """
    Calculate time-varying threshold using a rolling window.

    Uses a rolling window to compute a threshold that varies smoothly
    over time. Useful for removing short-term variations while preserving
    longer-term trends.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with time dimension
    quantile : float
        Quantile value between 0 and 1
    window_size : int, default 15
        Size of rolling window (in time steps) for computing the threshold
    time_dim : str, default 'time'
        Name of the time dimension
    center : bool, default True
        If True, set window labels at center of window
    min_periods : int, optional
        Minimum number of observations required in window. If None, uses window_size.

    Returns
    -------
    xr.DataArray
        Time-varying threshold values with smooth temporal variation

    Examples
    --------
    >>> # Rolling 15-day 95th percentile threshold
    >>> thresh = rolling_threshold(da, 0.95, window_size=15)

    >>> # Rolling 30-day 99th percentile threshold
    >>> thresh = rolling_threshold(da, 0.99, window_size=30)

    >>> # Non-centered window (past values only)
    >>> thresh = rolling_threshold(da, 0.95, window_size=30, center=False)

    Notes
    -----
    Rolling window approach provides smoother temporal variation than
    groupby methods. Good for removing high-frequency variations while
    keeping the seasonal cycle.
    """
    if min_periods is None:
        min_periods = window_size

    rolling = da.rolling(
        {time_dim: window_size}, center=center, min_periods=min_periods
    )
    # xarray removed DataArrayRolling.quantile; materialize the window axis
    # with construct and take the quantile along it.
    windowed = rolling.construct("_window")
    return windowed.quantile(quantile, dim="_window")


# Alias for backward compatibility
def seasonal_threshold(
    da: xr.DataArray, quantile: float = 0.95, time_dim: str = "time"
) -> xr.DataArray:
    """
    Calculate seasonal threshold (backward compatibility wrapper).

    This is a convenience wrapper around temporal_threshold with
    groupby='time.season'. For more control, use temporal_threshold directly.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    quantile : float, default 0.95
        Quantile value between 0 and 1
    time_dim : str, default 'time'
        Name of the time dimension

    Returns
    -------
    xr.DataArray
        Seasonal threshold values

    Examples
    --------
    >>> # 95th percentile by season
    >>> thresh = seasonal_threshold(da, 0.95)

    See Also
    --------
    temporal_threshold : More flexible temporal grouping
    rolling_threshold : Smooth rolling window threshold
    """
    # Derive the groupby key from `time_dim` instead of hard-coding
    # "time.season"; otherwise datasets whose temporal axis is named
    # something else (e.g. "date") fail despite the API claiming support
    # via the `time_dim` argument.
    return temporal_threshold(
        da, quantile, groupby=f"{time_dim}.season", time_dim=time_dim
    )
