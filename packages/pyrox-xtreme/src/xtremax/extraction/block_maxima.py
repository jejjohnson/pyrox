from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import xarray as xr


def temporal_block_maxima(
    da: xr.DataArray, freq: str, time_dim: str = "time", min_periods: int | None = None
) -> xr.DataArray:
    """
    Extract temporal block maximum values from an xarray DataArray.

    Uses time-based resampling to compute maximum values over temporal blocks.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    freq : str
        Frequency string for temporal grouping (e.g., '1Y' for annual,
        '1M' for monthly, '7D' for weekly, '3H' for 3-hourly)
    time_dim : str, default 'time'
        Name of the time dimension
    min_periods : int, optional
        Minimum number of observations required for a valid maximum.
        If None, all blocks are used regardless of missing data.

    Returns
    -------
    xr.DataArray
        Temporal block maximum values

    Examples
    --------
    >>> # Annual maxima
    >>> annual_max = temporal_block_maxima(da, '1Y')

    >>> # Monthly maxima with at least 20 observations per month
    >>> monthly_max = temporal_block_maxima(da, '1M', min_periods=20)

    >>> # Seasonal maxima (3-month blocks)
    >>> seasonal_max = temporal_block_maxima(da, '3M')
    """
    resampler = da.resample({time_dim: freq})

    if min_periods is not None:
        # Count valid observations in each block
        counts = resampler.count(dim=time_dim)
        maxima = resampler.max(dim=time_dim, keep_attrs=True)
        # Mask blocks with insufficient data
        maxima = maxima.where(counts >= min_periods)
        return maxima
    else:
        return resampler.max(dim=time_dim, keep_attrs=True)


def spatial_block_maxima(
    da: xr.DataArray,
    block_size: int | dict[Hashable, int],
    dims: Hashable | Sequence[Hashable] = None,
    min_periods: int | None = None,
    boundary: str = "trim",
) -> xr.DataArray:
    """
    Extract spatial block maximum values from an xarray DataArray.

    Uses coarsening to compute maximum values over spatial blocks of fixed size.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    block_size : int or dict
        Size of spatial blocks. If int, uses same size for all specified dims.
        If dict, maps dimension names to block sizes (e.g., {'x': 10, 'y': 10})
    dims : Hashable or Sequence[Hashable], optional
        Dimension(s) to coarsen. If None and block_size is int,
        raises error. If None and block_size is dict, uses dict keys.
    min_periods : int, optional
        Minimum number of valid observations required within each block.
        If None, computes maximum even with missing data.
    boundary : str, default 'trim'
        How to handle boundaries. Options: 'trim', 'pad', 'exact'

    Returns
    -------
    xr.DataArray
        Spatial block maximum values with reduced resolution

    Examples
    --------
    >>> # 10x10 spatial blocks
    >>> spatial_max = spatial_block_maxima(da, {'x': 10, 'y': 10})

    >>> # 5x5 blocks along lat/lon dimensions
    >>> spatial_max = spatial_block_maxima(da, 5, dims=['lat', 'lon'])

    >>> # Require at least 50 valid points per block
    >>> spatial_max = spatial_block_maxima(
    ...     da, {'x': 10, 'y': 10}, min_periods=50
    ... )
    """
    # Handle block_size specification
    if isinstance(block_size, dict):
        coarsen_kwargs = block_size
    else:
        if dims is None:
            raise ValueError("Must specify 'dims' when block_size is an integer")
        if isinstance(dims, str):
            dims = [dims]
        coarsen_kwargs = {dim: block_size for dim in dims}

    # Create coarsen object
    coarsened = da.coarsen(dim=coarsen_kwargs, boundary=boundary)

    if min_periods is not None:
        # Count valid observations in each block
        counts = coarsened.count()
        maxima = coarsened.max(keep_attrs=True)
        # Mask blocks with insufficient data
        maxima = maxima.where(counts >= min_periods)
        return maxima
    else:
        return coarsened.max(keep_attrs=True)


def sliding_block_maxima(
    da: xr.DataArray,
    window_size: int,
    dim: str = "time",
    stride: int = 1,
    min_periods: int | None = None,
    center: bool = False,
) -> xr.DataArray:
    """
    Extract block maxima using a sliding window approach.

    Sliding windows provide overlapping blocks, which helps:
    - Avoid sensitivity to arbitrary block boundaries
    - Increase the number of block maxima for better statistical estimation
    - Capture extreme events that might span fixed block boundaries

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    window_size : int
        Size of the sliding window (number of observations)
    dim : str, default 'time'
        Dimension along which to slide the window
    stride : int, default 1
        Step size between windows. stride=1 gives fully overlapping windows,
        stride=window_size gives non-overlapping blocks (equivalent to coarsen)
    min_periods : int, optional
        Minimum number of valid observations required for a valid maximum.
        If None, defaults to window_size.
    center : bool, default False
        If True, set the window labels at the center of the window

    Returns
    -------
    xr.DataArray
        Sliding block maximum values

    Examples
    --------
    >>> # 365-day sliding window with daily stride
    >>> sliding_max = sliding_block_maxima(da, window_size=365, stride=1)

    >>> # Non-overlapping 30-day blocks (equivalent to fixed blocks)
    >>> block_max = sliding_block_maxima(da, window_size=30, stride=30)

    >>> # 90-day window with 7-day stride (weekly updates)
    >>> weekly_max = sliding_block_maxima(da, window_size=90, stride=7)
    """
    if min_periods is None:
        min_periods = window_size

    # Use rolling window
    rolling = da.rolling({dim: window_size}, center=center, min_periods=min_periods)
    maxima = rolling.max()

    # Apply stride by subsampling
    if stride > 1:
        maxima = maxima.isel({dim: slice(None, None, stride)})

    return maxima


def declustered_block_maxima(
    da: xr.DataArray,
    threshold: float,
    min_separation: int,
    dim: str = "time",
    method: str = "runs",
) -> xr.DataArray:
    """
    Extract declustered block maxima by identifying independent extreme events.

    Declustering removes temporal dependence in extremes by ensuring sufficient
    separation between events. This is crucial for extreme value theory which
    assumes independence.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    threshold : float
        Threshold value for identifying extreme events
    min_separation : int
        Minimum separation (in time steps) between independent events
    dim : str, default 'time'
        Dimension along which to decluster
    method : str, default 'runs'
        Declustering method:
        - 'runs': Select maximum from each exceedance run
        - 'separation': Enforce minimum time separation between peaks

    Returns
    -------
    xr.DataArray
        Declustered block maxima with reduced temporal dependence

    Examples
    --------
    >>> # Extract independent storm maxima (3-day separation)
    >>> storm_max = declustered_block_maxima(
    ...     da, threshold=100, min_separation=3, method='runs'
    ... )

    >>> # Heat wave maxima with 7-day separation
    >>> heatwave_max = declustered_block_maxima(
    ...     da, threshold=35, min_separation=7, method='separation'
    ... )

    Notes
    -----
    The runs method identifies continuous periods above threshold and takes
    the maximum from each run. The separation method ensures peaks are
    separated by at least min_separation time steps.
    """
    from xtremax.extraction.decluster import decluster_runs, decluster_separation

    if method == "runs":
        # Delegate so run IDs are computed per non-`dim` slice and cannot
        # collide across batch rows (e.g. different sites).
        reduced = decluster_runs(da, threshold=threshold, dim=dim, reduction="max")
        return reduced.dropna(dim, how="all")

    elif method == "separation":
        # Delegate to the separation-based declustering helper, which
        # actually applies `min_separation` in units of original time steps.
        return decluster_separation(
            da, threshold=threshold, min_separation=min_separation, dim=dim
        )

    else:
        raise ValueError(f"Unknown method: {method}. Use 'runs' or 'separation'")


def r_largest_block_maxima(
    da: xr.DataArray,
    block_size: int | str,
    r: int = 3,
    dim: str = "time",
    min_periods: int | None = None,
) -> xr.DataArray:
    """
    Extract r-largest order statistics from each block.

    Instead of just the maximum, extract the r largest values from each block.
    This provides more information for extreme value analysis and can improve
    statistical efficiency.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    block_size : int or str
        Size of blocks. If int, number of elements. If str, frequency string.
    r : int, default 3
        Number of largest values to extract from each block
    dim : str, default 'time'
        Dimension along which to compute order statistics
    min_periods : int, optional
        Minimum number of observations required in a block

    Returns
    -------
    xr.DataArray
        Array with r-largest values from each block, with new dimension 'order'

    Examples
    --------
    >>> # 3 largest values from each year
    >>> annual_r_largest = r_largest_block_maxima(da, '1Y', r=3)

    >>> # Top 5 values from 100-element blocks
    >>> block_r_largest = r_largest_block_maxima(da, 100, r=5)

    Notes
    -----
    The r-largest order statistics model is useful when you want to use
    more information than just the block maximum for fitting extreme
    value distributions.
    """

    def _top_r_1d(values: np.ndarray) -> np.ndarray:
        """Top-r descending from a 1-D array, NaN-padded to length r.

        Operating per 1-D slice is what keeps non-``dim`` axes independent
        — flattening a multi-dim block pools every non-``dim`` axis into
        one sample and corrupts per-site r-largest extraction.
        """
        clean = values[~np.isnan(values)]
        out = np.full(r, np.nan, dtype=float)
        if min_periods is not None and clean.size < min_periods:
            return out
        if clean.size == 0:
            return out
        sorted_asc = np.sort(clean)
        k = min(clean.size, r)
        out[:k] = sorted_asc[-k:][::-1]
        return out

    if isinstance(block_size, str):
        # Time-based resampling; delegate to apply_ufunc over `dim` so
        # non-``dim`` axes are processed independently per group.
        groups = da.resample({dim: block_size})

        def _group_top_r(group: xr.DataArray) -> xr.DataArray:
            return xr.apply_ufunc(
                _top_r_1d,
                group,
                input_core_dims=[[dim]],
                output_core_dims=[["order"]],
                vectorize=True,
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"order": r}},
            )

        result = groups.map(_group_top_r)
        result = result.assign_coords(order=np.arange(1, r + 1))
        result.attrs = da.attrs
        return result
    else:
        # Fixed-size blocks. Label each position along `dim` with its
        # block id and groupby; for each block, run the 1-D selector per
        # non-``dim`` slice via apply_ufunc. This preserves the multi-dim
        # structure instead of flattening every non-``dim`` axis into
        # the block-wise top-r pool.
        n_blocks = da.sizes[dim] // block_size
        trimmed_length = n_blocks * block_size
        trimmed = da.isel({dim: slice(0, trimmed_length)})

        block_ids = np.arange(trimmed_length) // block_size
        trimmed = trimmed.assign_coords(_block=(dim, block_ids))

        def _per_block_top_r(group: xr.DataArray) -> xr.DataArray:
            return xr.apply_ufunc(
                _top_r_1d,
                group,
                input_core_dims=[[dim]],
                output_core_dims=[["order"]],
                vectorize=True,
                output_dtypes=[float],
                dask_gufunc_kwargs={"output_sizes": {"order": r}},
            )

        result = trimmed.groupby("_block").map(_per_block_top_r)
        block_coords = da[dim].values[:trimmed_length:block_size]
        result = result.rename({"_block": "block"}).assign_coords(
            block=block_coords,
            order=np.arange(1, r + 1),
        )
        result.attrs = da.attrs
        return result
