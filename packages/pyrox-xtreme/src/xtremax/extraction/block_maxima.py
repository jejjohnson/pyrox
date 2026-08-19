from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import xarray as xr


def _unique_dim(base: str, taken: set[Hashable]) -> str:
    """An internal dimension name guaranteed not to collide with ``taken``.

    The window/stack dimensions below are implementation details, but a caller's
    array may legitimately already carry a dimension or coordinate of the same
    name, which would otherwise produce duplicate dims and fail.
    """
    name = base
    suffix = 0
    while name in taken:
        suffix += 1
        name = f"{base}{suffix}"
    return name


def _na_for(dtype: np.dtype) -> float | np.datetime64:
    """The missing-value marker matching ``dtype`` (``NaT`` for datetimes)."""
    return np.datetime64("NaT") if dtype.kind == "M" else np.nan


def _argmax_coord(
    values: xr.DataArray,
    coord_windows: xr.DataArray,
    core_dims: list[str],
    na: float | np.datetime64,
) -> xr.DataArray:
    """Coordinate of the maximum of ``values`` over ``core_dims``.

    The position is found by locating the first entry equal to the slice
    maximum, rather than with ``argmax`` over a ``-inf``-filled copy. Both
    ``fillna(-inf).argmax()`` and ``np.nanargmax`` (which fills with ``-inf``
    internally) let a missing sample tie with a genuine ``-inf`` observation and
    win on position, reporting the wrong coordinate.

    The lookup runs inside the same ufunc because xarray refuses vectorized
    indexing with a chunked indexer; gathering where the positions are computed
    is what keeps chunked (Dask-backed) inputs working.

    Slices that are entirely missing yield ``na``; callers additionally mask
    against the maxima themselves to honour ``min_periods``.
    """

    def _pick(vals: np.ndarray, cwin: np.ndarray) -> np.ndarray:
        n = len(core_dims)
        vals = np.asarray(vals, dtype=float).reshape(*vals.shape[: vals.ndim - n], -1)
        cwin = np.asarray(cwin).reshape(*cwin.shape[: cwin.ndim - n], -1)
        # The coordinate grid carries no batch dims (a coordinate is shared by
        # every non-``dim`` slice), so line it up with the values before gathering.
        cwin = np.broadcast_to(cwin, vals.shape)
        empty = np.all(np.isnan(vals), axis=-1)
        # All-missing slices would make nanmax warn and return NaN; give them a
        # dummy row and discard the answer afterwards.
        safe = np.where(empty[..., None], 0.0, vals)
        largest = np.nanmax(safe, axis=-1, keepdims=True)
        # argmax over the boolean hit mask yields the first matching position.
        idx = np.argmax((safe == largest) & ~np.isnan(safe), axis=-1)
        picked = np.take_along_axis(cwin, idx[..., None], axis=-1)[..., 0]
        return np.where(empty, na, picked)

    return xr.apply_ufunc(
        _pick,
        values,
        coord_windows,
        input_core_dims=[core_dims, core_dims],
        dask="parallelized",
        output_dtypes=[coord_windows.dtype],
        dask_gufunc_kwargs={
            # Core-dim inputs chunked along the reduced dimension must be
            # consolidated before the gufunc runs. Window dims are always a
            # single chunk; the resampled ``time`` core dim need not be.
            "allow_rechunk": True,
        },
    )


def temporal_block_maxima(
    da: xr.DataArray,
    freq: str,
    time_dim: str = "time",
    min_periods: int | None = None,
    keep_time: bool = False,
) -> xr.DataArray:
    """
    Extract temporal block maximum values from an xarray DataArray.

    Uses time-based resampling to compute maximum values over temporal blocks.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    freq : str
        Frequency string for temporal grouping (e.g., 'YS' for annual,
        'MS' for monthly, '7D' for weekly, '3h' for 3-hourly)
    time_dim : str, default 'time'
        Name of the time dimension
    min_periods : int, optional
        Minimum number of observations required for a valid maximum.
        If None, all blocks are used regardless of missing data.
    keep_time : bool, default False
        If True, attach a ``f"{time_dim}_of_max"`` coordinate holding the
        **original timestamp** at which each block maximum occurred (rather than
        only the resampled block label). Empty or ``min_periods``-masked blocks
        get ``NaT``. Useful for plotting the maxima at the moment they actually
        happened instead of at the block boundary.

    Returns
    -------
    xr.DataArray
        Temporal block maximum values. When ``keep_time=True`` the result also
        carries a ``f"{time_dim}_of_max"`` coordinate with the argmax timestamps.

    Examples
    --------
    >>> # Annual maxima
    >>> annual_max = temporal_block_maxima(da, 'YS')

    >>> # Monthly maxima with at least 20 observations per month
    >>> monthly_max = temporal_block_maxima(da, 'MS', min_periods=20)

    >>> # Seasonal maxima (3-month blocks)
    >>> seasonal_max = temporal_block_maxima(da, '3MS')

    >>> # Annual maxima labelled with the day each maximum occurred
    >>> am = temporal_block_maxima(da, 'YS', keep_time=True)
    >>> am['time_of_max']  # actual timestamps, NaT for empty blocks
    """
    resampler = da.resample({time_dim: freq})

    maxima = resampler.max(dim=time_dim, keep_attrs=True)
    if min_periods is not None:
        # Mask blocks with insufficient data.
        counts = resampler.count(dim=time_dim)
        maxima = maxima.where(counts >= min_periods)

    if keep_time:
        na = _na_for(np.asarray(da[time_dim].values).dtype)

        def _arg_time(block: xr.DataArray) -> xr.DataArray:
            stamps = xr.DataArray(np.asarray(block[time_dim].values), dims=[time_dim])
            return _argmax_coord(block, stamps, [time_dim], na)

        arg_time = da.resample({time_dim: freq}).map(_arg_time)
        # Align NaT with the maxima's own NaNs (e.g. min_periods masking).
        arg_time = arg_time.where(maxima.notnull())
        maxima = maxima.assign_coords({f"{time_dim}_of_max": arg_time})

    return maxima


def spatial_block_maxima(
    da: xr.DataArray,
    block_size: int | dict[Hashable, int],
    dims: Hashable | Sequence[Hashable] = None,
    min_periods: int | None = None,
    boundary: str = "trim",
    keep_coords: bool = False,
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
    keep_coords : bool, default False
        If True, attach a ``f"{dim}_of_max"`` coordinate for each coarsened
        ``dim`` holding the **original coordinate** at which each block maximum
        occurred (i.e. *where* in the block the extreme is), rather than only the
        coarsened block label. Empty / masked blocks get NaN.

    Returns
    -------
    xr.DataArray
        Spatial block maximum values with reduced resolution. When
        ``keep_coords=True`` the result also carries a ``f"{dim}_of_max"``
        coordinate per coarsened ``dim`` with the argmax locations.

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

    >>> # 5x5 blocks, keeping the (lat, lon) of each block's maximum
    >>> sm = spatial_block_maxima(da, 5, dims=['lat', 'lon'], keep_coords=True)
    >>> sm['lat_of_max'], sm['lon_of_max']
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

    maxima = coarsened.max(keep_attrs=True)
    if min_periods is not None:
        counts = coarsened.count()
        maxima = maxima.where(counts >= min_periods)

    if keep_coords:
        cdims = list(coarsen_kwargs)
        taken = set(da.dims) | set(da.coords)
        win: dict[Hashable, str] = {}
        for d in cdims:
            win[d] = _unique_dim(f"_{d}_win", taken)
            taken.add(win[d])
        # Split each coarsened dim into (block, within-block) and find the
        # argmax over the flattened within-block window.
        con = coarsened.construct({d: (d, win[d]) for d in cdims})
        window_dims = [win[d] for d in cdims]
        # Align the argmax locations with the maxima's own NaNs, so empty and
        # ``min_periods``-masked blocks report no location.
        valid = maxima.notnull()
        # Build each coordinate's within-block grid, broadcast over every window
        # dim so the argmax runs over all of them at once — robust to any
        # number of coarsened dims.
        coord_windows = [
            xr.DataArray(da[d].values, dims=[d])
            .coarsen({d: coarsen_kwargs[d]}, boundary=boundary)
            .construct({d: (d, win[d])})
            for d in cdims
        ]
        for d, cwin in zip(cdims, xr.broadcast(*coord_windows), strict=True):
            d_of_max = _argmax_coord(con, cwin, window_dims, _na_for(cwin.dtype)).where(
                valid
            )
            maxima = maxima.assign_coords({f"{d}_of_max": d_of_max})

    return maxima


def sliding_block_maxima(
    da: xr.DataArray,
    window_size: int,
    dim: str = "time",
    stride: int = 1,
    min_periods: int | None = None,
    center: bool = False,
    keep_time: bool = False,
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
    keep_time : bool, default False
        If True, attach a ``f"{dim}_of_max"`` coordinate holding the **original
        coordinate** at which each window's maximum occurred (which can differ
        from the window's own label). Windows whose maximum is masked — no valid
        data, or fewer than ``min_periods`` observations — get NaN/NaT.

    Returns
    -------
    xr.DataArray
        Sliding block maximum values. When ``keep_time=True`` the result also
        carries a ``f"{dim}_of_max"`` coordinate with the argmax coordinates.

    Examples
    --------
    >>> # 365-day sliding window with daily stride
    >>> sliding_max = sliding_block_maxima(da, window_size=365, stride=1)

    >>> # Non-overlapping 30-day blocks (equivalent to fixed blocks)
    >>> block_max = sliding_block_maxima(da, window_size=30, stride=30)

    >>> # 90-day window with 7-day stride (weekly updates)
    >>> weekly_max = sliding_block_maxima(da, window_size=90, stride=7)

    >>> # Track the day each window maximum actually occurred
    >>> sm = sliding_block_maxima(da, window_size=30, keep_time=True)
    >>> sm['time_of_max']
    """
    if min_periods is None:
        min_periods = window_size

    # Use rolling window
    rolling = da.rolling({dim: window_size}, center=center, min_periods=min_periods)
    maxima = rolling.max()

    if keep_time:
        wdim = _unique_dim("_w", set(da.dims) | set(da.coords))
        windows = rolling.construct(wdim)
        coord_windows = (
            xr.DataArray(da[dim].values, dims=[dim], coords={dim: da[dim]})
            .rolling({dim: window_size}, center=center)
            .construct(wdim)
        )
        # Align with the maxima's own NaNs so windows with no data — or too few
        # valid observations for ``min_periods`` — report no coordinate.
        arg_coord = _argmax_coord(
            windows, coord_windows, [wdim], _na_for(coord_windows.dtype)
        ).where(maxima.notnull())
        # Assigned before striding so the coordinate is subsampled with it.
        maxima = maxima.assign_coords({f"{dim}_of_max": arg_coord})

    # Apply stride by subsampling, starting from the first *valid*
    # window — the first position whose window holds at least
    # min_periods samples. With the default min_periods == window_size
    # that is the first complete window, so stride == window_size
    # reproduces non-overlapping blocks (coarsen equivalence); an
    # explicit smaller min_periods keeps the valid partial edge windows
    # instead of discarding them. Rolling labels sit at the window end
    # (center=False) or centre (center=True, window spanning
    # [i - w//2, i + (w-1)//2]).
    if stride > 1:
        if center:
            offset = max(0, min_periods - 1 - (window_size - 1) // 2)
        else:
            offset = min_periods - 1
        maxima = maxima.isel({dim: slice(offset, None, stride)})

    return maxima


def declustered_block_maxima(
    da: xr.DataArray,
    threshold: float,
    min_separation: int,
    dim: str = "time",
    method: str = "runs",
    keep_time: bool = False,
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
        - 'runs': Select maximum from each exceedance run, merging runs
          whose below-threshold gap is shorter than ``min_separation``
        - 'separation': Enforce minimum time separation between peaks
    keep_time : bool, default False
        If True, attach a ``f"{dim}_of_max"`` coordinate holding the **original
        timestamp** at which each cluster's maximum occurred. The ``'runs'``
        method otherwise labels the representative at the *end* of its run, so
        this recovers the true peak time; for ``'separation'`` the peaks already
        sit at their original positions.

    Returns
    -------
    xr.DataArray
        Declustered block maxima with reduced temporal dependence.

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
        # `min_separation` is the runs parameter r: below-threshold gaps
        # shorter than it are intra-cluster.
        reduced = decluster_runs(
            da,
            threshold=threshold,
            dim=dim,
            reduction="max",
            min_separation=min_separation,
        )
        if keep_time:
            reduced = _attach_runs_argtime(reduced, da, threshold, min_separation, dim)
        return reduced.dropna(dim, how="all")

    elif method == "separation":
        # Delegate to the separation-based declustering helper, which
        # actually applies `min_separation` in units of original time steps.
        peaks = decluster_separation(
            da, threshold=threshold, min_separation=min_separation, dim=dim
        )
        if keep_time:
            # Separation keeps peaks at their own positions, so the maximum's
            # coordinate is the position itself.
            arg = xr.full_like(peaks, np.nan, dtype=da[dim].dtype)
            arg = arg.where(peaks.isnull(), peaks[dim])
            peaks = peaks.assign_coords({f"{dim}_of_max": arg})
        return peaks

    else:
        raise ValueError(f"Unknown method: {method}. Use 'runs' or 'separation'")


def _attach_runs_argtime(
    reduced: xr.DataArray, da: xr.DataArray, threshold, min_separation: int, dim: str
) -> xr.DataArray:
    """Attach a ``f"{dim}_of_max"`` coord giving the true peak time per cluster.

    ``decluster_runs`` stores each cluster's max at the *last exceedance* position
    of that cluster; this recovers the index of the actual maximum within the
    cluster and maps it to the original coordinate, aligned to ``reduced`` (NaT
    where there is no cluster representative).

    The clustering mirrors :func:`decluster_runs` exactly — exceedance positions
    split where at least ``min_separation`` below-threshold steps intervene, and
    the maximum taken over the cluster's *exceedance* values only — so the
    reported time always belongs to the value stored in ``reduced``.
    """
    exceedances = da > threshold
    coords = np.asarray(da[dim].values)

    def _runs_argpos_1d(values: np.ndarray, exc: np.ndarray) -> np.ndarray:
        out = np.full(values.shape[0], -1, dtype=np.int64)
        positions = np.flatnonzero(exc)
        if positions.size == 0:
            return out
        breaks = np.flatnonzero(np.diff(positions) - 1 >= min_separation) + 1
        for cluster in np.split(positions, breaks):
            out[cluster[-1]] = int(cluster[int(np.nanargmax(values[cluster]))])
        return out

    argpos = xr.apply_ufunc(
        _runs_argpos_1d,
        da,
        exceedances,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        output_dtypes=[np.int64],
    )
    arg_time = xr.where(
        argpos >= 0,
        xr.DataArray(coords[np.clip(argpos.values, 0, None)], dims=argpos.dims),
        np.datetime64("NaT") if coords.dtype.kind == "M" else np.nan,
    )
    arg_time = arg_time.where(reduced.notnull())
    return reduced.assign_coords({f"{dim}_of_max": arg_time})


def r_largest_block_maxima(
    da: xr.DataArray,
    block_size: int | str,
    r: int = 3,
    dim: str = "time",
    min_periods: int | None = None,
    keep_time: bool = False,
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
    keep_time : bool, default False
        If True, attach a ``f"{dim}_of_max"`` coordinate (with the same
        ``order`` axis) holding the **original coordinate** at which each of the
        r largest values occurred. Missing slots get NaN/NaT.

    Returns
    -------
    xr.DataArray
        Array with r-largest values from each block, with new dimension 'order'.
        For integer ``block_size``, a trailing partial block (fewer than
        ``block_size`` samples) is trimmed and does not contribute values.
        When ``keep_time=True`` it also carries a ``f"{dim}_of_max"`` coordinate
        over ``(..., order)`` with the timestamps of those r values.

    Examples
    --------
    >>> # 3 largest values from each year
    >>> annual_r_largest = r_largest_block_maxima(da, 'YS', r=3)

    >>> # Top 5 values from 100-element blocks
    >>> block_r_largest = r_largest_block_maxima(da, 100, r=5)

    >>> # ... and the day each of the 3 largest occurred
    >>> rl = r_largest_block_maxima(da, 'YS', r=3, keep_time=True)
    >>> rl['time_of_max']

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

    def _top_r_pos_1d(values: np.ndarray) -> np.ndarray:
        """Original indices of the r largest (descending), -1-padded to r."""
        valid = np.flatnonzero(~np.isnan(values))
        out = np.full(r, -1, dtype=np.int64)
        if min_periods is not None and valid.size < min_periods:
            return out
        if valid.size == 0:
            return out
        order = valid[np.argsort(values[valid])][::-1]
        k = min(order.size, r)
        out[:k] = order[:k]
        return out

    def _apply(values: xr.DataArray, func, out_dtype) -> xr.DataArray:
        return xr.apply_ufunc(
            func,
            values,
            input_core_dims=[[dim]],
            output_core_dims=[["order"]],
            vectorize=True,
            output_dtypes=[out_dtype],
            dask="parallelized",
            dask_gufunc_kwargs={
                "output_sizes": {"order": r},
                # Core-dim inputs chunked along `dim` must be
                # consolidated before the gufunc runs.
                "allow_rechunk": True,
            },
        )

    def _coord_from_pos(pos: xr.DataArray, coords: np.ndarray) -> xr.DataArray:
        # .where fills unselected slots with NaN (NaT for datetime coords).
        picked = coords[np.clip(pos.values, 0, None)]
        out = xr.DataArray(picked, dims=pos.dims, coords=pos.coords)
        return out.where(pos >= 0)

    if isinstance(block_size, str):
        # Time-based resampling; delegate to ``_apply`` over `dim` so
        # non-``dim`` axes are processed independently per group.
        groups = da.resample({dim: block_size})
        result = groups.map(lambda g: _apply(g, _top_r_1d, float))
        result = result.assign_coords(order=np.arange(1, r + 1))
        result.attrs = da.attrs
        if keep_time:
            # The block index of each top-r value, mapped to original coords
            # per block (positions are within-block, so resolve per group).
            def _group_times(g: xr.DataArray) -> xr.DataArray:
                pos = _apply(g, _top_r_pos_1d, np.int64)
                return _coord_from_pos(pos, np.asarray(g[dim].values))

            times = groups.map(_group_times)
            result = result.assign_coords({f"{dim}_of_max": times})
        return result
    else:
        # Fixed-size blocks. Label each position along `dim` with its
        # block id and groupby; for each block, run the 1-D selector per
        # non-``dim`` slice via ``_apply``. This preserves the multi-dim
        # structure instead of flattening every non-``dim`` axis into
        # the block-wise top-r pool.
        n_blocks = da.sizes[dim] // block_size
        if n_blocks == 0:
            raise ValueError(
                f"Series has {da.sizes[dim]} samples along {dim!r}, shorter "
                f"than one block of size {block_size}; no block maxima can "
                "be extracted."
            )
        trimmed_length = n_blocks * block_size
        trimmed = da.isel({dim: slice(0, trimmed_length)})
        block_ids = np.arange(trimmed_length) // block_size
        trimmed = trimmed.assign_coords(_block=(dim, block_ids))

        result = trimmed.groupby("_block").map(lambda g: _apply(g, _top_r_1d, float))
        block_coords = da[dim].values[:trimmed_length:block_size]
        result = result.rename({"_block": "block"}).assign_coords(
            block=block_coords, order=np.arange(1, r + 1)
        )
        result.attrs = da.attrs
        if keep_time:

            def _block_times(g: xr.DataArray) -> xr.DataArray:
                pos = _apply(g, _top_r_pos_1d, np.int64)
                return _coord_from_pos(pos, np.asarray(g[dim].values))

            times = trimmed.groupby("_block").map(_block_times)
            times = times.rename({"_block": "block"}).assign_coords(
                block=block_coords, order=np.arange(1, r + 1)
            )
            result = result.assign_coords({f"{dim}_of_max": times})
        return result
