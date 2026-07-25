"""Smoke tests for extraction utilities (block maxima, thresholds, decluster)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xtremax.extraction import (
    constant_threshold,
    decluster_runs,
    decluster_separation,
    declustered_block_maxima,
    estimate_extremal_index,
    quantile_threshold,
    r_largest_block_maxima,
    rolling_threshold,
    seasonal_threshold,
    sliding_block_maxima,
    spatial_block_maxima,
    temporal_block_maxima,
    temporal_threshold,
)


@pytest.fixture
def daily_series():
    rng = np.random.default_rng(0)
    time = pd.date_range("2000-01-01", periods=365 * 3, freq="D")
    values = rng.standard_normal(len(time)) + 0.1 * np.arange(len(time)) / len(time)
    return xr.DataArray(values, dims="time", coords={"time": time})


class TestBlockMaxima:
    def test_annual_block_maxima(self, daily_series):
        annual = temporal_block_maxima(daily_series, freq="YS")
        assert "time" in annual.dims
        assert annual.sizes["time"] == 3  # 3 years
        assert bool(np.all(np.isfinite(annual.values)))

    def test_r_largest_with_int_block_size_runs(self, daily_series):
        """Regression: fixed-size (int) block_size branch used to call
        `.values` on numpy sub-arrays yielded by reshape+iterate, raising
        AttributeError before producing any output.
        """
        out = r_largest_block_maxima(daily_series, block_size=30, r=3)
        assert "block" in out.dims and "order" in out.dims
        assert out.sizes["order"] == 3
        assert bool(np.all(np.isfinite(out.values)))
        # Order-1 ≥ order-2 ≥ order-3 within each block (descending).
        assert bool(np.all(np.diff(out.values, axis=1) <= 0))

    def test_r_largest_int_block_preserves_non_dim_axes(self):
        """Regression: the int-block path used `np.asarray(block).flatten()`
        which pooled every non-``dim`` axis into the top-r sort, returning
        shared maxima across all sites instead of per-site order statistics.
        """
        # Two sites with deliberately non-overlapping value ranges so any
        # accidental pooling would surface immediately.
        time = pd.date_range("2020-01-01", periods=20, freq="D")
        # Site 0: values 1..20. Site 1: values 101..120.
        values = np.stack(
            [np.arange(1, 21, dtype=float), np.arange(101, 121, dtype=float)],
            axis=-1,
        )
        da = xr.DataArray(
            values,
            dims=("time", "site"),
            coords={"time": time, "site": [0, 1]},
        )
        out = r_largest_block_maxima(da, block_size=10, r=3, dim="time")

        # Two blocks (10 elements each) × two sites × r=3.
        assert out.sizes == {"block": 2, "order": 3, "site": 2}
        # Per-site top-3 from block 0 must be {10,9,8} for site 0 and
        # {110,109,108} for site 1. If pooled, site 0 would incorrectly
        # contain the site-1 values.
        site0_b0 = out.isel(block=0).sel(site=0).values
        site1_b0 = out.isel(block=0).sel(site=1).values
        np.testing.assert_array_equal(site0_b0, [10.0, 9.0, 8.0])
        np.testing.assert_array_equal(site1_b0, [110.0, 109.0, 108.0])

    def test_r_largest_str_block_preserves_non_dim_axes(self):
        """Same regression check for the resample (str block_size) path."""
        time = pd.date_range("2020-01-01", periods=365 * 2, freq="D")
        rng = np.random.default_rng(0)
        # Two sites shifted far apart; each-site top values must be from
        # that site's own range, not a pooled union.
        values = np.stack(
            [rng.standard_normal(730), rng.standard_normal(730) + 100.0],
            axis=-1,
        )
        da = xr.DataArray(
            values,
            dims=("time", "site"),
            coords={"time": time, "site": [0, 1]},
        )
        out = r_largest_block_maxima(da, block_size="YS", r=3, dim="time")

        assert "site" in out.dims
        # All site-0 extracted values must be within site-0's sensible
        # range (well below site-1's +100 offset).
        site0 = out.sel(site=0).values
        site1 = out.sel(site=1).values
        assert float(np.nanmax(site0)) < 20.0  # no site-1 leakage
        assert float(np.nanmin(site1)) > 80.0  # no site-0 leakage


class TestThreshold:
    def test_constant(self, daily_series):
        assert constant_threshold(daily_series, 1.5) == 1.5

    def test_quantile(self, daily_series):
        u = quantile_threshold(daily_series, 0.95)
        empirical = float(np.quantile(daily_series.values, 0.95))
        assert np.isclose(float(u), empirical, rtol=1e-5)

    def test_temporal(self, daily_series):
        u = temporal_threshold(daily_series, 0.95, groupby="time.year")
        assert "year" in u.dims or "time" in u.dims

    def test_rolling(self, daily_series):
        u = rolling_threshold(daily_series, 0.95, window_size=30)
        assert u.sizes["time"] == daily_series.sizes["time"]

    def test_seasonal(self, daily_series):
        u = seasonal_threshold(daily_series, 0.95)
        # seasonal threshold returns one value per season
        assert "season" in u.dims or "time" in u.dims

    def test_seasonal_respects_custom_time_dim(self):
        """Regression: `seasonal_threshold` hard-coded `groupby="time.season"`,
        so datasets whose temporal axis is not named `time` (e.g. `date`)
        raised KeyError despite the API advertising a `time_dim` argument.
        The grouping key must be derived from `time_dim`.
        """
        rng = np.random.default_rng(0)
        dates = pd.date_range("2000-01-01", periods=365 * 2, freq="D")
        da = xr.DataArray(
            rng.standard_normal(len(dates)),
            dims="date",
            coords={"date": dates},
        )
        u = seasonal_threshold(da, 0.95, time_dim="date")
        # Success: call didn't raise; output has a season coord or axis.
        assert "season" in u.dims or "season" in u.coords


class TestDecluster:
    def test_runs(self, daily_series):
        out = decluster_runs(daily_series, threshold=1.5, reduction="max")
        assert isinstance(out, xr.DataArray)

    def test_separation_measures_original_time_steps(self):
        """Regression: min_separation must count original time steps, not peaks apart.

        Construct a sparse series with three isolated exceedance peaks at
        positions [10, 25, 30] (values 5, 4, 3). With min_separation=10 and
        the old buggy code (peak_times = 0..n_peaks-1), every peak is 1 apart
        so only the largest would be kept. The correct behaviour keeps the
        peak at 10 (value 5) and the one at 25 (5 > 10 steps away); 30 is
        within 5 of 25 so gets dropped.
        """
        values = np.zeros(40, dtype=float)
        # Each peak flanked by zeros so the "local max above threshold"
        # detector picks only the peak indices.
        values[10] = 5.0
        values[25] = 4.0
        values[30] = 3.0
        time = pd.date_range("2020-01-01", periods=40, freq="D")
        da = xr.DataArray(values, dims="time", coords={"time": time})

        out = decluster_separation(da, threshold=0.5, min_separation=10)

        assert out.sizes["time"] == 2
        assert set(out.values.tolist()) == {5.0, 4.0}

    def test_separation_keeps_only_largest_when_all_too_close(self):
        values = np.zeros(20, dtype=float)
        values[5], values[8], values[11] = 3.0, 2.0, 4.0
        time = pd.date_range("2020-01-01", periods=20, freq="D")
        da = xr.DataArray(values, dims="time", coords={"time": time})

        out = decluster_separation(da, threshold=0.5, min_separation=10)

        assert out.sizes["time"] == 1
        assert float(out.values[0]) == 4.0

    def test_separation_keeps_boundary_peaks(self):
        """Regression: `is_peak` required both a left AND a right neighbour,
        so the first and last positions along `dim` could never be peaks
        (one neighbour was always NaN after shifting). In short windows
        or chunked workflows this silently dropped valid boundary events.
        """
        # First and last positions are exceedances with a single
        # strictly-lower existing neighbour; they must now count as peaks.
        values = np.array([5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0])
        time = pd.date_range("2020-01-01", periods=len(values), freq="D")
        da = xr.DataArray(values, dims="time", coords={"time": time})

        out = decluster_separation(da, threshold=0.5, min_separation=3)

        # Both boundary peaks (positions 0 and -1) must be retained —
        # they're 7 steps apart, well above min_separation=3.
        assert out.sizes["time"] == 2
        assert set(float(v) for v in out.values) == {5.0, 4.0}

    def test_separation_vectorises_over_extra_dims(self):
        """Regression: `np.flatnonzero(is_peak.values)` used linear indices
        across all dims, so for multi-D arrays positions didn't correspond
        to offsets along `dim`. `apply_ufunc` now applies the 1-D
        selector per batch row.
        """
        time = pd.date_range("2020-01-01", periods=30, freq="D")
        # Two sites with different peak layouts; each has peaks at
        # positions with differing separations.
        values = np.zeros((30, 2), dtype=float)
        values[5, 0], values[20, 0] = 5.0, 3.0  # 15 apart
        values[10, 1], values[14, 1] = 4.0, 2.0  # 4 apart → one kept
        da = xr.DataArray(
            values,
            dims=("time", "site"),
            coords={"time": time, "site": [0, 1]},
        )

        out = decluster_separation(da, threshold=0.5, min_separation=10)

        site0_vals = out.sel(site=0).dropna("time").values.tolist()
        site1_vals = out.sel(site=1).dropna("time").values.tolist()
        assert set(site0_vals) == {5.0, 3.0}  # both kept (15 apart)
        assert set(site1_vals) == {4.0}  # only the larger (too close)

    def test_runs_isolates_per_slice(self):
        """Regression: grouping by a flat run_ids DataArray pooled clusters
        across non-``dim`` slices because IDs restart per slice, so e.g. a
        ``time × site`` input with run id 1 on both sites would merge them
        into one group. Per-slice 1-D processing via apply_ufunc fixes this.
        """
        # Build two sites, each with one exceedance run, at different times.
        # Site 0: values [0,0,3,4,0,0,0,0]  -> one run with max=4
        # Site 1: values [0,0,0,0,0,5,6,0]  -> one run with max=6
        time = pd.date_range("2020-01-01", periods=8, freq="D")
        values = np.zeros((8, 2), dtype=float)
        values[2, 0], values[3, 0] = 3.0, 4.0
        values[5, 1], values[6, 1] = 5.0, 6.0
        da = xr.DataArray(
            values,
            dims=("time", "site"),
            coords={"time": time, "site": [0, 1]},
        )

        out = decluster_runs(da, threshold=0.5, reduction="max")

        # Output preserves shape; each site gets its own run representative.
        assert out.sizes == da.sizes
        site0_vals = out.sel(site=0).dropna("time").values.tolist()
        site1_vals = out.sel(site=1).dropna("time").values.tolist()
        assert site0_vals == [4.0]
        assert site1_vals == [6.0]

    def test_runs_matches_per_slice_loop(self):
        """Per-slice output must equal calling decluster_runs on each slice."""
        rng = np.random.default_rng(0)
        time = pd.date_range("2020-01-01", periods=50, freq="D")
        values = rng.standard_normal((50, 3))
        da = xr.DataArray(
            values,
            dims=("time", "site"),
            coords={"time": time, "site": [0, 1, 2]},
        )

        vectorised = decluster_runs(da, threshold=0.5, reduction="max")
        for s in [0, 1, 2]:
            per_site = decluster_runs(da.sel(site=s), threshold=0.5, reduction="max")
            np.testing.assert_array_equal(
                vectorised.sel(site=s).values, per_site.values
            )

    def test_extremal_index_returns_one_per_slice(self):
        """Regression: counting unique run_ids across all dims merged
        independent clusters from different slices (both restart at 1).
        The per-slice estimator returns one θ per batch row instead.
        """
        # Site 0: two isolated exceedances (2 runs of 1 → θ = 1.0).
        # Site 1: one run of length 2 (1 run / 2 exceedances → θ = 0.5).
        time = pd.date_range("2020-01-01", periods=8, freq="D")
        values = np.zeros((8, 2), dtype=float)
        values[1, 0] = 3.0
        values[4, 0] = 3.0
        values[2, 1] = 3.0
        values[3, 1] = 3.0
        da = xr.DataArray(
            values,
            dims=("time", "site"),
            coords={"time": time, "site": [0, 1]},
        )

        theta = estimate_extremal_index(da, threshold=0.5)

        assert isinstance(theta, xr.DataArray)
        assert set(theta.dims) == {"site"}
        assert float(theta.sel(site=0)) == pytest.approx(1.0)
        assert float(theta.sel(site=1)) == pytest.approx(0.5)

    def test_extremal_index_scalar_for_1d(self):
        """1-D inputs keep the historical float return type."""
        values = np.zeros(8, dtype=float)
        values[2] = 3.0  # one isolated run
        values[5] = 3.0  # another isolated run
        time = pd.date_range("2020-01-01", periods=8, freq="D")
        da = xr.DataArray(values, dims="time", coords={"time": time})

        theta = estimate_extremal_index(da, threshold=0.5)
        assert isinstance(theta, float)
        assert theta == pytest.approx(1.0)

    def test_declustered_block_maxima_runs_isolates_per_slice(self):
        """Regression: the runs branch of declustered_block_maxima used to
        call groupby on numeric run_ids that restart per slice, mixing
        clusters across sites. It now delegates to decluster_runs.
        """
        time = pd.date_range("2020-01-01", periods=8, freq="D")
        values = np.zeros((8, 2), dtype=float)
        values[2, 0], values[3, 0] = 3.0, 4.0
        values[5, 1], values[6, 1] = 5.0, 6.0
        da = xr.DataArray(
            values,
            dims=("time", "site"),
            coords={"time": time, "site": [0, 1]},
        )

        out = declustered_block_maxima(
            da, threshold=0.5, min_separation=1, method="runs"
        )

        site0_vals = out.sel(site=0).dropna("time").values.tolist()
        site1_vals = out.sel(site=1).dropna("time").values.tolist()
        assert site0_vals == [4.0]
        assert site1_vals == [6.0]

    def test_declustered_block_maxima_runs_honors_min_separation(self):
        """#69 — the runs branch previously ignored `min_separation`
        entirely: two exceedances one step apart were always counted as
        independent events. With gap-tolerant runs, a below-threshold gap
        shorter than min_separation is intra-cluster."""
        values = np.array([0.0, 3.0, 0.0, 4.0, 0.0])
        time = pd.date_range("2020-01-01", periods=5, freq="D")
        da = xr.DataArray(values, dims="time", coords={"time": time})

        merged = declustered_block_maxima(
            da, threshold=0.5, min_separation=2, method="runs"
        )
        assert merged.dropna("time").values.tolist() == [4.0]

        split = declustered_block_maxima(
            da, threshold=0.5, min_separation=1, method="runs"
        )
        assert sorted(split.dropna("time").values.tolist()) == [3.0, 4.0]

    def test_declustered_block_maxima_separation_applies_min_separation(self):
        """Regression: the `method='separation'` branch of
        declustered_block_maxima previously returned all peaks unchanged.
        It must now enforce min_separation (via decluster_separation).
        """
        values = np.zeros(40, dtype=float)
        values[10], values[25], values[30] = 5.0, 4.0, 3.0
        time = pd.date_range("2020-01-01", periods=40, freq="D")
        da = xr.DataArray(values, dims="time", coords={"time": time})

        out = declustered_block_maxima(
            da, threshold=0.5, min_separation=10, method="separation"
        )

        assert out.sizes["time"] == 2
        assert set(out.values.tolist()) == {5.0, 4.0}


class TestSlidingBlockMaxima:
    """#67 — stride subsampling must start at the first complete window."""

    @pytest.mark.parametrize("center", [False, True])
    @pytest.mark.parametrize("window_size", [3, 4])
    def test_coarsen_equivalence(self, center, window_size):
        """stride == window_size is documented as equivalent to
        `coarsen(dim=window_size).max()`. The old subsampling started at
        label 0 (an incomplete window), shifting every block by one and
        silently dropping the final block's maximum."""
        da = xr.DataArray(np.arange(12.0), dims="time")
        out = sliding_block_maxima(
            da, window_size=window_size, stride=window_size, center=center
        )
        expected = da.coarsen(time=window_size).max()
        np.testing.assert_array_equal(out.values, expected.values)

    def test_no_leading_nan_and_last_block_present(self):
        da = xr.DataArray(np.arange(12.0), dims="time")
        out = sliding_block_maxima(da, window_size=4, stride=4)
        # Exactly the non-overlapping block maxima — no NaN padding, and
        # the final block's maximum (11) is present.
        np.testing.assert_array_equal(out.values, [3.0, 7.0, 11.0])

    def test_stride_keeps_valid_partial_windows_with_min_periods(self):
        """Codex round on #86: an explicit min_periods < window_size
        declares partial edge windows valid, so the stride offset must
        anchor at the first window satisfying min_periods — not discard
        those windows by jumping to the first complete one."""
        da = xr.DataArray(np.arange(3.0), dims="time")
        out = sliding_block_maxima(da, window_size=5, stride=2, min_periods=1)
        # Window maxima at labels 0..2 are [0, 1, 2]; offset 0, stride 2.
        np.testing.assert_array_equal(out.values, [0.0, 2.0])

    def test_stride_min_periods_offset_center(self):
        da = xr.DataArray(np.arange(8.0), dims="time")
        out = sliding_block_maxima(
            da, window_size=4, stride=2, min_periods=2, center=True
        )
        # center=True window at label i spans [i-2, i+1]; label 0 already
        # holds two in-bounds samples (0, 1), so nothing is skipped.
        full = da.rolling(time=4, center=True, min_periods=2).max()
        np.testing.assert_array_equal(out.values, full.values[0::2])


class TestDeclusterPlateau:
    """#68 — flat-topped exceedances must yield exactly one peak."""

    def test_separation_plateau_kept_once(self):
        values = np.array([0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0])
        time = pd.date_range("2020-01-01", periods=len(values), freq="D")
        da = xr.DataArray(values, dims="time", coords={"time": time})

        out = decluster_separation(da, threshold=1.0, min_separation=3)

        # The plateau contributes exactly one event (5.0) plus the
        # isolated 3.0 — previously the 5.0 event vanished entirely.
        assert sorted(float(v) for v in out.values) == [3.0, 5.0]

    def test_separation_plateau_at_boundary(self):
        values = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0])
        time = pd.date_range("2020-01-01", periods=len(values), freq="D")
        da = xr.DataArray(values, dims="time", coords={"time": time})

        out = decluster_separation(da, threshold=1.0, min_separation=3)
        assert sorted(float(v) for v in out.values) == [3.0, 5.0]

    def test_rising_shelf_is_not_a_peak(self):
        """Codex round 2 on #86: a plateau followed by a higher value
        (quantized rising shelf) is not a local maximum — only the true
        summit may be selected, even when min_separation is small."""
        values = np.array([0.0, 5.0, 5.0, 6.0, 0.0])
        time = pd.date_range("2020-01-01", periods=len(values), freq="D")
        da = xr.DataArray(values, dims="time", coords={"time": time})

        out = decluster_separation(da, threshold=1.0, min_separation=1)
        assert out.values.tolist() == [6.0]

    def test_falling_shelf_is_not_a_peak(self):
        values = np.array([0.0, 6.0, 5.0, 5.0, 0.0])
        time = pd.date_range("2020-01-01", periods=len(values), freq="D")
        da = xr.DataArray(values, dims="time", coords={"time": time})

        out = decluster_separation(da, threshold=1.0, min_separation=1)
        assert out.values.tolist() == [6.0]

    def test_constant_exceedance_run_single_peak(self):
        """A longer plateau still yields exactly one peak — no
        double-counting from the asymmetric tie-break."""
        values = np.array([0.0, 4.0, 4.0, 4.0, 4.0, 0.0])
        time = pd.date_range("2020-01-01", periods=len(values), freq="D")
        da = xr.DataArray(values, dims="time", coords={"time": time})

        out = decluster_separation(da, threshold=1.0, min_separation=1)
        assert out.sizes["time"] == 1
        assert float(out.values[0]) == 4.0


class TestIgnoredParameters:
    """#69 — parameters that used to be silently ignored."""

    def test_rolling_threshold_min_periods_masks_partial_windows(self):
        da = xr.DataArray(np.arange(10.0), dims="time")
        u = rolling_threshold(da, 0.5, window_size=5, center=False)
        # First four windows hold fewer than 5 samples → NaN, not the
        # quantile of a 1–4-sample window.
        assert bool(np.all(np.isnan(u.values[:4])))
        assert bool(np.all(np.isfinite(u.values[4:])))

    def test_rolling_threshold_explicit_min_periods(self):
        da = xr.DataArray(np.arange(10.0), dims="time")
        u = rolling_threshold(da, 0.5, window_size=5, center=False, min_periods=2)
        # Only the very first window (1 sample) is below min_periods=2.
        assert np.isnan(u.values[0])
        assert bool(np.all(np.isfinite(u.values[1:])))

    def test_decluster_runs_gap_tolerant(self):
        values = np.array([0.0, 3.0, 0.0, 4.0, 0.0])
        time = pd.date_range("2020-01-01", periods=5, freq="D")
        da = xr.DataArray(values, dims="time", coords={"time": time})

        # Gap of one below-threshold step: min_separation=2 merges the
        # two exceedances into one cluster; the default keeps them split.
        merged = decluster_runs(da, threshold=0.5, min_separation=2)
        assert merged.dropna("time").values.tolist() == [4.0]
        split = decluster_runs(da, threshold=0.5)
        assert sorted(split.dropna("time").values.tolist()) == [3.0, 4.0]

    def test_extremal_index_exposes_run_parameter(self):
        values = np.array([0.0, 3.0, 0.0, 4.0, 0.0])
        time = pd.date_range("2020-01-01", periods=5, freq="D")
        da = xr.DataArray(values, dims="time", coords={"time": time})

        assert estimate_extremal_index(da, threshold=0.5) == pytest.approx(1.0)
        assert estimate_extremal_index(
            da, threshold=0.5, min_separation=2
        ) == pytest.approx(0.5)


class TestSpatialBlockMaxima:
    """#75 — first tests for spatial_block_maxima."""

    def _grid(self):
        rng = np.random.default_rng(0)
        return xr.DataArray(
            rng.standard_normal((8, 6)),
            dims=("y", "x"),
            coords={"y": np.arange(8), "x": np.arange(6)},
        )

    def test_dict_block_size_matches_coarsen(self):
        da = self._grid()
        out = spatial_block_maxima(da, {"y": 4, "x": 3})
        expected = da.coarsen(y=4, x=3).max()
        np.testing.assert_array_equal(out.values, expected.values)

    def test_int_block_size_with_dims(self):
        da = self._grid()
        out = spatial_block_maxima(da, 2, dims=["y", "x"])
        expected = da.coarsen(y=2, x=2).max()
        np.testing.assert_array_equal(out.values, expected.values)

    def test_int_without_dims_raises(self):
        with pytest.raises(ValueError, match="dims"):
            spatial_block_maxima(self._grid(), 2)

    def test_min_periods_masks_sparse_blocks(self):
        da = self._grid()
        # Knock out most of one 4x3 block.
        values = da.values.copy()
        values[0:4, 0:2] = np.nan  # 8 of 12 cells NaN in block (0, 0)
        da = da.copy(data=values)
        out = spatial_block_maxima(da, {"y": 4, "x": 3}, min_periods=6)
        assert np.isnan(out.values[0, 0])  # only 4 valid cells
        assert np.isfinite(out.values[1, 0])

    def test_boundary_trim(self):
        da = self._grid()  # 8 x 6
        out = spatial_block_maxima(da, {"y": 3, "x": 4}, boundary="trim")
        assert out.sizes == {"y": 2, "x": 1}


class TestSlidingBlockMaximaNaN:
    """#75 — NaN handling with min_periods."""

    def test_nan_gap_respects_min_periods(self):
        values = np.arange(10.0)
        values[3:7] = np.nan
        da = xr.DataArray(values, dims="time")
        out_strict = sliding_block_maxima(da, window_size=4, min_periods=4)
        out_loose = sliding_block_maxima(da, window_size=4, min_periods=1)
        # Window at label 7 covers samples [4..7]: one valid value (7.0).
        # It fails min_periods=4 but passes min_periods=1 with the max
        # taken over the valid samples only.
        assert np.isnan(out_strict.values[7])
        assert float(out_loose.values[7]) == 7.0
        # Window at label 9 covers [6..9]: three valid values.
        assert np.isnan(out_strict.values[9])
        assert float(out_loose.values[9]) == 9.0


class TestDeclusterProperties:
    """#75 — DataArray thresholds and non-max reductions."""

    def test_dataarray_threshold(self):
        time = pd.date_range("2020-01-01", periods=10, freq="D")
        values = np.array([0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 5.0, 0.0, 0.0])
        da = xr.DataArray(values, dims="time", coords={"time": time})
        # Time-varying threshold: 1.0 early, 3.0 from day 3 on — the
        # 2.0 event at position 4 is below its local threshold.
        thresh = xr.DataArray(
            np.where(np.arange(10) < 3, 1.0, 3.0),
            dims="time",
            coords={"time": time},
        )
        out = decluster_runs(da, threshold=thresh, reduction="max")
        assert sorted(out.dropna("time").values.tolist()) == [2.0, 5.0]

    @pytest.mark.parametrize(
        ("reduction", "expected"),
        [("mean", 3.0), ("sum", 9.0), ("min", 2.0), ("max", 4.0)],
    )
    def test_reductions(self, reduction, expected):
        time = pd.date_range("2020-01-01", periods=7, freq="D")
        values = np.array([0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0])
        da = xr.DataArray(values, dims="time", coords={"time": time})
        out = decluster_runs(da, threshold=0.5, reduction=reduction)
        vals = out.dropna("time").values.tolist()
        assert vals == [expected]


class TestQuantileThresholdReturn:
    """#71 item 7 — return type and coordinate hygiene."""

    def test_scalar_is_float(self, daily_series):
        u = quantile_threshold(daily_series, 0.95)
        assert isinstance(u, float)

    def test_no_quantile_coord_leak(self, daily_series):
        da2 = daily_series.expand_dims(site=[0, 1])
        u = quantile_threshold(da2, 0.95, dim="time")
        assert "quantile" not in u.coords
        # Rolling and temporal variants must not leak either.
        assert "quantile" not in rolling_threshold(daily_series, 0.95).coords
        assert (
            "quantile"
            not in temporal_threshold(daily_series, 0.95, groupby="time.year").coords
        )
