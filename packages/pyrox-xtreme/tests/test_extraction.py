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


class TestKeepOriginalCoordinates:
    """The keep_time / keep_coords option records where each extreme occurred."""

    def test_temporal_keep_time(self):
        time = pd.date_range("2020-01-01", periods=730, freq="D")
        values = np.zeros(730)
        values[100] = 40.0  # 2020 max
        values[565] = 42.0  # 2021 max
        da = xr.DataArray(values, dims="time", coords={"time": time})
        am = temporal_block_maxima(da, "YE", keep_time=True)
        assert "time_of_max" in am.coords
        got = pd.to_datetime(np.asarray(am["time_of_max"].values).ravel())
        assert got[0] == time[100]
        assert got[1] == time[565]
        # default keeps the old behaviour (no extra coordinate)
        assert "time_of_max" not in temporal_block_maxima(da, "YE").coords

    def test_temporal_keep_time_empty_block_is_nat(self):
        time = pd.date_range("2020-01-01", periods=400, freq="D")
        values = np.full(400, np.nan)
        values[10] = 30.0  # only 2020 has data
        da = xr.DataArray(values, dims="time", coords={"time": time})
        am = temporal_block_maxima(da, "YE", keep_time=True)
        tom = np.asarray(am["time_of_max"].values).ravel()
        assert pd.Timestamp(tom[0]) == time[10]
        assert pd.isnull(tom[1])  # 2021 block has no data -> NaT

    def test_sliding_keep_time(self):
        time = pd.date_range("2020-01-01", periods=10, freq="D")
        values = np.array([1, 5, 2, 9, 1, 1, 7, 2, 8, 3.0])
        da = xr.DataArray(values, dims="time", coords={"time": time})
        sm = sliding_block_maxima(da, 3, keep_time=True, min_periods=1)
        tom = pd.to_datetime(np.asarray(sm["time_of_max"].values))
        assert tom[5] == time[3]  # window [3,4,5] max 9 sits at index 3

    def test_spatial_keep_coords(self):
        arr = np.arange(16).reshape(4, 4).astype(float)
        da = xr.DataArray(
            arr,
            dims=["y", "x"],
            coords={"y": np.arange(4) * 10.0, "x": np.arange(4) * 100.0},
        )
        sm = spatial_block_maxima(da, {"y": 2, "x": 2}, keep_coords=True)
        sm = sm.transpose("y", "x")
        assert float(sm["y_of_max"].values[0, 0]) == 10.0
        assert float(sm["x_of_max"].values[0, 0]) == 100.0
        assert float(sm["x_of_max"].values[0, 1]) == 300.0

    def test_r_largest_keep_time_resample(self):
        time = pd.date_range("2020-01-01", periods=6, freq="D")
        values = np.array([3, 1, 9, 2, 7, 5.0])
        da = xr.DataArray(values, dims="time", coords={"time": time})
        rl = r_largest_block_maxima(da, "YE", r=3, keep_time=True)
        assert "time_of_max" in rl.coords
        tom = pd.to_datetime(np.asarray(rl["time_of_max"].values).ravel())
        assert tom[0] == time[2]  # largest (9) at index 2
        assert tom[1] == time[4]  # second (7) at index 4

    def test_r_largest_keep_time_fixed_blocks(self):
        time = pd.date_range("2020-01-01", periods=6, freq="D")
        values = np.array([3, 1, 9, 2, 7, 5.0])
        da = xr.DataArray(values, dims="time", coords={"time": time})
        rl = r_largest_block_maxima(da, 3, r=2, keep_time=True)
        tom = np.asarray(rl["time_of_max"].transpose("block", "order").values)
        assert pd.Timestamp(tom[0, 0]) == time[2]  # block 0 largest (9)
        assert pd.Timestamp(tom[1, 0]) == time[4]  # block 1 largest (7)

    def test_declustered_runs_keep_time(self):
        time = pd.date_range("2020-01-01", periods=9, freq="D")
        values = np.array([1, 2, 9, 7, 8, 1, 1, 6, 1.0])
        da = xr.DataArray(values, dims="time", coords={"time": time})
        dc = declustered_block_maxima(
            da, threshold=5, min_separation=1, method="runs", keep_time=True
        )
        peaks = set(pd.to_datetime(np.asarray(dc["time_of_max"].values).ravel()))
        assert time[2] in peaks  # run [9,7,8] max at index 2
        assert time[7] in peaks  # isolated 6 at index 7

    def test_sliding_keep_time_masked_by_min_periods(self):
        """Windows failing ``min_periods`` must not report a time_of_max."""
        time = pd.date_range("2020-01-01", periods=6, freq="D")
        values = np.array([1, 5, 2, 9, 1, 7.0])
        da = xr.DataArray(values, dims="time", coords={"time": time})
        sm = sliding_block_maxima(da, 3, keep_time=True)
        tom = np.asarray(sm["time_of_max"].values)
        assert pd.isnull(tom[0]) and pd.isnull(tom[1])  # partial leading windows
        assert pd.Timestamp(tom[3]) == time[3]  # first full window with max 9

    def test_spatial_keep_coords_masked_by_min_periods(self):
        """Underfilled coarsened blocks must not report a location."""
        arr = np.arange(16).reshape(4, 4).astype(float)
        arr[0, 0:3] = np.nan  # block (0, 0) keeps 1 of 4 values, (0, 1) keeps 3
        da = xr.DataArray(
            arr,
            dims=["y", "x"],
            coords={"y": np.arange(4) * 10.0, "x": np.arange(4) * 100.0},
        )
        sm = spatial_block_maxima(
            da, {"y": 2, "x": 2}, min_periods=4, keep_coords=True
        ).transpose("y", "x")
        assert np.isnan(sm.values[0, 0]) and np.isnan(sm.values[0, 1])
        assert np.isnan(sm["y_of_max"].values[0, 0])
        assert np.isnan(sm["x_of_max"].values[0, 1])
        assert float(sm["y_of_max"].values[1, 0]) == 30.0  # full block unaffected

    def test_declustered_runs_keep_time_merged_clusters(self):
        """min_separation merges runs; the peak time must follow the merged max."""
        time = pd.date_range("2020-01-01", periods=9, freq="D")
        values = np.array([1, 2, 9, 7, 8, 1, 1, 6, 1.0])
        da = xr.DataArray(values, dims="time", coords={"time": time})
        # Gap of 2 below-threshold steps between the [9,7,8] run and the
        # isolated 6, so min_separation=3 folds them into one cluster.
        dc = declustered_block_maxima(
            da, threshold=5, min_separation=3, method="runs", keep_time=True
        )
        assert dc.values.tolist() == [9.0]
        tom = pd.to_datetime(np.asarray(dc["time_of_max"].values).ravel())
        assert tom[0] == time[2]  # merged cluster's max (9) at index 2

    def test_sliding_keep_time_survives_stride(self):
        """The argmax coordinate must stay aligned through stride subsampling."""
        time = pd.date_range("2020-01-01", periods=12, freq="D")
        values = np.array([1, 5, 2, 9, 1, 1, 7, 2, 8, 3, 4, 6.0])
        da = xr.DataArray(values, dims="time", coords={"time": time})
        sm = sliding_block_maxima(da, 3, stride=3, keep_time=True)
        tom = pd.to_datetime(np.asarray(sm["time_of_max"].values))
        for value, stamp in zip(sm.values, tom, strict=True):
            assert float(da.sel(time=stamp)) == float(value)

    def test_keep_time_distinguishes_missing_from_neg_inf(self):
        """A genuine -inf max must not lose a tie to a missing sample.

        Filling NaN with -inf (or np.nanargmax, which does so internally) makes
        the two indistinguishable, and the earlier position wins.
        """
        time = pd.date_range("2020-01-01", periods=4, freq="D")
        values = np.array([np.nan, -np.inf, -np.inf, -np.inf])
        da = xr.DataArray(values, dims="time", coords={"time": time})

        am = temporal_block_maxima(da, "YS", keep_time=True)
        assert pd.Timestamp(am["time_of_max"].values.ravel()[0]) == time[1]

        sm = sliding_block_maxima(da, 4, min_periods=1, keep_time=True)
        assert pd.Timestamp(np.asarray(sm["time_of_max"].values)[3]) == time[1]

        grid = xr.DataArray(
            np.array([[np.nan, -np.inf], [-np.inf, -np.inf]]),
            dims=["y", "x"],
            coords={"y": [0.0, 1.0], "x": [0.0, 1.0]},
        )
        gm = spatial_block_maxima(grid, 2, dims=["y", "x"], keep_coords=True)
        assert float(gm["y_of_max"].values.ravel()[0]) == 0.0
        assert float(gm["x_of_max"].values.ravel()[0]) == 1.0

    def test_keep_time_survives_internal_dim_name_collision(self):
        """Internal window dims must not clash with a caller's own dims."""
        time = pd.date_range("2020-01-01", periods=8, freq="D")
        da = xr.DataArray(
            np.arange(16.0).reshape(8, 2),
            dims=("time", "_w"),
            coords={"time": time},
        )
        sm = sliding_block_maxima(da, 3, keep_time=True)
        assert f"{'time'}_of_max" in sm.coords

        for clashing in ("_bw", "_y_win", "_x_win"):
            grid = xr.DataArray(
                np.arange(32.0).reshape(4, 4, 2),
                dims=("y", "x", clashing),
                coords={"y": np.arange(4.0), "x": np.arange(4.0)},
            )
            gm = spatial_block_maxima(grid, 2, dims=["y", "x"], keep_coords=True)
            assert "y_of_max" in gm.coords and "x_of_max" in gm.coords

    def test_keep_time_matches_numpy_on_dask_backed_input(self):
        """Chunked inputs must give the same answer, not raise on the indexer."""
        dask = pytest.importorskip("dask")  # noqa: F841
        time = pd.date_range("2020-01-01", periods=12, freq="D")
        values = np.array([1, 5, 2, 9, 1, 1, 7, 2, 8, 3, 4, 6.0])
        da = xr.DataArray(values, dims="time", coords={"time": time})
        chunked = da.chunk({"time": 4})

        # The coordinate must not force computation.
        assert sliding_block_maxima(chunked, 3, keep_time=True).chunks is not None
        assert temporal_block_maxima(chunked, "MS", keep_time=True).chunks is not None

        def same(got, want):
            # NaT never compares equal, so match the null masks separately.
            got, want = np.asarray(got), np.asarray(want)
            gnull, wnull = pd.isnull(got), pd.isnull(want)
            return np.array_equal(gnull, wnull) and np.array_equal(
                got[~gnull], want[~wnull]
            )

        assert same(
            sliding_block_maxima(chunked, 3, keep_time=True).compute()["time_of_max"],
            sliding_block_maxima(da, 3, keep_time=True)["time_of_max"],
        )
        assert same(
            temporal_block_maxima(chunked, "MS", keep_time=True).compute()[
                "time_of_max"
            ],
            temporal_block_maxima(da, "MS", keep_time=True)["time_of_max"],
        )
        grid = xr.DataArray(
            np.arange(16.0).reshape(4, 4),
            dims=("y", "x"),
            coords={"y": np.arange(4.0), "x": np.arange(4.0)},
        )
        gm = spatial_block_maxima(
            grid.chunk({"y": 2, "x": 2}), 2, dims=["y", "x"], keep_coords=True
        ).compute()
        expected = spatial_block_maxima(grid, 2, dims=["y", "x"], keep_coords=True)
        assert same(gm["y_of_max"], expected["y_of_max"])
        assert same(gm["x_of_max"], expected["x_of_max"])

    def test_keep_time_on_batched_input(self):
        """Batched (multi-dim) input: each slice's stamp must match its own max.

        The coordinate grid carries no batch dims, so it has to be lined up
        with the values before the gather.
        """
        time = pd.date_range("2020-01-01", periods=8, freq="D")
        rng = np.random.RandomState(0)
        da = xr.DataArray(
            rng.rand(3, 8),
            dims=("site", "time"),
            coords={"time": time, "site": [0, 1, 2]},
        )
        for result in (
            sliding_block_maxima(da, 3, keep_time=True),
            temporal_block_maxima(da, "MS", keep_time=True),
        ):
            for site in da["site"].values:
                row = result.sel(site=site)
                stamps = pd.to_datetime(np.asarray(row["time_of_max"].values).ravel())
                values = np.asarray(row.values).ravel()
                for stamp, value in zip(stamps, values, strict=True):
                    if pd.isnull(stamp) or pd.isnull(value):
                        continue
                    assert float(da.sel(site=site, time=stamp)) == pytest.approx(
                        float(value)
                    )

    def test_keep_coords_on_batched_spatial_input(self):
        """Spatial argmax locations must be per-slice, not shared across a batch."""
        rng = np.random.RandomState(1)
        da = xr.DataArray(
            rng.rand(2, 4, 4),
            dims=("t", "y", "x"),
            coords={"t": [0, 1], "y": np.arange(4.0), "x": np.arange(4.0)},
        )
        sm = spatial_block_maxima(da, 2, dims=["y", "x"], keep_coords=True)
        for t in da["t"].values:
            row = sm.sel(t=t).transpose("y", "x")
            for iy in range(2):
                for ix in range(2):
                    yv = float(row["y_of_max"].values[iy, ix])
                    xv = float(row["x_of_max"].values[iy, ix])
                    assert float(da.sel(t=t, y=yv, x=xv)) == pytest.approx(
                        float(row.values[iy, ix])
                    )
