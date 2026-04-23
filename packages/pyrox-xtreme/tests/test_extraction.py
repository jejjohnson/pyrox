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


# Quantile-regression threshold selection needs scikit-learn (optional
# `[threshold]` extra). Skip when sklearn isn't installed.
pytest.importorskip("sklearn")
from xtremax.extraction.quantile_regression import (
    XarrayQuantileRegressor,
    quantile_regression_threshold,
)


class TestQuantileRegression:
    def test_threshold_aligns_shuffled_covariate(self):
        """Regression: `_build_feature_matrix` read `covariates.values`
        without reindexing to the response's time coord, so a covariate
        in a different order paired with the wrong targets and produced
        a numerically wrong threshold with no error raised.
        """
        rng = np.random.default_rng(0)
        time = pd.date_range("2000-01-01", periods=365, freq="D")
        # Response with strong dependence on an integer covariate.
        cov_values = rng.standard_normal(len(time))
        response = 5.0 * cov_values + 0.1 * rng.standard_normal(len(time))
        da = xr.DataArray(response, dims="time", coords={"time": time})
        cov = xr.DataArray(cov_values, dims="time", coords={"time": time})

        # Shuffle the covariate's time axis; its *values* still correspond
        # to the correct timestamps via the coord, so after alignment the
        # fit should match the unshuffled case.
        perm = rng.permutation(len(time))
        cov_shuffled = cov.isel(time=perm)

        t_ordered = quantile_regression_threshold(
            da, quantile=0.9, time_dim="time", covariates=cov
        )
        t_shuffled = quantile_regression_threshold(
            da, quantile=0.9, time_dim="time", covariates=cov_shuffled
        )
        # After reindexing, shuffled input produces the same threshold
        # as the ordered one (within solver tolerance).
        np.testing.assert_allclose(
            t_ordered.values, t_shuffled.values, rtol=1e-4, atol=1e-4
        )

    def test_regressor_fits(self, daily_series):
        reg = XarrayQuantileRegressor(quantile=0.95)
        # Use time as covariate
        t = xr.DataArray(
            np.arange(daily_series.sizes["time"], dtype=float),
            dims="time",
            coords={"time": daily_series["time"]},
        )
        reg.fit(t.expand_dims("feature", axis=-1), daily_series)
        preds = reg.predict(t.expand_dims("feature", axis=-1))
        assert preds.shape == daily_series.shape

    def test_threshold_function(self, daily_series):
        u = quantile_regression_threshold(daily_series, quantile=0.95, time_dim="time")
        assert u.shape == daily_series.shape
