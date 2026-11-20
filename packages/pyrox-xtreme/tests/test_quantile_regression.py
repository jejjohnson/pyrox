"""Tests for quantile-regression threshold selection.

Kept separate from ``test_extraction.py`` so the module-level
``importorskip("sklearn")`` only skips these tests — not the whole
extraction suite — in environments without the optional ``[threshold]``
extra (#66).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


pytest.importorskip("sklearn")

from xtremax.extraction.quantile_regression import (
    XarrayQuantileRegressor,
    quantile_regression_threshold,
)


@pytest.fixture
def daily_series():
    rng = np.random.default_rng(0)
    time = pd.date_range("2000-01-01", periods=365 * 3, freq="D")
    values = rng.standard_normal(len(time)) + 0.1 * np.arange(len(time)) / len(time)
    return xr.DataArray(values, dims="time", coords={"time": time})


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

    def test_pinball_coverage(self, daily_series):
        """#75 — the fitted τ-quantile threshold must leave ≈ (1 − τ)
        of the responses above it (pinball-loss coverage property)."""
        tau = 0.9
        u = quantile_regression_threshold(daily_series, quantile=tau, time_dim="time")
        frac_below = float((daily_series < u).mean())
        assert abs(frac_below - tau) < 0.03

    def test_misaligned_covariate_raises(self):
        """Covariates missing timestamps become NaN rows after reindexing;
        sklearn would raise an opaque error (or silently misfit), so the
        feature builder must fail fast with a clear message."""
        time = pd.date_range("2000-01-01", periods=100, freq="D")
        rng = np.random.default_rng(0)
        da = xr.DataArray(
            rng.standard_normal(len(time)), dims="time", coords={"time": time}
        )
        cov = xr.DataArray(
            rng.standard_normal(50), dims="time", coords={"time": time[:50]}
        )
        with pytest.raises(ValueError, match="NaN"):
            quantile_regression_threshold(
                da, quantile=0.9, time_dim="time", covariates=cov
            )
