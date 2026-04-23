"""Smoke tests for simulation generators (temporal, spatial, extremes)."""

from __future__ import annotations

import numpy as np
import pytest

from xtremax.simulations import (
    compute_climate_signal,
    generate_fractal_terrain,
    generate_gmst_trajectory,
    generate_iberia_mask,
    generate_physical_gmst,
    generate_spatial_field,
    simulate_precip_extremes,
    simulate_temp_extremes,
    simulate_wind_extremes,
)


class TestTemporal:
    @pytest.mark.parametrize("trend", ["linear", "exponential", "logistic"])
    def test_gmst_trajectory_shape(self, trend):
        n_years = 50
        da = generate_gmst_trajectory(
            n_years=n_years, start_year=1970, trend_type=trend, seed=0
        )
        assert da.dims == ("year",)
        assert da.sizes["year"] == n_years
        assert np.all(np.isfinite(da.values))

    @pytest.mark.parametrize("n_years", [5, 9])
    def test_physical_gmst_short_runs(self, n_years):
        """Regression: `rng.uniform(5, n_years - 5, ...)` raised ValueError
        for n_years < 10 because `high < low`. Short runs must now return
        a valid dataset with an empty eruption schedule.
        """
        ds = generate_physical_gmst(n_years=n_years, seed=0)
        assert "gmst" in ds.data_vars
        assert np.all(np.isfinite(ds["gmst"].values))

    @pytest.mark.parametrize("n_years", [10, 20, 50])
    def test_physical_gmst_year_count(self, n_years):
        """Regression: `np.linspace(0, n_years, n_years * 12)` included
        `t = n_years`, so the floor(t) grouping created an extra final bin
        with a single sample and produced `n_years + 1` annual rows
        instead of `n_years`.
        """
        ds = generate_physical_gmst(n_years=n_years, seed=0)
        assert ds.sizes["year"] == n_years


class TestSpatial:
    def test_iberia_mask(self):
        lat = np.linspace(36.0, 44.0, 20)
        lon = np.linspace(-9.0, 3.0, 30)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
        mask = generate_iberia_mask(lat_grid, lon_grid)
        assert mask.shape == lat_grid.shape
        assert mask.dtype == bool
        assert mask.any()

    def test_fractal_terrain(self):
        terrain = generate_fractal_terrain(shape=(32, 32), seed=0)
        assert terrain.shape == (32, 32)
        assert np.all(np.isfinite(terrain))

    def test_iberian_domain_honors_requested_resolution(self):
        """Regression: `n = int((max-min)/res)` + `np.linspace(min, max, n)`
        created `n` points over a closed interval, so realized spacing
        was `(max-min)/(n-1)` — not `res_deg`. E.g. 36–44 at 0.1 yielded
        ~0.1013 spacing, silently distorting downstream distance-derived
        features.
        """
        from xtremax.simulations.spatial import create_iberian_domain

        ds = create_iberian_domain(res_deg=0.1, bounds=(-10, 5, 36, 44), seed=0)
        # Both axes must have exactly `res_deg` spacing.
        lat_spacing = np.diff(ds["lat"].values)
        lon_spacing = np.diff(ds["lon"].values)
        np.testing.assert_allclose(lat_spacing, 0.1, atol=1e-10)
        np.testing.assert_allclose(lon_spacing, 0.1, atol=1e-10)

    def test_fractal_terrain_does_not_mutate_global_rng(self):
        """Regression: `generate_fractal_terrain` called `np.random.seed(seed)`
        at the top, which leaks reproducibility coupling into unrelated
        NumPy random draws afterwards. With a local Generator the global
        RNG state must be untouched.
        """
        # Capture a pre-call draw from the global RNG.
        before = np.random.default_rng()  # local handle
        np.random.seed(12345)
        expected_after_global_draw = np.random.standard_normal(3)

        # Re-seed the global RNG to a known state; then call the helper.
        np.random.seed(12345)
        _ = generate_fractal_terrain(shape=(16, 16), seed=999)
        # If the helper had not mutated global state, drawing 3 values
        # from the global RNG now should still match the pre-recorded draw.
        actual_after_global_draw = np.random.standard_normal(3)
        np.testing.assert_array_equal(
            expected_after_global_draw, actual_after_global_draw
        )
        # Avoid unused-variable lint warning.
        del before


class TestExtremes:
    def test_spatial_field(self):
        ds = generate_spatial_field(n_sites=20, seed=0)
        assert ds.sizes["site"] == 20
        for name in ("lon", "lat", "elevation"):
            assert name in ds.data_vars

    def test_temp_extremes(self):
        gmst = generate_gmst_trajectory(n_years=10, seed=0)
        space = generate_spatial_field(n_sites=5, seed=0)
        mu = compute_climate_signal(
            space,
            gmst,
            base_val=20.0,
            coeffs={"elevation": -6.5, "lat": -0.5, "gmst": 1.0, "interaction": 0.1},
        )
        ds = simulate_temp_extremes(mu, scale=1.5, shape=-0.1, seed=0)
        assert "tmax" in ds.data_vars
        assert ds["tmax"].shape == (10, 5)

    def test_precip_extremes(self):
        gmst = generate_gmst_trajectory(n_years=10, seed=0)
        space = generate_spatial_field(n_sites=5, seed=0)
        ds = simulate_precip_extremes(space, gmst, seed=0)
        assert "rx1day" in ds.data_vars
        assert "cwd" in ds.data_vars

    def test_wind_extremes(self):
        gmst = generate_gmst_trajectory(n_years=10, seed=0)
        space = generate_spatial_field(n_sites=5, seed=0)
        ds = simulate_wind_extremes(space, gmst, seed=0)
        assert "wind_max" in ds.data_vars
