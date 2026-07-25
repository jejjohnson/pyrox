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


class TestRNGDiscipline:
    """#70 — per-variable salted streams and Generator-API unification."""

    def test_cross_variable_independence(self):
        """Default (equal) seeds must not make nominally independent
        variables comonotonic. Previously temp and wind shared the same
        legacy `random_state=42` uniform stream, giving corr ≈ 0.9992."""
        gmst = generate_gmst_trajectory(n_years=200, seed=0)
        space = generate_spatial_field(n_sites=1, seed=0)
        mu = compute_climate_signal(space, gmst, base_val=20.0, coeffs={"gmst": 1.0})
        temp = simulate_temp_extremes(mu)  # default seed
        wind = simulate_wind_extremes(space, gmst)  # default seed

        # Detrend by removing the shared GMST-driven signal component:
        # correlate the residuals around each variable's own year-mean.
        t = temp["tmax"].values[:, 0] - np.asarray(mu.values)[:, 0]
        w = wind["wind_max"].values[:, 0] - (15.0 + 0.5 * gmst.values)
        corr = np.corrcoef(t, w)[0, 1]
        assert abs(corr) < 0.15

    def test_same_seed_reproducible(self):
        gmst = generate_gmst_trajectory(n_years=10, seed=0)
        space = generate_spatial_field(n_sites=5, seed=0)
        a = simulate_wind_extremes(space, gmst, seed=7)
        b = simulate_wind_extremes(space, gmst, seed=7)
        np.testing.assert_array_equal(a["wind_max"].values, b["wind_max"].values)
        c = simulate_wind_extremes(space, gmst, seed=8)
        assert not np.array_equal(a["wind_max"].values, c["wind_max"].values)


class TestCleanupBatch:
    """#71 — assorted API-hygiene regressions."""

    def test_invalid_trend_type_raises_value_error(self):
        """Previously an invalid trend_type fell through every branch and
        crashed with UnboundLocalError on `trend`."""
        with pytest.raises(ValueError, match="trend_type"):
            generate_gmst_trajectory(trend_type="quadratic")  # type: ignore[arg-type]

    def test_augment_spatial_features_does_not_mutate_input(self):
        from xtremax.simulations.spatial import (
            augment_spatial_features,
            create_iberian_domain,
        )

        ds = create_iberian_domain(res_deg=0.5, seed=0)
        before = set(ds.data_vars)
        out = augment_spatial_features(ds)
        assert set(ds.data_vars) == before  # input untouched
        assert "slope" in out.data_vars and "roughness" in out.data_vars

    def test_spatial_features_handle_transposed_input(self):
        """Codex round on #86: a dataset storing its variables as
        (lon, lat) must produce the same labeled features as the
        (lat, lon) layout — the extractors transpose by name."""
        from xtremax.simulations.spatial import (
            augment_spatial_features,
            create_iberian_domain,
        )

        ds = create_iberian_domain(res_deg=0.5, seed=0)
        ds_t = ds.transpose("lon", "lat")
        out = augment_spatial_features(ds)
        out_t = augment_spatial_features(ds_t)
        for var in ("dist_to_coast", "slope", "aspect", "roughness"):
            np.testing.assert_allclose(
                out[var].transpose("lat", "lon").values,
                out_t[var].transpose("lat", "lon").values,
                equal_nan=True,
            )

    def test_simulators_reject_transposed_parameter_fields(self):
        """Blind `.values.transpose()` used to silently mislabel data when
        inputs arrived with unexpected dims; named transposes now raise."""
        gmst = generate_gmst_trajectory(n_years=10, seed=0)
        space = generate_spatial_field(n_sites=5, seed=0)
        bad_gmst = gmst.rename({"year": "t"})
        with pytest.raises((ValueError, KeyError)):
            simulate_wind_extremes(space, bad_gmst, seed=0)


class TestSimulatorStatistics:
    """#75 — statistical property tests with fixed seeds and generous
    tolerances."""

    def test_gev_samples_respect_weibull_upper_bound(self):
        """shape = -0.1 (Weibull domain) bounds samples above by
        μ − σ/ξ = μ + 10σ. A scipy sign-convention slip would produce a
        heavy upper tail instead and blow through the bound."""
        gmst = generate_gmst_trajectory(n_years=200, seed=0)
        space = generate_spatial_field(n_sites=10, seed=0)
        mu = compute_climate_signal(space, gmst, base_val=20.0, coeffs={"gmst": 1.0})
        scale, shape = 1.5, -0.1
        ds = simulate_temp_extremes(mu, scale=scale, shape=shape, seed=0)
        upper = np.asarray(mu.values) + scale / abs(shape)
        assert np.all(ds["tmax"].values <= upper + 1e-6)

    def test_gamma_intensity_mean_matches_cc_scaling(self):
        """rx1day is Gamma with mean equal to the Clausius-Clapeyron-
        scaled location parameter; the empirical mean ratio over all
        cells must be ≈ 1."""
        gmst = generate_gmst_trajectory(n_years=200, seed=0)
        space = generate_spatial_field(n_sites=20, seed=0)
        ds = simulate_precip_extremes(space, gmst, seed=0)

        base_intensity = 40.0 + 0.01 * space["elevation"]
        loc_intensity = (base_intensity * (1.0 + 0.07 * gmst)).transpose("year", "site")
        ratio = ds["rx1day"].values / loc_intensity.values
        # 4000 cells, per-cell sd of the ratio is 0.5 (Gamma shape 4).
        assert abs(float(ratio.mean()) - 1.0) < 0.05


class TestSpatialGeometry:
    """#75 — mask geometry, elevation clipping, and output dims/coords
    (previously entirely untested)."""

    def test_domain_geometry_and_clipping(self):
        from xtremax.simulations.spatial import create_iberian_domain

        ds = create_iberian_domain(res_deg=0.25, seed=0)
        land_fraction = float(ds["mask"].mean())
        assert 0.1 < land_fraction < 0.9
        assert float(ds["elevation"].min()) >= -100.0
        assert float(ds["elevation"].max()) <= 3400.0
        # Ocean is flagged at exactly -100 m.
        assert float(ds["elevation"].where(~ds["mask"]).max()) == -100.0

    def test_advanced_climate_signal_dims_and_masking(self):
        from xtremax.simulations.spatial import (
            augment_spatial_features,
            compute_advanced_climate_signal,
            create_iberian_domain,
        )

        ds = augment_spatial_features(create_iberian_domain(res_deg=0.5, seed=0))
        gmst = generate_gmst_trajectory(n_years=5, seed=0)
        out = compute_advanced_climate_signal(ds, gmst)

        for var in ("mu_tmax", "mu_precip"):
            assert set(out[var].dims) == {"year", "lat", "lon"}
            assert out[var].sizes["year"] == 5
            # Ocean cells are masked to NaN, land cells are finite.
            ocean = out[var].where(~ds["mask"])
            land = out[var].where(ds["mask"])
            assert bool(ocean.isnull().all())
            assert bool(np.isfinite(land).any())


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
