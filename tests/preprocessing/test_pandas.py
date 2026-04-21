"""Tests for `pyrox.preprocessing._pandas`."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from pyrox.nn import (
    FourierFeatures,
    InteractionFeatures,
    SeasonalFeatures,
    Standardization,
)
from pyrox.preprocessing import (
    SpatiotemporalFit,
    encode_time_column,
    fit_spatiotemporal,
    fit_standardization,
)


# -- fit_standardization ----------------------------------------------------


def test_fit_standardization_basic():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    layer = fit_standardization(df, ["a", "b"])
    assert isinstance(layer, Standardization)
    assert jnp.allclose(layer.mu, jnp.array([2.5, 25.0]))
    # std uses ddof=0 (population std) for stability.
    std_a = float(jnp.std(jnp.array([1.0, 2.0, 3.0, 4.0])))
    std_b = float(jnp.std(jnp.array([10.0, 20.0, 30.0, 40.0])))
    assert jnp.allclose(layer.std, jnp.array([std_a, std_b]), rtol=1e-5)


def test_fit_standardization_floors_zero_variance():
    """Constant column should not produce zero std (would NaN downstream)."""
    df = pd.DataFrame({"const": [3.0, 3.0, 3.0]})
    layer = fit_standardization(df, ["const"], eps=1e-3)
    assert float(layer.std[0]) >= 1e-3


# -- encode_time_column -----------------------------------------------------


def test_encode_time_column_int():
    s = pd.Series([10, 11, 12, 13])
    t, time_min, time_scale = encode_time_column(s, timetype="int")
    assert time_min == 10.0
    assert time_scale == 1.0
    assert jnp.allclose(t, jnp.array([0.0, 1.0, 2.0, 3.0]))


def test_encode_time_column_int_uses_supplied_offset():
    s = pd.Series([10, 11, 12])
    t, _, _ = encode_time_column(s, timetype="int", time_min=5.0)
    assert jnp.allclose(t, jnp.array([5.0, 6.0, 7.0]))


def test_encode_time_column_datetime_days():
    s = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    t, _time_min, _time_scale = encode_time_column(s, timetype="datetime", freq="D")
    # Days since 2024-01-01 at the float32 precision JAX uses by default
    # (float64 truncates to float32 without jax_enable_x64; tolerance ~1e-3
    # is enough to catch real regressions).
    assert jnp.allclose(t, jnp.array([0.0, 1.0, 2.0]), atol=1e-2)


def test_encode_time_column_unknown_timetype():
    with pytest.raises(ValueError, match="Unsupported timetype"):
        encode_time_column(pd.Series([1, 2, 3]), timetype="quantum")


# -- fit_spatiotemporal -----------------------------------------------------


def test_fit_spatiotemporal_returns_pytree_bundle():
    df = pd.DataFrame(
        {
            "t": list(range(20)),
            "lat": [0.1 * i for i in range(20)],
            "lon": [-0.05 * i for i in range(20)],
            "z": [i**0.5 for i in range(20)],
        }
    )
    fit = fit_spatiotemporal(
        df,
        feature_cols=["t", "lat", "lon"],
        target_col="z",
        seasonality_periods=(7.0,),
        num_seasonal_harmonics=(3,),
        fourier_degrees=(0, 2, 2),
        interactions=((0, 1),),
    )
    assert isinstance(fit, SpatiotemporalFit)
    assert isinstance(fit.standardize_layer, Standardization)
    assert isinstance(fit.fourier_layer, FourierFeatures)
    assert isinstance(fit.seasonal_layer, SeasonalFeatures)
    assert isinstance(fit.interaction_layer, InteractionFeatures)
    assert fit.feature_cols == ("t", "lat", "lon")
    assert fit.target_col == "z"
    assert fit.fourier_layer.degrees == (0, 2, 2)
    assert fit.seasonal_layer.periods == (7.0,)
    assert fit.seasonal_layer.harmonics == (3,)
    assert fit.interaction_layer.pairs == ((0, 1),)


def test_fit_spatiotemporal_default_degrees_all_zero():
    """Without explicit fourier_degrees, all columns get degree=0."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "z": [5.0, 6.0]})
    fit = fit_spatiotemporal(df, feature_cols=["a", "b"], target_col="z")
    assert fit.fourier_layer.degrees == (0, 0)


def test_encode_time_column_int_preserves_unix_second_precision():
    """Unix-second timestamps must keep unit-level resolution after centering.

    Regression: casting to float32 *before* subtracting `time_min`
    collapses neighboring Unix-second integers because float32 has only
    ~7 decimal digits. Subtract in float64, then cast.
    """
    # Three consecutive Unix seconds around 2024-01-01.
    base = 1_704_067_200  # 2024-01-01T00:00:00Z
    series = pd.Series([base, base + 1, base + 2])
    t, t_min, scale = encode_time_column(series, timetype="int")
    assert t_min == float(base)
    assert scale == 1.0
    t_np = np.asarray(t)
    assert t_np.shape == (3,)
    # Unit-level deltas must survive.
    assert t_np[1] - t_np[0] == pytest.approx(1.0)
    assert t_np[2] - t_np[1] == pytest.approx(1.0)


def test_fit_spatiotemporal_default_does_not_standardize_time():
    """The default `standardize=None` must leave the time column untouched.

    Seasonal features interpret `seasonality_periods` in the *original*
    time units, so z-scoring time would miscalibrate the seasonal
    frequencies unless the user overrides the default.
    """
    df = pd.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0],
            "lat": [10.0, 20.0, 30.0, 40.0],
            "z": [0.1, 0.2, 0.3, 0.4],
        }
    )
    fit = fit_spatiotemporal(df, feature_cols=["t", "lat"], target_col="z")
    # Time axis (index 0) must remain identity.
    assert float(fit.standardize_layer.mu[0]) == 0.0
    assert float(fit.standardize_layer.std[0]) == 1.0
    # Non-time columns are standardized.
    assert float(fit.standardize_layer.mu[1]) == pytest.approx(25.0)


def test_encode_time_column_datetime_rejects_unknown_freq():
    """Typos in `freq` must raise, not silently default to days.

    Regression for a `.get(freq, day_default)` fallback that masked
    typos like `'h'` (vs `'H'`) and silently mis-scaled time.
    """
    series = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    with pytest.raises(ValueError, match="Unsupported freq"):
        encode_time_column(pd.Series(series), timetype="datetime", freq="h")


def test_encode_time_column_datetime_handles_tz_aware():
    """Timezone-aware columns must be normalized to UTC, not crash."""
    series = pd.to_datetime(
        ["2024-01-01T00:00:00", "2024-01-02T00:00:00", "2024-01-03T00:00:00"],
        utc=True,
    )
    t, _t_min, _scale = encode_time_column(pd.Series(series), timetype="datetime")
    t_np = np.asarray(t)
    assert t_np.shape == (3,)
    # Day-unit deltas: 1.0, 1.0
    assert t_np[1] - t_np[0] == pytest.approx(1.0)
    assert t_np[2] - t_np[1] == pytest.approx(1.0)


def test_fit_spatiotemporal_rejects_misaligned_seasonal():
    """`seasonality_periods` and `num_seasonal_harmonics` must match length."""
    df = pd.DataFrame({"t": [0.0, 1.0, 2.0], "z": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="same length"):
        fit_spatiotemporal(
            df,
            feature_cols=["t"],
            target_col="z",
            seasonality_periods=(7.0,),
            num_seasonal_harmonics=(),  # forgot harmonics — silent drop before fix
        )


def test_fit_spatiotemporal_rejects_time_col_in_standardize():
    """Explicitly standardizing the time column must raise."""
    df = pd.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0],
            "lat": [10.0, 20.0, 30.0, 40.0],
            "z": [0.1, 0.2, 0.3, 0.4],
        }
    )
    with pytest.raises(ValueError, match="time column"):
        fit_spatiotemporal(
            df,
            feature_cols=["t", "lat"],
            target_col="z",
            standardize=["t", "lat"],
        )


def test_fit_spatiotemporal_standardize_subset_is_full_size():
    """A subset `standardize=` must still yield a layer aligned with feature_cols.

    Regression for a bug where `standardize_layer.mu / std` were sized
    to the subset, causing `_df_to_design` to broadcast-fail (or silently
    mis-scale) when applied to the full `(N, D)` design matrix.
    """
    df = pd.DataFrame(
        {
            "t": [0.0, 1.0, 2.0, 3.0],
            "lat": [10.0, 20.0, 30.0, 40.0],  # mean=25, std≈11.18
            "lon": [-1.0, -2.0, -3.0, -4.0],
            "z": [0.1, 0.2, 0.3, 0.4],
        }
    )
    fit = fit_spatiotemporal(
        df,
        feature_cols=["t", "lat", "lon"],
        target_col="z",
        standardize=["lat"],  # only lat
    )
    # Layer must be sized to (D,) = (3,), with identity on t and lon.
    assert fit.standardize_layer.mu.shape == (3,)
    assert fit.standardize_layer.std.shape == (3,)
    assert float(fit.standardize_layer.mu[0]) == 0.0  # t identity
    assert float(fit.standardize_layer.std[0]) == 1.0
    assert float(fit.standardize_layer.mu[2]) == 0.0  # lon identity
    assert float(fit.standardize_layer.std[2]) == 1.0
    # lat was standardized.
    assert float(fit.standardize_layer.mu[1]) == pytest.approx(25.0)
    assert float(fit.standardize_layer.std[1]) == pytest.approx(
        float(df["lat"].std(ddof=0))
    )


def test_fit_spatiotemporal_standardize_rejects_unknown_column():
    df = pd.DataFrame({"a": [1.0, 2.0], "z": [5.0, 6.0]})
    with pytest.raises(ValueError, match="subset of feature_cols"):
        fit_spatiotemporal(
            df, feature_cols=["a"], target_col="z", standardize=["not_a_col"]
        )


def test_fit_spatiotemporal_rejects_mismatched_fourier_length():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "z": [5.0, 6.0]})
    with pytest.raises(ValueError, match="fourier_degrees"):
        fit_spatiotemporal(
            df, feature_cols=["a", "b"], target_col="z", fourier_degrees=(1, 2, 3)
        )


def test_pandas_isolation():
    """`pyrox.nn` must not import pandas (layers are pandas-free by design).

    `pyrox.api._bnf` does import pandas, since the estimator facade
    takes pandas DataFrames — that's intentional. We only enforce the
    isolation on `pyrox.nn`.
    """
    import ast
    import pathlib

    nn_dir = pathlib.Path(__file__).parents[2] / "src" / "pyrox" / "nn"
    for py in nn_dir.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "pandas", py
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "pandas", py
