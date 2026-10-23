"""Tests for `pyrox_nn.api._bnf` — BNFEstimator family on tiny synthetic data."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from pyrox_nn.api import BNFEstimator, BNFEstimatorMLE, BNFEstimatorVI, FittedBNF


def _toy_dataset(seed: int = 42, n: int = 200, train_frac: float = 0.5):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n)
    y_truth = np.sin(x)
    y = y_truth + 0.1 * rng.normal(size=n)
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    train, test = perm[:n_train], perm[n_train:]
    df_train = pd.DataFrame({"t": x[train], "z": y[train]})
    df_test = pd.DataFrame({"t": x[test], "z": y[test]})
    return df_train, df_test, y_truth[test], y[test]


# -- API contract tests -----------------------------------------------------


def test_bnf_estimator_returns_fitted_bnf():
    df_train, _, _, _ = _toy_dataset()
    est = BNFEstimator(
        feature_cols=("t",),
        target_col="z",
        width=8,
        depth=2,
        fourier_degrees=(2,),
        sigma_obs=0.1,
        ensemble_size=2,
        num_epochs=200,
    )
    fitted = est.fit(df_train, seed=0)
    assert isinstance(fitted, FittedBNF)
    assert fitted.losses.shape == (2, 200)


def test_bnf_estimator_predict_shape():
    df_train, df_test, _, _ = _toy_dataset()
    est = BNFEstimator(
        feature_cols=("t",),
        target_col="z",
        width=8,
        depth=2,
        fourier_degrees=(2,),
        sigma_obs=0.1,
        ensemble_size=2,
        num_epochs=200,
    )
    fitted = est.fit(df_train, seed=0)
    mean = fitted.predict(df_test)
    assert mean.shape == (len(df_test),)


def test_bnf_estimator_predict_quantiles_shape():
    df_train, df_test, _, _ = _toy_dataset()
    est = BNFEstimator(
        feature_cols=("t",),
        target_col="z",
        width=8,
        depth=2,
        fourier_degrees=(2,),
        sigma_obs=0.1,
        ensemble_size=2,
        num_epochs=200,
    )
    fitted = est.fit(df_train, seed=0)
    mean, qs = fitted.predict(df_test, quantiles=(0.1, 0.5, 0.9))
    assert mean.shape == (len(df_test),)
    assert len(qs) == 3
    assert all(q.shape == (len(df_test),) for q in qs)


def test_bnf_estimator_mle_default_prior_weight_zero():
    est = BNFEstimatorMLE(feature_cols=("t",), target_col="z")
    assert est.prior_weight == 0.0


def test_bnf_estimator_vi_inference_kind():
    est = BNFEstimatorVI(feature_cols=("t",), target_col="z")
    assert est.inference_kind == "vi"


def test_bnf_estimator_standardize_subset_end_to_end():
    """`standardize_columns` as a strict subset must not break fit/predict.

    Regression for the subset broadcasting bug: previously the fitted
    `Standardization` layer was sized to the subset, then
    `_df_to_design` applied it to the full `(N, D)` design matrix and
    either mis-scaled or raised a broadcast error.
    """
    rng = np.random.default_rng(0)
    n = 40
    t = np.linspace(0, 10, n)
    lat = 50.0 + 10.0 * rng.normal(size=n)
    z = np.sin(t) + 0.1 * rng.normal(size=n)
    df = pd.DataFrame({"t": t, "lat": lat, "z": z})
    est = BNFEstimator(
        feature_cols=("t", "lat"),
        target_col="z",
        width=4,
        depth=2,
        fourier_degrees=(1, 0),
        sigma_obs=0.1,
        ensemble_size=2,
        num_epochs=50,
        standardize_columns=("lat",),
    )
    fitted = est.fit(df, seed=0)
    mean = fitted.predict(df)
    assert mean.shape == (n,)
    assert np.all(np.isfinite(np.asarray(mean)))


def test_bnf_estimator_rejects_unsupported_observation_model():
    df_train, _, _, _ = _toy_dataset()
    est = BNFEstimator(
        feature_cols=("t",),
        target_col="z",
        observation_model="NB",
        width=4,
        depth=2,
    )
    with pytest.raises(NotImplementedError, match="gaussx"):
        est.fit(df_train, seed=0)


# -- Convergence sanity checks (slow) ---------------------------------------


@pytest.mark.slow
def test_bnf_estimator_converges_on_sin_curve():
    """MLE BNFEstimator should fit sin(x) below the noise floor on train data."""
    df_train, df_test, y_test_truth, _ = _toy_dataset(n=400)
    est = BNFEstimatorMLE(
        feature_cols=("t",),
        target_col="z",
        width=32,
        depth=3,
        fourier_degrees=(3,),
        sigma_obs=0.1,
        ensemble_size=4,
        num_epochs=2000,
        learning_rate=5e-3,
    )
    fitted = est.fit(df_train, seed=0)
    mean_test = fitted.predict(df_test)
    rmse = float(jnp.sqrt(jnp.mean((mean_test - y_test_truth) ** 2)))
    # Should be well within the noise floor on a random in-distribution split.
    assert rmse < 0.2, f"RMSE {rmse:.4f} too high; the BNF is not fitting"


@pytest.mark.slow
def test_bnf_estimator_quantile_coverage():
    """95% quantile band should cover ~95% of the noisy test points."""
    df_train, df_test, _, y_test_obs = _toy_dataset(n=400)
    est = BNFEstimator(
        feature_cols=("t",),
        target_col="z",
        width=32,
        depth=3,
        fourier_degrees=(3,),
        sigma_obs=0.1,
        ensemble_size=8,
        num_epochs=5000,
        learning_rate=5e-3,
    )
    fitted = est.fit(df_train, seed=0)
    _, qs = fitted.predict(df_test, quantiles=(0.025, 0.5, 0.975))
    lower, _, upper = qs
    lo_np = np.asarray(lower)
    hi_np = np.asarray(upper)
    coverage = float(((y_test_obs >= lo_np) & (y_test_obs <= hi_np)).mean())
    assert coverage >= 0.85, f"95% interval coverage {coverage:.2f} too low"


# -- Encoder ------------------------------------------------------------------


def _sphere_field(lon, lat):
    """A smooth global field (degree <= 2 harmonics), continuous at ±180°."""
    return np.cos(lat) * np.cos(lon - 0.5) + 0.5 * (3.0 * np.sin(lat) ** 2 - 1.0)


def _sphere_dataset(seed: int, n: int, noise: float = 0.05):
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-np.pi, np.pi, n)
    lat = np.arcsin(rng.uniform(-1.0, 1.0, n))  # uniform on the sphere
    t = rng.integers(0, 10, n).astype(float)
    z = _sphere_field(lon, lat) + noise * rng.normal(size=n)
    return pd.DataFrame({"t": t, "lon": lon, "lat": lat, "z": z})


def test_bnf_estimator_encoder_sees_columns_in_dataframe_units():
    """The encoder is composed with the inverse of the column standardization."""
    import equinox as eqx
    from pyrox_nn.api._bnf import _build_bnf, _df_to_design
    from pyrox_nn.preprocessing import fit_spatiotemporal

    df = _sphere_dataset(0, 50)
    fit = fit_spatiotemporal(df, feature_cols=("t", "lon", "lat"), target_col="z")
    bnf = _build_bnf(fit, 4, 1, encoder=eqx.nn.Lambda(lambda z: z), encoder_cols=(2, 1))
    x = _df_to_design(df, fit)
    seen = np.asarray(jax.vmap(lambda xi: bnf.encoder(xi[jnp.asarray([2, 1])]))(x))
    # float32 design matrix: standardize then invert, ~1e-7 relative.
    np.testing.assert_allclose(seen, df[["lat", "lon"]].to_numpy(), atol=1e-5)


def test_bnf_estimator_with_encoder_fits_and_predicts():
    from geonnax import SphericalHarmonicEncoder

    df = _sphere_dataset(0, 64)
    enc = SphericalHarmonicEncoder(l_max=2, input_mode="lonlat")
    est = BNFEstimator(
        feature_cols=("t", "lon", "lat"),
        target_col="z",
        width=8,
        depth=1,
        encoder=enc,
        encoder_cols=(1, 2),
        ensemble_size=2,
        num_epochs=20,
    )
    fitted = est.fit(df, seed=0)
    assert "bnf.feature_gain_1" in fitted.params  # the encoder block's gain
    assert fitted.predict(df.iloc[:5]).shape == (5,)


@pytest.mark.slow
def test_bnf_estimator_spherical_harmonic_encoder_has_no_dateline_seam():
    """A SH-encoded BNF fits a smooth global field with no seam at ±180°.

    The estimator feeds lon/lat only to the encoder, so the field is periodic
    in longitude by construction; this checks it end to end, through the
    standardization round trip, on a fitted model.
    """
    from geonnax import SphericalHarmonicEncoder

    df_train = _sphere_dataset(0, 1500)
    df_test = _sphere_dataset(1, 500)
    est = BNFEstimatorMLE(
        feature_cols=("t", "lon", "lat"),
        target_col="z",
        width=32,
        depth=2,
        encoder=SphericalHarmonicEncoder(l_max=4, input_mode="lonlat"),
        encoder_cols=(1, 2),
        sigma_obs=0.05,
        ensemble_size=2,
        num_epochs=1500,
        learning_rate=5e-3,
    )
    fitted = est.fit(df_train, seed=0)

    truth = _sphere_field(df_test["lon"].to_numpy(), df_test["lat"].to_numpy())
    rmse = float(np.sqrt(np.mean((np.asarray(fitted.predict(df_test)) - truth) ** 2)))
    # The field's standard deviation on the sphere is ~0.7; a fit with
    # RMSE < 0.1 explains > 98% of its variance.
    assert rmse < 0.1, f"RMSE {rmse:.4f}: the SH-encoded BNF is not fitting"

    lat = np.deg2rad(np.linspace(-80.0, 80.0, 41))
    t = np.full_like(lat, 5.0)
    east = pd.DataFrame({"t": t, "lon": np.full_like(lat, np.pi), "lat": lat})
    west = pd.DataFrame({"t": t, "lon": np.full_like(lat, -np.pi), "lat": lat})
    jump = np.abs(np.asarray(fitted.predict(east)) - np.asarray(fitted.predict(west)))
    # Float32 round-off only: the design matrix is float32 and lon = ±π
    # passes through standardize / destandardize (eps 1.2e-7 at |lon| ~ 3)
    # before sin/cos, and the trained network amplifies that by its
    # Lipschitz constant (O(10) here); 1e-4 is three orders of magnitude
    # below the fit error, where a seam would be invisible anyway.
    assert jump.max() < 1e-4, f"max seam jump {jump.max():.2e}"
