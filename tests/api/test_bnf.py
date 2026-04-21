"""Tests for `pyrox.api._bnf` — BNFEstimator family on tiny synthetic data."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from pyrox.api import BNFEstimator, BNFEstimatorMLE, BNFEstimatorVI, FittedBNF


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
