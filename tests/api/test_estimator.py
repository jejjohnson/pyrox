"""Tests for the minimal `EstimatorBase` / `FittedEstimator` facade."""

from __future__ import annotations

import pandas as pd
import pytest

from pyrox.api import EstimatorBase, FittedEstimator


def test_base_fit_raises():
    est = EstimatorBase(feature_cols=("x",), target_col="y")
    with pytest.raises(NotImplementedError, match="fit"):
        est.fit(pd.DataFrame({"x": [1.0], "y": [2.0]}), seed=0)


def test_base_predict_raises():
    fitted = FittedEstimator(config=EstimatorBase(feature_cols=("x",), target_col="y"))
    with pytest.raises(NotImplementedError, match="predict"):
        fitted.predict(pd.DataFrame({"x": [1.0]}))


def test_subclass_can_implement_fit_predict():
    class FakeEstimator(EstimatorBase):
        def fit(self, df, *, seed):
            mean = float(df[self.target_col].mean())
            return FakeFitted(config=self, mean=mean)

    class FakeFitted(FittedEstimator):
        mean: float

        def predict(self, df, *, quantiles=None):
            import jax.numpy as jnp

            n = len(df)
            mean_arr = jnp.full((n,), self.mean)
            if quantiles is None:
                return mean_arr
            return mean_arr, tuple(mean_arr for _ in quantiles)

    est = FakeEstimator(feature_cols=("x",), target_col="y")
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 5.0]})
    fitted = est.fit(df, seed=0)
    out = fitted.predict(df)
    assert out.shape == (2,)
    assert float(out[0]) == 4.0  # mean of [3, 5]
