"""
Quantile regression utilities for extremes thresholding.

Provides two layers:

1. ``XarrayQuantileRegressor`` — a thin sklearn-compatible wrapper around
   ``sklearn.linear_model.QuantileRegressor`` that accepts and returns
   ``xr.DataArray`` objects, preserving coordinates through fit → predict.

2. ``quantile_regression_threshold`` — domain-specific function that builds
   the feature matrix (numeric time, optional covariates) and delegates to
   the regressor, returning a time-varying threshold suitable for
   peak-over-threshold analysis.

Example
-------
>>> import xarray as xr, numpy as np, pandas as pd
>>> da = xr.DataArray(
...     np.random.default_rng(0).standard_normal(200),
...     dims="time",
...     coords={"time": pd.date_range("2000", periods=200, freq="D")},
... )
>>> threshold = quantile_regression_threshold(da, quantile=0.95, time_dim="time")
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import QuantileRegressor as _SklearnQuantileRegressor
from sklearn.utils.validation import check_is_fitted


# ---------------------------------------------------------------------------
# Layer 1: sklearn-compatible regressor that operates on xr.DataArray
# ---------------------------------------------------------------------------


class XarrayQuantileRegressor(BaseEstimator, RegressorMixin):
    """Quantile regressor that operates on :class:`xr.DataArray` inputs.

    Thin wrapper around :class:`sklearn.linear_model.QuantileRegressor`.
    Both ``fit`` and ``predict`` expect ``DataArray`` objects; coordinate
    metadata on the leading dimension is preserved through the round-trip.

    Parameters
    ----------
    quantile : float
        Target quantile ∈ (0, 1).
    alpha : float
        L1 regularisation strength.  ``0`` gives the classical
        unregularised linear quantile regression (matching the default
        behaviour of ``statsmodels.QuantReg``).
    solver : str
        Solver forwarded to ``QuantileRegressor``.
        ``"highs"`` (the default) is robust for most problem sizes.
    fit_intercept : bool
        Whether the sklearn estimator should fit its own intercept.
        Set to ``False`` if you supply a constant column yourself.
    """

    def __init__(
        self,
        quantile: float = 0.5,
        alpha: float = 0.0,
        solver: str = "highs",
        fit_intercept: bool = True,
    ):
        self.quantile = quantile
        self.alpha = alpha
        self.solver = solver
        self.fit_intercept = fit_intercept

    # -- sklearn API --------------------------------------------------------

    def fit(self, X: xr.DataArray, y: xr.DataArray) -> XarrayQuantileRegressor:
        """Fit the quantile regression model.

        Parameters
        ----------
        X : xr.DataArray, shape (n_samples,) or (n_samples, n_features)
            Feature matrix.
        y : xr.DataArray, shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        X_np = X.values
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        self.model_ = _SklearnQuantileRegressor(
            quantile=self.quantile,
            alpha=self.alpha,
            solver=self.solver,
            fit_intercept=self.fit_intercept,
        )
        self.model_.fit(X_np, y.values.ravel())
        self.n_features_in_ = X_np.shape[1]
        return self

    def predict(self, X: xr.DataArray) -> xr.DataArray:
        """Predict quantile values.

        Parameters
        ----------
        X : xr.DataArray
            Feature matrix (same schema as training *X*).

        Returns
        -------
        xr.DataArray
            Predicted quantile values with the leading dimension's
            coordinates preserved from *X*.
        """
        check_is_fitted(self, "model_")

        first_dim = X.dims[0]
        X_np = X.values
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        y_hat = self.model_.predict(X_np)
        return xr.DataArray(
            y_hat,
            dims=[first_dim],
            coords={first_dim: X.coords[first_dim]},
        )

    # -- introspection ------------------------------------------------------

    @property
    def coef_(self) -> np.ndarray:
        """Regression coefficients (delegates to the inner model)."""
        check_is_fitted(self, "model_")
        return self.model_.coef_

    @property
    def intercept_(self) -> float:
        """Intercept (delegates to the inner model)."""
        check_is_fitted(self, "model_")
        return self.model_.intercept_


# ---------------------------------------------------------------------------
# Layer 2: domain wrapper for peak-over-threshold analysis
# ---------------------------------------------------------------------------


def _time_to_numeric(time_values: np.ndarray) -> np.ndarray:
    """Convert datetime-like values to float days since first observation.

    Handles ``np.datetime64`` and already-numeric arrays.
    """
    if np.issubdtype(np.asarray(time_values).dtype, np.datetime64):
        t0 = time_values[0]
        return (time_values - t0) / np.timedelta64(1, "D")
    return np.asarray(time_values, dtype=np.float64)


def _build_feature_matrix(
    da: xr.DataArray,
    time_dim: str,
    covariates: xr.DataArray | None = None,
) -> xr.DataArray:
    """Assemble the (n_samples, n_features) design matrix as a DataArray.

    Features are ``[t, cov_1, ..., cov_k]`` where *t* is numeric time
    and ``cov_i`` are optional covariate columns.  An intercept is *not*
    added here — the regressor handles that via ``fit_intercept=True``.

    Returns
    -------
    xr.DataArray
        Shape ``(n_time, n_features)`` with a ``"feature"`` dimension.
    """
    time_vals = da.coords[time_dim].values
    t_numeric = _time_to_numeric(time_vals)

    columns = [t_numeric]
    feature_names = ["time"]

    if covariates is not None:
        # Align covariates to the response's time coordinate before
        # extracting raw values. Without this, a covariate series in a
        # different order (or with gaps) silently pairs with the wrong
        # targets and produces numerically wrong thresholds with no
        # error raised.
        covariates = covariates.reindex({time_dim: da.coords[time_dim]})
        cov_np = covariates.values
        if cov_np.ndim == 1:
            columns.append(cov_np)
            feature_names.append("covariate_0")
        else:
            for i in range(cov_np.shape[1]):
                columns.append(cov_np[:, i])
                feature_names.append(f"covariate_{i}")

    X = np.column_stack(columns)
    return xr.DataArray(
        X,
        dims=[time_dim, "feature"],
        coords={time_dim: da.coords[time_dim], "feature": feature_names},
    )


def quantile_regression_threshold(
    da: xr.DataArray,
    quantile: float,
    time_dim: str,
    covariates: xr.DataArray | None = None,
    *,
    alpha: float = 0.0,
    solver: str = "highs",
) -> xr.DataArray:
    """Estimate a time-varying threshold via linear quantile regression.

    Fits the model

    .. math::

        Q_{\\tau}(y \\mid t, \\mathbf{z}) = \\beta_0
            + \\beta_1 t + \\boldsymbol{\\beta}_z^\\top \\mathbf{z}

    where *t* is numeric time and **z** are optional covariates, then
    returns the fitted quantile surface evaluated at the observed times.
    The result is suitable as a non-stationary threshold for
    peak-over-threshold (POT) extreme value analysis.

    Parameters
    ----------
    da : xr.DataArray
        Response variable (1-D along *time_dim*).
    quantile : float
        Target quantile τ ∈ (0, 1), e.g. 0.95 for the 95th percentile.
    time_dim : str
        Name of the time dimension in *da*.
    covariates : xr.DataArray, optional
        Additional covariates aligned along *time_dim*.
    alpha : float
        L1 regularisation passed to the regressor (0 = unregularised).
    solver : str
        LP solver, forwarded to ``QuantileRegressor``.

    Returns
    -------
    xr.DataArray
        Fitted threshold values with the same coords as *da*.

    Examples
    --------
    >>> # 95th-percentile linear-trend threshold
    >>> threshold = quantile_regression_threshold(da, 0.95, "time")

    >>> # With a covariate (e.g., sea-surface temperature)
    >>> threshold = quantile_regression_threshold(
    ...     da, 0.99, "time", covariates=sst
    ... )
    """
    X = _build_feature_matrix(da, time_dim, covariates)

    reg = XarrayQuantileRegressor(
        quantile=quantile,
        alpha=alpha,
        solver=solver,
        fit_intercept=True,
    )
    reg.fit(X, da)
    return reg.predict(X)
