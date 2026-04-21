"""Pandas-side helpers for fitting BNF feature layers.

Pandas appears here and *only* here. The output is always a JAX-only
PyTree (an :class:`equinox.Module` tree), so downstream layers and
estimators stay pandas-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import pandas as pd
from jaxtyping import Array, Float

from pyrox.nn._bnf import (
    FourierFeatures,
    InteractionFeatures,
    SeasonalFeatures,
    Standardization,
)


class SpatiotemporalFit(eqx.Module):
    """Immutable bundle of fitted feature layers + time-encoding scalars.

    Replaces bayesnf's mutable ``SpatiotemporalDataHandler`` with a pure
    PyTree so the whole bundle is JIT-friendly and picklable.

    Attributes:
        standardize_layer: :class:`Standardization` layer applied to the
            feature columns at predict time.
        fourier_layer: :class:`FourierFeatures` layer (may have all-zero
            degrees if the user opted out of Fourier features).
        seasonal_layer: :class:`SeasonalFeatures` layer (zero-period if
            the user opted out of seasonal features).
        interaction_layer: :class:`InteractionFeatures` layer
            (zero-pair if the user opted out).
        time_min: Minimum time value across the training set, used as
            an offset by :func:`encode_time_column`.
        time_scale: Multiplicative factor applied to ``(t - time_min)``
            to produce the array passed to the seasonal layer (typically
            ``1.0`` for ``int`` time columns and a unit-conversion
            factor for ``datetime``).
        feature_cols: Names of the columns the standardization/feature
            layers expect, in order.
        target_col: Name of the target column.
    """

    standardize_layer: Standardization
    fourier_layer: FourierFeatures
    seasonal_layer: SeasonalFeatures
    interaction_layer: InteractionFeatures
    time_min: float = eqx.field(static=True)
    time_scale: float = eqx.field(static=True)
    feature_cols: tuple[str, ...] = eqx.field(static=True)
    target_col: str = eqx.field(static=True)


def fit_standardization(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    eps: float = 1e-12,
) -> Standardization:
    """Build a :class:`Standardization` layer from per-column mean / std.

    Args:
        df: Source DataFrame.
        columns: Columns to standardize, in the order they will appear
            in the array passed to :meth:`Standardization.__call__`.
        eps: Floor for the standard deviation; protects against
            division by zero on constant columns.

    Returns:
        :class:`Standardization` layer.
    """
    sub = df[list(columns)]
    mu = jnp.asarray(sub.mean().to_numpy(), dtype=jnp.float32)
    std = jnp.asarray(sub.std(ddof=0).to_numpy(), dtype=jnp.float32)
    std = jnp.maximum(std, eps)
    return Standardization(mu=mu, std=std)  # ty: ignore[invalid-return-type]


def encode_time_column(
    series: pd.Series,
    *,
    timetype: Literal["int", "datetime"] = "int",
    freq: str | None = None,
    time_min: float | None = None,
) -> tuple[Float[Array, " N"], float, float]:
    """Convert a pandas time column into a unit-scale JAX float array.

    For ``timetype="int"``, the series is cast directly to ``float32``
    and offset by its minimum (or by ``time_min`` if supplied).

    For ``timetype="datetime"``, the series is converted to integer
    nanoseconds, offset by its minimum, and divided by a unit factor
    derived from ``freq`` (``"D"`` ⇒ days, ``"H"`` ⇒ hours, ``"W"`` ⇒
    weeks). When ``freq`` is ``None``, the unit is days.

    Args:
        series: 1D time column.
        timetype: ``"int"`` for already-numeric series, ``"datetime"``
            for ``pd.Timestamp``-valued series.
        freq: Optional unit string for the datetime path.
        time_min: Optional fixed offset (use the value from a previous
            ``fit`` to align test data with training).

    Returns:
        ``(t, time_min, time_scale)`` — the encoded array, the offset
        used, and the multiplicative scale (``1`` for the ``int`` path,
        ``1 / nanoseconds-per-unit`` for ``datetime``).
    """
    if timetype == "int":
        arr = jnp.asarray(series.to_numpy(), dtype=jnp.float32)
        if time_min is None:
            time_min = float(arr.min())
        return arr - time_min, float(time_min), 1.0
    if timetype == "datetime":
        # Keep everything in int64 (numpy-side) to avoid float32 precision
        # loss — nanoseconds since epoch overflow float32 badly.
        #
        # Cast explicitly to `datetime64[ns]` because pandas 2.x may pick
        # microsecond resolution by default, which would make the int64
        # representation off by a factor of 1000 from what `unit_ns`
        # below assumes.
        import numpy as np

        ns_np = (
            pd.to_datetime(series).astype("datetime64[ns]").astype("int64").to_numpy()
        )
        unit_ns = {
            None: 24 * 3600 * 1_000_000_000,
            "D": 24 * 3600 * 1_000_000_000,
            "H": 3600 * 1_000_000_000,
            "W": 7 * 24 * 3600 * 1_000_000_000,
            "min": 60 * 1_000_000_000,
        }.get(freq, 24 * 3600 * 1_000_000_000)
        scale = 1.0 / float(unit_ns)
        if time_min is None:
            time_min = float(ns_np.min()) * scale
        # Subtract the offset in ns (int64), *then* scale; keeps the
        # result well within float32 range.
        origin_ns = int(time_min / scale)
        centered = (ns_np - origin_ns).astype(np.float64)
        return jnp.asarray(centered * scale, dtype=jnp.float32), float(time_min), scale
    raise ValueError(f"Unsupported timetype {timetype!r}")


def fit_spatiotemporal(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    timetype: Literal["int", "datetime"] = "int",
    freq: str | None = None,
    seasonality_periods: Sequence[float] = (),
    num_seasonal_harmonics: Sequence[int] = (),
    fourier_degrees: Sequence[int] = (),
    interactions: Sequence[tuple[int, int]] = (),
    standardize: Sequence[str] | None = None,
    time_col: int = 0,
) -> SpatiotemporalFit:
    """Build a complete :class:`SpatiotemporalFit` from a DataFrame.

    The training-side workflow is::

        fit = fit_spatiotemporal(df, feature_cols=..., target_col=...)

    and the predict-side workflow re-uses the *same* ``fit`` to encode
    new data — concretely, by calling :func:`encode_time_column` with
    the stored ``time_min`` and applying the layers stored on the
    bundle.

    Args:
        df: Training DataFrame.
        feature_cols: Names of the input columns, in order. The first
            column (``time_col=0``) is treated as the time axis for
            seasonal features.
        target_col: Name of the target column.
        timetype: ``"int"`` or ``"datetime"`` — see
            :func:`encode_time_column`.
        freq: Optional unit string for ``datetime`` time columns.
        seasonality_periods: Periods (in time-unit) for seasonal
            features. Empty ⇒ no seasonal features.
        num_seasonal_harmonics: Harmonics per period; same length as
            ``seasonality_periods``.
        fourier_degrees: Per-input dyadic Fourier degrees. Length must
            match ``feature_cols``; use ``0`` to skip a column.
        interactions: Pairs of input-column indices for interaction
            features. Empty ⇒ no interactions.
        standardize: Optional subset of feature columns to standardize.
            ``None`` ⇒ standardize them all.
        time_col: Index of the time column inside ``feature_cols``;
            defaults to 0.

    Returns:
        Fitted :class:`SpatiotemporalFit` bundle.
    """
    if standardize is None:
        standardize = list(feature_cols)
    standardize_layer = fit_standardization(df, standardize)

    # If fourier_degrees was not provided, default to all-zero (no Fourier).
    if not fourier_degrees:
        fourier_degrees = [0] * len(feature_cols)
    if len(fourier_degrees) != len(feature_cols):
        raise ValueError(
            f"fourier_degrees must have length {len(feature_cols)}, "
            f"got {len(fourier_degrees)}"
        )
    fourier_layer = FourierFeatures(
        degrees=tuple(int(d) for d in fourier_degrees),
        rescale=True,
    )
    seasonal_layer = SeasonalFeatures(
        periods=tuple(float(p) for p in seasonality_periods),
        harmonics=tuple(int(h) for h in num_seasonal_harmonics),
        rescale=True,
    )
    interaction_layer = InteractionFeatures(
        pairs=tuple((int(a), int(b)) for a, b in interactions),
    )

    # Encode the time column once to capture (time_min, time_scale).
    _, time_min, time_scale = encode_time_column(
        df[feature_cols[time_col]], timetype=timetype, freq=freq
    )

    return SpatiotemporalFit(  # ty: ignore[invalid-return-type]
        standardize_layer=standardize_layer,
        fourier_layer=fourier_layer,
        seasonal_layer=seasonal_layer,
        interaction_layer=interaction_layer,
        time_min=time_min,
        time_scale=time_scale,
        feature_cols=tuple(feature_cols),
        target_col=target_col,
    )


# Re-export for convenience.
__all__ = [
    "SpatiotemporalFit",
    "encode_time_column",
    "fit_spatiotemporal",
    "fit_standardization",
]


# Suppress an unused-name warning on `Any`/`pd` if any downstream
# refactoring stops using them.
_ = Any
