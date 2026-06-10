"""Pandas-side preprocessing helpers.

Isolated here so that `pyrox.nn` stays pandas-free. The output of
every helper is an immutable bundle of fitted layers + scalars, ready
to be plugged into the JAX-side stack.

Public surface:

* `SpatiotemporalFit` — immutable record of the fitted feature
  layers and time-encoding constants.
* `fit_standardization` — build a `Standardization` layer
  from a DataFrame's column statistics.
* `encode_time_column` — convert a pandas time column (``datetime``
  index or ``int`` column) into a JAX float array on a fixed scale.
* `fit_spatiotemporal` — one-call construction of a
  `SpatiotemporalFit` covering standardization + Fourier +
  seasonal + interaction layers from a DataFrame.
"""

from pyrox.preprocessing._pandas import (
    SpatiotemporalFit,
    encode_time_column,
    fit_spatiotemporal,
    fit_standardization,
)


__all__ = [
    "SpatiotemporalFit",
    "encode_time_column",
    "fit_spatiotemporal",
    "fit_standardization",
]
