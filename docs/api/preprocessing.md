# Preprocessing API

The `pyrox.preprocessing` subpackage holds the **only** pandas-touching code in `pyrox`. Layers, models, and inference runners stay pandas-free; this module is the bridge between user-supplied DataFrames and the JAX-only `pyrox.nn` layers.

## `SpatiotemporalFit`

::: pyrox.preprocessing.SpatiotemporalFit

## `fit_spatiotemporal`

::: pyrox.preprocessing.fit_spatiotemporal

## `fit_standardization`

::: pyrox.preprocessing.fit_standardization

## `encode_time_column`

::: pyrox.preprocessing.encode_time_column
