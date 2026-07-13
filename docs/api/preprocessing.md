# Preprocessing API

The `pyrox_nn.preprocessing` subpackage holds the **only** pandas-touching code in `pyrox`. Layers, models, and inference runners stay pandas-free; this module is the bridge between user-supplied DataFrames and the JAX-only `pyrox_nn` layers.

## `SpatiotemporalFit`

::: pyrox_nn.preprocessing.SpatiotemporalFit

## `fit_spatiotemporal`

::: pyrox_nn.preprocessing.fit_spatiotemporal

## `fit_standardization`

::: pyrox_nn.preprocessing.fit_standardization

## `encode_time_column`

::: pyrox_nn.preprocessing.encode_time_column
