# Estimator API

The `pyrox_nn.api` subpackage exposes sklearn-style immutable estimator facades. Each estimator wraps a `pyrox_nn` model + the inference runners in `pyrox.inference` behind a one-call `fit`/`predict` ergonomic.

## Base contracts

::: pyrox_nn.api.EstimatorBase

::: pyrox_nn.api.FittedEstimator

## Bayesian Neural Field family

::: pyrox_nn.api.BNFEstimator

::: pyrox_nn.api.BNFEstimatorMLE

::: pyrox_nn.api.BNFEstimatorVI

::: pyrox_nn.api.FittedBNF
