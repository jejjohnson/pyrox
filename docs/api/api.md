# Estimator API

The `pyrox.api` subpackage exposes sklearn-style immutable estimator facades. Each estimator wraps a `pyrox.nn` model + the inference runners in `pyrox.inference` behind a one-call `fit`/`predict` ergonomic.

## Base contracts

::: pyrox.api.EstimatorBase

::: pyrox.api.FittedEstimator

## Bayesian Neural Field family

::: pyrox.api.BNFEstimator

::: pyrox.api.BNFEstimatorMLE

::: pyrox.api.BNFEstimatorVI

::: pyrox.api.FittedBNF
