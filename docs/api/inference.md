# Inference API

The `pyrox.inference` subpackage exposes ensemble-of-MAP and ensemble-of-VI runners as a layered surface — pick the level of control that fits your use case.

## Layer 1 — Functional primitives

Roll your own training loop on top of the vmapped state primitives.

::: pyrox.inference.ensemble_init

::: pyrox.inference.ensemble_loss

::: pyrox.inference.ensemble_step

## Layer 2 — NumPyro-like inference ops

`init` / `update` / `run` triplets that mirror `numpyro.infer.SVI`.

::: pyrox.inference.EnsembleMAP

::: pyrox.inference.EnsembleVI

## Layer 3 — One-shot sugar

::: pyrox.inference.ensemble_map

::: pyrox.inference.ensemble_vi

::: pyrox.inference.ensemble_predict

## Result containers

::: pyrox.inference.EnsembleState

::: pyrox.inference.EnsembleResult
