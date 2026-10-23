"""Inference runners for pyrox models — three layers of control.

**Layer 1 — Functional primitives** (roll your own training loop):

* `ensemble_init` — vmapped init of params + optimizer state
* `ensemble_loss` — ``filter_value_and_grad`` from a log-joint
* `ensemble_step` — one ensemble update step on a batch

**Layer 2 — NumPyro-like inference ops** (init/update/run, like SVI):

* `EnsembleMAP` — MAP/MLE runner via optax
* `EnsembleVI` — VI runner wrapping `numpyro.infer.SVI`

**Layer 3 — One-shot sugar:**

* `ensemble_map` — fit MAP/MLE in one call
* `ensemble_vi` — fit mean-field VI in one call
* `ensemble_predict` — vmap a predictive over the ensemble axis

**Post-hoc posteriors:**

* `last_layer_laplace` — GGN Laplace posterior over a parameter group of
  a fitted model, with a linearized predictive (`LastLayerLaplace`)

**Optimizer helpers:**

* `param_group_optimizer` — different optax optimizers (typically
  different learning rates) per labelled parameter group
"""

from pyrox.inference._ensemble import (
    EnsembleMAP,
    EnsembleResult,
    EnsembleState,
    EnsembleVI,
    ensemble_init,
    ensemble_loss,
    ensemble_map,
    ensemble_predict,
    ensemble_step,
    ensemble_vi,
)
from pyrox.inference._laplace import LastLayerLaplace, last_layer_laplace
from pyrox.inference._param_groups import param_group_optimizer


__all__ = [
    "EnsembleMAP",
    "EnsembleResult",
    "EnsembleState",
    "EnsembleVI",
    "LastLayerLaplace",
    "ensemble_init",
    "ensemble_loss",
    "ensemble_map",
    "ensemble_predict",
    "ensemble_step",
    "ensemble_vi",
    "last_layer_laplace",
    "param_group_optimizer",
]
