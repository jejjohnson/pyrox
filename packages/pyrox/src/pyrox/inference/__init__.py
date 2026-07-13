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


__all__ = [
    "EnsembleMAP",
    "EnsembleResult",
    "EnsembleState",
    "EnsembleVI",
    "ensemble_init",
    "ensemble_loss",
    "ensemble_map",
    "ensemble_predict",
    "ensemble_step",
    "ensemble_vi",
]
