"""Inference runners for pyrox models — three layers of control.

**Layer 1 — Functional primitives** (roll your own training loop):

* :func:`ensemble_init` — vmapped init of params + optimizer state
* :func:`ensemble_loss` — ``filter_value_and_grad`` from a log-joint
* :func:`ensemble_step` — one ensemble update step on a batch

**Layer 2 — NumPyro-like inference ops** (init/update/run, like SVI):

* :class:`EnsembleMAP` — MAP/MLE runner via optax
* :class:`EnsembleVI` — VI runner wrapping :class:`numpyro.infer.SVI`

**Layer 3 — One-shot sugar:**

* :func:`ensemble_map` — fit MAP/MLE in one call
* :func:`ensemble_vi` — fit mean-field VI in one call
* :func:`ensemble_predict` — vmap a predictive over the ensemble axis
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
