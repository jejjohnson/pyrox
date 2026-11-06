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

**Optimizer helpers:**

* `param_group_optimizer` — different optax optimizers (typically
  different learning rates) per labelled parameter group
"""

import importlib
import warnings
from typing import Any

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
from pyrox.inference._param_groups import param_group_optimizer


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
    "param_group_optimizer",
]


def __getattr__(name: str) -> Any:
    """Serve what moved to pyrox-nn, with a `DeprecationWarning`.

    Looked up at access time, so importing the core never loads pyrox-nn and the
    core does not depend on it.
    """
    if name not in {"LastLayerLaplace", "last_layer_laplace"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"pyrox.inference.{name} is deprecated; use `from pyrox_nn import {name}`.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        pyrox_nn = importlib.import_module("pyrox_nn")
    except ImportError as error:
        raise ImportError(
            f"pyrox.inference.{name} moved to pyrox-nn; install pyrox-nn and use "
            f"`from pyrox_nn import {name}`."
        ) from error
    return getattr(pyrox_nn, name)
