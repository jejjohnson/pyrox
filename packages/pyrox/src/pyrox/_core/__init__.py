"""Core: Equinox-to-NumPyro bridge primitives.

Public surface:

- `PyroxModule` — Equinox module with pyrox_param / pyrox_sample
- `PyroxParam` — declarative parameter descriptor
- `PyroxSample` — declarative sample descriptor
- `Parameterized` — param registry with priors, guides, and modes
- `pyrox_method` — decorator that activates the per-call context
- `call_context`, `call_contexts` — hold the per-call cache open across calls
- `freeze` — resolve a module's hyperparameters once (``module.frozen()``)
"""

from pyrox._core.context import call_context, call_contexts, freeze
from pyrox._core.descriptors import PyroxParam, PyroxSample
from pyrox._core.parameterized import Parameterized
from pyrox._core.pyrox_module import PyroxModule, pyrox_method


__all__ = [
    "Parameterized",
    "PyroxModule",
    "PyroxParam",
    "PyroxSample",
    "call_context",
    "call_contexts",
    "freeze",
    "pyrox_method",
]
