"""Core: Equinox-to-NumPyro bridge primitives.

Public surface:

- `PyroxModule` — Equinox module with pyrox_param / pyrox_sample
- `PyroxParam` — declarative parameter descriptor
- `PyroxSample` — declarative sample descriptor
- `Parameterized` — param registry with priors, guides, and modes
- `pyrox_method` — decorator that activates the per-call context
"""

from pyrox._core.descriptors import PyroxParam, PyroxSample
from pyrox._core.parameterized import Parameterized
from pyrox._core.pyrox_module import PyroxModule, pyrox_method


__all__ = [
    "Parameterized",
    "PyroxModule",
    "PyroxParam",
    "PyroxSample",
    "pyrox_method",
]
