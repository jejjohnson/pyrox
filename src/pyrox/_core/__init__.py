"""Core: Equinox-to-NumPyro bridge primitives.

Public surface:

- :class:`PyroxModule` — Equinox module with pyrox_param / pyrox_sample
- :class:`PyroxParam` — declarative parameter descriptor
- :class:`PyroxSample` — declarative sample descriptor
- :class:`Parameterized` — param registry with priors, guides, and modes
- :func:`pyrox_method` — decorator that activates the per-call context
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
