"""Deprecated aliases of `pyrox._core.call_context` and `call_contexts`.

The per-call cache is `PyroxModule` machinery, so it moved down to the pyrox
core, where pyrox-gp and pyrox-nn both reach it.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from contextlib import AbstractContextManager
from typing import Any

from pyrox._core import call_context, call_contexts


def _kernel_context(kernel: Any) -> AbstractContextManager[None]:
    """Deprecated: use `pyrox._core.call_context`."""
    warnings.warn(
        "pyrox_gp._context._kernel_context is deprecated; use "
        "pyrox._core.call_context.",
        DeprecationWarning,
        stacklevel=2,
    )
    return call_context(kernel)


def _kernel_contexts(kernels: Iterable[Any]) -> AbstractContextManager[None]:
    """Deprecated: use `pyrox._core.call_contexts`."""
    warnings.warn(
        "pyrox_gp._context._kernel_contexts is deprecated; use "
        "pyrox._core.call_contexts.",
        DeprecationWarning,
        stacklevel=2,
    )
    return call_contexts(kernels)
