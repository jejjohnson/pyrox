"""Shared kernel-context helper for GP entry points.

The :func:`_kernel_context` context manager scopes multiple kernel calls
under a single per-call pyrox context for kernels derived from
:class:`pyrox.PyroxModule` (Pattern B / C with priors). Without this
scoping, evaluating ``kernel(X1, X2)`` and ``kernel.diag(X3)``
back-to-back in the same NumPyro trace would either raise duplicate-site
errors (under tracing) or silently resample independent hyperparameter
draws for each call (under seed), decoupling the kernel matrices that
need to share hyperparameters (e.g.\\ ``K_zz``, ``K_xz``, ``K_xx_diag``
in the SVGP predictive).

The per-call ``_Context`` is reentrant, so nesting an inner
``_kernel_context`` inside an outer ``pyrox_method``-decorated call (or
inside another ``_kernel_context``) is safe and a no-op for the inner
scope. For pure :class:`equinox.Module` kernels (no ``_get_context``),
the context manager is itself a no-op.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from pyrox.gp._protocols import Kernel


@contextlib.contextmanager
def _kernel_context(kernel: Kernel) -> Iterator[None]:
    """Scope multiple kernel calls under a single per-call pyrox context."""
    ctx = getattr(kernel, "_get_context", None)
    if ctx is None:
        yield
        return
    with ctx():
        yield
