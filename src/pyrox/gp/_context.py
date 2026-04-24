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
from collections.abc import Iterable, Iterator

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


@contextlib.contextmanager
def _kernel_contexts(kernels: Iterable[Kernel]) -> Iterator[None]:
    """Scope a group of kernel calls under one shared per-call context each.

    Opens one :func:`_kernel_context` per **unique kernel instance** (by
    :func:`id`) and holds them all open for the duration of the block.
    This is the right primitive for multi-output builders that loop over
    several latent kernels: if the caller reuses the same
    :class:`pyrox.PyroxModule`-based kernel instance across latents (for
    hyperparameter tying), we must not open and close a fresh per-call
    context on every iteration — that would clear the sample-site cache
    between iterations and cause duplicate-site registration under a
    NumPyro trace, or resample independent hyperparameter draws per
    iteration under :func:`numpyro.handlers.seed`.

    Distinct kernel instances get distinct contexts, which is fine — the
    per-call cache is per-instance.
    """
    seen: set[int] = set()
    with contextlib.ExitStack() as stack:
        for kernel in kernels:
            key = id(kernel)
            if key in seen:
                continue
            seen.add(key)
            stack.enter_context(_kernel_context(kernel))
        yield
