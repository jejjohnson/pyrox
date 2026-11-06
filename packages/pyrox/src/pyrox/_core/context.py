"""One per-call site cache across several calls of a `PyroxModule`, and freezing.

`call_context` holds a module's per-call cache open, so evaluating it several
times inside the block reads each sample site once. `freeze` resolves a module's
hyperparameters once, for code that takes a plain array-valued object.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator
from typing import Any


@contextlib.contextmanager
def call_context(module: Any) -> Iterator[None]:
    """Hold one per-call site cache open for ``module``.

    Without it, two calls of a module with priors in one model collide under
    `numpyro.handlers.trace` (a duplicate sample site) or resample under
    `numpyro.handlers.seed` (two independent draws of one hyperparameter). The
    cache is reentrant, so nesting this inside a `pyrox_method` call or another
    `call_context` is a no-op for the inner block. An object that has no
    per-call cache, such as a plain kernellib kernel, makes it a no-op too.

    Args:
        module: A `PyroxModule`, or any other object.

    Examples:
        >>> import numpyro.distributions as dist
        >>> from numpyro import handlers
        >>> from pyrox._core import PyroxModule, call_context, pyrox_method
        >>> class Scale(PyroxModule):
        ...     pyrox_name: str = "Scale"
        ...
        ...     @pyrox_method
        ...     def __call__(self):
        ...         return self.pyrox_sample("s", dist.LogNormal(0.0, 1.0))
        >>> scale = Scale()
        >>> with handlers.seed(rng_seed=0), call_context(scale):
        ...     first, second = scale(), scale()
        >>> bool(first == second)
        True
    """
    opener = getattr(module, "_get_context", None)
    if opener is None:
        yield
        return
    with opener():
        yield


@contextlib.contextmanager
def call_contexts(modules: Iterable[Any]) -> Iterator[None]:
    """Hold one per-call site cache open for each distinct module.

    Modules are compared by identity. A module listed twice (tied
    hyperparameters across the latents of a multi-output model) gets one cache,
    because closing and reopening it between iterations would clear the sites
    read so far: a duplicate site under `numpyro.handlers.trace`, an independent
    draw per iteration under `numpyro.handlers.seed`.

    Args:
        modules: The modules to hold open together.
    """
    seen: set[int] = set()
    with contextlib.ExitStack() as stack:
        for module in modules:
            key = id(module)
            if key in seen:
                continue
            seen.add(key)
            stack.enter_context(call_context(module))
        yield


def freeze(module: Any) -> Any:
    """``module.frozen()`` under one `call_context`, or ``module`` if it cannot.

    One context covers every hyperparameter, so a prior registers one sample
    site. Call it inside an enclosing `call_context` to share that draw with
    other evaluations of the module. An object with no ``frozen`` method, or one
    whose ``frozen`` raises `NotImplementedError` because it has no plain
    counterpart, comes back unchanged.

    Args:
        module: A module with a ``frozen`` method, or any other object.

    Returns:
        The frozen module, or ``module`` itself.

    Examples:
        >>> from pyrox._core import freeze
        >>> plain = object()
        >>> freeze(plain) is plain
        True
    """
    frozen = getattr(module, "frozen", None)
    if frozen is None:
        return module
    try:
        with call_context(module):
            return frozen()
    except NotImplementedError:
        return module
