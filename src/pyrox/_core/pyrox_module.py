"""PyroxModule and pyrox_method — Equinox-to-NumPyro bridge primitives.

Defines the base :class:`PyroxModule` that lets Equinox modules register
deterministic parameters and random sample sites with NumPyro, plus a
per-call ``_Context`` cache that prevents duplicate site registration
within a single probabilistic call.

Pattern B usage::

    class BayesianLinear(PyroxModule):
        in_features: int
        out_features: int

        @pyrox_method
        def __call__(self, x):
            W = self.pyrox_sample(
                "weight",
                dist.Normal(0, 1)
                    .expand([self.in_features, self.out_features])
                    .to_event(2),
            )
            b = self.pyrox_param("bias", jnp.zeros(self.out_features))
            return x @ W + b
"""

from __future__ import annotations

import contextlib
import functools
import weakref
from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx
import numpyro
import numpyro.distributions as dist


class _Context:
    """Per-call site cache with re-entrant scope depth tracking.

    Enter to start a probabilistic call; exit clears the cache when the
    outermost scope closes. Re-entry (nested ``pyrox_method`` calls on the
    same module) increments the depth so the inner scope does not clobber
    the outer cache.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._depth: int = 0

    def __enter__(self) -> _Context:
        self._depth += 1
        return self

    def __exit__(self, *exc: Any) -> None:
        self._depth -= 1
        if self._depth == 0:
            self._cache.clear()

    @property
    def active(self) -> bool:
        return self._depth > 0

    def get(self, name: str) -> Any | None:
        return self._cache.get(name)

    def set(self, name: str, value: Any) -> Any:
        if self._depth > 0:
            self._cache[name] = value
        return value


class PyroxModule(eqx.Module):
    """Equinox module with NumPyro site registration and per-call caching.

    Subclasses register deterministic parameters via :meth:`pyrox_param`
    and random variables via :meth:`pyrox_sample`. Wrap the method that
    drives registration (typically ``__call__``) with :func:`pyrox_method`
    so the per-call ``_Context`` is active for the duration of the call.

    Without the decorator the cache is inactive and duplicate references
    to the same site within one trace will hit NumPyro's uniqueness check.
    """

    _contexts: ClassVar[dict[int, _Context]] = {}

    def _get_context(self) -> _Context:
        key = id(self)
        ctx = PyroxModule._contexts.get(key)
        if ctx is None:
            ctx = _Context()
            PyroxModule._contexts[key] = ctx
            with contextlib.suppress(TypeError):
                weakref.finalize(self, PyroxModule._contexts.pop, key, None)
        return ctx

    def _pyrox_scope_name(self) -> str:
        """Per-instance scope used when building fully-qualified site names.

        Uses an explicit ``pyrox_name`` attribute if the subclass defines
        one (as a field or class variable); otherwise falls back to a
        ``{ClassName}_{id}`` tag so sibling instances of the same class
        register distinct sites within a single trace. The id-based
        fallback is stable within a Python process but not across runs —
        set ``pyrox_name`` explicitly for checkpoint-portable names.
        """
        name = getattr(self, "pyrox_name", None)
        if isinstance(name, str) and name:
            return name
        return f"{type(self).__name__}_{id(self):x}"

    def _pyrox_fullname(self, name: str) -> str:
        return f"{self._pyrox_scope_name()}.{name}"

    def pyrox_param(
        self,
        name: str,
        init_value: Any,
        *,
        constraint: Any = None,
        event_dim: int | None = None,
    ) -> Any:
        ctx = self._get_context()
        fullname = self._pyrox_fullname(name)
        if ctx.active:
            cached = ctx.get(fullname)
            if cached is not None:
                return cached
        kwargs: dict[str, Any] = {}
        if constraint is not None:
            kwargs["constraint"] = constraint
        if event_dim is not None:
            kwargs["event_dim"] = event_dim
        value = numpyro.param(fullname, init_value, **kwargs)
        return ctx.set(fullname, value)

    def pyrox_sample(self, name: str, prior: Any) -> Any:
        ctx = self._get_context()
        fullname = self._pyrox_fullname(name)
        if ctx.active:
            cached = ctx.get(fullname)
            if cached is not None:
                return cached
        resolved = (
            prior(self)
            if callable(prior) and not isinstance(prior, dist.Distribution)
            else prior
        )
        if isinstance(resolved, dist.Distribution):
            value = numpyro.sample(fullname, resolved)
        else:
            value = numpyro.deterministic(fullname, resolved)
        return ctx.set(fullname, value)

    def _teardown(self) -> None:
        """Remove this instance's cached context.

        Class-level registries are keyed by ``id(self)``. Equinox modules
        are typically weak-referenceable, so cleanup normally happens via
        :mod:`weakref.finalize`. Call this explicitly in environments where
        weak refs are not available or when you need deterministic cleanup.
        """
        PyroxModule._contexts.pop(id(self), None)


def pyrox_method(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a method so its body runs inside the module's per-call context.

    Apply to ``__call__`` (and any other method that registers pyrox sites)
    so the ``_Context`` cache is active for the duration of the call. The
    cache is cleared when the outermost decorated call returns.
    """

    @functools.wraps(fn)
    def wrapper(self: PyroxModule, *args: Any, **kwargs: Any) -> Any:
        with self._get_context():
            return fn(self, *args, **kwargs)

    return wrapper
