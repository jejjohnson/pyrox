"""PyroxModule and pyrox_method — Equinox-to-NumPyro bridge primitives.

Defines the base `PyroxModule` that lets Equinox modules register
deterministic parameters and random sample sites with NumPyro, plus a
per-call ``_Context`` cache that prevents duplicate site registration
within a single probabilistic call.

Pattern B usage:

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

import builtins
import contextlib
import functools
import weakref
from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx
import numpyro
import numpyro.distributions as dist


# Sentinel for "no cached value" so a legitimately cached ``None`` (e.g. the
# result of ``pyrox_param(name, None)`` lookup mode outside a substitute
# handler) is still treated as a cache hit rather than a miss.
_MISSING: Any = object()


def _visible_traces(fullname: str) -> list[tuple[dict[str, Any], str]]:
    """Predict which traces will record a param ``fullname``, and as what.

    Used by the duplicate-scope guard in `PyroxModule.pyrox_param`:
    NumPyro's trace only asserts name uniqueness for *sample* sites, so
    two modules silently sharing a scope would alias their *param* sites
    (shared weights under SVI/MAP) without a check. The guard cannot rely
    on inspecting the registration's side effects (``handlers.lift``
    serves cached duplicates without re-recording), so this walks
    NumPyro's handler stack the way ``apply_stack`` processes a param
    message — innermost handler first:

    * ``handlers.trace`` — collected; every reached trace records the
      message (with its **final** name, since the message dict is
      mutated in place before any trace postprocesses it).
    * ``handlers.scope`` — folds its prefix into the name
      (``outer/inner/name``), unless ``hide_types`` exempts params.
    * ``handlers.block`` — if its ``hide_fn`` hides this message, the
      walk stops: handlers outside the block (traces *and* scopes)
      never see it.

    Returns ``(trace_mapping, recorded_name)`` pairs — the recorded name
    is identical for every reached trace. Empty when no trace is active
    or the message is blocked before reaching one.
    """
    from numpyro.primitives import _PYRO_STACK

    name = fullname
    traces: list[dict[str, Any]] = []
    for handler in reversed(_PYRO_STACK):
        if isinstance(handler, numpyro.handlers.trace):
            tr = getattr(handler, "trace", None)
            if tr is not None:
                traces.append(tr)
        elif isinstance(handler, numpyro.handlers.scope):
            hide_types = getattr(handler, "hide_types", None) or ()
            prefix = getattr(handler, "prefix", None)
            if prefix and "param" not in hide_types:
                divider = getattr(handler, "divider", "/")
                name = f"{prefix}{divider}{name}"
        elif isinstance(handler, numpyro.handlers.block):
            hide_fn = getattr(handler, "hide_fn", None)
            if hide_fn is None:
                continue
            try:
                hidden = bool(hide_fn({"type": "param", "name": name}))
            except Exception:
                # A hide_fn needing more message context than we can
                # synthesize: assume it hides, which only *narrows* the
                # guard (a missed exotic collision, never a false alarm).
                hidden = True
            if hidden:
                break
    return [(tr, name) for tr in traces]


class _Context:
    """Per-call site cache with re-entrant scope depth tracking.

    Enter to start a probabilistic call; exit clears the cache when the
    outermost scope closes. Re-entry (nested ``pyrox_method`` calls on the
    same module) increments the depth so the inner scope does not clobber
    the outer cache.

    ``trace_owned`` tracks the param site names this instance registered
    **in the currently active trace** (as recorded by the trace, i.e.
    after handler rewriting such as ``handlers.scope`` prefixes). It lets
    the duplicate-scope guard distinguish "this instance re-uses its own
    param site across calls in one trace" (legitimate weight sharing)
    from "a *different* instance with the same scope registered this
    name" (silent parameter aliasing — an error). Ownership is keyed to
    the trace *object* via a weak reference, so it resets for every new
    trace — a stale claim from an earlier trace must not bypass the
    guard.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._depth: int = 0
        # ``builtins.set`` in the annotations: the bare name ``set`` resolves
        # to the ``_Context.set`` method inside this class body.
        self._trace_owned: list[tuple[Any, builtins.set[str]]] = []

    def trace_owned(self, tr: dict[str, Any]) -> builtins.set[str]:
        """Return the set of site names owned by this instance in ``tr``.

        One entry per live trace object (weakly referenced; dead entries are
        pruned), because nested traces are simultaneously active and each
        records every site.
        """
        self._trace_owned = [(r, s) for r, s in self._trace_owned if r() is not None]
        for ref, owned in self._trace_owned:
            if ref() is tr:
                return owned
        owned: builtins.set[str] = set()
        self._trace_owned.append((weakref.ref(tr), owned))
        return owned

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

    def get(self, name: str) -> Any:
        return self._cache.get(name, _MISSING)

    def set(self, name: str, value: Any) -> Any:
        if self._depth > 0:
            self._cache[name] = value
        return value


class PyroxModule(eqx.Module):
    """Equinox module with NumPyro site registration and per-call caching.

    Subclasses register deterministic parameters via `pyrox_param`
    and random variables via `pyrox_sample`. Wrap the method that
    drives registration (typically ``__call__``) with `pyrox_method`
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

        Uses an explicit ``pyrox_name`` attribute if the module defines one
        (as a field or class variable); otherwise falls back to the **class
        name**. Both are deterministic, so site names are stable across
        Python runs, checkpoint round-trips, and — critically — Equinox
        pytree reconstruction (``eqx.tree_at``, ``eqx.filter_jit`` with the
        module as an argument, ``jax.tree.unflatten``, deserialization).
        The previous ``{ClassName}_{id}`` fallback changed on every
        reconstruction, silently desynchronizing site names between e.g.
        an MCMC run and a later ``Predictive`` on a rebuilt copy.

        The scope must be **unique among the instances participating in a
        single trace**. With the class-name fallback, two *unnamed*
        instances of the same class collide: under ``handlers.trace`` this
        raises loudly — NumPyro's uniqueness assertion for sample sites,
        and pyrox's duplicate-scope guard in `pyrox_param` for param
        sites (which NumPyro would otherwise silently alias). Under a
        bare ``handlers.seed`` (no trace) there is no uniqueness check,
        so the collision is silent. When stacking several instances of
        one class in a model, give each a distinct ``pyrox_name`` (a
        per-instance field or constructor argument).
        """
        name = getattr(self, "pyrox_name", None)
        if isinstance(name, str) and name:
            return name
        return type(self).__name__

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
            if cached is not _MISSING:
                return cached
        kwargs: dict[str, Any] = {}
        if constraint is not None:
            kwargs["constraint"] = constraint
        if event_dim is not None:
            kwargs["event_dim"] = event_dim
        # NumPyro's trace asserts uniqueness for sample sites but silently
        # tolerates duplicate param registrations (last write wins). Without
        # a guard, two instances sharing a scope (e.g. unnamed siblings of
        # one class under the class-name fallback) would silently alias
        # their parameters under SVI/MAP. The guard pre-checks the
        # predicted recorded name against every trace this message will
        # actually reach (see `_visible_traces` for the handler-stack
        # semantics: nested traces, scope prefixes incl. hide_types, and
        # block visibility). Same-instance re-registration across calls in
        # one trace (weight sharing) stays allowed via the per-trace
        # ownership sets; ownership does not leak across traces.
        visible = _visible_traces(fullname)
        for tr, recorded in visible:
            if recorded in tr and recorded not in ctx.trace_owned(tr):
                raise ValueError(
                    f"param site {recorded!r} was already registered "
                    "in this trace by a different module instance. Two "
                    f"instances of {type(self).__name__} are sharing "
                    f"the scope {self._pyrox_scope_name()!r} — give "
                    "each a distinct pyrox_name."
                )
        value = numpyro.param(fullname, init_value, **kwargs)
        for tr, recorded in visible:
            ctx.trace_owned(tr).add(recorded)
        return ctx.set(fullname, value)

    def pyrox_sample(self, name: str, prior: Any) -> Any:
        ctx = self._get_context()
        fullname = self._pyrox_fullname(name)
        if ctx.active:
            cached = ctx.get(fullname)
            if cached is not _MISSING:
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
        `weakref.finalize`. Call this explicitly in environments where
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
