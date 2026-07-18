"""Tests for pyrox._core PyroxModule, context caching, and descriptors."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from pyrox._core import (
    PyroxModule,
    PyroxParam,
    PyroxSample,
    pyrox_method,
)
from pyrox._core.pyrox_module import _MISSING, _Context


# --- _Context ---------------------------------------------------------------


def test_context_clears_on_outermost_exit():
    ctx = _Context()
    with ctx:
        ctx.set("a", 1)
        assert ctx.get("a") == 1
    assert ctx.get("a") is _MISSING


def test_context_reentrant_preserves_cache_in_nested_scope():
    ctx = _Context()
    with ctx:
        ctx.set("a", 1)
        with ctx:
            assert ctx.get("a") == 1
            ctx.set("b", 2)
        # inner exit must not clear while outer still active
        assert ctx.get("a") == 1
        assert ctx.get("b") == 2
    assert ctx.get("a") is _MISSING


def test_context_inactive_set_is_noop():
    ctx = _Context()
    ctx.set("a", 1)
    assert ctx.get("a") is _MISSING


def test_context_caches_none_value_as_hit():
    """A legitimately cached ``None`` must count as a cache hit, not a miss."""
    ctx = _Context()
    with ctx:
        ctx.set("a", None)
        assert ctx.get("a") is None
        assert ctx.get("a") is not _MISSING


# --- Pattern B: PyroxModule -------------------------------------------------


class BayesianLinear(PyroxModule):
    pyrox_name = "BayesianLinear"
    in_features: int
    out_features: int

    @pyrox_method
    def __call__(self, x):
        W = self.pyrox_sample(
            "weight",
            dist.Normal(0, 1).expand([self.in_features, self.out_features]).to_event(2),
        )
        b = self.pyrox_param("bias", jnp.zeros(self.out_features))
        return x @ W + b


def test_pattern_b_registers_sample_and_param_sites():
    m = BayesianLinear(in_features=3, out_features=2)
    x = jnp.ones((4, 3))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        y = m(x)
    assert y.shape == (4, 2)
    assert "BayesianLinear.weight" in tr
    assert "BayesianLinear.bias" in tr
    assert tr["BayesianLinear.weight"]["type"] == "sample"
    assert tr["BayesianLinear.bias"]["type"] == "param"


def test_pyrox_method_deduplicates_repeated_sample_access():
    """Two reads of the same site inside one call must hit the cache once."""

    class TwiceReferenced(PyroxModule):
        pyrox_name = "TwiceReferenced"

        @pyrox_method
        def __call__(self):
            a = self.pyrox_sample("w", dist.Normal(0.0, 1.0))
            b = self.pyrox_sample("w", dist.Normal(0.0, 1.0))
            return a, b

    m = TwiceReferenced()
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        a, b = m()
    assert a == b
    # Exactly one site registered for two references.
    assert sum(k == "TwiceReferenced.w" for k in tr) == 1


def test_lookup_mode_param_cached_once():
    """``pyrox_param(name, None)`` (lookup mode) returns ``None`` outside a
    substitute handler, but the two references inside one call must still
    register the site exactly once — the cache must treat ``None`` as a hit.
    """

    class LookupParam(PyroxModule):
        pyrox_name = "LookupParam"

        @pyrox_method
        def __call__(self):
            a = self.pyrox_param("w", None)
            b = self.pyrox_param("w", None)
            return a, b

    m = LookupParam()
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        a, b = m()
    assert a is None and b is None
    assert sum(k == "LookupParam.w" for k in tr) == 1


def test_duplicate_pyrox_name_raises_under_trace():
    """Two instances sharing a ``pyrox_name`` collide on site names; under a
    trace NumPyro must reject the duplicate loudly (documented contract).
    """

    class Dup(PyroxModule):
        pyrox_name = "Dup"

        @pyrox_method
        def __call__(self):
            return self.pyrox_sample("w", dist.Normal(0.0, 1.0))

    a, b = Dup(), Dup()
    with (
        pytest.raises(AssertionError, match="unique names"),
        handlers.trace(),
        handlers.seed(rng_seed=0),
    ):
        a()
        b()


def test_dependent_prior_resolves_callable():
    class LocationScale(PyroxModule):
        pyrox_name = "LocationScale"

        @pyrox_method
        def __call__(self):
            loc = self.pyrox_sample("loc", dist.Normal(0.0, 1.0))
            scale = self.pyrox_sample(
                "scale",
                lambda self_: dist.LogNormal(loc, 0.1),
            )
            return loc, scale

    m = LocationScale()
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        _ = m()
    # Both sites present; scale's prior was resolved via callable.
    assert "LocationScale.loc" in tr
    assert "LocationScale.scale" in tr


def test_fullname_uses_pyrox_name_when_set():
    m = BayesianLinear(in_features=1, out_features=1)
    assert m._pyrox_scope_name() == "BayesianLinear"
    assert m._pyrox_fullname("w") == "BayesianLinear.w"


def test_fullname_falls_back_to_class_name_without_pyrox_name():
    """The unnamed fallback is the class name — deterministic, identical
    across instances, and stable across pytree reconstruction (#184
    Option C: the old ``{ClassName}_{id}`` fallback silently renamed sites
    whenever Equinox rebuilt the module).
    """

    class Anon(PyroxModule):
        @pyrox_method
        def __call__(self):
            return self.pyrox_sample("w", dist.Normal(0.0, 1.0))

    a = Anon()
    b = Anon()
    assert a._pyrox_scope_name() == b._pyrox_scope_name() == "Anon"
    assert a._pyrox_fullname("w") == "Anon.w"


def test_scope_name_stable_across_pytree_reconstruction():
    """Site names must not change when the module is rebuilt by
    flatten/unflatten (the reconstruction path used by eqx.tree_at,
    filter_jit-with-module-arg, and checkpoint loads). Regression for
    #184: the id-based fallback desynchronized MCMC draws from a later
    Predictive on the rebuilt copy.
    """

    class Anon(PyroxModule):
        @pyrox_method
        def __call__(self):
            return self.pyrox_sample("w", dist.Normal(0.0, 1.0))

    a = Anon()
    leaves, treedef = jax.tree.flatten(a)
    a2 = jax.tree.unflatten(treedef, leaves)
    with handlers.trace() as tr1, handlers.seed(rng_seed=0):
        a()
    with handlers.trace() as tr2, handlers.seed(rng_seed=0):
        a2()
    assert list(tr1) == list(tr2) == ["Anon.w"]


def test_unnamed_same_class_siblings_collide_loudly_under_trace():
    """With the class-name fallback, two *unnamed* instances of one class
    share a scope; a trace must reject the duplicate loudly. Users who
    stack several instances of a class give each a distinct pyrox_name
    (see the named-siblings test below).
    """

    class Layer(PyroxModule):
        @pyrox_method
        def __call__(self, x):
            return self.pyrox_sample("w", dist.Normal(0.0, 1.0)) + x

    a, b = Layer(), Layer()
    with (
        pytest.raises(AssertionError, match="unique names"),
        handlers.trace(),
        handlers.seed(rng_seed=0),
    ):
        b(a(jnp.array(0.0)))


def test_unnamed_param_only_siblings_collide_loudly_under_trace():
    """NumPyro's trace only asserts uniqueness for *sample* sites; duplicate
    param registrations are silently tolerated (last write wins), which
    would let two unnamed param-only siblings share weights under SVI/MAP.
    pyrox's duplicate-scope guard must reject this loudly. Regression for
    the #187 Codex P1 finding.
    """

    class ParamLayer(PyroxModule):
        @pyrox_method
        def __call__(self, x):
            return x + self.pyrox_param("b", jnp.zeros(2))

    a, b = ParamLayer(), ParamLayer()
    with (
        pytest.raises(ValueError, match="different module instance"),
        handlers.trace(),
        handlers.seed(rng_seed=0),
    ):
        b(a(jnp.zeros(2)))


def test_param_sibling_collision_detected_under_handlers_scope():
    """The duplicate-param guard must see site names as the trace records
    them: handlers.scope rewrites names (``enc/Class.b``) before recording,
    so a raw-fullname pre-check would miss collisions inside a scope.
    Regression for the #187 Codex scope-rewrite finding.
    """

    class ParamLayer(PyroxModule):
        @pyrox_method
        def __call__(self, x):
            return x + self.pyrox_param("b", jnp.zeros(2))

    a, b = ParamLayer(), ParamLayer()
    with (
        pytest.raises(ValueError, match="different module instance"),
        handlers.trace(),
        handlers.seed(rng_seed=0),
        handlers.scope(prefix="enc"),
    ):
        b(a(jnp.zeros(2)))


def test_param_ownership_does_not_leak_across_traces():
    """Ownership is per-trace: an instance that registered a site in an
    earlier (e.g. warm-up) trace must not bypass the guard in a later
    trace where a sibling registered the same name first. Regression for
    the #187 Codex stale-ownership finding.
    """

    class ParamLayer(PyroxModule):
        @pyrox_method
        def __call__(self, x):
            return x + self.pyrox_param("b", jnp.zeros(2))

    w1, w2 = ParamLayer(), ParamLayer()
    with handlers.trace(), handlers.seed(rng_seed=0):
        w1(jnp.zeros(2))  # warm-up trace: only w1
    with (
        pytest.raises(ValueError, match="different module instance"),
        handlers.trace(),
        handlers.seed(rng_seed=0),
    ):
        w1(w2(jnp.zeros(2)))  # w2 registers first; w1's stale claim must not pass

    # And a lone instance across repeated traces stays fine.
    f = ParamLayer()
    for _ in range(2):
        with handlers.trace() as tr, handlers.seed(rng_seed=0):
            f(jnp.zeros(2))
    assert list(tr) == ["ParamLayer.b"]


def test_same_instance_param_reuse_across_calls_stays_allowed():
    """Calling one module twice in one trace re-uses its own param site
    (legitimate weight sharing) — the duplicate-scope guard must only fire
    for *different* instances sharing a scope.
    """

    class ParamLayer(PyroxModule):
        @pyrox_method
        def __call__(self, x):
            return x + self.pyrox_param("b", jnp.zeros(2))

    c = ParamLayer()
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        c(jnp.zeros(2))
        c(jnp.ones(2))
    assert list(tr) == ["ParamLayer.b"]


def test_named_same_class_instances_register_distinct_sites_in_one_trace():
    """Two instances of one class with distinct per-instance pyrox_name
    fields produce distinct sites — the supported stacking pattern.
    """

    class Layer(PyroxModule):
        pyrox_name: str

        @pyrox_method
        def __call__(self, x):
            return self.pyrox_sample("w", dist.Normal(0.0, 1.0)) + x

    a, b = Layer(pyrox_name="layer0"), Layer(pyrox_name="layer1")

    def model():
        y = a(jnp.array(0.0))
        return b(y)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model()
    assert list(tr) == ["layer0.w", "layer1.w"]


def test_pyrox_sample_with_non_distribution_uses_deterministic():
    class PlainValue(PyroxModule):
        pyrox_name = "PlainValue"

        @pyrox_method
        def __call__(self):
            return self.pyrox_sample("v", jnp.array(3.14))

    m = PlainValue()
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        v = m()
    assert float(v) == pytest.approx(3.14)
    assert tr["PlainValue.v"]["type"] == "deterministic"


def test_pattern_b_jits_end_to_end():
    m = BayesianLinear(in_features=3, out_features=2)
    x = jnp.ones((4, 3))

    def model(x):
        return m(x)

    seeded = handlers.seed(model, rng_seed=0)
    jitted = jax.jit(seeded)
    y = jitted(x)
    assert y.shape == (4, 2)


# --- Descriptors ------------------------------------------------------------


def test_pyrox_param_defaults_are_unconstrained():
    p = PyroxParam(init_value=jnp.array(1.0))
    assert p.constraint is None
    assert p.event_dim is None


def test_pyrox_sample_is_frozen():
    s = PyroxSample(prior=dist.Normal(0.0, 1.0))
    with pytest.raises((AttributeError, TypeError)):
        s.prior = dist.Normal(1.0, 1.0)  # type: ignore[misc]


# --- Teardown ---------------------------------------------------------------


def test_teardown_drops_context_entry():
    m = BayesianLinear(in_features=1, out_features=1)
    _ = m._get_context()
    assert id(m) in PyroxModule._contexts
    m._teardown()
    assert id(m) not in PyroxModule._contexts
