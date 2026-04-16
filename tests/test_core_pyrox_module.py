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
from pyrox._core.pyrox_module import _Context


# --- _Context ---------------------------------------------------------------


def test_context_clears_on_outermost_exit():
    ctx = _Context()
    with ctx:
        ctx.set("a", 1)
        assert ctx.get("a") == 1
    assert ctx.get("a") is None


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
    assert ctx.get("a") is None


def test_context_inactive_set_is_noop():
    ctx = _Context()
    ctx.set("a", 1)
    assert ctx.get("a") is None


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


def test_fullname_falls_back_to_class_plus_id_without_pyrox_name():
    """Unnamed sibling instances of the same class must NOT collide."""

    class Anon(PyroxModule):
        @pyrox_method
        def __call__(self):
            return self.pyrox_sample("w", dist.Normal(0.0, 1.0))

    a = Anon()
    b = Anon()
    # Instance-qualified scope → distinct fullnames for distinct instances.
    assert a._pyrox_scope_name() != b._pyrox_scope_name()
    assert a._pyrox_fullname("w") != b._pyrox_fullname("w")
    assert a._pyrox_scope_name().startswith("Anon_")


def test_two_same_class_instances_register_distinct_sites_in_one_trace():
    """Two module instances of the same class inside one model should
    produce distinct sites — regression for PR #57 review: site-name
    collisions when multiple layers of the same class appear together.
    """

    class Layer(PyroxModule):
        @pyrox_method
        def __call__(self, x):
            return self.pyrox_sample("w", dist.Normal(0.0, 1.0)) + x

    a, b = Layer(), Layer()

    def model():
        y = a(jnp.array(0.0))
        return b(y)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model()
    site_names = list(tr)
    assert len(site_names) == 2
    assert site_names[0] != site_names[1]


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
