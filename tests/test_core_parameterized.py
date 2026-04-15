"""Tests for pyrox._core.Parameterized: registry, priors, guides, modes."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro import handlers

from pyrox._core import Parameterized, pyrox_method
from pyrox._core.parameterized import _State


class RBFKernel(Parameterized):
    @pyrox_method
    def __call__(self, X1, X2):
        v = self.get_param("variance")
        ls = self.get_param("lengthscale")
        sq = jnp.sum((X1[:, None] - X2[None, :]) ** 2 / ls**2, axis=-1)
        return v * jnp.exp(-0.5 * sq)

    def setup(self):
        self.register_param(
            "variance",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        self.register_param(
            "lengthscale",
            jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        self.set_prior("variance", dist.LogNormal(0.0, 1.0))


# --- setup() and registry --------------------------------------------------


def test_setup_is_invoked_on_construction():
    k = RBFKernel()
    state = k._state()
    assert set(state.params) == {"variance", "lengthscale"}
    assert state.params["variance"].prior is not None
    assert state.params["lengthscale"].prior is None
    assert state.mode == "model"


def test_register_param_before_set_prior_raises_keyerror():
    class Empty(Parameterized):
        pass

    k = Empty()
    with pytest.raises(KeyError, match="not registered"):
        k.set_prior("missing", dist.Normal(0.0, 1.0))


# --- mode switching --------------------------------------------------------


def test_model_mode_with_prior_registers_sample_site():
    k = RBFKernel()
    X = jnp.array([[0.0], [1.0]])
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        _ = k(X, X)
    assert tr["RBFKernel.variance"]["type"] == "sample"
    # lengthscale has no prior — stays as param
    assert tr["RBFKernel.lengthscale"]["type"] == "param"


def test_guide_mode_delta_uses_param_site():
    k = RBFKernel()
    k.set_mode("guide")
    X = jnp.array([[0.0], [1.0]])
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        _ = k(X, X)
    assert tr["RBFKernel.variance"]["type"] == "param"


def test_guide_mode_normal_adds_variational_params():
    class NK(Parameterized):
        @pyrox_method
        def __call__(self):
            return self.get_param("v")

        def setup(self):
            self.register_param("v", jnp.array(1.0))
            self.set_prior("v", dist.Normal(0.0, 1.0))
            self.autoguide("v", "normal")

    k = NK()
    k.set_mode("guide")
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        _ = k()
    assert "NK.v_loc" in tr
    assert "NK.v_scale" in tr
    assert tr["NK.v"]["type"] == "sample"


def test_autoguide_rejects_unknown_guide_type():
    k = RBFKernel()
    with pytest.raises(ValueError, match="guide_type must be"):
        k.autoguide("variance", "bogus")  # type: ignore[arg-type]


def test_set_mode_rejects_unknown_mode():
    k = RBFKernel()
    with pytest.raises(ValueError, match="mode must be"):
        k.set_mode("bogus")  # type: ignore[arg-type]


def test_mvn_guide_raises_not_implemented_at_get_param():
    class MK(Parameterized):
        @pyrox_method
        def __call__(self):
            return self.get_param("v")

        def setup(self):
            self.register_param("v", jnp.array(1.0))
            self.set_prior("v", dist.Normal(0.0, 1.0))
            self.autoguide("v", "mvn")

    k = MK()
    k.set_mode("guide")
    with pytest.raises(NotImplementedError), handlers.seed(rng_seed=0):
        _ = k()


# --- load_pyro_samples -----------------------------------------------------


def test_load_pyro_samples_touches_every_site():
    k = RBFKernel()
    with (
        handlers.trace() as tr,
        handlers.seed(rng_seed=0),
        k._get_context(),
    ):
        k.load_pyro_samples()
    assert "RBFKernel.variance" in tr
    assert "RBFKernel.lengthscale" in tr


# --- teardown --------------------------------------------------------------


def test_teardown_removes_registry_entry():
    k = RBFKernel()
    _ = k._state()
    assert id(k) in Parameterized._registry
    k._teardown()
    assert id(k) not in Parameterized._registry


# --- instance isolation ----------------------------------------------------


def test_distinct_instances_have_distinct_state():
    k1 = RBFKernel()
    k2 = RBFKernel()
    assert isinstance(k1._state(), _State)
    assert k1._state() is not k2._state()
    k1.set_mode("guide")
    assert k2._state().mode == "model"
