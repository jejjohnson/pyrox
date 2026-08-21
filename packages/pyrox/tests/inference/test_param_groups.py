"""Tests for `pyrox.inference.param_group_optimizer`."""

from __future__ import annotations

import jax.numpy as jnp
import optax
import pytest
from pyrox.inference import param_group_optimizer


def _by_name(path, _leaf) -> str:
    return "latents" if "z" in str(path) else "globals"


@pytest.fixture()
def params():
    return {"z": jnp.ones(4), "lengthscale": jnp.ones(2)}


@pytest.fixture()
def grads():
    return {"z": jnp.ones(4), "lengthscale": jnp.ones(2)}


def test_groups_get_different_updates(params, grads):
    """A ``set_to_zero`` group must stay put while the other group moves."""
    opt = param_group_optimizer(
        {"latents": optax.sgd(1e-1), "globals": optax.set_to_zero()},
        _by_name,
    )
    state = opt.init(params)
    updates, _ = opt.update(grads, state, params)
    new = optax.apply_updates(params, updates)
    assert not jnp.allclose(new["z"], params["z"])
    assert jnp.array_equal(new["lengthscale"], params["lengthscale"])


def test_learning_rate_ratio_is_honoured(params, grads):
    """SGD updates from identical gradients must differ by the rate ratio."""
    opt = param_group_optimizer(
        {"latents": optax.sgd(1e-2), "globals": optax.sgd(1e-3)},
        _by_name,
    )
    state = opt.init(params)
    updates, _ = opt.update(grads, state, params)
    ratio = jnp.abs(updates["z"][0]) / jnp.abs(updates["lengthscale"][0])
    assert ratio == pytest.approx(10.0, rel=1e-6)


def test_single_group_is_a_noop_wrapper(params, grads):
    """One group must match calling that optimizer directly."""
    plain = optax.adam(1e-2)
    wrapped = param_group_optimizer({"all": optax.adam(1e-2)}, lambda p, v: "all")
    u_plain, _ = plain.update(grads, plain.init(params), params)
    u_wrapped, _ = wrapped.update(grads, wrapped.init(params), params)
    for k in params:
        assert jnp.allclose(u_plain[k], u_wrapped[k], atol=1e-12)


def test_unlabelled_parameter_fails_loudly(params):
    """A label absent from ``groups`` must raise, naming the label."""
    opt = param_group_optimizer(
        {"latents": optax.adam(1e-2)},  # no "globals" entry
        _by_name,
    )
    with pytest.raises(ValueError, match="globals"):
        opt.init(params)


def test_label_fn_sees_updates_not_params_during_update(params, grads):
    """optax evaluates a callable ``param_labels`` on the parameter tree at
    ``init`` but on the *update* tree at ``update``. This pins that fact,
    which is why ``label_fn`` must key off the path rather than leaf values.
    """
    seen: list[tuple[str, float]] = []

    def spy(path, leaf):
        seen.append((str(path), float(jnp.ravel(leaf)[0])))
        return "latents" if "z" in str(path) else "globals"

    opt = param_group_optimizer(
        {"latents": optax.sgd(1e-2), "globals": optax.sgd(1e-3)}, spy
    )
    params = {"z": jnp.full((2,), 5.0), "lengthscale": jnp.full((2,), 5.0)}
    grads = {"z": jnp.full((2,), -1.0), "lengthscale": jnp.full((2,), -1.0)}
    state = opt.init(params)
    at_init = dict(seen)
    seen.clear()
    opt.update(grads, state, params)
    at_update = dict(seen)

    # Same paths both times, but the values differ (params vs gradients) —
    # so a value-dependent label_fn would be unstable across the two.
    assert set(at_init) == set(at_update)
    assert any(at_init[k] != at_update[k] for k in at_init)


def test_path_based_labels_are_stable_across_updates(params, grads):
    """The documented contract: a path-keyed ``label_fn`` keeps each group
    on its own transform, so a frozen group stays frozen after init."""
    opt = param_group_optimizer(
        {"latents": optax.sgd(1e-1), "globals": optax.set_to_zero()}, _by_name
    )
    state = opt.init(params)
    current = params
    for _ in range(3):
        updates, state = opt.update(grads, state, current)
        current = optax.apply_updates(current, updates)
    assert jnp.array_equal(current["lengthscale"], params["lengthscale"])
    assert not jnp.allclose(current["z"], params["z"])
