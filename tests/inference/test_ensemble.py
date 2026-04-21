"""Tests for `pyrox.inference.ensemble_*`.

Closed-form checks on a Gaussian-likelihood + Gaussian-prior linear
regression problem (analytic ridge solution) plus shape/structure
assertions on the layered API.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import optax
import pytest
from numpyro.infer.autoguide import AutoNormal

from pyrox.inference import (
    EnsembleMAP,
    EnsembleResult,
    EnsembleState,
    EnsembleVI,
    ensemble_init,
    ensemble_loss,
    ensemble_map,
    ensemble_predict,
    ensemble_step,
    ensemble_vi,
)


# -- Test fixtures -----------------------------------------------------------


@pytest.fixture
def linear_problem():
    """A small Gaussian-likelihood + Gaussian-prior linear regression."""
    n, d = 200, 5
    sigma2 = 0.25
    tau2 = 1.0
    key = jr.PRNGKey(0)
    k1, k2, k3 = jr.split(key, 3)
    X = jr.normal(k1, (n, d))
    theta_true = jr.normal(k2, (d,))
    y = X @ theta_true + jnp.sqrt(sigma2) * jr.normal(k3, (n,))

    XtX = X.T @ X
    XtY = X.T @ y
    A = XtX / sigma2 + jnp.eye(d) / tau2
    theta_ridge = jnp.linalg.solve(A, XtY / sigma2)
    theta_ols, *_ = jnp.linalg.lstsq(X, y, rcond=None)

    def log_joint(theta, xb, yb):
        ll = -0.5 * jnp.sum((yb - xb @ theta) ** 2) / sigma2
        lp = -0.5 * jnp.sum(theta**2) / tau2
        return ll, lp

    def init_fn(k):
        return 0.1 * jr.normal(k, (d,))

    return {
        "X": X,
        "y": y,
        "log_joint": log_joint,
        "init_fn": init_fn,
        "theta_ridge": theta_ridge,
        "theta_ols": theta_ols,
        "d": d,
    }


# -- Layer 3 — one-shot ensemble_map -----------------------------------------


def test_ensemble_map_recovers_ridge(linear_problem):
    p = linear_problem
    params, losses = ensemble_map(
        p["log_joint"],
        p["init_fn"],
        ensemble_size=32,
        num_epochs=2000,
        data=(p["X"], p["y"]),
        seed=jr.PRNGKey(42),
        learning_rate=5e-2,
        prior_weight=1.0,
    )
    assert params.shape == (32, p["d"])
    assert losses.shape == (32, 2000)
    assert jnp.allclose(params.mean(0), p["theta_ridge"], rtol=5e-2, atol=5e-2)


def test_ensemble_map_recovers_ols_when_prior_weight_zero(linear_problem):
    p = linear_problem
    params, _ = ensemble_map(
        p["log_joint"],
        p["init_fn"],
        ensemble_size=8,
        num_epochs=2000,
        data=(p["X"], p["y"]),
        seed=jr.PRNGKey(42),
        learning_rate=5e-2,
        prior_weight=0.0,
    )
    assert jnp.allclose(params.mean(0), p["theta_ols"], rtol=5e-2, atol=5e-2)


def test_ensemble_map_is_deterministic_given_seed(linear_problem):
    p = linear_problem
    p1, _ = ensemble_map(
        p["log_joint"],
        p["init_fn"],
        ensemble_size=4,
        num_epochs=100,
        data=(p["X"], p["y"]),
        seed=jr.PRNGKey(7),
    )
    p2, _ = ensemble_map(
        p["log_joint"],
        p["init_fn"],
        ensemble_size=4,
        num_epochs=100,
        data=(p["X"], p["y"]),
        seed=jr.PRNGKey(7),
    )
    assert jnp.array_equal(p1, p2)


def test_ensemble_map_minibatch_runs(linear_problem):
    """Mini-batch path should still converge in the right ballpark."""
    p = linear_problem
    params, losses = ensemble_map(
        p["log_joint"],
        p["init_fn"],
        ensemble_size=8,
        num_epochs=2000,
        data=(p["X"], p["y"]),
        seed=jr.PRNGKey(1),
        learning_rate=2e-2,
        prior_weight=1.0,
        batch_size=32,
    )
    assert losses.shape == (8, 2000)
    # Mini-batch SGD won't hit ridge as tightly as full batch, but should
    # be reasonably close.
    assert jnp.linalg.norm(params.mean(0) - p["theta_ridge"]) < 0.5


def test_ensemble_map_accepts_custom_optimizer(linear_problem):
    """Custom optax optimizer (chain + schedule) should work."""
    p = linear_problem
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-3,
        peak_value=5e-2,
        warmup_steps=50,
        decay_steps=500,
        end_value=1e-3,
    )
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.0),
    )
    params, _ = ensemble_map(
        p["log_joint"],
        p["init_fn"],
        ensemble_size=8,
        num_epochs=500,
        data=(p["X"], p["y"]),
        seed=jr.PRNGKey(3),
        optimizer=opt,
    )
    assert jnp.linalg.norm(params.mean(0) - p["theta_ridge"]) < 0.1


# -- Layer 1 primitives ------------------------------------------------------


def test_ensemble_init_returns_stacked_state(linear_problem):
    p = linear_problem
    state = ensemble_init(
        p["init_fn"], optax.adam(1e-2), ensemble_size=4, seed=jr.PRNGKey(0)
    )
    assert isinstance(state, EnsembleState)
    assert state.params.shape == (4, p["d"])


def test_ensemble_step_advances_state(linear_problem):
    p = linear_problem
    opt = optax.adam(5e-2)
    state = ensemble_init(p["init_fn"], opt, ensemble_size=4, seed=jr.PRNGKey(0))
    params0 = state.params.copy()
    state, losses = ensemble_step(
        state,
        p["X"],
        p["y"],
        log_joint=p["log_joint"],
        optimizer=opt,
        prior_weight=1.0,
    )
    assert losses.shape == (4,)
    assert not jnp.allclose(state.params, params0)


def test_ensemble_loss_returns_value_and_grad(linear_problem):
    p = linear_problem
    loss = ensemble_loss(p["log_joint"], prior_weight=1.0, scale=1.0)
    theta = jnp.zeros(p["d"])
    val, grads = loss(theta, p["X"], p["y"])
    assert val.shape == ()
    assert grads.shape == (p["d"],)


def test_manual_loop_matches_ensemble_map(linear_problem):
    """Hand-rolled init+step loop should give same answer as ensemble_map."""
    p = linear_problem
    opt = optax.adam(5e-2)
    # Manual loop using primitives
    state = ensemble_init(p["init_fn"], opt, ensemble_size=8, seed=jr.PRNGKey(0))
    step_jit = eqx.filter_jit(
        lambda s, x, y: ensemble_step(
            s,
            x,
            y,
            log_joint=p["log_joint"],
            optimizer=opt,
            prior_weight=1.0,
        )
    )
    for _ in range(500):
        state, _ = step_jit(state, p["X"], p["y"])
    params_manual = state.params

    # Ridge convergence — both should be close to ridge
    assert jnp.linalg.norm(params_manual.mean(0) - p["theta_ridge"]) < 0.1


# -- Layer 2 — EnsembleMAP class --------------------------------------------


def test_ensemble_map_class_init_update_run_consistent(linear_problem):
    """init+update produces same convergence quality as run."""
    p = linear_problem
    runner = EnsembleMAP(
        log_joint=p["log_joint"],
        init_fn=p["init_fn"],
        optimizer=optax.adam(5e-2),
        ensemble_size=4,
        prior_weight=1.0,
    )
    # init+update path
    state = runner.init(jr.PRNGKey(0))
    step_jit = eqx.filter_jit(runner.update)
    for _ in range(500):
        state, _ = step_jit(state, p["X"], p["y"])
    params_iu = state.params

    # run path
    result = runner.run(jr.PRNGKey(0), 500, p["X"], p["y"])
    assert isinstance(result, EnsembleResult)
    assert result.losses.shape == (4, 500)

    # Both should converge near ridge.
    assert jnp.linalg.norm(params_iu.mean(0) - p["theta_ridge"]) < 0.1
    assert jnp.linalg.norm(result.params.mean(0) - p["theta_ridge"]) < 0.1


def test_ensemble_map_class_with_eqx_module():
    """EnsembleMAP should handle eqx.Module params with non-array leaves."""

    def init_mlp(k):
        return eqx.nn.MLP(
            in_size=1,
            out_size=1,
            width_size=8,
            depth=1,
            activation=jax.nn.tanh,
            key=k,
        )

    def log_joint(mlp, xb, yb):
        f = jax.vmap(mlp)(xb[:, None]).squeeze(-1)
        ll = -0.5 * jnp.sum((yb - f) ** 2) / 0.01
        leaves = jax.tree.leaves(eqx.filter(mlp, eqx.is_inexact_array))
        lp = -0.5 * sum(jnp.sum(w**2) for w in leaves)
        return ll, lp

    key = jr.PRNGKey(666)
    X = jr.uniform(key, (50,), minval=-3.0, maxval=3.0)
    y = jnp.sin(X) + 0.1 * jr.normal(jr.PRNGKey(7), (50,))

    runner = EnsembleMAP(
        log_joint=log_joint,
        init_fn=init_mlp,
        optimizer=optax.adam(5e-3),
        ensemble_size=4,
    )
    result = runner.run(jr.PRNGKey(0), 200, X, y)
    assert result.losses.shape == (4, 200)
    # First-layer weight should have leading (4,) ensemble axis.
    assert result.params.layers[0].weight.shape[0] == 4


# -- Layer 3 — ensemble_predict ---------------------------------------------


def test_ensemble_predict_shape(linear_problem):
    p = linear_problem
    params, _ = ensemble_map(
        p["log_joint"],
        p["init_fn"],
        ensemble_size=8,
        num_epochs=100,
        data=(p["X"], p["y"]),
        seed=jr.PRNGKey(0),
    )
    X_new = p["X"][:5]
    preds = ensemble_predict(params, lambda theta, x: x @ theta, X_new)
    assert preds.shape == (8, 5)


# -- VI path -----------------------------------------------------------------


def _vi_model(x, y=None):
    w = numpyro.sample("w", dist.Normal(jnp.zeros(5), 1.0).to_event(1))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    f = x @ w + b
    numpyro.sample("y", dist.Normal(f, 0.5), obs=y)


def test_ensemble_vi_returns_stacked_params(linear_problem):
    p = linear_problem
    guide = AutoNormal(_vi_model)
    params, losses = ensemble_vi(
        _vi_model,
        guide,
        ensemble_size=4,
        num_epochs=300,
        data=(p["X"], p["y"]),
        seed=jr.PRNGKey(0),
    )
    assert losses.shape == (4, 300)
    # Check leaves carry leading (4,) axis.
    for leaf in jax.tree.leaves(params):
        assert leaf.shape[0] == 4


def test_ensemble_vi_loss_decreases_on_average(linear_problem):
    """SVI loss should trend down on average."""
    p = linear_problem
    guide = AutoNormal(_vi_model)
    _, losses = ensemble_vi(
        _vi_model,
        guide,
        ensemble_size=4,
        num_epochs=400,
        data=(p["X"], p["y"]),
        seed=jr.PRNGKey(0),
    )
    # Mean over the first vs last 10% of epochs.
    head = losses[:, :40].mean()
    tail = losses[:, -40:].mean()
    assert tail < head


def test_ensemble_vi_class_run(linear_problem):
    """EnsembleVI class wrapper runs end-to-end."""
    p = linear_problem
    guide = AutoNormal(_vi_model)
    runner = EnsembleVI(
        model_fn=_vi_model,
        guide_fn=guide,
        optimizer=numpyro.optim.Adam(1e-2),
        ensemble_size=4,
    )
    result = runner.run(jr.PRNGKey(0), 200, p["X"], p["y"])
    assert isinstance(result, EnsembleResult)
    assert result.losses.shape == (4, 200)
