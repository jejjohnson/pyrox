"""Tests for ``pyrox.nn._ensemble`` rank-1 ensemble layers."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from pyrox._core.pyrox_module import PyroxModule
from pyrox.nn import DenseRank1


# --- forward shape & init ---------------------------------------------------


def test_dense_rank1_output_shape():
    layer = DenseRank1.init(
        jr.PRNGKey(0), in_features=4, out_features=2, ensemble_size=3
    )
    x = jnp.ones((5, 4))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert y.shape == (3, 5, 2)


def test_dense_rank1_no_bias_output_shape():
    layer = DenseRank1.init(
        jr.PRNGKey(0), in_features=4, out_features=2, ensemble_size=3, bias=False
    )
    x = jnp.ones((5, 4))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert y.shape == (3, 5, 2)


def test_dense_rank1_init_validates_positive_dims():
    with pytest.raises(ValueError, match="must all be > 0"):
        DenseRank1.init(jr.PRNGKey(0), in_features=0, out_features=2, ensemble_size=3)
    with pytest.raises(ValueError, match="must all be > 0"):
        DenseRank1.init(jr.PRNGKey(0), in_features=4, out_features=2, ensemble_size=0)


def test_dense_rank1_prior_scale_only_validated_in_bayesian_mode():
    """Deterministic configs may carry a sentinel ``prior_scale`` that's unused."""
    # bayesian=False ⇒ prior_scale=0.0 is fine (just unused).
    layer = DenseRank1.init(
        jr.PRNGKey(0),
        in_features=3,
        out_features=2,
        ensemble_size=2,
        bayesian=False,
        prior_scale=0.0,
    )
    assert layer.bayesian is False
    # bayesian=True ⇒ prior_scale must be positive.
    with pytest.raises(ValueError, match="prior_scale must be > 0"):
        DenseRank1.init(
            jr.PRNGKey(0),
            in_features=3,
            out_features=2,
            ensemble_size=2,
            bayesian=True,
            prior_scale=0.0,
        )


def test_dense_rank1_supports_arbitrary_batch_dims():
    """Forward accepts ``(*batch, D_in)`` and returns ``(M, *batch, D_out)``."""
    layer = DenseRank1.init(
        jr.PRNGKey(0), in_features=4, out_features=2, ensemble_size=3
    )
    x_3d = jnp.ones((6, 5, 4))  # e.g. (batch, time, D_in)
    with handlers.seed(rng_seed=0):
        y_3d = layer(x_3d)
    assert y_3d.shape == (3, 6, 5, 2)

    # Unbatched single example: (D_in,) → (M, D_out).
    x_1d = jnp.ones((4,))
    with handlers.seed(rng_seed=0):
        y_1d = layer(x_1d)
    assert y_1d.shape == (3, 2)


def test_dense_rank1_validates_init_arrays_shape():
    """Bypassing ``init`` with mis-sized arrays must fail loudly."""
    with pytest.raises(ValueError, match="W_init shape"):
        DenseRank1(
            in_features=4,
            out_features=2,
            ensemble_size=3,
            W_init=jnp.zeros((3, 2)),  # wrong: should be (4, 2)
            r_init=jnp.ones((3, 2)),
            s_init=jnp.ones((3, 4)),
        )


def test_dense_rank1_requires_init_arrays():
    with pytest.raises(ValueError, match=r"DenseRank1\.init"):
        DenseRank1(in_features=4, out_features=2, ensemble_size=3)


# --- deterministic mode (BatchEnsemble) ------------------------------------


def test_dense_rank1_deterministic_registers_param_sites():
    layer = DenseRank1.init(
        jr.PRNGKey(0),
        in_features=4,
        out_features=2,
        ensemble_size=3,
        pyrox_name="be",
    )
    x = jnp.ones((5, 4))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(x)
    for k in ("be.W", "be.r", "be.s", "be.b"):
        assert tr[k]["type"] == "param", f"{k} should be a param site"


def test_dense_rank1_deterministic_is_deterministic_across_seeds():
    """In BatchEnsemble mode the forward is purely deterministic."""
    layer = DenseRank1.init(
        jr.PRNGKey(0), in_features=4, out_features=2, ensemble_size=3
    )
    x = jr.normal(jr.PRNGKey(7), (5, 4))
    with handlers.seed(rng_seed=0):
        y1 = layer(x)
    with handlers.seed(rng_seed=1):
        y2 = layer(x)
    assert jnp.allclose(y1, y2)


def test_dense_rank1_members_are_distinct():
    """Random per-member init (init_scale > 0) gives distinct outputs."""
    layer = DenseRank1.init(
        jr.PRNGKey(0),
        in_features=4,
        out_features=2,
        ensemble_size=3,
        init_scale=0.5,
    )
    x = jr.normal(jr.PRNGKey(7), (5, 4))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    # No two members produce identical outputs.
    assert not jnp.allclose(y[0], y[1])
    assert not jnp.allclose(y[1], y[2])


def test_dense_rank1_zero_init_scale_collapses_members():
    """init_scale=0 ⇒ all r_i = s_i = 1 ⇒ every member is identical."""
    layer = DenseRank1.init(
        jr.PRNGKey(0),
        in_features=4,
        out_features=2,
        ensemble_size=3,
        init_scale=0.0,
    )
    x = jr.normal(jr.PRNGKey(7), (5, 4))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert jnp.allclose(y[0], y[1])
    assert jnp.allclose(y[1], y[2])


def test_dense_rank1_matches_explicit_per_member_kernel():
    """The efficient einsum path equals materialising W_i = (s_i⊗r_i)∘W."""
    layer = DenseRank1.init(
        jr.PRNGKey(0), in_features=3, out_features=2, ensemble_size=4
    )
    x = jr.normal(jr.PRNGKey(11), (6, 3))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    W = layer.W_init
    r = layer.r_init
    s = layer.s_init
    b = jnp.zeros((4, 2))
    expected = jnp.stack(
        [(x * s[i]) @ W * r[i] + b[i] for i in range(4)],
        axis=0,
    )
    assert jnp.allclose(y, expected, atol=1e-5)


# --- Bayesian mode (rank-1 BNN) --------------------------------------------


def test_dense_rank1_bayesian_registers_sample_sites():
    layer = DenseRank1.init(
        jr.PRNGKey(0),
        in_features=4,
        out_features=2,
        ensemble_size=3,
        bayesian=True,
        pyrox_name="rb",
    )
    x = jnp.ones((5, 4))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(x)
    assert tr["rb.W"]["type"] == "param"
    assert tr["rb.r"]["type"] == "sample"
    assert tr["rb.s"]["type"] == "sample"
    assert tr["rb.b"]["type"] == "param"


def test_dense_rank1_bayesian_stochastic_across_seeds():
    layer = DenseRank1.init(
        jr.PRNGKey(0),
        in_features=4,
        out_features=2,
        ensemble_size=3,
        bayesian=True,
    )
    x = jr.normal(jr.PRNGKey(7), (5, 4))
    with handlers.seed(rng_seed=0):
        y1 = layer(x)
    with handlers.seed(rng_seed=1):
        y2 = layer(x)
    assert not jnp.allclose(y1, y2)


def test_dense_rank1_bayesian_priors_centred_on_inits():
    """Each per-member prior is Normal(r_init[i], prior_scale)."""
    layer = DenseRank1.init(
        jr.PRNGKey(0),
        in_features=4,
        out_features=2,
        ensemble_size=3,
        bayesian=True,
        prior_scale=0.25,
        pyrox_name="rb",
    )
    x = jnp.ones((1, 4))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(x)
    r_fn = tr["rb.r"]["fn"]
    assert jnp.allclose(r_fn.base_dist.loc, layer.r_init)
    assert jnp.allclose(r_fn.base_dist.scale, 0.25)


def test_dense_rank1_bayesian_svi_elbo_decreases_with_observation_plate():
    """Canonical SVI pattern: forward outside the data plate, obs inside."""
    rng = jr.PRNGKey(0)
    x = jnp.linspace(-1.0, 1.0, 16)[:, None]
    y = 2.0 * x.squeeze(-1) + 0.5

    def model(x, y=None):
        layer = DenseRank1.init(
            jr.PRNGKey(1),
            in_features=1,
            out_features=1,
            ensemble_size=4,
            bayesian=True,
            pyrox_name="rb",
        )
        # Average over ensemble members to get a single regression output.
        f = layer(x).mean(axis=0).squeeze(-1)
        sigma = numpyro.param(
            "sigma", jnp.array(0.5), constraint=dist.constraints.positive
        )
        with numpyro.plate("data", x.shape[0]):
            numpyro.sample("obs", dist.Normal(f, sigma), obs=y)

    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(1e-2), Trace_ELBO())
    state = svi.init(rng, x, y)
    losses = []
    for _ in range(40):
        state, loss = svi.update(state, x, y)
        losses.append(float(loss))
    assert all(jnp.isfinite(jnp.asarray(losses)))
    assert losses[-1] < losses[0] - 1.0


def test_dense_rank1_is_pyrox_module():
    layer = DenseRank1.init(
        jr.PRNGKey(0), in_features=2, out_features=2, ensemble_size=2
    )
    assert isinstance(layer, PyroxModule)
