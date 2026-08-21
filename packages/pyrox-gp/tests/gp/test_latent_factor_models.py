"""Tests for the collapsed latent-factor model surface.

Covers `LatentFactorGPPrior` / `ConditionedLatentFactorGP` /
`lfr_factor` / `lfr_model` — the user-facing layer on top of
`pyrox_gp._latent_factor`. Mirrors the structure of
``test_oilmm_models.py``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import optax
import pytest
from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.initialization import init_to_median, init_to_sample
from pyrox_gp import (
    RBF,
    LatentFactorGPPrior,
    lfr_factor,
    lfr_model,
)


N, P, Q, D = 8, 5, 3, 2


def _init_latents_by_sampling(site):
    """Sample ``Z_T`` from its GP prior (breaks the rotational symmetry);
    init everything else deterministically."""
    return init_to_sample(site) if site["name"] == "Z_T" else init_to_median(site)


def _kernels(Q: int = Q) -> tuple[RBF, ...]:
    return tuple(RBF(pyrox_name=f"RBF_q{q}") for q in range(Q))


def _data(N: int = N, D: int = D, P: int = P, seed: int = 0):
    X = jr.uniform(jr.PRNGKey(seed), (N, D))
    Y = jr.normal(jr.PRNGKey(seed + 1), (N, P))
    return X, Y


@pytest.fixture()
def prior() -> LatentFactorGPPrior:
    X, _ = _data()
    return LatentFactorGPPrior(kernels=_kernels(), X=X)


# --- shapes -------------------------------------------------------------


def test_prior_shapes(prior):
    assert prior.num_latents == Q
    assert len(prior.latent_priors()) == Q
    chol = prior.latent_cholesky()
    assert chol.shape == (Q, N, N)
    assert jnp.all(jnp.isfinite(chol))


def test_condition_and_predict_shapes(prior):
    _, Y = _data()
    Z = jr.normal(jr.PRNGKey(2), (N, Q))
    cond = prior.condition(Y, Z, jnp.asarray(0.1))
    assert cond.mu_W.shape == (Q, P)
    assert cond.Sigma_W.shape == (Q, Q)
    X_new = jr.uniform(jr.PRNGKey(3), (4, D))
    mean, var = cond.predict(X_new)
    assert mean.shape == (4, P)
    assert var.shape == (4, P)


def test_positive_variance_and_noise_strictly_larger(prior):
    _, Y = _data()
    Z = jr.normal(jr.PRNGKey(2), (N, Q))
    cond = prior.condition(Y, Z, jnp.asarray(0.1))
    X_new = jr.uniform(jr.PRNGKey(3), (4, D))
    _, var = cond.predict(X_new)
    _, var_noise = cond.predict(X_new, include_noise=True)
    assert bool((var > 0.0).all())
    assert bool((var_noise > var).all())


def test_q_greater_than_p_is_allowed():
    """The concrete behavioural difference from ``OILMMKernel``, which
    rejects ``Q > P``."""
    X, _ = _data(P=2)
    Y = jr.normal(jr.PRNGKey(1), (N, 2))
    prior = LatentFactorGPPrior(kernels=_kernels(4), X=X)
    Z = jr.normal(jr.PRNGKey(2), (N, 4))
    cond = prior.condition(Y, Z, jnp.asarray(0.1))
    assert cond.mu_W.shape == (4, 2)


# --- validation ---------------------------------------------------------


def test_mismatched_y_rows_raise(prior):
    Y_bad = jnp.zeros((N + 1, P))
    Z = jnp.zeros((N, Q))
    with pytest.raises(ValueError, match=f"{N} rows"):
        prior.condition(Y_bad, Z, jnp.asarray(0.1))


def test_mismatched_z_shape_raises(prior):
    _, Y = _data()
    Z_bad = jnp.zeros((N, Q + 1))
    with pytest.raises(ValueError, match=rf"\({N}, {Q}\)"):
        prior.condition(Y, Z_bad, jnp.asarray(0.1))


def test_kernel_scope_collision_raises():
    X, _ = _data()
    with pytest.raises(ValueError, match="pyrox scope"):
        LatentFactorGPPrior(kernels=(RBF(), RBF()), X=X)


def test_warp_field_default_and_conditional_rejection():
    X, _ = _data()
    assert LatentFactorGPPrior(kernels=_kernels(), X=X, warp=None).warp is None

    class _FakeConditionalWarp:
        cond_shape = (2,)

    with pytest.raises(ValueError, match="Conditional warps"):
        LatentFactorGPPrior(kernels=_kernels(), X=X, warp=_FakeConditionalWarp())


# --- NumPyro integration ------------------------------------------------


def test_lfr_factor_tempering_default_and_override():
    _, Y = _data()
    Z = jr.normal(jr.PRNGKey(2), (N, Q))

    def model():
        Z_T = numpyro.sample(
            "Z_T",
            numpyro.distributions.Normal(jnp.zeros((Q, N)), 1.0).to_event(2),
        )
        lfr_factor(Y, Z_T.T, jnp.asarray(0.1))

    tr = handlers.trace(handlers.seed(model, jr.PRNGKey(0))).get_trace()
    assert tr["collapsed_lfr"]["scale"] == pytest.approx(Q / P)
    assert tr["Z_T"].get("scale") in (None, 1.0)

    def model_beta():
        lfr_factor(Y, Z, jnp.asarray(0.1), beta=0.5)

    tr = handlers.trace(handlers.seed(model_beta, jr.PRNGKey(0))).get_trace()
    assert tr["collapsed_lfr"]["scale"] == pytest.approx(0.5)


def test_model_traces_and_one_svi_step_runs(prior):
    X, Y = _data()
    tr = handlers.trace(handlers.seed(lfr_model, jr.PRNGKey(0))).get_trace(X, Y, prior)
    assert set(tr) >= {"Z_T", "noise", "collapsed_lfr"}
    assert tr["Z_T"]["value"].shape == (Q, N)

    guide = AutoDelta(
        lfr_model,
        init_loc_fn=partial(_init_latents_by_sampling),
    )
    svi = SVI(lfr_model, guide, optax.adam(1e-2), loss=Trace_ELBO())
    state = svi.init(jr.PRNGKey(0), X, Y, prior)
    state, loss = svi.update(state, X, Y, prior)
    assert jnp.isfinite(loss)


@pytest.mark.slow
def test_recovers_planted_low_rank_signal():
    """Fitted predictions must beat a predict-the-mean baseline on held-out
    points of a smooth planted low-rank signal."""
    n_all, p, q = 40, 6, 2
    key = jr.PRNGKey(0)
    X_all = jnp.linspace(0.0, 1.0, n_all)[:, None]
    Z_true = jnp.stack(
        [jnp.sin(4.0 * X_all[:, 0]), jnp.cos(7.0 * X_all[:, 0])], axis=-1
    )
    W_true = jr.normal(key, (q, p))
    Y_all = Z_true @ W_true + 0.05 * jr.normal(jr.PRNGKey(1), (n_all, p))

    train = jnp.arange(n_all) % 2 == 0
    X, Y = X_all[train], Y_all[train]
    X_test, Y_test = X_all[~train], Y_all[~train]

    prior = LatentFactorGPPrior(
        kernels=tuple(
            RBF(pyrox_name=f"RBF_q{i}", init_lengthscale=0.2) for i in range(q)
        ),
        X=X,
    )
    guide = AutoDelta(
        lfr_model,
        init_loc_fn=partial(_init_latents_by_sampling),
    )
    svi = SVI(lfr_model, guide, optax.adam(1e-2), loss=Trace_ELBO())
    result = svi.run(jr.PRNGKey(2), 3000, X, Y, prior, progress_bar=False)

    Z_map = result.params["Z_T_auto_loc"].T
    noise_var = result.params["noise_auto_loc"] ** 2
    cond = prior.condition(Y, Z_map, noise_var)
    mean, _ = cond.predict(X_test)

    rmse_model = jnp.sqrt(jnp.mean((mean - Y_test) ** 2))
    rmse_baseline = jnp.sqrt(jnp.mean((Y.mean(axis=0) - Y_test) ** 2))
    assert rmse_model < 0.5 * rmse_baseline


# --- diagnostics --------------------------------------------------------


def test_latent_total_correlation_separates_independent_from_mixed():
    pytest.importorskip("gauss_flows")
    from pyrox_gp import latent_total_correlation

    key = jr.PRNGKey(0)
    Z_indep = jr.normal(key, (4000, 4))
    mix = jr.normal(jr.PRNGKey(1), (4, 4)) + 2.0 * jnp.eye(4)
    Z_mixed = Z_indep @ mix
    tc_indep = latent_total_correlation(Z_indep)
    tc_mixed = latent_total_correlation(Z_mixed)
    assert tc_indep < 0.05
    assert tc_mixed > 0.2


# --- exports ------------------------------------------------------------


def test_public_names_exported():
    import pyrox_gp

    for name in (
        "LatentFactorGPPrior",
        "ConditionedLatentFactorGP",
        "lfr_factor",
        "lfr_model",
        "latent_total_correlation",
    ):
        assert name in pyrox_gp.__all__
        assert hasattr(pyrox_gp, name)


def test_gradients_reach_latents_and_noise(prior):
    X, Y = _data()

    def loss(Z, noise_var):
        return -numpyro.infer.util.log_density(
            handlers.substitute(lfr_model, {"Z_T": Z.T, "noise": jnp.sqrt(noise_var)}),
            (X, Y, prior),
            {},
            {},
        )[0]

    g_z, g_n = jax.grad(loss, argnums=(0, 1))(
        jr.normal(jr.PRNGKey(0), (N, Q)), jnp.asarray(0.1)
    )
    assert jnp.all(jnp.isfinite(g_z))
    assert jnp.isfinite(g_n)


def test_lfr_model_rejects_x_that_is_not_the_prior_inputs(prior):
    """``X`` contributes only shape/dtype — the latent covariance comes from
    ``prior.X`` — so a differently-shaped ``X`` must raise rather than fit
    against the wrong locations."""
    _, Y = _data()
    X_wrong = jr.uniform(jr.PRNGKey(9), (N, D + 1))
    with pytest.raises(ValueError, match="prior's training inputs"):
        handlers.seed(lfr_model, jr.PRNGKey(0))(X_wrong, Y, prior)


def test_conditioned_latent_gps_are_built_once(prior):
    """The per-latent training solves are cached on the conditioned object,
    so repeated prediction does not repeat the O(Q N^3) work."""
    _, Y = _data()
    Z = jr.normal(jr.PRNGKey(2), (N, Q))
    cond = prior.condition(Y, Z, jnp.asarray(0.1))
    assert len(cond.latents) == Q
    X_a = jr.uniform(jr.PRNGKey(3), (4, D))
    m1, v1 = cond.predict(X_a)
    m2, v2 = cond.predict(X_a)
    assert jnp.allclose(m1, m2) and jnp.allclose(v1, v2)


def test_predictions_use_substituted_kernel_parameters(prior):
    """Conditioning under a substitution handler must pick up the supplied
    lengthscales rather than the kernels' initial values."""
    _, Y = _data()
    Z = jr.normal(jr.PRNGKey(2), (N, Q))
    X_new = jr.uniform(jr.PRNGKey(3), (4, D))

    def _predict():
        return prior.condition(Y, Z, jnp.asarray(0.1)).predict(X_new)

    base_mean, _ = _predict()
    subs = {f"RBF_q{q}.lengthscale": jnp.asarray(5.0) for q in range(Q)}
    alt_mean, _ = handlers.substitute(_predict, subs)()
    assert not jnp.allclose(base_mean, alt_mean)


def test_empty_kernel_tuple_is_rejected():
    """Q = 0 constructs but fails later inside ``latent_cholesky``."""
    X, _ = _data()
    with pytest.raises(ValueError, match="at least one latent kernel"):
        LatentFactorGPPrior(kernels=(), X=X)


def test_latent_total_correlation_handles_a_single_factor():
    """``jnp.cov`` collapses to a scalar at Q = 1; the diagnostic must
    still return the mathematically expected zero."""
    pytest.importorskip("gauss_flows")
    from pyrox_gp import latent_total_correlation

    Z = jr.normal(jr.PRNGKey(0), (500, 1))
    assert jnp.allclose(latent_total_correlation(Z), 0.0, atol=1e-6)
