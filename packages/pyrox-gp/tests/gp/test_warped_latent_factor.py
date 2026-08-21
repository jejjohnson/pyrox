"""Tests for warped collapsed latent-factor regression.

The load-bearing facts pinned here: the warped density is a proper
density (the log-det term is not cosmetic), an identity-init warp reduces
*exactly* to the unwarped likelihood, the decoder conjugacy survives the
warp, and the closed-form warp direction is numerically equivalent to the
bisection direction — the recommended ``Invert`` workaround is not a
silent change of model.

Tests that need ``gauss_flows`` (``RQSplineMarginal``,
``MixtureGaussianCDF``) skip when the ``flows`` extra is absent; the rest
run with ``flowjax`` alone.
"""

from __future__ import annotations

import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.bijections import Affine, Exp
from pyrox_gp import (
    RBF,
    LatentFactorGPPrior,
    collapsed_lfr_log_prob,
    decoder_posterior,
    lfr_predictive_moments,
    warp_to_base,
    warped_decoder_posterior,
    warped_lfr_log_prob,
)


jax.config.update("jax_enable_x64", True)


N, P, Q = 6, 4, 2


@pytest.fixture()
def problem():
    k_y, k_z = jr.split(jr.key(0))
    Y = jr.normal(k_y, (N, P))
    Z = jr.normal(k_z, (N, Q))
    return Y, Z, jnp.asarray(0.3)


def _affine_warp(p: int = P) -> Affine:
    """A non-identity closed-form warp with event shape ``(p,)``."""
    return Affine(
        loc=0.3 * jnp.arange(p, dtype=jnp.float64),
        scale=jnp.linspace(0.7, 1.4, p),
    )


def _perturbed_spline(p: int, key=None, scale: float = 0.3):
    """An ``RQSplineMarginal`` pushed off the identity."""
    gauss_flows = pytest.importorskip("gauss_flows")
    warp = gauss_flows.RQSplineMarginal(n_bins=6, shape=(p,), interval=4.0)
    leaves, treedef = jax.tree_util.tree_flatten(warp)
    key = jr.key(7) if key is None else key
    keys = jr.split(key, len(leaves))
    leaves = [
        leaf + scale * jr.normal(k, jnp.shape(leaf))
        if eqx.is_inexact_array(leaf)
        else leaf
        for leaf, k in zip(leaves, keys, strict=True)
    ]
    return jax.tree_util.tree_unflatten(treedef, leaves)


# --- proper density -----------------------------------------------------


def test_warped_density_integrates_to_one_and_log_det_is_load_bearing():
    """With ``N = P = Q = 1``, the warped density must integrate to one —
    and must *not* when the log-det term is dropped."""
    Z = jnp.asarray([[0.8]])
    s2 = jnp.asarray(0.4)
    warp = Affine(loc=jnp.asarray([0.5]), scale=jnp.asarray([0.7]))

    ys = jnp.linspace(-15.0, 15.0, 400_001)
    dy = ys[1] - ys[0]

    def log_p(y):
        return warped_lfr_log_prob(y[None, None], Z, s2, warp, jitter=0.0)

    log_vals = jax.vmap(log_p)(ys)
    integral = jnp.sum(jnp.exp(log_vals)) * dy
    assert jnp.abs(integral - 1.0) < 1e-6

    def log_p_no_det(y):
        ytil, _ = warp_to_base(warp, y[None, None])
        return collapsed_lfr_log_prob(ytil, Z, s2, jitter=0.0)

    no_det = jnp.sum(jnp.exp(jax.vmap(log_p_no_det)(ys))) * dy
    assert jnp.abs(no_det - 1.0) > 1e-3


def test_warped_density_integrates_to_one_spline():
    """Same proper-density check with a genuinely nonlinear warp."""
    pytest.importorskip("gauss_flows")
    Z = jnp.asarray([[0.8]])
    s2 = jnp.asarray(0.4)
    warp = _perturbed_spline(1)

    ys = jnp.linspace(-25.0, 25.0, 400_001)
    dy = ys[1] - ys[0]
    log_vals = jax.vmap(
        lambda y: warped_lfr_log_prob(y[None, None], Z, s2, warp, jitter=0.0)
    )(ys)
    integral = jnp.sum(jnp.exp(log_vals)) * dy
    assert jnp.abs(integral - 1.0) < 1e-6


# --- exact reduction ----------------------------------------------------


def test_identity_init_spline_reduces_exactly(problem):
    """``RQSplineMarginal`` at construction is the exact identity, so the
    warped likelihood must equal the plain one with zero tolerance."""
    gauss_flows = pytest.importorskip("gauss_flows")
    Y, Z, s2 = problem
    warp = gauss_flows.RQSplineMarginal(n_bins=8, shape=(P,), interval=4.0)
    warped = warped_lfr_log_prob(Y, Z, s2, warp)
    plain = collapsed_lfr_log_prob(Y, Z, s2)
    assert float(warped) == float(plain)


# --- conjugacy ----------------------------------------------------------


def test_warped_decoder_posterior_matches_blr_on_warped_data(problem):
    Y, Z, s2 = problem
    warp = _affine_warp()
    mean, row_cov = warped_decoder_posterior(Y, Z, s2, warp, jitter=0.0)
    Ytil, _ = warp_to_base(warp, Y)
    exp_mean, exp_cov = decoder_posterior(Ytil, Z, s2, jitter=0.0)
    psi = jnp.eye(Q) + Z.T @ Z / s2
    for j in range(P):
        blr = jnp.linalg.solve(psi, Z.T @ Ytil[:, j] / s2)
        assert jnp.allclose(mean[:, j], blr, atol=1e-10)
    assert jnp.allclose(row_cov, jnp.linalg.inv(psi), atol=1e-10)
    assert jnp.allclose(mean, exp_mean, atol=1e-12)
    assert jnp.allclose(row_cov, exp_cov, atol=1e-12)


# --- gradients ----------------------------------------------------------


def test_gradients_reach_z_noise_and_warp(problem):
    Y, Z, s2 = problem
    warp = _perturbed_spline(P)

    g_z, g_s2 = jax.grad(warped_lfr_log_prob, argnums=(1, 2))(Y, Z, s2, warp)
    assert jnp.all(jnp.isfinite(g_z))
    assert jnp.isfinite(g_s2)

    g_warp = eqx.filter_grad(lambda w: warped_lfr_log_prob(Y, Z, s2, w))(warp)
    leaves = jax.tree_util.tree_leaves(eqx.filter(g_warp, eqx.is_inexact_array))
    assert len(leaves) > 0
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_all_four_blocks_move_under_one_svi_step():
    """Z, noise, kernel hyperparameters, and warp params must all update
    under one SVI step of ``lfr_model`` with a warp on the prior."""
    pytest.importorskip("gauss_flows")
    import optax
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoDelta
    from pyrox_gp import lfr_model

    X = jr.uniform(jr.key(1), (N, 2))
    Y = jnp.abs(jr.normal(jr.key(2), (N, P))) + 0.1
    prior = LatentFactorGPPrior(
        kernels=tuple(RBF(pyrox_name=f"RBF_q{q}") for q in range(Q)),
        X=X,
        warp=_perturbed_spline(P, scale=0.05),
    )
    guide = AutoDelta(lfr_model)
    svi = SVI(lfr_model, guide, optax.adam(1e-2), loss=Trace_ELBO())
    state = svi.init(jr.key(0), X, Y, prior)
    params0 = svi.get_params(state)
    state, loss = svi.update(state, X, Y, prior)
    params1 = svi.get_params(state)
    assert jnp.isfinite(loss)

    moved = {
        name: any(
            not jnp.allclose(a, b)
            for a, b in zip(
                jax.tree_util.tree_leaves(params0[name]),
                jax.tree_util.tree_leaves(params1[name]),
                strict=True,
            )
        )
        for name in params0
    }
    assert moved["Z_T_auto_loc"], "latents did not move"
    assert moved["noise_auto_loc"], "noise did not move"
    assert moved["warp_params"], "warp did not move"
    kernel_keys = [k for k in params0 if k.startswith("RBF_q")]
    assert kernel_keys and any(moved[k] for k in kernel_keys), (
        "no kernel hyperparameter moved"
    )


# --- direction trap -----------------------------------------------------


def test_inverting_a_mixture_cdf_warp_changes_the_model(problem):
    """Wrapping a warp in ``Invert`` is not a free speedup.

    ``warped_lfr_log_prob`` applies ``warp.inverse``, which is
    ``M.inverse`` for ``M`` but ``M.transform`` for ``Invert(M)``. The two
    therefore map ``Y`` to different base values and define different
    likelihoods — pinned here so the docstring's warning cannot quietly
    become a recommendation again.
    """
    gauss_flows = pytest.importorskip("gauss_flows")
    from flowjax.bijections import Invert

    Y, Z, s2 = problem
    warp = gauss_flows.MixtureGaussianCDF(n_components=4, shape=(P,))
    # Push off the identity so the two directions genuinely differ.
    warp = jax.tree_util.tree_map(
        lambda leaf: leaf + 0.4 if eqx.is_inexact_array(leaf) else leaf, warp
    )
    direct = warped_lfr_log_prob(Y, Z, s2, warp)
    inverted = warped_lfr_log_prob(Y, Z, s2, Invert(warp))
    assert jnp.isfinite(direct) and jnp.isfinite(inverted)
    assert not jnp.allclose(direct, inverted, rtol=1e-3)


def test_bisection_inverse_agrees_with_its_own_closed_form_identity(problem):
    """The expensive ``inverse`` and the identity
    ``log|dG^-1/dy|(y) = -log|dG/dx|(G^-1(y))`` describe the same model, so
    they must agree to the bijection's inversion tolerance."""
    gauss_flows = pytest.importorskip("gauss_flows")
    Y, Z, s2 = problem
    warp = gauss_flows.MixtureGaussianCDF(n_components=4, shape=(P,))

    lp_bisection = warped_lfr_log_prob(Y, Z, s2, warp)
    Ytil, _ = warp_to_base(warp, Y)
    _, fwd_log_det = jax.vmap(warp.transform_and_log_det)(Ytil)
    lp_identity = collapsed_lfr_log_prob(Ytil, Z, s2) - jnp.sum(fwd_log_det)
    assert jnp.allclose(lp_bisection, lp_identity, atol=1e-5)


# --- validation ---------------------------------------------------------


def test_conditional_warp_rejected(problem):
    Y, _Z, _s2 = problem

    class _FakeConditionalWarp:
        shape = (P,)
        cond_shape = (3,)

    with pytest.raises(ValueError, match="Conditional warps"):
        warp_to_base(_FakeConditionalWarp(), Y)


def test_wrong_event_shape_rejected(problem):
    Y, Z, s2 = problem
    warp = _affine_warp(P + 1)
    with pytest.raises(ValueError, match=rf"\({P},\)"):
        warped_lfr_log_prob(Y, Z, s2, warp)


def test_unwarped_path_does_not_import_gauss_flows(monkeypatch, problem):
    """The ``warp=None`` path must work with ``gauss_flows`` absent."""
    Y, Z, s2 = problem
    monkeypatch.delitem(sys.modules, "gauss_flows", raising=False)
    monkeypatch.setitem(sys.modules, "gauss_flows", None)

    X = jr.uniform(jr.key(1), (N, 2))
    prior = LatentFactorGPPrior(
        kernels=tuple(RBF(pyrox_name=f"RBF_q{q}") for q in range(Q)), X=X
    )
    cond = prior.condition(Y, Z, s2)
    mean, var = cond.predict(X[:3])
    assert mean.shape == (3, P)
    assert bool((var > 0).all())


# --- predictive pushforward ---------------------------------------------


def _conditioned_with_warp(warp):
    X = jr.uniform(jr.key(1), (N, 2))
    Y_base = jr.normal(jr.key(2), (N, P))
    Y = jax.vmap(warp.transform)(Y_base)
    prior = LatentFactorGPPrior(
        kernels=tuple(RBF(pyrox_name=f"RBF_q{q}") for q in range(Q)),
        X=X,
        warp=warp,
    )
    Z = jr.normal(jr.key(3), (N, Q))
    return prior.condition(Y, Z, jnp.asarray(0.1)), X


def test_predictive_mean_is_not_warp_of_mean():
    """``E[G(f)] != G(E[f])`` for a nonlinear warp — pins the pushforward
    so nobody replaces the quadrature with a point evaluation."""
    warp = Exp(shape=(P,))
    cond, X = _conditioned_with_warp(warp)
    mean, var = cond.predict(X[:3])
    z_mean, z_var = cond.predict_latents(X[:3])
    w_mean, _ = lfr_predictive_moments(z_mean, z_var, cond.mu_W, cond.Sigma_W)
    naive = jax.vmap(warp.transform)(w_mean)
    assert bool((var > 0).all())
    assert not jnp.allclose(mean, naive, rtol=1e-3)
    # For exp, E[G(f)] > G(E[f]) strictly (Jensen).
    assert bool((mean > naive).all())


@pytest.mark.slow
def test_predictive_moments_vs_monte_carlo():
    """Observation-space quadrature moments vs Monte Carlo, 2% relative."""
    warp = Exp(shape=(P,))
    cond, X = _conditioned_with_warp(warp)
    X_new = X[:3]
    mean, var = cond.predict(X_new, quad_order=64)

    z_mean, z_var = cond.predict_latents(X_new)
    w_mean, w_var = lfr_predictive_moments(z_mean, z_var, cond.mu_W, cond.Sigma_W)
    samples = w_mean + jnp.sqrt(w_var) * jr.normal(jr.key(9), (400_000, 3, P))
    g = jnp.exp(samples)
    mc_mean = g.mean(axis=0)
    mc_var = g.var(axis=0)
    assert jnp.max(jnp.abs(mean - mc_mean) / mc_mean) < 0.02
    assert jnp.max(jnp.abs(var - mc_var) / mc_var) < 0.05


# --- scaling ------------------------------------------------------------


@pytest.mark.slow
def test_cost_stays_linear_in_p():
    gauss_flows = pytest.importorskip("gauss_flows")
    n, q = 9, 3
    Z = jr.normal(jr.key(0), (n, q))
    # Plain jax.jit fails here: flowjax spline pytrees carry string leaves
    # (include_endpoints), so the filtered variant is required.
    fn = eqx.filter_jit(warped_lfr_log_prob)

    def _best_time(p: int) -> float:
        warp = gauss_flows.RQSplineMarginal(n_bins=8, shape=(p,), interval=4.0)
        Y = jr.normal(jr.key(p), (n, p))
        fn(Y, Z, jnp.asarray(0.5), warp).block_until_ready()
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            fn(Y, Z, jnp.asarray(0.5), warp).block_until_ready()
            best = min(best, time.perf_counter() - t0)
        return best

    t_small, t_mid, t_big = (_best_time(p) for p in (10**3, 10**4, 5 * 10**4))
    assert t_mid / max(t_small, 1e-9) < 50.0
    assert t_big / max(t_mid, 1e-9) < 25.0  # 5x size; quadratic would be 25x


def test_warped_predictive_variance_survives_a_large_offset():
    """``E[G^2] - E[G]^2`` cancels catastrophically when the transformed
    values sit far from zero relative to their spread; the centered
    accumulation must keep the variance."""
    warp = Affine(loc=jnp.full((P,), 1e4), scale=jnp.ones(P))
    cond, X = _conditioned_with_warp(warp)
    _mean, var = cond.predict(X[:3])
    assert jnp.all(jnp.isfinite(var))
    assert bool((var > 0).all())
    # An affine warp is exact under the pushforward: variance is unchanged.
    # predict adds the latent nugget before propagating, so mirror that.
    z_mean, z_var = cond.predict_latents(X[:3])
    _, w_var = lfr_predictive_moments(
        z_mean, z_var + cond.prior.latent_noise, cond.mu_W, cond.Sigma_W
    )
    assert jnp.allclose(var, w_var, rtol=1e-6)
