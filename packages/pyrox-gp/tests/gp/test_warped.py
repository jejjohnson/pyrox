"""Tests for the transformed-GP warped likelihood.

The load-bearing facts pinned here: an identity-init warp reproduces the
closed-form Gaussian expected log-likelihood exactly; Gauss-Hermite
convergence is warp-dependent (spectral for the smooth
``MixtureGaussianCDF``, plateaued and non-monotone for splines — so
"raise the order until it converges" is not a valid diagnostic); the warp
is a trainable child module (the direct contrast with the
``DistLikelihood`` static-field trap); and the predictive mean is
``E[G(f)]``, not ``G(E[f])``.
"""

from __future__ import annotations

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from flowjax.bijections import RationalQuadraticSpline
from gaussx import GaussHermiteIntegrator
from pyrox_gp import (
    RBF,
    FullRankGuide,
    SparseGPPrior,
    WarpedGaussianLikelihood,
    svgp_elbo,
    warped_predictive_moments,
)
from pyrox_gp._inference import _ell_numerical


jax.config.update("jax_enable_x64", True)


def _identity_spline() -> RationalQuadraticSpline:
    """Exact identity at construction."""
    return RationalQuadraticSpline(knots=8, interval=4.0)


def _perturbed(bijection, scale: float = 0.4, seed: int = 7):
    """Push a bijection's inexact leaves off their initial values."""
    leaves, treedef = jax.tree_util.tree_flatten(bijection)
    keys = jr.split(jr.key(seed), len(leaves))
    leaves = [
        leaf + scale * jr.normal(k, jnp.shape(leaf))
        if eqx.is_inexact_array(leaf)
        else leaf
        for leaf, k in zip(leaves, keys, strict=True)
    ]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _toy(M: int = 5, N: int = 10, seed: int = 0):
    Z = jnp.linspace(-2.0, 2.0, M).reshape(-1, 1)
    X = jnp.linspace(-3.0, 3.0, N).reshape(-1, 1)
    y = jnp.sin(X.squeeze(-1)) + 0.1 * jr.normal(jr.PRNGKey(seed), (N,))
    prior = SparseGPPrior(
        kernel=RBF(init_variance=1.0, init_lengthscale=0.5),
        Z=Z,
        jitter=1e-4,
    )
    return prior, X, y


# --- identity reduction -------------------------------------------------


def test_identity_warp_reduces_to_closed_form_gaussian_ell():
    """The quadrature ELL with an identity warp must equal the closed-form
    Gaussian expected log-likelihood to 1e-12."""
    y = jnp.asarray([1.4, -0.3, 0.8])
    f_loc = jnp.asarray([0.3, 0.1, -0.5])
    f_var = jnp.asarray([0.7, 0.2, 1.1])
    s2 = 0.1
    lik = WarpedGaussianLikelihood(warp=_identity_spline(), noise_var=jnp.asarray(s2))
    ell = _ell_numerical(lik, y, f_loc, f_var, GaussHermiteIntegrator(order=32))
    closed = jnp.sum(
        -0.5 * jnp.log(2.0 * jnp.pi * s2) - ((y - f_loc) ** 2 + f_var) / (2.0 * s2)
    )
    assert jnp.allclose(ell, closed, atol=1e-12)


# --- quadrature convergence is warp-dependent ---------------------------


def _ell_at_order(warp, order: int) -> float:
    y = jnp.asarray([1.4])
    f_loc = jnp.asarray([0.3])
    f_var = jnp.asarray([0.7])
    lik = WarpedGaussianLikelihood(warp=warp, noise_var=jnp.asarray(0.1))
    return float(
        _ell_numerical(lik, y, f_loc, f_var, GaussHermiteIntegrator(order=order))
    )


def test_quadrature_convergence_smooth_warp():
    """An analytic warp (``MixtureGaussianCDF``) must converge spectrally:
    successive deltas decreasing past order 32 and below 1e-9 by 128."""
    gauss_flows = pytest.importorskip("gauss_flows")
    warp = _perturbed(
        gauss_flows.MixtureGaussianCDF(n_components=4, shape=(1,)), scale=0.3
    )
    orders = [32, 64, 128, 256]
    vals = [_ell_at_order(warp, o) for o in orders]
    deltas = [abs(b - a) for a, b in itertools.pairwise(vals)]
    assert deltas[1] < deltas[0]
    assert deltas[2] < deltas[1]
    assert deltas[-1] < 1e-9


def test_quadrature_convergence_spline_plateaus():
    """A piecewise warp must *not* reach the analytic floor — this pins the
    docs' warning so nobody 'simplifies' it away later."""
    warp = _perturbed(_identity_spline(), scale=1.5)
    orders = [32, 64, 128, 256]
    vals = [_ell_at_order(warp, o) for o in orders]
    deltas = [abs(b - a) for a, b in itertools.pairwise(vals)]
    assert max(deltas) > 1e-7


# --- Monte Carlo cross-check --------------------------------------------


@pytest.mark.slow
def test_nonidentity_warp_ell_matches_monte_carlo():
    # A moderate perturbation: a violent spline warp pushes the quadrature
    # against its piecewise error floor (see the plateau test), which is
    # exactly what this MC cross-check must not conflate with a bug.
    warp = _perturbed(_identity_spline(), scale=0.3)
    y = jnp.asarray([1.4])
    f_loc = jnp.asarray([0.3])
    f_var = jnp.asarray([0.7])
    s2 = 0.1
    lik = WarpedGaussianLikelihood(warp=warp, noise_var=jnp.asarray(s2))
    ell = _ell_numerical(lik, y, f_loc, f_var, GaussHermiteIntegrator(order=64))

    n = 300_000
    f = f_loc + jnp.sqrt(f_var) * jr.normal(jr.key(1), (n, 1))
    log_p = jax.vmap(lambda fi: lik.log_prob(fi, y))(f)
    mc = log_p.mean()
    stderr = log_p.std() / jnp.sqrt(n)
    assert jnp.abs(ell - mc) < 2.0 * stderr


# --- gradients ----------------------------------------------------------


def test_gradients_reach_warp_and_noise():
    """The direct contrast with the ``DistLikelihood`` static-field trap:
    the warp is a child module, so its leaves receive gradients."""
    y = jnp.asarray([1.4, -0.3])
    f = jnp.asarray([0.3, 0.5])
    lik = WarpedGaussianLikelihood(
        warp=_perturbed(_identity_spline(), scale=0.3),
        noise_var=jnp.asarray(0.1),
    )
    grad = eqx.filter_grad(lambda m: -m.log_prob(f, y))(lik)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grad, eqx.is_inexact_array))
    assert len(leaves) > 1  # warp leaves plus noise_var
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert jnp.isfinite(grad.noise_var)


# --- shape handling -----------------------------------------------------


def test_log_prob_is_shape_agnostic():
    lik = WarpedGaussianLikelihood(
        warp=_perturbed(_identity_spline(), scale=0.5),
        noise_var=jnp.asarray(0.1),
    )
    y = jnp.asarray([1.4, -0.3, 0.8])
    f = jnp.asarray([0.3, 0.1, -0.5])

    per_point = sum(float(lik.log_prob(f[i][None], y[i][None])) for i in range(3))
    batched = float(lik.log_prob(f, y))
    assert batched == pytest.approx(per_point, abs=1e-10)

    stacked = jnp.stack([f, f + 0.1])
    y_stacked = jnp.stack([y, y])
    both = float(lik.log_prob(stacked, y_stacked))
    assert both == pytest.approx(
        float(lik.log_prob(f, y)) + float(lik.log_prob(f + 0.1, y)), abs=1e-10
    )


def test_invalid_event_shape_raises():
    from flowjax.bijections import Affine

    warp = Affine(loc=jnp.zeros(3), scale=jnp.ones(3))
    lik = WarpedGaussianLikelihood(warp=warp, noise_var=jnp.asarray(0.1))
    with pytest.raises(ValueError, match=r"\(\) or \(1,\)"):
        lik.log_prob(jnp.zeros(3), jnp.zeros(3))


def test_scalar_and_shape1_warps_both_work():
    y = jnp.asarray([0.4, -0.2])
    f = jnp.asarray([0.1, 0.3])
    lik1 = WarpedGaussianLikelihood(warp=_identity_spline(), noise_var=jnp.asarray(0.1))
    from flowjax.bijections import Affine

    lik2 = WarpedGaussianLikelihood(
        warp=Affine(loc=jnp.zeros(1), scale=jnp.ones(1)),
        noise_var=jnp.asarray(0.1),
    )
    assert jnp.allclose(lik1.log_prob(f, y), lik2.log_prob(f, y), atol=1e-12)


# --- end to end ---------------------------------------------------------


def test_one_svi_step_through_svgp_elbo_moves_warp():
    prior, X, y = _toy()
    guide = FullRankGuide.init(num_inducing=prior.num_inducing)
    lik = WarpedGaussianLikelihood(
        warp=_perturbed(_identity_spline(), scale=0.1),
        noise_var=jnp.asarray(0.1),
    )

    # Differentiate w.r.t. the likelihood only. (Reverse-mode through the
    # guide/prior blocks hits a pre-existing lax.while_loop limitation that
    # is independent of this likelihood — the Gaussian path fails the same
    # way — so the trainability being pinned here is the warp's.)
    def loss(m):
        return -svgp_elbo(
            prior, guide, m, X, y, integrator=GaussHermiteIntegrator(order=20)
        )

    value, grads = eqx.filter_value_and_grad(loss)(lik)
    assert jnp.isfinite(value)

    params = eqx.filter(lik, eqx.is_inexact_array)
    opt = optax.adam(1e-2)
    updates, _ = opt.update(
        eqx.filter(grads, eqx.is_inexact_array), opt.init(params), params
    )
    new_lik = eqx.apply_updates(lik, updates)
    warp_before = jax.tree_util.tree_leaves(eqx.filter(lik.warp, eqx.is_inexact_array))
    warp_after = jax.tree_util.tree_leaves(
        eqx.filter(new_lik.warp, eqx.is_inexact_array)
    )
    assert any(
        not jnp.allclose(a, b) for a, b in zip(warp_before, warp_after, strict=True)
    )


# --- predictive moments -------------------------------------------------


@pytest.mark.slow
def test_predictive_moments_vs_monte_carlo():
    warp = _perturbed(_identity_spline(), scale=0.8)
    lik = WarpedGaussianLikelihood(warp=warp, noise_var=jnp.asarray(0.1))
    f_loc = jnp.asarray([0.3, -0.4])
    f_var = jnp.asarray([0.7, 0.4])
    mean, var = warped_predictive_moments(lik, f_loc, f_var, order=64)

    n = 400_000
    f = f_loc + jnp.sqrt(f_var) * jr.normal(jr.key(2), (n, 2))
    g = jax.vmap(jax.vmap(warp.transform))(f)
    mc_mean = g.mean(axis=0)
    mc_var = lik.noise_var + g.var(axis=0)
    assert jnp.max(jnp.abs(mean - mc_mean) / jnp.abs(mc_mean)) < 0.02
    assert jnp.max(jnp.abs(var - mc_var) / mc_var) < 0.02


def test_predictive_mean_is_not_warp_of_mean():
    warp = _perturbed(_identity_spline(), scale=1.0)
    lik = WarpedGaussianLikelihood(warp=warp, noise_var=jnp.asarray(0.1))
    f_loc = jnp.asarray([0.3])
    f_var = jnp.asarray([0.9])
    mean, _ = warped_predictive_moments(lik, f_loc, f_var)
    naive = warp.transform(f_loc[0])
    assert not jnp.allclose(mean[0], naive, atol=1e-3)


def test_order_out_of_range_raises():
    lik = WarpedGaussianLikelihood(warp=_identity_spline(), noise_var=jnp.asarray(0.1))
    f = jnp.zeros(2)
    with pytest.raises(ValueError, match="order"):
        warped_predictive_moments(lik, f, jnp.ones(2), order=512)
    with pytest.raises(ValueError, match="order"):
        warped_predictive_moments(lik, f, jnp.ones(2), order=0)


# --- jit ----------------------------------------------------------------


def test_filter_jit_works_and_plain_jit_fails():
    """flowjax spline pytrees carry string leaves, so ``eqx.filter_jit`` is
    required and plain ``jax.jit`` fails loudly — worth pinning so the
    requirement stays discoverable."""
    lik = WarpedGaussianLikelihood(warp=_identity_spline(), noise_var=jnp.asarray(0.1))
    f = jnp.asarray([0.1, 0.2])
    y = jnp.asarray([0.0, 0.4])

    filtered = eqx.filter_jit(lambda m, a, b: m.log_prob(a, b))(lik, f, y)
    assert jnp.isfinite(filtered)

    with pytest.raises(TypeError, match="str"):
        jax.jit(lambda m, a, b: m.log_prob(a, b))(lik, f, y)


def test_predictive_variance_survives_a_large_offset():
    """A warp whose output sits far from zero relative to its spread: the
    raw ``E[G^2] - E[G]^2`` form cancels away the variance in float32, so
    the centered accumulation is what keeps it."""
    from flowjax.bijections import Affine

    lik = WarpedGaussianLikelihood(
        warp=Affine(loc=jnp.asarray(1e4), scale=jnp.asarray(1.0)),
        noise_var=jnp.asarray(0.1),
    )
    f_loc = jnp.asarray([0.3, -0.4])
    f_var = jnp.asarray([1.0, 4.0])
    mean, var = warped_predictive_moments(lik, f_loc, f_var, order=32)
    assert jnp.all(jnp.isfinite(var))
    # Affine warp with unit scale: variance passes through unchanged.
    assert jnp.allclose(var, f_var + lik.noise_var, rtol=1e-6)
    assert jnp.allclose(mean, f_loc + 1e4, rtol=1e-9)


def test_conditional_warp_is_rejected():
    """The likelihood never supplies a condition, so a conditional warp
    must be refused up front rather than failing inside ``transform``."""
    gauss_flows = pytest.importorskip("gauss_flows")

    warp = gauss_flows.Conditioner(
        key=jr.key(3),
        inner=gauss_flows.MixtureGaussianCDF(n_components=4, shape=(1,)),
        cond_shape=(2,),
        nn_width=8,
        nn_depth=2,
    )
    lik = WarpedGaussianLikelihood(warp=warp, noise_var=jnp.asarray(0.1))
    with pytest.raises(ValueError, match="Conditional warps"):
        lik.log_prob(jnp.zeros(2), jnp.zeros(2))
