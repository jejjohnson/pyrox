"""Tests for pure-JAX extreme value primitives.

Exercises round-trip (cdf ∘ icdf = id), the Gumbel limit of GEV, grad
safety (no NaNs in gradients under sensible parameters), and vmap
compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtremax import (
    frechet_cdf,
    frechet_icdf,
    frechet_log_prob,
    frechet_mean,
    frechet_return_level,
    gev_cdf,
    gev_icdf,
    gev_log_prob,
    gev_mean,
    gev_return_level,
    gpd_cdf,
    gpd_icdf,
    gpd_log_prob,
    gpd_mean,
    gumbel_cdf,
    gumbel_icdf,
    gumbel_log_prob,
    gumbel_mean,
    gumbel_return_level,
    weibull_cdf,
    weibull_icdf,
    weibull_log_prob,
    weibull_mean,
    weibull_return_level,
)


Q_GRID = jnp.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])


class TestGEV:
    def test_cdf_icdf_round_trip_positive_shape(self):
        x = gev_icdf(Q_GRID, loc=0.0, scale=1.0, shape=0.2)
        q = gev_cdf(x, loc=0.0, scale=1.0, shape=0.2)
        assert jnp.allclose(q, Q_GRID, atol=1e-5)

    def test_cdf_icdf_round_trip_negative_shape(self):
        x = gev_icdf(Q_GRID, loc=0.0, scale=1.0, shape=-0.2)
        q = gev_cdf(x, loc=0.0, scale=1.0, shape=-0.2)
        assert jnp.allclose(q, Q_GRID, atol=1e-5)

    def test_gumbel_limit_shape_zero(self):
        """GEV at shape=0 must match the Gumbel closed form exactly."""
        x = jnp.linspace(-2.0, 3.0, 20)
        gev_lp = gev_log_prob(x, 0.0, 1.0, 0.0)
        gumbel_lp = gumbel_log_prob(x, 0.0, 1.0)
        assert jnp.allclose(gev_lp, gumbel_lp, atol=1e-6)

        gev_q = gev_icdf(Q_GRID, 0.0, 1.0, 0.0)
        gumbel_q = gumbel_icdf(Q_GRID, 0.0, 1.0)
        assert jnp.allclose(gev_q, gumbel_q, atol=1e-6)

    def test_mean_finite_for_small_shape(self):
        m = gev_mean(0.0, 1.0, 0.2)
        assert jnp.isfinite(m)

    def test_mean_infinite_when_shape_ge_one(self):
        m = gev_mean(0.0, 1.0, 1.0)
        assert jnp.isposinf(m)

    def test_return_level_matches_icdf(self):
        period = jnp.array([2.0, 10.0, 100.0])
        rl = gev_return_level(period, 0.0, 1.0, 0.1)
        expected = gev_icdf(1.0 - 1.0 / period, 0.0, 1.0, 0.1)
        assert jnp.allclose(rl, expected)

    def test_grad_log_prob_finite(self):
        """Gradient w.r.t. all three params at an in-support point is finite."""
        grad_fn = jax.grad(gev_log_prob, argnums=(1, 2, 3))
        g = grad_fn(jnp.array(1.5), 0.0, 1.0, 0.1)
        assert all(jnp.all(jnp.isfinite(gi)) for gi in g)

    def test_vmap_over_data(self):
        x = jnp.linspace(0.1, 3.0, 64)
        vfn = jax.vmap(gev_log_prob, in_axes=(0, None, None, None))
        lp = vfn(x, 0.0, 1.0, 0.1)
        assert lp.shape == x.shape
        assert jnp.all(jnp.isfinite(lp))

    def test_jit_compiles(self):
        fn = jax.jit(gev_log_prob)
        lp = fn(jnp.array(1.5), 0.0, 1.0, 0.1)
        assert jnp.isfinite(lp)


class TestGumbel:
    def test_cdf_icdf_round_trip(self):
        x = gumbel_icdf(Q_GRID, 0.0, 1.0)
        q = gumbel_cdf(x, 0.0, 1.0)
        assert jnp.allclose(q, Q_GRID, atol=1e-5)

    def test_log_prob_normalizes(self):
        """∫ pdf dx ≈ 1 via trapezoid over a wide range."""
        x = jnp.linspace(-10.0, 20.0, 4096)
        pdf = jnp.exp(gumbel_log_prob(x, 0.0, 1.0))
        total = jnp.trapezoid(pdf, x)
        assert jnp.allclose(total, 1.0, atol=1e-3)

    def test_mean_closed_form(self):
        m = gumbel_mean(0.0, 1.0)
        euler_gamma = 0.5772156649015329
        assert jnp.allclose(m, euler_gamma, atol=1e-6)

    def test_return_level(self):
        rl = gumbel_return_level(jnp.array([2.0, 100.0]), 0.0, 1.0)
        expected = gumbel_icdf(jnp.array([0.5, 0.99]), 0.0, 1.0)
        assert jnp.allclose(rl, expected)

    def test_grad(self):
        grad_fn = jax.grad(gumbel_log_prob, argnums=(1, 2))
        g_loc, g_scale = grad_fn(jnp.array(1.0), 0.0, 1.0)
        assert jnp.isfinite(g_loc) and jnp.isfinite(g_scale)


class TestGPD:
    def test_cdf_icdf_round_trip_positive_shape(self):
        x = gpd_icdf(Q_GRID, scale=1.0, shape=0.2)
        q = gpd_cdf(x, scale=1.0, shape=0.2)
        assert jnp.allclose(q, Q_GRID, atol=1e-5)

    def test_cdf_icdf_round_trip_negative_shape(self):
        x = gpd_icdf(Q_GRID, scale=1.0, shape=-0.2)
        q = gpd_cdf(x, scale=1.0, shape=-0.2)
        assert jnp.allclose(q, Q_GRID, atol=1e-5)

    def test_exponential_limit(self):
        """GPD at shape=0 is the exponential."""
        x = jnp.linspace(0.01, 5.0, 20)
        gpd_lp = gpd_log_prob(x, 1.0, 0.0)
        # exp(x; scale=1) has log pdf -log(1) - x/1 = -x
        expected = -x
        assert jnp.allclose(gpd_lp, expected, atol=1e-5)

    def test_mean_formula(self):
        m = gpd_mean(scale=2.0, shape=0.3)
        assert jnp.allclose(m, 2.0 / (1.0 - 0.3), atol=1e-6)

    def test_mean_infinite_when_shape_ge_one(self):
        m = gpd_mean(1.0, 1.0)
        assert jnp.isposinf(m)

    def test_log_prob_outside_support_is_neg_inf(self):
        lp = gpd_log_prob(jnp.array([-0.5, -1.0]), 1.0, 0.2)
        assert jnp.all(jnp.isneginf(lp))

    def test_cdf_at_upper_bound_is_one(self):
        """For shape < 0, CDF at x = -σ/ξ must be exactly 1 (not 0)."""
        scale, shape = 1.0, -0.3
        upper = -scale / shape
        # at the endpoint and just past it
        x = jnp.array([upper, upper + 1e-6])
        c = gpd_cdf(x, scale, shape)
        assert jnp.allclose(c, 1.0)

    def test_log_prob_continuous_across_shape_zero(self):
        """Regression: for tiny negative shape, the primitive previously
        set upper_bound = -scale/1 (due to _safe_shape substitution), so
        log_prob returned -inf everywhere. Verify continuity across ξ=0.
        """
        x = jnp.linspace(0.1, 3.0, 10)
        lp_tiny_neg = gpd_log_prob(x, 1.0, -1e-8)
        lp_zero = gpd_log_prob(x, 1.0, 0.0)
        lp_tiny_pos = gpd_log_prob(x, 1.0, 1e-8)
        assert jnp.all(jnp.isfinite(lp_tiny_neg))
        assert jnp.allclose(lp_tiny_neg, lp_zero, atol=1e-5)
        assert jnp.allclose(lp_tiny_pos, lp_zero, atol=1e-5)

    def test_cdf_approaches_one_near_upper_bound(self):
        """CDF should be continuous at the upper endpoint, approaching 1."""
        scale, shape = 1.0, -0.3
        upper = float(-scale / shape)
        x_inside = jnp.linspace(upper - 1e-3, upper, 8)
        c = gpd_cdf(x_inside, scale, shape)
        assert float(c[-1]) == pytest.approx(1.0, abs=1e-5)
        assert jnp.all(jnp.diff(c) >= -1e-7)  # monotone non-decreasing


class TestFrechet:
    def test_delegation_matches_gev(self):
        x = jnp.linspace(1.1, 5.0, 10)
        lp_frechet = frechet_log_prob(x, 0.0, 1.0, 0.2)
        lp_gev = gev_log_prob(x, 0.0, 1.0, 0.2)
        assert jnp.allclose(lp_frechet, lp_gev)

    def test_round_trip(self):
        x = frechet_icdf(Q_GRID, 0.0, 1.0, 0.3)
        q = frechet_cdf(x, 0.0, 1.0, 0.3)
        assert jnp.allclose(q, Q_GRID, atol=1e-5)

    def test_mean_infinite_for_heavy_tail(self):
        m = frechet_mean(0.0, 1.0, 1.5)
        assert jnp.isposinf(m)

    def test_return_level(self):
        rl = frechet_return_level(10.0, 0.0, 1.0, 0.2)
        assert jnp.isfinite(rl)


class TestWeibull:
    def test_delegation_matches_gev(self):
        x = jnp.linspace(-5.0, 0.5, 10)  # below upper bound for shape=-0.2
        lp_weibull = weibull_log_prob(x, 0.0, 1.0, -0.2)
        lp_gev = gev_log_prob(x, 0.0, 1.0, -0.2)
        assert jnp.allclose(lp_weibull, lp_gev)

    def test_round_trip(self):
        x = weibull_icdf(Q_GRID, 0.0, 1.0, -0.2)
        q = weibull_cdf(x, 0.0, 1.0, -0.2)
        assert jnp.allclose(q, Q_GRID, atol=1e-5)

    def test_mean_finite(self):
        m = weibull_mean(0.0, 1.0, -0.3)
        assert jnp.isfinite(m)

    def test_return_level(self):
        rl = weibull_return_level(10.0, 0.0, 1.0, -0.2)
        assert jnp.isfinite(rl)


class TestClassPrimitiveParity:
    """Each class method should produce identical output to its primitive."""

    @pytest.fixture
    def x(self):
        return jnp.linspace(0.5, 3.0, 20)

    def test_gevd_log_prob_parity(self, x):
        from xtremax import GeneralizedExtremeValueDistribution

        d = GeneralizedExtremeValueDistribution(0.0, 1.0, 0.2)
        assert jnp.allclose(d.log_prob(x), gev_log_prob(x, 0.0, 1.0, 0.2))

    def test_gumbel_log_prob_parity(self, x):
        from xtremax import GumbelType1GEVD

        d = GumbelType1GEVD(0.0, 1.0)
        assert jnp.allclose(d.log_prob(x), gumbel_log_prob(x, 0.0, 1.0))

    def test_gpd_log_prob_parity(self, x):
        from xtremax import GeneralizedParetoDistribution

        d = GeneralizedParetoDistribution(scale=1.0, shape=0.2)
        assert jnp.allclose(d.log_prob(x), gpd_log_prob(x, 1.0, 0.2))

    def test_frechet_log_prob_parity(self, x):
        from xtremax import FrechetType2GEVD

        d = FrechetType2GEVD(0.0, 1.0, 0.2)
        assert jnp.allclose(d.log_prob(x), frechet_log_prob(x, 0.0, 1.0, 0.2))

    def test_weibull_log_prob_parity(self):
        from xtremax import WeibullType3GEVD

        d = WeibullType3GEVD(0.0, 1.0, -0.2)
        x = jnp.linspace(-3.0, 2.0, 20)
        assert jnp.allclose(d.log_prob(x), weibull_log_prob(x, 0.0, 1.0, -0.2))
