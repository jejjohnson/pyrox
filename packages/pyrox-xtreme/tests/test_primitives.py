"""Tests for pure-JAX extreme value primitives.

Exercises round-trip (cdf ∘ icdf = id), the Gumbel limit of GEV, grad
safety (no NaNs in gradients under sensible parameters), and vmap
compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import scipy.stats as st

from xtremax import (
    frechet_cdf,
    frechet_icdf,
    frechet_log_prob,
    frechet_mean,
    frechet_return_level,
    gev_cdf,
    gev_icdf,
    gev_log_prob,
    gev_log_survival,
    gev_mean,
    gev_return_level,
    gev_survival,
    gpd_cdf,
    gpd_icdf,
    gpd_log_prob,
    gpd_log_survival,
    gpd_mean,
    gpd_return_level,
    gpd_survival,
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

# Shape values straddling the Gumbel/exponential limit, including the float32
# danger band 1e-7 < |ξ| < 1e-4 where the old kernels lost precision.
XI_BAND = [-0.5, -0.05, -1e-4, -1e-6, 0.0, 1e-6, 1e-4, 0.05, 0.2, 0.9]


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

    def test_gpd_survival_delegates_to_primitive(self, x):
        """Class survival / cumulative-hazard must match the stable primitives
        (incl. the small-ξ band), not the old threshold-based branches."""
        from xtremax import GeneralizedParetoDistribution

        for xi in (0.2, 5e-8, -0.2):
            d = GeneralizedParetoDistribution(scale=1.0, concentration=xi)
            assert jnp.allclose(d.survival_function(x), gpd_survival(x, 1.0, xi))
            assert jnp.allclose(d.exceedance_probability(x), gpd_survival(x, 1.0, xi))
            assert jnp.allclose(
                d.cumulative_hazard_rate(x), -gpd_log_survival(x, 1.0, xi)
            )

    def test_frechet_log_prob_parity(self, x):
        from xtremax import FrechetType2GEVD

        d = FrechetType2GEVD(0.0, 1.0, 0.2)
        assert jnp.allclose(d.log_prob(x), frechet_log_prob(x, 0.0, 1.0, 0.2))

    def test_weibull_log_prob_parity(self):
        from xtremax import WeibullType3GEVD

        d = WeibullType3GEVD(0.0, 1.0, -0.2)
        x = jnp.linspace(-3.0, 2.0, 20)
        assert jnp.allclose(d.log_prob(x), weibull_log_prob(x, 0.0, 1.0, -0.2))

    def test_gevd_survival_delegates_to_primitive(self, x):
        """Class survival / log-survival must match the stable primitives (incl.
        the small-ξ band), not the old threshold-based branches."""
        from xtremax import GeneralizedExtremeValueDistribution

        for xi in (0.2, 5e-8, -0.2):
            d = GeneralizedExtremeValueDistribution(0.0, 1.0, xi)
            assert jnp.allclose(d.survival_function(x), gev_survival(x, 0.0, 1.0, xi))
            assert jnp.allclose(
                d.log_survival_function(x),
                gev_log_survival(x, 0.0, 1.0, xi),
                equal_nan=True,
            )


class TestGEVOracle:
    """Compare GEV primitives against scipy.stats.genextreme (c = -ξ).

    Guards against systematic parameterization errors that the circular
    class-vs-primitive parity tests cannot catch, and pins the float32
    accuracy of the small-ξ band (the reformulated log1p/expm1 kernels).
    """

    @pytest.mark.parametrize("xi", XI_BAND)
    def test_cdf_logpdf_match_scipy(self, xi):
        x = jnp.linspace(-1.5, 4.0, 15)
        ref = st.genextreme(-xi, loc=0.0, scale=1.0)
        assert jnp.allclose(gev_cdf(x, 0.0, 1.0, xi), ref.cdf(x), atol=1e-5)
        # rtol on the log-density: near the support endpoint |logpdf| is huge
        # (density ≈ 0), so an absolute tolerance is meaningless there.
        assert jnp.allclose(
            gev_log_prob(x, 0.0, 1.0, xi), ref.logpdf(x), atol=1e-4, rtol=1e-4
        )

    @pytest.mark.parametrize("xi", XI_BAND)
    def test_icdf_matches_scipy(self, xi):
        ref = st.genextreme(-xi, loc=0.0, scale=1.0)
        assert jnp.allclose(gev_icdf(Q_GRID, 0.0, 1.0, xi), ref.ppf(Q_GRID), atol=1e-4)

    def test_small_shape_band_continuous(self):
        """cdf/log_prob/mean must not jump across the Gumbel threshold."""
        for xi in [1e-6, -1e-6, 1e-4, -1e-4]:
            assert float(gev_cdf(1.0, 0.0, 1.0, xi)) == pytest.approx(
                float(gev_cdf(1.0, 0.0, 1.0, 0.0)), abs=1e-4
            )
            assert float(gev_log_prob(1.0, 0.0, 1.0, xi)) == pytest.approx(
                float(gev_log_prob(1.0, 0.0, 1.0, 0.0)), abs=1e-4
            )

    @pytest.mark.parametrize("xi", [2e-7, 1e-6, 1e-4, 1e-3, 1e-2, 0.2, -0.3])
    def test_mean_matches_scipy_small_shape(self, xi):
        """Regression: gev_mean returned garbage / negative-scale values in the
        small-ξ band from the Γ(1-ξ)-1 cancellation (issue #44)."""
        got = float(gev_mean(0.0, 1.0, xi))
        ref = float(st.genextreme(-xi).mean())
        assert got == pytest.approx(ref, abs=1e-3)


class TestGEVGradientStability:
    """Gradient regressions for the reformulated GEV kernels (issue #45)."""

    @pytest.mark.parametrize(
        "fn, x, xi",
        [
            (gev_log_prob, -90.0, -0.1),
            (gev_cdf, -120.0, -0.1),
            (gev_return_level, 100.0, 0.2),
        ],
    )
    def test_grad_finite_in_weibull_tail(self, fn, x, xi):
        """Deep-tail Weibull points used to yield NaN gradients from the
        unsanitized ``exp(-z)`` Gumbel branch (double-where trap)."""
        g = jax.grad(fn, argnums=(1, 2, 3))(jnp.asarray(x), 0.0, 1.0, xi)
        assert all(jnp.all(jnp.isfinite(gi)) for gi in g)

    def test_dshape_grad_nonzero_at_zero(self):
        """∂/∂ξ was identically 0 in the |ξ|<threshold dead zone, so a GEV fit
        started at the Gumbel model could never move ξ."""
        h = 1e-3
        for fn, arg in [(gev_log_prob, 1.5), (gev_icdf, 0.9)]:
            grad = float(jax.grad(fn, argnums=3)(jnp.asarray(arg), 0.0, 1.0, 0.0))
            fd = float(fn(arg, 0.0, 1.0, h) - fn(arg, 0.0, 1.0, -h)) / (2 * h)
            assert abs(grad) > 1e-2
            assert grad == pytest.approx(fd, rel=5e-2)


class TestReturnLevelPrecision:
    """Large-T return levels lost precision from the 1 - 1/T cancellation."""

    @pytest.mark.parametrize("period", [1e2, 1e4, 1e6])
    @pytest.mark.parametrize("xi", [0.1, -0.1])
    def test_gev_return_level_large_period(self, period, xi):
        got = float(gev_return_level(period, 0.0, 1.0, xi))
        ref = float(st.genextreme(-xi).ppf(1.0 - 1.0 / period))
        assert got == pytest.approx(ref, rel=1e-4)

    @pytest.mark.parametrize("period", [1e2, 1e4, 1e6])
    @pytest.mark.parametrize("xi", [0.2, -0.2])
    def test_gpd_return_level_large_period(self, period, xi):
        got = float(gpd_return_level(period, 1.0, xi))
        ref = float(st.genpareto(xi).ppf(1.0 - 1.0 / period))
        assert got == pytest.approx(ref, rel=1e-4)


class TestGPDOracle:
    """GPD primitives vs scipy.stats.genpareto (c = ξ), incl. new survival."""

    @pytest.mark.parametrize("xi", XI_BAND)
    def test_cdf_logpdf_match_scipy(self, xi):
        x = jnp.linspace(0.05, 6.0, 15)
        ref = st.genpareto(xi, loc=0.0, scale=1.0)
        assert jnp.allclose(gpd_cdf(x, 1.0, xi), ref.cdf(x), atol=1e-5)
        assert jnp.allclose(
            gpd_log_prob(x, 1.0, xi), ref.logpdf(x), atol=1e-4, rtol=1e-4
        )

    @pytest.mark.parametrize("xi", XI_BAND)
    def test_survival_matches_scipy(self, xi):
        x = jnp.linspace(0.05, 6.0, 15)
        ref = st.genpareto(xi, loc=0.0, scale=1.0)
        assert jnp.allclose(gpd_survival(x, 1.0, xi), ref.sf(x), atol=1e-6)
        assert jnp.allclose(
            gpd_log_survival(x, 1.0, xi), ref.logsf(x), atol=1e-4, rtol=1e-4
        )

    def test_survival_equals_one_minus_cdf(self):
        x = jnp.linspace(0.05, 6.0, 15)
        for xi in [-0.2, 5e-8, 1e-4, 0.3]:
            s = gpd_survival(x, 1.0, xi)
            c = gpd_cdf(x, 1.0, xi)
            assert jnp.allclose(s + c, 1.0, atol=1e-6)

    def test_cdf_tail_accuracy_small_argument(self):
        """gpd_cdf(1e-6) was ~19% off from ``1 - power`` cancellation."""
        got = float(gpd_cdf(1e-6, 1.0, 0.2))
        ref = float(st.genpareto(0.2).cdf(1e-6))
        assert got == pytest.approx(ref, rel=1e-4)


class TestExtendedRealLimits:
    """The smooth log1p/expm1 reformulation must keep the support-endpoint and
    ±inf limits that the old explicit branches returned (0·inf → NaN otherwise).
    """

    INF = float("inf")

    @pytest.mark.parametrize(
        "q, xi, expected",
        [
            (1.0, -0.2, 5.0),  # Weibull upper endpoint loc - σ/ξ
            (1.0, 0.0, INF),  # Gumbel upper endpoint
            (1.0, 0.2, INF),  # Fréchet unbounded above
            (0.0, 0.2, -5.0),  # Fréchet lower endpoint loc - σ/ξ
            (0.0, 0.0, -INF),  # Gumbel lower endpoint
            (0.0, -0.2, -INF),  # Weibull unbounded below
        ],
    )
    def test_gev_icdf_endpoints(self, q, xi, expected):
        got = float(gev_icdf(q, 0.0, 1.0, xi))
        if jnp.isinf(jnp.asarray(expected)):
            assert got == expected
        else:
            assert got == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize(
        "xi, expected",
        [(0.0, INF), (0.2, INF), (-0.3, 1.0 / 0.3)],  # -σ/ξ for ξ<0
    )
    def test_gpd_icdf_upper_endpoint(self, xi, expected):
        got = float(gpd_icdf(1.0, 1.0, xi))
        if jnp.isinf(jnp.asarray(expected)):
            assert got == expected
        else:
            assert got == pytest.approx(expected, abs=1e-5)

    def test_gev_cdf_survival_at_infinities_gumbel(self):
        assert float(gev_cdf(-self.INF, 0.0, 1.0, 0.0)) == 0.0
        assert float(gev_cdf(self.INF, 0.0, 1.0, 0.0)) == 1.0
        assert float(gev_survival(-self.INF, 0.0, 1.0, 0.0)) == 1.0
        assert float(gev_log_survival(-self.INF, 0.0, 1.0, 0.0)) == 0.0

    def test_gpd_cdf_survival_at_positive_infinity(self):
        # Exponential limit: F(+inf)=1, S(+inf)=0, and they must agree.
        assert float(gpd_cdf(self.INF, 1.0, 0.0)) == 1.0
        assert float(gpd_survival(self.INF, 1.0, 0.0)) == 0.0

    @pytest.mark.parametrize("fn", [gpd_survival, gpd_cdf, gpd_log_survival])
    @pytest.mark.parametrize("xi", [0.0, 0.2, -0.2])
    def test_gpd_below_support_grad_finite(self, fn, xi):
        """Deep sub-threshold x used to overflow exp(-w) into a NaN gradient in
        the masked branch (constant below x=0, so the gradient must be 0)."""
        g = jax.grad(lambda s: fn(-100.0, s, xi))(1.0)
        assert jnp.isfinite(g)

    @pytest.mark.parametrize(
        "x, xi, expected",
        [
            (INF, 0.2, 1.0),  # Fréchet unbounded above
            (-INF, -0.2, 0.0),  # Weibull unbounded below
            (-INF, 0.2, 0.0),  # below Fréchet lower endpoint
            (INF, -0.2, 1.0),  # above Weibull upper endpoint
        ],
    )
    def test_gev_cdf_at_infinity_nonzero_shape(self, x, xi, expected):
        """log1p(u)/u is inf/inf=NaN at the unbounded tail; the w-limit override
        must restore the analytic 0/1 cdf for nonzero shape too."""
        assert float(gev_cdf(x, 0.0, 1.0, xi)) == expected

    def test_gpd_cdf_survival_at_infinity_nonzero_shape(self):
        assert float(gpd_cdf(self.INF, 1.0, 0.2)) == 1.0
        assert float(gpd_survival(self.INF, 1.0, 0.2)) == 0.0

    def test_gev_icdf_invalid_probability_is_nan(self):
        """q=+inf (invalid) must stay NaN, not be treated as the q=0 endpoint."""
        assert jnp.isnan(gev_icdf(self.INF, 0.0, 1.0, 0.2))

    @pytest.mark.parametrize("xi", [-0.2, -0.5])
    def test_gev_icdf_endpoint_grad_finite_and_correct(self, xi):
        """The discarded inf·0 endpoint branch must not poison the endpoint
        gradient: d/dσ (loc - σ/ξ) = -1/ξ, d/dξ = 1/ξ²."""
        g_scale = float(jax.grad(lambda s: gev_icdf(1.0, 0.0, s, xi))(1.0))
        g_shape = float(jax.grad(lambda z: gev_icdf(1.0, 0.0, 1.0, z))(xi))
        assert g_scale == pytest.approx(-1.0 / xi, rel=1e-4)
        assert g_shape == pytest.approx(1.0 / xi**2, rel=1e-4)

    def test_gpd_icdf_endpoint_grad_finite(self):
        g = jax.grad(lambda s: gpd_icdf(1.0, s, -0.3))(1.0)
        assert jnp.isfinite(g)
        assert float(g) == pytest.approx(1.0 / 0.3, rel=1e-4)

    @pytest.mark.parametrize(
        "fn", [gev_log_prob, gev_cdf, gev_survival, gev_log_survival]
    )
    @pytest.mark.parametrize("x, xi", [(-100.0, 0.2), (100.0, -0.2)])
    def test_gev_out_of_support_grad_finite(self, fn, x, xi):
        """A finite observation outside the parameter-dependent support must not
        overflow exp(-w) into a NaN parameter gradient (breaks fitting when a
        proposed parameter excludes a data point)."""
        for argnums in (2, 3):  # scale, shape
            g = jax.grad(fn, argnums=argnums)(jnp.asarray(x), 0.0, 1.0, xi)
            assert jnp.isfinite(g)

    @pytest.mark.parametrize("fn", [gev_cdf, gev_survival, gev_log_survival])
    @pytest.mark.parametrize("x", [INF, -INF])
    def test_gev_infinity_grad_finite(self, fn, x):
        """Differentiating at x=±inf must give the (zero) gradient of the
        constant extended-real limit, not a 0·inf NaN."""
        for argnums in (2, 3):
            g = jax.grad(fn, argnums=argnums)(jnp.asarray(x), 0.0, 1.0, 0.2)
            assert jnp.isfinite(g)

    @pytest.mark.parametrize("fn", [gpd_cdf, gpd_survival, gpd_log_survival])
    def test_gpd_infinity_grad_finite(self, fn):
        for argnums in (1, 2):  # scale, shape
            g = jax.grad(fn, argnums=argnums)(jnp.asarray(self.INF), 1.0, 0.2)
            assert jnp.isfinite(g)

    @pytest.mark.parametrize(
        "fn", [gev_log_prob, gev_cdf, gev_survival, gev_log_survival]
    )
    @pytest.mark.parametrize("z", [-100.0, -500.0, -1e6])
    @pytest.mark.parametrize("argnums", [1, 2, 3])  # loc, scale, shape
    def test_gumbel_deep_tail_grad_finite(self, fn, z, argnums):
        """The Gumbel (ξ=0) deep lower tail drives exp(-w) past the float32
        overflow; every parameter gradient (incl. the shape gradient, which
        scales as exp(-w)·z²) must stay finite rather than 0·inf = NaN."""
        g = jax.grad(fn, argnums=argnums)(jnp.asarray(z), 0.0, 1.0, 0.0)
        assert jnp.isfinite(g)

    def test_gumbel_log_density_preserves_exp_term(self):
        """The log density must keep the exp(-w) term wherever the dtype can
        represent it — not cap it early like the CDF/survival stabilizer. At
        z=-85 (Gumbel), exp(85) ≈ 8e36 is representable in float32, so the value
        must match -z - exp(-z) rather than a value clipped at exp(80)."""
        z = -85.0
        ref = -z - jnp.exp(jnp.asarray(-z, dtype=jnp.float32))  # -log σ = 0
        got = gev_log_prob(jnp.asarray(z), 0.0, 1.0, 0.0)
        assert float(got) == pytest.approx(float(ref), rel=1e-4)

    def test_gumbel_log_density_saturates_finite_past_overflow(self):
        """Past the exp(-w) overflow point (z=-90, exp(90) > float32 max), the
        density term must saturate at finfo.max — not inf — so the log density
        stays a large finite negative rather than collapsing to -inf."""
        got = gev_log_prob(jnp.asarray(-90.0), 0.0, 1.0, 0.0)
        assert jnp.isfinite(got)
        assert float(got) < 0.0

    @pytest.mark.parametrize("fn", [gev_return_level, gpd_return_level])
    def test_huge_shape_return_level_is_inf_not_nan(self, fn):
        """A huge shape overflows the quantile exponent; expm1_over_x(+inf) must
        yield the +inf tail quantile rather than inf/inf = NaN."""
        args = (10.0, 0.0, 1.0, 2e38) if fn is gev_return_level else (10.0, 1.0, 2e38)
        assert jnp.isposinf(fn(*args))

    @pytest.mark.parametrize("fn", [gev_cdf, gev_survival, gpd_cdf, gpd_survival])
    def test_huge_shape_cdf_survival_not_nan(self, fn):
        """A huge shape overflows ξ·z even after z is clamped; log1p_over_x(+inf)
        must fall back to its 0 limit so the bounded CDF / survival stays finite
        in [0, 1] instead of NaN."""
        args = (
            (10.0, 0.0, 1e-38, 1e20)
            if fn in (gev_cdf, gev_survival)
            else (10.0, 1e-38, 1e20)
        )
        out = fn(*args)
        assert jnp.isfinite(out)
        assert 0.0 <= float(out) <= 1.0

    @pytest.mark.parametrize("fn", [gev_cdf, gev_survival, gpd_cdf, gpd_survival])
    def test_tiny_scale_cdf_survival_finite(self, fn):
        """A finite observation with a tiny (even subnormal) scale overflows the
        raw (x-loc)/σ; the clamped standardization must keep the bounded CDF /
        survival finite in [0, 1], not NaN."""
        args = (
            (10.0, 0.0, 1e-38, 0.2)
            if fn in (gev_cdf, gev_survival)
            else (10.0, 1e-38, 0.2)
        )
        out = fn(*args)
        assert jnp.isfinite(out)
        assert 0.0 <= float(out) <= 1.0

    @pytest.mark.parametrize("fn", [gev_log_prob, gpd_log_prob])
    def test_tiny_scale_log_prob_not_nan(self, fn):
        """The log density at a tiny scale is a degenerate ±inf limit — the
        observation sits infinitely far into the tail as σ → 0, and the -log σ
        normalizer diverges — but it must never collapse to NaN."""
        args = (10.0, 0.0, 1e-38, 0.2) if fn is gev_log_prob else (10.0, 1e-38, 0.2)
        assert not jnp.isnan(fn(*args))

    def test_large_representable_standardized_value_not_clipped(self):
        """The overflow guard must clip only genuinely non-finite ratios, not
        representable-but-large ones: x/σ = 1e20 is finite in float32, so
        gpd_survival(1e20, 1, 10) must equal (1 + 10·1e20)^-0.1 ≈ 0.00794,
        not a distorted value from a prematurely clipped standardization."""
        got = gpd_survival(jnp.asarray(1e20), 1.0, 10.0)
        ref = (1.0 + 10.0 * 1e20) ** (-1.0 / 10.0)
        assert float(got) == pytest.approx(ref, rel=1e-4)

    @pytest.mark.parametrize(
        "fn, args",
        [
            (gev_cdf, (0.0, 1.0, 0.0)),
            (gev_survival, (0.0, 1.0, 0.2)),
            (gev_log_prob, (0.0, 1.0, -0.2)),
            (gev_log_survival, (0.0, 1.0, 0.0)),
            (gpd_cdf, (1.0, 0.2)),
            (gpd_survival, (1.0, 0.0)),
            (gpd_log_prob, (1.0, -0.2)),
            (gpd_log_survival, (1.0, 0.2)),
        ],
    )
    def test_nan_input_propagates(self, fn, args):
        """A NaN observation must stay NaN, not be read as a support boundary."""
        assert jnp.isnan(fn(jnp.asarray(float("nan")), *args))
