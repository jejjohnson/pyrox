"""Smoke tests for the five NumPyro-compatible extreme value distributions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtremax.distributions import (
    FrechetType2GEVD,
    GeneralizedExtremeValueDistribution,
    GeneralizedParetoDistribution,
    GumbelType1GEVD,
    WeibullType3GEVD,
)


@pytest.fixture
def key():
    return jax.random.key(0)


class TestGEVD:
    def test_log_prob_and_sample_shape(self, key):
        dist = GeneralizedExtremeValueDistribution(
            loc=0.0, scale=1.0, concentration=0.1
        )
        samples = dist.sample(key, sample_shape=(32,))
        assert samples.shape == (32,)
        lp = dist.log_prob(samples)
        assert lp.shape == (32,)
        assert jnp.all(jnp.isfinite(lp))

    def test_cdf_icdf_round_trip(self):
        dist = GeneralizedExtremeValueDistribution(
            loc=0.0, scale=1.0, concentration=0.2
        )
        q = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])
        x = dist.icdf(q)
        q_round = dist.cdf(x)
        assert jnp.allclose(q, q_round, atol=1e-4)

    def test_small_negative_shape(self, key):
        dist = GeneralizedExtremeValueDistribution(
            loc=0.0, scale=1.0, concentration=-0.2
        )
        samples = dist.sample(key, sample_shape=(16,))
        lp = dist.log_prob(samples)
        assert jnp.all(jnp.isfinite(lp))

    def test_support_reflects_shape_dependent_bounds(self):
        """Regression: GEVD declared `support = constraints.real` for all
        shapes, but the true support depends on ξ: [μ-σ/ξ, ∞) when ξ>0,
        (-∞, μ-σ/ξ] when ξ<0, the real line only when ξ=0. Now the
        constraint reflects the actual shape-dependent support.
        """
        # ξ > 0: Fréchet branch, lower-bounded at μ - σ/ξ
        d_pos = GeneralizedExtremeValueDistribution(
            loc=0.0, scale=1.0, concentration=0.2
        )
        lower = float(d_pos.lower_bound())  # -5
        assert bool(d_pos.support(jnp.array(lower + 1.0)))
        assert not bool(d_pos.support(jnp.array(lower - 1.0)))

        # ξ < 0: Weibull branch, upper-bounded at μ - σ/ξ
        d_neg = GeneralizedExtremeValueDistribution(
            loc=0.0, scale=1.0, concentration=-0.2
        )
        upper = float(d_neg.upper_bound())  # 5
        assert bool(d_neg.support(jnp.array(upper - 1.0)))
        assert not bool(d_neg.support(jnp.array(upper + 1.0)))

        # ξ = 0: Gumbel branch, unbounded real line
        d_gumbel = GeneralizedExtremeValueDistribution(
            loc=0.0, scale=1.0, concentration=0.0
        )
        assert bool(d_gumbel.support(jnp.array(-1e6)))
        assert bool(d_gumbel.support(jnp.array(1e6)))

    def test_mean_excess_normalizes_by_full_survival(self):
        """Regression: the quantile-space quadrature previously normalised
        by ``(1 - ε) - F(u)`` (truncated mass) instead of ``1 - F(u)``
        and used a linear-p grid that underresolved heavy tails. For a
        Fréchet branch with ξ close to 1, the truncated tail could
        contribute materially to the finite mean, producing systematic
        underestimation. The log-tail-probability grid plus normalisation
        by S(u) fixes both issues.
        """
        d = GeneralizedExtremeValueDistribution(loc=0.0, scale=1.0, concentration=0.5)
        # At u=0 the mean excess should grow with ξ but remain finite
        # and well above the GPD-linear POT estimate (σ/(1-ξ) = 2.0 here).
        me = float(d.conditional_excess_mean(jnp.array(0.0)))
        assert jnp.isfinite(me)
        assert me > 1.5  # well above σ

    def test_mean_excess_varies_with_threshold_at_gumbel_limit(self):
        """Regression: GEVD mean excess used the GPD linear POT
        approximation `(σ + ξ(u-μ))/(1-ξ)`, which collapses to a constant
        `σ` for every threshold when ξ=0 — even though the true Gumbel
        mean excess depends on u and only asymptotes to σ in the far tail.
        """
        from xtremax.distributions import GumbelType1GEVD

        gev0 = GeneralizedExtremeValueDistribution(
            loc=0.0, scale=1.0, concentration=0.0
        )
        gumbel_ref = GumbelType1GEVD(loc=0.0, scale=1.0)
        thresholds = jnp.array([0.0, 1.0, 3.0, 5.0])
        me_gev = gev0.conditional_excess_mean(thresholds)
        me_gumbel = gumbel_ref.conditional_excess_mean(thresholds)
        # Must match the independently-implemented Gumbel quadrature
        # (reviewed / corrected in an earlier round).
        assert jnp.allclose(me_gev, me_gumbel, atol=1e-2)
        # Must not be constant = scale.
        assert float(jnp.std(me_gev)) > 0.05
        # At u=0 mean excess is well above σ; at u=5 it's close to σ.
        assert float(me_gev[0]) > 1.1
        assert abs(float(me_gev[2]) - 1.0) < 0.05

    def test_mode_matches_argmax_of_pdf(self):
        """Regression: the non-Gumbel mode used `(1+ξ)^ξ` with the wrong
        exponent sign. The correct stationary point is `(1+ξ)^(-ξ)`.
        Verify by grid-search argmax of the pdf for a few shapes.
        """
        for shape in [-0.3, -0.1, 0.1, 0.3]:
            d = GeneralizedExtremeValueDistribution(
                loc=0.0, scale=1.0, concentration=shape
            )
            # Grid covers a wide range relative to the typical mode of
            # σ·((1+ξ)^(-ξ) - 1)/ξ (a fraction of σ around μ).
            grid = jnp.linspace(-5.0, 5.0, 20001)
            pdf = jnp.exp(d.log_prob(grid))
            safe_pdf = jnp.where(jnp.isfinite(pdf), pdf, -1.0)
            empirical = float(grid[jnp.argmax(safe_pdf)])
            analytical = float(d.mode)
            assert abs(empirical - analytical) < 5e-3, (
                f"shape={shape}: empirical {empirical} vs analytical {analytical}"
            )

    def test_mode_equals_upper_bound_when_shape_le_minus_one(self):
        """For ξ ≤ -1 the GEV density is maximized at the upper endpoint,
        not at the interior stationary-point formula.
        """
        for shape in [-1.0, -1.2, -2.0]:
            d = GeneralizedExtremeValueDistribution(
                loc=0.0, scale=1.0, concentration=shape
            )
            assert jnp.allclose(d.mode, d.upper_bound(), atol=1e-6)

    def test_log_survival_stays_finite_in_gumbel_far_tail(self):
        """Regression: generic GEV log survival used `log(1 - cdf)` and
        underflowed to `-inf` in the far Gumbel tail.
        """
        d = GeneralizedExtremeValueDistribution(loc=0.0, scale=1.5, concentration=0.0)
        ref = GumbelType1GEVD(loc=0.0, scale=1.5)
        x = jnp.array(20.0)
        assert jnp.isfinite(d.log_survival_function(x))
        assert jnp.allclose(d.log_survival_function(x), ref.log_survival_function(x))

    def test_frechet_survival_stays_positive_in_far_tail(self):
        """Regression: generic GEV survival used `1 - cdf` and rounded to
        zero in the far Fréchet tail, making the cumulative hazard blow up.
        """
        d = GeneralizedExtremeValueDistribution(loc=0.0, scale=1.0, concentration=0.2)
        x = jnp.array(1000.0)
        survival = d.survival_function(x)
        cumulative_hazard = d.cumulative_hazard_rate(x)
        assert float(survival) > 0.0
        assert jnp.isfinite(cumulative_hazard)
        assert jnp.allclose(cumulative_hazard, -d.log_survival_function(x), atol=1e-6)


class TestGPD:
    def test_log_prob_and_sample_shape(self, key):
        dist = GeneralizedParetoDistribution(scale=1.0, concentration=0.2)
        samples = dist.sample(key, sample_shape=(32,))
        assert samples.shape == (32,)
        lp = dist.log_prob(samples)
        assert jnp.all(jnp.isfinite(lp))
        assert jnp.all(samples >= 0)  # GPD support is x >= 0 when loc = 0

    def test_percentile_residual_life_at_p_zero_returns_zero(self):
        """Regression: conditional CDF was `1 - p*S(t)` (wrong) instead of
        `1 - (1-p)*S(t)`. At p=0 the correct residual life is 0 (the
        conditional 0th-percentile equals the threshold).
        """
        d = GeneralizedExtremeValueDistribution(0.0, 1.0, 0.1)
        r = d.percentile_residual_life(jnp.array(1.0), percentile=0.0)
        assert jnp.allclose(r, 0.0, atol=1e-5)

    def test_entropy_matches_closed_form(self):
        """Regression: the ξ≠0 branch used `log σ + 1 + ξ + γξ`; the
        correct formula is `log σ + 1 + γ(1 + ξ)`.
        """
        scale, xi = 2.0, 0.2
        d = GeneralizedExtremeValueDistribution(0.0, scale, xi)
        euler_gamma = 0.5772156649015329
        expected = float(jnp.log(scale) + 1.0 + euler_gamma * (1.0 + xi))
        assert jnp.allclose(d.entropy(), expected, atol=1e-5)

    def test_entropy_continuous_at_shape_zero(self):
        """Entropy should not jump at ξ=0 (Gumbel limit is the same formula)."""
        scale = 2.0
        euler_gamma = 0.5772156649015329
        d_gumbel = GeneralizedExtremeValueDistribution(0.0, scale, 0.0)
        d_tiny = GeneralizedExtremeValueDistribution(0.0, scale, 1e-8)
        expected_gumbel = float(jnp.log(scale) + 1.0 + euler_gamma)
        assert jnp.allclose(d_gumbel.entropy(), expected_gumbel, atol=1e-5)
        assert jnp.allclose(d_tiny.entropy(), d_gumbel.entropy(), atol=1e-6)

    def test_support_rejects_above_upper_bound_when_shape_negative(self):
        """Regression: GPD declared `support = constraints.nonnegative` for
        all shapes, but when ξ<0 the support is `[0, -σ/ξ]`. A sample
        above the finite upper endpoint is outside the domain and must be
        rejected by the support constraint (so `validate_args=True` does
        its job instead of deferring to log_prob → -∞).
        """
        d = GeneralizedParetoDistribution(scale=1.0, concentration=-0.5)
        upper = float(d.upper_bound())  # = 2.0
        # Within support
        assert bool(d.support(jnp.array(1.0)))
        # Above the finite upper endpoint
        assert not bool(d.support(jnp.array(upper + 0.1)))
        # Below the lower bound (x < 0)
        assert not bool(d.support(jnp.array(-0.1)))

    def test_support_accepts_all_nonnegative_when_shape_positive(self):
        """For ξ ≥ 0 the support is [0, +∞), so any nonnegative x must
        be accepted (no upper-bound leakage from the constraint change).
        """
        d = GeneralizedParetoDistribution(scale=1.0, concentration=0.2)
        assert bool(d.support(jnp.array(0.5)))
        assert bool(d.support(jnp.array(1e6)))
        assert not bool(d.support(jnp.array(-0.1)))

    def test_expand_preserves_state(self):
        """Regression: the custom GPD.expand() override called
        `_get_checked_instance` (which does not exist on the current
        NumPyro Distribution) and bypassed __init__, so attributes like
        ``_exponential_threshold`` were missing on the returned instance
        and every cdf/skew/kurtosis call raised AttributeError. Rebuilding
        via ``__init__`` restores all cached state.
        """
        d = GeneralizedParetoDistribution(scale=1.0, concentration=0.2)
        expanded = d.expand((3,))
        assert expanded.batch_shape == (3,)
        assert hasattr(expanded, "_exponential_threshold")
        x = jnp.array([0.1, 0.5, 1.0])
        _ = expanded.cdf(x)
        _ = expanded.skew()
        _ = expanded.kurtosis()

    def test_hazard_rate_zero_below_support(self):
        """Regression: GPD hazard_rate previously only checked `σ + ξx > 0`,
        so for `x < 0` (outside GPD support where f=0, S=1, h=0) it returned
        positive hazards instead of zero.
        """
        d = GeneralizedParetoDistribution(scale=1.5, concentration=0.2)
        x_below = jnp.array([-5.0, -1.0, -0.01])
        h = d.hazard_rate(x_below)
        assert jnp.all(h == 0.0)

    def test_hazard_rate_matches_pdf_over_survival_on_support(self):
        """Within the support hazard must equal f/S."""
        d = GeneralizedParetoDistribution(scale=1.5, concentration=0.2)
        x = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0])
        pdf = jnp.exp(d.log_prob(x))
        surv = d.survival_function(x)
        expected = pdf / surv
        assert jnp.allclose(d.hazard_rate(x), expected, rtol=1e-5)

    def test_survival_uses_scale_not_shape(self):
        """Regression: survival_function previously aliased `scale = self.shape`.

        With scale=2 and shape=0.3 the correct S(1) is a specific number;
        the buggy version used `shape` as both the scale and shape, giving
        a wildly different value. Verify against 1 - cdf().
        """
        d = GeneralizedParetoDistribution(scale=2.0, concentration=0.3)
        x = jnp.array([0.5, 1.0, 2.0, 5.0])
        assert jnp.allclose(d.survival_function(x), 1.0 - d.cdf(x), atol=1e-6)

    def test_survival_and_cumulative_hazard_stable_in_far_tail(self):
        """Regression: `1 - cdf` cancelled to zero in the far tail even when
        the true survival probability was still representable.
        """
        d = GeneralizedParetoDistribution(scale=1.0, concentration=0.2)
        x = jnp.array(1000.0)
        survival = d.survival_function(x)
        cumulative_hazard = d.cumulative_hazard_rate(x)
        expected_survival = jnp.power(1.0 + 0.2 * x, -5.0)
        expected_cumulative_hazard = 5.0 * jnp.log(1.0 + 0.2 * x)
        assert float(survival) > 0.0
        assert jnp.isfinite(cumulative_hazard)
        assert jnp.allclose(survival, expected_survival, rtol=1e-5)
        assert jnp.allclose(cumulative_hazard, expected_cumulative_hazard, rtol=1e-5)


class TestGumbel:
    def test_log_prob_and_sample_shape(self, key):
        dist = GumbelType1GEVD(loc=0.0, scale=1.0)
        samples = dist.sample(key, sample_shape=(32,))
        assert samples.shape == (32,)
        lp = dist.log_prob(samples)
        assert jnp.all(jnp.isfinite(lp))

    def test_expand_preserves_state(self):
        """Regression: the custom Gumbel.expand() bypassed __init__ and
        called a non-existent `_get_checked_instance`, losing cached
        constants (`_pi_squared_over_six`, `_gumbel_skewness`,
        `_gumbel_kurtosis`, `_euler_gamma`). Rebuilding via ``__init__``
        keeps variance/skew/kurtosis/entropy reachable.
        """
        d = GumbelType1GEVD(loc=0.0, scale=1.0)
        expanded = d.expand((4,))
        assert expanded.batch_shape == (4,)
        for attr in (
            "_euler_gamma",
            "_pi_squared_over_six",
            "_gumbel_skewness",
            "_gumbel_kurtosis",
        ):
            assert hasattr(expanded, attr), f"missing {attr}"
        _ = expanded.variance
        _ = expanded.skew
        _ = expanded.kurtosis
        _ = expanded.entropy()

    def test_hazard_rate_matches_f_over_S(self):
        """h(x) must equal f(x) / S(x), not exp(z)/σ."""
        d = GumbelType1GEVD(loc=0.0, scale=2.0)
        x = jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        pdf = jnp.exp(d.log_prob(x))
        surv = 1.0 - d.cdf(x)
        expected = pdf / surv
        assert jnp.allclose(d.hazard_rate(x), expected, rtol=1e-5)

    def test_cumulative_hazard_matches_neg_log_survival(self):
        """Λ(x) = -log S(x)."""
        d = GumbelType1GEVD(loc=0.0, scale=2.0)
        x = jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        expected = -jnp.log(1.0 - d.cdf(x))
        assert jnp.allclose(d.cumulative_hazard_rate(x), expected, rtol=1e-5)

    def test_hazard_asymptotes_to_inverse_scale(self):
        """At the far upper tail, h(x) → 1/σ (exponential-tail limit)."""
        scale = 1.5
        d = GumbelType1GEVD(loc=0.0, scale=scale)
        x_far = jnp.array([20.0])  # deep in upper tail
        h = d.hazard_rate(x_far)
        assert jnp.allclose(h, 1.0 / scale, atol=1e-5)

    def test_conditional_excess_mean_uses_survival_not_cdf(self):
        """Regression: survival_u was previously `exp(-exp(-z_u))` (the CDF).

        With loc=0, scale=1, threshold=-2 we have F(-2) ≈ 5.7e-4 but
        S(-2) ≈ 0.9994. The bug returned NaN at low thresholds where the
        CDF is tiny; the fix should return a finite mean excess.
        """
        d = GumbelType1GEVD(loc=0.0, scale=1.0)
        m = d.conditional_excess_mean(jnp.array(-2.0))
        assert jnp.isfinite(m)

    def test_conditional_excess_mean_upper_tail_asymptote(self):
        """Regression: the old closed form `σ·(exp(-z_u) + γ)` converged to
        γσ (≈ 0.577σ) instead of σ in the upper tail. The correct limit
        for a Gumbel tail is σ (the exponential-tail mean-residual-life).
        """
        scale = 1.5
        d = GumbelType1GEVD(loc=0.0, scale=scale)
        m = d.conditional_excess_mean(jnp.array(20.0))
        assert jnp.allclose(m, scale, atol=5e-3)

    def test_conditional_excess_mean_handles_far_negative_threshold(self):
        """Regression: the fixed ``u + 50σ`` integration cap truncated
        mass when u sat far below the location. For u = -1000σ below μ,
        the true mean excess is ≈ γσ - u ≈ |u|; the old formula missed
        by several orders of magnitude. The adaptive cap fixes this.
        """
        scale = 1.0
        euler_gamma = 0.5772156649015329
        d = GumbelType1GEVD(loc=0.0, scale=scale)
        for u_val in [-1000.0, -100.0]:
            me = float(d.conditional_excess_mean(jnp.array(u_val)))
            expected = euler_gamma * scale - u_val
            # Allow ~1% error for quadrature approximation.
            assert abs(me - expected) / abs(expected) < 0.01, (
                f"u={u_val}: me={me}, expected≈{expected}"
            )

    def test_conditional_excess_mean_vectorizes_over_thresholds(self):
        """Regression: the trapezoidal integration previously collapsed on
        vector thresholds because the grid axis collided with the batch
        axis. Passing an array of thresholds must now broadcast cleanly.
        """
        d = GumbelType1GEVD(loc=0.0, scale=1.0)
        thresholds = jnp.array([0.0, 1.0, 5.0, 10.0])
        m = d.conditional_excess_mean(thresholds)
        assert m.shape == thresholds.shape
        assert jnp.all(jnp.isfinite(m))
        # Each element must match the corresponding scalar call.
        for i, u in enumerate(thresholds):
            assert jnp.allclose(m[i], d.conditional_excess_mean(u), atol=1e-4)

    def test_characteristic_function_handles_complex_gamma(self):
        """Regression: characteristic_function called `jax.scipy.special.gammaln`
        on a complex argument `1 - iσt`. `gammaln` is real-only, so the
        method either failed or returned wrong values. Now delegated to
        `scipy.special.loggamma` via `jax.pure_callback`.
        """
        d = GumbelType1GEVD(loc=0.5, scale=1.5)
        t = jnp.array([0.0, 0.5, 1.0, 2.0], dtype=jnp.float32)
        phi = d.characteristic_function(t)
        # φ(0) = 1.
        assert jnp.allclose(phi[0], 1.0 + 0.0j, atol=1e-5)
        # |φ(t)| ≤ 1 for any characteristic function.
        magnitudes = jnp.abs(phi)
        assert jnp.all(magnitudes <= 1.0 + 1e-4)
        # φ(-t) = conj(φ(t)) for real-valued X.
        phi_neg = d.characteristic_function(-t)
        assert jnp.allclose(phi_neg, jnp.conj(phi), atol=1e-4)

    def test_log_survival_upper_tail_asymptote(self):
        """Regression: the z > 5 branch returned ≈ -exp(-z), not ≈ -z.

        For Gumbel S(x) ≈ exp(-z) in the upper tail, so log S ≈ -z.
        """
        scale = 1.5
        d = GumbelType1GEVD(loc=0.0, scale=scale)
        x = jnp.array(20.0)
        log_s = d.log_survival_function(x)
        expected_z = float(x / scale)
        # Allow some numerical slack — but it must be close to -z ≈ -13.3,
        # not close to 0.
        assert float(log_s) == pytest.approx(-expected_z, abs=1e-3)


class TestFrechet:
    def test_log_prob_and_sample_shape(self, key):
        dist = FrechetType2GEVD(loc=0.0, scale=1.0, concentration=0.2)
        samples = dist.sample(key, sample_shape=(32,))
        assert samples.shape == (32,)
        lp = dist.log_prob(samples)
        assert jnp.all(jnp.isfinite(lp))

    def test_log_survival_matches_log_of_survival(self):
        """Regression: log_survival previously returned log F(x), not log S(x)."""
        d = FrechetType2GEVD(loc=0.0, scale=1.0, concentration=0.2)
        x = jnp.linspace(1.5, 10.0, 10)
        log_s = d.log_survival_function(x)
        expected = jnp.log(1.0 - d.cdf(x))
        assert jnp.allclose(log_s, expected, atol=1e-5)

    def test_mode_matches_argmax_of_pdf(self):
        """Regression: Fréchet mode used `(1+ξ)^ξ` — the GEV-parameterisation
        stationary point is `(1+ξ)^(-ξ)`. Verify by grid argmax of the pdf.
        """
        for shape in [0.1, 0.3, 0.5]:
            d = FrechetType2GEVD(loc=0.0, scale=1.0, concentration=shape)
            # Support is x > μ - σ/ξ; for μ=0,σ=1 this is x > -1/ξ.
            # The mode sits slightly below 0 for small ξ, slightly above
            # for larger ξ, so scan a wide window inside the support.
            lower = -1.0 / shape + 1e-4
            grid = jnp.linspace(max(lower, -5.0), 10.0, 40001)
            pdf = jnp.exp(d.log_prob(grid))
            safe_pdf = jnp.where(jnp.isfinite(pdf), pdf, -1.0)
            empirical = float(grid[jnp.argmax(safe_pdf)])
            analytical = float(d.mode)
            assert abs(empirical - analytical) < 5e-3, (
                f"shape={shape}: empirical {empirical} vs analytical {analytical}"
            )

    def test_construct_under_jit_succeeds(self):
        """Regression: the Fréchet constructor branched on
        ``if jnp.any(shape <= 0)``, which forced Python truth-value
        evaluation of a traced JAX array. Building the distribution
        inside ``jit`` then raised a tracer concretization error before
        any sampling could happen. The guard is removed — domain
        validation lives in ``arg_constraints``.
        """

        @jax.jit
        def make_and_logprob(shape):
            d = FrechetType2GEVD(loc=0.0, scale=1.0, concentration=shape)
            return d.log_prob(jnp.array(1.0))

        result = make_and_logprob(jnp.array(0.3))
        assert jnp.isfinite(result)

    def test_entropy_matches_gev_formula(self):
        """Regression: Fréchet entropy used `log σ + ξ + 1 + γξ`; the
        correct GEV-branch entropy is `log σ + 1 + γ(1 + ξ)`.
        """
        scale, shape = 1.7, 0.25
        d = FrechetType2GEVD(loc=0.0, scale=scale, concentration=shape)
        euler_gamma = 0.5772156649015329
        expected = float(jnp.log(scale) + 1.0 + euler_gamma * (1.0 + shape))
        assert jnp.allclose(d.entropy(), expected, atol=1e-6)

    def test_mean_excess_varies_with_threshold(self):
        """Regression: Fréchet mean excess used the GPD linear POT form,
        which is only the asymptotic limit. The quantile-space quadrature
        now returns threshold-dependent values that grow sub-linearly with
        u in the heavy-tail regime (ξ > 0).
        """
        d = FrechetType2GEVD(loc=0.0, scale=1.0, concentration=0.3)
        thresholds = jnp.array([0.0, 1.0, 2.0, 5.0])
        me = d.conditional_excess_mean(thresholds)
        # Monotonically increasing with threshold (heavier tail).
        assert bool(jnp.all(jnp.diff(me) > 0.0))
        # All finite for ξ < 1.
        assert bool(jnp.all(jnp.isfinite(me)))

    def test_support_accepts_lower_endpoint(self):
        """Fréchet support is closed at the lower endpoint x = μ - σ/ξ."""
        d = FrechetType2GEVD(loc=0.0, scale=1.0, concentration=0.2)
        assert bool(d.support(d.lower_bound()))

    def test_survival_stays_positive_in_far_tail(self):
        """Regression: ``survival_function`` used ``1 - self.cdf(x)`` and
        catastrophically cancelled in the far right tail — once
        ``F(x) ≈ 1 - 1e-8`` float32 rounds ``1 - F`` to zero, and
        ``exceedance_probability``/``hazard_rate`` then return impossible
        zero tail probabilities at finite inputs. The stable
        ``-expm1(log F)`` form preserves the tail down to subnormals.
        """
        d = FrechetType2GEVD(loc=0.0, scale=1.0, concentration=0.3)
        x = jnp.array([10.0, 100.0, 1000.0, 1e5])
        s = d.survival_function(x)
        # All finite inputs must give strictly positive survival.
        assert bool(jnp.all(s > 0.0))
        # And must match log_survival_function to round-off.
        log_s = d.log_survival_function(x)
        assert jnp.allclose(jnp.log(s), log_s, atol=1e-5)


class TestWeibull:
    def test_log_prob_and_sample_shape(self, key):
        dist = WeibullType3GEVD(loc=0.0, scale=1.0, concentration=-0.2)
        samples = dist.sample(key, sample_shape=(32,))
        assert samples.shape == (32,)
        lp = dist.log_prob(samples)
        assert jnp.all(jnp.isfinite(lp))

    def test_percentile_residual_life_at_p_zero_returns_zero(self):
        """Same conditional-CDF fix as the GEVD case."""
        d = WeibullType3GEVD(0.0, 1.0, -0.3)
        r = d.percentile_residual_life(jnp.array(-1.0), percentile=0.0)
        assert jnp.allclose(r, 0.0, atol=1e-5)

    def test_construct_under_jit_succeeds(self):
        """Regression: the Weibull constructor branched on
        ``if jnp.any(shape >= 0)``, which broke under ``jit`` with
        traced inputs. Guard removed; domain validation is via
        ``arg_constraints``.
        """

        @jax.jit
        def make_and_logprob(shape):
            d = WeibullType3GEVD(loc=0.0, scale=1.0, concentration=shape)
            return d.log_prob(jnp.array(-1.0))

        result = make_and_logprob(jnp.array(-0.3))
        assert jnp.isfinite(result)

    def test_moments_finite_across_full_valid_shape_range(self):
        """Regression: variance/skew/kurtosis used guards ``ξ > -1/2``,
        ``ξ > -1/3``, ``ξ > -1/4`` inherited from the Fréchet moment
        existence conditions reflected to negative ξ. Weibull Type III
        has ξ<0 (bounded support), so ALL moments are finite for every
        valid ξ. The guards silently returned NaN for e.g. ξ = -1.
        """
        for xi in [-0.1, -0.5, -0.7, -1.0, -2.0]:
            d = WeibullType3GEVD(loc=0.0, scale=1.0, concentration=xi)
            v = float(d.variance)
            s = float(d.skew())
            k = float(d.kurtosis())
            assert jnp.isfinite(v), f"variance NaN at ξ={xi}"
            assert jnp.isfinite(s), f"skew NaN at ξ={xi}"
            assert jnp.isfinite(k), f"kurtosis NaN at ξ={xi}"
            # Sanity: variance must be positive.
            assert v > 0.0

    def test_mean_excess_decays_to_zero_near_upper_bound(self):
        """Regression: Weibull mean excess used the GPD linear POT form,
        which does NOT vanish as the threshold approaches the finite
        upper endpoint μ - σ/ξ. Quantile-space quadrature correctly
        returns values tending to zero as u → upper bound (and NaN once
        F(u) ≥ 1 - 1e-6, beyond the quadrature's float32 reach).
        """
        d = WeibullType3GEVD(loc=0.0, scale=1.0, concentration=-0.3)
        ub = float(d.upper_bound())
        thresholds = jnp.array([ub * 0.3, ub * 0.6, ub * 0.9, ub * 0.95])
        me = d.conditional_excess_mean(thresholds)
        # All values must be finite in this range.
        assert bool(jnp.all(jnp.isfinite(me)))
        # Monotonically decreasing toward the upper bound.
        assert bool(jnp.all(jnp.diff(me) < 0.0))
        # By u = 0.9 * upper_bound the mean excess is well below scale=1.
        assert float(me[2]) < 0.15

    def test_entropy_matches_gev_formula(self):
        """Regression: Weibull entropy used `log σ + ξ + 1 + γξ`; the
        correct GEV-branch formula is `log σ + 1 + γ(1 + ξ)`. At ξ = 0
        it must also reduce to the Gumbel entropy `log σ + 1 + γ`.
        """
        scale, xi = 1.7, -0.25
        d = WeibullType3GEVD(loc=0.0, scale=scale, concentration=xi)
        euler_gamma = 0.5772156649015329
        expected = float(jnp.log(scale) + 1.0 + euler_gamma * (1.0 + xi))
        assert jnp.allclose(d.entropy(), expected, atol=1e-6)

    def test_entropy_continuous_at_shape_zero(self):
        """Weibull entropy at ξ→0⁻ must match the Gumbel formula."""
        scale = 1.7
        euler_gamma = 0.5772156649015329
        d_tiny = WeibullType3GEVD(loc=0.0, scale=scale, concentration=-1e-8)
        expected_gumbel = float(jnp.log(scale) + 1.0 + euler_gamma)
        assert jnp.allclose(d_tiny.entropy(), expected_gumbel, atol=1e-5)

    def test_mode_matches_argmax_of_pdf(self):
        """Regression: Weibull Type III mode used `(1+ξ)^ξ` — the
        GEV-parameterisation stationary point is `(1+ξ)^(-ξ)`.
        """
        for shape in [-0.3, -0.2, -0.1]:
            d = WeibullType3GEVD(loc=0.0, scale=1.0, concentration=shape)
            # Support has upper bound μ - σ/ξ = 1/|ξ|; scan below it.
            upper = -1.0 / shape
            grid = jnp.linspace(upper - 10.0, upper - 1e-4, 20001)
            pdf = jnp.exp(d.log_prob(grid))
            safe_pdf = jnp.where(jnp.isfinite(pdf), pdf, -1.0)
            empirical = float(grid[jnp.argmax(safe_pdf)])
            analytical = float(d.mode)
            assert abs(empirical - analytical) < 5e-3, (
                f"shape={shape}: empirical {empirical} vs analytical {analytical}"
            )

    def test_mode_equals_upper_bound_when_shape_le_minus_one(self):
        """For ξ ≤ -1 the reverse-Weibull density peaks at the upper endpoint."""
        for shape in [-1.0, -1.2, -2.0]:
            d = WeibullType3GEVD(loc=0.0, scale=1.0, concentration=shape)
            assert jnp.allclose(d.mode, d.upper_bound(), atol=1e-6)

    def test_support_accepts_upper_endpoint(self):
        """Weibull support is closed at the upper endpoint x = μ - σ/ξ."""
        d = WeibullType3GEVD(loc=0.0, scale=1.0, concentration=-0.2)
        assert bool(d.support(d.upper_bound()))

    def test_survival_stays_positive_near_upper_endpoint(self):
        """Regression: ``survival_function`` used ``1 - self.cdf(x)`` which
        cancels to zero near the finite upper endpoint. Downstream
        ``hazard_rate``/``cumulative_hazard_rate`` (via ``log(S)``) then
        produced ``-inf``/``nan`` at valid in-support inputs. The stable
        ``-expm1(log F)`` form preserves the tail down to subnormals.
        """
        d = WeibullType3GEVD(loc=0.0, scale=1.0, concentration=-0.3)
        upper = float(d.upper_bound())
        # Stay safely below the endpoint — the bug was that S collapsed
        # to zero while still strictly inside support.
        x = jnp.array([frac * upper for frac in [0.9, 0.99, 0.999, 0.9999]])
        s = d.survival_function(x)
        assert bool(jnp.all(s > 0.0))
        log_s = d.log_survival_function(x)
        # log_survival_function must agree with log(survival_function).
        assert jnp.allclose(jnp.log(s), log_s, atol=1e-5)
        # cumulative_hazard_rate must equal -log S and be finite here.
        ch = d.cumulative_hazard_rate(x)
        assert bool(jnp.all(jnp.isfinite(ch)))
        assert jnp.allclose(ch, -log_s, atol=1e-6)

    def test_survival_boundary_values(self):
        """Weibull: S(x) = 1 below lower (in the unbounded-below part of
        the real line) and S(x) = 0 above the finite upper endpoint.
        """
        d = WeibullType3GEVD(loc=0.0, scale=1.0, concentration=-0.3)
        upper = float(d.upper_bound())
        # Above upper bound → S = 0, log S = -inf.
        assert float(d.survival_function(jnp.array(upper + 1.0))) == 0.0
        assert float(d.log_survival_function(jnp.array(upper + 1.0))) == -float("inf")
        # Well below support → S ≈ 1.
        assert jnp.isclose(d.survival_function(jnp.array(-1e3)), 1.0, atol=1e-6)


class TestPRNGKeyValidation:
    """sample() must raise TypeError (not AssertionError) on bad keys."""

    def test_rejects_non_key_with_typeerror(self):
        d = GumbelType1GEVD(loc=0.0, scale=1.0)
        with pytest.raises(TypeError, match="JAX PRNG key"):
            d.sample(42, sample_shape=(4,))  # plain int, not a key

    def test_accepts_legacy_prngkey(self):
        d = GumbelType1GEVD(loc=0.0, scale=1.0)
        legacy = jax.random.PRNGKey(0)
        samples = d.sample(legacy, sample_shape=(4,))
        assert samples.shape == (4,)

    def test_rejects_float_array_with_typeerror(self):
        """Regression: `is_typed = not issubdtype(dtype, integer)` used to
        classify any non-integer array as a typed PRNG key, so a plain
        float32 array was silently accepted and broke deep inside the
        sampling call. The validator now uses `jax.dtypes.prng_key` and
        rejects non-keys up front with a clear TypeError.
        """
        d = GumbelType1GEVD(loc=0.0, scale=1.0)
        not_a_key = jnp.array([1.0, 2.0], dtype=jnp.float32)
        with pytest.raises(TypeError, match="JAX PRNG key"):
            d.sample(not_a_key, sample_shape=(4,))

    def test_sample_does_not_emit_plus_minus_infinity(self):
        """Regression: inverse-transform sampling used Uniform(0, 1) whose
        JAX sampler can emit exact 0 (and 1 in some dtypes), sending
        icdf to -inf (Gumbel/GEV at p=0) or +inf (Fréchet at p=1).
        Samples must now all be finite.
        """
        from xtremax.distributions import (
            FrechetType2GEVD,
            GeneralizedExtremeValueDistribution,
            GeneralizedParetoDistribution,
            WeibullType3GEVD,
        )

        key = jax.random.key(0)
        dists = [
            GumbelType1GEVD(loc=0.0, scale=1.0),
            GeneralizedExtremeValueDistribution(loc=0.0, scale=1.0, concentration=0.0),
            GeneralizedExtremeValueDistribution(loc=0.0, scale=1.0, concentration=0.2),
            GeneralizedExtremeValueDistribution(loc=0.0, scale=1.0, concentration=-0.2),
            GeneralizedParetoDistribution(scale=1.0, concentration=0.2),
            GeneralizedParetoDistribution(scale=1.0, concentration=-0.2),
            FrechetType2GEVD(loc=0.0, scale=1.0, concentration=0.3),
            WeibullType3GEVD(loc=0.0, scale=1.0, concentration=-0.3),
        ]
        for d in dists:
            samples = d.sample(key, sample_shape=(4096,))
            assert bool(jnp.all(jnp.isfinite(samples))), (
                f"{type(d).__name__} emitted non-finite samples"
            )

    def test_rejects_wrong_shape_uint32_with_typeerror(self):
        """A uint32 array that isn't shaped like a legacy key must still
        be rejected (e.g. `uint32[5]` where the trailing dim != 2).
        """
        d = GumbelType1GEVD(loc=0.0, scale=1.0)
        not_a_key = jnp.array([1, 2, 3, 4, 5], dtype=jnp.uint32)
        with pytest.raises(TypeError, match="JAX PRNG key"):
            d.sample(not_a_key, sample_shape=(4,))


class TestShapeParameterAlias:
    """Regression: GEVD/GPD/Frechet/Weibull previously stored the tail-
    index parameter on ``self.shape``, which shadowed NumPyro's
    ``Distribution.shape()`` method. Tooling that calls
    ``fn.shape(sample_shape)`` would hit a non-callable tensor. The
    attribute is now ``self.concentration`` and ``shape=`` remains as a
    deprecated constructor alias.
    """

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: GeneralizedExtremeValueDistribution(
                loc=0.0, scale=1.0, concentration=0.2
            ),
            lambda: GeneralizedParetoDistribution(scale=1.0, concentration=0.2),
            lambda: FrechetType2GEVD(loc=0.0, scale=1.0, concentration=0.3),
            lambda: WeibullType3GEVD(loc=0.0, scale=1.0, concentration=-0.3),
        ],
    )
    def test_distribution_shape_method_is_callable(self, factory):
        d = factory()
        # shape() must be NumPyro's bound method, not a tensor.
        assert callable(d.shape), (
            f"{type(d).__name__}.shape shadowed by parameter tensor"
        )
        # And returns a tuple for the given sample_shape.
        assert d.shape((3,)) == (3, *d.batch_shape, *d.event_shape)

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (GeneralizedParetoDistribution, {"scale": 1.0}),
            (FrechetType2GEVD, {"loc": 0.0, "scale": 1.0}),
            (WeibullType3GEVD, {"loc": 0.0, "scale": 1.0}),
        ],
    )
    def test_shape_kwarg_emits_deprecation_and_still_works(self, cls, kwargs):
        xi = -0.3 if cls is WeibullType3GEVD else 0.2
        with pytest.warns(DeprecationWarning, match="'shape' is deprecated"):
            d = cls(**kwargs, shape=xi)
        assert jnp.allclose(d.concentration, jnp.asarray(xi))

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (GeneralizedParetoDistribution, {"scale": 1.0}),
            (FrechetType2GEVD, {"loc": 0.0, "scale": 1.0}),
            (WeibullType3GEVD, {"loc": 0.0, "scale": 1.0}),
        ],
    )
    def test_passing_both_concentration_and_shape_raises(self, cls, kwargs):
        with pytest.raises(ValueError, match="Pass only one"):
            cls(**kwargs, concentration=0.2, shape=0.2)


class TestMeanExcessFarTailMonotonicGrid:
    """Regression: the log-tail quadrature used a fixed lower bound
    ``log(1e-6)`` and interpolated up to ``log S(u)``. When
    ``S(u) < 1e-6`` (far-out threshold) the grid ran backward —
    ``jnp.trapezoid(..., x=v_grid)`` then integrated with negative ``dx``
    and returned a sign-flipped / negative conditional-excess mean.
    The lower endpoint is now widened to ``log_s_u - 20`` in that
    regime so ``v_grid`` stays strictly ascending.
    """

    def test_gevd_mean_excess_stays_non_negative_far_out(self):
        # Fréchet branch (ξ=0.7): ME grows in u, but must never go
        # negative just because S(u) is tiny. At u=500 with σ=1,
        # S(u) ≈ 5e-4 ** (1/0.7) ≈ 1e-5 (below 1e-6 in float32).
        d = GeneralizedExtremeValueDistribution(loc=0.0, scale=1.0, concentration=0.7)
        for u in [100.0, 500.0, 1000.0]:
            me = float(d.conditional_excess_mean(jnp.array(u)))
            assert jnp.isfinite(me), f"ME non-finite at u={u}: {me}"
            assert me > 0.0, f"ME sign-flipped at u={u}: {me}"

    def test_frechet_mean_excess_stays_non_negative_far_out(self):
        d = FrechetType2GEVD(loc=0.0, scale=1.0, concentration=0.5)
        for u in [50.0, 500.0, 5000.0]:
            me = float(d.conditional_excess_mean(jnp.array(u)))
            assert jnp.isfinite(me), f"ME non-finite at u={u}: {me}"
            assert me > 0.0, f"ME sign-flipped at u={u}: {me}"

    def test_weibull_mean_excess_stays_non_negative_near_upper_endpoint(self):
        # Weibull upper bound is μ - σ/ξ = 0 - 1/(-0.3) ≈ 3.333.
        # Just below the endpoint, S(u) gets very small. We stay at
        # fractions where S(u) > 1e-12 so the NaN mask doesn't fire;
        # the bug symptom was a *negative* ME, not NaN.
        d = WeibullType3GEVD(loc=0.0, scale=1.0, concentration=-0.3)
        upper = float(d.upper_bound())
        for frac in [0.9, 0.95, 0.99]:
            u = frac * upper
            me = float(d.conditional_excess_mean(jnp.array(u)))
            assert jnp.isfinite(me), f"ME non-finite at u={u}: {me}"
            assert me >= -1e-6, f"ME sign-flipped at u={u}: {me}"
