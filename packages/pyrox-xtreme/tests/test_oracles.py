"""External-oracle tests: every EVT quantity pinned to scipy.stats (#72).

The pre-existing "parity" tests compare the class layer against the
primitives — the same formulas — so a systematic parameterization or
sign error passes everything. These tests pin values to an independent
implementation instead.

Sign conventions:

* ``scipy.stats.genextreme(c=-xi)``  (scipy flips the GEV shape sign)
* ``scipy.stats.genpareto(c=xi)``    (same sign, loc = 0 here)
* ``scipy.stats.gumbel_r``
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import jax
import numpy as np
import pytest
import scipy.stats as st

from xtremax.distributions import (
    FrechetType2GEVD,
    GeneralizedExtremeValueDistribution,
    GeneralizedParetoDistribution,
    GumbelType1GEVD,
    WeibullType3GEVD,
)
from xtremax.primitives.gev import (
    gev_cdf,
    gev_icdf,
    gev_log_prob,
    gev_log_survival,
    gev_mean,
    gev_return_level,
    gev_survival,
)
from xtremax.primitives.gpd import (
    gpd_cdf,
    gpd_icdf,
    gpd_log_prob,
    gpd_log_survival,
    gpd_mean,
    gpd_return_level,
    gpd_survival,
)
from xtremax.primitives.gumbel import (
    gumbel_cdf,
    gumbel_icdf,
    gumbel_log_prob,
    gumbel_mean,
    gumbel_return_level,
)


# Non-unit loc/scale everywhere so parameterization slips can't hide.
LOC, SCALE = 1.5, 2.0
# ξ grid straddling the Gumbel threshold, including the (1e-7, 1e-4)
# band where the smooth-kernel numerics live.
XI_GRID = [-0.5, -0.2, -1e-6, 0.0, 1e-6, 1e-4, 0.2, 0.9]
Q_GRID = np.array([0.01, 0.1, 0.5, 0.9, 0.99])


def _gev_frozen(xi):
    return st.genextreme(-xi, loc=LOC, scale=SCALE)


def _gev_x_grid(xi, margin=0.05, n=9):
    """Interior x values for this ξ's support."""
    lo, hi = _gev_frozen(xi).support()
    return np.linspace(max(lo, -6.0) + margin, min(hi, 9.0) - margin, n)


def _gpd_frozen(xi):
    return st.genpareto(xi, loc=0.0, scale=SCALE)


def _gpd_x_grid(xi, margin=0.05, n=9):
    lo, hi = _gpd_frozen(xi).support()
    return np.linspace(lo + margin, min(hi, 9.0) - margin, n)


class TestGEVPrimitiveOracles:
    @pytest.mark.parametrize("xi", XI_GRID)
    def test_cdf(self, xi):
        x = _gev_x_grid(xi)
        got = np.asarray(gev_cdf(x, LOC, SCALE, xi))
        np.testing.assert_allclose(got, _gev_frozen(xi).cdf(x), rtol=2e-4, atol=1e-6)

    @pytest.mark.parametrize("xi", XI_GRID)
    def test_log_prob(self, xi):
        x = _gev_x_grid(xi)
        got = np.asarray(gev_log_prob(x, LOC, SCALE, xi))
        np.testing.assert_allclose(got, _gev_frozen(xi).logpdf(x), rtol=2e-4, atol=2e-3)

    @pytest.mark.parametrize("xi", XI_GRID)
    def test_icdf(self, xi):
        got = np.asarray(gev_icdf(Q_GRID, LOC, SCALE, xi))
        np.testing.assert_allclose(
            got, _gev_frozen(xi).ppf(Q_GRID), rtol=2e-4, atol=1e-4
        )

    @pytest.mark.parametrize("xi", XI_GRID)
    def test_survival(self, xi):
        x = _gev_x_grid(xi)
        got = np.asarray(gev_survival(x, LOC, SCALE, xi))
        np.testing.assert_allclose(got, _gev_frozen(xi).sf(x), rtol=5e-4, atol=1e-6)

    @pytest.mark.parametrize("xi", XI_GRID)
    def test_log_survival(self, xi):
        x = _gev_x_grid(xi)
        got = np.asarray(gev_log_survival(x, LOC, SCALE, xi))
        np.testing.assert_allclose(got, _gev_frozen(xi).logsf(x), rtol=5e-4, atol=2e-3)

    @pytest.mark.parametrize("xi", XI_GRID)
    def test_mean(self, xi):
        got = float(gev_mean(LOC, SCALE, xi))
        np.testing.assert_allclose(got, _gev_frozen(xi).mean(), rtol=2e-4)

    @pytest.mark.parametrize("xi", XI_GRID)
    @pytest.mark.parametrize("period", [10.0, 100.0, 1e4])
    def test_return_level(self, xi, period):
        got = float(gev_return_level(period, LOC, SCALE, xi))
        ref = _gev_frozen(xi).ppf(1.0 - 1.0 / period)
        np.testing.assert_allclose(got, ref, rtol=5e-4)


class TestGPDPrimitiveOracles:
    @pytest.mark.parametrize("xi", XI_GRID)
    def test_cdf(self, xi):
        x = _gpd_x_grid(xi)
        got = np.asarray(gpd_cdf(x, SCALE, xi))
        np.testing.assert_allclose(got, _gpd_frozen(xi).cdf(x), rtol=2e-4, atol=1e-6)

    @pytest.mark.parametrize("xi", XI_GRID)
    def test_log_prob(self, xi):
        x = _gpd_x_grid(xi)
        got = np.asarray(gpd_log_prob(x, SCALE, xi))
        np.testing.assert_allclose(got, _gpd_frozen(xi).logpdf(x), rtol=2e-4, atol=2e-3)

    @pytest.mark.parametrize("xi", XI_GRID)
    def test_icdf(self, xi):
        got = np.asarray(gpd_icdf(Q_GRID, SCALE, xi))
        np.testing.assert_allclose(
            got, _gpd_frozen(xi).ppf(Q_GRID), rtol=2e-4, atol=1e-4
        )

    @pytest.mark.parametrize("xi", XI_GRID)
    def test_survival_and_log_survival(self, xi):
        x = _gpd_x_grid(xi)
        frozen = _gpd_frozen(xi)
        np.testing.assert_allclose(
            np.asarray(gpd_survival(x, SCALE, xi)),
            frozen.sf(x),
            rtol=5e-4,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(gpd_log_survival(x, SCALE, xi)),
            frozen.logsf(x),
            rtol=5e-4,
            atol=2e-3,
        )

    @pytest.mark.parametrize("xi", [-0.5, -0.2, 0.0, 1e-4, 0.2, 0.9])
    def test_mean(self, xi):
        got = float(gpd_mean(SCALE, xi))
        np.testing.assert_allclose(got, _gpd_frozen(xi).mean(), rtol=2e-4)

    @pytest.mark.parametrize("xi", XI_GRID)
    @pytest.mark.parametrize("period", [10.0, 100.0, 1e4])
    def test_return_level(self, xi, period):
        """First tests for gpd_return_level (never previously imported
        by the suite)."""
        got = float(gpd_return_level(period, SCALE, xi))
        ref = _gpd_frozen(xi).ppf(1.0 - 1.0 / period)
        np.testing.assert_allclose(got, ref, rtol=5e-4)


class TestGumbelPrimitiveOracles:
    def test_all_quantities(self):
        frozen = st.gumbel_r(loc=LOC, scale=SCALE)
        x = np.linspace(-4.0, 9.0, 11)
        np.testing.assert_allclose(
            np.asarray(gumbel_cdf(x, LOC, SCALE)), frozen.cdf(x), rtol=2e-4, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(gumbel_log_prob(x, LOC, SCALE)),
            frozen.logpdf(x),
            rtol=2e-4,
            atol=2e-3,
        )
        np.testing.assert_allclose(
            np.asarray(gumbel_icdf(Q_GRID, LOC, SCALE)),
            frozen.ppf(Q_GRID),
            rtol=2e-4,
        )
        np.testing.assert_allclose(
            float(gumbel_mean(LOC, SCALE)), frozen.mean(), rtol=2e-4
        )
        np.testing.assert_allclose(
            float(gumbel_return_level(100.0, LOC, SCALE)),
            frozen.ppf(0.99),
            rtol=2e-4,
        )


# ---------------------------------------------------------------------------
# Class layer
# ---------------------------------------------------------------------------


def _class_and_frozen(family, xi):
    """Build (xtremax distribution, scipy frozen) pairs per family."""
    if family == "gev":
        return (
            GeneralizedExtremeValueDistribution(LOC, SCALE, concentration=xi),
            st.genextreme(-xi, loc=LOC, scale=SCALE),
        )
    if family == "gpd":
        return (
            GeneralizedParetoDistribution(SCALE, concentration=xi),
            st.genpareto(xi, loc=0.0, scale=SCALE),
        )
    if family == "frechet":
        return (
            FrechetType2GEVD(LOC, SCALE, concentration=xi),
            st.genextreme(-xi, loc=LOC, scale=SCALE),
        )
    if family == "weibull":
        return (
            WeibullType3GEVD(LOC, SCALE, concentration=xi),
            st.genextreme(-xi, loc=LOC, scale=SCALE),
        )
    if family == "gumbel":
        return GumbelType1GEVD(LOC, SCALE), st.gumbel_r(loc=LOC, scale=SCALE)
    raise ValueError(family)


CLASS_CASES = [
    ("gev", -0.3),
    ("gev", 0.0),
    ("gev", 1e-5),
    ("gev", 0.2),
    ("gpd", -0.3),
    ("gpd", 0.2),
    ("frechet", 0.2),
    ("frechet", 0.5),
    ("weibull", -0.3),
    ("weibull", -0.5),
    ("gumbel", None),
]


def _interior_x(frozen, margin=0.05, n=7):
    lo, hi = frozen.support()
    return np.linspace(max(lo, -6.0) + margin, min(hi, 9.0) - margin, n)


class TestClassOracles:
    @pytest.mark.parametrize(("family", "xi"), CLASS_CASES)
    def test_cdf_log_prob_icdf(self, family, xi):
        d, frozen = _class_and_frozen(family, xi)
        x = _interior_x(frozen)
        np.testing.assert_allclose(
            np.asarray(d.cdf(x)), frozen.cdf(x), rtol=2e-4, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(d.log_prob(x)), frozen.logpdf(x), rtol=2e-4, atol=2e-3
        )
        np.testing.assert_allclose(
            np.asarray(d.icdf(Q_GRID)), frozen.ppf(Q_GRID), rtol=2e-4, atol=1e-4
        )

    @pytest.mark.parametrize(("family", "xi"), CLASS_CASES)
    def test_survival_hazard(self, family, xi):
        d, frozen = _class_and_frozen(family, xi)
        x = _interior_x(frozen)
        np.testing.assert_allclose(
            np.asarray(d.survival_function(x)), frozen.sf(x), rtol=5e-4, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(d.hazard_rate(x)),
            frozen.pdf(x) / frozen.sf(x),
            rtol=1e-3,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(d.cumulative_hazard_rate(x)),
            -frozen.logsf(x),
            rtol=1e-3,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            np.asarray(d.exceedance_probability(x)),
            frozen.sf(x),
            rtol=5e-4,
            atol=1e-6,
        )

    @pytest.mark.parametrize(("family", "xi"), CLASS_CASES)
    def test_moments(self, family, xi):
        d, frozen = _class_and_frozen(family, xi)
        mean, var, skew, kurt = frozen.stats(moments="mvsk")
        np.testing.assert_allclose(float(d.mean), mean, rtol=5e-4)
        np.testing.assert_allclose(float(d.variance), var, rtol=1e-3)
        np.testing.assert_allclose(float(d.skew()), skew, rtol=2e-3, atol=1e-5)
        # scipy's own genextreme moments degrade inside |ξ| ≲ 1e-3 (its
        # dead band, ~1e-3 relative on the excess kurtosis), so the
        # standardized moments are pinned loosely here and exactly
        # against mpmath in TestSmallShapeMoments.
        np.testing.assert_allclose(float(d.kurtosis()), kurt, rtol=5e-3, atol=1e-3)

    @pytest.mark.parametrize(("family", "xi"), CLASS_CASES)
    def test_return_level(self, family, xi):
        d, frozen = _class_and_frozen(family, xi)
        for period in (10.0, 100.0, 1e4):
            np.testing.assert_allclose(
                float(d.return_level(period)),
                frozen.ppf(1.0 - 1.0 / period),
                rtol=5e-4,
            )

    def test_gpd_conditional_excess_mean_closed_form(self):
        """Mean-excess oracle: e(u) = (σ + ξu) / (1 - ξ) for the GPD."""
        xi, u = 0.2, np.array([0.5, 1.0, 3.0])
        d = GeneralizedParetoDistribution(SCALE, concentration=xi)
        ref = (SCALE + xi * u) / (1.0 - xi)
        np.testing.assert_allclose(
            np.asarray(d.conditional_excess_mean(u)), ref, rtol=1e-3
        )

    def test_validate_args_rejects_bad_scale(self):
        with pytest.raises(ValueError):
            GeneralizedExtremeValueDistribution(
                LOC, -1.0, concentration=0.1, validate_args=True
            )
        with pytest.raises(ValueError):
            GeneralizedParetoDistribution(-1.0, concentration=0.1, validate_args=True)


# ---------------------------------------------------------------------------
# Statistical validation of sample()
# ---------------------------------------------------------------------------


SAMPLE_CASES = [
    ("gev", 0.2),
    ("gpd", 0.2),
    ("frechet", 0.3),  # ξ < 1/2 so the variance oracle is finite
    ("weibull", -0.3),
    ("gumbel", None),
]


class TestSampleStatistics:
    @pytest.mark.parametrize(("family", "xi"), SAMPLE_CASES)
    def test_moments_of_draws(self, family, xi):
        d, frozen = _class_and_frozen(family, xi)
        n = 100_000
        samples = np.asarray(d.sample(jax.random.PRNGKey(0), (n,)))
        mean, var = frozen.stats(moments="mv")
        # 3-sigma-ish Monte-Carlo bands with fixed seed (deterministic).
        np.testing.assert_allclose(samples.mean(), mean, atol=4.0 * np.sqrt(var / n))
        np.testing.assert_allclose(samples.var(), var, rtol=0.05)

    @pytest.mark.parametrize(("family", "xi"), SAMPLE_CASES)
    def test_ks_against_scipy_cdf(self, family, xi):
        d, frozen = _class_and_frozen(family, xi)
        samples = np.asarray(d.sample(jax.random.PRNGKey(1), (20_000,)))
        result = st.kstest(samples, frozen.cdf)
        # float32 quantile granularity costs a little KS resolution; a
        # parameterization/sign error costs orders of magnitude more.
        assert result.pvalue > 1e-4, (
            f"{family}: KS p-value {result.pvalue:.2e} (stat {result.statistic:.4f})"
        )


# ---------------------------------------------------------------------------
# Extreme-quantile round-trips
# ---------------------------------------------------------------------------


ROUNDTRIP_Q = [1e-4, 0.5, 1.0 - 1e-4]


class TestQuantileRoundTrips:
    @pytest.mark.parametrize("xi", [-0.3, 0.0, 1e-5, 0.3])
    @pytest.mark.parametrize("q", ROUNDTRIP_Q)
    def test_gev_float32(self, xi, q):
        x = gev_icdf(q, LOC, SCALE, xi)
        assert np.isfinite(float(x))
        q_back = float(gev_cdf(x, LOC, SCALE, xi))
        np.testing.assert_allclose(q_back, q, rtol=5e-3, atol=5e-6)

    @pytest.mark.parametrize("xi", [-0.3, 0.0, 1e-5, 0.3])
    @pytest.mark.parametrize("q", ROUNDTRIP_Q)
    def test_gpd_float32(self, xi, q):
        x = gpd_icdf(q, SCALE, xi)
        assert np.isfinite(float(x))
        q_back = float(gpd_cdf(x, SCALE, xi))
        np.testing.assert_allclose(q_back, q, rtol=5e-3, atol=5e-6)

    @pytest.mark.parametrize("xi", [-0.3, 0.0, 0.3])
    def test_gev_icdf_of_cdf_direction(self, xi):
        """icdf ∘ cdf ≈ identity at interior points (the other
        direction of the round-trip)."""
        x = np.array([-1.0, 0.5, 2.0])
        lo, hi = _gev_frozen(xi).support()
        x = np.clip(x, lo + 0.1, hi - 0.1)
        x_back = np.asarray(gev_icdf(gev_cdf(x, LOC, SCALE, xi), LOC, SCALE, xi))
        np.testing.assert_allclose(x_back, x, rtol=5e-4, atol=5e-4)


class TestFloat64Isolation:
    """Float64 checks run in a subprocess so the x64 flag never mutates
    this session's global JAX config (#72: the old in-process
    ``config.update("jax_enable_x64", ...)`` pattern is flaky under
    parallel runners and leaks JIT-cache state)."""

    @pytest.mark.integration
    def test_float64_roundtrips_and_deep_tail(self):
        script = textwrap.dedent(
            """
            import numpy as np
            import scipy.stats as st
            from xtremax.primitives.gev import gev_cdf, gev_icdf, gev_survival
            from xtremax.primitives.gpd import gpd_cdf, gpd_icdf

            LOC, SCALE = 1.5, 2.0
            for xi in (-0.3, 0.0, 1e-5, 0.3):
                for q in (1e-4, 0.5, 1.0 - 1e-4):
                    tol = 1e-9 * max(q, 1e-4) + 1e-12
                    x = gev_icdf(np.float64(q), LOC, SCALE, np.float64(xi))
                    q_back = float(gev_cdf(x, LOC, SCALE, np.float64(xi)))
                    assert abs(q_back - q) < tol
                    ref = st.genextreme(-xi, loc=LOC, scale=SCALE).ppf(q)
                    assert np.isclose(float(x), ref, rtol=1e-8), (xi, q, float(x), ref)
                    xp = gpd_icdf(np.float64(q), SCALE, np.float64(xi))
                    qp_back = float(gpd_cdf(xp, SCALE, np.float64(xi)))
                    assert abs(qp_back - q) < tol

            # Deep-tail survival keeps precision where 1 - cdf cancels to 0.
            s = float(gev_survival(np.float64(40.0), 0.0, 1.0, np.float64(0.0)))
            assert s > 0.0 and np.isclose(s, np.exp(-40.0), rtol=1e-6)
            naive = 1.0 - float(gev_cdf(np.float64(40.0), 0.0, 1.0, np.float64(0.0)))
            assert naive == 0.0
            print("OK")
            """
        )
        env = dict(os.environ, JAX_ENABLE_X64="1")
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout


# (ξ, Var, skew, excess kurtosis) at LOC/SCALE, computed to 40 decimal
# digits with mpmath from the exact Γ-difference formulas and rounded to
# 12 significant figures. mpmath rather than scipy because
# ``scipy.stats.genextreme`` has a dead band of its own — inside
# |ξ| ≲ 1e-3 its moments drift by up to ~4e-4 (skew) and ~1e-3 (excess
# kurtosis), which is the very region this table has to pin. ξ = 0 uses
# the exact Gumbel limits σ²π²/6, 12√6 ζ(3)/π³ and 12/5.
GEV_MOMENT_GOLDEN = (
    (-1.0, 4.0, -2.0, 6.0),
    (-0.5, 3.43362938564, -0.631110657819, 0.245089300688),
    (-0.3, 3.9138532693, -0.068742099421, -0.289399162108),
    (-0.24, 4.19003133275, 0.119634097592, -0.234039348657),
    (-0.2, 4.42299779831, 0.254109603707, -0.119709936218),
    (-0.15, 4.78316749944, 0.43574332954, 0.137656966413),
    (-0.1, 5.24018202939, 0.637637133903, 0.570166483567),
    (-0.05, 5.82434367673, 0.867965095175, 1.26720075926),
    (-0.01, 6.41219664801, 1.08107375981, 2.12544588659),
    (-0.003, 6.52851835404, 1.12175681071, 2.31460271537),
    (-0.001, 6.56257072594, 1.13359273066, 2.37123426411),
    (-0.0001, 6.57801550513, 1.13895056093, 2.39710975666),
    (-1e-05, 6.57956414899, 1.13948743451, 2.39971083838),
    (-1e-06, 6.57971905513, 1.1395411328, 2.39997108246),
    (-1e-08, 6.57973609527, 1.13954703974, 2.39999971158),
    (0.0, 6.57973626739, 1.1395470994, 2.4),
    (1e-08, 6.57973643952, 1.13954715907, 2.40000029032),
    (1e-06, 6.57975347975, 1.13955306603, 2.40002891784),
    (1e-05, 6.57990839517, 1.13960676676, 2.40028919215),
    (0.0001, 6.58145796712, 1.14014388348, 2.40289329566),
    (0.001, 6.59699555602, 1.14552602793, 2.42907097371),
    (0.003, 6.63179793854, 1.15755844145, 2.48814483951),
    (0.01, 6.75665517419, 1.20047851976, 2.70513497837),
    (0.05, 7.57241072454, 1.4738841313, 4.33349431517),
    (0.1, 8.90496429283, 1.91033913417, 7.97856623935),
    (0.15, 10.7440475863, 2.53024989387, 16.2741593308),
    (0.2, 13.3761422492, 3.53507160462, 45.0915121258),
    (0.24, 16.3949134617, 5.02446249514, 309.608411304),
)


class TestSmallShapeMoments:
    """#88 — the Γ-difference moment formulas cancelled to noise for
    small ξ. In float32 the variance evaluated to exactly ``0`` at
    ξ = 1e-5 (true value 6.58) and the skew/kurtosis to ``nan`` / ~1e6;
    even in float64 the excess kurtosis came out as 3e4 instead of 2.4.
    The Gumbel closed forms only took over below ``GUMBEL_THRESHOLD``
    (1e-7), leaving everything between there and ξ ≈ 0.1 garbage.
    """

    @pytest.mark.parametrize(("xi", "var", "skew", "kurt"), GEV_MOMENT_GOLDEN)
    def test_gev_moments_match_mpmath(self, xi, var, skew, kurt):
        d = GeneralizedExtremeValueDistribution(LOC, SCALE, concentration=xi)
        np.testing.assert_allclose(float(d.variance), var, rtol=5e-4)
        np.testing.assert_allclose(float(d.skew()), skew, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(float(d.kurtosis()), kurt, rtol=1e-2, atol=5e-3)

    @pytest.mark.parametrize(("xi", "var", "skew", "kurt"), GEV_MOMENT_GOLDEN)
    def test_frechet_weibull_share_the_gev_moments(self, xi, var, skew, kurt):
        """The ξ > 0 / ξ < 0 subclasses carried their own copies of the
        same cancelling formulas — Fréchet's ``concentration`` constraint
        is plain positivity, so ξ = 1e-5 was reachable there too.
        """
        if xi == 0.0:
            pytest.skip("ξ = 0 is in neither the Fréchet nor Weibull domain")
        cls = FrechetType2GEVD if xi > 0 else WeibullType3GEVD
        d = cls(LOC, SCALE, concentration=xi)
        np.testing.assert_allclose(float(d.variance), var, rtol=5e-4)
        np.testing.assert_allclose(float(d.skew()), skew, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(float(d.kurtosis()), kurt, rtol=1e-2, atol=5e-3)

    def test_variance_never_collapses_across_the_old_dead_band(self):
        """The failure was a *collapse*, not drift: the Γ difference
        rounded to 0 (or negative) in float32. Sweep the band densely and
        require the variance to stay near the Gumbel value throughout.
        """
        xi = np.concatenate([np.logspace(-8, -2, 200), -np.logspace(-8, -2, 200)])
        got = np.asarray(
            GeneralizedExtremeValueDistribution(
                LOC, SCALE, concentration=xi.astype(np.float32)
            ).variance
        )
        gumbel_var = SCALE**2 * np.pi**2 / 6.0
        assert np.all(np.isfinite(got))
        # |ξ| ≤ 1e-2 moves the true variance by at most ~3% off the limit.
        np.testing.assert_allclose(got, gumbel_var, rtol=0.03)

    def test_integral_parameters_are_promoted_before_arithmetic(self):
        """``scale=50000`` and ``concentration=0`` are legal arguments and
        must not stay integral: an int32 ``scale ** 2`` wraps round to a
        *negative* variance, and an integral concentration has no floating
        epsilon to pick the series crossover from.
        """
        got = GeneralizedExtremeValueDistribution(0, 50000, concentration=0)
        ref = GeneralizedExtremeValueDistribution(0.0, 50000.0, concentration=0.0)
        assert float(got.variance) > 0.0
        assert float(got.variance) == float(ref.variance)
        assert float(got.skew()) == float(ref.skew())
        assert float(got.kurtosis()) == float(ref.kurtosis())

    def test_series_coefficients_outlive_narrow_input_dtypes(self):
        """The Taylor tables reach ~1e18. Narrowed to the input dtype they
        overflow anything below float32, and Horner then returns 0·inf =
        nan at exactly the ξ = 0 limit the series exists to cover.
        """
        xi = jax.numpy.asarray(0.0, dtype=jax.numpy.float16)
        d = GeneralizedExtremeValueDistribution(LOC, SCALE, concentration=xi)
        np.testing.assert_allclose(float(d.skew()), 1.1395470994046486, rtol=1e-3)
        np.testing.assert_allclose(float(d.kurtosis()), 2.4, rtol=1e-3)

    def test_moment_gradients_stay_finite_through_zero(self):
        """The series branch also has to be differentiable: a ``0/0``
        reaching autodiff yields a nan gradient even where the value
        looks right. Forward mode and the second derivative are checked
        too — the inactive exact arm is evaluated at a fixed sentinel ξ,
        and picking one that sits on a ``Γ(1 - kξ)`` pole would leave the
        masking of its nan cotangent to do all the work.
        """

        def gev(shape):
            return GeneralizedExtremeValueDistribution(LOC, SCALE, concentration=shape)

        moments = (
            lambda s: gev(s).variance,
            lambda s: gev(s).skew(),
            lambda s: gev(s).kurtosis(),
        )
        one = np.float32(1.0)
        for xi in (0.0, 1e-8, 1e-5, 1e-2, 0.15, -0.15, -0.5):
            x = np.float32(xi)
            for moment in moments:
                d1 = float(jax.grad(moment)(x))
                fwd = float(jax.jvp(moment, (x,), (one,))[1])
                d2 = float(jax.grad(jax.grad(moment))(x))
                assert np.isfinite(d1) and np.isfinite(fwd) and np.isfinite(d2), (
                    xi,
                    d1,
                    fwd,
                    d2,
                )
                np.testing.assert_allclose(fwd, d1, rtol=1e-4, atol=1e-5)
