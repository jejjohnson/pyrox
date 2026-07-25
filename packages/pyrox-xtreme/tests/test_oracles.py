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
        mean, var, skew = frozen.stats(moments="mvs")
        np.testing.assert_allclose(float(d.mean), mean, rtol=5e-4)
        if xi is not None and 0 < abs(xi) < 1e-3:
            # Known float32 limitation (dead band): for tiny nonzero ξ
            # the exact Γ-difference variance/skew formulas cancel to ~0
            # in float32 — the class switches to the Gumbel closed forms
            # only below GUMBEL_THRESHOLD (1e-7). Pin only finiteness
            # here (the ξ=0 crash regression) and the mean, which uses a
            # cancellation-free expm1∘gammaln form.
            assert np.isfinite(float(d.variance))
        else:
            np.testing.assert_allclose(float(d.variance), var, rtol=1e-3)
            np.testing.assert_allclose(float(d.skew()), skew, rtol=2e-3, atol=1e-5)

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
