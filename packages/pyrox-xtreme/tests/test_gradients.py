"""Gradient-correctness and small-ξ band tests (#73).

Before this module the only gradient coverage was a finiteness check at
one benign point, ξ was never probed inside (1e-7, 1e-4), continuity
across the Gumbel branch switch was tested only at exactly ξ = 0, and
no distribution statistical method was ever called with batched
parameters. Gradients are checked against central finite differences of
``scipy.stats`` (float64), so the reference is independent of the JAX
implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
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
    gev_mean,
    gev_survival,
)
from xtremax.primitives.gpd import (
    gpd_cdf,
    gpd_icdf,
    gpd_log_prob,
    gpd_mean,
    gpd_survival,
)
from xtremax.primitives.spatial import pairwise_distances


LOC, SCALE = 0.0, 1.0
XI_BAND = [-0.3, -1e-6, 0.0, 1e-6, 1e-4, 0.3]
X_POINTS = [-2.0, 0.5, 3.0]


def _fd(f, x0: float, h: float) -> float:
    """Central finite difference in float64."""
    return (f(x0 + h) - f(x0 - h)) / (2.0 * h)


def _gev_ref(x, loc, scale, xi):
    return st.genextreme(-xi, loc=loc, scale=scale)


def _in_support(x, xi, margin=0.25):
    lo, hi = (
        _gev_ref(0.0, LOC, SCALE, xi).support()
        if xi == 0
        else st.genextreme(-xi, loc=LOC, scale=SCALE).support()
    )
    return (x - lo) > margin and (hi - x) > margin


GEV_POINTS = [(x, xi) for x in X_POINTS for xi in XI_BAND if _in_support(x, xi)]


class TestGEVLogProbGradients:
    @pytest.mark.parametrize(("x", "xi"), GEV_POINTS)
    def test_grad_wrt_x_loc_scale(self, x, xi):
        g_x = float(jax.grad(gev_log_prob, argnums=0)(x, LOC, SCALE, xi))
        g_loc = float(jax.grad(gev_log_prob, argnums=1)(x, LOC, SCALE, xi))
        g_scale = float(jax.grad(gev_log_prob, argnums=2)(x, LOC, SCALE, xi))

        fd_x = _fd(
            lambda v: st.genextreme(-xi, loc=LOC, scale=SCALE).logpdf(v), x, 1e-6
        )
        fd_loc = _fd(
            lambda v: st.genextreme(-xi, loc=v, scale=SCALE).logpdf(x), LOC, 1e-6
        )
        fd_scale = _fd(
            lambda v: st.genextreme(-xi, loc=LOC, scale=v).logpdf(x), SCALE, 1e-6
        )
        np.testing.assert_allclose(g_x, fd_x, rtol=2e-3, atol=2e-4)
        np.testing.assert_allclose(g_loc, fd_loc, rtol=2e-3, atol=2e-4)
        np.testing.assert_allclose(g_scale, fd_scale, rtol=2e-3, atol=2e-4)

    @pytest.mark.parametrize(("x", "xi"), GEV_POINTS)
    def test_grad_wrt_shape(self, x, xi):
        """d/dξ against FD of scipy over c = -ξ — nonzero through the
        small-ξ band, including exactly ξ = 0 (pins the dead zone where
        a hard branch switch would zero the gradient)."""
        g_xi = float(jax.grad(gev_log_prob, argnums=3)(x, LOC, SCALE, xi))
        # d/dξ = -d/dc.
        fd_xi = -_fd(
            lambda c: st.genextreme(c, loc=LOC, scale=SCALE).logpdf(x), -xi, 1e-5
        )
        np.testing.assert_allclose(g_xi, fd_xi, rtol=5e-3, atol=5e-4)

    def test_shape_grad_at_zero_is_nonzero(self):
        g = float(jax.grad(gev_log_prob, argnums=3)(0.5, LOC, SCALE, 0.0))
        assert g != 0.0
        fd = -_fd(lambda c: st.genextreme(c).logpdf(0.5), 0.0, 1e-5)
        np.testing.assert_allclose(g, fd, rtol=5e-3)

    def test_deep_tail_weibull_grads_finite(self):
        """x = -90 with ξ = -0.1: value is astronomically small but the
        gradients must stay finite (pins #45)."""
        for argnums in range(4):
            g = float(jax.grad(gev_log_prob, argnums=argnums)(-90.0, LOC, SCALE, -0.1))
            assert np.isfinite(g), f"argnums={argnums}: {g}"


class TestGEVCdfIcdfGradients:
    @pytest.mark.parametrize(("x", "xi"), GEV_POINTS)
    def test_cdf_grads(self, x, xi):
        g_x = float(jax.grad(gev_cdf, argnums=0)(x, LOC, SCALE, xi))
        fd_x = _fd(lambda v: st.genextreme(-xi, loc=LOC, scale=SCALE).cdf(v), x, 1e-6)
        np.testing.assert_allclose(g_x, fd_x, rtol=2e-3, atol=2e-4)

        g_xi = float(jax.grad(gev_cdf, argnums=3)(x, LOC, SCALE, xi))
        fd_xi = -_fd(lambda c: st.genextreme(c, loc=LOC, scale=SCALE).cdf(x), -xi, 1e-5)
        np.testing.assert_allclose(g_xi, fd_xi, rtol=5e-3, atol=5e-4)

    @pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
    @pytest.mark.parametrize("xi", XI_BAND)
    def test_icdf_grads(self, q, xi):
        g_q = float(jax.grad(gev_icdf, argnums=0)(q, LOC, SCALE, xi))
        fd_q = _fd(lambda v: st.genextreme(-xi, loc=LOC, scale=SCALE).ppf(v), q, 1e-6)
        np.testing.assert_allclose(g_q, fd_q, rtol=2e-3, atol=2e-4)

        g_xi = float(jax.grad(gev_icdf, argnums=3)(q, LOC, SCALE, xi))
        fd_xi = -_fd(lambda c: st.genextreme(c, loc=LOC, scale=SCALE).ppf(q), -xi, 1e-5)
        np.testing.assert_allclose(g_xi, fd_xi, rtol=5e-3, atol=5e-4)


GPD_POINTS = [
    (x, xi)
    for x in (0.3, 1.0, 2.5)
    for xi in XI_BAND
    if xi >= 0 or x < SCALE / abs(xi) - 0.25
]


class TestGPDGradients:
    @pytest.mark.parametrize(("x", "xi"), GPD_POINTS)
    def test_log_prob_grads(self, x, xi):
        g_x = float(jax.grad(gpd_log_prob, argnums=0)(x, SCALE, xi))
        fd_x = _fd(lambda v: st.genpareto(xi, scale=SCALE).logpdf(v), x, 1e-6)
        np.testing.assert_allclose(g_x, fd_x, rtol=2e-3, atol=2e-4)

        g_xi = float(jax.grad(gpd_log_prob, argnums=2)(x, SCALE, xi))
        fd_xi = _fd(lambda c: st.genpareto(c, scale=SCALE).logpdf(x), xi, 1e-5)
        np.testing.assert_allclose(g_xi, fd_xi, rtol=5e-3, atol=5e-4)

    @pytest.mark.parametrize("xi", XI_BAND)
    def test_icdf_shape_grad(self, xi):
        q = 0.9
        g_xi = float(jax.grad(gpd_icdf, argnums=2)(q, SCALE, xi))
        fd_xi = _fd(lambda c: st.genpareto(c, scale=SCALE).ppf(q), xi, 1e-5)
        np.testing.assert_allclose(g_xi, fd_xi, rtol=5e-3, atol=5e-4)


class TestGumbelThresholdContinuity:
    """|f(ξ = ±1e-6) − f(ξ = 0)| must be tiny for every quantity —
    both sides of the branch switch (pins #43/#44)."""

    EPS = 1e-6
    TOL = 1e-4

    @pytest.mark.parametrize("side", [-1e-6, 1e-6])
    def test_gev_quantities(self, side):
        x = jnp.array([-1.5, 0.0, 1.0, 3.0])
        q = jnp.array([0.05, 0.5, 0.95])
        for fn, arg in (
            (gev_cdf, x),
            (gev_log_prob, x),
            (gev_survival, x),
            (gev_icdf, q),
        ):
            at_eps = np.asarray(fn(arg, LOC, SCALE, side))
            at_zero = np.asarray(fn(arg, LOC, SCALE, 0.0))
            np.testing.assert_allclose(at_eps, at_zero, atol=self.TOL)
        np.testing.assert_allclose(
            float(gev_mean(LOC, SCALE, side)),
            float(gev_mean(LOC, SCALE, 0.0)),
            atol=self.TOL,
        )

    @pytest.mark.parametrize("side", [-1e-6, 1e-6])
    def test_gpd_quantities(self, side):
        x = jnp.array([0.2, 1.0, 3.0])
        q = jnp.array([0.05, 0.5, 0.95])
        for fn, arg in (
            (gpd_cdf, x),
            (gpd_log_prob, x),
            (gpd_survival, x),
            (gpd_icdf, q),
        ):
            at_eps = np.asarray(fn(arg, SCALE, side))
            at_zero = np.asarray(fn(arg, SCALE, 0.0))
            np.testing.assert_allclose(at_eps, at_zero, atol=self.TOL)
        np.testing.assert_allclose(
            float(gpd_mean(SCALE, side)), float(gpd_mean(SCALE, 0.0)), atol=self.TOL
        )


# ---------------------------------------------------------------------------
# Batched-parameter smoke tests (pins #52's class of crash)
# ---------------------------------------------------------------------------


def _batched_distributions():
    return [
        GeneralizedExtremeValueDistribution(
            loc=jnp.array([0.0, 1.0]),
            scale=jnp.array([1.0, 2.0]),
            concentration=jnp.array([0.1, -0.2]),
        ),
        GeneralizedParetoDistribution(
            scale=jnp.array([1.0, 2.0]), concentration=jnp.array([0.1, -0.2])
        ),
        FrechetType2GEVD(
            loc=jnp.array([0.0, 1.0]),
            scale=jnp.array([1.0, 2.0]),
            concentration=jnp.array([0.2, 0.4]),
        ),
        WeibullType3GEVD(
            loc=jnp.array([0.0, 1.0]),
            scale=jnp.array([1.0, 2.0]),
            concentration=jnp.array([-0.2, -0.4]),
        ),
        GumbelType1GEVD(loc=jnp.array([0.0, 1.0]), scale=jnp.array([1.0, 2.0])),
    ]


NULLARY = [
    "mean",
    "variance",
    "mode",
    "skew",
    "kurtosis",
    "entropy",
    "tail_index",
    "upper_bound",
    "lower_bound",
]
UNARY_X = [
    "log_prob",
    "cdf",
    "survival_function",
    "log_survival_function",
    "hazard_rate",
    "cumulative_hazard_rate",
    "exceedance_probability",
    "conditional_excess_mean",
]


def _evaluate(d, name, *args):
    attr = getattr(d, name)
    return attr(*args) if callable(attr) else attr


class TestBatchedParameters:
    """Every statistical method must broadcast over batched parameters
    instead of crashing or silently collapsing the batch axis."""

    @pytest.mark.parametrize(
        "d", _batched_distributions(), ids=lambda d: type(d).__name__
    )
    def test_nullary_methods_broadcast(self, d):
        for name in NULLARY:
            if not hasattr(d, name):
                continue
            out = np.asarray(_evaluate(d, name))
            assert out.shape == (2,), f"{type(d).__name__}.{name}: {out.shape}"

    @pytest.mark.parametrize(
        "d", _batched_distributions(), ids=lambda d: type(d).__name__
    )
    def test_unary_methods_broadcast(self, d):
        # 1.6 sits inside every batch row's support for the params above.
        x = jnp.asarray(1.6)
        for name in UNARY_X:
            if not hasattr(d, name):
                continue
            out = np.asarray(_evaluate(d, name, x))
            assert out.shape == (2,), f"{type(d).__name__}.{name}: {out.shape}"
            assert np.all(np.isfinite(out)), f"{type(d).__name__}.{name}: {out}"

    @pytest.mark.parametrize(
        "d", _batched_distributions(), ids=lambda d: type(d).__name__
    )
    def test_icdf_and_return_level_broadcast(self, d):
        out_q = np.asarray(d.icdf(jnp.asarray(0.9)))
        assert out_q.shape == (2,)
        assert np.all(np.isfinite(out_q))
        out_rl = np.asarray(d.return_level(50.0))
        assert out_rl.shape == (2,)
        assert np.all(np.isfinite(out_rl))

    @pytest.mark.parametrize(
        "d", _batched_distributions(), ids=lambda d: type(d).__name__
    )
    def test_sample_appends_batch_shape(self, d):
        s = np.asarray(d.sample(jax.random.PRNGKey(0), (5,)))
        assert s.shape == (5, 2)


class TestPairwiseDistancesGradient:
    def test_grad_matches_fd(self):
        """FD-vs-grad for the masked-sqrt construction (pins #48): the
        gradient must match finite differences of the summed distances,
        not just be finite."""
        coords = jnp.array([[0.0, 0.0], [1.0, 0.5], [0.3, 2.0]])

        def total(c):
            return jnp.sum(pairwise_distances(c))

        g = np.asarray(jax.grad(total)(coords))
        h = 1e-3
        fd = np.zeros_like(np.asarray(coords))
        base = np.asarray(coords)
        for i in range(base.shape[0]):
            for j in range(base.shape[1]):
                up = base.copy()
                dn = base.copy()
                up[i, j] += h
                dn[i, j] -= h
                fd[i, j] = (
                    float(total(jnp.asarray(up))) - float(total(jnp.asarray(dn)))
                ) / (2 * h)
        np.testing.assert_allclose(g, fd, rtol=2e-3, atol=2e-3)
