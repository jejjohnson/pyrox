"""Tests for non-stationary GEV return-level and spatial primitives.

Covers the stationary/non-stationary consistency of return levels and
periods, the round-trip ``return_level`` ↔ ``return_period`` identity, the
parameter-field assembly, and the spatial helpers (distance matrix, design
matrix, two-range copula correlation), including ``jax.jit`` / ``jax.vmap``
safety.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtremax import (
    assemble_nonstationary_gev_fields,
    design_matrix,
    expected_exceedances,
    gev_cdf,
    gev_log_survival,
    gev_return_level,
    gev_survival,
    nonstationary_return_level,
    nonstationary_return_period,
    pairwise_distances,
    two_range_correlation,
)


class TestAssembleFields:
    def test_shapes_scalar_covariate(self):
        n_sites, n_blocks = 4, 6
        mu0 = jnp.linspace(10.0, 13.0, n_sites)
        mu1 = jnp.full((n_sites,), 0.5)
        log_sigma = jnp.zeros(n_sites)
        time = jnp.linspace(0.0, 1.0, n_blocks)

        loc, scale, shape = assemble_nonstationary_gev_fields(
            mu0, mu1, log_sigma, -0.1, time
        )
        assert loc.shape == (n_blocks, n_sites)
        assert scale.shape == (n_blocks, n_sites)
        assert shape.shape == (n_blocks, n_sites)

    def test_location_trend(self):
        """loc(t, s) = mu0(s) + x_t * mu1(s)."""
        mu0 = jnp.array([10.0, 20.0])
        mu1 = jnp.array([1.0, 2.0])
        time = jnp.array([0.0, 1.0, 2.0])
        loc, scale, _ = assemble_nonstationary_gev_fields(
            mu0, mu1, jnp.log(jnp.array([2.0, 3.0])), 0.0, time
        )
        expected = mu0[None, :] + time[:, None] * mu1[None, :]
        assert jnp.allclose(loc, expected)
        # Scale is exp(log_sigma), constant over blocks.
        assert jnp.allclose(scale[0], jnp.array([2.0, 3.0]))
        assert jnp.allclose(scale[0], scale[-1])

    def test_multi_covariate(self):
        n_sites, n_blocks, n_cov = 3, 5, 2
        mu0 = jnp.zeros(n_sites)
        mu1 = jnp.ones((n_sites, n_cov))
        time = jnp.ones((n_blocks, n_cov))
        loc, _, _ = assemble_nonstationary_gev_fields(
            mu0, mu1, jnp.zeros(n_sites), 0.0, time
        )
        # Each block sums the two covariate contributions: 1*1 + 1*1 = 2.
        assert jnp.allclose(loc, 2.0)


class TestSurvival:
    @pytest.mark.slow
    def test_matches_one_minus_cdf_in_bulk(self):
        """In the bulk, S(x) and 1 - F(x) agree to high precision."""
        x = jnp.linspace(-1.0, 4.0, 25)
        for shape in (-0.2, 0.0, 0.3):
            s = gev_survival(x, 0.0, 1.0, shape)
            ref = 1.0 - gev_cdf(x, 0.0, 1.0, shape)
            assert jnp.allclose(s, ref, atol=1e-5)

    def test_log_survival_consistent(self):
        x = jnp.linspace(-1.0, 4.0, 25)
        s = gev_survival(x, 0.0, 1.0, -0.1)
        log_s = gev_log_survival(x, 0.0, 1.0, -0.1)
        assert jnp.allclose(jnp.exp(log_s), s, atol=1e-6)

    @pytest.mark.slow
    def test_deep_tail_accuracy(self):
        """`1 - cdf` cancels to 0 in the deep tail; `gev_survival` does not.

        Float32 in-process check. The float64 twin runs in a subprocess
        (``tests/test_oracles.py::TestFloat64Isolation``) so the global
        ``jax_enable_x64`` flag is never mutated mid-session — the old
        in-process ``config.update`` pattern was flaky under parallel
        runners and leaked JIT-cache state (#72).
        """
        x = jnp.asarray(20.0)  # Gumbel, S ~ e^-20 ~ 2e-9
        s = gev_survival(x, 0.0, 1.0, 0.0)
        naive = 1.0 - gev_cdf(x, 0.0, 1.0, 0.0)
        assert float(s) > 0.0
        assert jnp.allclose(s, jnp.exp(-20.0), rtol=1e-5)
        # The naive form has lost all precision (rounds to exactly 0).
        assert float(naive) == 0.0

    def test_out_of_support_boundaries(self):
        # Fréchet (ξ>0): below lower endpoint S = 1.
        s_lo = gev_survival(-100.0, 0.0, 1.0, 0.5)
        assert jnp.allclose(s_lo, 1.0)
        # Weibull (ξ<0): above upper endpoint μ - σ/ξ = 5.0, S = 0.
        s_hi = gev_survival(100.0, 0.0, 1.0, -0.2)
        assert jnp.allclose(s_hi, 0.0)


class TestStationaryConsistency:
    @pytest.mark.slow
    def test_return_level_matches_closed_form(self):
        """time_axis=None must equal the closed-form GEV return level."""
        periods = jnp.array([2.0, 10.0, 100.0])
        rl = nonstationary_return_level(periods, 10.0, 2.0, -0.1)
        ref = gev_return_level(periods, 10.0, 2.0, -0.1)
        assert jnp.allclose(rl, ref)

    def test_return_period_inverts_return_level(self):
        period = 50.0
        z = nonstationary_return_level(period, 5.0, 1.5, 0.1)
        recovered = nonstationary_return_period(z, 5.0, 1.5, 0.1)
        assert jnp.allclose(recovered, period, rtol=1e-4)


class TestNonStationary:
    def _fields(self):
        # Warming trend: location drifts upward across 30 blocks.
        n_blocks, n_sites = 30, 4
        time = jnp.linspace(0.0, 1.0, n_blocks)
        mu0 = jnp.linspace(10.0, 13.0, n_sites)
        mu1 = jnp.full((n_sites,), 2.0)
        return assemble_nonstationary_gev_fields(
            mu0, mu1, jnp.zeros(n_sites), -0.15, time
        )

    @pytest.mark.slow
    def test_expected_exceedances_decreasing(self):
        loc, scale, shape = self._fields()
        low = expected_exceedances(11.0, loc, scale, shape, time_axis=0)
        high = expected_exceedances(20.0, loc, scale, shape, time_axis=0)
        # Higher threshold -> fewer expected exceedances.
        assert jnp.all(high < low)

    @pytest.mark.slow
    def test_return_level_solves_exceedance_target(self):
        loc, scale, shape = self._fields()
        period = 20.0
        z = nonstationary_return_level(period, loc, scale, shape, time_axis=0)
        n_blocks = loc.shape[0]
        # Threshold is shaped like the non-time dims; the block axis is inserted.
        got = expected_exceedances(z, loc, scale, shape, time_axis=0)
        # By definition sum_t P(Y_t > z) = N_blocks / T.
        assert jnp.allclose(got, n_blocks / period, atol=1e-3)

    @pytest.mark.slow
    def test_return_period_roundtrip(self):
        loc, scale, shape = self._fields()
        period = 25.0
        z = nonstationary_return_level(period, loc, scale, shape, time_axis=0)
        recovered = nonstationary_return_period(z, loc, scale, shape, time_axis=0)
        assert jnp.allclose(recovered, period, rtol=1e-3)

    @pytest.mark.slow
    def test_non_leading_time_axis(self):
        """time_axis need not be leading: (n_sites, n_blocks) with time_axis=1
        must match the transposed (n_blocks, n_sites) / time_axis=0 result."""
        loc0, scale0, shape0 = self._fields()  # (n_blocks, n_sites)
        loc1, scale1, shape1 = loc0.T, scale0.T, shape0.T  # (n_sites, n_blocks)
        period = 20.0

        z0 = nonstationary_return_level(period, loc0, scale0, shape0, time_axis=0)
        z1 = nonstationary_return_level(period, loc1, scale1, shape1, time_axis=1)
        assert z1.shape == (loc0.shape[1],)
        assert jnp.allclose(z0, z1, rtol=1e-4)

        # A threshold shaped like the (n_sites,) non-time dims broadcasts along
        # the trailing block axis without raising.
        thr = z1  # (n_sites,)
        count = expected_exceedances(thr, loc1, scale1, shape1, time_axis=1)
        assert count.shape == (loc0.shape[1],)
        assert jnp.allclose(count, loc0.shape[0] / period, atol=1e-3)

    @pytest.mark.slow
    def test_heavy_frechet_tail_brackets_root(self):
        """Regression: heavy Fréchet tails / large T must not clamp to the
        initial 20*scale upper bracket.

        With flat fields the non-stationary solve reduces to the closed-form
        stationary quantile sum_t S(z) = N/T  =>  S(z) = 1/T, so the answer
        must equal gev_return_level(T, ...). Previously this returned ~20.0.
        """
        n_blocks = 30
        loc = jnp.zeros((n_blocks, 1))
        scale = jnp.ones((n_blocks, 1))
        for shape, period in [(0.3, 1000.0), (0.5, 1000.0), (0.5, 10_000.0)]:
            shape_field = jnp.full((n_blocks, 1), shape)
            z = nonstationary_return_level(period, loc, scale, shape_field, time_axis=0)
            ref = gev_return_level(period, 0.0, 1.0, shape)
            assert z[0] > 20.0  # the old buggy clamp value
            assert jnp.allclose(z[0], ref, rtol=1e-3)

    def test_jit_and_grad_safe(self):
        loc, scale, shape = self._fields()
        fn = jax.jit(
            lambda p: nonstationary_return_level(p, loc, scale, shape, time_axis=0)
        )
        out = fn(20.0)
        assert out.shape == (loc.shape[1],)
        assert jnp.all(jnp.isfinite(out))

    def test_stationary_large_period_uses_stable_path(self):
        """Stationary mode must route through the stable return-level path so a
        large period does not round ``1 - 1/T`` to 1 (returning an endpoint)."""
        period = float(2**25)
        stat = nonstationary_return_level(period, 0.0, 1.0, 0.1, time_axis=None)
        assert jnp.isfinite(stat)
        assert jnp.allclose(stat, gev_return_level(period, 0.0, 1.0, 0.1), rtol=1e-6)

    @pytest.mark.slow
    def test_reverse_mode_grad_matches_finite_difference(self):
        """The non-stationary solve is reverse-mode differentiable via the
        implicit function theorem (issue #47): the bracketing ``while_loop``
        used to raise under ``jax.grad``, and bisection alone gives 0 gradient.
        """
        loc, scale, shape = self._fields()

        def z_of_scale(factor):
            return nonstationary_return_level(
                100.0, loc, factor * scale, shape, time_axis=0
            )[0]

        grad = float(jax.grad(z_of_scale)(1.0))
        h = 1e-4
        fd = float(z_of_scale(1.0 + h) - z_of_scale(1.0 - h)) / (2 * h)
        assert jnp.isfinite(grad)
        assert abs(grad) > 1e-3
        assert grad == pytest.approx(fd, rel=1e-3)

        def z_of_shape(xi):
            return nonstationary_return_level(
                100.0, loc, scale, jnp.full_like(shape, xi), time_axis=0
            )[0]

        grad_xi = float(jax.grad(z_of_shape)(0.1))
        fd_xi = float(z_of_shape(0.1 + h) - z_of_shape(0.1 - h)) / (2 * h)
        assert grad_xi == pytest.approx(fd_xi, rel=1e-3)


class TestSpatial:
    @pytest.mark.slow
    def test_pairwise_distances(self):
        coords = jnp.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
        d = pairwise_distances(coords)
        assert d.shape == (3, 3)
        assert jnp.allclose(jnp.diag(d), 0.0)
        assert jnp.allclose(d[0, 1], 5.0)
        assert jnp.allclose(d, d.T)

    @pytest.mark.slow
    def test_pairwise_distances_grad_finite(self):
        """The zero diagonal made ‖Δ‖'s Δ/‖Δ‖ gradient 0/0 → all-NaN (#48)."""
        coords = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
        g = jax.grad(lambda c: pairwise_distances(c).sum())(coords)
        assert jnp.all(jnp.isfinite(g))

    def test_pairwise_distances_propagates_nan_coord(self):
        """A NaN coordinate must surface as NaN distances, not be masked to a
        zero (coincident) distance that fakes a perfect correlation."""
        coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [jnp.nan, 0.0]])
        d = pairwise_distances(coords)
        assert jnp.isnan(d[2, 0]) and jnp.isnan(d[0, 2])
        assert float(d[0, 1]) == pytest.approx(1.0)  # valid pair unaffected

    def test_design_matrix(self):
        cov = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        x = design_matrix(cov)
        assert x.shape == (2, 3)
        assert jnp.allclose(x[:, 0], 1.0)
        assert jnp.allclose(x[:, 1:], cov)

    def test_design_matrix_1d(self):
        x = design_matrix(jnp.array([1.0, 2.0, 3.0]))
        assert x.shape == (3, 2)

    @pytest.mark.slow
    def test_two_range_correlation_is_valid(self):
        coords = jnp.linspace(0.0, 1.0, 8).reshape(-1, 1)
        d = pairwise_distances(coords)
        c = two_range_correlation(d, weight=0.6, range_short=0.1, range_long=0.5)
        assert c.shape == (8, 8)
        # Unit diagonal and symmetry.
        assert jnp.allclose(jnp.diag(c), 1.0, atol=1e-6)
        assert jnp.allclose(c, c.T, atol=1e-6)
        # Positive definite (valid correlation matrix).
        assert jnp.all(jnp.linalg.eigvalsh(c) > 0)

    @pytest.mark.slow
    def test_vmap_over_sites(self):
        keys = jax.random.split(jax.random.PRNGKey(0), 5)
        coords = jax.vmap(lambda k: jax.random.normal(k, (6, 2)))(keys)
        dists = jax.vmap(pairwise_distances)(coords)
        assert dists.shape == (5, 6, 6)
