"""Tests for :mod:`xtremax.point_processes.primitives.ipp`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import random

from xtremax.point_processes._integration import integrate_log_intensity
from xtremax.point_processes.primitives import (
    ipp_cumulative_hazard,
    ipp_cumulative_intensity,
    ipp_hazard,
    ipp_intensity,
    ipp_inter_event_log_prob,
    ipp_log_prob,
    ipp_predict_count,
    ipp_sample_inversion,
    ipp_sample_thinning,
    ipp_survival,
)


T_WINDOW = 10.0


def constant_log_intensity(log_rate: float):
    def fn(t):
        return jnp.full_like(t, log_rate)

    return fn


def linear_log_intensity(t):
    # λ(t) = 1 + 0.5 t ⇒ log λ = log(1 + 0.5 t). Λ(T) = T + 0.25 T².
    return jnp.log(1.0 + 0.5 * t)


def sinusoidal_log_intensity(t):
    # λ(t) = 2 + sin(t). Minimum is 1 so always positive.
    return jnp.log(2.0 + jnp.sin(t))


class TestIppLogProb:
    def test_matches_hpp_when_constant(self):
        rate = 2.0
        fn = constant_log_intensity(jnp.log(rate))
        times = jnp.array([0.5, 1.2, 3.3, 5.5, 7.7])
        mask = jnp.ones_like(times, dtype=jnp.bool_)
        Lambda_T = rate * T_WINDOW
        ipp_val = ipp_log_prob(times, mask, fn, Lambda_T)
        # HPP log L = n log λ - λT.
        expected = 5 * jnp.log(rate) - rate * T_WINDOW
        assert jnp.allclose(ipp_val, expected)

    def test_padding_does_not_contribute(self):
        fn = constant_log_intensity(jnp.log(3.0))
        times = jnp.array([1.0, 2.0, 0.0, 0.0])  # last two are padding
        mask = jnp.array([True, True, False, False])
        Lambda_T = 3.0 * T_WINDOW
        expected = 2 * jnp.log(3.0) - 3.0 * T_WINDOW
        assert jnp.allclose(ipp_log_prob(times, mask, fn, Lambda_T), expected)

    def test_grad_through_intensity_fn(self):
        # Parameterised log-intensity: log λ(t) = θ. ∂ log L / ∂ θ = n - exp(θ) T.
        times = jnp.array([0.5, 1.0, 1.5])
        mask = jnp.ones_like(times, dtype=jnp.bool_)

        def loss(theta):
            fn = lambda t: jnp.full_like(t, theta)
            Lambda_T = jnp.exp(theta) * T_WINDOW
            return -ipp_log_prob(times, mask, fn, Lambda_T)

        theta = jnp.asarray(0.5)
        grad = jax.grad(loss)(theta)
        expected = -(3.0 - jnp.exp(theta) * T_WINDOW)
        assert jnp.allclose(grad, expected)


class TestIppSampleThinning:
    def test_reduces_to_hpp_when_lambda_max_tight(self):
        # Constant intensity with λ_max exactly equal: acceptance probability 1.
        rate = 1.5
        fn = constant_log_intensity(jnp.log(rate))
        key = random.PRNGKey(0)
        _times, mask, n_uncapped = ipp_sample_thinning(
            key, fn, T=T_WINDOW, lambda_max=rate, max_candidates=200
        )
        # All candidates should be accepted.
        accepted = mask.sum()
        n_candidates = jnp.minimum(n_uncapped, 200)
        assert accepted == n_candidates

    def test_thinning_scales_count_correctly(self):
        # Over many seeds, empirical mean accepted = λ_avg * T.
        rate = 2.0
        fn = constant_log_intensity(jnp.log(rate))
        keys = random.split(random.PRNGKey(1), 1000)

        def once(k):
            _, m, _ = ipp_sample_thinning(k, fn, T_WINDOW, rate * 1.5, 256)
            return m.sum()

        counts = jax.vmap(once)(keys)
        mean_count = jnp.mean(counts.astype(jnp.float32))
        # Expected 20. Monte Carlo sem ≈ sqrt(20/1000) ≈ 0.14. Tolerate 1.0.
        assert jnp.abs(mean_count - 20.0) < 1.0

    def test_thinning_jits(self):
        rate = 1.0
        fn = constant_log_intensity(jnp.log(rate))
        jitted = jax.jit(lambda k: ipp_sample_thinning(k, fn, T_WINDOW, rate, 128))
        times, _mask, _n = jitted(random.PRNGKey(0))
        assert times.shape == (128,)

    def test_thinning_log_prob_consistency(self):
        # Generate events, then check log_prob is finite and reasonable.
        rate = 2.0
        fn = constant_log_intensity(jnp.log(rate))
        times, mask, _ = ipp_sample_thinning(random.PRNGKey(3), fn, T_WINDOW, rate, 256)
        log_L = ipp_log_prob(times, mask, fn, rate * T_WINDOW)
        assert jnp.isfinite(log_L)


class TestIppSampleInversion:
    def test_inversion_reduces_to_uniform_sort(self):
        # Constant intensity: Λ(t) = λ t, Λ⁻¹(y) = y/λ.
        rate = 2.0
        Lambda_T = rate * T_WINDOW

        def inv_cum(y):
            return y / rate

        times, mask, _n = ipp_sample_inversion(
            random.PRNGKey(0), inv_cum, Lambda_T, max_events=256
        )
        valid = times[mask]
        # Real events sorted and within [0, T].
        assert jnp.all(jnp.diff(valid) >= -1e-7)
        assert jnp.all((valid >= 0.0) & (valid <= T_WINDOW + 1e-6))


class TestIppIntensityHazardSurvival:
    def test_intensity_exp_of_log(self):
        fn = sinusoidal_log_intensity
        t = jnp.array([0.0, 1.0, 2.0])
        assert jnp.allclose(ipp_intensity(t, fn), 2.0 + jnp.sin(t))

    def test_hazard_equals_intensity(self):
        t = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(
            ipp_hazard(t, sinusoidal_log_intensity),
            ipp_intensity(t, sinusoidal_log_intensity),
        )

    def test_cumulative_intensity_matches_integrate(self):
        # Linear intensity: λ(t) = 1 + 0.5 t ⇒ Λ(T) = T + 0.25 T².
        result = ipp_cumulative_intensity(
            jnp.asarray(4.0), linear_log_intensity, n_points=500
        )
        expected = 4.0 + 0.25 * 16.0
        assert jnp.allclose(result, expected, atol=1e-2)

    def test_survival_positive_and_bounded(self):
        s = ipp_survival(
            jnp.asarray(3.0), jnp.asarray(1.0), sinusoidal_log_intensity, n_points=200
        )
        assert 0.0 < float(s) < 1.0

    def test_cumulative_hazard_matches_integrate_between(self):
        result = ipp_cumulative_hazard(
            jnp.asarray(4.0), jnp.asarray(1.0), linear_log_intensity, n_points=500
        )
        expected = integrate_log_intensity(linear_log_intensity, 1.0, 4.0, n_points=500)
        assert jnp.allclose(result, expected)


class TestIppInterEvent:
    def test_inter_event_negative_is_neg_inf(self):
        val = ipp_inter_event_log_prob(
            jnp.asarray(-0.1),
            jnp.asarray(1.0),
            sinusoidal_log_intensity,
        )
        assert jnp.isneginf(val)

    def test_inter_event_density_integrates_to_one(self):
        # Numerically: ∫₀^∞ f(τ|s=0) dτ ≈ 1.
        taus = jnp.linspace(0.0, 20.0, 1001)
        logs = jax.vmap(
            lambda tau: ipp_inter_event_log_prob(
                tau, jnp.asarray(0.0), linear_log_intensity, n_points=200
            )
        )(taus)
        density = jnp.exp(logs)
        integral = jnp.trapezoid(density, taus)
        assert jnp.allclose(integral, 1.0, atol=0.05)


class TestIppPredictCount:
    def test_predict_count_constant(self):
        rate = 3.0
        fn = constant_log_intensity(jnp.log(rate))
        assert jnp.allclose(ipp_predict_count(1.0, 4.0, fn, n_points=50), 9.0)


@pytest.mark.parametrize("seed", range(3))
def test_thinning_output_is_sorted(seed):
    rate = 1.0
    fn = constant_log_intensity(jnp.log(rate))
    times, mask, _ = ipp_sample_thinning(random.PRNGKey(seed), fn, T_WINDOW, rate, 128)
    # Real events are sorted.
    valid = times[mask]
    assert jnp.all(jnp.diff(valid) >= -1e-7)
