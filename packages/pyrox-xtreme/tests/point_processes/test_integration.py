"""Tests for the shared quadrature utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtremax.point_processes._integration import (
    cumulative_log_intensity,
    integrate_log_intensity,
)


def test_trapezoid_integrates_constant_exactly():
    # Constant intensity λ = 2 ⇒ Λ(T) = 2T.
    def log_lam(t):
        return jnp.full_like(t, jnp.log(2.0))

    result = integrate_log_intensity(log_lam, 0.0, 5.0, n_points=50)
    assert jnp.allclose(result, 10.0, atol=1e-5)


def test_simpson_integrates_quadratic_exactly():
    # Simpson is exact on cubics; λ(t) = 2t + 1 integrates to t² + t.
    # So log λ(t) = log(2t + 1).
    def log_lam(t):
        return jnp.log(2.0 * t + 1.0)

    # Over [0, 3]: Λ = 9 + 3 = 12.
    result = integrate_log_intensity(log_lam, 0.0, 3.0, n_points=51, method="simpson")
    assert jnp.allclose(result, 12.0, atol=1e-3)


def test_trapezoid_batched_limits():
    def log_lam(t):
        return jnp.full_like(t, jnp.log(3.0))

    t0 = jnp.array([0.0, 1.0, 2.0])
    t1 = jnp.array([1.0, 3.0, 5.0])
    result = integrate_log_intensity(log_lam, t0, t1, n_points=50)
    expected = jnp.array([3.0, 6.0, 9.0])
    assert jnp.allclose(result, expected, atol=1e-4)


def test_cumulative_log_intensity_matches_integrate_from_zero():
    def log_lam(t):
        return jnp.log(2.0 + jnp.sin(t))

    t = 4.0
    a = cumulative_log_intensity(log_lam, t, n_points=200)
    b = integrate_log_intensity(log_lam, 0.0, t, n_points=200)
    assert jnp.allclose(a, b)


def test_integrate_is_jit_safe():
    def log_lam(t):
        return jnp.log(2.0 + 0.5 * t)

    jitted = jax.jit(lambda a, b: integrate_log_intensity(log_lam, a, b, n_points=80))
    result = jitted(jnp.asarray(0.0), jnp.asarray(4.0))
    assert jnp.isfinite(result)


def test_integrate_requires_n_points_at_least_2():
    def log_lam(t):
        return jnp.zeros_like(t)

    with pytest.raises(ValueError):
        integrate_log_intensity(log_lam, 0.0, 1.0, n_points=1)


def test_simpson_auto_odd_padding():
    # Even n_points should be bumped to odd without a crash.
    def log_lam(t):
        return jnp.zeros_like(t)

    result = integrate_log_intensity(log_lam, 0.0, 1.0, n_points=10, method="simpson")
    assert jnp.allclose(result, 1.0, atol=1e-8)
