"""Tests for pure kernel function math in pyrox.gp._src.kernels.

Pin the math against direct closed-form computations on small hand-checkable
arrays. These functions are the canonical math definitions; numerically
stable / scalable variants live in gaussx and are not in scope here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pyrox.gp._src.kernels import (
    constant_kernel,
    cosine_kernel,
    kernel_add,
    kernel_mul,
    linear_kernel,
    matern_kernel,
    periodic_kernel,
    polynomial_kernel,
    rational_quadratic_kernel,
    rbf_kernel,
    white_kernel,
)


# --- rbf -------------------------------------------------------------------


def test_rbf_diagonal_equals_variance():
    X = jnp.array([[0.0], [1.0], [2.0]])
    K = rbf_kernel(X, X, jnp.array(2.5), jnp.array(1.0))
    assert jnp.allclose(jnp.diag(K), 2.5)


def test_rbf_matches_closed_form_scalar_pair():
    x1 = jnp.array([[0.0]])
    x2 = jnp.array([[1.5]])
    var = jnp.array(0.7)
    ls = jnp.array(0.4)
    expected = var * jnp.exp(-0.5 * (1.5 / ls) ** 2)
    K = rbf_kernel(x1, x2, var, ls)
    assert jnp.allclose(K, expected)


def test_rbf_is_symmetric():
    X = jax.random.normal(jax.random.PRNGKey(0), (5, 3))
    K = rbf_kernel(X, X, jnp.array(1.0), jnp.array(0.7))
    assert jnp.allclose(K, K.T)


def test_rbf_gram_is_psd():
    X = jax.random.normal(jax.random.PRNGKey(1), (8, 2))
    K = rbf_kernel(X, X, jnp.array(1.3), jnp.array(0.5))
    eigs = jnp.linalg.eigvalsh(K + 1e-8 * jnp.eye(8))
    assert float(eigs.min()) > 0.0


# --- matern ---------------------------------------------------------------


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_matern_diagonal_equals_variance(nu):
    X = jnp.array([[0.0], [1.0], [2.0]])
    K = matern_kernel(X, X, jnp.array(1.7), jnp.array(0.9), nu)
    assert jnp.allclose(jnp.diag(K), 1.7, atol=1e-5)


def test_matern_half_matches_exponential():
    x1 = jnp.array([[0.0]])
    x2 = jnp.array([[1.2]])
    var = jnp.array(0.4)
    ls = jnp.array(0.6)
    K = matern_kernel(x1, x2, var, ls, 0.5)
    expected = var * jnp.exp(-1.2 / ls)
    assert jnp.allclose(K, expected)


def test_matern_three_halves_matches_closed_form():
    x1 = jnp.array([[0.0]])
    x2 = jnp.array([[0.8]])
    var = jnp.array(1.1)
    ls = jnp.array(0.5)
    a = jnp.sqrt(3.0) * 0.8 / ls
    expected = var * (1.0 + a) * jnp.exp(-a)
    K = matern_kernel(x1, x2, var, ls, 1.5)
    assert jnp.allclose(K, expected)


def test_matern_five_halves_matches_closed_form():
    x1 = jnp.array([[0.0]])
    x2 = jnp.array([[0.7]])
    var = jnp.array(0.9)
    ls = jnp.array(0.4)
    a = jnp.sqrt(5.0) * 0.7 / ls
    expected = var * (1.0 + a + (a * a) / 3.0) * jnp.exp(-a)
    K = matern_kernel(x1, x2, var, ls, 2.5)
    assert jnp.allclose(K, expected)


def test_matern_unsupported_nu_raises():
    X = jnp.zeros((1, 1))
    with pytest.raises(ValueError, match="nu"):
        matern_kernel(X, X, jnp.array(1.0), jnp.array(1.0), 1.0)


def test_matern_grad_is_finite_at_zero_distance():
    """Sqrt clipping must keep grad finite when X1 == X2."""

    def loss(ls):
        X = jnp.array([[0.5], [0.5]])
        K = matern_kernel(X, X, jnp.array(1.0), ls, 1.5)
        return jnp.sum(K)

    g = jax.grad(loss)(jnp.array(0.7))
    assert jnp.isfinite(g)


# --- periodic --------------------------------------------------------------


def test_periodic_repeats_with_period():
    """k(0, p) should equal k(0, 0) since the period brings us back."""
    var = jnp.array(1.3)
    ls = jnp.array(0.5)
    period = jnp.array(2.0)
    x1 = jnp.array([[0.0]])
    x2_zero = jnp.array([[0.0]])
    x2_period = jnp.array([[2.0]])  # one full period away
    k_zero = periodic_kernel(x1, x2_zero, var, ls, period)
    k_period = periodic_kernel(x1, x2_period, var, ls, period)
    assert jnp.allclose(k_zero, k_period, atol=1e-5)


def test_periodic_diagonal_equals_variance():
    X = jnp.array([[0.0], [1.0], [2.0]])
    K = periodic_kernel(X, X, jnp.array(2.0), jnp.array(0.7), jnp.array(1.5))
    assert jnp.allclose(jnp.diag(K), 2.0)


# --- linear ----------------------------------------------------------------


def test_linear_matches_dot_product():
    X1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    X2 = jnp.array([[0.5, 0.5]])
    var = jnp.array(2.0)
    bias = jnp.array(0.3)
    K = linear_kernel(X1, X2, var, bias)
    expected = var * (X1 @ X2.T) + bias
    assert jnp.allclose(K, expected)


# --- rational quadratic ----------------------------------------------------


def test_rational_quadratic_large_alpha_matches_rbf():
    """As alpha -> infty, RQ converges to RBF. Moderate alpha + bounded distances
    keep the check inside float32 precision."""
    X = jnp.array([[0.0], [0.1], [0.2], [0.3]])
    var = jnp.array(1.0)
    ls = jnp.array(1.0)
    K_rq = rational_quadratic_kernel(X, X, var, ls, jnp.array(1e4))
    K_rbf = rbf_kernel(X, X, var, ls)
    assert jnp.allclose(K_rq, K_rbf, atol=1e-3)


def test_rational_quadratic_diagonal_equals_variance():
    X = jnp.array([[0.0], [1.0], [2.0]])
    K = rational_quadratic_kernel(X, X, jnp.array(1.5), jnp.array(0.5), jnp.array(2.0))
    assert jnp.allclose(jnp.diag(K), 1.5)


# --- polynomial ------------------------------------------------------------


def test_polynomial_degree_one_matches_shifted_dot_product():
    X1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    X2 = jnp.array([[0.5, 0.5]])
    var = jnp.array(2.0)
    bias = jnp.array(0.3)
    K = polynomial_kernel(X1, X2, var, bias, 1)
    expected = var * (X1 @ X2.T + bias)
    assert jnp.allclose(K, expected)


def test_polynomial_degree_two_matches_closed_form():
    X = jnp.array([[1.0], [2.0]])
    var = jnp.array(1.0)
    bias = jnp.array(0.5)
    K = polynomial_kernel(X, X, var, bias, 2)
    expected = (X @ X.T + bias) ** 2
    assert jnp.allclose(K, expected)


def test_polynomial_rejects_degree_zero():
    X = jnp.zeros((2, 1))
    with pytest.raises(ValueError, match="degree"):
        polynomial_kernel(X, X, jnp.array(1.0), jnp.array(0.0), 0)


# --- cosine ----------------------------------------------------------------


def test_cosine_equals_variance_at_zero_distance():
    X = jnp.array([[0.5], [1.5]])
    K = cosine_kernel(X, X, jnp.array(1.7), jnp.array(2.0))
    assert jnp.allclose(jnp.diag(K), 1.7)


def test_cosine_negates_at_half_period():
    x1 = jnp.array([[0.0]])
    x2 = jnp.array([[1.0]])  # half of period 2 -> cos(pi) = -1
    K = cosine_kernel(x1, x2, jnp.array(1.0), jnp.array(2.0))
    assert jnp.allclose(K, -1.0, atol=1e-5)


# --- white -----------------------------------------------------------------


def test_white_is_diagonal_when_X1_is_X2():
    X = jnp.array([[0.0], [1.0], [2.0]])
    K = white_kernel(X, X, jnp.array(0.5))
    assert jnp.allclose(K, 0.5 * jnp.eye(3))


def test_white_is_zero_between_distinct_points():
    X1 = jnp.array([[0.0]])
    X2 = jnp.array([[1.0]])
    K = white_kernel(X1, X2, jnp.array(2.0))
    assert jnp.allclose(K, 0.0)


# --- constant --------------------------------------------------------------


def test_constant_is_uniform():
    X1 = jax.random.normal(jax.random.PRNGKey(0), (3, 2))
    X2 = jax.random.normal(jax.random.PRNGKey(1), (4, 2))
    K = constant_kernel(X1, X2, jnp.array(1.8))
    assert K.shape == (3, 4)
    assert jnp.allclose(K, 1.8)


# --- composition -----------------------------------------------------------


def test_kernel_add_is_pointwise_sum():
    X = jnp.array([[0.0], [1.0]])
    K1 = rbf_kernel(X, X, jnp.array(1.0), jnp.array(1.0))
    K2 = linear_kernel(X, X, jnp.array(0.5), jnp.array(0.0))
    assert jnp.allclose(kernel_add(K1, K2), K1 + K2)


def test_kernel_mul_is_pointwise_product():
    X = jnp.array([[0.0], [1.0]])
    K1 = rbf_kernel(X, X, jnp.array(1.0), jnp.array(1.0))
    K2 = periodic_kernel(X, X, jnp.array(1.0), jnp.array(0.5), jnp.array(1.0))
    assert jnp.allclose(kernel_mul(K1, K2), K1 * K2)


# --- jit / grad smoke ------------------------------------------------------


def test_rbf_jits_and_grads():
    X = jax.random.normal(jax.random.PRNGKey(2), (4, 2))

    @jax.jit
    def loss(ls):
        return jnp.sum(rbf_kernel(X, X, jnp.array(1.0), ls))

    g = jax.grad(loss)(jnp.array(0.7))
    assert jnp.isfinite(g)
