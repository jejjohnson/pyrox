"""Tests for the Kalman-based :class:`MarkovGPPrior`.

The contract is small but load-bearing: on any sorted 1-D grid, the Kalman
filter / RTS smoother path must agree with the dense GP using the same
kernel and the same data — log marginal likelihood, posterior mean, and
posterior variance, all to numerical precision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpyro
import pytest
from numpyro.infer.util import log_density

from pyrox.gp import (
    ConditionedMarkovGP,
    MarkovGPPrior,
    MaternSDE,
    SumSDE,
    markov_gp_factor,
)
from pyrox.gp._src.kernels import matern_kernel


jax.config.update("jax_enable_x64", True)


# --- helpers -------------------------------------------------------------


def _dense_log_marginal(K: jax.Array, y: jax.Array, noise_var: jax.Array) -> jax.Array:
    """Reference dense ``log N(y | 0, K + sigma^2 I)`` via Cholesky."""
    n = y.shape[0]
    K_y = K + noise_var * jnp.eye(n, dtype=K.dtype)
    L = jnp.linalg.cholesky(K_y)
    alpha = jsl.solve_triangular(L, y, lower=True)
    return (
        -0.5 * (alpha @ alpha)
        - jnp.sum(jnp.log(jnp.diag(L)))
        - 0.5 * n * jnp.log(2.0 * jnp.pi)
    )


def _dense_predict(
    K_train: jax.Array,
    K_cross: jax.Array,
    K_test_diag: jax.Array,
    y: jax.Array,
    noise_var: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Reference dense ``(mean, var)`` predictive marginals."""
    n = y.shape[0]
    K_y = K_train + noise_var * jnp.eye(n, dtype=K_train.dtype)
    L = jnp.linalg.cholesky(K_y)
    alpha = jsl.cho_solve((L, True), y)
    mean = K_cross @ alpha
    v = jsl.solve_triangular(L, K_cross.T, lower=True)
    var = K_test_diag - jnp.sum(v * v, axis=0)
    return mean, var


# --- structural / shape contract -----------------------------------------


def test_markov_gp_state_dim_inherits_from_kernel() -> None:
    times = jnp.linspace(0.0, 1.0, 5)
    for order in (0, 1, 2):
        prior = MarkovGPPrior(
            MaternSDE(variance=1.0, lengthscale=0.5, order=order), times
        )
        assert prior.state_dim == order + 1


def test_invalid_times_raises() -> None:
    sde = MaternSDE(order=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        MarkovGPPrior(sde, jnp.array([0.0, 0.5, 0.5, 1.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        MarkovGPPrior(sde, jnp.array([0.0, 0.4, 0.3]))
    with pytest.raises(ValueError, match="must be 1-D"):
        MarkovGPPrior(sde, jnp.zeros((3, 2)))


def test_invalid_obs_noise_floor_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MarkovGPPrior(
            MaternSDE(order=1), jnp.linspace(0.0, 1.0, 3), obs_noise_floor=-0.1
        )


# --- equivalence with dense GP -------------------------------------------


@pytest.mark.parametrize(("order", "nu"), [(0, 0.5), (1, 1.5), (2, 2.5)])
def test_log_marginal_matches_dense_matern(order: int, nu: float) -> None:
    sigma2, ell, noise_var = 0.7, 0.4, 0.05
    # Irregular grid stresses the discretisation-per-step path.
    times = jnp.array([0.0, 0.13, 0.55, 1.2, 1.9, 2.5, 3.7, 4.2])
    key = jax.random.PRNGKey(0)
    y = jnp.sin(2.0 * times) + 0.1 * jax.random.normal(key, (times.shape[0],))

    sde = MaternSDE(variance=sigma2, lengthscale=ell, order=order)
    prior = MarkovGPPrior(sde, times)
    log_marg_kf = prior.log_marginal(y, jnp.asarray(noise_var))

    K = matern_kernel(
        times[:, None], times[:, None], jnp.asarray(sigma2), jnp.asarray(ell), nu=nu
    )
    log_marg_dense = _dense_log_marginal(K, y, jnp.asarray(noise_var))

    assert jnp.allclose(log_marg_kf, log_marg_dense, atol=1e-8)


@pytest.mark.slow
@pytest.mark.parametrize(("order", "nu"), [(0, 0.5), (1, 1.5), (2, 2.5)])
def test_predict_at_training_times_matches_dense(order: int, nu: float) -> None:
    sigma2, ell, noise_var = 0.6, 0.5, 0.04
    times = jnp.array([0.0, 0.2, 0.7, 1.4, 2.0, 3.1])
    key = jax.random.PRNGKey(1)
    y = 0.5 * jnp.cos(times) + 0.05 * jax.random.normal(key, (times.shape[0],))

    sde = MaternSDE(variance=sigma2, lengthscale=ell, order=order)
    prior = MarkovGPPrior(sde, times)
    cond = prior.condition(y, jnp.asarray(noise_var))
    m_kf, v_kf = cond.predict(times)

    X = times[:, None]
    K = matern_kernel(X, X, jnp.asarray(sigma2), jnp.asarray(ell), nu=nu)
    K_diag = jnp.diag(K)
    m_dense, v_dense = _dense_predict(K, K, K_diag, y, jnp.asarray(noise_var))

    assert jnp.allclose(m_kf, m_dense, atol=1e-7)
    assert jnp.allclose(v_kf, v_dense, atol=1e-7)


@pytest.mark.slow
@pytest.mark.parametrize(("order", "nu"), [(0, 0.5), (1, 1.5), (2, 2.5)])
def test_predict_off_grid_matches_dense(order: int, nu: float) -> None:
    sigma2, ell, noise_var = 0.8, 0.3, 0.02
    times = jnp.array([0.0, 0.4, 1.1, 1.8, 2.7, 3.6])
    t_star = jnp.array([-0.2, 0.5, 1.5, 2.0, 4.5])  # mix of inside/outside
    key = jax.random.PRNGKey(2)
    y = jnp.sin(times) + 0.05 * jax.random.normal(key, (times.shape[0],))

    sde = MaternSDE(variance=sigma2, lengthscale=ell, order=order)
    prior = MarkovGPPrior(sde, times)
    cond = prior.condition(y, jnp.asarray(noise_var))
    m_kf, v_kf = cond.predict(t_star)

    X, X_star = times[:, None], t_star[:, None]
    K = matern_kernel(X, X, jnp.asarray(sigma2), jnp.asarray(ell), nu=nu)
    K_cross = matern_kernel(X_star, X, jnp.asarray(sigma2), jnp.asarray(ell), nu=nu)
    K_test_diag = jnp.diag(
        matern_kernel(X_star, X_star, jnp.asarray(sigma2), jnp.asarray(ell), nu=nu)
    )
    m_dense, v_dense = _dense_predict(
        K, K_cross, K_test_diag, y, jnp.asarray(noise_var)
    )

    assert jnp.allclose(m_kf, m_dense, atol=1e-7)
    assert jnp.allclose(v_kf, v_dense, atol=1e-7)


# --- filter / smooth consistency -----------------------------------------


@pytest.mark.slow
def test_filter_and_smooth_log_marginal_agree() -> None:
    """``filter`` and ``smooth`` must report the same log marginal.

    Also exercises the smoothed-covariance PSD invariant on every step.
    """
    times = jnp.linspace(0.0, 4.0, 25)
    key = jax.random.PRNGKey(3)
    y = jnp.sin(times) + 0.05 * jax.random.normal(key, (times.shape[0],))
    sde = MaternSDE(variance=0.8, lengthscale=0.6, order=2)
    prior = MarkovGPPrior(sde, times)

    *_, log_marg_filter = prior.filter(y, jnp.asarray(0.01))
    _m_smooth, P_smooth, log_marg_smooth = prior.smooth(y, jnp.asarray(0.01))
    assert jnp.allclose(log_marg_filter, log_marg_smooth)
    for P in P_smooth:
        eig = jnp.linalg.eigvalsh(0.5 * (P + P.T))
        assert jnp.all(eig > -1e-8)


# --- composition kernels flow through ------------------------------------


def test_sum_sde_log_marginal_well_defined() -> None:
    """``SumSDE`` of two Matern kernels gives a well-defined log marginal.

    No closed-form dense reference (no ``+`` operator on the dense ``Matern``
    ``Parameterized`` class), so verify finite output, PSD smoothed covs,
    and consistency between ``filter`` and ``smooth``.
    """
    times = jnp.linspace(0.0, 4.0, 25)
    key = jax.random.PRNGKey(7)
    y = (
        jnp.sin(times)
        + 0.5 * jnp.cos(5.0 * times)
        + 0.05 * jax.random.normal(key, (times.shape[0],))
    )
    sde = SumSDE(
        (
            MaternSDE(variance=0.8, lengthscale=1.5, order=2),
            MaternSDE(variance=0.2, lengthscale=0.1, order=0),
        )
    )
    prior = MarkovGPPrior(sde, times)
    log_marg = prior.log_marginal(y, jnp.asarray(0.01))
    assert jnp.isfinite(log_marg)

    *_, log_marg_filter = prior.filter(y, jnp.asarray(0.01))
    _m_smooth, P_smooth, log_marg_smooth = prior.smooth(y, jnp.asarray(0.01))
    assert jnp.allclose(log_marg, log_marg_filter)
    assert jnp.allclose(log_marg, log_marg_smooth)
    for P in P_smooth:
        eig = jnp.linalg.eigvalsh(0.5 * (P + P.T))
        assert jnp.all(eig > -1e-8)


# --- gradients -----------------------------------------------------------


@pytest.mark.slow
def test_log_marginal_gradient_matches_dense() -> None:
    """Autodiff through the Kalman filter must match dense-GP gradients."""
    times = jnp.array([0.0, 0.2, 0.6, 1.1, 1.7, 2.3, 3.0])
    key = jax.random.PRNGKey(4)
    y = jnp.sin(times) + 0.05 * jax.random.normal(key, (times.shape[0],))

    def f_kf(params):
        sigma2, ell, noise_var = params
        sde = MaternSDE(variance=sigma2, lengthscale=ell, order=1)
        prior = MarkovGPPrior(sde, times)
        return prior.log_marginal(y, noise_var)

    def f_dense(params):
        sigma2, ell, noise_var = params
        K = matern_kernel(times[:, None], times[:, None], sigma2, ell, nu=1.5)
        return _dense_log_marginal(K, y, noise_var)

    params = (jnp.asarray(0.7), jnp.asarray(0.4), jnp.asarray(0.05))
    g_kf = jax.grad(f_kf)(params)
    g_dense = jax.grad(f_dense)(params)
    for a, b in zip(g_kf, g_dense, strict=True):
        assert jnp.allclose(a, b, atol=1e-6)


# --- jit compatibility ---------------------------------------------------


def test_log_marginal_jit_compatible() -> None:
    times = jnp.linspace(0.0, 3.0, 20)
    y = jnp.sin(times)

    @jax.jit
    def go(prior, y, nv):
        return prior.log_marginal(y, nv)

    prior = MarkovGPPrior(MaternSDE(order=1), times)
    out = go(prior, y, jnp.asarray(0.05))
    assert jnp.isfinite(out)


def test_predict_jit_compatible() -> None:
    times = jnp.linspace(0.0, 3.0, 15)
    t_star = jnp.linspace(-0.2, 4.0, 25)
    y = jnp.sin(times)

    @jax.jit
    def go(prior, y, nv, t_star):
        cond = prior.condition(y, nv)
        return cond.predict(t_star)

    prior = MarkovGPPrior(MaternSDE(order=2), times)
    m, v = go(prior, y, jnp.asarray(0.05), t_star)
    assert m.shape == (25,) and v.shape == (25,)
    assert jnp.all(v >= -1e-8)


# --- NaN-safety on degenerate masked steps -------------------------------


def test_filter_stays_finite_when_masked_step_has_degenerate_S() -> None:
    """Masked steps must not contaminate outputs even when the innovation
    variance ``S`` is zero on those steps. Earlier the masked update
    computed ``K_full = ... / S`` and ``log(S)`` unconditionally, so a
    degenerate ``S`` produced ``inf``/``NaN`` that survived ``0 *`` masking.
    """
    from pyrox.gp._markov import _kalman_filter

    d = 2
    F = jnp.zeros((d, d))
    H = jnp.array([[1.0, 0.0]])
    P_inf = jnp.eye(d)
    A_seq = jnp.broadcast_to(jnp.eye(d), (3, d, d))
    Q_seq = jnp.zeros((3, d, d))  # deterministic transition → H P H^T = 0
    residual = jnp.array([0.0, 1.0, 0.0])
    mask = jnp.array([0.0, 0.0, 0.0])  # all steps masked
    R_seq = jnp.zeros(3)  # forces S = 0 on every step

    _m_pred, _P_pred, m_filt, P_filt, ll = _kalman_filter(
        F, H, P_inf, A_seq, Q_seq, residual, mask, R_seq
    )
    assert jnp.all(jnp.isfinite(m_filt))
    assert jnp.all(jnp.isfinite(P_filt))
    assert jnp.isfinite(ll)
    assert ll == 0.0  # no observations contribute


# --- numpyro factor hook -------------------------------------------------


def test_markov_gp_factor_matches_log_marginal() -> None:
    times = jnp.linspace(0.0, 2.0, 12)
    y = jnp.sin(times)

    def model(times, y):
        sde = MaternSDE(variance=0.5, lengthscale=0.3, order=1)
        prior = MarkovGPPrior(sde, times)
        markov_gp_factor("obs", prior, y, jnp.asarray(0.02))

    log_density_value, _ = log_density(model, (times, y), {}, {})

    sde = MaternSDE(variance=0.5, lengthscale=0.3, order=1)
    prior = MarkovGPPrior(sde, times)
    expected = prior.log_marginal(y, jnp.asarray(0.02))
    assert jnp.allclose(log_density_value, expected, atol=1e-9)


# --- mean function -------------------------------------------------------


@pytest.mark.slow
def test_constant_mean_function_shifts_predictions() -> None:
    times = jnp.linspace(0.0, 3.0, 15)
    t_star = jnp.linspace(0.5, 2.5, 7)
    key = jax.random.PRNGKey(5)
    y_centered = jnp.sin(times) + 0.05 * jax.random.normal(key, (times.shape[0],))
    offset = 3.0
    y_shifted = y_centered + offset

    sde = MaternSDE(variance=0.6, lengthscale=0.5, order=1)

    prior_zero = MarkovGPPrior(sde, times)
    cond_zero = prior_zero.condition(y_centered, jnp.asarray(0.01))
    m_zero, v_zero = cond_zero.predict(t_star)

    prior_const = MarkovGPPrior(sde, times, mean_fn=lambda t: jnp.full_like(t, offset))
    cond_const = prior_const.condition(y_shifted, jnp.asarray(0.01))
    m_const, v_const = cond_const.predict(t_star)

    # Constant mean shifts the predictive mean by ``offset`` and leaves the
    # predictive variance unchanged.
    assert jnp.allclose(m_const, m_zero + offset, atol=1e-7)
    assert jnp.allclose(v_const, v_zero, atol=1e-7)


def test_conditioned_object_carries_log_marginal() -> None:
    times = jnp.linspace(0.0, 1.0, 8)
    y = jnp.sin(times)
    prior = MarkovGPPrior(MaternSDE(order=1), times)
    cond = prior.condition(y, jnp.asarray(0.01))
    assert isinstance(cond, ConditionedMarkovGP)
    assert jnp.allclose(cond.log_marginal, prior.log_marginal(y, jnp.asarray(0.01)))


# numpyro shouldn't trace anything weird; sanity-check we can still import it.
def test_numpyro_import_smoke() -> None:
    assert hasattr(numpyro, "factor")
