"""Tests for `nonlinear_ssm_factor`, the moment-matched nonlinear SSM factor.

The contracts:

* with affine dynamics and observations, the factor's log likelihood is the
  exact Kalman log marginal of the matching `LinearGaussianSSMPrior` for
  every gaussx integrator (the deterministic rules to ``rtol=1e-10``; Monte
  Carlo to the tolerance its sample size implies);
* on the log-emission AR(1) toy, ``y_k = exp(x_k) + eps_k`` with
  ``x_k = a x_{k-1} + eta_k``, the gradient with respect to ``a`` matches
  central finite differences, under ``jit`` too;
* masking a step or an entry equals dropping that observation, and ``NaN``
  at masked entries leaves the value and the gradient finite;
* the factor registers its one site and SVI recovers the surrogate's
  maximiser in ``a``.
"""

from __future__ import annotations

import gaussx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from pyrox_ssm import LinearGaussianSSMPrior, lgssm_factor, nonlinear_ssm_factor


jax.config.update("jax_enable_x64", True)

T = 15


# --- fixtures ------------------------------------------------------------


def _linear_model():
    """D = 2, M = 2 affine model with correlated noises."""
    A = jnp.array([[0.9, 0.2], [-0.1, 0.8]])
    H = jnp.array([[1.0, 0.5], [-0.3, 1.0]])
    Q = jnp.array([[0.2, 0.05], [0.05, 0.1]])
    R = jnp.array([[0.3, 0.1], [0.1, 0.4]])
    m0 = jnp.array([0.3, -0.2])
    P0 = jnp.array([[1.0, 0.2], [0.2, 0.5]])
    y = jr.normal(jr.key(0), (T, 2))
    return A, H, Q, R, m0, P0, y


def _log_emission_data(a: float = 0.8, *, T: int = T, seed: int = 1):
    """Simulate ``x_k = a x_{k-1} + eta_k``, ``y_k = exp(x_k) + eps_k``."""
    k_eta, k_eps = jr.split(jr.key(seed))
    eta = jnp.sqrt(0.1) * jr.normal(k_eta, (T,))

    def step(x, e):
        x = a * x + e
        return x, x

    _, x = jax.lax.scan(step, jnp.asarray(0.0), eta)
    y = jnp.exp(x) + jnp.sqrt(0.05) * jr.normal(k_eps, (T,))
    return y[:, None]


def _log_emission_loglik(a, y, *, mask=None, integrator=None):
    """The factor's log likelihood on the log-emission AR(1) toy."""
    with handlers.seed(rng_seed=0):
        state = nonlinear_ssm_factor(
            "y",
            lambda x: a * x,
            jnp.exp,
            0.1 * jnp.eye(1),
            0.05 * jnp.eye(1),
            y,
            jnp.zeros(1),
            jnp.eye(1),
            integrator=integrator,
            mask=mask,
        )
    return state.log_likelihood


def _two_channel_obs(x):
    """A nonlinear two-channel observation map of a two-dimensional state."""
    return jnp.stack([jnp.exp(0.5 * x[0]), x[0] + jnp.sin(x[1])])


def _two_channel_dynamics(x):
    return jnp.array([0.9 * x[0] + 0.1 * jnp.tanh(x[1]), 0.8 * x[1]])


_Q2 = jnp.array([[0.1, 0.02], [0.02, 0.05]])
_R2 = jnp.array([[0.2, 0.05], [0.05, 0.3]])
_M02, _P02 = jnp.array([0.1, -0.1]), jnp.eye(2)


def _two_channel_loglik(y, mask):
    with handlers.seed(rng_seed=0):
        state = nonlinear_ssm_factor(
            "y", _two_channel_dynamics, _two_channel_obs, _Q2, _R2, y, _M02, _P02,
            mask=mask,
        )  # fmt: skip
    return state.log_likelihood


def _dropped_reference(y, keep):
    """Filter step by step, updating on the kept channels of ``y`` only.

    ``keep`` is ``(T, M)``; a step with no kept channel runs the predict only,
    and a step with some updates on the sub-map and sub-noise of those
    channels, so the dropped observations never enter.
    """
    integrator = gaussx.UnscentedIntegrator(alpha=1.0)
    mean, cov, total = _M02, _P02, jnp.asarray(0.0)
    for k in range(y.shape[0]):
        mean, cov = gaussx.nonlinear_kalman_predict(
            _two_channel_dynamics, mean, cov, _Q2, integrator=integrator
        )
        idx = [j for j in range(y.shape[1]) if bool(keep[k, j])]
        if not idx:
            continue
        sel = jnp.array(idx)
        mean, cov, step = gaussx.nonlinear_kalman_update(
            lambda x, sel=sel: _two_channel_obs(x)[sel],
            mean,
            cov,
            y[k, sel],
            _R2[jnp.ix_(sel, sel)],
            integrator=integrator,
        )
        total = total + step
    return total


# --- linear case: every integrator is the Kalman filter ------------------


_DETERMINISTIC_INTEGRATORS = {
    "default": None,
    "unscented-alpha1e-3": gaussx.UnscentedIntegrator(),
    "taylor": gaussx.TaylorIntegrator(),
    "cubature": gaussx.CubatureIntegrator(),
    "fifth-order-cubature": gaussx.FifthOrderCubatureIntegrator(),
    "gauss-hermite": gaussx.GaussHermiteIntegrator(),
}


@pytest.mark.parametrize(
    "integrator",
    list(_DETERMINISTIC_INTEGRATORS.values()),
    ids=list(_DETERMINISTIC_INTEGRATORS),
)
def test_linear_case_equals_kalman_log_marginal(integrator) -> None:
    A, H, Q, R, m0, P0, y = _linear_model()
    # Same convention on both sides: predict before every update, so y[0]
    # is scored against A x_0.
    prior = LinearGaussianSSMPrior(
        jnp.broadcast_to(A, (T, 2, 2)), H, jnp.broadcast_to(Q, (T, 2, 2)), R, m0, P0
    )
    with handlers.seed(rng_seed=0):
        state = nonlinear_ssm_factor(
            "y",
            lambda x: A @ x,
            lambda x: H @ x,
            Q,
            R,
            y,
            m0,
            P0,
            integrator=integrator,
        )
    # Every deterministic rule is exact for affine maps; under x64 the
    # recursions round differently by ~1e-15 (the scaled unscented rule's
    # cancellation costs ~1e-11), well inside 1e-10.
    assert jnp.allclose(state.log_likelihood, prior.log_marginal(y), rtol=1e-10)

    def lgssm_model():
        lgssm_factor("y", prior, y)

    lgssm_trace = handlers.trace(handlers.seed(lgssm_model, 0)).get_trace()
    factor = lgssm_trace["y"]["fn"].log_factor
    assert jnp.allclose(state.log_likelihood, factor, rtol=1e-10)


def test_linear_case_monte_carlo_within_sampling_error() -> None:
    A, H, Q, R, m0, P0, y = _linear_model()
    prior = LinearGaussianSSMPrior(
        jnp.broadcast_to(A, (T, 2, 2)), H, jnp.broadcast_to(Q, (T, 2, 2)), R, m0, P0
    )
    n = 10_000
    with handlers.seed(rng_seed=0):
        state = nonlinear_ssm_factor(
            "y",
            lambda x: A @ x,
            lambda x: H @ x,
            Q,
            R,
            y,
            m0,
            P0,
            integrator=gaussx.MonteCarloIntegrator(n_samples=n, key=jr.key(0)),
        )
    # Monte Carlo is not exact for affine maps: the filter uses the empirical
    # moments of n draws, whose covariance is off by a relative O(sqrt(2/n)).
    # Each of the T steps' M-dimensional log density moves by about M times
    # that, so the log likelihood is within T M sqrt(2/n) nats (0.42 here;
    # the measured difference is ~0.02).
    tol = T * 2 * jnp.sqrt(2.0 / n)
    assert jnp.abs(state.log_likelihood - prior.log_marginal(y)) < tol


# --- gradients on the log-emission toy -----------------------------------


@pytest.mark.parametrize(
    "integrator",
    [
        None,
        gaussx.TaylorIntegrator(),
        gaussx.CubatureIntegrator(),
        gaussx.GaussHermiteIntegrator(),
    ],
    ids=["unscented", "taylor", "cubature", "gauss-hermite"],
)
def test_log_emission_gradient_matches_finite_differences(integrator) -> None:
    y = _log_emission_data()
    a, h = 0.7, 1e-5

    def loglik(a):
        return _log_emission_loglik(a, y, integrator=integrator)

    grad = jax.grad(loglik)(a)
    fd = (loglik(a + h) - loglik(a - h)) / (2 * h)
    # Central differences have O(h^2) truncation and O(eps / h) rounding
    # error, ~1e-10 relative at h = 1e-5 under x64.
    assert jnp.isfinite(grad)
    assert jnp.allclose(grad, fd, rtol=1e-6)


def test_jit_and_grad_with_respect_to_dynamics_parameters() -> None:
    A, H, Q, R, m0, P0, y = _linear_model()

    def loglik(A):
        with handlers.seed(rng_seed=0):
            state = nonlinear_ssm_factor(
                "y", lambda x: jnp.tanh(A @ x), lambda x: H @ x, Q, R, y, m0, P0
            )
        return state.log_likelihood

    assert jnp.allclose(jax.jit(loglik)(A), loglik(A), rtol=1e-12)
    grad = jax.jit(jax.grad(loglik))(A)
    assert jnp.allclose(grad, jax.grad(loglik)(A), rtol=1e-10)
    h = 1e-5
    E = jnp.zeros((2, 2)).at[0, 1].set(1.0)
    fd = (loglik(A + h * E) - loglik(A - h * E)) / (2 * h)
    assert jnp.allclose(grad[0, 1], fd, rtol=1e-6)


# --- masks ---------------------------------------------------------------


def test_step_mask_equals_dropping_the_step() -> None:
    y = jr.normal(jr.key(2), (T, 2)) + jnp.array([1.0, 0.0])
    mask = jnp.ones((T,), dtype=bool).at[jnp.array([3, 7, 8])].set(False)
    keep = jnp.broadcast_to(mask[:, None], (T, 2))
    expected = _dropped_reference(y, keep)
    assert jnp.allclose(_two_channel_loglik(y, mask), expected, rtol=1e-10)


def test_entry_mask_equals_dropping_the_entries() -> None:
    y = jr.normal(jr.key(3), (T, 2)) + jnp.array([1.0, 0.0])
    keep = jnp.ones((T, 2), dtype=bool)
    keep = keep.at[2, 0].set(False).at[5, 1].set(False).at[9].set(False)
    expected = _dropped_reference(y, keep)
    assert jnp.allclose(_two_channel_loglik(y, keep), expected, rtol=1e-10)


def test_masking_trailing_steps_equals_truncating() -> None:
    y = _log_emission_data()
    mask = jnp.arange(T) < T - 3
    truncated = _log_emission_loglik(0.7, y[: T - 3])
    assert jnp.allclose(_log_emission_loglik(0.7, y, mask=mask), truncated, rtol=1e-12)


@pytest.mark.parametrize("per_entry", [False, True], ids=["step", "entry"])
def test_nan_at_masked_entries_leaves_value_and_gradient_finite(per_entry) -> None:
    y = _log_emission_data()
    step_mask = jnp.ones((T,), dtype=bool).at[jnp.array([2, 6])].set(False)
    mask = step_mask[:, None] if per_entry else step_mask
    y_nan = jnp.where(step_mask[:, None], y, jnp.nan)
    y_zero = jnp.where(step_mask[:, None], y, 0.0)

    def loglik(a, y):
        return _log_emission_loglik(a, y, mask=mask)

    value, grad = jax.value_and_grad(loglik)(0.7, y_nan)
    assert jnp.isfinite(value) and jnp.isfinite(grad)
    assert value == loglik(0.7, y_zero)
    # Batched under vmap as well, where the filter's per-step gate is
    # evaluated as a select over both branches.
    grads = jax.vmap(jax.grad(loglik), in_axes=(0, None))(jnp.array([0.5, 0.7]), y_nan)
    assert jnp.all(jnp.isfinite(grads))


def test_rejects_misshapen_observations_and_mask() -> None:
    y = _log_emission_data()
    with pytest.raises(ValueError, match="y must have shape"):
        _log_emission_loglik(0.7, y[:, 0])
    with pytest.raises(ValueError, match="mask has shape"):
        _log_emission_loglik(0.7, y, mask=jnp.ones((T - 1,), dtype=bool))


# --- NumPyro -------------------------------------------------------------


def _ar1_model(y):
    a = numpyro.sample("a", dist.Uniform(-1.0, 1.0))
    nonlinear_ssm_factor(
        "y",
        lambda x: a * x,
        jnp.exp,
        0.1 * jnp.eye(1),
        0.05 * jnp.eye(1),
        y,
        jnp.zeros(1),
        jnp.eye(1),
    )


def test_trace_registers_the_factor_site() -> None:
    y = _log_emission_data()
    trace = handlers.trace(handlers.seed(_ar1_model, 0)).get_trace(y)
    assert set(trace) == {"a", "y"}
    assert isinstance(trace["y"]["fn"], dist.Unit)
    a = trace["a"]["value"]
    assert jnp.allclose(trace["y"]["fn"].log_factor, _log_emission_loglik(a, y))


def test_svi_recovers_the_surrogate_maximiser() -> None:
    y = _log_emission_data(a=0.8, T=60)
    # The prior on a is flat on (-1, 1), so the MAP that AutoDelta finds is
    # the maximiser of the surrogate log likelihood, found here on a grid.
    grid = jnp.linspace(-0.99, 0.99, 1981)
    a_star = grid[jnp.argmax(jax.vmap(lambda a: _log_emission_loglik(a, y))(grid))]

    guide = AutoDelta(_ar1_model)
    svi = SVI(_ar1_model, guide, numpyro.optim.Adam(0.02), Trace_ELBO())
    result = svi.run(jr.key(0), 600, y, progress_bar=False)
    a_hat = guide.median(result.params)["a"]
    # The grid spacing is 1e-3 and Adam at lr 0.02 settles within a few 1e-3.
    assert jnp.abs(a_hat - a_star) < 1e-2
