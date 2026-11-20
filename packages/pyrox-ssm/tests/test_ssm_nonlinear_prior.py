"""`NonlinearGaussianSSMPrior` and `ConditionedNonlinearGaussianSSM`.

The contracts:

* with affine dynamics and observations, ``filter``, ``log_marginal``,
  ``smooth`` and ``condition(...).predict`` equal `LinearGaussianSSMPrior`'s
  for every deterministic gaussx integrator, to ``rtol=1e-10`` (the scaled
  unscented rule with ``alpha=1e-3`` recovers moments by cancellation, so it
  is compared at the looser tolerance its lost digits leave);
* ``log_marginal`` is the value `nonlinear_ssm_factor` registers;
* masking a step equals dropping it from the smoother's view, and ``NaN`` at
  masked entries leaves the value and the gradient finite;
* on the log-emission AR(1) toy the gradient of ``log_marginal`` with
  respect to the AR coefficient matches central finite differences.
"""

from __future__ import annotations

import gaussx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro import handlers
from pyrox_ssm import (
    ConditionedNonlinearGaussianSSM,
    LinearGaussianSSMPrior,
    NonlinearGaussianSSMPrior,
    nonlinear_ssm_factor,
)


jax.config.update("jax_enable_x64", True)

T = 15
A = jnp.array([[0.9, 0.2], [-0.1, 0.8]])
H = jnp.array([[1.0, 0.5], [-0.3, 1.0]])
Q = jnp.array([[0.2, 0.05], [0.05, 0.1]])
R = jnp.array([[0.3, 0.1], [0.1, 0.4]])
M0 = jnp.array([0.3, -0.2])
P0 = jnp.array([[1.0, 0.2], [0.2, 0.5]])
Y = jr.normal(jr.key(0), (T, 2))

# The deterministic rules, and the tolerance each is exact to on affine maps
# under x64: ~1e-15 of round-off, except the alpha=1e-3 unscented rule, whose
# cancellation costs about six digits.
_INTEGRATORS = {
    "default": (None, 1e-10),
    "unscented-alpha1e-3": (gaussx.UnscentedIntegrator(), 1e-6),
    "taylor": (gaussx.TaylorIntegrator(), 1e-10),
    "cubature": (gaussx.CubatureIntegrator(), 1e-10),
    "fifth-order-cubature": (gaussx.FifthOrderCubatureIntegrator(), 1e-10),
    "gauss-hermite": (gaussx.GaussHermiteIntegrator(), 1e-10),
}


def _affine(integrator=None) -> NonlinearGaussianSSMPrior:
    return NonlinearGaussianSSMPrior(
        lambda x: A @ x, lambda x: H @ x, Q, R, M0, P0, integrator=integrator
    )


def _linear() -> LinearGaussianSSMPrior:
    return LinearGaussianSSMPrior(
        jnp.broadcast_to(A, (T, 2, 2)), H, jnp.broadcast_to(Q, (T, 2, 2)), R, M0, P0
    )


@pytest.mark.parametrize(
    ("integrator", "rtol"),
    list(_INTEGRATORS.values()),
    ids=list(_INTEGRATORS),
)
def test_affine_model_equals_the_linear_gaussian_ssm(integrator, rtol):
    mask = jnp.ones((T, 2), dtype=bool).at[3, 1].set(False).at[7, 0].set(False)
    y = jnp.where(mask, Y, jnp.nan)
    nonlinear, linear = _affine(integrator), _linear()

    state, state_ref = nonlinear.filter(y, mask=mask), linear.filter(y, mask=mask)
    assert jnp.allclose(
        state.filtered_means, state_ref.filtered_means, rtol=rtol, atol=rtol
    )
    assert jnp.allclose(state.log_likelihood, state_ref.log_likelihood, rtol=rtol)
    assert jnp.allclose(
        nonlinear.log_marginal(y, mask=mask),
        linear.log_marginal(y, mask=mask),
        rtol=rtol,
    )

    m, P, lm = nonlinear.smooth(y, mask=mask)
    m_ref, P_ref, lm_ref = linear.smooth(y, mask=mask)
    assert jnp.allclose(m, m_ref, rtol=rtol, atol=rtol)
    assert jnp.allclose(P, P_ref, rtol=rtol, atol=rtol)
    assert jnp.allclose(lm, lm_ref, rtol=rtol)

    cond = nonlinear.condition(y, mask=mask)
    assert isinstance(cond, ConditionedNonlinearGaussianSSM)
    cond_ref = linear.condition(y, mask=mask)
    for include_noise in (False, True):
        mean, var = cond.predict(include_noise=include_noise)
        mean_ref, var_ref = cond_ref.predict(include_noise=include_noise)
        assert jnp.allclose(mean, mean_ref, rtol=rtol, atol=rtol)
        assert jnp.allclose(var, var_ref, rtol=rtol, atol=rtol)


def test_log_marginal_is_the_value_the_factor_registers():
    prior = _affine()

    def model():
        nonlinear_ssm_factor("y", lambda x: A @ x, lambda x: H @ x, Q, R, Y, M0, P0)

    trace = handlers.trace(handlers.seed(model, 0)).get_trace()
    assert jnp.array_equal(trace["y"]["fn"].log_factor, prior.log_marginal(Y))


def test_condition_zeroes_masked_entries_and_keeps_the_mask():
    mask = jnp.arange(T) != 4
    cond = _affine().condition(jnp.where(mask[:, None], Y, jnp.nan), mask=mask)
    assert jnp.array_equal(cond.y[4], jnp.zeros(2))
    assert cond.mask is not None and jnp.array_equal(cond.mask, mask)
    assert jnp.all(jnp.isfinite(cond.smoothed_means))


def _log_emission(a):
    return NonlinearGaussianSSMPrior(
        lambda x: a * x,
        jnp.exp,
        0.1 * jnp.eye(1),
        0.05 * jnp.eye(1),
        jnp.zeros(1),
        jnp.eye(1),
    )


def _log_emission_data():
    k_eta, k_eps = jr.split(jr.key(1))
    eta = jnp.sqrt(0.1) * jr.normal(k_eta, (T,))

    def step(x, e):
        x = 0.8 * x + e
        return x, x

    _, x = jax.lax.scan(step, jnp.asarray(0.0), eta)
    return (jnp.exp(x) + jnp.sqrt(0.05) * jr.normal(k_eps, (T,)))[:, None]


def test_log_emission_gradient_matches_finite_differences():
    y = _log_emission_data()
    lm = lambda a: _log_emission(a).log_marginal(y)
    a0, h = 0.7, 1e-5
    fd = (lm(a0 + h) - lm(a0 - h)) / (2 * h)
    # central differences in x64: O(h^2) truncation plus ~1e-11 round-off.
    assert jnp.allclose(jax.grad(lm)(a0), fd, rtol=1e-6)
    assert jnp.allclose(jax.jit(jax.grad(lm))(a0), fd, rtol=1e-6)


def test_nan_at_masked_entries_leaves_value_gradient_and_smoother_finite():
    y = _log_emission_data().at[5, 0].set(jnp.nan)
    mask = jnp.arange(T) != 5

    def lm(a):
        return _log_emission(a).log_marginal(y, mask=mask)

    assert jnp.isfinite(lm(0.7)) and jnp.isfinite(jax.grad(lm)(0.7))
    cond = _log_emission(0.7).condition(y, mask=mask)
    mean, var = cond.predict(include_noise=True)
    assert jnp.all(jnp.isfinite(mean)) and jnp.all(var > 0)
    # Masking a step equals leaving it out of the update: the filter's value
    # equals the factor on the same mask.
    with handlers.seed(rng_seed=0):
        state = nonlinear_ssm_factor(
            "y",
            lambda x: 0.7 * x,
            jnp.exp,
            0.1 * jnp.eye(1),
            0.05 * jnp.eye(1),
            y,
            jnp.zeros(1),
            jnp.eye(1),
            mask=mask,
        )
    assert jnp.array_equal(state.log_likelihood, lm(0.7))


def test_shapes_are_checked():
    prior = _affine()
    with pytest.raises(ValueError, match="y must have shape"):
        prior.log_marginal(Y[:, 0])
    with pytest.raises(ValueError, match="mask has shape"):
        prior.condition(Y, mask=jnp.ones(T - 1, dtype=bool))
