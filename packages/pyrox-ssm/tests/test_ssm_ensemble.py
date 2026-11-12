"""Tests for `pyrox_ssm.ensemble.ensemble_factor` (the ``ensemble`` extra).

Pinned here: the factor is the sum of filterax's per-step log likelihoods; for
linear-Gaussian dynamics it equals the exact Kalman log marginal of
`pyrox_ssm.lgssm_factor` when the initial ensemble matches the prior's first
two moments, and is within Monte Carlo error of it for a random ensemble of
E = 2000; the log density is identical on repeated calls; stochastic filters
and inflators raise with the reason; and the factor runs inside SVI and NUTS.
"""

from __future__ import annotations

import filterax as flx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpyro
import numpyro.distributions as dist
import pyrox_ssm as pss
import pytest
from jaxtyping import Array, Float
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from pyrox_ssm.ensemble import ensemble_factor


T, D, M = 10, 2, 1
A = jnp.array([[0.9, 0.2], [-0.1, 0.8]])
H = jnp.array([[1.0, 0.5]])
R = 0.3 * jnp.eye(M)
M0 = jnp.array([0.5, -0.3])
P0 = jnp.array([[1.0, 0.3], [0.3, 0.5]])
OBS_TIMES = jnp.arange(1.0, T + 1.0)


class LinearDynamics(flx.AbstractDynamics):
    transition: Float[Array, "D D"]

    def __call__(self, state, t0, t1):
        return self.transition @ state


class LinearObs(flx.AbstractObsOperator):
    obs_model: Float[Array, "M D"]

    def __call__(self, state):
        return self.obs_model @ state


def _observations() -> Float[Array, "T M"]:
    """One draw of y from the linear-Gaussian model with no process noise."""
    key_x, key_y = jr.split(jr.key(0))
    x = M0 + jnp.linalg.cholesky(P0) @ jr.normal(key_x, (D,))
    states = []
    for _ in range(T):
        x = A @ x
        states.append(x)
    noise = jnp.sqrt(R[0, 0]) * jr.normal(key_y, (T, M))
    return jnp.stack(states) @ H.T + noise


def _kalman_log_marginal(y: Float[Array, "T M"]) -> Float[Array, ""]:
    """The exact log marginal, as the factor `pss.lgssm_factor` registers.

    The ensemble filter's dynamics are deterministic, so the reference has
    zero process noise; both forecast before the first update.
    """
    prior = pss.LinearGaussianSSMPrior(
        jnp.broadcast_to(A, (T, D, D)), H, jnp.zeros((T, D, D)), R, M0, P0
    )
    trace = handlers.trace(lambda: pss.lgssm_factor("y", prior, y)).get_trace()
    return trace["y"]["fn"].log_factor


def _random_ensemble(n_members: int, seed: int) -> Float[Array, "E D"]:
    z = jr.normal(jr.key(seed), (n_members, D))
    return M0 + z @ jnp.linalg.cholesky(P0).T


def _moment_matched_ensemble(n_members: int) -> Float[Array, "E D"]:
    """An ensemble whose mean is m0 and whose (Bessel) covariance is P0."""
    z = jr.normal(jr.key(1), (n_members, D))
    z = z - z.mean(axis=0)
    sample_cov = z.T @ z / (n_members - 1)
    z = z @ jnp.linalg.inv(jnp.linalg.cholesky(sample_cov)).T
    return M0 + z @ jnp.linalg.cholesky(P0).T


def _ensemble_factor(init_ensemble, y, obs_noise=R):
    return ensemble_factor(
        "y",
        flx.filters.ETKF(),
        LinearDynamics(A),
        LinearObs(H),
        init_ensemble,
        y,
        OBS_TIMES,
        obs_noise,
    )


def _ensemble_log_density(init_ensemble, y, obs_noise=R):
    model = lambda: _ensemble_factor(init_ensemble, y, obs_noise)
    trace = handlers.trace(model).get_trace()
    return trace["y"]["fn"].log_factor, trace


def test_registers_one_factor_equal_to_the_summed_log_likelihoods():
    y = _observations()
    X0 = _random_ensemble(32, seed=0)
    log_factor, trace = _ensemble_log_density(X0, y)
    assert list(trace) == ["y"]
    assert log_factor.shape == ()
    returned = _ensemble_factor(X0, y)
    assert isinstance(returned, flx.AssimilationResult)
    assert returned.log_likelihoods.shape == (T,)
    assert returned.analysis_history.shape == (T, 32, D)
    # the same computation, so equal to rounding only
    assert jnp.allclose(log_factor, returned.log_likelihoods.sum(), rtol=1e-12)


def test_moment_matched_ensemble_gives_the_exact_kalman_log_marginal():
    # A square-root filter on a linear-Gaussian model propagates the ensemble's
    # first two moments exactly, so with an ensemble whose moments are m0 and
    # P0 the ensemble log likelihood is the Kalman one; 1e-10 is the x64
    # tolerance the package uses for exact equivalences.
    y = _observations()
    log_factor, _ = _ensemble_log_density(_moment_matched_ensemble(50), y)
    assert jnp.allclose(log_factor, _kalman_log_marginal(y), rtol=1e-10)


def test_large_random_ensemble_is_within_monte_carlo_error_of_kalman():
    # rtol = 5e-2 is the acceptance tolerance set for E = 2000; over five
    # seeds of the initial ensemble the measured relative gap was 1e-3 to 5e-3.
    y = _observations()
    log_factor, _ = _ensemble_log_density(_random_ensemble(2000, seed=0), y)
    assert jnp.allclose(log_factor, _kalman_log_marginal(y), rtol=5e-2)


def test_obs_noise_matrix_and_operator_agree():
    y = _observations()
    X0 = _random_ensemble(32, seed=0)
    from_matrix, _ = _ensemble_log_density(X0, y, obs_noise=R)
    operator = lx.MatrixLinearOperator(R, lx.positive_semidefinite_tag)
    from_operator, _ = _ensemble_log_density(X0, y, obs_noise=operator)
    assert jnp.allclose(from_matrix, from_operator, rtol=1e-12)


def test_same_parameters_and_ensemble_give_an_identical_log_density():
    y = _observations()
    X0 = _random_ensemble(32, seed=0)

    def model():
        scale = numpyro.sample("scale", dist.Uniform(0.5, 1.0))
        ensemble_factor(
            "y",
            flx.filters.ETKF(),
            LinearDynamics(scale * A),
            LinearObs(H),
            X0,
            y,
            OBS_TIMES,
            R,
        )

    fixed = handlers.condition(model, {"scale": 0.9})

    def log_density(seed):
        trace = handlers.trace(handlers.seed(fixed, seed)).get_trace()
        return trace["y"]["fn"].log_factor

    log_densities = [log_density(seed) for seed in (0, 1, 0)]
    # bit-for-bit: no random numbers are drawn inside the factor
    assert all(jnp.array_equal(log_densities[0], v) for v in log_densities[1:])
    jitted = jax.jit(lambda: log_density(2))
    assert jnp.array_equal(jitted(), jitted())


@pytest.mark.parametrize(
    ("filter_", "inflator", "culprit"),
    [
        (flx.filters.StochasticEnKF(), None, "StochasticEnKF"),
        (flx.filters.ETKF_Livings(), None, "ETKF_Livings"),
        (
            flx.filters.ETKF(),
            flx.AdditiveInflator(lx.DiagonalLinearOperator(jnp.ones(D)), jr.key(0)),
            "AdditiveInflator",
        ),
    ],
)
def test_stochastic_components_raise_with_the_reason(filter_, inflator, culprit):
    with pytest.raises(ValueError, match="common random numbers") as info:
        ensemble_factor(
            "y",
            filter_,
            LinearDynamics(A),
            LinearObs(H),
            _random_ensemble(8, seed=0),
            _observations(),
            OBS_TIMES,
            R,
            inflator=inflator,
        )
    assert culprit in str(info.value)
    # filterax's own refusal is kept as the cause
    assert isinstance(info.value.__cause__, ValueError)


# A 2-parameter toy for SVI and NUTS: a damped rotation x_k = a Rot(w) x_{k-1}
# of a 2-D state, observed in its first coordinate.
TOY_T, TOY_E = 20, 32
TOY_TIMES = jnp.arange(1.0, TOY_T + 1.0)
TOY_R = 0.05 * jnp.eye(1)
TOY_TRUTH = {"a": 0.9, "w": 0.5}


class DampedRotation(flx.AbstractDynamics):
    a: Float[Array, ""]
    w: Float[Array, ""]

    def __call__(self, state, t0, t1):
        c, s = jnp.cos(self.w), jnp.sin(self.w)
        return self.a * jnp.array([[c, -s], [s, c]]) @ state


def _toy_data():
    X0 = jnp.array([1.0, 0.0]) + 0.1 * jr.normal(jr.key(3), (TOY_E, 2))
    dynamics = DampedRotation(jnp.asarray(TOY_TRUTH["a"]), jnp.asarray(TOY_TRUTH["w"]))
    x, states = jnp.array([1.0, 0.0]), []
    for k in range(TOY_T):
        x = dynamics(x, TOY_TIMES[k] - 1.0, TOY_TIMES[k])
        states.append(x[:1])
    y = jnp.stack(states) + jnp.sqrt(TOY_R[0, 0]) * jr.normal(jr.key(4), (TOY_T, 1))
    return X0, y


def _toy_model(X0, y):
    a = numpyro.sample("a", dist.Uniform(0.5, 1.0))
    w = numpyro.sample("w", dist.Uniform(0.0, 1.0))
    ensemble_factor(
        "y",
        flx.filters.ETKF(),
        DampedRotation(a, w),
        lambda x: x[:1],
        X0,
        y,
        TOY_TIMES,
        TOY_R,
    )


def test_runs_inside_svi_on_a_two_parameter_model():
    X0, y = _toy_data()
    svi = SVI(_toy_model, AutoDelta(_toy_model), numpyro.optim.Adam(0.02), Trace_ELBO())
    result = svi.run(jr.key(0), 300, X0, y, progress_bar=False)
    assert jnp.all(jnp.isfinite(result.losses))
    assert result.losses[-1] < result.losses[0]


def test_nuts_draws_100_samples():
    # about 5 s on the dev VM, so it stays in the fast tier
    X0, y = _toy_data()
    mcmc = MCMC(NUTS(_toy_model), num_warmup=100, num_samples=100, progress_bar=False)
    mcmc.run(jr.key(0), X0, y)
    samples = mcmc.get_samples()
    assert samples["a"].shape == samples["w"].shape == (100,)
    assert all(jnp.all(jnp.isfinite(v)) for v in samples.values())
