"""LGMPrior.condition and condition_nongauss against dense and inla() references."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pyrox_lgm as lgm
import pytest
from pyrox_lgm._condition import row_entries


N = 10
T = np.arange(N)
THETA = {"t.tau": 1.7, "t.rho": 0.4}
NOISE = 0.3


def _prior(component):
    return lgm.LGMPrior((component,), lgm.FixedEffects(("intercept", "z")))


def _data(n_obs=14):
    rng = np.random.default_rng(0)
    index = np.arange(n_obs) % (N - 2)  # the last two nodes are never observed
    return {
        "y": np.sin(index / 2.0) + 0.5 + 0.3 * rng.normal(size=n_obs),
        "t": index,
        "z": rng.normal(size=n_obs),
        "offset": rng.normal(size=n_obs),
    }


# New rows: unobserved nodes, and a node with a covariate it never met.
NEW = {
    "t": np.array([N - 1, N - 2, 0, 5]),
    "z": np.array([0.0, 2.0, -3.0, 0.5]),
    "offset": np.array([0.0, 0.1, 0.2, -1.0]),
}


def _dense_posterior(prior, theta, data, noise_var):
    """Mean and covariance of x | y, kriged onto the hard constraints."""
    latent = prior.latent_prior(theta)
    intrinsic = hasattr(latent, "null_space")
    Q = np.asarray(
        latent.structure.as_matrix() if intrinsic else latent.precision.as_matrix()
    )
    A = np.asarray(prior.projector(data).as_matrix())
    H = Q + A.T @ A / noise_var
    cov = np.linalg.inv(H)
    if intrinsic:
        V = np.asarray(latent.null_space)
        W = cov @ V
        cov = cov - W @ np.linalg.solve(V.T @ W, W.T)
    residual = data["y"] - data["offset"]
    return cov @ (A.T @ residual) / noise_var, cov


COMPONENTS = [lgm.RW1(N, name="t"), lgm.RW2(N, name="t"), lgm.AR1(N, name="t")]


# --- Gaussian noise: exact ---------------------------------------------------


@pytest.mark.parametrize("component", COMPONENTS, ids=["rw1", "rw2", "ar1"])
def test_condition_is_the_dense_posterior(component):
    prior, data = _prior(component), _data()
    posterior = prior.condition(THETA, data, NOISE)
    mean, cov = _dense_posterior(prior, THETA, data, NOISE)
    latent_mean, latent_var = posterior.predict_latent()
    # H has the fixed effects' 1e-3 beside 1/0.3 * n_obs: condition number
    # around 1e5, so double precision leaves about 1e-10.
    assert np.allclose(latent_mean, mean, rtol=1e-8, atol=1e-9)
    assert np.allclose(latent_var, np.diag(cov), rtol=1e-8, atol=1e-9)
    for rows in (data, NEW):
        A = np.asarray(prior.projector(rows).as_matrix())
        eta, var = posterior.predict(rows)
        assert np.allclose(eta, rows["offset"] + A @ mean, rtol=1e-8, atol=1e-9)
        assert np.allclose(var, np.diag(A @ cov @ A.T), rtol=1e-8, atol=1e-9)
    eta_noisy, var_noisy = posterior.predict(NEW, include_noise=True)
    assert np.array_equal(eta, eta_noisy)
    assert np.allclose(var_noisy, var + NOISE, rtol=1e-12)


def test_condition_carries_the_evidence_and_its_hyperparameters():
    prior, data = _prior(COMPONENTS[0]), _data()
    posterior = prior.condition(THETA, data, NOISE)
    evidence = prior.log_marginal(THETA, data, NOISE)
    assert float(posterior.log_marginal) == float(evidence)
    assert float(posterior.theta["t.tau"]) == THETA["t.tau"]
    assert posterior.prior is prior


def test_the_gaussian_observation_model_gives_the_same_posterior():
    prior, data = _prior(COMPONENTS[0]), _data()
    theta = {**THETA, "lik.prec": 1.0 / NOISE}
    laplace = prior.condition_nongauss(theta, lgm.Gaussian(), data)
    assert bool(laplace.converged)
    for a, b in zip(
        laplace.predict(NEW),
        prior.condition(THETA, data, NOISE).predict(NEW),
        strict=True,
    ):
        assert np.allclose(a, b, rtol=1e-8, atol=1e-9)


def test_draws_have_the_posterior_moments_and_meet_the_constraint():
    prior, data = _prior(COMPONENTS[0]), _data()
    posterior = prior.condition(THETA, data, NOISE)
    n = 20_000
    latent = np.asarray(posterior.sample(jax.random.key(0), n))
    assert np.allclose(latent[:, :N].sum(axis=1), 0.0, atol=1e-9)
    eta = np.asarray(posterior.sample(jax.random.key(0), n, NEW))
    mean, var = (np.asarray(v) for v in posterior.predict(NEW))
    # Five standard errors of a Gaussian mean and variance.
    assert np.all(np.abs(eta.mean(axis=0) - mean) < 5.0 * np.sqrt(var / n))
    assert np.all(np.abs(eta.var(axis=0) - var) < 5.0 * var * np.sqrt(2.0 / n))
    # The two kinds of draw come from the same latent vectors.
    A = np.asarray(prior.projector(NEW).as_matrix())
    assert np.allclose(eta, latent @ A.T + NEW["offset"], atol=1e-10)


# --- any likelihood: Laplace -------------------------------------------------


def _counts():
    data = _data()
    return {**data, "y": np.random.default_rng(1).poisson(np.exp(data["y"]))}


def test_condition_nongauss_is_stationary_at_its_mode():
    prior, data = _prior(COMPONENTS[2]), _counts()
    posterior = prior.condition_nongauss(THETA, lgm.Poisson(), data)
    assert bool(posterior.converged)
    A = np.asarray(prior.projector(data).as_matrix())
    Q = np.asarray(prior.latent_prior(THETA).precision.as_matrix())
    mode = np.asarray(posterior.fit.mode)
    rate = np.exp(A @ mode + data["offset"])
    # grad log p(y, x) = A^T (y - rate) - Q x, zero at the mode; the Newton
    # loop stops on a relative step of 1e-8.
    assert np.allclose(A.T @ (data["y"] - rate) - Q @ mode, 0.0, atol=1e-6)
    cov = np.linalg.inv(Q + A.T @ (rate[:, None] * A))
    A_new = np.asarray(prior.projector(NEW).as_matrix())
    _, var = posterior.predict(NEW)
    assert np.allclose(var, np.diag(A_new @ cov @ A_new.T), rtol=1e-7)
    evidence = lgm.LGM(prior, likelihood=lgm.Poisson()).log_marginal(THETA, data)
    assert float(posterior.log_marginal_approx) == float(evidence)


def test_an_unconverged_fit_says_so():
    prior, data = _prior(COMPONENTS[2]), _counts()
    posterior = prior.condition_nongauss(THETA, lgm.Poisson(), data, max_newton=1)
    assert not bool(posterior.converged)


# --- transforms --------------------------------------------------------------


def test_condition_under_jit_and_its_gradient():
    prior, data = _prior(COMPONENTS[0]), _data()
    A = prior.projector(data)

    @eqx.filter_jit
    def predict(tau):
        theta = {"t.tau": tau}
        return prior.condition(theta, data, NOISE, projector=A).predict(NEW)

    eager = prior.condition({"t.tau": 1.3}, data, NOISE).predict(NEW)
    for a, b in zip(predict(jnp.asarray(1.3)), eager, strict=True):
        assert np.allclose(a, b, rtol=1e-10)

    def total(tau):
        mean, var = predict(tau)
        return jnp.sum(mean) + jnp.sum(var)

    h = 1e-5
    fd = (total(1.3 + h) - total(1.3 - h)) / (2 * h)
    assert np.isclose(float(jax.grad(total)(1.3)), float(fd), rtol=1e-6)


def test_a_conditioned_model_returns_from_jit():
    prior, data = _prior(COMPONENTS[0]), _data()
    posterior = eqx.filter_jit(lambda v: prior.condition(THETA, data, v))(NOISE)
    for a, b in zip(
        posterior.predict(NEW),
        prior.condition(THETA, data, NOISE).predict(NEW),
        strict=True,
    ):
        assert np.allclose(a, b, rtol=1e-10)


def test_row_entries_rebuild_the_matrix():
    prior = lgm.LGMPrior((lgm.SPDE(grid=(3, 3), name="s"), lgm.IID(2, name="g")))
    rows = {"s": np.array([[0.2, 0.7], [0.5, 0.5]]), "g": np.array([1, 0])}
    A = prior.projector(rows)
    cols, vals = row_entries(A)
    dense = np.zeros((2, prior.n_latent))
    for i in range(2):
        np.add.at(dense[i], np.asarray(cols[i]), np.asarray(vals[i]))
    assert np.allclose(dense, A.as_matrix())


# --- against inla() ----------------------------------------------------------


@pytest.mark.slow
def test_condition_nongauss_is_what_inla_mixes():
    prior, data = _prior(COMPONENTS[0]), _counts()
    model = lgm.LGM(prior, likelihood=lgm.Poisson())
    res = lgm.inla(model, data, strategy="gaussian")
    k = int(jnp.argmax(res.theta_weights))
    theta = model.unflatten(res.theta_points[k])
    posterior = prior.condition_nongauss(theta, lgm.Poisson(), data)
    mean, var = posterior.predict_latent()
    assert np.allclose(mean, res.latent_means[k], rtol=1e-7, atol=1e-9)
    assert np.allclose(var, res.latent_variances[k], rtol=1e-7, atol=1e-9)
    eta, eta_var = posterior.predict(data)
    assert np.allclose(eta, res.predictor_means[k], rtol=1e-7, atol=1e-9)
    assert np.allclose(eta_var, res.predictor_variances[k], rtol=1e-7, atol=1e-9)
