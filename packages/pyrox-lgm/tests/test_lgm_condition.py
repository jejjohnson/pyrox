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
    # restrict to the complement of the constraints: H may be singular off it
    N = np.eye(Q.shape[0])
    if intrinsic:
        V = np.asarray(latent.hard_constraint)
        N = np.linalg.svd(V)[0][:, V.shape[1] :]
    cov = N @ np.linalg.inv(N.T @ H @ N) @ N.T
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


# --- what the data must pin down ---------------------------------------------


SEVERAL = {
    # two intrinsic components: their constants cancel in the predictor
    "two_walks": (lgm.RW1(N, name="t"), lgm.RW1(4, name="w")),
    "walk_and_trend": (lgm.RW2(N, name="t"), lgm.RW1(4, name="w")),
    "three": (
        lgm.RW2(N + 1, name="t"),
        lgm.RW1(4, name="w"),
        lgm.Kronecker(lgm.RW1(4, name="m"), lgm.AR1(3, name="k"), name="g"),
    ),
    # a replicate, and a slice of a Kronecker, that no observation reads
    "unread_replicate": (lgm.Replicate(lgm.RW1(N, name="r"), 2, name="t"),),
    "unread_slice": (
        lgm.RW1(N, name="t"),
        lgm.Kronecker(lgm.RW1(4, name="m"), lgm.AR1(5, name="k"), name="g"),
    ),
}


def _several(name):
    components = SEVERAL[name]
    rng = np.random.default_rng(3)
    n_obs = 40
    data = {
        "y": rng.normal(size=n_obs),
        "t": rng.integers(0, N - 2, n_obs),
        "w": rng.integers(0, 4, n_obs),
        # time slices 0 to 2 only
        "g": rng.integers(0, 4, n_obs) * components[-1].group.n_index
        + rng.integers(0, 3, n_obs)
        if name in ("three", "unread_slice")
        else None,
        "offset": np.zeros(n_obs),
    }
    return components, {k: v for k, v in data.items() if v is not None}


@pytest.mark.parametrize("fixed", [(), ("intercept",)], ids=["bare", "intercept"])
@pytest.mark.parametrize("name", SEVERAL)
def test_a_singular_precision_still_gives_the_exact_posterior(name, fixed):
    components, data = _several(name)
    prior = lgm.LGMPrior(components, lgm.FixedEffects(fixed) if fixed else None)
    theta = prior.unflatten(0.3 * jnp.ones(prior.n_theta))
    mean, cov = _dense_posterior(prior, theta, data, NOISE)
    posterior = prior.condition(theta, data, NOISE)
    got_mean, got_var = posterior.predict_latent()
    assert np.allclose(got_mean, mean, atol=1e-8)
    assert np.allclose(got_var, np.diag(cov), atol=1e-8)
    # joint draws: the variance of 4000 is within 7 of its own standard errors
    draws = np.asarray(posterior.sample(jax.random.key(1), 4000))
    free = np.diag(cov) > 1e-10
    ratio = draws.var(0)[free] / np.diag(cov)[free]
    assert np.all(np.abs(ratio - 1.0) < 7 * np.sqrt(2 / 4000))
    # the linear predictor at every index of the first component
    new = {k: np.zeros(N, dtype=int) for k in data if k not in ("y", "offset")}
    new = new | {"t": np.arange(N)}
    A = np.asarray(prior.projector(new).as_matrix())
    eta, eta_var = posterior.predict(new)
    assert np.allclose(eta, A @ mean, atol=1e-8)
    assert np.allclose(eta_var, np.diag(A @ cov @ A.T), atol=1e-8)


def test_an_improper_posterior_is_an_error():
    # An RW2 leaves its trend to the data. Nothing fixes it when the only reading
    # is at the centre, where the trend is zero, or in a replicate with none.
    prior = lgm.LGMPrior((lgm.RW2(7, name="t"),))
    data = {"y": np.ones(3), "t": np.array([3, 3, 3])}
    with pytest.raises(Exception, match="posterior is improper"):
        prior.condition({"t.tau": 1.0}, data, NOISE)
    with pytest.raises(Exception, match="posterior is improper"):
        prior.condition_nongauss({"t.tau": 1.0}, lgm.Poisson(), data)
    replicate = lgm.LGMPrior((lgm.Replicate(lgm.RW2(6, name="r"), 2, name="t"),))
    with pytest.raises(Exception, match="posterior is improper"):
        replicate.condition({"t.tau": 1.0}, {"y": np.ones(6), "t": np.arange(6)}, NOISE)


def test_one_reading_and_the_sum_fix_a_trend():
    # Off the centre, one reading and the sum-to-zero constraint are enough, under
    # jit too.
    prior = lgm.LGMPrior((lgm.RW2(6, name="t"),))
    data = {"y": np.ones(3), "t": np.array([2, 2, 2]), "offset": np.zeros(3)}
    mean, cov = _dense_posterior(prior, {"t.tau": 1.0}, data, NOISE)

    @jax.jit
    def moments(tau):
        return prior.condition({"t.tau": tau}, data, NOISE).predict_latent()

    got_mean, got_var = moments(1.0)
    assert np.allclose(got_mean, mean, atol=1e-8)
    assert np.allclose(got_var, np.diag(cov), rtol=1e-8)


@pytest.mark.slow
def test_inla_fits_two_intrinsic_components():
    components, data = _several("two_walks")
    model = lgm.LGM(
        lgm.LGMPrior(components, lgm.FixedEffects(("intercept",))),
        likelihood=lgm.Gaussian(),
    )
    res = lgm.inla(model, data, strategy="gaussian")
    k = int(jnp.argmax(res.theta_weights))
    theta = model.unflatten(res.theta_points[k])
    noise_var = 1.0 / float(theta["lik.prec"])
    mean, cov = _dense_posterior(model.prior, theta, data, noise_var)
    assert np.allclose(res.latent_means[k], mean, atol=1e-6)
    assert np.allclose(res.latent_variances[k], np.diag(cov), rtol=1e-6)


def test_a_projector_lets_an_spde_predict_under_jit():
    prior = lgm.LGMPrior((lgm.SPDE(grid=(4, 4), name="s"),))
    points = np.random.default_rng(0).uniform(0.2, 2.8, size=(6, 2))
    data = {"y": np.ones(6), "s": points}
    new = {"s": points[:2] + 0.05}
    A, A_new = prior.projector(data), prior.projector(new)

    @eqx.filter_jit
    def predict(sigma):
        theta = {"s.range_sigma": jnp.array([1.5, 1.0]) * sigma}
        posterior = prior.condition(theta, data, NOISE, projector=A)
        draws = posterior.sample(jax.random.key(0), 3, new, projector=A_new)
        return *posterior.predict(new, projector=A_new), draws

    theta = {"s.range_sigma": jnp.array([1.5, 1.0])}
    eager = prior.condition(theta, data, NOISE).predict(new)
    mean, var, draws = predict(jnp.asarray(1.0))
    assert np.allclose(mean, eager[0], rtol=1e-8)
    assert np.allclose(var, eager[1], rtol=1e-8)
    assert draws.shape == (3, 2)


def test_the_result_keeps_only_the_hyperparameters_it_used():
    prior, data = _prior(COMPONENTS[0]), _counts()
    theta = {**THETA, "lik.size": 2.0, "note": "ignored"}
    posterior = prior.condition_nongauss(theta, lgm.NegativeBinomial(), data)
    assert set(posterior.theta) == {"t.tau", "lik.size"}
