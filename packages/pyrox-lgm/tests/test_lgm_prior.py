"""LGMPrior, the latent field of an LGM without its likelihood."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pyrox_lgm as lgm
import pytest


N = 12
T = np.arange(N)


def _parts():
    return (lgm.RW1(N, name="t"), lgm.IID(3, name="g")), lgm.FixedEffects(("z",))


def _data():
    rng = np.random.default_rng(0)
    return {
        "y": rng.poisson(2.0, N).astype(float),
        "t": T,
        "g": T % 3,
        "z": rng.normal(size=N),
    }


# --- the prior and the model that holds it -----------------------------------


def test_an_lgm_built_from_a_prior_is_the_same_model():
    components, fixed = _parts()
    data, u = _data(), jnp.array([0.3, 0.7, -0.2])
    likelihood = lgm.NegativeBinomial()
    direct = lgm.LGM(components, fixed, likelihood)
    prior = lgm.LGMPrior(components, fixed)
    through = lgm.LGM(prior, likelihood=likelihood)
    assert through.prior is prior
    assert through.consts is prior.consts
    assert list(direct.theta_spec()) == list(through.theta_spec())
    assert list(direct.theta_spec()) == ["t.tau", "g.tau", "lik.size"]
    a = direct.log_posterior_theta(u, data)
    b = through.log_posterior_theta(u, data)
    assert float(a) == float(b)


def test_the_prior_declares_only_the_components_hyperparameters():
    prior = lgm.LGMPrior(*_parts())
    assert list(prior.theta_spec()) == ["t.tau", "g.tau"]
    assert prior.n_theta == 2
    assert prior.n_latent == N + 3 + 1
    assert prior.slices() == {"t": (0, N), "g": (N, N + 3), "z": (N + 3, N + 4)}
    # Keys of the likelihood are ignored, so an LGM's whole theta can be passed.
    theta = {"t.tau": 2.0, "g.tau": 3.0}
    x = jnp.zeros(prior.n_latent)
    a = prior.latent_prior(theta).log_prob(x)
    b = prior.latent_prior({**theta, "lik.size": 5.0}).log_prob(x)
    assert float(a) == float(b)


def test_prior_validation():
    components, fixed = _parts()
    with pytest.raises(ValueError, match="distinct"):
        lgm.LGMPrior((components[0], components[0]))
    with pytest.raises(ValueError, match="observation data"):
        lgm.LGMPrior((lgm.IID(3, name="y"),))
    with pytest.raises(ValueError, match="its own fixed effects"):
        lgm.LGM(lgm.LGMPrior(components, fixed), fixed)
    with pytest.raises(ValueError, match="observation model's name"):
        lgm.LGM(lgm.LGMPrior((lgm.IID(3, name="lik"),)))


# --- rows --------------------------------------------------------------------


def test_the_projector_needs_no_response():
    prior, data = lgm.LGMPrior(*_parts()), _data()
    without = {k: v for k, v in data.items() if k != "y"}
    a, b = prior.projector(data), prior.projector(without)
    assert np.array_equal(a.as_matrix(), b.as_matrix())
    # An intercept alone has no column of its own: the offset sizes it.
    intercept = lgm.LGMPrior((), lgm.FixedEffects(("intercept",)))
    assert intercept.projector({"offset": np.zeros(4)}).as_matrix().shape == (4, 1)
    with pytest.raises(KeyError, match="no number of rows"):
        intercept.projector({})


# --- draws -------------------------------------------------------------------


def test_draws_of_a_proper_prior_have_its_covariance():
    prior = lgm.LGMPrior(
        (lgm.AR1(4, name="t"),), lgm.FixedEffects(("z",), prior_precision=2.0)
    )
    theta = {"t.tau": 1.5, "t.rho": 0.6}
    n = 20_000
    draws = np.asarray(prior.sample(jax.random.key(0), theta, n))
    cov = np.linalg.inv(np.asarray(prior.latent_prior(theta).precision.as_matrix()))
    # Entry (i, j) of a Gaussian sample covariance has variance
    # (c_ii c_jj + c_ij^2) / n; five standard errors, over 25 entries.
    sd = np.sqrt((np.outer(np.diag(cov), np.diag(cov)) + cov**2) / n)
    assert np.all(np.abs(np.cov(draws.T) - cov) < 5.0 * sd)


def test_draws_of_an_intrinsic_component_leave_its_null_space_empty():
    # RW2's null space is the constant and the linear trend; an LGM constrains
    # only the first, but a draw is zero along both.
    prior = lgm.LGMPrior((lgm.RW2(8, name="t"),))
    draws = np.asarray(prior.sample(jax.random.key(1), {"t.tau": 4.0}, 5))
    assert draws.shape == (5, 8)
    trend = np.arange(8) - 3.5
    assert np.allclose(draws.sum(axis=1), 0.0, atol=1e-8)
    assert np.allclose(draws @ trend, 0.0, atol=1e-7)
    assert draws.std() > 0.1


def test_every_component_family_draws():
    import kernellib as kl

    graph = kl.grid_graph((3, 3))
    components = (
        lgm.BYM2(graph, name="b"),
        lgm.SPDE(grid=(3, 3), name="s"),
        lgm.Kronecker(lgm.RW1(4, name="m"), lgm.AR1(3, name="k"), name="kron"),
        lgm.Replicate(lgm.RW1(4, name="r"), 2, name="rep"),
    )
    prior = lgm.LGMPrior(components)
    theta = prior.unflatten(jnp.zeros(prior.n_theta))
    draws = prior.sample(jax.random.key(2), theta, 2)
    empty = lgm.LGMPrior(()).sample(jax.random.key(2), {}, 2)
    assert empty.shape == (2, 0)
    assert draws.shape == (2, prior.n_latent)
    assert bool(jnp.all(jnp.isfinite(draws)))


# --- the log marginal likelihood ---------------------------------------------


def _dense_marginal(prior, theta, data, noise_var):
    """log N(y | 0, A Q^-1 A^T + s^2 I) with the intrinsic part pseudo-inverted."""
    from scipy.stats import multivariate_normal

    cov = np.zeros((prior.n_latent, prior.n_latent))
    for c, (a, b) in zip(prior.components, prior.slices().values(), strict=False):
        theta_c = prior._component_theta(theta, c)
        gmrf = c.prior(theta_c)
        Q = np.asarray(c.assembly_precision(theta_c, gmrf).as_matrix())
        cov[a:b, a:b] = np.linalg.pinv(Q, hermitian=True, rcond=1e-10)
    n = prior.n_fixed
    if n:
        cov[-n:, -n:] = np.eye(n) / prior.fixed.prior_precision
    A = np.asarray(prior.projector(data).as_matrix())
    cov_y = A @ cov @ A.T + noise_var * np.eye(A.shape[0])
    return multivariate_normal(np.zeros(A.shape[0]), cov_y).logpdf(data["y"])


@pytest.mark.parametrize("component", [lgm.RW1(N, name="t"), lgm.AR1(N, name="t")])
def test_log_marginal_is_the_dense_gaussian_density(component):
    prior = lgm.LGMPrior((component,), lgm.FixedEffects(("intercept",)))
    data = _data()
    theta = {"t.tau": 1.7, "t.rho": 0.4}
    ref = _dense_marginal(prior, theta, data, 0.3)
    assert np.isclose(float(prior.log_marginal(theta, data, 0.3)), ref, atol=1e-9)
    # The same evidence, through a Gaussian observation model's precision.
    model = lgm.LGM(prior)
    through = model.log_marginal({**theta, "lik.prec": 1.0 / 0.3}, data)
    assert np.isclose(float(through), ref, atol=1e-9)


def test_log_marginal_reads_the_offset():
    prior = lgm.LGMPrior((lgm.AR1(N, name="t"),))
    data, theta = _data(), {"t.tau": 1.7, "t.rho": 0.4}
    shift = np.linspace(-1.0, 1.0, N)
    a = prior.log_marginal(theta, data, 0.3)
    b = prior.log_marginal(
        theta, {**data, "y": data["y"] + shift, "offset": shift}, 0.3
    )
    assert np.isclose(float(a), float(b), atol=1e-9)


@pytest.mark.parametrize("likelihood", [lgm.Gaussian(), lgm.Poisson()])
def test_log_posterior_theta_is_the_log_marginal_plus_the_hyperprior(likelihood):
    model = lgm.LGM(*_parts(), likelihood)
    data = _data()
    u = jnp.linspace(-0.4, 0.9, model.n_theta)
    whole = model.log_posterior_theta(u, data)
    parts = model.log_marginal(model.unflatten(u), data) + model.log_prior(u)
    assert float(whole) == float(parts)


def test_log_marginal_gradient_matches_central_differences():
    prior, data = lgm.LGMPrior(*_parts()), _data()
    model = lgm.LGM(prior, likelihood=lgm.Poisson())
    h = 1e-5

    def gaussian(v):
        return prior.log_marginal({"t.tau": v, "g.tau": 2.0}, data, 0.5)

    def poisson(v):
        return model.log_marginal({"t.tau": v, "g.tau": 2.0}, data)

    for f in (gaussian, poisson):
        fd = (f(1.3 + h) - f(1.3 - h)) / (2 * h)
        assert np.isclose(float(jax.grad(f)(1.3)), float(fd), rtol=1e-6)


@pytest.mark.parametrize(
    ("component", "index"),
    [
        (lgm.Replicate(lgm.RW1(4, name="r"), 2, name="t"), np.arange(4)),
        (
            lgm.Kronecker(lgm.RW1(4, name="m"), lgm.AR1(3, name="k"), name="t"),
            np.arange(4) * 3,
        ),
    ],
    ids=["replicate", "kronecker"],
)
def test_a_part_that_no_observation_reads_is_an_error(component, index):
    # H is singular along the null direction of the unread part, so its solves
    # are round-off: the new methods raise instead of returning them.
    prior = lgm.LGMPrior((component,), lgm.FixedEffects(("intercept",)))
    theta = prior.unflatten(jnp.zeros(prior.n_theta))
    data = {"y": np.ones(4), "t": index}
    with pytest.raises(Exception, match="no observation reads"):
        prior.log_marginal(theta, data, 0.3)
    with pytest.raises(Exception, match="no observation reads"):
        lgm.LGM(prior, likelihood=lgm.Poisson()).log_marginal(theta, data)
    # With every part read there is no error.
    full = {"y": np.ones(prior.n_latent - 1), "t": np.arange(prior.n_latent - 1)}
    assert np.isfinite(float(prior.log_marginal(theta, full, 0.3)))


def test_the_noise_variance_is_a_scalar():
    with pytest.raises(ValueError, match="noise_var must be a scalar"):
        lgm.LGMPrior(*_parts()).log_marginal({"t.tau": 1.0, "g.tau": 1.0}, _data(), [1])


def test_a_disconnected_graph_is_an_error_only_where_a_part_is_unread():
    import kernellib as kl

    edges = np.array([[0, 1], [1, 2], [3, 4], [4, 5]])
    weights = np.zeros((6, 6))
    weights[edges[:, 0], edges[:, 1]] = weights[edges[:, 1], edges[:, 0]] = 1.0
    graph = kl.graph_from_adjacency(weights)
    for component in (lgm.Besag(graph, name="t"), lgm.BYM2(graph, name="t")):
        prior = lgm.LGMPrior((component,), lgm.FixedEffects(("intercept",)))
        theta = prior.unflatten(jnp.zeros(prior.n_theta))
        both = {"y": np.ones(4), "t": np.array([0, 1, 3, 4])}
        assert np.isfinite(float(prior.log_marginal(theta, both, 0.3)))
    one = {"y": np.ones(4), "t": np.array([0, 1, 2, 0])}
    prior = lgm.LGMPrior((lgm.Besag(graph, name="t"),))
    with pytest.raises(Exception, match="no observation reads"):
        prior.log_marginal({"t.tau": 1.0}, one, 0.3)
