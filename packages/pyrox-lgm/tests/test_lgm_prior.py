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
