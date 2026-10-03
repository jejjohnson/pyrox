"""Temporal and generic components against dense references.

Every reference is closed-form on small dense matrices (numpy), and keys are
pinned: the checks are correctness properties, not sampling statements.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import numpyro
import numpyro.distributions as dist
import pyrox_lgm as lgm
import pytest
from numpyro import handlers


jax.config.update("jax_enable_x64", True)


def _diff_structure(n, order):
    D = np.diff(np.eye(n), order, axis=0)
    return D.T @ D


def _scale(R):
    """R-INLA's scale.model: geometric mean of diag(R+) on the constraint."""
    return np.exp(np.mean(np.log(np.diag(np.linalg.pinv(R, hermitian=True)))))


# --- scale_model ------------------------------------------------------------
# gaussx's generalized_variance_scale factors R + eps I (eps = sqrt(machine eps)
# times the mean diagonal, as R-INLA does), so it agrees with the exact
# pseudo-inverse to ~1e-6 relative; RW2's padding correction raises it to the
# power N/n. Hence rtol=1e-5.


@pytest.mark.parametrize("n", [5, 8])
def test_rw1_scale_matches_dense(n):
    assert np.isclose(float(lgm.RW1(n).scale), _scale(_diff_structure(n, 1)), rtol=1e-5)


@pytest.mark.parametrize("n", [6, 7, 11])
def test_rw2_scale_matches_dense_including_odd_n(n):
    assert np.isclose(float(lgm.RW2(n).scale), _scale(_diff_structure(n, 2)), rtol=1e-5)


def test_scale_model_false_is_unscaled():
    assert float(lgm.RW1(6, scale_model=False).scale) == 1.0
    assert float(lgm.RW2(6, scale_model=False).scale) == 1.0


# --- densities --------------------------------------------------------------


def test_iid_density():
    comp = lgm.IID(5)
    x = jnp.asarray(np.random.default_rng(0).normal(size=5))
    lp = comp.prior({"tau": jnp.asarray(2.5)}).log_prob(x)
    ref = dist.Normal(0.0, 1.0 / np.sqrt(2.5)).log_prob(x).sum()
    assert np.isclose(float(lp), float(ref))


def test_ar1_density_is_the_stationary_gaussian():
    n, tau, rho = 6, 3.0, 0.6
    i = np.arange(n)
    cov = rho ** np.abs(i[:, None] - i[None, :]) / tau
    x = np.random.default_rng(1).normal(size=n)
    lp = (
        lgm.AR1(n)
        .prior({"tau": jnp.asarray(tau), "rho": jnp.asarray(rho)})
        .log_prob(jnp.asarray(x))
    )
    ref = dist.MultivariateNormal(jnp.zeros(n), jnp.asarray(cov)).log_prob(x)
    assert np.isclose(float(lp), float(ref), atol=1e-10)


@pytest.mark.parametrize(("cls", "order", "n"), [(lgm.RW1, 1, 7), (lgm.RW2, 2, 8)])
def test_random_walk_tau_dependence(cls, order, n):
    # Without the tau-free normaliser the log-density is
    # (n - c)/2 log(tau s) - tau s / 2 x'Rx; check its exact tau-dependence.
    comp = cls(n)
    R = _diff_structure(n, order)
    s = float(comp.scale)
    x = np.random.default_rng(2).normal(size=n)
    x -= x.mean()
    if order == 2:
        t = np.arange(n) - (n - 1) / 2
        x -= (x @ t) / (t @ t) * t
    quad = x @ R @ x

    def lp(tau):
        return float(comp.prior({"tau": jnp.asarray(tau)}).log_prob(jnp.asarray(x)))

    c = order
    expected = (n - c) / 2 * np.log(4.0 / 0.5) - (4.0 - 0.5) * s * quad / 2
    assert np.isclose(lp(4.0) - lp(0.5), expected, atol=1e-8)


def test_rw2_odd_padding_never_enters_the_projector():
    comp = lgm.RW2(7)
    assert (comp.n_nodes, comp.n_index) == (8, 7)
    A = np.asarray(comp.projector([0, 6, 3]).as_matrix())
    assert A.shape == (3, 8) and np.all(A[:, 7] == 0)
    with pytest.raises(ValueError, match="out of range"):
        comp.projector([7])


def test_generic_proper_and_intrinsic():
    L = np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    op = lx.MatrixLinearOperator(jnp.asarray(L), lx.positive_semidefinite_tag)
    proper = lgm.Generic(
        lx.MatrixLinearOperator(
            jnp.asarray(L + np.eye(3)), lx.positive_semidefinite_tag
        )
    )
    Q = proper.prior({"tau": jnp.asarray(2.0)}).precision.as_matrix()
    assert np.allclose(Q, 2.0 * (L + np.eye(3)))
    intrinsic = lgm.Generic(op, jnp.ones(3) / np.sqrt(3.0))
    g = intrinsic.prior({"tau": jnp.asarray(2.0)})
    draws = g.sample(jax.random.key(0), (4,))
    assert np.allclose(draws.sum(axis=1), 0.0, atol=1e-8)  # hard constraint
    with pytest.raises(ValueError, match="rows"):
        lgm.Generic(op, jnp.ones(4))


# --- theta_spec -------------------------------------------------------------


@pytest.mark.parametrize(
    "comp", [lgm.IID(3), lgm.RW1(4), lgm.RW2(5), lgm.AR1(4)], ids=type
)
def test_theta_spec_transforms_map_the_reals_into_the_support(comp):
    for prior, transform in comp.theta_spec().values():
        y = transform(jnp.linspace(-5.0, 5.0, 7))
        assert bool(jnp.all(prior.support(y)))


# --- NumPyro face -----------------------------------------------------------


def test_sample_sites_and_shapes():
    def model():
        a = lgm.AR1(6, name="season").sample()
        b = lgm.RW2(7, name="trend").sample(jnp.array([0, 2, 6]))
        numpyro.deterministic("eta", a[:3] + b)

    tr = handlers.trace(handlers.seed(model, 0)).get_trace()
    assert list(tr) == [
        "season_tau",
        "season_rho",
        "season",
        "trend_tau",
        "trend",
        "eta",
    ]
    assert tr["season"]["value"].shape == (6,)
    assert tr["trend"]["value"].shape == (8,)  # the field keeps its padding node
    assert tr["eta"]["value"].shape == (3,)


def test_sample_is_soft_constrained_and_has_a_finite_log_density():
    def model():
        lgm.RW1(10, name="trend").sample()

    tr = handlers.trace(handlers.seed(model, 0)).get_trace()
    site = tr["trend"]
    assert np.isfinite(float(site["fn"].log_prob(site["value"])))
    assert site["fn"].constraint == "soft"
    # The soft constraint pins the sum to ~ N(0, 1e-3^2): tiny but not zero.
    assert abs(float(site["value"].sum())) < 0.05
