"""Areal components against dense references (numpy), pinned keys."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import kernellib as kl
import numpy as np
import numpyro.distributions as dist
import pyrox_lgm as lgm
import pytest
from numpyro import handlers


jax.config.update("jax_enable_x64", True)


def _irregular_graph():
    """Six areas: a 4-cycle with a chord, plus a pendant path 3 - 4 - 5."""
    s = [0, 1, 2, 3, 0, 3, 4]
    r = [1, 2, 3, 0, 2, 4, 5]
    return kl.graph_from_edges(s, r, 6)


def _dense_laplacian(graph):
    if isinstance(graph, kl.GridGraph):
        graph = kl.Graph(graph.topology, graph.weights)
    return np.asarray(graph.laplacian_operator().as_matrix())


GRAPHS = {"irregular": _irregular_graph, "grid": lambda: kl.grid_graph((3, 4))}


def _mvn_logpdf(x, Q):
    _, logdet = np.linalg.slogdet(Q)
    return 0.5 * (logdet - x @ Q @ x - x.size * np.log(2 * np.pi))


# --- CAR / Leroux: proper, dense densities ----------------------------------


@pytest.mark.parametrize("graph", list(GRAPHS))
@pytest.mark.parametrize(("tau", "rho"), [(1.5, 0.3), (0.4, 0.95)])
def test_car_density_matches_dense(graph, tau, rho):
    g = GRAPHS[graph]()
    L = _dense_laplacian(g)
    D = np.diag(np.diag(L))
    W = D - L
    Q = tau * (D - rho * W)
    x = np.random.default_rng(0).normal(size=L.shape[0])
    gmrf = lgm.CAR(g).prior({"tau": jnp.asarray(tau), "rho": jnp.asarray(rho)})
    assert np.allclose(gmrf.precision.as_matrix(), Q)
    assert np.isclose(
        float(gmrf.log_prob(jnp.asarray(x))), _mvn_logpdf(x, Q), atol=1e-9
    )


@pytest.mark.parametrize("graph", list(GRAPHS))
@pytest.mark.parametrize(("tau", "rho"), [(1.5, 0.3), (0.4, 0.95), (2.0, 0.0)])
def test_leroux_density_matches_dense(graph, tau, rho):
    g = GRAPHS[graph]()
    L = _dense_laplacian(g)
    Q = tau * (rho * L + (1 - rho) * np.eye(L.shape[0]))
    x = np.random.default_rng(1).normal(size=L.shape[0])
    gmrf = lgm.Leroux(g).prior({"tau": jnp.asarray(tau), "rho": jnp.asarray(rho)})
    assert np.allclose(gmrf.precision.as_matrix(), Q)
    assert np.isclose(
        float(gmrf.log_prob(jnp.asarray(x))), _mvn_logpdf(x, Q), atol=1e-9
    )


def test_leroux_stays_a_kronecker_sum_on_a_grid():
    import gaussx as gx

    gmrf = lgm.Leroux(kl.grid_graph((3, 4))).prior(
        {"tau": jnp.asarray(1.0), "rho": jnp.asarray(0.5)}
    )
    assert isinstance(gmrf.precision, gx.KroneckerSum)


def test_car_rejects_isolated_nodes():
    with pytest.raises(ValueError, match="isolated"):
        lgm.CAR(kl.graph_from_edges([0], [1], 3))


# --- Besag ------------------------------------------------------------------


def test_besag_constraints_per_connected_component():
    g = kl.graph_from_edges([0, 1, 3], [1, 2, 4], 5)  # {0,1,2} and {3,4}
    comp = lgm.Besag(g)
    assert comp.null_space.shape == (5, 2)
    x = comp.prior({"tau": jnp.asarray(2.0)}).sample(jax.random.key(0), (3,))
    assert np.allclose(x[:, :3].sum(axis=1), 0.0, atol=1e-8)
    assert np.allclose(x[:, 3:].sum(axis=1), 0.0, atol=1e-8)


@pytest.mark.parametrize("graph", list(GRAPHS))
def test_besag_tau_dependence(graph):
    g = GRAPHS[graph]()
    comp = lgm.Besag(g)
    R = np.asarray(comp.structure.as_matrix())
    x = np.random.default_rng(2).normal(size=R.shape[0])
    x -= x.mean()  # connected: one sum-to-zero constraint
    quad = x @ R @ x

    def lp(tau):
        return float(comp.prior({"tau": jnp.asarray(tau)}).log_prob(jnp.asarray(x)))

    n = R.shape[0]
    expected = (n - 1) / 2 * np.log(3.0 / 0.5) - (3.0 - 0.5) * quad / 2
    assert np.isclose(lp(3.0) - lp(0.5), expected, atol=1e-8)


# --- BYM2 -------------------------------------------------------------------


@pytest.mark.parametrize("graph", list(GRAPHS))
def test_bym2_marginal_variances_are_the_dense_bym2_ones(graph):
    g = GRAPHS[graph]()
    comp = lgm.BYM2(g)
    n = comp.n_index
    Rp = np.linalg.pinv(np.asarray(comp.structure.as_matrix()), hermitian=True)
    tau, phi = 2.0, 0.3
    var = comp.prior({"tau": jnp.asarray(tau), "phi": jnp.asarray(phi)})
    var = np.asarray(var.marginal_variances())
    assert np.allclose(var[:n], ((1 - phi) + phi * np.diag(Rp)) / tau, rtol=1e-6)
    assert np.allclose(var[n:], np.diag(Rp), rtol=1e-6)


def test_bym2_grid_and_graph_agree():
    # A grid built as a GridGraph (Kronecker spectrum) and as a plain Graph
    # (dense spectrum) is the same BYM2 model.
    grid = kl.grid_graph((3, 4))
    a = lgm.BYM2(grid)
    b = lgm.BYM2(kl.Graph(grid.topology, grid.weights))
    assert np.allclose(a.structure.as_matrix(), b.structure.as_matrix())
    assert np.isclose(float(a.log_pdet), float(b.log_pdet))
    phis = jnp.array([0.1, 0.5, 0.9])
    assert np.allclose(a.phi_prior.log_prob(phis), b.phi_prior.log_prob(phis))


def test_bym2_projector_sees_b_only():
    comp = lgm.BYM2(_irregular_graph())
    A = np.asarray(comp.projector([0, 5]).as_matrix())
    assert A.shape == (2, 12) and np.all(A[:, 6:] == 0)
    with pytest.raises(ValueError, match="out of range"):
        comp.projector([6])


# --- theta_spec / NumPyro face ----------------------------------------------


@pytest.mark.parametrize(
    "comp",
    [
        lgm.Besag(_irregular_graph()),
        lgm.BYM2(_irregular_graph()),
        lgm.CAR(_irregular_graph()),
        lgm.Leroux(_irregular_graph()),
    ],
    ids=type,
)
def test_theta_transforms_and_sample_sites(comp):
    for prior, transform in comp.theta_spec().values():
        assert bool(jnp.all(prior.support(transform(jnp.linspace(-5.0, 5.0, 7)))))
    tr = handlers.trace(handlers.seed(comp.sample, 0)).get_trace()
    assert list(tr) == [f"{comp.name}_{k}" for k in comp.theta_spec()] + [comp.name]
    assert tr[comp.name]["value"].shape == (comp.n_nodes,)
    out = handlers.seed(comp.sample, 0)()
    assert out.shape == (comp.n_index,)
    site = tr[comp.name]
    assert np.isfinite(float(site["fn"].log_prob(site["value"])))


def test_default_rho_prior_is_uniform():
    assert isinstance(lgm.CAR(_irregular_graph()).rho_prior, dist.Uniform)
