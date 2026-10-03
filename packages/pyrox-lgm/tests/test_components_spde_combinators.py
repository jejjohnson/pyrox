"""SPDE, Kronecker and Replicate against dense / direct gaussx references.

Keys are pinned. The NUTS run at the end is the P7 integration criterion: a
BYM2 Poisson model under NUTS has no divergences.
"""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import kernellib as kl
import numpy as np
import numpyro
import numpyro.distributions as dist
import pyrox_lgm as lgm
import pytest
from numpyro.infer import MCMC, NUTS


jax.config.update("jax_enable_x64", True)


def _square_mesh(m=5):
    """Unit square, m x m vertices, two triangles per cell."""
    xs = np.linspace(0.0, 1.0, m)
    V = np.stack(np.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    tri = []
    for i in range(m - 1):
        for j in range(m - 1):
            a, b, c, d = i * m + j, (i + 1) * m + j, i * m + j + 1, (i + 1) * m + j + 1
            tri += [[a, b, d], [a, d, c]]
    return V, np.asarray(tri)


# --- SPDE -------------------------------------------------------------------


def test_spde_grid_is_gaussx_grid_precision():
    spde = lgm.SPDE(grid=(6, 5), spacing=0.5)
    range_, sigma = 2.0, 1.5
    Q = spde.prior({"range_sigma": jnp.array([range_, sigma])}).precision
    kappa, tau, alpha = gx.matern_spde_params(range_, sigma, 1.0, 2)
    ref = gx.spde_precision_grid((6, 5), kappa, tau, alpha, spacing=0.5)
    assert np.allclose(Q.as_matrix(), ref.as_matrix())


def test_spde_mesh_is_gaussx_fem_precision():
    V, T = _square_mesh()
    spde = lgm.SPDE(mesh=(V, T))
    range_, sigma = 0.4, 2.0
    Q = spde.prior({"range_sigma": jnp.array([range_, sigma])}).precision
    C, G = gx.fem_matrices(V, T)
    kappa, tau, alpha = gx.matern_spde_params(range_, sigma, 1.0, 2)
    ref = gx.spde_precision(C, G, kappa, tau, alpha)
    assert np.allclose(Q.as_matrix(), ref.as_matrix())
    assert spde.n_nodes == V.shape[0]


def test_spde_mesh_projector_interpolates_linear_functions_exactly():
    V, T = _square_mesh()
    spde = lgm.SPDE(mesh=(V, T))
    pts = np.random.default_rng(0).uniform(0.05, 0.95, size=(20, 2))
    A = np.asarray(spde.project_points(pts).as_matrix())
    assert np.allclose(A.sum(axis=1), 1.0)
    f = 2.0 * V[:, 0] - 3.0 * V[:, 1] + 0.5
    assert np.allclose(A @ f, 2.0 * pts[:, 0] - 3.0 * pts[:, 1] + 0.5)


def test_spde_validation():
    with pytest.raises(ValueError, match="exactly one"):
        lgm.SPDE()
    with pytest.raises(ValueError, match="nu"):
        lgm.SPDE(grid=(4, 4), alpha=1)  # nu = 0 in 2-D
    with pytest.raises(ValueError, match="mesh"):
        lgm.SPDE(grid=(4, 4)).project_points(np.zeros((1, 2)))


@pytest.mark.slow
def test_spde_grid_marginal_variance_approaches_sigma_squared():
    # Lattice discretisation error shrinks as the range covers more cells.
    sigma = 2.0
    errs = []
    for cells in (8.0, 16.0):
        n = int(4 * cells)
        spde = lgm.SPDE(grid=(n, n))
        g = spde.prior({"range_sigma": jnp.array([cells, sigma])})
        var = g.marginal_variances().reshape(n, n)[n // 2, n // 2]
        errs.append(abs(float(var) - sigma**2))
    assert errs[1] < errs[0] and errs[1] < 0.05 * sigma**2


# --- Kronecker / Replicate --------------------------------------------------


def test_kronecker_of_proper_factors_is_the_dense_kronecker():
    main = lgm.Leroux(kl.grid_graph((2, 3)), name="space")
    group = lgm.AR1(4, name="time")
    st = lgm.Kronecker(main, group)
    theta = {
        "space_tau": jnp.asarray(2.0),
        "space_rho": jnp.asarray(0.6),
        "time_rho": jnp.asarray(0.8),
    }
    Q = st.prior(theta).precision.as_matrix()
    Qs = main.prior({"tau": 2.0, "rho": 0.6}).precision.as_matrix()
    Qt = group.prior({"tau": 1.0, "rho": 0.8}).precision.as_matrix()  # tau fixed
    assert np.allclose(Q, np.kron(Qs, Qt))
    x = np.random.default_rng(1).normal(size=24)
    ref = 0.5 * (np.linalg.slogdet(Q)[1] - x @ Q @ x - 24 * np.log(2 * np.pi))
    assert np.isclose(float(st.prior(theta).log_prob(jnp.asarray(x))), ref)


def test_kronecker_with_an_intrinsic_main_keeps_its_constraint_at_every_time():
    st = lgm.Kronecker(
        lgm.Besag(kl.grid_graph((3, 3)), name="space"), lgm.AR1(4, name="time")
    )
    theta = {"space_tau": jnp.asarray(1.0), "time_rho": jnp.asarray(0.5)}
    x = st.prior(theta).sample(jax.random.key(0)).reshape(9, 4)  # (space, time)
    assert np.allclose(x.sum(axis=0), 0.0, atol=1e-8)


def test_kronecker_and_replicate_reject_unsupported_factors():
    with pytest.raises(ValueError, match="padding"):
        lgm.Kronecker(lgm.RW2(7), lgm.AR1(3))
    with pytest.raises(ValueError, match="padding"):  # BYM2: n_index != n_nodes
        lgm.Kronecker(lgm.BYM2(kl.grid_graph((2, 2))), lgm.AR1(3))
    both = lgm.Kronecker(lgm.RW1(4, name="a"), lgm.RW1(3, name="b"))
    with pytest.raises(NotImplementedError, match="two intrinsic"):
        both.prior({"a_tau": jnp.asarray(1.0)})


def test_replicate_is_block_diagonal_and_shares_theta():
    rep = lgm.Replicate(lgm.RW1(5, name="trend"), 3)
    assert list(rep.theta_spec()) == ["tau"]
    g = rep.prior({"tau": jnp.asarray(2.0)})
    x = g.sample(jax.random.key(1)).reshape(3, 5)
    assert np.allclose(x.sum(axis=1), 0.0, atol=1e-8)  # each copy constrained
    R = np.asarray(lgm.RW1(5).structure.as_matrix())
    assert np.allclose(g.structure.as_matrix(), np.kron(np.eye(3), R))


# --- NumPyro face: soft constraint scale ------------------------------------


def test_soft_constraint_scale_reaches_the_gmrf():
    from numpyro import handlers

    comp = lgm.RW1(8, name="trend")
    tr = handlers.trace(
        handlers.seed(lambda: comp.sample(soft_constraint_scale=0.05), 0)
    ).get_trace()
    assert tr["trend"]["fn"].soft_constraint_scale == 0.05
    default = handlers.trace(handlers.seed(comp.sample, 0)).get_trace()
    assert default["trend"]["fn"].soft_constraint_scale == 1e-2


@pytest.mark.slow
def test_bym2_poisson_under_nuts_has_no_divergences():
    # P7's integration criterion: BYM2 on a 12 x 12 grid with Poisson counts.
    g = kl.grid_graph((12, 12))
    n = 144
    bym2 = lgm.BYM2(g)
    truth = bym2.prior({"tau": jnp.asarray(2.0), "phi": jnp.asarray(0.7)})
    b_true = truth.sample(jax.random.key(1))[:n]
    E = jnp.full(n, 20.0)
    y = jax.random.poisson(jax.random.key(2), E * jnp.exp(-0.5 + b_true))

    def model(y=None):
        mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
        b = bym2.sample()
        numpyro.sample("y", dist.Poisson(E * jnp.exp(mu + b)), obs=y)

    mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, progress_bar=False)
    mcmc.run(jax.random.key(0), y=y, extra_fields=("diverging",))
    assert int(mcmc.get_extra_fields()["diverging"].sum()) == 0
    # The intercept is recovered (its posterior sd is ~0.03 here).
    assert abs(float(mcmc.get_samples()["mu"].mean()) + 0.5) < 0.15
