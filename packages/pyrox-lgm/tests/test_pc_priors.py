"""PC priors: each calibration statement holds, and each density integrates to 1.

Calibration and normalisation are checked by adaptive quadrature
(``scipy.integrate.quad``) of ``exp(log_prob)``, so the tolerances are
quadrature tolerances, not sampling ones. The one sampling check bounds the
empirical tail frequency by its binomial standard error.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import kernellib as kl
import numpy as np
import pyrox_lgm as lgm
import pytest
from scipy import integrate


jax.config.update("jax_enable_x64", True)


def _quad(f, a, b, **kw):
    return integrate.quad(lambda x: float(f(x)), a, b, limit=200, **kw)[0]


def _density(prior):
    return lambda x: jnp.exp(prior.log_prob(jnp.asarray(x)))


# --- PCPrecision ------------------------------------------------------------


@pytest.mark.parametrize(("U", "alpha"), [(1.0, 0.01), (0.3, 0.5), (5.0, 0.1)])
def test_pc_precision_calibration(U, alpha):
    prior = lgm.PCPrecision(U, alpha)
    p = _density(prior)

    # sigma > U  <=>  tau < U^-2; integrate in log tau for a heavy-tailed density.
    def in_log(f):
        return lambda s: f(np.exp(s)) * np.exp(s)

    t = np.log(U**-2.0)
    tail = _quad(in_log(p), -60.0, t)
    total = tail + _quad(in_log(p), t, 60.0)
    assert np.isclose(total, 1.0, atol=1e-6)
    assert np.isclose(tail, alpha, atol=1e-6)
    assert np.isclose(float(prior.cdf(U**-2.0)), alpha)


# --- PCAR1Rho ---------------------------------------------------------------


@pytest.mark.parametrize(("U", "alpha"), [(0.5, 0.5), (0.9, 0.1), (0.2, 0.7)])
def test_pc_ar1_rho_calibration(U, alpha):
    prior = lgm.PCAR1Rho(U, alpha)
    p = _density(prior)

    # rho = tanh(s): the density piles up at +-1. Integrate to rho_max and add
    # the mass beyond it, P(|rho| > r) = exp(-lambda d(r)), analytically.
    def in_s(s):
        return p(np.tanh(s)) / np.cosh(s) ** 2

    a, b = np.arctanh(U), np.arctanh(1.0 - 1e-12)
    beyond = float(jnp.exp(-prior.rate * jnp.sqrt(-jnp.log1p(-((1.0 - 1e-12) ** 2)))))
    tail = 2.0 * _quad(in_s, a, b) + beyond  # symmetric
    total = tail + _quad(in_s, -a, a)
    assert np.isclose(total, 1.0, atol=1e-6)
    assert np.isclose(tail, alpha, atol=1e-6)


def test_pc_ar1_rho_finite_value_and_gradient_at_zero():
    prior = lgm.PCAR1Rho(0.5, 0.5)
    value, grad = jax.value_and_grad(prior.log_prob)(jnp.asarray(0.0))
    assert jnp.isfinite(value) and jnp.isfinite(grad)
    # The PC prior has a cusp at its base model, log p ~ c - lambda |rho|, so
    # the one-sided gradients are -+lambda; check their antisymmetry.
    g_pos = jax.grad(prior.log_prob)(jnp.asarray(1e-6))
    g_neg = jax.grad(prior.log_prob)(jnp.asarray(-1e-6))
    assert jnp.isclose(g_pos, -g_neg) and jnp.isclose(g_pos, -prior.rate, rtol=1e-3)


# --- PCBYM2Phi --------------------------------------------------------------


def _spectrum(shape=(6, 6), **kw):
    g = kl.grid_graph(shape)
    R = kl.structure_matrix(g, scaled=True)
    return lgm.structure_spectrum(R, kl.graph_null_space(g), **kw)


@pytest.mark.parametrize(("U", "alpha"), [(0.5, 2 / 3), (0.2, 0.3), (0.8, 0.9)])
def test_pc_bym2_phi_calibration(U, alpha):
    prior = lgm.PCBYM2Phi(U, alpha, structure_spectrum=_spectrum())
    p = _density(prior)

    # phi = sigmoid(s): the density piles up at 1, where d grows only like
    # sqrt(-log(1 - phi)). Integrate to phi_max and add the mass beyond it,
    # exp(-lambda d(phi_max)), analytically.
    def in_s(s):
        phi = 1.0 / (1.0 + np.exp(-s))
        return p(phi) * phi * (1.0 - phi)

    top = 1.0 - 1e-12
    beyond = float(jnp.exp(-prior.rate * prior.distance(top)))
    below = _quad(in_s, -40.0, np.log(U / (1 - U)))
    total = below + _quad(in_s, np.log(U / (1 - U)), np.log(top / (1 - top))) + beyond
    assert np.isclose(total, 1.0, atol=1e-5)
    assert np.isclose(below, alpha, atol=1e-5)
    assert np.isclose(float(prior.cdf(U)), alpha)


def test_pc_bym2_phi_distance_matches_the_dense_kld():
    # d(phi)^2 = 2 KLD between N(0, (1-phi) I + phi R+) and N(0, I), straight
    # from the Gaussian KL formula on the dense matrices.
    g = kl.grid_graph((5, 4))
    R = kl.structure_matrix(g, scaled=True)
    V = kl.graph_null_space(g)
    prior = lgm.PCBYM2Phi(0.5, 0.5, structure_spectrum=lgm.structure_spectrum(R, V))
    Rp = np.linalg.pinv(np.asarray(R.as_matrix()), hermitian=True)
    n = Rp.shape[0]
    for phi in [1e-4, 0.1, 0.5, 0.9]:
        S = (1 - phi) * np.eye(n) + phi * Rp
        two_kld = np.trace(S) - n - np.linalg.slogdet(S)[1]
        assert np.isclose(float(prior.distance(phi)) ** 2, two_kld, rtol=1e-8)


def test_pc_bym2_phi_requires_a_null_space():
    spec = lgm.StructureSpectrum(jnp.array([0.5, 1.5, 3.0]), jnp.ones(3))
    with pytest.raises(ValueError, match="null space"):
        lgm.PCBYM2Phi(0.5, 0.5, structure_spectrum=spec)


def test_pc_bym2_phi_gradient_is_finite_near_zero():
    prior = lgm.PCBYM2Phi(0.5, 2 / 3, structure_spectrum=_spectrum())
    for phi in [1e-9, 1e-4, 0.5, 0.999]:
        value, grad = jax.value_and_grad(prior.log_prob)(jnp.asarray(phi))
        assert jnp.isfinite(value) and jnp.isfinite(grad)


def test_pc_bym2_phi_samples_hit_the_calibration():
    prior = lgm.PCBYM2Phi(0.5, 2 / 3, structure_spectrum=_spectrum())
    n = 20_000
    phi = prior.sample(jax.random.key(0), (n,))
    assert phi.shape == (n,)
    assert bool(jnp.all((phi > 0) & (phi < 1)))
    # Binomial standard error of the empirical P(phi < U); 5 sigma.
    se = np.sqrt(2 / 3 * (1 / 3) / n)
    assert abs(float((phi < 0.5).mean()) - 2 / 3) < 5 * se


@pytest.mark.slow
def test_lanczos_spectrum_agrees_with_dense_at_n_500():
    # 25 x 20 grid (n = 500), but not as a Kronecker sum, so "dense" is a
    # plain eigh and "lanczos" is SLQ on the projected operator.
    g = kl.grid_graph((25, 20))
    edges = g.topology
    graph = kl.Graph(edges, jnp.ones(edges.senders.shape[0]))
    R = kl.structure_matrix(graph, scaled=True)
    V = kl.graph_null_space(graph)
    dense = lgm.PCBYM2Phi(
        0.5, 2 / 3, structure_spectrum=lgm.structure_spectrum(R, V, method="dense")
    )
    slq = lgm.PCBYM2Phi(
        0.5,
        2 / 3,
        structure_spectrum=lgm.structure_spectrum(
            R, V, method="lanczos", num_probes=64, order=60, key=jax.random.key(0)
        ),
    )
    phis = jnp.array([0.05, 0.2, 0.5, 0.8, 0.95])
    # Deflating the 32 largest eigenvalues of R+ leaves the probes a small
    # remainder; measured relative error ~1e-4 here, bound at 1e-2.
    assert jnp.allclose(slq.distance(phis), dense.distance(phis), rtol=1e-2)
    assert jnp.allclose(slq.log_prob(phis), dense.log_prob(phis), atol=2e-2)


def test_grid_spectrum_is_the_exact_kronecker_spectrum():
    g = kl.grid_graph((5, 4))
    R = kl.structure_matrix(g, scaled=True)  # a KroneckerSum
    V = kl.graph_null_space(g)
    fast = lgm.structure_spectrum(R, V)
    lam = np.linalg.eigvalsh(np.asarray(R.as_matrix()))
    gamma = np.where(np.arange(lam.size) < 1, 0.0, 1.0 / np.where(lam > 1e-9, lam, 1))
    assert np.allclose(np.sort(np.asarray(fast.eigenvalues)), np.sort(gamma))


# --- PCMatern ---------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 2])
def test_pc_matern_calibration(d):
    prior = lgm.PCMatern(range0=3.0, alpha_range=0.05, sigma0=2.0, alpha_sigma=0.1, d=d)
    logp = jax.jit(prior.log_prob)
    lr, ls = float(prior.rate_range), float(prior.rate_sigma)

    # The density is a product, so a slice at fixed sigma divided by the
    # sigma factor lam_s exp(-lam_s s) is the range marginal (and vice versa).
    def range_marginal(r, s=0.7):
        return float(jnp.exp(logp(jnp.array([r, s])))) / (ls * np.exp(-ls * s))

    def sigma_marginal(s, r=4.0):
        h = d / 2.0
        p_r = h * lr * r ** (-h - 1.0) * np.exp(-lr * r**-h)
        return float(jnp.exp(logp(jnp.array([r, s])))) / p_r

    assert np.isclose(_quad(range_marginal, 1e-6, 3.0), 0.05, atol=1e-6)
    assert np.isclose(_quad(sigma_marginal, 2.0, np.inf), 0.1, atol=1e-6)
    assert np.isclose(_quad(sigma_marginal, 0.0, np.inf), 1.0, atol=1e-6)


def test_samples_have_the_declared_support():
    key = jax.random.key(1)
    assert bool(jnp.all(lgm.PCPrecision(1.0, 0.01).sample(key, (100,)) > 0))
    rho = lgm.PCAR1Rho(0.5, 0.5).sample(key, (100,))
    assert bool(jnp.all(jnp.abs(rho) < 1))
    x = lgm.PCMatern(1.0, 0.5, 1.0, 0.5, d=2).sample(key, (100,))
    assert x.shape == (100, 2) and bool(jnp.all(x > 0))
