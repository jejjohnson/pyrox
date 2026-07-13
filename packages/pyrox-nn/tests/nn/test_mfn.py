"""Tests for Multiplicative Filter Networks (Fathony et al., ICLR 2021).

Covers :class:`pyrox_nn.FourierFilter`, :class:`pyrox_nn.GaborFilter`,
:class:`pyrox_nn.FourierNet`, :class:`pyrox_nn.GaborNet`,
:class:`pyrox_nn.BayesianFourierNet`, :class:`pyrox_nn.BayesianGaborNet`,
and :func:`pyrox_nn.mfn_forward`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from numpyro import handlers
from pyrox_nn import (
    BayesianFourierNet,
    BayesianGaborNet,
    FourierFilter,
    FourierNet,
    GaborFilter,
    GaborNet,
    mfn_forward,
)


# ---------------------------------------------------------------------------
# FourierFilter
# ---------------------------------------------------------------------------


def test_fourier_filter_output_shape():
    key = jr.PRNGKey(0)
    f = FourierFilter.init(in_features=3, out_features=16, key=key)
    x = jnp.ones((5, 3))
    y = jax.vmap(f)(x)
    assert y.shape == (5, 16)


def test_fourier_filter_output_shape_single_point():
    """Single-point (D,) input returns (H,) under the single-example convention."""
    key = jr.PRNGKey(1)
    f = FourierFilter.init(in_features=3, out_features=8, key=key)
    x = jnp.ones((3,))
    y = f(x)
    assert y.shape == (8,)


# ---------------------------------------------------------------------------
# GaborFilter
# ---------------------------------------------------------------------------


def test_gabor_filter_output_shape():
    key = jr.PRNGKey(2)
    f = GaborFilter.init(in_features=3, out_features=16, key=key)
    x = jnp.ones((5, 3))
    y = jax.vmap(f)(x)
    assert y.shape == (5, 16)


def test_gabor_filter_envelope_decays():
    """At x = mu the envelope is 1; far from mu it is approximately 0."""
    key = jr.PRNGKey(3)
    # Manually set a known gamma and mu for a predictable test.
    f = GaborFilter.init(in_features=2, out_features=4, key=key)
    # Replace mu with a known value so we can test at x = mu exactly.
    import equinox as eqx

    mu_known = jnp.ones((4, 2)) * 0.5
    log_gamma_known = jnp.zeros((4,))  # gamma = 1 for all filters
    f_known = eqx.tree_at(
        lambda ff: (ff.mu, ff.log_gamma), f, (mu_known, log_gamma_known)
    )

    # At x = mu, envelope = exp(-gamma/2 * 0) = 1 -> g(x) = sin(Omega*mu + phi)
    x_at_mu = mu_known[:1, :]  # shape (1, 2), equals mu[0]
    g_at_mu = jax.vmap(f_known)(x_at_mu)  # (1, 4)
    g_sin_only = jnp.sin(x_at_mu @ f_known.Omega.T + f_known.phi)
    np.testing.assert_allclose(np.asarray(g_at_mu), np.asarray(g_sin_only), atol=1e-5)

    # Far from mu, envelope should be very small.
    x_far = jnp.ones((1, 2)) * 10.0
    g_far = jax.vmap(f_known)(x_far)
    # ||x - mu||^2 = 2 * 9.5^2 = 180.5 (sum across the D=2 dims);
    # envelope = exp(-0.5 * gamma * 180.5) ≈ 1e-39 with gamma=1.
    assert float(jnp.max(jnp.abs(g_far))) < 1e-10


# ---------------------------------------------------------------------------
# FourierNet
# ---------------------------------------------------------------------------


def test_fourier_net_output_shape():
    key = jr.PRNGKey(4)
    net = FourierNet.init(
        in_features=2, hidden_features=32, out_features=4, depth=3, key=key
    )
    x = jnp.ones((10, 2))
    y = jax.vmap(net)(x)
    assert y.shape == (10, 4)


def test_fourier_net_single_point_squeezed():
    """Input (D,) → output (O,)."""
    key = jr.PRNGKey(5)
    net = FourierNet.init(
        in_features=2, hidden_features=16, out_features=3, depth=2, key=key
    )
    x = jnp.ones((2,))
    y = net(x)
    assert y.shape == (3,)


# ---------------------------------------------------------------------------
# GaborNet
# ---------------------------------------------------------------------------


def test_gabor_net_output_shape():
    key = jr.PRNGKey(6)
    net = GaborNet.init(
        in_features=2, hidden_features=32, out_features=4, depth=3, key=key
    )
    x = jnp.ones((10, 2))
    y = jax.vmap(net)(x)
    assert y.shape == (10, 4)


def test_gabor_net_single_point_squeezed():
    key = jr.PRNGKey(7)
    net = GaborNet.init(
        in_features=2, hidden_features=16, out_features=3, depth=2, key=key
    )
    x = jnp.ones((2,))
    y = net(x)
    assert y.shape == (3,)


# ---------------------------------------------------------------------------
# Depth-1 sanity: FourierNet(depth=1) == readout(FourierFilter(x))
# ---------------------------------------------------------------------------


def test_mfn_depth_1_matches_single_filter():
    """FourierNet(depth=1): output equals readout applied to first filter."""
    key = jr.PRNGKey(8)
    net = FourierNet.init(
        in_features=2, hidden_features=8, out_features=3, depth=1, key=key
    )
    x = jnp.ones((4, 2))
    y_net = jax.vmap(net)(x)
    # Manually compute: z = filter(x); y = vmap(linear)(z)
    z = jax.vmap(net.filters[0])(x)  # (4, 8)
    y_manual = jax.vmap(net.linears[0])(z)  # (4, 3)
    np.testing.assert_allclose(np.asarray(y_net), np.asarray(y_manual), atol=1e-5)


# ---------------------------------------------------------------------------
# Kernel approximation: GaborNet(depth=1, μ=0) → theoretical kernel
# ---------------------------------------------------------------------------


def test_gabor_depth_1_matches_rbf_rff_kernel():
    r"""Depth-1 GaborFilter with mu=0 approximates its theoretical kernel.

    For mu=0, gamma ~ Gamma(alpha, 1), Omega ~ N(0, gamma * I), phi ~ U(-pi, pi):

        E[g(x) g(x')] = (1/2) * (1 + t/2)^{-alpha}

    where t = ||x||^2 + ||x'||^2 + ||x - x'||^2 and alpha = gamma_alpha.

    With H=512 filters, the empirical estimate should be within ~0.04
    of the analytical value.
    """
    import equinox as eqx

    D = 2
    H = 512
    alpha = 6.0
    key = jr.PRNGKey(42)

    net = GaborNet.init(
        in_features=D,
        hidden_features=H,
        out_features=1,
        depth=1,
        key=key,
        gamma_alpha=alpha,
        gamma_beta=1.0,
    )
    # Set μ = 0 (zero-mean envelope) so the theoretical formula holds.
    filt = net.filters[0]
    filt_zero_mu = eqx.tree_at(lambda f: f.mu, filt, jnp.zeros_like(filt.mu))

    # Small inputs so the envelope doesn't vanish.
    x = jnp.array([[0.1, 0.2]])  # (1, D)
    xp = jnp.array([[0.3, 0.1]])  # (1, D)

    phi_x = jax.vmap(filt_zero_mu)(x)[0]  # (H,)
    phi_xp = jax.vmap(filt_zero_mu)(xp)[0]  # (H,)

    K_hat = float(jnp.dot(phi_x, phi_xp)) / H

    # Analytical: (1/2) * (1 + t/2)^{-alpha}
    t = float(jnp.sum(x**2) + jnp.sum(xp**2) + jnp.sum((x - xp) ** 2))
    K_analytical = 0.5 * (1.0 + t / 2.0) ** (-alpha)

    # Monte-Carlo tolerance: std ≈ sqrt(Var/H) ≈ 0.04 for H=512.
    assert abs(K_hat - K_analytical) < 0.06, (
        f"K_hat={K_hat:.4f}, K_analytical={K_analytical:.4f}"
    )


# ---------------------------------------------------------------------------
# JIT compilation
# ---------------------------------------------------------------------------


def test_fourier_net_jits():
    key = jr.PRNGKey(10)
    net = FourierNet.init(
        in_features=2, hidden_features=16, out_features=1, depth=2, key=key
    )

    @jax.jit
    def fwd(x):
        return jax.vmap(net)(x)

    y = fwd(jnp.ones((5, 2)))
    assert jnp.all(jnp.isfinite(y))


def test_gabor_net_jits():
    key = jr.PRNGKey(11)
    net = GaborNet.init(
        in_features=2, hidden_features=16, out_features=1, depth=2, key=key
    )

    @jax.jit
    def fwd(x):
        return jax.vmap(net)(x)

    y = fwd(jnp.ones((5, 2)))
    assert jnp.all(jnp.isfinite(y))


# ---------------------------------------------------------------------------
# Gradient flow through all parameters
# ---------------------------------------------------------------------------


def test_mfn_gradient_flows_through_all_params():
    """grad of scalar loss w.r.t. full PyTree has finite non-zero leaf grads."""
    key = jr.PRNGKey(12)
    net = FourierNet.init(
        in_features=2, hidden_features=8, out_features=1, depth=2, key=key
    )
    x = jnp.ones((3, 2))

    def loss_fn(m):
        return jnp.sum(jax.vmap(m)(x) ** 2)

    grads = jax.grad(loss_fn)(net)
    leaves = jax.tree_util.tree_leaves(grads)
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf)), "non-finite gradient"
        assert jnp.any(leaf != 0.0), "zero gradient leaf"


def test_gabor_mfn_gradient_flows_through_all_params():
    key = jr.PRNGKey(13)
    net = GaborNet.init(
        in_features=2, hidden_features=8, out_features=1, depth=2, key=key
    )
    x = jnp.ones((3, 2))

    def loss_fn(m):
        return jnp.sum(jax.vmap(m)(x) ** 2)

    grads = jax.grad(loss_fn)(net)
    leaves = jax.tree_util.tree_leaves(grads)
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf)), "non-finite gradient"
        assert jnp.any(leaf != 0.0), "zero gradient leaf"


# ---------------------------------------------------------------------------
# Bayesian variants — site registration
# ---------------------------------------------------------------------------


def test_bayesian_fourier_net_registers_sites():
    """handlers.trace() captures exactly 4*depth sample sites."""
    depth = 3
    key = jr.PRNGKey(14)
    net = BayesianFourierNet.init(
        in_features=2,
        hidden_features=8,
        out_features=1,
        depth=depth,
        key=key,
        pyrox_name="bfn",
    )
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        net(jnp.ones((4, 2)))
    sample_sites = {n for n, s in tr.items() if s["type"] == "sample"}
    # 2 per filter (Omega, phi) + 2 per linear (W, b) = 4 * depth
    assert len(sample_sites) == 4 * depth, (
        f"expected {4 * depth} sites, got {len(sample_sites)}: {sample_sites}"
    )


def test_bayesian_gabor_net_registers_sites():
    """handlers.trace() captures exactly 6*depth sample sites."""
    depth = 2
    key = jr.PRNGKey(15)
    net = BayesianGaborNet.init(
        in_features=2,
        hidden_features=8,
        out_features=1,
        depth=depth,
        key=key,
        pyrox_name="bgn",
    )
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        net(jnp.ones((4, 2)))
    sample_sites = {n for n, s in tr.items() if s["type"] == "sample"}
    # 4 per filter (Omega, phi, mu, log_gamma) + 2 per linear (W, b) = 6 * depth
    assert len(sample_sites) == 6 * depth, (
        f"expected {6 * depth} sites, got {len(sample_sites)}: {sample_sites}"
    )


# ---------------------------------------------------------------------------
# mfn_forward helper
# ---------------------------------------------------------------------------


def test_mfn_forward_helper_matches_fourier_net():
    """mfn_forward with FourierNet's filters and linears matches net(x)."""
    key = jr.PRNGKey(16)
    net = FourierNet.init(
        in_features=2, hidden_features=8, out_features=3, depth=3, key=key
    )
    x = jnp.ones((5, 2))
    y_net = jax.vmap(net)(x)
    y_helper = jax.vmap(lambda xi: mfn_forward(xi, net.filters, net.linears))(x)
    np.testing.assert_allclose(np.asarray(y_net), np.asarray(y_helper), atol=1e-5)


# ---------------------------------------------------------------------------
# Depth validation
# ---------------------------------------------------------------------------


def test_mfn_rejects_mismatched_depth():
    """FourierNet.init with depth=0 raises ValueError mentioning 'depth'."""
    key = jr.PRNGKey(17)
    with pytest.raises(ValueError, match="depth"):
        FourierNet.init(
            in_features=2, hidden_features=8, out_features=1, depth=0, key=key
        )


def test_gabor_net_rejects_zero_depth():
    key = jr.PRNGKey(18)
    with pytest.raises(ValueError, match="depth"):
        GaborNet.init(
            in_features=2, hidden_features=8, out_features=1, depth=0, key=key
        )


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------


def test_fourier_filter_rejects_zero_in_features():
    """FourierFilter.init raises ValueError instead of dividing by sqrt(0)."""
    key = jr.PRNGKey(19)
    with pytest.raises(ValueError, match="in_features"):
        FourierFilter.init(in_features=0, out_features=4, key=key)


def test_fourier_filter_rejects_nonpositive_freq_scale():
    key = jr.PRNGKey(20)
    with pytest.raises(ValueError, match="freq_scale"):
        FourierFilter.init(in_features=2, out_features=4, key=key, freq_scale=0.0)


def test_gabor_filter_rejects_bad_domain():
    """domain[0] >= domain[1] is rejected at init time."""
    key = jr.PRNGKey(21)
    with pytest.raises(ValueError, match="domain"):
        GaborFilter.init(in_features=2, out_features=4, key=key, domain=(1.0, 1.0))


def test_gabor_filter_rejects_nonpositive_gamma():
    key = jr.PRNGKey(22)
    with pytest.raises(ValueError, match="gamma_beta"):
        GaborFilter.init(in_features=2, out_features=4, key=key, gamma_beta=0.0)


def test_mfn_forward_rejects_empty_sequences():
    """mfn_forward with empty filters/linears raises ValueError, not IndexError."""
    x = jnp.ones((3, 2))
    with pytest.raises(ValueError, match="non-empty"):
        mfn_forward(x, [], [])


def test_mfn_forward_rejects_mismatched_lengths():
    """Length mismatch between filters and linears is rejected up-front."""
    key = jr.PRNGKey(23)
    net = FourierNet.init(
        in_features=2, hidden_features=4, out_features=1, depth=2, key=key
    )
    x = jnp.ones((3, 2))
    with pytest.raises(ValueError, match="equal length"):
        mfn_forward(x, net.filters, net.linears[:1])


# ---------------------------------------------------------------------------
# Bayesian variants — init accepts prior_std
# ---------------------------------------------------------------------------


def test_bayesian_fourier_net_init_accepts_prior_std():
    """BayesianFourierNet.init(prior_std=...) plumbs through to __call__."""
    net = BayesianFourierNet.init(
        in_features=2,
        hidden_features=4,
        out_features=1,
        depth=2,
        key=jr.PRNGKey(24),
        prior_std=2.5,
    )
    assert net.prior_std == 2.5


def test_bayesian_gabor_net_init_accepts_prior_std():
    """BayesianGaborNet.init(prior_std=...) plumbs through to __call__."""
    net = BayesianGaborNet.init(
        in_features=2,
        hidden_features=4,
        out_features=1,
        depth=2,
        key=jr.PRNGKey(25),
        prior_std=0.5,
    )
    assert net.prior_std == 0.5
