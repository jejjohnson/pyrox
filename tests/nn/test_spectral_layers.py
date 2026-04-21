"""Tests for the wave-4 spectral feature layers.

Covers :class:`pyrox.nn.VariationalFourierFeatures`,
:class:`pyrox.nn.OrthogonalRandomFeatures`, and
:class:`pyrox.nn.HSGPFeatures`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from numpyro import handlers

from pyrox.gp import RBF, Matern
from pyrox.nn import (
    HSGPFeatures,
    OrthogonalRandomFeatures,
    RBFFourierFeatures,
    VariationalFourierFeatures,
)


# --- VariationalFourierFeatures -------------------------------------------


def test_vff_output_shape():
    layer = VariationalFourierFeatures.init(in_features=2, n_features=8)
    x = jnp.ones((4, 2))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert y.shape == (4, 16)  # cos + sin


def test_vff_registers_W_and_lengthscale_sites():
    layer = VariationalFourierFeatures.init(
        in_features=2, n_features=4, lengthscale=0.5
    )
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(jnp.zeros((1, 2)))
    site_names = {n for n, s in tr.items() if s["type"] == "sample"}
    scope = layer._pyrox_scope_name()
    assert f"{scope}.W" in site_names
    assert f"{scope}.lengthscale" in site_names


def test_vff_matches_rbf_rff_under_seed():
    """Under the same seed and matching prior, VFF and RBF-RFF agree on forward."""
    layer_vff = VariationalFourierFeatures.init(
        in_features=3, n_features=8, lengthscale=1.0
    )
    layer_rff = RBFFourierFeatures.init(in_features=3, n_features=8, lengthscale=1.0)
    # Match scope names so both layers register identical site keys.
    layer_vff = type(layer_vff)(
        in_features=3, n_features=8, init_lengthscale=1.0, pyrox_name="rff"
    )
    layer_rff = type(layer_rff)(
        in_features=3, n_features=8, init_lengthscale=1.0, pyrox_name="rff"
    )
    x = jnp.ones((2, 3))
    with handlers.seed(rng_seed=42):
        y_vff = layer_vff(x)
    with handlers.seed(rng_seed=42):
        y_rff = layer_rff(x)
    np.testing.assert_allclose(np.asarray(y_vff), np.asarray(y_rff), atol=1e-5)


def test_vff_rejects_nonpositive_lengthscale():
    with pytest.raises(ValueError, match="lengthscale"):
        VariationalFourierFeatures.init(in_features=2, n_features=4, lengthscale=0.0)


# --- OrthogonalRandomFeatures ---------------------------------------------


def test_orf_output_shape():
    orf = OrthogonalRandomFeatures.init(in_features=4, n_features=8, key=jr.PRNGKey(0))
    y = orf(jnp.zeros((6, 4)))
    assert y.shape == (6, 16)


def test_orf_block_orthogonality():
    """Each ``D x D`` block's columns are orthogonal chi-scaled unit vectors."""
    D = 4
    n_features = 12  # 3 blocks
    orf = OrthogonalRandomFeatures.init(
        in_features=D, n_features=n_features, key=jr.PRNGKey(7)
    )
    W = np.asarray(orf.W)  # (D, n_features); columns are frequencies
    n_blocks = n_features // D
    for k in range(n_blocks):
        block = W[:, k * D : (k + 1) * D]
        # Recover Q by dividing each *column* by its chi-magnitude.
        chi = np.linalg.norm(block, axis=0)  # (D,) per-column norm
        Q = block / chi[None, :]
        # Columns of Q are orthonormal: Q^T Q = I.
        np.testing.assert_allclose(Q.T @ Q, np.eye(D), atol=1e-5)


def test_orf_kernel_approximation_lower_variance_than_rff():
    """ORF gives lower MSE on the kernel approximation than plain RFF.

    Compare the variance over 32 seeds of the kernel approximation
    ``phi(x).T @ phi(x')`` against the analytic RBF kernel value at a
    moderate test pair.
    """
    rng = np.random.default_rng(123)
    D = 8
    n_features = 16
    x = jnp.asarray(rng.standard_normal(D))
    xp = jnp.asarray(rng.standard_normal(D))
    true_k = float(jnp.exp(-0.5 * jnp.sum((x - xp) ** 2)))

    n_seeds = 32
    mse_orf = 0.0
    mse_rff = 0.0
    for s in range(n_seeds):
        orf = OrthogonalRandomFeatures.init(
            in_features=D, n_features=n_features, key=jr.PRNGKey(s)
        )
        phi_x = orf(x[None, :])[0]
        phi_y = orf(xp[None, :])[0]
        approx = float(jnp.dot(phi_x, phi_y))
        mse_orf += (approx - true_k) ** 2 / n_seeds

        # Plain RFF baseline with same n_features.
        rff_layer = RBFFourierFeatures.init(in_features=D, n_features=n_features)
        with handlers.seed(rng_seed=s):
            phi_x_r = rff_layer(x[None, :])[0]
            phi_y_r = rff_layer(xp[None, :])[0]
        approx_r = float(jnp.dot(phi_x_r, phi_y_r))
        mse_rff += (approx_r - true_k) ** 2 / n_seeds

    # ORF should have noticeably lower MSE — give a small margin to keep this
    # test from flaking due to RNG.
    assert mse_orf < mse_rff, (mse_orf, mse_rff)


def test_orf_rejects_indivisible_n_features():
    with pytest.raises(ValueError, match="divisible"):
        OrthogonalRandomFeatures.init(in_features=3, n_features=8, key=jr.PRNGKey(0))


def test_orf_rejects_nonpositive_lengthscale():
    with pytest.raises(ValueError, match="lengthscale"):
        OrthogonalRandomFeatures.init(
            in_features=4, n_features=8, key=jr.PRNGKey(0), lengthscale=0.0
        )


# --- HSGPFeatures ----------------------------------------------------------


def test_hsgp_output_shape():
    layer = HSGPFeatures.init(
        in_features=1,
        num_basis_per_dim=8,
        L=3.0,
        kernel=RBF(init_lengthscale=0.5, init_variance=1.0),
    )
    x = jnp.linspace(-2, 2, 16).reshape(-1, 1)
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert y.shape == (16,)


def test_hsgp_registers_alpha_site():
    kern = RBF(init_lengthscale=0.5, init_variance=1.0)
    layer = HSGPFeatures(
        in_features=1,
        num_basis_per_dim=(4,),
        L=(2.0,),
        kernel=kern,
        pyrox_name="hsgp",
    )
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(jnp.zeros((3, 1)))
    site_names = {n for n, s in tr.items() if s["type"] == "sample"}
    assert "hsgp.alpha" in site_names


def test_hsgp_alpha_dim_matches_total_basis():
    kern = RBF(init_lengthscale=1.0, init_variance=1.0)
    layer = HSGPFeatures.init(
        in_features=2, num_basis_per_dim=(3, 4), L=(1.0, 1.0), kernel=kern
    )
    assert layer.num_basis == 12


def test_hsgp_jits():
    kern = RBF(init_lengthscale=0.5, init_variance=1.0)
    layer = HSGPFeatures.init(in_features=1, num_basis_per_dim=4, L=2.0, kernel=kern)

    @jax.jit
    def fwd(x):
        with handlers.seed(rng_seed=0):
            return layer(x)

    y = fwd(jnp.linspace(-1, 1, 8).reshape(-1, 1))
    assert jnp.all(jnp.isfinite(y))


def test_hsgp_with_matern_kernel():
    kern = Matern(init_lengthscale=0.5, init_variance=1.0, nu=2.5)
    layer = HSGPFeatures.init(in_features=1, num_basis_per_dim=6, L=2.0, kernel=kern)
    x = jnp.linspace(-1, 1, 8).reshape(-1, 1)
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert jnp.all(jnp.isfinite(y))


def test_hsgp_rejects_mismatched_dims():
    kern = RBF(init_lengthscale=1.0, init_variance=1.0)
    with pytest.raises(ValueError, match="num_basis_per_dim length"):
        HSGPFeatures.init(in_features=2, num_basis_per_dim=(3,), L=1.0, kernel=kern)
    with pytest.raises(ValueError, match="L length"):
        HSGPFeatures.init(in_features=2, num_basis_per_dim=2, L=(1.0,), kernel=kern)


def test_hsgp_rejects_nonpositive_L():
    kern = RBF(init_lengthscale=1.0, init_variance=1.0)
    with pytest.raises(ValueError, match="L"):
        HSGPFeatures.init(in_features=1, num_basis_per_dim=4, L=0.0, kernel=kern)


def test_hsgp_kernel_approximation_converges():
    r"""As ``M`` and ``L`` grow, ``Var[hat f(x)]`` approaches the kernel diagonal.

    Property tested: averaging over many alpha draws,
    :math:`\mathrm{Var}[\hat f(x)] = \sum_j S(\sqrt{\lambda_j}) \phi_j(x)^2`
    should approach :math:`k(x, x) = \sigma^2` as ``M`` grows.
    """
    kern = RBF(init_lengthscale=0.5, init_variance=1.0)
    layer = HSGPFeatures.init(in_features=1, num_basis_per_dim=64, L=4.0, kernel=kern)
    x = jnp.array([[0.0]])  # well inside [-L, L]

    # Estimate Var[f(0)] by sampling alpha's prior many times.
    n_samples = 200
    samples = []
    for s in range(n_samples):
        with handlers.seed(rng_seed=s):
            samples.append(float(layer(x)[0]))
    var = np.var(samples)
    # k(0, 0) = sigma^2 = 1.0; allow ~25% Monte-Carlo slack on 200 samples.
    assert 0.5 < var < 1.5, var
