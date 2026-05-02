"""Tests for ``pyrox.nn`` uncertainty-aware dense and random-feature layers.

Covers forward shapes, NumPyro site registration, stochastic semantics,
the NCP deterministic + stochastic decomposition, and random-feature
kernel approximation sanity checks.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from numpyro import handlers

from pyrox.nn import (
    ArcCosineFourierFeatures,
    DenseFlipout,
    DenseNCP,
    DenseReparameterization,
    DenseVariational,
    DenseVariationalDropout,
    LaplaceCosineFeatures,
    LaplaceFourierFeatures,
    MaternCosineFeatures,
    MaternFourierFeatures,
    MCDropout,
    RandomKitchenSinks,
    RBFCosineFeatures,
    RBFFourierFeatures,
)


# --- DenseReparameterization -----------------------------------------------


def test_reparam_output_shape():
    layer = DenseReparameterization(in_features=3, out_features=5)
    x = jnp.ones((4, 3))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert y.shape == (4, 5)


def test_reparam_no_bias():
    layer = DenseReparameterization(in_features=3, out_features=5, bias=False)
    x = jnp.ones((2, 3))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert y.shape == (2, 5)


def test_reparam_registers_weight_and_bias_sites():
    layer = DenseReparameterization(in_features=3, out_features=2, pyrox_name="reparam")
    x = jnp.ones((1, 3))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(x)
    assert "reparam.weight" in tr
    assert "reparam.bias" in tr
    assert tr["reparam.weight"]["type"] == "sample"
    assert tr["reparam.bias"]["type"] == "sample"


def test_reparam_stochastic_across_seeds():
    layer = DenseReparameterization(in_features=3, out_features=2)
    x = jnp.ones((1, 3))
    with handlers.seed(rng_seed=0):
        y1 = layer(x)
    with handlers.seed(rng_seed=1):
        y2 = layer(x)
    assert not jnp.allclose(y1, y2)


# --- DenseFlipout ----------------------------------------------------------


def test_flipout_output_shape():
    layer = DenseFlipout(in_features=3, out_features=5)
    x = jnp.ones((4, 3))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert y.shape == (4, 5)


def test_flipout_registers_weight_site():
    layer = DenseFlipout(in_features=3, out_features=2, pyrox_name="flipout")
    x = jnp.ones((1, 3))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(x)
    assert "flipout.weight" in tr


def test_flipout_stochastic_across_seeds():
    layer = DenseFlipout(in_features=3, out_features=2)
    x = jnp.ones((2, 3))
    with handlers.seed(rng_seed=0):
        y1 = layer(x)
    with handlers.seed(rng_seed=1):
        y2 = layer(x)
    assert not jnp.allclose(y1, y2)


# --- DenseVariational ------------------------------------------------------


def _std_prior(d_in, d_out):
    return dist.Normal(jnp.zeros((d_in, d_out)), 1.0).to_event(2)


def test_variational_output_shape():
    layer = DenseVariational(
        in_features=3,
        out_features=5,
        make_prior=_std_prior,
    )
    x = jnp.ones((4, 3))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert y.shape == (4, 5)


def test_variational_registers_weight_site():
    layer = DenseVariational(
        in_features=3,
        out_features=2,
        make_prior=_std_prior,
        pyrox_name="var",
    )
    x = jnp.ones((1, 3))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(x)
    assert "var.weight" in tr


# --- MCDropout -------------------------------------------------------------


def test_mcdropout_output_shape():
    drop = MCDropout(rate=0.5)
    x = jnp.ones((4, 10))
    y = drop(x, key=jr.PRNGKey(0))
    assert y.shape == (4, 10)


def test_mcdropout_zeros_some_elements():
    drop = MCDropout(rate=0.5)
    x = jnp.ones((100, 10))
    y = drop(x, key=jr.PRNGKey(0))
    assert jnp.any(y == 0.0)
    assert jnp.any(y != 0.0)


def test_mcdropout_scales_survivors():
    drop = MCDropout(rate=0.5)
    x = jnp.ones((1000, 10))
    y = drop(x, key=jr.PRNGKey(0))
    survivors = y[y != 0.0]
    assert jnp.allclose(survivors, 1.0 / 0.5, atol=1e-5)


def test_mcdropout_stochastic_across_keys():
    drop = MCDropout(rate=0.5)
    x = jnp.ones((10, 10))
    y1 = drop(x, key=jr.PRNGKey(0))
    y2 = drop(x, key=jr.PRNGKey(1))
    assert not jnp.allclose(y1, y2)


def test_mcdropout_is_not_pyrox_module():
    from pyrox._core.pyrox_module import PyroxModule

    drop = MCDropout()
    assert not isinstance(drop, PyroxModule)


# --- DenseNCP --------------------------------------------------------------


def test_ncp_output_shape():
    layer = DenseNCP(in_features=3, out_features=5)
    x = jnp.ones((4, 3))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert y.shape == (4, 5)


def test_ncp_registers_det_and_stoch_sites():
    layer = DenseNCP(in_features=3, out_features=2, pyrox_name="ncp")
    x = jnp.ones((1, 3))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(x)
    assert "ncp.weight_stoch" in tr
    assert "ncp.bias_stoch" in tr
    assert tr["ncp.weight_stoch"]["type"] == "sample"


def test_ncp_registers_scale_site():
    """The scale parameter is a sample site with a LogNormal prior."""
    layer = DenseNCP(in_features=3, out_features=2, pyrox_name="ncp2")
    x = jnp.ones((1, 3))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(x)
    assert "ncp2.scale" in tr
    assert tr["ncp2.scale"]["type"] == "sample"


def test_ncp_stochastic_when_scale_nonzero():
    layer = DenseNCP(in_features=3, out_features=2, init_scale=1.0)
    x = jnp.ones((2, 3))
    with handlers.seed(rng_seed=0):
        y1 = layer(x)
    with handlers.seed(rng_seed=1):
        y2 = layer(x)
    assert not jnp.allclose(y1, y2)


# --- DenseVariationalDropout -----------------------------------------------


def test_vd_output_shape():
    layer = DenseVariationalDropout(in_features=3, out_features=5)
    x = jnp.ones((4, 3))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert y.shape == (4, 5)


def test_vd_no_bias():
    layer = DenseVariationalDropout(in_features=3, out_features=5, bias=False)
    x = jnp.ones((2, 3))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert y.shape == (2, 5)


def test_vd_registers_param_and_kl_sites():
    layer = DenseVariationalDropout(in_features=3, out_features=2, pyrox_name="vd")
    x = jnp.ones((1, 3))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(x)
    assert tr["vd.theta"]["type"] == "param"
    assert tr["vd.log_alpha"]["type"] == "param"
    assert tr["vd.bias"]["type"] == "param"
    # numpyro.factor registers as a sample-type site backed by a Unit
    # distribution whose log_factor carries the value.
    assert "vd.kl" in tr


def test_vd_kl_factor_is_non_positive():
    layer = DenseVariationalDropout(
        in_features=3, out_features=2, pyrox_name="vd", log_alpha_init=-5.0
    )
    x = jnp.ones((1, 3))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        layer(x)
    log_factor = float(tr["vd.kl"]["fn"].log_factor)
    # Factor adds -KL to the log density; KL >= 0 so log_factor <= 0.
    assert log_factor <= 0.0


def test_vd_stochastic_across_seeds():
    """With non-zero theta and non-trivial alpha, output is stochastic."""
    layer = DenseVariationalDropout(
        in_features=3, out_features=2, pyrox_name="vd", log_alpha_init=0.0
    )
    x = jnp.ones((4, 3))
    theta = jnp.ones((3, 2))
    with (
        handlers.substitute(data={"vd.theta": theta}),
        handlers.seed(rng_seed=0),
    ):
        y1 = layer(x)
    with (
        handlers.substitute(data={"vd.theta": theta}),
        handlers.seed(rng_seed=1),
    ):
        y2 = layer(x)
    assert not jnp.allclose(y1, y2)


def test_vd_deterministic_limit_at_low_log_alpha():
    """log_alpha << 0 → alpha → 0 → noise term vanishes; output ≈ x @ theta."""
    layer = DenseVariationalDropout(
        in_features=3, out_features=2, pyrox_name="vd", log_alpha_init=-10.0
    )
    x = jnp.ones((4, 3))
    theta = jnp.ones((3, 2))
    expected = x @ theta
    with (
        handlers.substitute(data={"vd.theta": theta}),
        handlers.seed(rng_seed=0),
    ):
        y = layer(x)
    # alpha = exp(-10) ≈ 4.54e-5; per-output noise std at this clamp is
    # ~sqrt(in_features * alpha) ≈ 0.012. A few-sigma tolerance is safe.
    assert jnp.allclose(y, expected, atol=0.1)


def test_vd_zero_theta_init_gives_zero_output():
    """With the default theta=0 init, gamma=0 and delta=0 ⇒ y = bias only."""
    layer = DenseVariationalDropout(
        in_features=3, out_features=2, pyrox_name="vd", bias=False
    )
    x = jnp.ones((4, 3))
    with handlers.seed(rng_seed=0):
        y = layer(x)
    assert jnp.allclose(y, jnp.zeros_like(y))


def test_vd_sparsity_threshold_selects_pruned_weights():
    layer = DenseVariationalDropout(in_features=3, out_features=2, threshold=3.0)
    log_alpha = jnp.array([[5.0, 1.0], [4.0, 2.0], [0.0, 3.5]])
    # Strictly above 3.0: 5.0, 4.0, 3.5 → three of six.
    s = float(layer.sparsity(log_alpha))
    assert s == pytest.approx(3.0 / 6.0)


def test_vd_is_pyrox_module():
    from pyrox._core.pyrox_module import PyroxModule

    layer = DenseVariationalDropout(in_features=2, out_features=2)
    assert isinstance(layer, PyroxModule)


# --- RBFFourierFeatures (SSGP-style) ---------------------------------------


def test_rbf_rff_output_shape():
    rff = RBFFourierFeatures.init(in_features=3, n_features=10)
    with handlers.seed(rng_seed=0):
        assert rff(jnp.ones((4, 3))).shape == (4, 20)


def test_rbf_rff_registers_w_and_lengthscale_sites():
    rff = RBFFourierFeatures(in_features=3, n_features=5, pyrox_name="rbf_rff")
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        rff(jnp.ones((2, 3)))
    assert "rbf_rff.W" in tr
    assert "rbf_rff.lengthscale" in tr
    assert tr["rbf_rff.W"]["type"] == "sample"
    assert tr["rbf_rff.lengthscale"]["type"] == "sample"


def test_rbf_rff_stochastic_across_seeds():
    rff = RBFFourierFeatures.init(in_features=3, n_features=10)
    x = jnp.ones((2, 3))
    with handlers.seed(rng_seed=0):
        y1 = rff(x)
    with handlers.seed(rng_seed=1):
        y2 = rff(x)
    assert not jnp.allclose(y1, y2)


def test_rbf_rff_kernel_approximation():
    """RFF inner product approximates the RBF kernel when W and
    lengthscale are conditioned to known values."""
    rff = RBFFourierFeatures(in_features=1, n_features=5000, pyrox_name="rff_approx")
    W_fixed = jr.normal(jr.PRNGKey(42), (1, 5000))
    with (
        handlers.seed(rng_seed=0),
        handlers.condition(
            data={
                "rff_approx.W": W_fixed,
                "rff_approx.lengthscale": jnp.array(1.0),
            }
        ),
    ):
        phi1 = rff(jnp.array([[0.0]]))
        phi2 = rff(jnp.array([[0.5]]))
    k_approx = float((phi1 @ phi2.T).squeeze())
    k_true = float(jnp.exp(-0.5 * 0.5**2))
    assert abs(k_approx - k_true) < 0.05


def test_rbf_rff_is_pyrox_module():
    from pyrox._core.pyrox_module import PyroxModule

    assert isinstance(RBFFourierFeatures.init(in_features=3, n_features=5), PyroxModule)


# --- MaternFourierFeatures (SSGP-style) ------------------------------------


def test_matern32_rff_output_shape():
    rff = MaternFourierFeatures.init(in_features=3, n_features=10, nu=1.5)
    with handlers.seed(rng_seed=0):
        assert rff(jnp.ones((4, 3))).shape == (4, 20)


def test_matern52_rff_output_shape():
    rff = MaternFourierFeatures.init(in_features=3, n_features=10, nu=2.5)
    with handlers.seed(rng_seed=0):
        assert rff(jnp.ones((4, 3))).shape == (4, 20)


def test_matern_rff_registers_w_site_with_student_t():
    rff = MaternFourierFeatures(
        in_features=2, n_features=5, nu=1.5, pyrox_name="mat_rff"
    )
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        rff(jnp.ones((1, 2)))
    assert "mat_rff.W" in tr
    assert tr["mat_rff.W"]["type"] == "sample"


# --- LaplaceFourierFeatures (SSGP-style) -----------------------------------


def test_laplace_rff_output_shape():
    rff = LaplaceFourierFeatures.init(in_features=3, n_features=10)
    with handlers.seed(rng_seed=0):
        assert rff(jnp.ones((4, 3))).shape == (4, 20)


def test_laplace_rff_registers_w_site():
    rff = LaplaceFourierFeatures(in_features=2, n_features=5, pyrox_name="lap_rff")
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        rff(jnp.ones((1, 2)))
    assert "lap_rff.W" in tr


# --- NCPContinuousPerturb --------------------------------------------------


def test_ncp_perturb_output_shape():
    from pyrox.nn import NCPContinuousPerturb

    perturb = NCPContinuousPerturb(scale=0.1)
    x = jnp.ones((4, 3))
    assert perturb(x, key=jr.PRNGKey(0)).shape == (4, 3)


def test_ncp_perturb_stochastic_across_keys():
    from pyrox.nn import NCPContinuousPerturb

    perturb = NCPContinuousPerturb(scale=1.0)
    x = jnp.ones((2, 3))
    y1 = perturb(x, key=jr.PRNGKey(0))
    y2 = perturb(x, key=jr.PRNGKey(1))
    assert not jnp.allclose(y1, y2)


def test_ncp_perturb_zero_scale_is_identity():
    from pyrox.nn import NCPContinuousPerturb

    perturb = NCPContinuousPerturb(scale=0.0)
    x = jnp.ones((2, 3))
    assert jnp.allclose(perturb(x, key=jr.PRNGKey(0)), x)


# --- RandomKitchenSinks ----------------------------------------------------


def test_rks_output_shape():
    rff = RBFFourierFeatures.init(in_features=3, n_features=10)
    rks = RandomKitchenSinks.init(rff, out_features=2)
    with handlers.seed(rng_seed=0):
        assert rks(jnp.ones((4, 3))).shape == (4, 2)


def test_rks_registers_beta_and_bias_sites():
    rff = RBFFourierFeatures.init(in_features=3, n_features=5)
    rks = RandomKitchenSinks.init(rff, out_features=2)
    rks = RandomKitchenSinks(
        rff=rks.rff,
        init_beta=rks.init_beta,
        init_bias=rks.init_bias,
        pyrox_name="rks",
    )
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        rks(jnp.ones((2, 3)))
    assert "rks.beta" in tr
    assert "rks.bias" in tr
    assert tr["rks.beta"]["type"] == "sample"


def test_rks_with_matern_rff():
    rff = MaternFourierFeatures.init(in_features=3, n_features=10, nu=2.5)
    rks = RandomKitchenSinks.init(rff, out_features=2)
    with handlers.seed(rng_seed=0):
        assert rks(jnp.ones((4, 3))).shape == (4, 2)


# --- RBFCosineFeatures (cos(Wx + b) variant) -------------------------------


def test_rbf_cosine_output_shape():
    rff = RBFCosineFeatures.init(in_features=3, n_features=10)
    with handlers.seed(rng_seed=0):
        assert rff(jnp.ones((4, 3))).shape == (4, 10)


def test_rbf_cosine_registers_w_b_lengthscale():
    rff = RBFCosineFeatures(in_features=3, n_features=5, pyrox_name="cos_rff")
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        rff(jnp.ones((2, 3)))
    assert "cos_rff.W" in tr
    assert "cos_rff.b" in tr
    assert "cos_rff.lengthscale" in tr


def test_rbf_cosine_stochastic_across_seeds():
    rff = RBFCosineFeatures.init(in_features=3, n_features=10)
    x = jnp.ones((2, 3))
    with handlers.seed(rng_seed=0):
        y1 = rff(x)
    with handlers.seed(rng_seed=1):
        y2 = rff(x)
    assert not jnp.allclose(y1, y2)


def test_rbf_cosine_output_dim_is_n_features():
    """The cos(Wx+b) variant has output dim = n_features (not 2*n_features
    like the [cos, sin] variant)."""
    rff = RBFCosineFeatures.init(in_features=5, n_features=20)
    with handlers.seed(rng_seed=0):
        phi = rff(jnp.ones((3, 5)))
    assert phi.shape == (3, 20)


# --- MaternCosineFeatures (cos(Wx + b) variant) ----------------------------


def test_matern_cosine_output_shape():
    rff = MaternCosineFeatures.init(in_features=3, n_features=10, nu=1.5)
    with handlers.seed(rng_seed=0):
        assert rff(jnp.ones((4, 3))).shape == (4, 10)


def test_matern_cosine_registers_w_b_lengthscale():
    rff = MaternCosineFeatures(
        in_features=3, n_features=5, nu=1.5, pyrox_name="cos_matern"
    )
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        rff(jnp.ones((2, 3)))
    assert "cos_matern.W" in tr
    assert "cos_matern.b" in tr
    assert "cos_matern.lengthscale" in tr


def test_matern_cosine_stochastic_across_seeds():
    rff = MaternCosineFeatures.init(in_features=3, n_features=10, nu=1.5)
    x = jnp.ones((2, 3))
    with handlers.seed(rng_seed=0):
        y1 = rff(x)
    with handlers.seed(rng_seed=1):
        y2 = rff(x)
    assert not jnp.allclose(y1, y2)


def test_matern_cosine_output_dim_is_n_features():
    rff = MaternCosineFeatures.init(in_features=5, n_features=20, nu=2.5)
    with handlers.seed(rng_seed=0):
        phi = rff(jnp.ones((3, 5)))
    assert phi.shape == (3, 20)


def test_matern_cosine_rejects_invalid_params():
    import pytest

    with pytest.raises(ValueError):
        MaternCosineFeatures.init(in_features=2, n_features=4, nu=0.0)
    with pytest.raises(ValueError):
        MaternCosineFeatures.init(in_features=2, n_features=4, lengthscale=-1.0)


# --- LaplaceCosineFeatures (cos(Wx + b) variant) ---------------------------


def test_laplace_cosine_output_shape():
    rff = LaplaceCosineFeatures.init(in_features=3, n_features=10)
    with handlers.seed(rng_seed=0):
        assert rff(jnp.ones((4, 3))).shape == (4, 10)


def test_laplace_cosine_registers_w_b_lengthscale():
    rff = LaplaceCosineFeatures(in_features=3, n_features=5, pyrox_name="cos_laplace")
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        rff(jnp.ones((2, 3)))
    assert "cos_laplace.W" in tr
    assert "cos_laplace.b" in tr
    assert "cos_laplace.lengthscale" in tr


def test_laplace_cosine_stochastic_across_seeds():
    rff = LaplaceCosineFeatures.init(in_features=3, n_features=10)
    x = jnp.ones((2, 3))
    with handlers.seed(rng_seed=0):
        y1 = rff(x)
    with handlers.seed(rng_seed=1):
        y2 = rff(x)
    assert not jnp.allclose(y1, y2)


def test_laplace_cosine_output_dim_is_n_features():
    rff = LaplaceCosineFeatures.init(in_features=5, n_features=20)
    with handlers.seed(rng_seed=0):
        phi = rff(jnp.ones((3, 5)))
    assert phi.shape == (3, 20)


# --- Cross-formulation kernel-equivalence ----------------------------------


def _exact_rbf_gram(x, lengthscale):
    diff = x[:, None, :] - x[None, :, :]
    return jnp.exp(-0.5 * jnp.sum(diff**2, axis=-1) / lengthscale**2)


def _exact_matern32_gram(x, lengthscale):
    r = jnp.sqrt(jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1) + 1e-30)
    s = jnp.sqrt(3.0) * r / lengthscale
    return (1.0 + s) * jnp.exp(-s)


def _exact_laplace_gram(x, lengthscale):
    r = jnp.sqrt(jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1) + 1e-30)
    return jnp.exp(-r / lengthscale)


def _phi_under_seed(layer, x, lengthscale, seed):
    """Trace a layer under a fixed lengthscale and seed, return ``phi``."""
    name = f"{layer.pyrox_name}.lengthscale"
    with (
        handlers.substitute(data={name: jnp.asarray(lengthscale)}),
        handlers.seed(rng_seed=seed),
    ):
        return layer(x)


def test_paired_and_cosine_recover_same_kernel_rbf():
    """Both [cos, sin] and cos(Wx+b) RFFs converge to the same RBF kernel."""
    import equinox as eqx

    n, d, m = 12, 1, 4096
    ell = 0.7
    x = jnp.linspace(-1.0, 1.0, n).reshape(-1, d)
    paired = eqx.tree_at(
        lambda r: r.pyrox_name,
        RBFFourierFeatures.init(in_features=d, n_features=m, lengthscale=ell),
        "rbf_paired",
    )
    cosine = eqx.tree_at(
        lambda r: r.pyrox_name,
        RBFCosineFeatures.init(in_features=d, n_features=2 * m, lengthscale=ell),
        "rbf_cos",
    )
    K_exact = _exact_rbf_gram(x, ell)
    K_paired = _phi_under_seed(paired, x, ell, seed=0)
    K_paired = K_paired @ K_paired.T
    K_cos = _phi_under_seed(cosine, x, ell, seed=1)
    K_cos = K_cos @ K_cos.T
    norm = jnp.linalg.norm(K_exact)
    assert float(jnp.linalg.norm(K_paired - K_exact) / norm) < 0.05
    assert float(jnp.linalg.norm(K_cos - K_exact) / norm) < 0.10


def test_paired_and_cosine_recover_same_kernel_matern32():
    import equinox as eqx

    n, d, m = 12, 1, 4096
    ell = 0.7
    x = jnp.linspace(-1.0, 1.0, n).reshape(-1, d)
    paired = eqx.tree_at(
        lambda r: r.pyrox_name,
        MaternFourierFeatures.init(
            in_features=d, n_features=m, nu=1.5, lengthscale=ell
        ),
        "m_paired",
    )
    cosine = eqx.tree_at(
        lambda r: r.pyrox_name,
        MaternCosineFeatures.init(
            in_features=d, n_features=2 * m, nu=1.5, lengthscale=ell
        ),
        "m_cos",
    )
    K_exact = _exact_matern32_gram(x, ell)
    phi_p = _phi_under_seed(paired, x, ell, seed=0)
    phi_c = _phi_under_seed(cosine, x, ell, seed=1)
    norm = jnp.linalg.norm(K_exact)
    # Heavier-tailed Student-t spectrum + finite m → looser tolerance.
    assert float(jnp.linalg.norm(phi_p @ phi_p.T - K_exact) / norm) < 0.20
    assert float(jnp.linalg.norm(phi_c @ phi_c.T - K_exact) / norm) < 0.30


def test_paired_and_cosine_recover_same_kernel_laplace():
    import equinox as eqx

    n, d, m = 12, 1, 4096
    ell = 0.7
    x = jnp.linspace(-1.0, 1.0, n).reshape(-1, d)
    paired = eqx.tree_at(
        lambda r: r.pyrox_name,
        LaplaceFourierFeatures.init(in_features=d, n_features=m, lengthscale=ell),
        "l_paired",
    )
    cosine = eqx.tree_at(
        lambda r: r.pyrox_name,
        LaplaceCosineFeatures.init(in_features=d, n_features=2 * m, lengthscale=ell),
        "l_cos",
    )
    K_exact = _exact_laplace_gram(x, ell)
    phi_p = _phi_under_seed(paired, x, ell, seed=0)
    phi_c = _phi_under_seed(cosine, x, ell, seed=1)
    norm = jnp.linalg.norm(K_exact)
    # Cauchy spectrum has infinite variance in 1D → very loose tolerance.
    assert float(jnp.linalg.norm(phi_p @ phi_p.T - K_exact) / norm) < 0.40
    assert float(jnp.linalg.norm(phi_c @ phi_c.T - K_exact) / norm) < 0.50


# --- ArcCosineFourierFeatures (ReLU features) ------------------------------


def test_arccosine_order1_output_shape():
    rff = ArcCosineFourierFeatures.init(in_features=3, n_features=10, order=1)
    with handlers.seed(rng_seed=0):
        assert rff(jnp.ones((4, 3))).shape == (4, 10)


def test_arccosine_order0_output_shape():
    rff = ArcCosineFourierFeatures.init(in_features=3, n_features=10, order=0)
    with handlers.seed(rng_seed=0):
        assert rff(jnp.ones((4, 3))).shape == (4, 10)


def test_arccosine_order2_output_shape():
    rff = ArcCosineFourierFeatures.init(in_features=3, n_features=10, order=2)
    with handlers.seed(rng_seed=0):
        assert rff(jnp.ones((4, 3))).shape == (4, 10)


def test_arccosine_registers_w_and_lengthscale():
    rff = ArcCosineFourierFeatures(in_features=3, n_features=5, pyrox_name="arc_rff")
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        rff(jnp.ones((2, 3)))
    assert "arc_rff.W" in tr
    assert "arc_rff.lengthscale" in tr


def test_arccosine_order1_nonnegative():
    """Order-1 (ReLU) features are non-negative."""
    rff = ArcCosineFourierFeatures.init(in_features=3, n_features=50, order=1)
    with handlers.seed(rng_seed=0):
        phi = rff(jnp.ones((10, 3)))
    assert jnp.all(phi >= 0.0)


def test_arccosine_order0_is_heaviside():
    """Order-0 features are binary (0 or scaled 1) — the Heaviside step,
    not 0**0 = 1 everywhere."""
    rff = ArcCosineFourierFeatures.init(in_features=3, n_features=50, order=0)
    with handlers.seed(rng_seed=0):
        phi = rff(jnp.ones((10, 3)))
    scale = jnp.sqrt(2.0 / 50)
    unique_vals = jnp.unique(phi)
    assert jnp.allclose(unique_vals, jnp.array([0.0, scale]), atol=1e-6)


def test_arccosine_stochastic_across_seeds():
    rff = ArcCosineFourierFeatures.init(in_features=3, n_features=10)
    x = jnp.ones((2, 3))
    with handlers.seed(rng_seed=0):
        y1 = rff(x)
    with handlers.seed(rng_seed=1):
        y2 = rff(x)
    assert not jnp.allclose(y1, y2)
