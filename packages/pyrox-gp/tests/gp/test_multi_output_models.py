"""Tests for the multi-output GP model layer.

Covers :class:`MultiOutputGPPrior` / :class:`MultiOutputConditionedGP` /
:func:`mo_gp_factor` (exact) and :class:`MultiOutputSparseGPPrior` /
:func:`mo_svgp_elbo` / :func:`mo_svgp_factor` (inducing inputs).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from gaussx import (
    GaussHermiteIntegrator,
    is_block_diagonal,
    log_marginal_likelihood,
)
from numpyro import handlers
from pyrox_gp import (
    RBF,
    DistLikelihood,
    FullRankGuide,
    GaussianLikelihood,
    GPPrior,
    ICMKernel,
    LMCKernel,
    MeanFieldGuide,
    MultiOutputGPPrior,
    MultiOutputSparseGPPrior,
    OILMMKernel,
    SharedInducingPoints,
    SparseGPPrior,
    WhitenedGuide,
    mo_gp_factor,
    mo_svgp_elbo,
    mo_svgp_factor,
    svgp_elbo,
)


def _lmc(P: int = 3, Q: int = 2) -> LMCKernel:
    kernels = tuple(
        RBF(
            pyrox_name=f"RBF_q{q}",
            init_variance=1.0 - 0.3 * q,
            init_lengthscale=0.6 + 0.4 * q,
        )
        for q in range(Q)
    )
    mixing = jnp.array([[1.0, 0.4], [0.3, -0.8], [0.5, 0.2]])[:P, :Q]
    return LMCKernel(kernels=kernels, mixing=mixing)


def _data(N: int = 8, D: int = 2, P: int = 3, seed: int = 0):
    X = jr.uniform(jr.PRNGKey(seed), (N, D))
    Y = jr.normal(jr.PRNGKey(seed + 1), (N, P))
    return X, Y


def _vec(Y: jnp.ndarray) -> jnp.ndarray:
    """Output-major ``(p n)`` flattening used by the Kronecker operators."""
    return Y.T.reshape(-1)


# ---------------------------------------------------------------------------
# MultiOutputGPPrior — exact dense workflow
# ---------------------------------------------------------------------------


def test_mo_prior_mean_and_sample_shapes():
    X, _ = _data()
    prior = MultiOutputGPPrior(kernel=_lmc(), X=X)
    assert prior.num_outputs == 3
    assert prior.mean(X).shape == (8, 3)
    assert prior.sample(jr.PRNGKey(0)).shape == (8, 3)


def test_mo_condition_predict_shapes_and_positive_variance():
    X, Y = _data()
    cond = MultiOutputGPPrior(kernel=_lmc(), X=X).condition(Y, noise_var=0.1)
    mean, var = cond.predict(X[:5])
    assert mean.shape == (5, 3)
    assert var.shape == (5, 3)
    assert bool((var > 0.0).all())


def test_mo_condition_matches_dense_reference():
    """Predictions must equal the naive dense multi-output GP regression."""
    X, Y = _data()
    kernel = _lmc()
    jitter, noise = 1e-5, 0.1
    cond = MultiOutputGPPrior(kernel=kernel, X=X, jitter=jitter).condition(
        Y, noise_var=noise
    )
    mean, var = cond.predict(X)

    K_ff = kernel.full_covariance(X)
    noisy = K_ff + (jitter + noise) * jnp.eye(K_ff.shape[0])
    alpha = jnp.linalg.solve(noisy, _vec(Y))
    K_cross = kernel.cross_covariance(X, X)
    mean_ref = (K_cross @ alpha).reshape(3, -1).T
    solve_cross = jnp.linalg.solve(noisy, K_cross.T)
    var_ref = (
        (jnp.diag(K_ff) - jnp.einsum("ij,ji->i", K_cross, solve_cross)).reshape(3, -1).T
    )
    assert jnp.allclose(mean, mean_ref, atol=1e-4)
    assert jnp.allclose(var, var_ref, atol=1e-3)


def test_mo_per_output_noise_matches_dense_reference():
    X, Y = _data()
    kernel = _lmc()
    noise = jnp.array([0.01, 0.1, 1.0])
    jitter = 1e-5
    cond = MultiOutputGPPrior(kernel=kernel, X=X, jitter=jitter).condition(
        Y, noise_var=noise
    )
    mean, _ = cond.predict(X)

    K_ff = kernel.full_covariance(X)
    noise_diag = jnp.repeat(noise, X.shape[0])
    noisy = K_ff + jnp.diag(jitter + noise_diag)
    mean_ref = (
        (kernel.cross_covariance(X, X) @ jnp.linalg.solve(noisy, _vec(Y)))
        .reshape(3, -1)
        .T
    )
    assert jnp.allclose(mean, mean_ref, atol=1e-4)


def test_mo_single_output_matches_gpprior():
    """P=1, Q=1 LMC with unit mixing must reproduce the scalar GPPrior."""
    X, Y = _data(P=1)
    rbf = RBF(init_variance=1.2, init_lengthscale=0.7)
    lmc = LMCKernel(kernels=(rbf,), mixing=jnp.array([[1.0]]))
    mo = MultiOutputGPPrior(kernel=lmc, X=X, jitter=1e-5).condition(Y, 0.1)
    so = GPPrior(kernel=rbf, X=X, jitter=1e-5).condition(Y[:, 0], 0.1)

    mean_mo, var_mo = mo.predict(X)
    mean_so, var_so = so.predict(X)
    assert jnp.allclose(mean_mo[:, 0], mean_so, atol=1e-5)
    assert jnp.allclose(var_mo[:, 0], var_so, atol=1e-5)


def test_mo_log_prob_matches_dense_reference():
    X, Y = _data(N=5)
    kernel = _lmc()
    jitter = 1e-4
    prior = MultiOutputGPPrior(kernel=kernel, X=X, jitter=jitter)
    K = kernel.full_covariance(X) + jitter * jnp.eye(15)
    ref = dist.MultivariateNormal(jnp.zeros(15), K).log_prob(_vec(Y))
    assert jnp.allclose(prior.log_prob(Y), ref, atol=1e-2, rtol=1e-4)


def test_mo_icm_condition_matches_dense_reference():
    X, Y = _data(P=2)
    icm = ICMKernel(
        kernel=RBF(init_variance=1.0, init_lengthscale=0.8),
        mixing=jnp.array([[1.0, 0.2], [0.5, -0.3]]),
    )
    jitter, noise = 1e-6, 0.05
    cond = MultiOutputGPPrior(kernel=icm, X=X, jitter=jitter).condition(Y, noise)
    mean, _ = cond.predict(X)

    K_ff = icm.full_covariance(X)
    noisy = K_ff + (jitter + noise) * jnp.eye(K_ff.shape[0])
    mean_ref = (
        (icm.cross_covariance(X, X) @ jnp.linalg.solve(noisy, _vec(Y))).reshape(2, -1).T
    )
    assert jnp.allclose(mean, mean_ref, atol=1e-4)


def test_mo_oilmm_kernel_accepted_by_exact_model():
    X, Y = _data(P=2)
    W, _ = jnp.linalg.qr(jr.normal(jr.PRNGKey(3), (2, 2)))
    oilmm = OILMMKernel(
        kernels=(
            RBF(pyrox_name="RBF_q0", init_lengthscale=0.5),
            RBF(pyrox_name="RBF_q1", init_lengthscale=1.5),
        ),
        mixing=W,
    )
    mean, var = MultiOutputGPPrior(kernel=oilmm, X=X).condition(Y, 0.1).predict(X[:4])
    assert mean.shape == (4, 2)
    assert bool(jnp.isfinite(var).all())


def test_mo_conditioned_sample_shape():
    X, Y = _data()
    cond = MultiOutputGPPrior(kernel=_lmc(), X=X).condition(Y, 0.1)
    samples = cond.sample(jr.PRNGKey(0), X[:4], n_samples=5)
    assert samples.shape == (5, 4, 3)
    assert bool(jnp.isfinite(samples).all())


def test_mo_mean_fn_added_to_predictions():
    X, Y = _data()
    offset = 2.5

    def mean_fn(X):
        return jnp.full((X.shape[0], 3), offset, dtype=X.dtype)

    base = MultiOutputGPPrior(kernel=_lmc(), X=X).condition(Y, 0.1)
    shifted = MultiOutputGPPrior(kernel=_lmc(), X=X, mean_fn=mean_fn).condition(
        Y + offset, 0.1
    )
    mean_base, _ = base.predict(X[:4])
    mean_shifted, _ = shifted.predict(X[:4])
    assert jnp.allclose(mean_shifted, mean_base + offset, atol=1e-4)


def test_mo_condition_rejects_wrong_target_shape():
    X, Y = _data()
    prior = MultiOutputGPPrior(kernel=_lmc(), X=X)
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        prior.condition(Y[:, :2], 0.1)
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        prior.condition(Y[:, 0], 0.1)


def test_mo_condition_rejects_bad_noise_shape():
    X, Y = _data()
    prior = MultiOutputGPPrior(kernel=_lmc(), X=X)
    with pytest.raises(ValueError, match="noise_var"):
        prior.condition(Y, noise_var=jnp.ones(2))


def test_mo_gp_factor_registers_marginal_likelihood():
    X, Y = _data()
    prior = MultiOutputGPPrior(kernel=_lmc(), X=X)

    def model():
        mo_gp_factor("mll", prior, Y, noise_var=0.1)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model()
    logp = tr["mll"]["fn"].log_factor
    ref = log_marginal_likelihood(jnp.zeros(24), prior._noisy_operator(0.1), _vec(Y))
    assert jnp.allclose(logp, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# MultiOutputSparseGPPrior — inducing-input workflow
# ---------------------------------------------------------------------------


def _sparse_setup(M: int = 4, P: int = 3, Q: int = 2, seed: int = 0):
    X, Y = _data(P=P, seed=seed)
    Z = jr.uniform(jr.PRNGKey(seed + 7), (M, X.shape[1]))
    prior = MultiOutputSparseGPPrior(
        kernel=_lmc(P=P, Q=Q),
        inducing=SharedInducingPoints(locations=Z),
        jitter=1e-5,
    )
    return prior, X, Y


def test_mo_sparse_predictive_blocks_shapes_and_structure():
    prior, X, _ = _sparse_setup()
    K_uu_op, K_fu, K_ff_diag = prior.predictive_blocks(X)
    assert prior.num_inducing == 8  # Q=2 latents x M=4 inducing
    assert is_block_diagonal(K_uu_op)
    assert K_uu_op.as_matrix().shape == (8, 8)
    assert K_fu.shape == (24, 8)
    assert K_ff_diag.shape == (24,)


def test_mo_sparse_blocks_match_kernel_level_assembly():
    """The prior's blocks must agree with the kernel-level helpers."""
    prior, X, _ = _sparse_setup()
    kernel, shared = prior.kernel, prior.inducing
    K_uu_op, K_fu, K_ff_diag = prior.predictive_blocks(X)

    K_uu_ref = shared.K_uu(kernel.kernels) + prior.jitter * jnp.eye(8)
    assert jnp.allclose(K_uu_op.as_matrix(), K_uu_ref, atol=1e-6)

    from pyrox_gp import MultiOutputInducingVariables

    iv = MultiOutputInducingVariables.from_kernel(kernel, shared)
    K_uf_ref = iv.K_uf(X, kernel.kernels)
    assert jnp.allclose(K_fu, K_uf_ref.T, atol=1e-6)
    assert jnp.allclose(K_ff_diag, _vec(kernel.diag(X)), atol=1e-6)


def test_mo_sparse_rejects_oilmm_kernel():
    Z = jnp.zeros((3, 1))
    W = jnp.eye(2)
    oilmm = OILMMKernel(
        kernels=(
            RBF(pyrox_name="RBF_q0"),
            RBF(pyrox_name="RBF_q1"),
        ),
        mixing=W,
    )
    with pytest.raises(TypeError, match="LMCKernel or ICMKernel"):
        MultiOutputSparseGPPrior(
            kernel=oilmm, inducing=SharedInducingPoints(locations=Z)
        )


def test_mo_sparse_rejects_icm_with_nonzero_kappa():
    Z = jnp.zeros((3, 1))
    icm = ICMKernel(
        kernel=RBF(),
        mixing=jnp.array([[1.0], [0.5]]),
        kappa=jnp.array([0.1, 0.2]),
    )
    with pytest.raises(ValueError, match="kappa"):
        MultiOutputSparseGPPrior(kernel=icm, inducing=SharedInducingPoints(locations=Z))


@pytest.mark.parametrize("guide_cls", [FullRankGuide, MeanFieldGuide, WhitenedGuide])
def test_mo_svgp_elbo_finite_across_guides(guide_cls):
    prior, X, Y = _sparse_setup()
    guide = guide_cls.init(num_inducing=prior.num_inducing)
    elbo = mo_svgp_elbo(prior, guide, GaussianLikelihood(noise_var=0.1), X, Y)
    assert elbo.shape == ()
    assert bool(jnp.isfinite(elbo))


def test_mo_svgp_elbo_lower_bounds_exact_marginal_likelihood():
    """For any guide, ELBO <= log p(Y) of the exact multi-output GP."""
    prior, X, Y = _sparse_setup()
    noise = 0.1
    guide = FullRankGuide.init(num_inducing=prior.num_inducing)
    elbo = mo_svgp_elbo(prior, guide, GaussianLikelihood(noise_var=noise), X, Y)

    exact = MultiOutputGPPrior(kernel=prior.kernel, X=X, jitter=prior.jitter)
    mll = log_marginal_likelihood(jnp.zeros(24), exact._noisy_operator(noise), _vec(Y))
    assert elbo <= mll + 1e-4


def test_mo_svgp_elbo_single_output_matches_svgp_elbo():
    """P=1, Q=1 must reduce to the single-output SVGP ELBO exactly."""
    X, Y = _data(P=1)
    Z = jr.uniform(jr.PRNGKey(9), (4, X.shape[1]))
    rbf = RBF(init_variance=1.1, init_lengthscale=0.6)
    lmc = LMCKernel(kernels=(rbf,), mixing=jnp.array([[1.0]]))
    mo_prior = MultiOutputSparseGPPrior(
        kernel=lmc, inducing=SharedInducingPoints(locations=Z), jitter=1e-5
    )
    so_prior = SparseGPPrior(kernel=rbf, Z=Z, jitter=1e-5)
    guide = FullRankGuide.init(num_inducing=4)
    lik = GaussianLikelihood(noise_var=0.1)

    elbo_mo = mo_svgp_elbo(mo_prior, guide, lik, X, Y)
    elbo_so = svgp_elbo(so_prior, guide, lik, X, Y[:, 0])
    assert jnp.allclose(elbo_mo, elbo_so, atol=1e-5)


def test_mo_svgp_elbo_improves_with_informative_guide():
    """The closed-form optimal guide mean must raise the ELBO over zero.

    For fixed guide covariance the ELBO is quadratic in the guide mean
    ``m`` with maximizer
    ``(A^T A / s2 + K_uu^{-1}) m = A^T y / s2`` where
    ``A = K_fu K_uu^{-1}`` maps ``m`` to the predictive mean.
    """
    prior, X, Y = _sparse_setup()
    noise = 0.1
    lik = GaussianLikelihood(noise_var=noise)
    guide_zero = FullRankGuide.init(num_inducing=prior.num_inducing)
    elbo_zero = mo_svgp_elbo(prior, guide_zero, lik, X, Y)

    K_uu_op, K_fu, _ = prior.predictive_blocks(X)
    K_uu_inv = jnp.linalg.inv(K_uu_op.as_matrix())
    A = K_fu @ K_uu_inv
    m_star = jnp.linalg.solve(A.T @ A / noise + K_uu_inv, A.T @ _vec(Y) / noise)
    guide_fit = FullRankGuide(mean=m_star, scale_tril=guide_zero.scale_tril)
    elbo_fit = mo_svgp_elbo(prior, guide_fit, lik, X, Y)
    assert elbo_fit > elbo_zero


def test_mo_sparse_predict_shapes_and_prior_variance_cap():
    prior, X, _ = _sparse_setup()
    guide = WhitenedGuide.init(num_inducing=prior.num_inducing, scale=1.0)
    mean, var = prior.predict(guide, X[:5])
    assert mean.shape == (5, 3)
    assert var.shape == (5, 3)
    # With a unit whitened guide the predictive variance equals the
    # prior variance; it can never exceed it plus numerical slack.
    assert bool((var <= prior.kernel.diag(X[:5]) + 1e-4).all())


def test_mo_sparse_mean_fn_added_to_predict():
    prior, X, _ = _sparse_setup()
    offset = 1.5
    shifted = MultiOutputSparseGPPrior(
        kernel=prior.kernel,
        inducing=prior.inducing,
        mean_fn=lambda X: jnp.full((X.shape[0], 3), offset, dtype=X.dtype),
        jitter=prior.jitter,
    )
    guide = FullRankGuide.init(num_inducing=prior.num_inducing)
    mean_base, _ = prior.predict(guide, X[:4])
    mean_shifted, _ = shifted.predict(guide, X[:4])
    assert jnp.allclose(mean_shifted, mean_base + offset, atol=1e-5)


def test_mo_svgp_elbo_rejects_wrong_target_shape():
    prior, X, Y = _sparse_setup()
    guide = FullRankGuide.init(num_inducing=prior.num_inducing)
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        mo_svgp_elbo(prior, guide, GaussianLikelihood(noise_var=0.1), X, Y[:, :2])


def test_mo_svgp_elbo_nonconjugate_requires_integrator():
    prior, X, Y = _sparse_setup()
    guide = FullRankGuide.init(num_inducing=prior.num_inducing)
    lik = DistLikelihood(lambda f: dist.Bernoulli(logits=f))
    with pytest.raises(ValueError, match="integrator"):
        mo_svgp_elbo(prior, guide, lik, X, Y)


def test_mo_svgp_elbo_nonconjugate_with_integrator():
    prior, X, Y = _sparse_setup()
    Y_bin = (Y > 0).astype(X.dtype)
    guide = FullRankGuide.init(num_inducing=prior.num_inducing)
    lik = DistLikelihood(lambda f: dist.Bernoulli(logits=f))
    elbo = mo_svgp_elbo(
        prior, guide, lik, X, Y_bin, integrator=GaussHermiteIntegrator(order=20)
    )
    assert bool(jnp.isfinite(elbo))


def test_mo_svgp_factor_registers_elbo():
    prior, X, Y = _sparse_setup()
    guide = FullRankGuide.init(num_inducing=prior.num_inducing)
    lik = GaussianLikelihood(noise_var=0.1)

    def model():
        mo_svgp_factor("elbo", prior, guide, lik, X, Y)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model()
    logp = tr["elbo"]["fn"].log_factor
    assert jnp.allclose(logp, mo_svgp_elbo(prior, guide, lik, X, Y), atol=1e-5)


def test_mo_icm_sparse_elbo_finite_and_ties_latent_kernel():
    """ICM shares one kernel instance across latents — the block builder
    must accept the repeated instance and produce a finite ELBO."""
    X, Y = _data(P=2)
    Z = jr.uniform(jr.PRNGKey(5), (3, X.shape[1]))
    icm = ICMKernel(
        kernel=RBF(init_variance=1.0, init_lengthscale=0.8),
        mixing=jnp.array([[1.0, 0.2], [0.5, -0.3]]),
    )
    prior = MultiOutputSparseGPPrior(
        kernel=icm, inducing=SharedInducingPoints(locations=Z)
    )
    assert prior.num_inducing == 6  # Q=2 tied latents x M=3
    guide = FullRankGuide.init(num_inducing=prior.num_inducing)
    elbo = mo_svgp_elbo(prior, guide, GaussianLikelihood(noise_var=0.1), X, Y)
    assert bool(jnp.isfinite(elbo))


# ---------------------------------------------------------------------------
# Priored kernels — shared-context regression tests
# ---------------------------------------------------------------------------


def test_mo_gp_factor_shares_context_for_priored_tied_kernel():
    """A priored kernel reused across latents registers each site once."""
    kernel = RBF()
    kernel.set_prior("variance", dist.LogNormal(0.0, 0.3))
    kernel.set_prior("lengthscale", dist.LogNormal(0.0, 0.3))
    lmc = LMCKernel(
        kernels=(kernel, kernel), mixing=jnp.array([[1.0, 0.5], [0.25, -1.0]])
    )
    X = jnp.array([[0.0], [0.5], [1.0]])
    Y = jnp.zeros((3, 2))
    prior = MultiOutputGPPrior(kernel=lmc, X=X)

    def model():
        mo_gp_factor("mll", prior, Y, noise_var=0.1)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model()
    assert "RBF.variance" in tr
    assert "RBF.lengthscale" in tr
    assert "mll" in tr


def test_mo_svgp_elbo_shares_context_for_priored_tied_kernel():
    """All three sparse blocks must see one hyperparameter draw."""
    kernel = RBF()
    kernel.set_prior("variance", dist.LogNormal(0.0, 0.3))
    lmc = LMCKernel(
        kernels=(kernel, kernel), mixing=jnp.array([[1.0, 0.5], [0.25, -1.0]])
    )
    X = jnp.array([[0.0], [0.5], [1.0]])
    Y = jnp.zeros((3, 2))
    prior = MultiOutputSparseGPPrior(
        kernel=lmc, inducing=SharedInducingPoints(locations=jnp.array([[-1.0], [1.0]]))
    )
    guide = FullRankGuide.init(num_inducing=prior.num_inducing)

    def model():
        mo_svgp_factor("elbo", prior, guide, GaussianLikelihood(noise_var=0.1), X, Y)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model()
    assert "RBF.variance" in tr
    assert "elbo" in tr
