"""Tests for ``pyrox.gp.SparseGPPrior``.

The sparse prior is a thin shell — kernel + inducing inputs + jitter +
helper accessors. Verify shapes, dtype-preservation, jitter regularization,
and that the helpers compose correctly with the kernel.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from gaussx import CGSolver, DenseSolver
from numpyro import handlers

from pyrox.gp import RBF, Matern, SparseGPPrior


def _toy_inducing(M: int = 5, D: int = 1) -> jnp.ndarray:
    return jnp.linspace(-2.0, 2.0, M).reshape(M, D)


def test_num_inducing_property_returns_m():
    Z = _toy_inducing(7)
    prior = SparseGPPrior(kernel=RBF(), Z=Z)
    assert prior.num_inducing == 7


def test_zero_mean_default():
    Z = _toy_inducing()
    prior = SparseGPPrior(kernel=RBF(), Z=Z)
    out = prior.mean(Z)
    assert out.shape == (Z.shape[0],)
    assert jnp.allclose(out, 0.0)


def test_callable_mean_fn_is_applied_to_input():
    Z = _toy_inducing()
    prior = SparseGPPrior(kernel=RBF(), Z=Z, mean_fn=lambda x: x.squeeze(-1) * 2.0)
    X = jnp.array([[0.5], [1.0], [-0.5]])
    assert jnp.allclose(prior.mean(X), jnp.array([1.0, 2.0, -1.0]))


def test_inducing_operator_includes_jitter_regularization():
    """``K_zz + jitter * I`` — diagonal must be lifted by exactly ``jitter``."""
    Z = _toy_inducing(5)
    jitter = 5e-3
    kern = RBF(init_variance=1.5, init_lengthscale=0.4)
    prior = SparseGPPrior(kernel=kern, Z=Z, jitter=jitter)
    K = kern(Z, Z)
    K_reg = prior.inducing_operator().as_matrix()
    assert jnp.allclose(K_reg, K + jitter * jnp.eye(Z.shape[0]), atol=1e-10)


def test_cross_covariance_matches_kernel_call():
    """``cross_covariance(X) == kernel(X, Z)`` — pure delegation."""
    Z = _toy_inducing(4)
    X = jnp.linspace(-3.0, 3.0, 7).reshape(-1, 1)
    kern = Matern(init_variance=0.8, init_lengthscale=0.5)
    prior = SparseGPPrior(kernel=kern, Z=Z)
    assert jnp.allclose(prior.cross_covariance(X), kern(X, Z))


def test_kernel_diag_matches_kernel_diag_call():
    """``kernel_diag(X) == kernel.diag(X)`` — pure delegation; stationary
    kernels override ``diag`` for an O(N) shortcut, which the sparse prior
    must preserve."""
    Z = _toy_inducing(4)
    X = jnp.linspace(-3.0, 3.0, 7).reshape(-1, 1)
    kern = RBF(init_variance=2.0, init_lengthscale=0.7)
    prior = SparseGPPrior(kernel=kern, Z=Z)
    assert jnp.allclose(prior.kernel_diag(X), kern.diag(X))


def test_resolved_solver_defaults_to_dense_solver():
    Z = _toy_inducing()
    prior = SparseGPPrior(kernel=RBF(), Z=Z)
    assert isinstance(prior._resolved_solver(), DenseSolver)


def test_resolved_solver_passes_through_explicit_solver():
    Z = _toy_inducing()
    explicit = CGSolver()
    prior = SparseGPPrior(kernel=RBF(), Z=Z, solver=explicit)
    assert prior._resolved_solver() is explicit


def test_inducing_operator_is_psd_tagged():
    """The PSD tag is what unlocks Cholesky-routed solvers in gaussx."""
    import lineax as lx

    Z = _toy_inducing(3)
    op = SparseGPPrior(kernel=RBF(), Z=Z).inducing_operator()
    assert lx.is_positive_semidefinite(op)


# --- Pattern B/C regression: kernel context scoping ------------------------


def test_predictive_blocks_returns_consistent_shapes_and_jitter():
    """``predictive_blocks(X)`` returns the same three matrices that
    :meth:`inducing_operator`, :meth:`cross_covariance`, and
    :meth:`kernel_diag` produce for pure ``eqx.Module`` kernels — but in
    one shared kernel context (verified separately for Pattern B / C
    kernels)."""
    Z = _toy_inducing(4)
    X = jnp.linspace(-3.0, 3.0, 6).reshape(-1, 1)
    kern = RBF(init_variance=1.5, init_lengthscale=0.4)
    prior = SparseGPPrior(kernel=kern, Z=Z, jitter=2e-4)

    K_zz_op, K_xz, K_xx_diag = prior.predictive_blocks(X)
    assert K_zz_op.in_size() == Z.shape[0]
    assert K_xz.shape == (X.shape[0], Z.shape[0])
    assert K_xx_diag.shape == (X.shape[0],)

    K_zz_solo = prior.inducing_operator().as_matrix()
    assert jnp.allclose(K_zz_op.as_matrix(), K_zz_solo, atol=1e-10)
    assert jnp.allclose(K_xz, prior.cross_covariance(X), atol=1e-10)
    assert jnp.allclose(K_xx_diag, prior.kernel_diag(X), atol=1e-10)


def test_predictive_blocks_does_not_double_register_pattern_b_kernel_sites():
    """Regression for PR #64 P1: building ``K_zz``, ``K_xz``, and
    ``K_xx_diag`` for an SVGP predictive requires three kernel calls.
    For a kernel whose hyperparameters carry priors (Pattern B / C),
    those calls would re-register the same NumPyro sample sites without
    a shared kernel context — raising a duplicate-site error under
    tracing. ``predictive_blocks`` scopes one outer
    :class:`pyrox.PyroxModule` context so each prior'd hyperparameter
    site registers exactly once."""
    Z = _toy_inducing(4)
    X = jnp.linspace(-3.0, 3.0, 5).reshape(-1, 1)

    kernel = RBF()
    kernel.set_prior("variance", dist.LogNormal(0.0, 1.0))
    kernel.set_prior("lengthscale", dist.LogNormal(0.0, 1.0))
    prior = SparseGPPrior(kernel=kernel, Z=Z, jitter=1e-4)

    def model():
        return prior.predictive_blocks(X)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        K_zz_op, K_xz, K_xx_diag = model()

    assert K_zz_op.in_size() == Z.shape[0]
    assert K_xz.shape == (X.shape[0], Z.shape[0])
    assert K_xx_diag.shape == (X.shape[0],)
    # Each prior'd hyperparameter site registers exactly once across
    # the three kernel evaluations — duplicate registration would have
    # raised before this point.
    assert "RBF.variance" in tr
    assert "RBF.lengthscale" in tr


def test_predictive_blocks_uses_consistent_kernel_hyperparameter_draws():
    """Under one shared seed scope, ``predictive_blocks`` draws the
    kernel hyperparameter samples once and reuses them across the three
    kernel evaluations — so ``K_zz``, ``K_xz``, and ``K_xx_diag`` all
    reflect the same hyperparameter draw. Calling the three accessors
    separately under independent seed scopes would draw three
    independent samples and yield mutually inconsistent matrices."""
    Z = _toy_inducing(4)
    X = jnp.linspace(-2.0, 2.0, 5).reshape(-1, 1)

    kernel = RBF()
    kernel.set_prior("variance", dist.LogNormal(0.0, 1.0))
    kernel.set_prior("lengthscale", dist.LogNormal(0.0, 1.0))
    prior = SparseGPPrior(kernel=kernel, Z=Z, jitter=1e-4)

    with handlers.seed(rng_seed=0):
        K_zz_op, K_xz, K_xx_diag = prior.predictive_blocks(X)
    K_zz = K_zz_op.as_matrix()

    # Recover the variance from the diagonal of K_zz (RBF: K(z, z) = variance).
    variance_zz = K_zz[0, 0] - prior.jitter
    # Same variance must appear on the diagonal of K_xx (stationary RBF).
    assert jnp.allclose(K_xx_diag, variance_zz, atol=1e-6)
    # And K_xz is bounded by the same variance (k(x, z) <= variance).
    assert jnp.all(K_xz <= variance_zz + 1e-6)


# --- Prior-side sample / log_prob (PR #66 review) --------------------------


def test_inducing_log_prob_matches_numpyro_mvn():
    """``SparseGPPrior.log_prob(u)`` must match a reference numpyro MVN
    over the inducing prior :math:`p(u) = N(0, K_zz + jitter*I)`."""
    Z = _toy_inducing(5)
    prior = SparseGPPrior(kernel=RBF(init_variance=1.5, init_lengthscale=0.4), Z=Z)
    K = prior.inducing_operator().as_matrix()
    u = jr.normal(jr.PRNGKey(11), (Z.shape[0],))
    ref = dist.MultivariateNormal(jnp.zeros_like(u), covariance_matrix=K).log_prob(u)
    assert jnp.allclose(prior.log_prob(u), ref, atol=1e-5)


def test_inducing_log_prob_invariant_to_dense_solver_choice():
    """Default solver resolution (``None`` → :class:`DenseSolver`) matches
    an explicitly constructed :class:`DenseSolver` — confirms the field
    is wired through to :func:`gaussx.gaussian_log_prob`."""
    Z = _toy_inducing(5)
    p_default = SparseGPPrior(kernel=RBF(), Z=Z)
    p_dense = SparseGPPrior(kernel=RBF(), Z=Z, solver=DenseSolver())
    u = jr.normal(jr.PRNGKey(12), (Z.shape[0],))
    assert jnp.allclose(p_default.log_prob(u), p_dense.log_prob(u), atol=1e-5)


def test_inducing_sample_returns_correct_shape():
    Z = _toy_inducing(7)
    prior = SparseGPPrior(kernel=RBF(), Z=Z)
    u = prior.sample(jr.PRNGKey(13))
    assert u.shape == (7,)


def test_inducing_sample_is_deterministic_under_same_key():
    """Same key, same prior → same draw — ``MultivariateNormal.sample``
    is reparameterized so this is an explicit reproducibility check."""
    Z = _toy_inducing(5)
    prior = SparseGPPrior(kernel=RBF(), Z=Z)
    key = jr.PRNGKey(14)
    assert jnp.allclose(prior.sample(key), prior.sample(key), atol=0.0)
