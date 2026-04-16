"""Tests for pyrox.gp model-facing API.

Verify (1) exact GP regression against direct closed-form linear algebra,
(2) NumPyro integration — ``gp_factor`` under MCMC and SVI, ``gp_sample``
for latent-function workflows, and (3) that solvers are pluggable (any
``gaussx.AbstractSolverStrategy`` works).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from gaussx import CGSolver, ComposedSolver, DenseLogdet, DenseSolver
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

from pyrox.gp import (
    RBF,
    ConditionedGP,
    GPPrior,
    gp_factor,
    gp_sample,
)
from pyrox.gp._src import kernels as _k


# --- Fixtures --------------------------------------------------------------


def _toy_dataset(n: int = 6, noise_scale: float = 0.05, seed: int = 0):
    key = jr.PRNGKey(seed)
    X = jnp.linspace(-1.0, 1.0, n).reshape(-1, 1)
    y = jnp.sin(X.squeeze(-1)) + noise_scale * jr.normal(key, (n,))
    return X, y


# --- GPPrior construction -------------------------------------------------


def test_gpprior_defaults_zero_mean_and_dense_solver():
    X = jnp.zeros((3, 1))
    p = GPPrior(kernel=RBF(), X=X)
    assert jnp.allclose(p.mean(X), 0.0)
    # Solver is None until resolved; _resolved_solver returns a DenseSolver.
    assert isinstance(p._resolved_solver(), DenseSolver)


def test_gpprior_accepts_callable_mean_fn():
    X = jnp.array([[0.0], [1.0], [2.0]])
    p = GPPrior(kernel=RBF(), X=X, mean_fn=lambda x: x.squeeze(-1) + 1.0)
    assert jnp.allclose(p.mean(X), jnp.array([1.0, 2.0, 3.0]))


def test_gpprior_log_prob_matches_dense_mvn_log_prob():
    X, _ = _toy_dataset()
    p = GPPrior(kernel=RBF(init_variance=1.2, init_lengthscale=0.5), X=X, jitter=1e-6)
    f = jnp.array([0.1, -0.2, 0.0, 0.3, -0.1, 0.2])

    # Reference: numpyro MVN with dense covariance.
    with handlers.seed(rng_seed=0):
        K = _k.rbf_kernel(X, X, jnp.array(1.2), jnp.array(0.5))
    K = K + 1e-6 * jnp.eye(K.shape[0])
    ref = dist.MultivariateNormal(jnp.zeros(6), covariance_matrix=K).log_prob(f)

    with handlers.seed(rng_seed=0):
        got = p.log_prob(f)
    assert jnp.allclose(got, ref, atol=1e-4)


# --- Exact regression: closed-form posterior predictive --------------------


def test_exact_regression_mean_matches_closed_form():
    X, y = _toy_dataset()
    noise_var = jnp.array(0.01)
    kernel = RBF(init_variance=1.0, init_lengthscale=0.4)

    with handlers.seed(rng_seed=0):
        prior = GPPrior(kernel=kernel, X=X, jitter=1e-8)
        cond = prior.condition(y, noise_var)
        mu = cond.predict_mean(X)

    # Closed-form reference using dense numpy.
    with handlers.seed(rng_seed=0):
        K = kernel(X, X) + (1e-8 + 0.01) * jnp.eye(X.shape[0])
        K_star = kernel(X, X)
    alpha = jnp.linalg.solve(K, y)
    ref = K_star @ alpha
    assert jnp.allclose(mu, ref, atol=1e-4)


def test_exact_regression_variance_matches_closed_form():
    X, y = _toy_dataset()
    noise_var = jnp.array(0.05)
    kernel = RBF(init_variance=0.9, init_lengthscale=0.5)

    with handlers.seed(rng_seed=0):
        prior = GPPrior(kernel=kernel, X=X, jitter=1e-8)
        cond = prior.condition(y, noise_var)
        var = cond.predict_var(X)
        K = kernel(X, X)

    K_y = K + (1e-8 + 0.05) * jnp.eye(X.shape[0])
    K_diag = jnp.diag(K)
    ref = K_diag - jnp.sum(K * jnp.linalg.solve(K_y, K), axis=0)
    assert jnp.allclose(var, ref, atol=1e-4)


def test_predict_returns_mean_and_var_tuple():
    X, y = _toy_dataset()
    with handlers.seed(rng_seed=0):
        prior = GPPrior(kernel=RBF(), X=X)
        cond = prior.condition(y, jnp.array(0.05))
        mu, var = cond.predict(X)
    assert mu.shape == (X.shape[0],)
    assert var.shape == (X.shape[0],)


def test_sample_shape_and_diagonal_consistency():
    X, y = _toy_dataset()
    with handlers.seed(rng_seed=0):
        prior = GPPrior(kernel=RBF(), X=X)
        cond = prior.condition(y, jnp.array(0.05))
        samples = cond.sample(jr.PRNGKey(1), X, n_samples=64)
    assert samples.shape == (64, X.shape[0])
    # Empirical mean / var at enough samples should track predictive ones.
    with handlers.seed(rng_seed=0):
        mu = cond.predict_mean(X)
    emp_mu = jnp.mean(samples, axis=0)
    assert jnp.allclose(emp_mu, mu, atol=0.3)  # generous tolerance for 64 draws


# --- Solver pluggability ---------------------------------------------------


def test_gpprior_accepts_cg_solver():
    X, y = _toy_dataset()
    with handlers.seed(rng_seed=0):
        prior = GPPrior(kernel=RBF(), X=X, solver=CGSolver())
        cond = prior.condition(y, jnp.array(0.05))
        mu_cg = cond.predict_mean(X)

    with handlers.seed(rng_seed=0):
        prior_dense = GPPrior(kernel=RBF(), X=X)
        mu_dense = prior_dense.condition(y, jnp.array(0.05)).predict_mean(X)
    assert jnp.allclose(mu_cg, mu_dense, atol=1e-3)


def test_gpprior_accepts_composed_solver():
    X, y = _toy_dataset()
    composed = ComposedSolver(
        solve_strategy=DenseSolver(), logdet_strategy=DenseLogdet()
    )
    with handlers.seed(rng_seed=0):
        prior = GPPrior(kernel=RBF(), X=X, solver=composed)
        logp = prior.log_prob(y)
    assert jnp.isfinite(logp)


# --- gp_factor inside a NumPyro model --------------------------------------


def test_gp_factor_registers_in_trace():
    X, y = _toy_dataset()

    def model():
        kernel = RBF()
        prior = GPPrior(kernel=kernel, X=X)
        gp_factor("obs", prior, y, jnp.array(0.05))

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model()
    assert "obs" in tr
    assert tr["obs"]["type"] == "sample"  # numpyro.factor registers as sample
    assert jnp.isfinite(tr["obs"]["fn"].log_prob(jnp.array(0.0)))


def test_gp_factor_svi_step_runs():
    X, y = _toy_dataset()

    def model():
        # Kernel with priors on its hyperparameters — exercises the
        # Parameterized pipeline together with gp_factor.
        kernel = RBF()
        kernel.set_prior("variance", dist.LogNormal(0.0, 1.0))
        kernel.set_prior("lengthscale", dist.LogNormal(0.0, 1.0))
        prior = GPPrior(kernel=kernel, X=X)
        gp_factor("obs", prior, y, jnp.array(0.05))

    guide = AutoNormal(model)
    svi = SVI(model, guide, Adam(1e-2), Trace_ELBO())
    state = svi.init(jr.PRNGKey(0))
    for _ in range(3):
        state, loss = svi.update(state)
    assert jnp.isfinite(loss)


def test_gp_factor_mcmc_round_trip():
    X, y = _toy_dataset(n=5)

    def model():
        kernel = RBF()
        kernel.set_prior("variance", dist.LogNormal(0.0, 1.0))
        prior = GPPrior(kernel=kernel, X=X)
        gp_factor("obs", prior, y, jnp.array(0.05))

    mcmc = MCMC(
        NUTS(model),
        num_warmup=5,
        num_samples=5,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jr.PRNGKey(0))
    samples = mcmc.get_samples()
    assert "RBF.variance" in samples
    assert samples["RBF.variance"].shape == (5,)


# --- gp_sample (latent-function workflow) ----------------------------------


def test_gp_sample_registers_mvn_site():
    X, _ = _toy_dataset()

    def model():
        prior = GPPrior(kernel=RBF(), X=X)
        return gp_sample("f", prior)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        f = model()
    assert f.shape == (X.shape[0],)
    assert tr["f"]["type"] == "sample"


def test_gp_sample_delegates_to_guide_when_provided():
    X, _ = _toy_dataset()

    class _SentinelGuide:
        def sample(self, name, prior):
            return jnp.full(prior.X.shape[0], 42.0)

    def model():
        prior = GPPrior(kernel=RBF(), X=X)
        return gp_sample("f", prior, guide=_SentinelGuide())

    with handlers.seed(rng_seed=0):
        f = model()
    assert jnp.allclose(f, 42.0)


# --- jit composability -----------------------------------------------------


def test_gpprior_log_prob_jits():
    """Jit via the NumPyro closure pattern — the ``GPPrior`` / ``Kernel``
    is captured by reference in the model function, not passed through
    the pytree machinery. This is the same pattern the ``_core`` MCMC
    integration tests use for ``Parameterized`` kernels."""
    X, y = _toy_dataset()
    prior = GPPrior(kernel=RBF(), X=X)

    def model(y):
        return prior.log_prob(y)

    seeded = handlers.seed(model, rng_seed=0)
    v_eager = seeded(y)
    v_jit = jax.jit(handlers.seed(model, rng_seed=0))(y)
    assert jnp.isfinite(v_eager)
    assert jnp.allclose(v_eager, v_jit)


def test_condition_predict_jits():
    X, y = _toy_dataset()
    prior = GPPrior(kernel=RBF(), X=X)

    def model(X_star, y):
        return prior.condition(y, jnp.array(0.05)).predict_mean(X_star)

    seeded = handlers.seed(model, rng_seed=0)
    mu_eager = seeded(X, y)
    mu_jit = jax.jit(handlers.seed(model, rng_seed=0))(X, y)
    assert jnp.allclose(mu_eager, mu_jit)


def test_conditioned_gp_is_eqx_module():
    X, y = _toy_dataset()
    with handlers.seed(rng_seed=0):
        prior = GPPrior(kernel=RBF(), X=X)
        cond = prior.condition(y, jnp.array(0.05))
    # ConditionedGP is a PyTree — flatten/unflatten round-trips.
    leaves, treedef = jax.tree_util.tree_flatten(cond)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, ConditionedGP)


# --- error surfaces --------------------------------------------------------


def test_condition_shape_mismatch_is_caught_at_call_time():
    X, _ = _toy_dataset(n=5)
    y_wrong = jnp.zeros(3)
    prior = GPPrior(kernel=RBF(), X=X)
    with (
        pytest.raises(Exception),  # noqa: B017 — upstream error type varies
        handlers.seed(rng_seed=0),
    ):
        prior.condition(y_wrong, jnp.array(0.05))
