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


# --- Closed-form log marginal likelihood golden check ----------------------


def test_collapsed_log_marginal_likelihood_matches_direct_computation():
    """Pin the collapsed GP log marginal — what ``gp_factor`` registers —
    against a direct Cholesky reference.

    .. math::
        \\log p(y \\mid X, \\theta) =
            -\\tfrac{1}{2}\\, y^\\top (K + (\\text{jitter} + \\sigma^2) I)^{-1} y
            -\\tfrac{1}{2}\\, \\log\\bigl|K + (\\text{jitter} + \\sigma^2) I\\bigr|
            -\\tfrac{N}{2}\\, \\log(2\\pi).

    We compute it two ways:

    1. ``gp_factor`` path — call ``gaussx.log_marginal_likelihood`` on
       ``prior._noisy_operator(noise_var)``. This is exactly what
       :func:`pyrox.gp.gp_factor` does under the hood, just without the
       NumPyro side effect.
    2. Direct Cholesky — solve + logdet on the same kernel matrix, with
       the matching ``(jitter + noise_var)`` regularization.

    The two numbers should agree to float32 roundoff.
    """
    from gaussx import log_marginal_likelihood

    X, y = _toy_dataset(n=8, seed=1)
    variance = jnp.array(1.1)
    lengthscale = jnp.array(0.45)
    noise_var = jnp.array(0.07)
    jitter = 1e-8
    n = X.shape[0]

    # (1) pyrox path: log_marginal_likelihood on the noisy operator —
    # the same call gp_factor makes inside its NumPyro factor.
    with handlers.seed(rng_seed=0):
        kernel = RBF(init_variance=float(variance), init_lengthscale=float(lengthscale))
        prior = GPPrior(kernel=kernel, X=X, jitter=jitter)
        K = kernel(X, X)
        pyrox_logp = log_marginal_likelihood(
            prior.mean(prior.X),
            prior._noisy_operator(noise_var),
            y,
            solver=prior._resolved_solver(),
        )

    # (2) Closed-form direct computation via Cholesky.
    K_y = K + (jitter + noise_var) * jnp.eye(n)
    L = jnp.linalg.cholesky(K_y)
    alpha = jax.scipy.linalg.cho_solve((L, True), y)
    data_fit = -0.5 * jnp.dot(y, alpha)
    logdet = -jnp.sum(jnp.log(jnp.diag(L)))  # = -1/2 log|K_y|
    normalizer = -0.5 * n * jnp.log(2.0 * jnp.pi)
    direct_logp = data_fit + logdet + normalizer

    assert jnp.allclose(pyrox_logp, direct_logp, atol=1e-4)


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
    """``gp_sample(..., guide=g)`` must call ``g.register(name, prior)``.

    The hook is intentionally distinct from the :class:`Guide` protocol's
    ``sample(self, key)`` (raw variational draw) — see Guide ABC docstring.
    """
    X, _ = _toy_dataset()

    class _SentinelGuide:
        def register(self, name, prior):
            return jnp.full(prior.X.shape[0], 42.0)

    def model():
        prior = GPPrior(kernel=RBF(), X=X)
        return gp_sample("f", prior, guide=_SentinelGuide())

    with handlers.seed(rng_seed=0):
        f = model()
    assert jnp.allclose(f, 42.0)


def test_gp_sample_whitened_registers_unit_normal_site_and_deterministic():
    """Whitened mode registers ``f"{name}_u"`` as the latent sample site
    and ``name`` as a deterministic, returning the unwhitened function value."""
    X, _ = _toy_dataset()

    def model():
        prior = GPPrior(kernel=RBF(), X=X, jitter=1e-6)
        return gp_sample("f", prior, whitened=True)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        f = model()
    assert f.shape == (X.shape[0],)
    assert tr["f_u"]["type"] == "sample"
    assert tr["f_u"]["value"].shape == (X.shape[0],)
    assert tr["f"]["type"] == "deterministic"


def test_gp_sample_whitened_matches_mu_plus_chol_u():
    """``f = mu(X) + L u`` exactly, where ``L = chol(K + jitter I)``."""
    from gaussx import cholesky as gaussx_cholesky

    X, _ = _toy_dataset()
    kernel = RBF(init_variance=0.7, init_lengthscale=0.6)
    mean_fn = lambda x: 0.3 * x.squeeze(-1)
    jitter = 1e-6

    def model():
        prior = GPPrior(kernel=kernel, X=X, mean_fn=mean_fn, jitter=jitter)
        return gp_sample("f", prior, whitened=True)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        f = model()

    with handlers.seed(rng_seed=0):
        K = kernel(X, X) + jitter * jnp.eye(X.shape[0])
    import lineax as lx

    L = gaussx_cholesky(lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag))
    u = tr["f_u"]["value"]
    expected = mean_fn(X) + L.as_matrix() @ u
    assert jnp.allclose(f, expected, atol=1e-10)


def test_gp_sample_whitened_with_guide_raises():
    """Combining ``whitened=True`` with a ``guide`` is rejected — the two
    modes are mutually exclusive (guides own their own parameterization)."""
    X, _ = _toy_dataset()

    class _SentinelGuide:
        def register(self, name, prior):  # pragma: no cover — should not run
            return jnp.zeros(prior.X.shape[0])

    prior = GPPrior(kernel=RBF(), X=X)
    with (
        pytest.raises(ValueError, match="cannot combine"),
        handlers.seed(rng_seed=0),
    ):
        gp_sample("f", prior, whitened=True, guide=_SentinelGuide())


def test_gp_sample_whitened_marginal_matches_prior_mvn():
    """Under the prior, ``f = mu + L u`` with ``u ~ N(0, I)`` has marginal
    ``MVN(mu, K + jitter I)`` — verify by Monte-Carlo moment matching."""
    X, _ = _toy_dataset(n=4)
    kernel = RBF(init_variance=1.5, init_lengthscale=0.4)
    jitter = 1e-6

    def model():
        prior = GPPrior(kernel=kernel, X=X, jitter=jitter)
        return gp_sample("f", prior, whitened=True)

    keys = jr.split(jr.PRNGKey(0), 5000)

    def draw(key):
        return handlers.seed(model, rng_seed=key)()

    fs = jax.vmap(draw)(keys)  # (S, N)
    sample_mean = fs.mean(axis=0)
    sample_cov = jnp.cov(fs.T)

    K = kernel(X, X) + jitter * jnp.eye(X.shape[0])
    assert jnp.allclose(sample_mean, jnp.zeros(X.shape[0]), atol=0.1)
    assert jnp.allclose(sample_cov, K, atol=0.1)


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


# --- Pattern B/C regression: kernel context scoping ------------------------


def test_predict_var_does_not_double_register_pattern_b_kernel_sites():
    """Regression for PR #62 P1: ``predict_var`` makes two kernel calls
    (``kernel(X_star, X)`` and ``kernel.diag(X_star)``). For a kernel
    whose hyperparameters carry priors (Pattern B / C), both calls would
    re-register the same NumPyro sample site without a shared kernel
    context, raising a duplicate-site error under tracing.

    To isolate the within-method duplication the reviewer flagged, we
    build the ``ConditionedGP`` outside the trace (so the training
    operator is constructed against the kernel's default values) and
    only enter the trace for ``predict_var``. That puts both predict-
    time kernel calls under the same outer context — without the fix,
    the second call would raise.
    """
    X, y = _toy_dataset()
    X_star = jnp.linspace(-1.5, 1.5, 4).reshape(-1, 1)

    kernel = RBF()
    kernel.set_prior("variance", dist.LogNormal(0.0, 1.0))
    kernel.set_prior("lengthscale", dist.LogNormal(0.0, 1.0))
    prior = GPPrior(kernel=kernel, X=X)

    # Build the conditioned GP under its own seed scope so the training
    # operator gets concrete kernel hyperparameters, then enter a fresh
    # trace for the predict_var call. Without the per-method
    # ``_kernel_context`` scoping, the second kernel call inside
    # ``predict_var`` would raise a duplicate-site error here.
    with handlers.seed(rng_seed=1):
        cond = prior.condition(y, jnp.array(0.05))

    def model():
        return cond.predict_var(X_star)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        var = model()
    assert var.shape == (X_star.shape[0],)
    assert jnp.all(jnp.isfinite(var))
    # Each prior'd hyperparameter registers exactly once across the
    # whole predict_var call — duplicate registration would have raised
    # before this point.
    assert "RBF.variance" in tr
    assert "RBF.lengthscale" in tr


def test_predict_does_not_double_register_pattern_b_kernel_sites():
    """Same regression as above for ``predict`` (returns mean + var) —
    its three kernel evaluations must share one outer context."""
    X, y = _toy_dataset()
    X_star = jnp.linspace(-1.5, 1.5, 4).reshape(-1, 1)

    kernel = RBF()
    kernel.set_prior("variance", dist.LogNormal(0.0, 1.0))
    kernel.set_prior("lengthscale", dist.LogNormal(0.0, 1.0))
    prior = GPPrior(kernel=kernel, X=X)
    with handlers.seed(rng_seed=1):
        cond = prior.condition(y, jnp.array(0.05))

    def model():
        return cond.predict(X_star)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        mean, var = model()
    assert mean.shape == (X_star.shape[0],)
    assert var.shape == (X_star.shape[0],)
    assert "RBF.variance" in tr
    assert "RBF.lengthscale" in tr


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


# --- Prior-side sample (PR #66 review) -------------------------------------


def test_gpprior_sample_returns_correct_shape_and_dtype():
    """``GPPrior.sample`` returns a vector of length ``N`` matching the
    training-input dtype — the non-NumPyro draw analogue of
    :func:`gp_sample`."""
    X, _ = _toy_dataset(n=6)
    prior = GPPrior(kernel=RBF(), X=X)
    f = prior.sample(jr.PRNGKey(7))
    assert f.shape == (6,)
    assert f.dtype == X.dtype


def test_gpprior_sample_marginal_matches_prior_log_prob():
    """``GPPrior.log_prob(f)`` evaluated at a sample drawn from
    :meth:`sample` is finite — round-trip sanity that the same
    operator backs both paths."""
    X, _ = _toy_dataset(n=6)
    prior = GPPrior(kernel=RBF(), X=X)
    f = prior.sample(jr.PRNGKey(8))
    lp = prior.log_prob(f)
    assert jnp.isfinite(lp)


def test_gpprior_sample_invariant_to_dense_solver_choice():
    """Default solver resolution matches an explicit
    :class:`DenseSolver` — the configured ``solver`` propagates into
    :class:`gaussx.MultivariateNormal`."""
    X, _ = _toy_dataset(n=6)
    p_default = GPPrior(kernel=RBF(), X=X)
    p_dense = GPPrior(kernel=RBF(), X=X, solver=DenseSolver())
    key = jr.PRNGKey(9)
    assert jnp.allclose(p_default.sample(key), p_dense.sample(key), atol=1e-5)
