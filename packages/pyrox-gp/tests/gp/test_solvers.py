"""The matfree_solver and bbmm_solver presets (pyrox#557)."""

from __future__ import annotations

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest
from pyrox_gp import (
    GPPrior,
    Matern,
    bbmm_solver,
    matfree_solver,
    preconditioned_cg_solver,
)
from pyrox_gp._kernel_operator import KernelOperator


jax.config.update("jax_enable_x64", True)

N = 400
X = jr.uniform(jr.key(0), (N, 2))
Y = jnp.sin(6 * X[:, 0]) * jnp.cos(4 * X[:, 1]) + 0.2 * jr.normal(jr.key(1), (N,))
X_STAR = jr.uniform(jr.key(5), (30, 2))
NOISE_VAR = 0.04
# a frozen kernellib kernel: no NumPyro sites, so no seed handler is needed
KERNEL = Matern(nu=1.5, init_lengthscale=0.3).frozen()
PRESETS = {"matfree": matfree_solver, "bbmm": bbmm_solver}


def _dense():
    prior = GPPrior(KERNEL, X)
    mean, variance = prior.condition(Y, NOISE_VAR).predict(X_STAR)
    return prior.log_marginal(Y, NOISE_VAR), mean, variance


def _no_kernel_matrix(monkeypatch):
    # any materialisation of the N x N kernel matrix fails the test
    def forbidden(self):
        raise AssertionError("the kernel matrix was formed")

    monkeypatch.setattr(KernelOperator, "as_matrix", forbidden)


@pytest.mark.parametrize("name", list(PRESETS))
def test_preset_smoke(name, monkeypatch):
    # one key at rank 150 and rtol 1e-8: the predictions match dense at
    # 1e-4, and the log marginal within 1 nat (measured below 0.2)
    lml, mean, variance = _dense()
    solver = PRESETS[name](rank=150, rtol=1e-8, key=jr.key(0))
    _no_kernel_matrix(monkeypatch)
    prior = GPPrior(KERNEL, X, solver=solver, matrix_free=True)
    got_lml = prior.log_marginal(Y, NOISE_VAR)
    got_mean, got_variance = prior.condition(Y, NOISE_VAR).predict(X_STAR)
    assert abs(float(got_lml - lml)) < 1.0
    np.testing.assert_allclose(got_mean, mean, rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(got_variance, variance, rtol=1e-4, atol=1e-8)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_match_dense(name):
    # Gaussian GP at N = 400: over 20 keys the mean log marginal is within
    # 3 standard errors of the dense one, and the default rtol=1e-4 gives the
    # dense predictions at rtol=1e-4 (the issue's DoD).
    lml, mean, variance = _dense()

    def estimate(key):
        prior = GPPrior(KERNEL, X, solver=PRESETS[name](key=key), matrix_free=True)
        return prior.log_marginal(Y, NOISE_VAR)

    estimates = np.asarray(jax.vmap(estimate)(jr.split(jr.key(10), 20)))
    standard_error = estimates.std(ddof=1) / np.sqrt(len(estimates))
    assert abs(estimates.mean() - float(lml)) < 3 * standard_error + 1e-6

    prior = GPPrior(KERNEL, X, solver=PRESETS[name](key=jr.key(0)), matrix_free=True)
    got_mean, got_variance = prior.condition(Y, NOISE_VAR).predict(X_STAR)
    np.testing.assert_allclose(got_mean, mean, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(got_variance, variance, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_on_a_site_system(name, monkeypatch):
    # The system of a Laplace fit, K_op + diag(d) with d varying by point,
    # through gaussx.site_posterior: the preset matches the dense posterior
    # without forming K. (LaplaceInference itself still rejects a
    # matrix-free prior until pyrox#551.)
    site_precision = jnp.linspace(0.1, 0.25, 60)
    site_information = site_precision * (X[:60, 0] > 0.5)
    prior_mean = jnp.zeros(60)
    kernel_op = lx.TaggedLinearOperator(
        KernelOperator(KERNEL, X[:60]), lx.positive_semidefinite_tag
    )
    dense = gx.site_posterior(
        prior_mean,
        lx.MatrixLinearOperator(kernel_op.as_matrix(), lx.positive_semidefinite_tag),
        site_information,
        site_precision,
    )
    solver = PRESETS[name](rank=20, rtol=1e-10, key=jr.key(0))
    _no_kernel_matrix(monkeypatch)
    got = gx.site_posterior(
        prior_mean, kernel_op, site_information, site_precision, solver=solver
    )
    # CG at rtol 1e-10, so 1e-6 holds with margin
    np.testing.assert_allclose(got.mean, dense.mean, atol=1e-6)


@pytest.mark.parametrize("name", list(PRESETS))
def test_fitc_preset(name, monkeypatch):
    """preconditioner="fitc" builds gaussx's FITCPreconditioner (gaussx#824).

    Before gaussx v0.8.4 both presets raised for it. On a matrix-free
    Gaussian prior it predicts as the dense GP does, with the tolerance of
    `test_preset_smoke`, and on the per-point system of a site posterior it
    reaches the dense mean; the N x N kernel matrix is never formed.
    """
    lml, mean, variance = _dense()
    solver = PRESETS[name](preconditioner="fitc", rank=150, rtol=1e-8, key=jr.key(0))
    _no_kernel_matrix(monkeypatch)
    prior = GPPrior(KERNEL, X, solver=solver, matrix_free=True)
    got_lml = prior.log_marginal(Y, NOISE_VAR)
    got_mean, got_variance = prior.condition(Y, NOISE_VAR).predict(X_STAR)
    assert abs(float(got_lml - lml)) < 1.0
    np.testing.assert_allclose(got_mean, mean, rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(got_variance, variance, rtol=1e-4, atol=1e-8)

    site_precision = jnp.linspace(0.1, 0.25, 60)
    site_information = site_precision * (X[:60, 0] > 0.5)
    kernel_op = lx.TaggedLinearOperator(
        KernelOperator(KERNEL, X[:60]), lx.positive_semidefinite_tag
    )
    reference = KERNEL(X[:60], X[:60])
    dense = gx.site_posterior(
        jnp.zeros(60),
        lx.MatrixLinearOperator(reference, lx.positive_semidefinite_tag),
        site_information,
        site_precision,
    )
    site_solver = PRESETS[name](
        preconditioner="fitc", rank=20, rtol=1e-10, key=jr.key(0)
    )
    got = gx.site_posterior(
        jnp.zeros(60), kernel_op, site_information, site_precision, solver=site_solver
    )
    # CG at rtol 1e-10, so 1e-6 holds with margin
    np.testing.assert_allclose(got.mean, dense.mean, atol=1e-6)


def test_fitc_is_gaussx_fitc_at_drawn_training_inputs():
    """The preset's P is `FITCPreconditioner.from_kernel` at R drawn inputs.

    The same P whether the kernel part is a `KernelOperator` (kernel entries
    evaluated) or a dense matrix (entries read), and the diagonal is read
    from the system.
    """
    from pyrox_gp._preconditioned import _SystemFITCPreconditioner

    rank, n = 12, 60
    pre = _SystemFITCPreconditioner(rank, jr.key(3))
    shift = lx.DiagonalLinearOperator(jnp.linspace(0.05, 0.3, n))
    psd = lx.positive_semidefinite_tag
    matrix_free = lx.TaggedLinearOperator(KernelOperator(KERNEL, X[:n]) + shift, psd)
    dense = lx.TaggedLinearOperator(
        lx.MatrixLinearOperator(KERNEL(X[:n], X[:n])) + shift, psd
    )
    index = jr.choice(jr.key(3), n, (rank,), replace=False)
    expected = gx.FITCPreconditioner.from_kernel(KERNEL, X[:n], X[:n][index])
    want = expected.approximation(matrix_free).as_matrix()
    for system in (matrix_free, dense):
        # same blocks of the same K, so equal to round-off
        np.testing.assert_allclose(
            pre.approximation(system).as_matrix(), want, rtol=1e-10, atol=1e-12
        )
    np.testing.assert_allclose(pre.logdet(dense), expected.logdet(dense), rtol=1e-10)
    # a matrix-free prior is K_op + jitter I inside the site system: the
    # jitter stays on P's diagonal, so P still matches A there
    prior_cov = GPPrior(KERNEL, X[:n], matrix_free=True)._prior_operator()
    nested = lx.TaggedLinearOperator(prior_cov + shift, psd)
    jitter = GPPrior(KERNEL, X[:n]).jitter
    np.testing.assert_allclose(
        pre.approximation(nested).as_matrix(),
        want + jitter * jnp.eye(n),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jnp.diag(pre.approximation(nested).as_matrix()),
        jnp.diag(nested.as_matrix()),
        rtol=1e-10,
    )
    vector = Y[:n]
    np.testing.assert_allclose(
        pre(vector, dense), jnp.linalg.solve(want, vector), rtol=1e-8
    )


def test_fitc_rejects_what_it_cannot_build():
    from pyrox_gp._preconditioned import _SystemFITCPreconditioner

    one_matrix = GPPrior(KERNEL, X[:30])._noisy_operator(NOISE_VAR)
    with pytest.raises(ValueError, match=r"sum K \+ diag\(d\)"):
        _SystemFITCPreconditioner(10, jr.key(0)).as_operator(one_matrix)
    system = GPPrior(KERNEL, X[:30], matrix_free=True)._noisy_operator(NOISE_VAR)
    with pytest.raises(ValueError, match="rank 40 exceeds the 30"):
        _SystemFITCPreconditioner(40, jr.key(0)).as_operator(system)


def test_unknown_options_raise():
    with pytest.raises(ValueError, match="preconditioner must be"):
        bbmm_solver(
            preconditioner="jacobi",  # ty: ignore[invalid-argument-type]
            key=jr.key(0),
        )
    with pytest.raises(ValueError, match="logdet must be"):
        matfree_solver(
            logdet="exact",  # ty: ignore[invalid-argument-type]
            key=jr.key(0),
        )


def test_presets_are_ordinary_strategies():
    # compose_solver and KeyedSolver take them like any gaussx strategy
    matfree = matfree_solver(key=jr.key(0))
    bbmm = bbmm_solver(key=jr.key(0))
    assert isinstance(matfree, gx.ComposedSolver)
    assert isinstance(bbmm, gx.KeyedSolver)
    assert isinstance(bbmm.strategy, gx.BBMMSolver)
    assert bbmm.strategy.preconditioner is not None
    assert gx.compose_solver(matfree) is matfree
    keyed = gx.KeyedSolver(matfree, jr.key(1))
    assert isinstance(keyed, gx.AbstractSolverStrategy)


@pytest.mark.parametrize("logdet", ["slq", "nystrom"])
def test_deprecated_alias(logdet):
    # It warns, and builds the matfree_solver strategy with its old
    # defaults, so a scalar-noise system gets the same numbers.
    with pytest.warns(DeprecationWarning, match="matfree_solver"):
        alias = preconditioned_cg_solver(rank=50, logdet=logdet, key=jr.key(0))
    preset = matfree_solver(
        rank=50,
        logdet=logdet,
        num_probes=20,
        rtol=1e-6,
        atol=1e-6,
        key=jr.key(0),
    )
    assert eqx.tree_equal(alias, preset)
    operator = GPPrior(KERNEL, X[:100], matrix_free=True)._noisy_operator(NOISE_VAR)
    np.testing.assert_array_equal(
        alias.solve(operator, Y[:100]), preset.solve(operator, Y[:100])
    )
    np.testing.assert_array_equal(alias.logdet(operator), preset.logdet(operator))


def test_alias_shift_is_ignored_on_a_sum():
    # shift= has no place once the diagonal is read from the system: it is
    # accepted and ignored with a warning
    with pytest.warns(DeprecationWarning):
        alias = preconditioned_cg_solver(shift=0.01, rank=20, key=jr.key(0))
    operator = GPPrior(KERNEL, X[:60], matrix_free=True)._noisy_operator(NOISE_VAR)
    with pytest.warns(UserWarning, match="ignored"):
        solution = alias.solve(operator, Y[:60])
    expected = jnp.linalg.solve(operator.as_matrix(), Y[:60])
    np.testing.assert_allclose(solution, expected, rtol=1e-5)
