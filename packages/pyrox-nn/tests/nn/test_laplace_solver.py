"""The ``solver`` of `last_layer_laplace` (pyrox#561).

The problem is the docstring example: a sine MLP's last layer, 16 weights and
a bias, on 50 points. The suite runs with x64 on (root ``conftest.py``).
"""

from __future__ import annotations

import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from pyrox_nn import last_layer_laplace


MODEL = eqx.nn.MLP(1, 1, 16, 2, activation=jnp.sin, key=jr.key(0))
X = jnp.linspace(-1.0, 1.0, 50)[:, None]
Y = jnp.sin(3.0 * X[:, 0])
X_STAR = jnp.linspace(-1.5, 1.5, 7)[:, None]
LAST = f".layers[{len(MODEL.layers) - 1}]"


def _predict(model, inputs):
    return jax.vmap(model)(inputs)[:, 0]


def _is_last(path, _leaf):
    return jax.tree_util.keystr(path).startswith(LAST)


def _posterior(**kwargs):
    return last_layer_laplace(
        _predict, MODEL, X, Y, sigma=0.1, param_filter=_is_last, **kwargs
    )


class _SolverCalled(Exception):
    pass


class _RaisingSolver(gaussx.DenseSolver):
    def solve(self, operator, vector):
        raise _SolverCalled

    def logdet(self, operator, *, key=None):
        raise _SolverCalled


def test_solver_matches_dense():
    # CG at rtol = atol = 1e-10 against the dense solve, at λ = 0.5 and the
    # tuned λ; the evidence through CG with an exact log-determinant.
    dense = _posterior().optimize_prior_precision()
    cg = gaussx.CGSolver(rtol=1e-10, atol=1e-10)
    iterative = _posterior(solver=cg).optimize_prior_precision()
    np.testing.assert_allclose(iterative.prior_precision, dense.prior_precision)
    np.testing.assert_allclose(iterative.mode(0.5), dense.mode(0.5), rtol=1e-5)
    np.testing.assert_allclose(iterative.w_mean, dense.w_mean, rtol=1e-5)
    for got, expected in zip(
        iterative.predict(X_STAR), dense.predict(X_STAR), strict=True
    ):
        np.testing.assert_allclose(got, expected, rtol=1e-5)
    exact_logdet = gaussx.ComposedSolver(
        solve_strategy=cg, logdet_strategy=gaussx.DenseLogdet()
    )
    composed = eqx.tree_at(lambda p: p.solver, iterative, exact_logdet)
    np.testing.assert_allclose(
        composed.log_marginal_likelihood(), dense.log_marginal_likelihood(), rtol=1e-5
    )


def test_solver_is_jittable():
    posterior = _posterior(solver=gaussx.CGSolver(rtol=1e-10, atol=1e-10))
    mean, var = eqx.filter_jit(lambda p: p.predict(X_STAR))(posterior)
    eager_mean, eager_var = posterior.predict(X_STAR)
    # jit reorders float64 sums: 3e-10 relative measured
    np.testing.assert_allclose(mean, eager_mean, rtol=1e-8)
    np.testing.assert_allclose(var, eager_var, rtol=1e-8)
    evidence = eqx.filter_jit(lambda p: p.log_marginal_likelihood())(posterior)
    assert np.isfinite(float(evidence))


def test_solver_is_used():
    posterior = _posterior(solver=_RaisingSolver())
    with pytest.raises(_SolverCalled):
        posterior.mode()
    with pytest.raises(_SolverCalled):
        posterior.predict(X_STAR)
    with pytest.raises(_SolverCalled):
        posterior.log_marginal_likelihood()
    # "diag" has a diagonal precision, so it ignores the solver
    diagonal = _posterior(structure="diag", solver=_RaisingSolver())
    diagonal.predict(X_STAR)
    diagonal.log_marginal_likelihood()


# mode(0.5)[0], mean[0], var[0] at X_STAR, the evidence and λ after
# optimize_prior_precision, recorded at edbbd6d (before pyrox#561)
_DEFAULT_REFERENCE = {
    "full": [
        -7.496371233212321,
        2.9149388475365554,
        0.3885579609105094,
        38.55019607356791,
        0.0010356067377281424,
    ],
    "diag": [
        -1.5016627880710705,
        -9.00451439133669,
        0.007745995885752849,
        7146.30634204042,
        0.09352934173111971,
    ],
}


@pytest.mark.parametrize("structure", ["full", "diag"])
def test_default_unchanged(structure):
    posterior = _posterior(structure=structure)
    tuned = posterior.optimize_prior_precision()
    mean, var = tuned.predict(X_STAR)
    got = [
        posterior.mode(0.5)[0],
        mean[0],
        var[0],
        tuned.log_marginal_likelihood(),
        tuned.prior_precision,
    ]
    # Frozen on one machine. The tuned precision comes out of an optimiser
    # loop, which amplifies BLAS differences: a CI runner differed by 2.3e-7
    # relative, so 1e-10 failed there.
    np.testing.assert_allclose(
        np.asarray(got), _DEFAULT_REFERENCE[structure], rtol=1e-6
    )
