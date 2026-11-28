"""The pathwise samplers solve with the prior's solver (pyrox#558).

These run in the fast tier, unlike the rest of the pathwise tests
(``test_pathwise.py`` is marked slow as a whole): 60 points and 64 features.
"""

from __future__ import annotations

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from pyrox_gp import (
    RBF,
    DecoupledPathwiseSampler,
    FullRankGuide,
    GPPrior,
    PathwiseSampler,
    SparseGPPrior,
    preconditioned_cg_solver,
)
from pyrox_gp._kernel_operator import KernelOperator
from pyrox_gp._preconditioned import _NoiseSplitPreconditioner


jax.config.update("jax_enable_x64", True)

X = jr.uniform(jr.key(0), (60, 2))
Y = jnp.sin(6.0 * X[:, 0])


class _SolverCalled(Exception):
    pass


class _RaisingSolver(gx.DenseSolver):
    def solve(self, operator, vector):
        raise _SolverCalled


def _matrix_free_posterior():
    solver = preconditioned_cg_solver(rank=20, key=jr.key(1))
    prior = GPPrior(RBF(), X, solver=solver, matrix_free=True)
    return prior.condition(Y, jnp.asarray(0.1))


def test_matrix_free_paths_never_densify(monkeypatch):
    # The issue's reproducer: sample_paths called as_matrix once before
    # pyrox#558, for the N x N kernel matrix.
    count = [0]
    original = KernelOperator.as_matrix

    def counted(self):
        count[0] += 1
        return original(self)

    monkeypatch.setattr(KernelOperator, "as_matrix", counted)
    posterior = _matrix_free_posterior()
    posterior.predict(X[:5])
    assert count[0] == 0
    PathwiseSampler(posterior, n_features=64).sample_paths(jr.key(2), n_paths=3)
    assert count[0] == 0


def test_matrix_free_paths_match_dense():
    # Same key, so the same RFF prior path and noise draw; the paths differ
    # only by the CG residual of the correction solve (rtol = atol = 1e-6).
    dense = GPPrior(RBF(), X).condition(Y, jnp.asarray(0.1))
    X_star = jr.uniform(jr.key(4), (7, 2))
    paths = {}
    for name, posterior in (("dense", dense), ("free", _matrix_free_posterior())):
        sampler = PathwiseSampler(posterior, n_features=64)
        paths[name] = sampler.sample_paths(jr.key(2), n_paths=3)(X_star)
    np.testing.assert_allclose(paths["free"], paths["dense"], rtol=1e-4, atol=1e-6)


def test_preconditioner_is_built_once_for_many_paths(monkeypatch):
    # The correction solves are vmapped over paths, and the preconditioner
    # depends on the operator only, so it is traced once for the batch.
    count = [0]
    original = _NoiseSplitPreconditioner.as_operator

    def counted(self, operator=None):
        count[0] += 1
        return original(self, operator)

    posterior = _matrix_free_posterior()
    monkeypatch.setattr(_NoiseSplitPreconditioner, "as_operator", counted)
    PathwiseSampler(posterior, n_features=64).sample_paths(jr.key(2), n_paths=5)
    assert count[0] == 1


def test_paths_use_the_prior_solver():
    posterior = GPPrior(RBF(), X).condition(Y, jnp.asarray(0.1))
    raising = eqx.tree_at(
        lambda p: p.prior.solver,
        posterior,
        _RaisingSolver(),
        is_leaf=lambda node: node is None,
    )
    with pytest.raises(_SolverCalled):
        PathwiseSampler(raising, n_features=64).sample_paths(jr.key(2), n_paths=3)
    sparse = SparseGPPrior(kernel=RBF(), Z=X[:8], solver=_RaisingSolver())
    sampler = DecoupledPathwiseSampler(sparse, FullRankGuide.init(8), n_features=64)
    with pytest.raises(_SolverCalled):
        sampler.sample_paths(jr.key(3), n_paths=2)


# Draws at X[:4], recorded at edbbd6d (before pyrox#558) in float64
_EXACT_REFERENCE = [
    [0.0584198942790497, -1.029020307803007, -0.15429343559287934, -0.8602141878884442],
    [
        0.15765486983561638,
        -0.8199085374073953,
        -0.11974761601561029,
        -0.7114032461406776,
    ],
    [
        0.15510494850219914,
        -0.9024186921970139,
        -0.07951305274957621,
        -0.7872613482298458,
    ],
]
_DECOUPLED_REFERENCE = [
    [
        -0.9102430899534548,
        0.025806027374162932,
        -0.6116491687822629,
        1.6013777072886413,
    ],
    [-2.526138475642367, 0.17050778620181162, 0.3404524800049813, -0.402778601480579],
]


def test_dense_paths_unchanged():
    posterior = GPPrior(RBF(), X).condition(Y, jnp.asarray(0.1))
    paths = PathwiseSampler(posterior, n_features=64).sample_paths(jr.key(2), n_paths=3)
    np.testing.assert_allclose(paths(X[:4]), _EXACT_REFERENCE, rtol=1e-12)
    sparse = SparseGPPrior(kernel=RBF(), Z=X[:8])
    sampler = DecoupledPathwiseSampler(sparse, FullRankGuide.init(8), n_features=64)
    decoupled = sampler.sample_paths(jr.key(3), n_paths=2)
    np.testing.assert_allclose(decoupled(X[:4]), _DECOUPLED_REFERENCE, rtol=1e-12)
