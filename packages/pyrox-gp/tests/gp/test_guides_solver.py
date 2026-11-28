"""The sparse guides' ``solver`` reaches ``predict`` and ``kl_divergence`` (#560).

Before #560, `FullRankGuide`, `MeanFieldGuide` and `DeltaGuide` factorised
the inducing covariance by Cholesky in ``predict`` whatever ``solver`` was,
and `FullRankGuide` and `MeanFieldGuide` did the same in ``kl_divergence``;
`WhitenedGuide` has nothing to choose there (its ``u = L_ZZ v`` needs the
factor, and its KL is against N(0, I)).
"""

from __future__ import annotations

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest
from pyrox_gp import (
    RBF,
    DeltaGuide,
    FullRankGuide,
    GaussianLikelihood,
    MeanFieldGuide,
    SparseGPPrior,
    WhitenedGuide,
    svgp_elbo,
)
from pyrox_gp._guides import _svgp_predict_unwhitened


class Raising(gx.AbstractSolverStrategy):
    """A solver that fails on any use: the issue's reproducer."""

    def solve(self, operator, vector):
        raise RuntimeError("solver used")

    def logdet(self, operator, *, key=None):
        raise RuntimeError("solver used")


_UNWHITENED = (FullRankGuide, MeanFieldGuide, DeltaGuide)


def _blocks(n: int = 60, m: int = 8, seed: int = 0):
    X = jr.uniform(jr.key(seed), (n, 2))
    prior = SparseGPPrior(RBF(), Z=X[:m])
    return X, prior, prior.predictive_blocks(X)


def _random_guide(cls, m: int, key):
    mean_key, scale_key = jr.split(key)
    mean = jr.normal(mean_key, (m,))
    if cls is DeltaGuide:
        return DeltaGuide(loc=mean)
    if cls is MeanFieldGuide:
        return MeanFieldGuide(mean=mean, scale=0.1 + jr.uniform(scale_key, (m,)))
    tril = jnp.tril(0.3 * jr.normal(scale_key, (m, m))) + jnp.eye(m)
    return cls(mean=mean, scale_tril=tril)


def test_predict_and_kl_use_the_solver() -> None:
    """The issue's reproducer: a raising solver now raises where it applies."""
    _, _, (K_zz_op, K_xz, K_xx_diag) = _blocks()
    for cls in _UNWHITENED:
        guide = cls.init(8, solver=Raising())
        with pytest.raises(RuntimeError, match="solver used"):
            guide.predict(K_xz, K_zz_op, K_xx_diag)
        with pytest.raises(RuntimeError, match="solver used"):
            guide.kl_divergence(K_zz_op)
    # WhitenedGuide: u = L_ZZ v needs the Cholesky factor, and its KL is
    # against N(0, I), so the solver does not apply to either (documented)
    whitened = WhitenedGuide.init(8, solver=Raising())
    whitened.predict(K_xz, K_zz_op, K_xx_diag)
    whitened.kl_divergence(K_zz_op)


@pytest.mark.parametrize("cls", _UNWHITENED)
def test_default_unchanged(cls) -> None:
    """solver=None keeps the Cholesky route, bit for bit."""
    _, _, (K_zz_op, K_xz, K_xx_diag) = _blocks()
    guide = _random_guide(cls, 8, jr.key(1))
    mean, variance = guide.predict(K_xz, K_zz_op, K_xx_diag)
    if cls is DeltaGuide:
        m_size = guide.loc.shape[0]
        L_zz = gx.cholesky(K_zz_op)
        white = jnp.linalg.solve(L_zz.as_matrix(), guide.loc)
        want = gx.whitened_svgp_predict(
            K_zz_op, K_xz, white, jnp.zeros((m_size, m_size)), K_xx_diag
        )
        # the same factor, solved by lineax in the guide and jnp here
        np.testing.assert_allclose(mean, want[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(variance, want[1])
    else:
        scale = guide.scale_tril if cls is FullRankGuide else jnp.diag(guide.scale)
        want = _svgp_predict_unwhitened(
            K_xz, K_zz_op, K_xx_diag, guide.mean, scale @ scale.T
        )
        np.testing.assert_array_equal(mean, want[0])
        np.testing.assert_array_equal(variance, want[1])


@pytest.mark.parametrize("cls", _UNWHITENED)
def test_dense_solver_form_matches_the_cholesky_route(cls) -> None:
    """The solver form with DenseSolver equals the default to round-off."""
    _, _, (K_zz_op, K_xz, K_xx_diag) = _blocks()
    default = _random_guide(cls, 8, jr.key(2))
    explicit = eqx.tree_at(
        lambda g: g.solver, default, gx.DenseSolver(), is_leaf=lambda x: x is None
    )
    for got, want in zip(
        explicit.predict(K_xz, K_zz_op, K_xx_diag),
        default.predict(K_xz, K_zz_op, K_xx_diag),
        strict=True,
    ):
        # two exact factorisations of an RBF K_ZZ with jitter 1e-6
        np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        explicit.kl_divergence(K_zz_op), default.kl_divergence(K_zz_op), rtol=1e-8
    )


def _elbo_problem(dtype=jnp.float64):
    X = jr.uniform(jr.key(0), (200, 1), minval=-3.0, maxval=3.0, dtype=dtype)
    y = jnp.sin(2.0 * X[:, 0]) + 0.1 * jr.normal(jr.key(1), (200,), dtype=dtype)
    Z = jnp.linspace(-3.0, 3.0, 30, dtype=dtype)[:, None]
    prior = SparseGPPrior(RBF(init_lengthscale=0.5), Z=Z, jitter=1e-4)
    return prior, X, y


def _solvers():
    """CG, preconditioned CG and BBMM, each with an exact log-determinant.

    Each solver's own log-determinant is stochastic Lanczos quadrature; on
    this K_ZZ (M = 30) it is 9.3 nats off with CG's 20 probes, which moves
    the KL by half that, about 3e-4 of the ELBO. The exact log-determinant
    leaves only the solves, which are what pyrox#560 routes.
    """
    exact = gx.DenseLogdet()
    cg = gx.CGSolver(rtol=1e-10, atol=1e-10, max_steps=2000)
    return {
        "cg": gx.compose_solver(cg, logdet=exact),
        "pcg": gx.compose_solver(
            cg,
            preconditioner=gx.PartialCholeskyPreconditioner(rank=10, shift=1e-4),
            logdet=exact,
        ),
        "bbmm": gx.compose_solver(
            gx.BBMMSolver(rtol=1e-10, atol=1e-10, max_steps=2000), logdet=exact
        ),
    }


def _small_guide(cls, key):
    # a guide near the prior's scale, so the ELBO is O(10^3) nats
    return jax.tree.map(lambda leaf: 0.3 * leaf, _random_guide(cls, 30, key))


@pytest.mark.parametrize("cls", _UNWHITENED)
@pytest.mark.parametrize("name", ["cg", "pcg", "bbmm"])
def test_solvers_agree_on_elbo(cls, name) -> None:
    """200 points, 30 inducing: every solver gives the dense ELBO.

    The issue's bar is rtol 1e-4; with an exact log-determinant beside the
    solver the gap is the CG residual at 1e-10 against a K_ZZ of condition
    number 6e4, measured below 2e-10 relative, so 1e-8 is asserted.
    """
    prior, X, y = _elbo_problem()
    likelihood = GaussianLikelihood(0.1)
    dense = _small_guide(cls, jr.key(4))
    guide = eqx.tree_at(
        lambda g: g.solver, dense, _solvers()[name], is_leaf=lambda x: x is None
    )
    want = svgp_elbo(prior, dense, likelihood, X, y)
    got = svgp_elbo(prior, guide, likelihood, X, y)
    np.testing.assert_allclose(got, want, rtol=1e-8)


def test_float32_variance_is_never_negative() -> None:
    """The contraction form k_nn - k_n^T b_n + ... is clipped at zero."""
    prior, X, _ = _elbo_problem(jnp.float32)
    K_zz_op, K_xz, K_xx_diag = prior.predictive_blocks(X)
    solver = gx.CGSolver(rtol=1e-6, atol=1e-6, max_steps=2000)
    for cls in _UNWHITENED:
        guide = _small_guide(cls, jr.key(5))
        guide = jax.tree.map(
            lambda leaf: (
                leaf.astype(jnp.float32) if eqx.is_inexact_array(leaf) else leaf
            ),
            guide,
        )
        guide = eqx.tree_at(
            lambda g: g.solver, guide, solver, is_leaf=lambda x: x is None
        )
        mean, variance = guide.predict(K_xz, K_zz_op, K_xx_diag)
        assert mean.dtype == variance.dtype == jnp.float32
        assert bool(jnp.all(variance >= 0.0))


@pytest.mark.parametrize("name", ["cg", "pcg", "bbmm"])
def test_svgp_elbo_trains_without_nan(name) -> None:
    """Twenty Adam steps on the ELBO of a full-rank guide, for each solver."""
    prior, X, y = _elbo_problem()
    likelihood = GaussianLikelihood(0.1)
    guide = FullRankGuide.init(30, scale=0.3, solver=_solvers()[name])
    optimiser = optax.adam(1e-2)
    params, static = eqx.partition(guide, eqx.is_inexact_array)
    state = optimiser.init(params)

    @jax.jit
    def step(params, state):
        def loss(params):
            return -svgp_elbo(prior, eqx.combine(params, static), likelihood, X, y)

        value, grads = jax.value_and_grad(loss)(params)
        updates, state = optimiser.update(grads, state, params)
        return optax.apply_updates(params, updates), state, value

    values = []
    for _ in range(20):
        params, state, value = step(params, state)
        values.append(value)
    values = jnp.stack(values)
    assert bool(jnp.all(jnp.isfinite(values)))
    assert float(values[-1]) < float(values[0])
