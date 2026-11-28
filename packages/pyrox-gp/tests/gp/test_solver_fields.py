"""The solver, preconditioner and logdet fields of the GP priors (pyrox#550)."""

from __future__ import annotations

import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.util import log_density
from numpyro.optim import Adam
from pyrox_gp import (
    RBF,
    GPPrior,
    ICMKernel,
    Matern,
    MultiOutputGPPrior,
    OILMMGPPrior,
    OILMMKernel,
    SparseGPPrior,
    gp_factor,
)
from pyrox_gp._kernel_operator import KernelOperator


jax.config.update("jax_enable_x64", True)

N = 200
X = jr.uniform(jr.key(0), (N, 2))
Y = jnp.sin(6 * X[:, 0]) * jnp.cos(4 * X[:, 1]) + 0.2 * jr.normal(jr.key(1), (N,))
X_STAR = jr.uniform(jr.key(5), (30, 2))
NOISE_VAR = 0.04
# a frozen kernellib kernel: no NumPyro sites, so no seed handler is needed
KERNEL = Matern(nu=1.5, init_lengthscale=0.3).frozen()
# tight CG tolerances, so the solves match dense to ~1e-10 (measured)
TIGHT = {"rtol": 1e-10, "atol": 1e-10}


def _nystrom():
    return gx.NystromPreconditioner.lazy(rank=100, key=jr.key(1))


def _partial_cholesky():
    return gx.PartialCholeskyPreconditioner(
        rank=100, shift=None, pivoting="random", key=jr.key(1)
    )


def _dense_reference():
    prior = GPPrior(KERNEL, X)
    mean, variance = prior.condition(Y, NOISE_VAR).predict(X_STAR)
    return prior.log_marginal(Y, NOISE_VAR), mean, variance


# (fields, log-marginal tolerance). An exact log-determinant matches dense
# to 1e-12 (measured); a stochastic one carries its own error, measured at
# N = 200 and 50 probes: 1.8 nats for SLQ, 0.7 for NystromLogdet, 0.25 for
# BBMM with a preconditioner and 0.6 with a bound key, so the bound is 3.
CONFIGURATIONS = {
    "cg+nystrom+dense": (
        {
            "solver": gx.CGSolver(**TIGHT),
            "preconditioner": _nystrom(),
            "logdet": gx.DenseLogdet(),
        },
        1e-8,
    ),
    "cg+rpcholesky+dense": (
        {
            "solver": gx.CGSolver(**TIGHT),
            "preconditioner": _partial_cholesky(),
            "logdet": gx.DenseLogdet(),
        },
        1e-8,
    ),
    "cg+nystrom+slq": (
        {
            "solver": gx.CGSolver(**TIGHT),
            "preconditioner": _nystrom(),
            "logdet": gx.SLQLogdet(num_probes=50),
        },
        3.0,
    ),
    "cg+nystrom-logdet": (
        {
            "solver": gx.CGSolver(**TIGHT),
            "logdet": gx.NystromLogdet(shift=None, rank=100),
        },
        3.0,
    ),
    "bbmm+nystrom": (
        {
            "solver": gx.BBMMSolver(num_probes=50, **TIGHT),
            "preconditioner": _nystrom(),
        },
        3.0,
    ),
    "bbmm+rpcholesky": (
        {
            "solver": gx.BBMMSolver(num_probes=50, **TIGHT),
            "preconditioner": _partial_cholesky(),
        },
        3.0,
    ),
    "keyed-bbmm+nystrom": (
        {
            "solver": gx.KeyedSolver(gx.BBMMSolver(num_probes=50, **TIGHT), jr.key(7)),
            "preconditioner": _nystrom(),
        },
        3.0,
    ),
}


def test_default_is_unchanged():
    # GPPrior(kernel, X) composes to DenseSolver, and a solver alone is
    # returned as it is, so priors that set none of the new fields keep
    # their exact numbers.
    prior = GPPrior(KERNEL, X)
    assert prior.solver_strategy() == gx.DenseSolver()
    explicit = GPPrior(KERNEL, X, solver=gx.DenseSolver())
    assert prior.log_marginal(Y, NOISE_VAR) == explicit.log_marginal(Y, NOISE_VAR)
    cg = gx.CGSolver()
    assert GPPrior(KERNEL, X, solver=cg).solver_strategy() is cg
    # the new fields are keyword only, so positional jitter still works
    assert GPPrior(KERNEL, X, None, None, 1e-5).jitter == 1e-5


@pytest.mark.parametrize("name", list(CONFIGURATIONS))
def test_configurations_match_dense(name):
    fields, logdet_tolerance = CONFIGURATIONS[name]
    lml, mean, variance = _dense_reference()
    prior = GPPrior(KERNEL, X, matrix_free=True, **fields)
    got_lml = prior.log_marginal(Y, NOISE_VAR)
    got_mean, got_variance = prior.condition(Y, NOISE_VAR).predict(X_STAR)
    assert abs(float(got_lml - lml)) < logdet_tolerance
    # the solves run at CG's 1e-10, so 1e-7 holds with margin (measured 5e-11)
    np.testing.assert_allclose(got_mean, mean, atol=1e-7)
    np.testing.assert_allclose(got_variance, variance, atol=1e-7)


def _example_configurations():
    preconditioner = gx.NystromPreconditioner.lazy(rank=100, key=jr.key(0))
    return {
        "bbmm": {
            "solver": gx.BBMMSolver(num_probes=50),
            "preconditioner": preconditioner,
        },
        "cg+slq": {
            "solver": gx.CGSolver(rtol=1e-4),
            "preconditioner": preconditioner,
            "logdet": gx.SLQLogdet(num_probes=50),
        },
    }


@pytest.mark.parametrize("name", ["bbmm", "cg+slq"])
def test_matrix_free_never_densifies(name, monkeypatch):
    # a counting operator: any materialisation of the N x N kernel matrix
    # is counted, and none is allowed
    calls = []

    def counting_as_matrix(self):
        calls.append(self)
        raise AssertionError("the kernel matrix was formed")

    monkeypatch.setattr(KernelOperator, "as_matrix", counting_as_matrix)
    fields = _example_configurations()[name]
    prior = GPPrior(KERNEL, X, matrix_free=True, **fields)
    lml = prior.log_marginal(Y, NOISE_VAR)
    mean, variance = prior.condition(Y, NOISE_VAR).predict(X_STAR)
    assert not calls
    assert np.isfinite(float(lml))
    assert np.all(np.isfinite(np.asarray(mean)))
    assert np.all(np.isfinite(np.asarray(variance)))


def _model(fields, y):
    kernel = Matern(nu=1.5, init_lengthscale=0.3)
    prior = GPPrior(kernel, X, matrix_free=True, **fields)
    noise_var = numpyro.param("noise_var", 0.1, constraint=dist.constraints.positive)
    gp_factor("y", prior, y, noise_var)


@pytest.mark.parametrize("name", ["bbmm", "cg+slq"])
def test_gp_factor_under_jit_and_grad(name):
    fields = _example_configurations()[name]

    def log_joint(noise_var):
        # the fields are closed over, not passed through jit
        joint, _ = log_density(
            handlers.seed(_model, 0), (fields, Y), {}, {"noise_var": noise_var}
        )
        return joint

    value, grad = jax.jit(jax.value_and_grad(log_joint))(jnp.asarray(NOISE_VAR))
    assert np.isfinite(float(value))
    assert np.isfinite(float(grad))


@pytest.mark.slow
@pytest.mark.parametrize("name", ["bbmm", "cg+slq"])
def test_svi_step_with_bbmm(name):
    # SVI on the noise variance, with the kernel's priors inside the model:
    # every step builds the prior anew, and the lazy preconditioner follows.
    fields = _example_configurations()[name]
    svi = SVI(_model, lambda fields, y: None, Adam(1e-2), Trace_ELBO())
    result = svi.run(jr.key(0), 20, fields, Y, progress_bar=False)
    assert np.all(np.isfinite(np.asarray(result.losses)))
    assert result.losses[-1] < result.losses[0]


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        # a preconditioner with the default exact solver
        ({"preconditioner": _nystrom()}, "DenseSolver"),
        # a solver with no preconditioned solve
        ({"solver": gx.MINRESSolver(), "preconditioner": _nystrom()}, "MINRES"),
        # NystromLogdet builds its own preconditioner
        (
            {
                "solver": gx.CGSolver(),
                "preconditioner": _nystrom(),
                "logdet": gx.NystromLogdet(shift=None),
            },
            "NystromLogdet",
        ),
        # two preconditioners
        (
            {
                "solver": gx.CGSolver(preconditioner=_nystrom()),
                "preconditioner": _partial_cholesky(),
            },
            "already carries",
        ),
        # a preconditioner that does not know P, with BBMM's shared pass
        (
            {
                "solver": gx.BBMMSolver(),
                "preconditioner": gx.OperatorPreconditioner(
                    lx.IdentityLinearOperator(jax.ShapeDtypeStruct((N,), X.dtype))
                ),
            },
            "supports_slq",
        ),
    ],
)
def test_invalid_combinations_raise(fields, match):
    with pytest.raises(ValueError, match=match) as error:
        GPPrior(KERNEL, X, matrix_free=True, **fields)
    # the message names the three fields
    assert "GPPrior(solver=" in str(error.value)
    assert "preconditioner=" in str(error.value)
    assert "logdet=" in str(error.value)


def test_other_priors_take_the_fields():
    preconditioner = _nystrom()
    logdet = gx.SLQLogdet(num_probes=10)
    solver = gx.CGSolver()
    with handlers.seed(rng_seed=0):
        sparse = SparseGPPrior(
            RBF(),
            Z=X[:10],
            solver=solver,
            preconditioner=preconditioner,
            logdet=logdet,
        )
        multi = MultiOutputGPPrior(
            ICMKernel(kernel=RBF(), mixing=jnp.array([[1.0], [0.5]])),
            X[:20],
            solver=solver,
            preconditioner=preconditioner,
            logdet=logdet,
        )
        oilmm = OILMMGPPrior(
            OILMMKernel(
                kernels=(RBF(pyrox_name="RBF_q0"), RBF(pyrox_name="RBF_q1")),
                mixing=jnp.linalg.qr(jr.normal(jr.key(3), (3, 2)))[0],
            ),
            X[:20],
            solver=solver,
            preconditioner=preconditioner,
            logdet=logdet,
        )
    expected = gx.compose_solver(solver, preconditioner=preconditioner, logdet=logdet)
    assert sparse.solver_strategy() == expected
    assert multi.solver_strategy() == expected
    for latent in oilmm.latent_priors():
        assert latent.solver_strategy() == expected
    with pytest.raises(ValueError, match=r"SparseGPPrior\(solver=None"):
        SparseGPPrior(RBF(), Z=X[:10], preconditioner=preconditioner)
    with pytest.raises(ValueError, match=r"MultiOutputGPPrior\(solver=None"):
        MultiOutputGPPrior(
            ICMKernel(kernel=RBF(), mixing=jnp.array([[1.0], [0.5]])),
            X[:20],
            preconditioner=preconditioner,
        )
