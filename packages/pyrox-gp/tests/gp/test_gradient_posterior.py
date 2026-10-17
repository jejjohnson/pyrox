"""Gradient posterior of `ConditionedGP` (KD9, pyrox#305).

``predict_gradient`` / ``predict_var_gradient`` / ``predict_mean_hessian``
against autodiff of the value predictive, finite differences, closed forms
and a dense joint-Gaussian reference built from kernellib's
``DerivativeIndexed`` and ``gaussx.conditional``.
"""

from __future__ import annotations

import einx
import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import kernellib as kl
import lineax as lx
import numpy as np
import numpyro.distributions as dist
import pyrox_gp as px
import pytest
from numpyro import handlers
from pyrox_gp._context import _kernel_context
from pyrox_gp._kernel_operator import freeze_kernel


N, M = 20, 7
NOISE = 1e-2
JITTER = 1e-8


def _data(D: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    X = jr.uniform(jr.key(0), (N, D), minval=-1.0, maxval=1.0)
    y = jnp.sin(3.0 * X[:, 0]) + 0.5 * jnp.cos(2.0 * X[:, -1])
    X_star = jr.uniform(jr.key(1), (M, D), minval=-1.1, maxval=1.1)
    return X, y, X_star


def _condition(kernel, D: int, *, noise=NOISE, **prior_kwargs):
    X, y, X_star = _data(D)
    prior = px.GPPrior(kernel, X, jitter=JITTER, **prior_kwargs)
    return prior.condition(y, jnp.asarray(noise)), X_star


def _grad_of(fn, X_star):
    """Per-point ``jax.grad`` of a ``(M, D) -> (M,)`` predictor."""
    return jax.vmap(jax.grad(lambda x: fn(x[None])[0]))(X_star)


def _central_fd(fn, X_star, h=1e-5):
    D = X_star.shape[-1]
    cols = []
    for d in range(D):
        e = h * jnp.eye(D, dtype=X_star.dtype)[d]
        cols.append((fn(X_star + e) - fn(X_star - e)) / (2.0 * h))
    return jnp.stack(cols, axis=-1)


KERNELS_2D = {
    "rbf-ard": lambda: px.RBF(init_lengthscale=jnp.array([0.5, 0.8])),
    "matern32": lambda: px.Matern(nu=1.5, init_lengthscale=0.5),
    "matern52": lambda: px.Matern(nu=2.5, init_lengthscale=0.5),
    "rq": lambda: px.RationalQuadratic(init_lengthscale=0.5),
}


def _cases():
    for name, make in KERNELS_2D.items():
        for D in (1, 2):
            if name == "rbf-ard" and D == 1:
                yield pytest.param(lambda: px.RBF(init_lengthscale=0.5), 1, id="rbf-1d")
            else:
                yield pytest.param(make, D, id=f"{name}-{D}d")
    # Periodic on a Euclidean distance is only PSD in 1-D.
    yield pytest.param(
        lambda: px.RBF(init_lengthscale=0.5) + px.Periodic(init_period=2.0),
        1,
        id="rbf+periodic-1d",
    )


# --- mean --------------------------------------------------------------------


@pytest.mark.parametrize(("make", "D"), list(_cases()))
def test_mean_matches_autodiff_of_predict_mean(make, D):
    cond, X_star = _condition(make(), D)
    post = cond.predict_gradient(X_star)
    assert post.mean.shape == (M, D)
    np.testing.assert_allclose(
        post.mean, _grad_of(cond.predict_mean, X_star), rtol=0, atol=1e-10
    )


@pytest.mark.parametrize("D", [1, 2])
def test_mean_matches_finite_differences(D):
    cond, X_star = _condition(px.Matern(nu=2.5, init_lengthscale=0.5), D)
    np.testing.assert_allclose(
        cond.predict_gradient(X_star, cov="none").mean,
        _central_fd(cond.predict_mean, X_star),
        rtol=1e-6,
        atol=1e-8,
    )


def test_mean_matches_rbf_closed_form():
    """Johnson et al. (2020), §3: G_dn = -(x*_d - x_nd) / l_d^2 k(x*, x_n)."""
    ell = jnp.array([0.5, 0.8])
    cond, X_star = _condition(px.RBF(init_lengthscale=ell), 2)
    X = cond.prior.X
    K = cond.prior.kernel(X_star, X)
    diff = einx.subtract("m d, n d -> m n d", X_star, X)
    G = -einx.multiply(
        "m n d, m n -> m n d", einx.divide("m n d, d -> m n d", diff, ell**2), K
    )
    expected = einx.dot("m n d, n -> m d", G, cond.cache.alpha)
    np.testing.assert_allclose(cond.predict_gradient(X_star).mean, expected, atol=1e-10)


def test_linear_mean_fn_shifts_mean_by_its_slope():
    a = jnp.array([0.7, -1.3])
    X, y, X_star = _data(2)
    kernel = px.RBF(init_lengthscale=0.5)

    def mean_fn(Z):
        return einx.dot("n d, d -> n", Z, a)

    base = px.GPPrior(kernel, X, jitter=JITTER).condition(y, jnp.asarray(NOISE))
    shifted = px.GPPrior(kernel, X, mean_fn=mean_fn, jitter=JITTER).condition(
        y + mean_fn(X), jnp.asarray(NOISE)
    )
    np.testing.assert_allclose(
        shifted.predict_gradient(X_star).mean - base.predict_gradient(X_star).mean,
        jnp.broadcast_to(a, (M, 2)),
        atol=1e-12,
    )


# --- covariance --------------------------------------------------------------


def _dense_reference(cond, X_star):
    """Joint Gaussian over [f(X); grad f(X*)], conditioned on the value rows."""
    prior = cond.prior
    k = prior.kernel.frozen()
    rows = jnp.concatenate(
        [
            kl.derivative_inputs(prior.X, dims=()),
            kl.derivative_inputs(X_star, values=False),
        ]
    )
    K = kl.DerivativeIndexed(k)(rows, rows)
    n = prior.X.shape[0]
    noise = jnp.concatenate(
        [
            jnp.full((n,), prior.jitter + cond.noise_var),
            jnp.zeros(rows.shape[0] - n),
        ]
    )
    K = K + jnp.diag(noise)
    mean, cov = gx.conditional(
        jnp.zeros(rows.shape[0]),
        lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag),
        jnp.arange(n),
        cond.y,
    )
    D = X_star.shape[-1]
    return (
        einx.id("(d m) -> m d", mean, d=D),
        einx.id("(i a) (j b) -> a i b j", cov.as_matrix(), i=D, j=D),
    )


@pytest.mark.parametrize("D", [1, 2])
def test_full_cov_matches_dense_joint_reference(D):
    cond, X_star = _condition(px.RBF(init_lengthscale=0.5), D)
    post = cond.predict_gradient(X_star, cov="full")
    ref_mean, ref_cov = _dense_reference(cond, X_star)
    assert post.cov.shape == (M, D, M, D)
    np.testing.assert_allclose(post.mean, ref_mean, atol=1e-9)
    np.testing.assert_allclose(post.cov, ref_cov, atol=1e-9)


def test_full_cov_matrix_free_cg_matches_dense_joint_reference():
    solver = px.preconditioned_cg_solver(rank=10, key=jr.key(0), rtol=1e-10, atol=1e-10)
    cond, X_star = _condition(
        px.RBF(init_lengthscale=0.5), 2, solver=solver, matrix_free=True
    )
    assert cond.cache.factor is None
    post = cond.predict_gradient(X_star, cov="full")
    ref_mean, ref_cov = _dense_reference(cond, X_star)
    np.testing.assert_allclose(post.mean, ref_mean, atol=1e-5)
    np.testing.assert_allclose(post.cov, ref_cov, atol=1e-5)
    marginal = cond.predict_gradient(X_star).cov
    np.testing.assert_allclose(
        marginal, einx.id("a i a j -> a i j", ref_cov), atol=1e-5
    )


@pytest.mark.parametrize("D", [1, 2])
def test_marginal_blocks_are_the_diagonal_of_full(D):
    cond, X_star = _condition(px.Matern(nu=2.5, init_lengthscale=0.5), D)
    marginal = cond.predict_gradient(X_star, cov="marginal").cov
    full = cond.predict_gradient(X_star, cov="full").cov
    assert marginal.shape == (M, D, D)
    np.testing.assert_allclose(
        marginal, einx.id("a i a j -> a i j", full), rtol=0, atol=1e-12
    )
    assert cond.predict_gradient(X_star, cov="none").cov is None


def test_unknown_cov_option_raises():
    cond, X_star = _condition(px.RBF(), 1)
    with pytest.raises(ValueError, match="cov must be"):
        cond.predict_gradient(X_star, cov="diag")


@pytest.mark.parametrize(("make", "D"), list(_cases()))
def test_posterior_cov_is_psd_and_below_prior(make, D):
    cond, X_star = _condition(make(), D)
    post = cond.predict_gradient(X_star)
    k = freeze_kernel(make())
    H = jax.vmap(lambda x: jax.jacfwd(jax.grad(k.pairwise, 0), 1)(x, x))(X_star)
    assert float(jnp.linalg.eigvalsh(post.cov).min()) >= -1e-10
    assert float(jnp.linalg.eigvalsh(H - post.cov).min()) >= -1e-10
    full = einx.id(
        "a i b j -> (a i) (b j)", cond.predict_gradient(X_star, cov="full").cov
    )
    np.testing.assert_allclose(full, full.T, atol=0)
    assert float(jnp.linalg.eigvalsh(full).min()) >= -1e-10 * M * D


@pytest.mark.parametrize(
    ("kernel", "block"),
    [
        (
            px.RBF(init_variance=1.5, init_lengthscale=jnp.array([0.5, 0.8])),
            jnp.diag(1.5 / jnp.array([0.5, 0.8]) ** 2),
        ),
        (
            px.Matern(nu=2.5, init_variance=1.5, init_lengthscale=0.5),
            5.0 * 1.5 / (3.0 * 0.5**2) * jnp.eye(2),
        ),
    ],
    ids=["rbf-ard", "matern52"],
)
def test_huge_noise_returns_the_stationary_prior_block(kernel, block):
    cond, X_star = _condition(kernel, 2, noise=1e8)
    post = cond.predict_gradient(X_star)
    # The downdate G K^-1 G^T is O(N |G|^2 / noise) ~ 1e-8 here; atol covers
    # the off-diagonal zeros, rtol the diagonal.
    np.testing.assert_allclose(
        post.cov, jnp.broadcast_to(block, (M, 2, 2)), rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(post.mean, 0.0, atol=1e-6)


# --- variance gradient and mean Hessian --------------------------------------


@pytest.mark.parametrize(("make", "D"), list(_cases()))
def test_var_gradient_matches_autodiff(make, D):
    cond, X_star = _condition(make(), D)
    got = cond.predict_var_gradient(X_star)
    assert got.shape == (M, D)
    np.testing.assert_allclose(got, _grad_of(cond.predict_var, X_star), atol=1e-10)


@pytest.mark.parametrize("matrix_free", [False, True])
def test_var_gradient_matches_finite_differences(matrix_free):
    kwargs = {}
    if matrix_free:
        kwargs = {
            "matrix_free": True,
            "solver": px.preconditioned_cg_solver(
                rank=10, key=jr.key(0), rtol=1e-10, atol=1e-10
            ),
        }
    cond, X_star = _condition(px.RBF(init_lengthscale=0.5), 2, **kwargs)
    np.testing.assert_allclose(
        cond.predict_var_gradient(X_star),
        _central_fd(cond.predict_var, X_star),
        atol=1e-6,
    )


def test_var_gradient_vanishes_at_centroid_of_symmetric_design():
    X = jnp.linspace(-1.0, 1.0, N)[:, None]
    cond = px.GPPrior(px.RBF(init_lengthscale=0.5), X, jitter=JITTER).condition(
        jnp.sin(3.0 * X[:, 0]), jnp.asarray(NOISE)
    )
    np.testing.assert_allclose(
        cond.predict_var_gradient(jnp.zeros((1, 1))), 0.0, atol=1e-8
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda: px.RBF(init_lengthscale=jnp.array([0.5, 0.8])),
        lambda: px.Matern(nu=2.5, init_lengthscale=0.5),
    ],
    ids=["rbf-ard", "matern52"],
)
def test_mean_hessian_matches_autodiff_and_is_symmetric(make):
    X, y, X_star = _data(2)

    def mean_fn(Z):
        return 0.3 * einx.sum("n [d]", Z**2) + Z[:, 0]

    cond = px.GPPrior(make(), X, mean_fn=mean_fn, jitter=JITTER).condition(
        y, jnp.asarray(NOISE)
    )
    hess = cond.predict_mean_hessian(X_star)
    ref = jax.vmap(jax.hessian(lambda x: cond.predict_mean(x[None])[0]))(X_star)
    assert hess.shape == (M, 2, 2)
    np.testing.assert_allclose(hess, ref, atol=1e-9)
    np.testing.assert_allclose(hess, einx.id("m i j -> m j i", hess), atol=1e-12)


# --- errors and handlers -----------------------------------------------------


def test_rough_kernel_raises_kernellib_value_error():
    cond, X_star = _condition(px.Matern(nu=0.5), 1)
    for call in (
        cond.predict_gradient,
        cond.predict_var_gradient,
        cond.predict_mean_hessian,
    ):
        with pytest.raises(ValueError, match="mean-square differentiable"):
            call(X_star)


class _NoFrozenRBF(px.RBF):
    _frozen_cls = None
    pyrox_name: str = "NoFrozenRBF"


def test_kernel_without_frozen_raises_not_implemented_with_hint():
    cond, X_star = _condition(_NoFrozenRBF(), 1)
    with pytest.raises(NotImplementedError, match="_frozen_cls"):
        cond.predict_gradient(X_star)


def test_prior_hyperparameters_registered_once_and_shared():
    X, y, X_star = _data(2)
    kernel = px.RBF()
    kernel.set_prior("lengthscale", dist.LogNormal(jnp.log(0.5), 0.1))
    with handlers.seed(rng_seed=1):
        cond = px.GPPrior(kernel, X, jitter=JITTER).condition(y, jnp.asarray(NOISE))

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        cond.predict_gradient(X_star, cov="full")
    samples = [name for name, site in tr.items() if site["type"] == "sample"]
    assert samples == ["RBF.lengthscale"]

    with handlers.seed(rng_seed=0), _kernel_context(kernel):
        got = cond.predict_gradient(X_star).mean
        ref = _grad_of(cond.predict_mean, X_star)
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_jit_and_pytree_result():
    cond, X_star = _condition(px.RBF(init_lengthscale=0.5), 2)
    post = jax.jit(lambda Xs: cond.predict_gradient(Xs))(X_star)
    assert isinstance(post, px.GradientPosterior)
    np.testing.assert_allclose(
        post.mean, cond.predict_gradient(X_star).mean, atol=1e-12
    )


# --- Monte Carlo cross-check with pathwise samples ----------------------------


@pytest.mark.slow
@pytest.mark.parametrize("D", [1, 2])
def test_pathwise_gradient_samples_match_posterior(D):
    ell, var = 0.35, 1.0
    cond, X_star = _condition(px.RBF(init_variance=var, init_lengthscale=ell), D)
    paths = px.PathwiseSampler(cond, n_features=4096).sample_paths(
        jr.key(3), n_paths=1024
    )
    # (M, S, D): gradient of every path at every test point.
    grads = jax.lax.map(jax.jacfwd(lambda x: paths(x[None])[:, 0]), X_star)
    emp_mean = einx.mean("m [s] d", grads)
    centred = einx.subtract("m s d, m d -> m s d", grads, emp_mean)
    emp_cov = einx.dot("m s i, m s j -> m i j", centred, centred) / (grads.shape[1] - 1)
    post = cond.predict_gradient(X_star)
    np.testing.assert_allclose(emp_mean, post.mean, atol=0.15 * var**0.5 / ell)
    np.testing.assert_allclose(emp_cov, post.cov, atol=0.15 * var / ell**2)
