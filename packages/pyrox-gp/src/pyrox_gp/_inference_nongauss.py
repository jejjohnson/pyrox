"""Advanced GP inference strategies for non-conjugate likelihoods.

All strategies in this module produce a Gaussian approximation to the
posterior ``q(f) = N(m, V)`` over the latent function values at the
training inputs, given a non-conjugate likelihood ``p(y | f)``. They
share the same site-based view: each likelihood factor contributes a
diagonal Gaussian *site* with natural parameters
$(\\lambda^{(1)}, \\Lambda^{(2)}) = (J - H m, -H)$. Strategies
differ only in how the per-site curvature ``H`` and effective gradient
``J`` are obtained:

================================  =======================================
Strategy                          Curvature ``H`` from
================================  =======================================
`LaplaceInference`         exact ``-d^2 log p / df^2`` at the mode
`GaussNewtonInference`     generalized Gauss-Newton
                                  (positive-semidefinite by construction)
`PosteriorLinearization`   statistical linearization under cavity
                                  via cubature
`ExpectationPropagation`   moment matching against the tilted
                                  distribution per site
`QuasiNewtonInference`     L-BFGS optimization to MAP, exact
                                  Hessian at convergence
================================  =======================================

All five accept a `pyrox_gp.GPPrior` and a scalar-latent
`pyrox_gp.Likelihood`, and return an
`ConditionedNonGaussGP` that quacks like
`pyrox_gp.ConditionedGP` (``predict``, ``predict_mean``,
``predict_var``).

Everything here is pure JAX and ``equinox``-compatible (jit / grad /
vmap). The heavy lifting — cavity arithmetic, natural-parameter
updates, GGN curvature, Cholesky solves — comes from
`gaussx`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpyro
import optimistix as optx
from argminax.solve import laplace_mode_covariance, site_fixed_point
from gaussx import (
    AbstractIntegrator,
    AbstractLogdetStrategy,
    AbstractPreconditioner,
    AbstractSolverStrategy,
    DenseSolver,
    GaussHermiteIntegrator,
    PredictionCache,
    SitePosterior,
    cavity_distribution,
    damped_natural_update,
    ep_tilted_moments,
    newton_update,
    predict_variance,
    safe_cholesky,
    site_log_evidence,
    site_posterior,
    symmetrize,
)
from jaxtyping import Array, Bool, Float, Int, PyTree
from pyrox._core import call_context
from pyrox.likelihoods import AbstractObservation

from pyrox_gp._inference import _per_site_expectation
from pyrox_gp._kernel_operator import freeze_kernel
from pyrox_gp._likelihoods import _site_log_prob
from pyrox_gp._protocols import Kernel, Likelihood
from pyrox_gp._solvers import compose_solver_fields


if TYPE_CHECKING:
    from pyrox_gp._models import GPPrior


# --- result type ---------------------------------------------------------


class ConditionedNonGaussGP(eqx.Module):
    r"""GP conditioned on a non-Gaussian likelihood via an advanced strategy.

    Equivalent role to `pyrox_gp.ConditionedGP` but the posterior over
    training latents is the Gaussian of diagonal sites, a
    `gaussx.SitePosterior`: any site-based approximation of $p(f \mid y)$
    looks like a Gaussian-likelihood regression with per-point noise
    variance $d_n = 1 / \lambda_n$ and targets
    $\mu_{\text{site},n} = \lambda^{(1)}_n / \lambda_n$, so with
    $A = K + \mathrm{diag}(d)$ the fit keeps the weights
    $w = A^{-1} (\mu_\text{site} - \mu_f)$ and predicts with

    $$
    \mu_* = \mu(X_*) + K_{*X} w, \qquad
    \sigma_*^2 = k_{**} - k_*^\top A^{-1} k_* ,
    $$

    through the solver of the fit: triangular solves with the Cholesky
    factor of $A$ kept from the fit on the dense path, one solve per test
    point with an iterative solver, so a matrix-free prior stays matrix-free.

    The kernel's hyperparameters are those the fit used (pyrox#551): the fit
    resolves them once, and `predict` reuses them. Before pyrox#551 each
    ``predict`` evaluated the prior's kernel again, so a kernel with priors
    drew new hyperparameters there; refit to resample.

    Attributes:
        prior: The `GPPrior` the fit was given.
        y: Training targets (kept for round-trip / diagnostics).
        site_posterior: The `gaussx.SitePosterior` of the fitted sites: the
            prior covariance, the sites, the weights $w$, the posterior mean
            and the solver (with the Cholesky factor of $A$ on the dense
            path).
        log_marginal_approx: Approximate log marginal likelihood
            (the scalar each strategy reports — interpretation is
            strategy-specific; see the per-class docstring).
        n_iter: Iterations used by the strategy, a scalar integer array,
            so a fit runs under ``jax.jit`` and ``jax.vmap``.
        converged: Whether the convergence tolerance was met, a scalar
            boolean array; with an iterative solver it is also false when an
            inner solve of a Newton fit missed its tolerance.
        kernel: The prior's kernel with the hyperparameters of the fit, a
            plain kernellib kernel (``frozen()``); a kernel that cannot be
            frozen is kept as it is and evaluated once per ``predict``.
        batch_size: Solves vectorised together when `q_var` is read with an
            iterative solver.
    """

    prior: GPPrior
    y: Float[Array, " N"]
    site_posterior: SitePosterior
    log_marginal_approx: Float[Array, ""]
    n_iter: Int[Array, ""]
    converged: Bool[Array, ""]
    kernel: Kernel
    batch_size: int = eqx.field(static=True, default=32)

    @property
    def site_nat1(self) -> Float[Array, " N"]:
        r"""The site information $\lambda^{(1)} = \lambda \odot \mu_\text{site}$."""
        return self.site_posterior.site_information

    @property
    def site_nat2(self) -> Float[Array, " N"]:
        r"""The site precisions $\lambda$, all positive."""
        return self.site_posterior.site_precision

    @property
    def q_mean(self) -> Float[Array, " N"]:
        r"""The posterior mean $\mu_f + K w$ at the training inputs."""
        return self.site_posterior.mean

    @property
    def solver(self) -> AbstractSolverStrategy | None:
        """The iterative solver the fit used; ``None`` for a dense fit.

        A dense fit predicts by the Cholesky factor of $A$ kept from the fit,
        an iterative one by one solve per test point through this solver.
        """
        solver = self.site_posterior.solver
        return None if isinstance(solver, DenseSolver) else solver

    @property
    def q_var(self) -> Float[Array, " N"]:
        r"""The posterior marginal variances $\mathrm{diag}(K - K A^{-1} K)$.

        Computed when read, so a fit that needs only the mean or the
        evidence does no extra solves: triangular solves with the kept
        factor on the dense path, $N$ solves (``batch_size`` at a time) with
        an iterative solver.
        """
        return self.site_posterior.variance(method="exact", batch_size=self.batch_size)

    def _predict(
        self, X_star: Float[Array, "M D"], *, variance: bool
    ) -> tuple[Float[Array, " M"], Float[Array, " M"] | None]:
        """The predictive mean and, if asked, the marginal variances."""
        kernel = self.kernel
        posterior = self.site_posterior
        # a no-op for a frozen kernel; one draw for one that could not be
        with call_context(kernel):
            K_cross = kernel(X_star, self.prior.X)
            K_diag = kernel.diag(X_star) if variance else None
            prior_mean_test = self.prior.mean(X_star)
        # μ_* = μ(X_*) + K_{*X} w,  K_{*X}: (M, N), w: (N,) → (M,)
        mean = prior_mean_test + einx.dot("m n, n -> m", K_cross, posterior.weights)
        if K_diag is None:
            return mean, None
        cache = PredictionCache(
            alpha=posterior.weights,
            operator=posterior.system,
            factor=posterior.factor,
        )
        # σ²_* = k_** − k_*ᵀ A⁻¹ k_*
        var = predict_variance(cache, K_cross, K_diag, solver=posterior.solver)
        return mean, jnp.maximum(var, 0.0)

    def predict_mean(self, X_star: Float[Array, "M D"]) -> Float[Array, " M"]:
        r"""$\mu_* = \mu(X_*) + K_{*X} w$, from the weights of the fit."""
        return self._predict(X_star, variance=False)[0]

    def predict_var(self, X_star: Float[Array, "M D"]) -> Float[Array, " M"]:
        r"""$k_{**} - k_*^\top (K + \mathrm{diag}(1/\lambda))^{-1} k_*$."""
        var = self._predict(X_star, variance=True)[1]
        assert var is not None
        return var

    def predict(
        self, X_star: Float[Array, "M D"]
    ) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
        """Joint mean / marginal-variance prediction at ``X_star``.

        Both use the hyperparameters of the fit, so the pair is
        self-consistent and a kernel with priors registers no new site.
        """
        mean, var = self._predict(X_star, variance=True)
        assert var is not None
        return mean, var


# --- helpers -------------------------------------------------------------


def _check_scalar_latent(lik: Likelihood) -> None:
    latent_dim = getattr(lik, "latent_dim", 1)
    if latent_dim != 1:
        msg = (
            f"{type(lik).__name__} declares latent_dim={latent_dim}; the "
            "scalar-latent advanced inference strategies (Laplace, GN, EP, "
            "PL, QN) do not yet support multi-latent likelihoods. Use SVGP "
            "or wait for the multi-latent inference follow-up."
        )
        raise ValueError(msg)


_MATRIX_FREE_EPIC = "https://github.com/jejjohnson/pyrox/issues/353"


def _reject_matrix_free(prior: GPPrior) -> None:
    """Refuse ``GPPrior(matrix_free=True)`` in `QuasiNewtonInference`.

    A matrix-free prior promises never to form the N × N kernel matrix, and
    the quasi-Newton search whitens with a Cholesky root of it (pyrox#549).
    Laplace, Gauss-Newton (pyrox#551), posterior linearisation and EP
    (pyrox#552) solve through the prior's solver and accept it. Raising
    keeps that promise instead of allocating ``K`` silently.
    """
    if not prior.matrix_free:
        return
    raise ValueError(
        "QuasiNewtonInference does not support GPPrior(matrix_free=True): it "
        "whitens its L-BFGS search with a Cholesky root of the kernel matrix, "
        "so it never accepts a matrix-free prior; LaplaceInference with an "
        "iterative solver is the matrix-free route (pyrox#551). Use "
        f"matrix_free=False for this fit. See {_MATRIX_FREE_EPIC}."
    )


def _prior_K(prior: GPPrior) -> Float[Array, "N N"]:
    with call_context(prior.kernel):
        K = prior.kernel(prior.X, prior.X)
    return K.at[jnp.diag_indices_from(K)].add(prior.jitter)


def _one_site(site: PyTree) -> PyTree:
    """One site's data with a leading site axis of length one on every leaf."""
    return jax.tree.map(lambda leaf: leaf[None], site)


def _per_point_grad_hess(
    log_prob_per_point: Callable[[Float[Array, " N"], PyTree], Float[Array, " N"]],
    f: Float[Array, " N"],
    y: PyTree,
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    r"""Per-point ``g_n = d log p_n / df_n`` and ``h_n = d^2 log p_n / df_n^2``.

    ``log_prob_per_point`` must be vectorized over ``f`` and return one
    log-density per observation (no sum). For scalar likelihoods the
    Hessian is diagonal in ``f`` so per-point second derivatives suffice.
    ``y`` is the per-site data of `_site_data`, a leading site axis on
    every leaf.
    """

    def per_n(f_n: Float[Array, ""], y_n: PyTree) -> Float[Array, ""]:
        return log_prob_per_point(f_n[None], _one_site(y_n))[0]

    g = jax.vmap(jax.grad(per_n))(f, y)
    h = jax.vmap(jax.grad(jax.grad(per_n)))(f, y)
    return g, h


def _psd_operator(M: Float[Array, "N N"]) -> lx.AbstractLinearOperator:
    """Wrap a symmetric PSD array as a tagged ``lineax`` operator."""
    return lx.MatrixLinearOperator(M, lx.positive_semidefinite_tag)


def _psd_safe_cholesky(M: Float[Array, "N N"]) -> Float[Array, "N N"]:
    """Symmetrize ``M`` then call `gaussx.safe_cholesky`.

    ``gaussx.safe_cholesky`` runs an adaptive ``while_loop`` that
    multiplies a diagonal jitter from ``1e-8`` to ``1e-2`` until the
    Cholesky succeeds (or returns NaNs after 5 retries). It does not
    symmetrize its input, so we do that here before wrapping ``M`` as
    a PSD `lineax` operator. Float32 + densely-packed kernel
    inputs is the realistic failure case this guards against.
    """
    return safe_cholesky(_psd_operator(symmetrize(M)))


def _site_data(
    lik: Likelihood, y: Float[Array, " N"], X: Float[Array, "N D"]
) -> PyTree:
    """The per-site data `_log_prob_per_point_factory`'s function reads.

    ``(y, {"X": X})`` for an observation model, whose `build` reads its
    inputs from ``data["X"]``; ``y`` alone for any other `Likelihood`,
    whose ``log_prob`` these strategies never hand ``X``.
    """
    if isinstance(lik, AbstractObservation):
        return (y, {"X": X})
    return y


def _log_prob_per_point_factory(
    lik: Likelihood,
) -> Callable[[Float[Array, " N"], PyTree], Float[Array, " N"]]:
    """Wrap the likelihood so it returns per-point log-densities.

    pyrox's ``Likelihood.log_prob`` is summed over points by convention.
    For per-site curvature we need the unsummed per-point quantity,
    which we compute via ``vmap`` over the scalar log-density. An
    observation model's comes from the one-site gaussx likelihood it
    builds (`_site_log_prob`), on the per-site data of `_site_data`.
    """
    if isinstance(lik, AbstractObservation):
        observation = lik

        def per_one(f_n: Float[Array, ""], site: PyTree) -> Float[Array, ""]:
            y_n, data_n = site
            return _site_log_prob(observation, f_n, y_n, data_n)

    else:

        def per_one(f_n: Float[Array, ""], site: PyTree) -> Float[Array, ""]:
            return lik.log_prob(f_n[None], site[None])

    def per_point(f: Float[Array, " N"], sites: PyTree) -> Float[Array, " N"]:
        return jax.vmap(per_one)(f, sites)

    return per_point


def _strategy_solver(
    strategy: LaplaceInference
    | GaussNewtonInference
    | PosteriorLinearization
    | ExpectationPropagation,
    prior: GPPrior,
) -> AbstractSolverStrategy:
    """The strategy's solver, each unset field taken from the prior.

    ``solver``, ``preconditioner`` and ``logdet`` fall back one by one to
    the prior's fields, and the three are composed by
    `gaussx.compose_solver`; none set on either is `gaussx.DenseSolver`.
    """
    return compose_solver_fields(
        type(strategy).__name__,
        prior.solver if strategy.solver is None else strategy.solver,
        prior.preconditioner
        if strategy.preconditioner is None
        else strategy.preconditioner,
        prior.logdet if strategy.logdet is None else strategy.logdet,
    )


def _fitted_kernel(prior: GPPrior) -> Kernel:
    """The prior's kernel with its hyperparameters resolved, for `predict`.

    Called inside the fit's `call_context`, so the draw is the fit's. A
    kernel with no plain kernellib counterpart comes back unchanged.
    """
    try:
        return freeze_kernel(prior.kernel)
    except NotImplementedError:
        return prior.kernel


def _newton_fit(
    strategy: LaplaceInference | GaussNewtonInference,
    prior: GPPrior,
    likelihood: Likelihood,
    y: Float[Array, " N"],
    key: Array | None,
) -> ConditionedNonGaussGP:
    """Damped Newton to the Laplace mode, on `argminax.solve.laplace_mode_covariance`.

    The loop is a ``lax.while_loop`` and the mode is implicitly
    differentiable, so the fit runs under ``jax.jit`` and ``jax.vmap`` and a
    gradient does not unroll it. Every solve against
    $A = K + \\mathrm{diag}(1 / \\lambda)$ goes through the composed solver,
    on the prior's own operator, so a matrix-free prior with an iterative
    solver never forms $K$. The sites, the posterior and the evidence
    (`gaussx.site_log_evidence`) at the mode come from argminax too; the
    marginal variances wait until `ConditionedNonGaussGP.q_var` is read.
    """
    _check_scalar_latent(likelihood)
    solver = _strategy_solver(strategy, prior)
    # one hyperparameter draw for the operator, the mean and the predictions
    with call_context(prior.kernel):
        prior_cov = prior._prior_operator()
        prior_mean = jnp.asarray(prior.mean(prior.X))
        kernel = _fitted_kernel(prior)
    log_prob_per_point = _log_prob_per_point_factory(likelihood)
    site_data = _site_data(likelihood, y, prior.X)
    fit = laplace_mode_covariance(
        prior_cov,
        prior_mean,
        lambda f: log_prob_per_point(f, site_data),
        max_iter=strategy.max_iter,
        tol=strategy.tol,
        damping=strategy.damping,
        precision_floor=strategy.precision_floor,
        adjoint=strategy.adjoint,
        solver=solver,
        with_variance=False,
        key=key,
    )
    assert fit.log_marginal is not None  # with_log_marginal defaults to True
    return ConditionedNonGaussGP(
        prior=prior,
        y=y,
        site_posterior=fit.site_posterior,
        log_marginal_approx=fit.log_marginal,
        n_iter=fit.n_iter,
        converged=fit.converged,
        kernel=kernel,
    )


# --- Laplace -------------------------------------------------------------


class LaplaceInference(eqx.Module):
    r"""Laplace approximation via Newton iteration on the log posterior.

    Iterates the standard GP-Laplace fixed-point loop (Rasmussen &
    Williams Algorithm 3.1): at each iteration evaluate per-point
    gradient $g$ and Hessian diagonal $h$ of $\log p(y \mid f)$, set the
    sites $\lambda_n = \max(-h_n, \epsilon)$ and
    $\lambda_n \mu_{\text{site},n} = g_n + \lambda_n f_n$ (floor $\epsilon$,
    ``precision_floor``), and move $f$ towards the posterior mean of those
    sites, $\mu_f + K A^{-1} (\mu_\text{site} - \mu_f)$ with
    $A = K + \mathrm{diag}(1 / \lambda)$.

    The loop is `argminax.solve.laplace_mode_covariance`, a
    ``lax.while_loop`` with an implicitly differentiable mode, so ``fit``
    runs under ``jax.jit`` and ``jax.vmap``; close over the prior, whose
    kernel registry does not survive being passed through a transform.
    Every solve against $A$, in the loop, at the mode and in `predict`, goes
    through the solver composed from ``solver``, ``preconditioner`` and
    ``logdet`` (each unset one taken from the prior), on the prior's own
    operator, so ``GPPrior(matrix_free=True)`` with an iterative solver
    never forms the $N \times N$ kernel matrix (pyrox#551).

    The reported ``log_marginal_approx`` is the Laplace evidence
    `gaussx.site_log_evidence` at the mode,

    $$
    \log p(y) \approx \sum_n \log p(y_n \mid m_n)
      - \tfrac12 (m - \mu_f)^\top K^{-1} (m - \mu_f)
      - \tfrac12 \log \lvert I + \Lambda^{1/2} K \Lambda^{1/2} \rvert ,
    $$

    with $m$ the posterior mean of the sites at the mode, computed without
    $K^{-1}$. A dense solver (the default) gives it exactly; a stochastic
    log-determinant (CG's own SLQ, `gaussx.BBMMSolver`) gives the equal
    pseudo-observation form, an estimate whose probes come from the ``key``
    of `fit`. Pass ``logdet=gaussx.DenseLogdet()`` beside CG for an exact
    evidence at moderate $N$.

    With an iterative solver, give it a tolerance at least ten times tighter
    than ``tol`` (the loop cannot resolve a step below the solves'
    residual). An inner solve that misses its tolerance makes ``converged``
    false; the variances and the evidence solve through the solver itself,
    which raises at ``max_steps`` when built with ``throw=True``, so inside
    a sampler build it with ``throw=False``.

    Attributes:
        max_iter: Newton iterations. Default ``20``.
        tol: ``inf``-norm convergence tolerance on ``f``. Default ``1e-6``.
        damping: Step-size in (0, 1] applied to each Newton update:
            ``f_{k+1} = (1 - alpha) f_k + alpha f_k^{Newton}``. Default
            ``1.0`` (full Newton step). Drop below 1 for non-log-concave
            likelihoods where pure Newton oscillates.
        precision_floor: Lower bound on the diagonal precision to keep
            ``K + 1/Λ`` well-conditioned even for log-concave-but-flat
            likelihoods (e.g. Bernoulli at extreme logits). Default
            ``1e-6``.
        solver: The solve of $A$; ``None`` takes the prior's ``solver``.
        preconditioner: The preconditioner of an iterative ``solver``;
            ``None`` takes the prior's. A lazy one
            (``gaussx.NystromPreconditioner.lazy``) reads the site variances
            from the system at every Newton step.
        logdet: The log-determinant of the evidence; ``None`` takes the
            prior's (and, if that is unset too, the solver's own). The three
            are composed by `gaussx.compose_solver`; none set on either side
            is `gaussx.DenseSolver`.
        adjoint: How the mode is differentiated, an
            ``optimistix.ImplicitAdjoint``; ``None`` is argminax's
            matrix-free GMRES tangent solve.
    """

    max_iter: int = eqx.field(static=True, default=20)
    tol: float = eqx.field(static=True, default=1e-6)
    damping: float = eqx.field(static=True, default=1.0)
    precision_floor: float = eqx.field(static=True, default=1e-6)
    solver: AbstractSolverStrategy | None = None
    preconditioner: AbstractPreconditioner | None = None
    logdet: AbstractLogdetStrategy | None = None
    adjoint: optx.ImplicitAdjoint | None = None

    def fit(
        self,
        prior: GPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
        *,
        key: Array | None = None,
    ) -> ConditionedNonGaussGP:
        r"""Fit the Laplace approximation at the mode of $p(f \mid y)$.

        Args:
            prior: The `GPPrior`, dense or ``matrix_free=True``.
            likelihood: A scalar-latent `Likelihood`.
            y: Observations, shape ``(N,)``.
            key: PRNG key of a stochastic log-determinant in the evidence
                (CG's SLQ, `gaussx.BBMMSolver`); ``None`` uses the solver's
                own seed. Ignored by an exact log-determinant.

        Returns:
            The `ConditionedNonGaussGP` at the mode.

        Examples:
            >>> import gaussx as gx
            >>> import jax
            >>> import jax.numpy as jnp
            >>> import jax.random as jr
            >>> from pyrox_gp import RBF, BernoulliLikelihood, GPPrior
            >>> from pyrox_gp import LaplaceInference
            >>> X = jnp.linspace(-2.0, 2.0, 10)[:, None]
            >>> y = (X[:, 0] > 0.0).astype(X.dtype)
            >>> # CG on a matrix-free prior: K is never formed, and the
            >>> # evidence is CG's stochastic Lanczos estimate from the key
            >>> prior = GPPrior(kernel=RBF(), X=X, matrix_free=True)
            >>> strategy = LaplaceInference(solver=gx.CGSolver(rtol=1e-10, atol=1e-10))
            >>> fit = jax.jit(
            ...     lambda labels: strategy.fit(
            ...         prior, BernoulliLikelihood(), labels, key=jr.key(0)
            ...     )
            ... )(y)
        """
        return _newton_fit(self, prior, likelihood, y, key)


# --- Gauss-Newton --------------------------------------------------------


class GaussNewtonInference(eqx.Module):
    r"""Gauss-Newton inference: Newton loop with PSD-projected curvature.

    Identical to `LaplaceInference` for log-concave likelihoods,
    where ``-d^2 log p / df^2`` is already positive. For non-log-concave
    likelihoods (e.g. `StudentTLikelihood`, where the Hessian
    becomes positive in the tails) GN aggressively floors the curvature
    to a strictly-positive value via ``precision_floor``, guaranteeing a
    PSD site precision and stable Newton steps. Laplace uses the same
    floor but typically with a smaller default.

    For Bernoulli / Poisson the Fisher information equals the negative
    Hessian, so GGN ≡ Laplace; for StudentT the floor matters. The loop is
    the same `argminax.solve.laplace_mode_covariance` call, so ``fit`` runs
    under ``jax.jit`` and ``jax.vmap``, and it takes the same ``solver``,
    ``preconditioner`` and ``logdet`` as `LaplaceInference`, falling back to
    the prior's, so a matrix-free prior with an iterative solver never forms
    the kernel matrix (pyrox#551).

    Attributes:
        max_iter: Iterations. Default ``20``.
        tol: ``inf``-norm convergence tolerance on ``f``. Default ``1e-6``.
        damping: Step-size in (0, 1] applied to each Newton update.
            Default ``1.0`` (full Newton step). Drop below 1 for
            non-log-concave likelihoods where pure Newton oscillates.
        precision_floor: Lower bound on the diagonal precision. Default
            ``1e-3`` — larger than `LaplaceInference` to ensure
            stable updates on non-log-concave likelihoods.
        solver: As in `LaplaceInference`; ``None`` takes the prior's.
        preconditioner: As in `LaplaceInference`.
        logdet: As in `LaplaceInference`.
        adjoint: As in `LaplaceInference`.
    """

    max_iter: int = eqx.field(static=True, default=20)
    tol: float = eqx.field(static=True, default=1e-6)
    damping: float = eqx.field(static=True, default=1.0)
    precision_floor: float = eqx.field(static=True, default=1e-3)
    solver: AbstractSolverStrategy | None = None
    preconditioner: AbstractPreconditioner | None = None
    logdet: AbstractLogdetStrategy | None = None
    adjoint: optx.ImplicitAdjoint | None = None

    def fit(
        self,
        prior: GPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
        *,
        key: Array | None = None,
    ) -> ConditionedNonGaussGP:
        """Fit the Gauss-Newton approximation; as `LaplaceInference.fit`.

        Args:
            prior: The `GPPrior`, dense or ``matrix_free=True``.
            likelihood: A scalar-latent `Likelihood`.
            y: Observations, shape ``(N,)``.
            key: PRNG key of a stochastic log-determinant in the evidence;
                ignored by an exact one.

        Returns:
            The `ConditionedNonGaussGP` at the mode.
        """
        return _newton_fit(self, prior, likelihood, y, key)


# --- Posterior linearization and EP: the site loop ------------------------


_VarianceMethod = Literal["auto", "exact", "stochastic"]
_VARIANCE_METHODS = ("auto", "exact", "stochastic")
# The size above which gaussx's "auto" variance turns stochastic for an
# iterative solver (`gaussx.SitePosterior.variance`, `gaussx.diag_inv`).
_EXACT_VARIANCE_MAX_SIZE = 2048
_STOCHASTIC_VARIANCE_ISSUE = "https://github.com/jejjohnson/gaussx/issues/827"

_SiteNaturals = tuple[Float[Array, " N"], Float[Array, " N"]]


def _check_variance_method(strategy: str, variance: str) -> None:
    """Reject a ``variance`` that is none of the three methods."""
    if variance not in _VARIANCE_METHODS:
        raise ValueError(
            f"{strategy}: variance must be 'auto', 'exact' or 'stochastic', "
            f"got {variance!r}."
        )


def _require_exact_variance(
    strategy: PosteriorLinearization | ExpectationPropagation,
    solver: AbstractSolverStrategy,
    size: int,
) -> None:
    """Raise unless the marginal variances can be exact.

    ``"auto"`` is exact for a dense solver, and for an iterative one up to
    N = 2048; above, and for ``"stochastic"``, the variances would be a
    probe estimate, which waits on the diagonal estimators of gaussx#827.

    Raises:
        NotImplementedError: For the stochastic branch, naming gaussx#827.
    """
    method = strategy.variance
    if method == "auto":
        exact = isinstance(solver, DenseSolver) or size <= _EXACT_VARIANCE_MAX_SIZE
        method = "exact" if exact else "stochastic"
    if method == "stochastic":
        raise NotImplementedError(
            f"{type(strategy).__name__}: the stochastic marginal variances "
            f"(variance={strategy.variance!r} with an iterative solver at "
            f"N = {size}) wait on the diagonal estimators of gaussx#827 "
            f"({_STOCHASTIC_VARIANCE_ISSUE}). Pass variance='exact', which "
            f"costs N solves per iteration in batches of batch_size."
        )


def _site_marginals(
    sites: _SiteNaturals,
    prior_mean: Float[Array, " N"],
    prior_cov: lx.AbstractLinearOperator,
    solver: AbstractSolverStrategy,
    batch_size: int,
) -> tuple[SitePosterior, Float[Array, " N"]]:
    """The site posterior of ``sites`` and its exact marginal variances."""
    information, precision = sites
    posterior = site_posterior(
        prior_mean, prior_cov, information, precision, solver=solver
    )
    return posterior, posterior.variance(method="exact", batch_size=batch_size)


def _fit_cavity_sites(
    strategy: PosteriorLinearization | ExpectationPropagation,
    prior: GPPrior,
    likelihood: Likelihood,
    y: Float[Array, " N"],
    key: Array | None,
) -> ConditionedNonGaussGP:
    """The fit PL and EP share; they differ only in ``strategy._update``.

    The first update forms its cavities from the prior marginals, with every
    site at ``(0, precision_floor)``; every later one, inside
    `argminax.solve.site_fixed_point`, from the site posterior of the
    current sites (`gaussx.site_posterior`). The evidence is
    `gaussx.site_log_evidence` at the posterior mean.
    """
    _check_scalar_latent(likelihood)
    solver = _strategy_solver(strategy, prior)
    size = prior.X.shape[0]
    _require_exact_variance(strategy, solver, size)
    # one hyperparameter draw for the operator, the mean and the predictions
    with call_context(prior.kernel):
        prior_cov = prior._prior_operator()
        prior_mean = jnp.asarray(prior.mean(prior.X))
        kernel = _fitted_kernel(prior)
    site_data = _site_data(likelihood, y, prior.X)
    dtype = prior_cov.in_structure().dtype

    # zero sites with precision ε: the first cavity is the prior
    sites = (
        jnp.zeros(size, dtype=dtype),
        jnp.full((size,), strategy.precision_floor, dtype=dtype),
    )
    first = min(strategy.max_iter, 1)
    if first:
        sites = strategy._update(
            sites, prior_mean, lx.diagonal(prior_cov), likelihood, site_data
        )
    if strategy.max_iter > first:
        fit = site_fixed_point(
            strategy._step,
            sites,
            (prior_mean, prior_cov, solver, likelihood, site_data),
            max_iter=strategy.max_iter - first,
            tol=strategy.tol,
            adjoint=strategy.adjoint,
        )
        sites, n_steps, converged = fit.sites, fit.n_iter, fit.converged
    else:
        # the first update was the whole budget, and no stopping test ran
        n_steps, converged = jnp.asarray(0), jnp.asarray(False)

    information, precision = sites
    posterior = site_posterior(
        prior_mean, prior_cov, information, precision, solver=solver
    )
    # ∑ₙ log p(yₙ ∣ mₙ) at the posterior mean
    log_lik = _log_prob_per_point_factory(likelihood)(posterior.mean, site_data).sum()
    return ConditionedNonGaussGP(
        prior=prior,
        y=y,
        site_posterior=posterior,
        log_marginal_approx=site_log_evidence(posterior, log_lik, key=key),
        n_iter=n_steps + first,
        converged=converged,
        kernel=kernel,
        batch_size=strategy.batch_size,
    )


def _cavity_step(
    strategy: PosteriorLinearization | ExpectationPropagation,
    sites: _SiteNaturals,
    args: PyTree,
) -> _SiteNaturals:
    r"""One damped sweep $T_\alpha(s)$ from the site posterior of ``sites``."""
    prior_mean, prior_cov, solver, likelihood, site_data = args
    posterior, variance = _site_marginals(
        sites, prior_mean, prior_cov, solver, strategy.batch_size
    )
    return strategy._update(sites, posterior.mean, variance, likelihood, site_data)


class PosteriorLinearization(eqx.Module):
    r"""Iterated statistical-linearization site updates.

    At each iteration: form the cavity ``q_n^\\(¬n) = N(m^c_n, v^c_n)``
    by removing the current site, then under the cavity compute
    statistical-linearization moments
    $\bar g = E_{\rm cav}[\partial_f \log p]$ and
    $\bar h = E_{\rm cav}[\partial^2_f \log p]$ via the integrator,
    update the site naturals via `gaussx.newton_update` with
    a damping factor, recompute the global posterior, repeat. This is
    PL/CVI in the sense of Adam/Garcia-Fernandez/Sarkka — equivalent
    in the Gaussian-cavity limit to taking one EP-style step but using
    derivative expectations instead of moment matching.

    The posterior of the sites is `gaussx.site_posterior` through the
    strategy's solver, and the loop is `argminax.solve.site_fixed_point`, so
    ``fit`` runs under ``jax.jit`` and ``jax.vmap`` and its gradient in the
    kernel hyperparameters is implicit at the fixed point; close over the
    prior, whose kernel registry does not survive being passed through a
    transform. The cavities need the posterior *marginal variances* at every
    iteration, not only at the end: one Cholesky of
    $A = K + \mathrm{diag}(1 / \Lambda)$ with a dense solver, $N$ solves
    with an iterative one. The reported ``log_marginal_approx`` is the
    Laplace-type approximation `gaussx.site_log_evidence` at the posterior
    mean, not the EP evidence.

    Attributes:
        integrator: Cavity integrator. Default
            ``gaussx.GaussHermiteIntegrator(order=20)``.
        max_iter: Iterations. Default ``40`` (``20`` before pyrox#552,
            whose stopping test on the posterior mean let a fit stop a few
            iterations earlier: on a 16-point Bernoulli problem the site test
            needs 23 where the mean test stopped at 20).
        damping: Step size in (0, 1]. Default ``0.5``.
        tol: Convergence tolerance: the loop stops at the first iteration
            where every site natural changed by less than
            ``0.1 * tol * (1 + |new value|)``. Default ``1e-6``.
        precision_floor: Lower bound on the diagonal precision.
        solver: The solve of $A$; ``None`` takes the prior's ``solver``.
        preconditioner: The preconditioner of an iterative ``solver``;
            ``None`` takes the prior's.
        logdet: The log-determinant of the evidence; ``None`` takes the
            prior's (and, if that is unset too, the solver's own). The three
            are composed by `gaussx.compose_solver`; none set on either side
            is `gaussx.DenseSolver`.
        variance: How the marginal variances are computed, ``"auto"``,
            ``"exact"`` or ``"stochastic"``. ``"auto"`` is exact for a dense
            solver, and for an iterative one up to $N = 2048$. The stochastic
            estimate waits on gaussx#827, so ``"stochastic"``, and ``"auto"``
            above $N = 2048$ with an iterative solver, raise
            ``NotImplementedError``.
        num_probes: Probes of the stochastic variance (gaussx#827).
        adjoint: How gradients pass through the loop; ``None`` is the
            matrix-free implicit adjoint of `argminax.solve.site_fixed_point`.
        batch_size: Solves vectorised together for the exact variances of
            an iterative solver. Default ``32``.
    """

    integrator: AbstractIntegrator = eqx.field(
        default_factory=lambda: GaussHermiteIntegrator(order=20)
    )
    max_iter: int = eqx.field(static=True, default=40)
    damping: float = eqx.field(static=True, default=0.5)
    tol: float = eqx.field(static=True, default=1e-6)
    precision_floor: float = eqx.field(static=True, default=1e-6)
    solver: AbstractSolverStrategy | None = None
    preconditioner: AbstractPreconditioner | None = None
    logdet: AbstractLogdetStrategy | None = None
    variance: _VarianceMethod = eqx.field(static=True, default="auto")
    num_probes: int = eqx.field(static=True, default=100)
    adjoint: optx.AbstractAdjoint | None = None
    batch_size: int = eqx.field(static=True, default=32)

    def __check_init__(self) -> None:
        _check_variance_method("PosteriorLinearization", self.variance)

    def _update(
        self,
        sites: _SiteNaturals,
        mean: Float[Array, " N"],
        variance: Float[Array, " N"],
        likelihood: Likelihood,
        site_data: PyTree,
    ) -> _SiteNaturals:
        """The damped PL sites from the marginals ``(mean, variance)``."""
        nat1, nat2 = sites
        log_prob_per_point = _log_prob_per_point_factory(likelihood)

        def grad_at(f_n: Float[Array, ""], y_n: PyTree) -> Float[Array, ""]:
            return jax.grad(lambda f: log_prob_per_point(f[None], _one_site(y_n))[0])(
                f_n
            )

        def hess_at(f_n: Float[Array, ""], y_n: PyTree) -> Float[Array, ""]:
            return jax.grad(
                jax.grad(lambda f: log_prob_per_point(f[None], _one_site(y_n))[0])
            )(f_n)

        # q₋ₙ(fₙ) = q(fₙ) / siteₙ(fₙ)
        cav_mean, cav_var = cavity_distribution(
            mean, variance, nat1, nat2, precision_floor=self.precision_floor
        )
        assert isinstance(cav_var, jax.Array)  # diagonal path in, diagonal out
        # ḡₙ = 𝔼_{cav}[∂f log p(yₙ | f)], h̄ₙ = 𝔼_{cav}[∂²f log p(yₙ | f)]
        E_grad = _per_site_expectation(
            self.integrator, grad_at, cav_mean, cav_var, site_data
        )
        E_hess = _per_site_expectation(
            self.integrator, hess_at, cav_mean, cav_var, site_data
        )
        # λₙ ← max(−h̄ₙ, ε),  λₙ μ_site,n ← ḡₙ + λₙ μ_cav,n; damped
        nat1_target, nat2_target = newton_update(
            cav_mean, E_grad, E_hess, precision_floor=self.precision_floor
        )
        nat1, nat2 = damped_natural_update(
            nat1, nat2, nat1_target, nat2_target, lr=self.damping
        )
        assert isinstance(nat2, jax.Array)  # pyrox sites are diagonal arrays
        return nat1, jnp.maximum(nat2, self.precision_floor)

    def _step(self, sites: _SiteNaturals, args: PyTree) -> _SiteNaturals:
        return _cavity_step(self, sites, args)

    def fit(
        self,
        prior: GPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
        *,
        key: Array | None = None,
    ) -> ConditionedNonGaussGP:
        r"""Fit the sites by posterior linearisation.

        Args:
            prior: The `GPPrior`, dense or ``matrix_free=True``.
            likelihood: A scalar-latent `Likelihood`.
            y: Observations, shape ``(N,)``.
            key: PRNG key of a stochastic log-determinant in the evidence
                (for example `gaussx.BBMMSolver`); ignored otherwise.

        Returns:
            The `ConditionedNonGaussGP` of the fitted sites.

        Raises:
            NotImplementedError: If the marginal variances would be
                stochastic (see ``variance``), which waits on gaussx#827.

        Examples:
            >>> import gaussx as gx
            >>> import jax
            >>> import jax.numpy as jnp
            >>> from pyrox_gp import RBF, BernoulliLikelihood, GPPrior
            >>> from pyrox_gp import PosteriorLinearization
            >>> X = jnp.linspace(-2.0, 2.0, 10)[:, None]
            >>> y = (X[:, 0] > 0.0).astype(X.dtype)
            >>> prior = GPPrior(kernel=RBF(), X=X)
            >>> strategy = PosteriorLinearization(solver=gx.CGSolver())
            >>> fit = jax.jit(
            ...     lambda labels: strategy.fit(prior, BernoulliLikelihood(), labels)
            ... )(y)
        """
        return _fit_cavity_sites(self, prior, likelihood, y, key)


# --- Expectation propagation --------------------------------------------


def _require_gauss_hermite(integrator: AbstractIntegrator) -> GaussHermiteIntegrator:
    """Reject a rule EP cannot use: tilted moments are Gauss-Hermite only.

    Args:
        integrator: The rule passed to an EP strategy.

    Returns:
        The same rule, narrowed.

    Raises:
        TypeError: If ``integrator`` is not a `gaussx.GaussHermiteIntegrator`.
    """
    if not isinstance(integrator, GaussHermiteIntegrator):
        raise TypeError(
            "Expectation propagation takes its tilted moments from "
            "gaussx.ep_tilted_moments, a Gauss-Hermite rule, so it needs a "
            f"gaussx.GaussHermiteIntegrator; got {type(integrator).__name__}."
        )
    return integrator


class ExpectationPropagation(eqx.Module):
    r"""Parallel expectation propagation (Minka 2001) with damping.

    Per outer iteration: for every site simultaneously, form the
    cavity, compute the *tilted* moments
    ``E_{q^c(f) p(y|f)}[1, f, f^2]`` via the integrator, match them to
    a Gaussian, and update site naturals (damped). Parallel-EP is
    embarrassingly vectorizable and converges for well-behaved
    log-concave likelihoods; for problematic models reduce ``damping``.
    On a non-log-concave likelihood EP can cycle, and ``converged`` is the
    only signal of it.

    The posterior of the sites is `gaussx.site_posterior` through the
    strategy's solver, and the loop is `argminax.solve.site_fixed_point`, so
    ``fit`` runs under ``jax.jit`` and ``jax.vmap`` and its gradient is
    implicit at the fixed point. As for `PosteriorLinearization`, the
    cavities need the marginal variances at every iteration (one Cholesky
    with a dense solver, $N$ solves with an iterative one), and
    ``log_marginal_approx`` is the Laplace-type `gaussx.site_log_evidence`
    at the EP mean, not the EP evidence with the cavity normalisers.

    Attributes:
        integrator: Cavity integrator. Default
            ``gaussx.GaussHermiteIntegrator(order=20)``. EP reads only its
            ``order``, for `gaussx.ep_tilted_moments`, so any other rule
            raises ``TypeError``.
        max_iter: Iterations. Default ``40``.
        damping: Damping in (0, 1]. Default ``0.5``.
        tol: Convergence tolerance on the site naturals, as in
            `PosteriorLinearization`. Default ``1e-6``.
        precision_floor: Lower bound on the diagonal precision.
        solver: As in `PosteriorLinearization`; ``None`` takes the prior's.
        preconditioner: As in `PosteriorLinearization`.
        logdet: As in `PosteriorLinearization`.
        variance: As in `PosteriorLinearization`.
        num_probes: As in `PosteriorLinearization`.
        adjoint: As in `PosteriorLinearization`.
        batch_size: As in `PosteriorLinearization`.
    """

    integrator: AbstractIntegrator = eqx.field(
        default_factory=lambda: GaussHermiteIntegrator(order=20)
    )
    max_iter: int = eqx.field(static=True, default=40)
    damping: float = eqx.field(static=True, default=0.5)
    tol: float = eqx.field(static=True, default=1e-6)
    precision_floor: float = eqx.field(static=True, default=1e-6)
    solver: AbstractSolverStrategy | None = None
    preconditioner: AbstractPreconditioner | None = None
    logdet: AbstractLogdetStrategy | None = None
    variance: _VarianceMethod = eqx.field(static=True, default="auto")
    num_probes: int = eqx.field(static=True, default=100)
    adjoint: optx.AbstractAdjoint | None = None
    batch_size: int = eqx.field(static=True, default=32)

    def __check_init__(self) -> None:
        _require_gauss_hermite(self.integrator)
        _check_variance_method("ExpectationPropagation", self.variance)

    def _update(
        self,
        sites: _SiteNaturals,
        mean: Float[Array, " N"],
        variance: Float[Array, " N"],
        likelihood: Likelihood,
        site_data: PyTree,
    ) -> _SiteNaturals:
        """The damped EP sites from the marginals ``(mean, variance)``."""
        nat1, nat2 = sites
        log_prob_per_point = _log_prob_per_point_factory(likelihood)
        order = _require_gauss_hermite(self.integrator).order

        def _per_site_tilted(
            m_n: Float[Array, ""],
            v_n: Float[Array, ""],
            y_n: PyTree,
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            def lp(f: Float[Array, ""]) -> Float[Array, ""]:
                return log_prob_per_point(f[None], _one_site(y_n))[0]

            return ep_tilted_moments(lp, m_n, v_n, order=order)

        # q₋ₙ(fₙ) = q(fₙ) / siteₙ(fₙ)
        cav_mean, cav_var = cavity_distribution(
            mean, variance, nat1, nat2, precision_floor=self.precision_floor
        )
        assert isinstance(cav_var, jax.Array)  # diagonal path in, diagonal out
        # The tilted moments of q₋ₙ(fₙ) p(yₙ ∣ fₙ), one site at a time.
        tilted_mean, tilted_var = jax.vmap(_per_site_tilted)(
            cav_mean, cav_var, site_data
        )
        # λₙ ← max(1 / σ̂ₙ² − 1 / σ²_cav,n, ε), and the matching information
        new_prec = jnp.reciprocal(tilted_var) - jnp.reciprocal(cav_var)
        new_prec = jnp.maximum(new_prec, self.precision_floor)
        new_nat1 = tilted_mean / tilted_var - cav_mean / cav_var
        nat1, nat2 = damped_natural_update(
            nat1, nat2, new_nat1, new_prec, lr=self.damping
        )
        assert isinstance(nat2, jax.Array)  # pyrox sites are diagonal arrays
        return nat1, jnp.maximum(nat2, self.precision_floor)

    def _step(self, sites: _SiteNaturals, args: PyTree) -> _SiteNaturals:
        return _cavity_step(self, sites, args)

    def fit(
        self,
        prior: GPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
        *,
        key: Array | None = None,
    ) -> ConditionedNonGaussGP:
        r"""Fit the sites by parallel expectation propagation.

        Args:
            prior: The `GPPrior`, dense or ``matrix_free=True``.
            likelihood: A scalar-latent `Likelihood`.
            y: Observations, shape ``(N,)``.
            key: PRNG key of a stochastic log-determinant in the evidence
                (for example `gaussx.BBMMSolver`); ignored otherwise.

        Returns:
            The `ConditionedNonGaussGP` of the fitted sites.

        Raises:
            NotImplementedError: If the marginal variances would be
                stochastic (see ``variance``), which waits on gaussx#827.

        Examples:
            >>> import jax
            >>> import jax.numpy as jnp
            >>> from pyrox_gp import RBF, ExpectationPropagation, GPPrior
            >>> from pyrox_gp import StudentTLikelihood
            >>> X = jnp.linspace(-2.0, 2.0, 10)[:, None]
            >>> y = jnp.sin(X[:, 0]).at[3].set(4.0)
            >>> prior = GPPrior(kernel=RBF(), X=X)
            >>> strategy = ExpectationPropagation(damping=0.3)
            >>> fits = jax.jit(jax.vmap(
            ...     lambda labels: strategy.fit(prior, StudentTLikelihood(), labels)
            ... ))(jnp.stack([y, -y]))
        """
        return _fit_cavity_sites(self, prior, likelihood, y, key)


# --- Quasi-Newton --------------------------------------------------------


class QuasiNewtonInference(eqx.Module):
    r"""MAP optimization via L-BFGS, Laplace covariance at convergence.

    Optimizes the unnormalized log posterior
    $\log p(y | f) - \tfrac12 (f - \mu)^\top K^{-1} (f - \mu)$
    with `optimistix.LBFGS` in the whitened coordinates $v$ of
    $f = \mu + L v$, $L L^\top = K$, where the prior term is
    $\tfrac12 v^\top v$ and the problem is far better conditioned than in
    $f$; then it forms a Laplace-style Gaussian
    approximation centered at the optimum using the exact per-point
    Hessian. The optimization is the cheap path for very high N where
    you cannot afford dense Hessians per iteration; the final Laplace
    covariance is computed once.

    For full low-rank posterior covariance from L-BFGS history (without
    the final Hessian solve) see the follow-up issue — this class
    delivers the simpler "QN optimization, Laplace covariance"
    contract.

    The L-BFGS loop is `optimistix.minimise`, so ``fit`` runs under
    ``jax.jit`` and ``jax.vmap``.

    Dense only: the whitening needs a Cholesky root of the N × N kernel
    matrix, and the prior's ``solver`` is not used, so a
    ``GPPrior(matrix_free=True)`` raises ``ValueError``, permanently
    (pyrox#549); `LaplaceInference` with an iterative solver is the
    matrix-free route (pyrox#551).

    Attributes:
        max_iter: L-BFGS iterations. Default ``50``.
        tol: The ``rtol`` and ``atol`` of `optimistix.LBFGS`, which stops
            when both the step in ``v`` and the change in the objective are
            within ``atol + rtol * |.|``. Default ``1e-6``.
        precision_floor: Lower bound on the diagonal precision.
    """

    max_iter: int = eqx.field(static=True, default=50)
    tol: float = eqx.field(static=True, default=1e-6)
    precision_floor: float = eqx.field(static=True, default=1e-6)

    def fit(
        self,
        prior: GPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
        *,
        key: Array | None = None,
    ) -> ConditionedNonGaussGP:
        """Fit by L-BFGS to the mode, then the Laplace Gaussian there.

        Args:
            prior: A dense `GPPrior`.
            likelihood: A scalar-latent `Likelihood`.
            y: Observations, shape ``(N,)``.
            key: Accepted so that every strategy takes the keyword of
                `GPPrior.condition_nongauss`; the evidence here is exact,
                so it is ignored.

        Returns:
            The `ConditionedNonGaussGP` at the mode.

        Raises:
            ValueError: For ``GPPrior(matrix_free=True)``.
        """
        del key  # the dense evidence needs no probes
        _reject_matrix_free(prior)
        _check_scalar_latent(likelihood)
        # one hyperparameter draw for K, the mean and the predictions
        with call_context(prior.kernel):
            K = _prior_K(prior)
            prior_mean = jnp.asarray(prior.mean(prior.X))
            kernel = _fitted_kernel(prior)
        log_prob_per_point = _log_prob_per_point_factory(likelihood)
        sites = _site_data(likelihood, y, prior.X)
        L_K = _psd_safe_cholesky(K)

        def latent(v: Float[Array, " N"]) -> Float[Array, " N"]:
            # f = μ + L v, the whitened coordinates of the prior
            return prior_mean + einx.dot("i j, j -> i", L_K, v)

        def neg_log_post(v: Float[Array, " N"], args: None) -> Float[Array, ""]:
            del args
            # −log p(y | f) + ½ vᵀv, the negative log posterior in v
            return -log_prob_per_point(latent(v), sites).sum() + 0.5 * jnp.dot(v, v)

        # optimistix keeps the L-BFGS history in the default float dtype, so
        # float32 data under x64 runs the search in float64 and casts back.
        dtype = jnp.promote_types(prior_mean.dtype, jnp.result_type(float))
        solution = optx.minimise(
            neg_log_post,
            optx.LBFGS(rtol=self.tol, atol=self.tol),
            jnp.zeros(prior_mean.shape, dtype),
            max_steps=self.max_iter,
            throw=False,
        )
        mode = latent(solution.value).astype(prior_mean.dtype)
        # the Newton sites at the mode: λ = max(−h, ε), λ μ_site = g + λ f̂
        g, h = _per_point_grad_hess(log_prob_per_point, mode, sites)
        information, precision = newton_update(
            mode, g, h, precision_floor=self.precision_floor
        )
        posterior = site_posterior(
            prior_mean,
            _psd_operator(K),
            information,
            precision,
            solver=DenseSolver(),
        )
        # ∑ₙ log p(yₙ ∣ mₙ) at the posterior mean of the sites
        log_lik = log_prob_per_point(posterior.mean, sites).sum()
        return ConditionedNonGaussGP(
            prior=prior,
            y=y,
            site_posterior=posterior,
            log_marginal_approx=site_log_evidence(posterior, log_lik),
            n_iter=jnp.asarray(solution.stats["num_steps"]),
            converged=solution.result == optx.RESULTS.successful,
            kernel=kernel,
        )


# --- module-level conveniences ------------------------------------------


def gp_laplace_factor(
    name: str,
    prior: GPPrior,
    likelihood: Likelihood,
    y: Float[Array, " N"],
    *,
    strategy: LaplaceInference | GaussNewtonInference | None = None,
    key: Array | None = None,
) -> None:
    r"""Register the Laplace approximate GP evidence with NumPyro.

    Adds ``numpyro.factor(name, strategy.fit(prior, likelihood, y)
    .log_marginal_approx)``, the non-Gaussian analogue of `gp_factor`: the
    latent function is marginalised by the Laplace approximation at the
    mode $\hat f$,

    $$
    \log p(y \mid \theta) \approx \sum_n \log p(y_n \mid \hat f_n)
      - \tfrac12 (\hat f - \mu)^\top K^{-1} (\hat f - \mu)
      - \tfrac12 \log \lvert I + \Lambda^{1/2} K \Lambda^{1/2} \rvert ,
    $$

    with $\Lambda$ the site precisions. The Newton loop is
    `argminax.solve.laplace_mode_covariance` with an implicitly
    differentiable mode, so the factor traces under NUTS and its gradient in
    the kernel hyperparameters $\theta$ flows through $\hat f$ without
    unrolling the loop. Registers the one factor site ``name``, plus the
    kernel's hyperparameter sites, drawn once.

    The value is that of a fit that may not have converged; check
    ``strategy.fit(...).converged`` at the hyperparameters you care about.
    The strategy's ``solver``, ``preconditioner`` and ``logdet`` (or the
    prior's) apply, so a matrix-free prior with an iterative solver never
    forms the kernel matrix (pyrox#551); with a stochastic log-determinant
    the factor is an estimate, and a fixed ``key`` makes NUTS target a fixed
    approximation of the posterior.

    Args:
        name: The factor site name.
        prior: The `GPPrior`.
        likelihood: A scalar-latent `Likelihood`.
        y: Observations, shape ``(N,)``.
        strategy: `LaplaceInference` (the default) or `GaussNewtonInference`.
        key: PRNG key of a stochastic log-determinant in the evidence,
            passed to ``strategy.fit``; ``None`` uses the solver's own seed.

    Examples:
        >>> import jax.numpy as jnp
        >>> import numpyro.distributions as dist
        >>> from numpyro import handlers
        >>> from pyrox_gp import RBF, BernoulliLikelihood, GPPrior, gp_laplace_factor
        >>> X = jnp.linspace(-2.0, 2.0, 10)[:, None]
        >>> y = (X[:, 0] > 0.0).astype(X.dtype)
        >>> kernel = RBF()
        >>> kernel.set_prior("lengthscale", dist.LogNormal(0.0, 0.5))
        >>> prior = GPPrior(kernel=kernel, X=X)
        >>> def model():
        ...     gp_laplace_factor("y", prior, BernoulliLikelihood(), y)
        >>> trace = handlers.trace(handlers.seed(model, 0)).get_trace()
        >>> sorted(trace)
        ['RBF.lengthscale', 'RBF.variance', 'y']
    """
    strategy = LaplaceInference() if strategy is None else strategy
    fit = strategy.fit(prior, likelihood, y, key=key)
    numpyro.factor(name, fit.log_marginal_approx)
