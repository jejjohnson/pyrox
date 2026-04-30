"""Advanced GP inference strategies for non-conjugate likelihoods.

All strategies in this module produce a Gaussian approximation to the
posterior ``q(f) = N(m, V)`` over the latent function values at the
training inputs, given a non-conjugate likelihood ``p(y | f)``. They
share the same site-based view: each likelihood factor contributes a
diagonal Gaussian *site* with natural parameters
:math:`(\\lambda^{(1)}, \\Lambda^{(2)}) = (J - H m, -H)`. Strategies
differ only in how the per-site curvature ``H`` and effective gradient
``J`` are obtained:

================================  =======================================
Strategy                          Curvature ``H`` from
================================  =======================================
:class:`LaplaceInference`         exact ``-d^2 log p / df^2`` at the mode
:class:`GaussNewtonInference`     generalized Gauss-Newton
                                  (positive-semidefinite by construction)
:class:`PosteriorLinearization`   statistical linearization under cavity
                                  via cubature
:class:`ExpectationPropagation`   moment matching against the tilted
                                  distribution per site
:class:`QuasiNewtonInference`     L-BFGS optimization to MAP, exact
                                  Hessian at convergence
================================  =======================================

All five accept a :class:`pyrox.gp.GPPrior` and a scalar-latent
:class:`pyrox.gp.Likelihood`, and return an
:class:`NonGaussConditionedGP` that quacks like
:class:`pyrox.gp.ConditionedGP` (``predict``, ``predict_mean``,
``predict_var``).

Everything here is pure JAX and ``equinox``-compatible (jit / grad /
vmap). The heavy lifting — cavity arithmetic, natural-parameter
updates, GGN curvature, Cholesky solves — comes from
:mod:`gaussx`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from pyrox.gp._context import _kernel_context
from pyrox.gp._integrators import GaussHermite
from pyrox.gp._protocols import Integrator, Likelihood


if TYPE_CHECKING:
    from pyrox.gp._models import GPPrior


# --- result type ---------------------------------------------------------


class NonGaussConditionedGP(eqx.Module):
    """GP conditioned on a non-Gaussian likelihood via an advanced strategy.

    Equivalent role to :class:`pyrox.gp.ConditionedGP` but the
    posterior over training latents is a generic
    ``q(f) = N(q_mean, q_cov)`` rather than a Gaussian-likelihood
    closed form. Predictions at test inputs use the standard
    *site-as-pseudo-observation* trick: any site-based Gaussian
    approximation of ``p(f | y)`` looks identical to a Gaussian-
    likelihood regression with synthetic per-point noise variance
    :math:`\\sigma_n^2 = 1/\\Lambda_n^{(2)}` and synthetic targets
    :math:`\\tilde y_n = \\lambda_n^{(1)} / \\Lambda_n^{(2)}` (in the
    zero-mean prior frame). Predictions reconstruct ``K_reg = K +
    diag(1 / Lambda)`` and Cholesky-factorize it on each call (the
    full-Cholesky cost is ``O(N^3)`` per ``predict`` invocation; the
    cross-covariance contributions are ``O(M N)``). Caching the solve
    on the module would require freezing the kernel hyperparameters at
    fit time and is intentionally not done here so prior'd-kernel
    workflows keep resampling correctly.

    Attributes:
        prior: The :class:`GPPrior`.
        y: Training targets (kept for round-trip / diagnostics).
        site_nat1: Diagonal site naturals
            :math:`\\lambda^{(1)} \\in \\mathbb{R}^N`.
        site_nat2: Diagonal site precisions
            :math:`\\Lambda^{(2)} \\in \\mathbb{R}^N` (positive).
        q_mean: Posterior mean over training latents.
        q_var: Marginal posterior variance per training point.
        log_marginal_approx: Approximate log marginal likelihood
            (the scalar each strategy reports — interpretation is
            strategy-specific; see the per-class docstring).
        n_iter: Iterations used by the strategy.
        converged: Whether convergence tolerance was met.
    """

    prior: GPPrior
    y: Float[Array, " N"]
    site_nat1: Float[Array, " N"]
    site_nat2: Float[Array, " N"]
    q_mean: Float[Array, " N"]
    q_var: Float[Array, " N"]
    log_marginal_approx: Float[Array, ""]
    n_iter: int = eqx.field(static=True)
    converged: bool = eqx.field(static=True)

    def _pseudo_factor(self) -> Float[Array, "N N"]:
        """Cholesky of ``K + diag(1/Λ + jitter)`` via :func:`_stable_cholesky`."""
        K = self.prior.kernel(self.prior.X, self.prior.X)
        N = K.shape[0]
        eye = jnp.eye(N, dtype=K.dtype)
        K_reg = K + (jnp.reciprocal(self.site_nat2) + self.prior.jitter)[:, None] * eye
        return _stable_cholesky(K_reg)

    def predict_mean(self, X_star: Float[Array, "M D"]) -> Float[Array, " M"]:
        r""":math:`\mu_* = \mu(X_*) + K_{*f}\,\alpha` with
        ``alpha`` derived from the site naturals."""
        with _kernel_context(self.prior.kernel):
            L = self._pseudo_factor()
            K_cross = self.prior.kernel(X_star, self.prior.X)
            prior_mean_train = self.prior.mean(self.prior.X)
        # Effective synthetic targets in the centered (zero-mean) frame.
        y_tilde = self.site_nat1 / self.site_nat2
        residual = y_tilde - prior_mean_train
        alpha = jax.scipy.linalg.cho_solve((L, True), residual)
        return self.prior.mean(X_star) + K_cross @ alpha

    def predict_var(self, X_star: Float[Array, "M D"]) -> Float[Array, " M"]:
        r""":math:`\Sigma_{**} - K_{*f} (K + \mathrm{diag}(1/\Lambda))^{-1} K_{f*}`."""
        with _kernel_context(self.prior.kernel):
            L = self._pseudo_factor()
            K_cross = self.prior.kernel(X_star, self.prior.X)
            K_diag = self.prior.kernel.diag(X_star)
        v = jax.scipy.linalg.solve_triangular(L, K_cross.T, lower=True)
        return jnp.maximum(K_diag - jnp.sum(v * v, axis=0), 0.0)

    def predict(
        self, X_star: Float[Array, "M D"]
    ) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
        """Joint mean / marginal-variance prediction at ``X_star``.

        Both kernel evaluations share a single kernel context so
        Pattern B / C kernels with prior'd hyperparameters resample
        once and produce a self-consistent ``(mean, var)`` pair.
        """
        with _kernel_context(self.prior.kernel):
            L = self._pseudo_factor()
            K_cross = self.prior.kernel(X_star, self.prior.X)
            K_diag = self.prior.kernel.diag(X_star)
            prior_mean_train = self.prior.mean(self.prior.X)
            prior_mean_test = self.prior.mean(X_star)
        y_tilde = self.site_nat1 / self.site_nat2
        residual = y_tilde - prior_mean_train
        alpha = jax.scipy.linalg.cho_solve((L, True), residual)
        mean = prior_mean_test + K_cross @ alpha
        v = jax.scipy.linalg.solve_triangular(L, K_cross.T, lower=True)
        var = jnp.maximum(K_diag - jnp.sum(v * v, axis=0), 0.0)
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


def _prior_K(prior: GPPrior) -> Float[Array, "N N"]:
    with _kernel_context(prior.kernel):
        K = prior.kernel(prior.X, prior.X)
    return K + prior.jitter * jnp.eye(K.shape[0], dtype=K.dtype)


def _posterior_from_diag_sites(
    K: Float[Array, "N N"],
    nat1: Float[Array, " N"],
    nat2: Float[Array, " N"],
    prior_mean: Float[Array, " N"],
) -> tuple[Float[Array, " N"], Float[Array, "N N"], Float[Array, " N"]]:
    r"""Compute ``q(f) = N(m, V)`` from diagonal site naturals.

    Sites contribute a synthetic Gaussian likelihood with mean
    ``y_tilde = nat1 / nat2`` and variance ``1/nat2``. The posterior
    mean is
    :math:`m = \mu + K (K + \mathrm{diag}(1/\Lambda))^{-1} (y_{\rm tilde} - \mu)`,
    and the posterior covariance is
    :math:`V = K - K (K + \mathrm{diag}(1/\Lambda))^{-1} K`.

    Returns ``(q_mean, q_cov, q_var)``.
    """
    N = K.shape[0]
    eye = jnp.eye(N, dtype=K.dtype)
    sigma2 = jnp.reciprocal(nat2)
    K_reg = K + sigma2[:, None] * eye
    L = _stable_cholesky(K_reg)

    y_tilde = nat1 / nat2
    residual = y_tilde - prior_mean
    alpha = jax.scipy.linalg.cho_solve((L, True), residual)
    q_mean = prior_mean + K @ alpha

    # V = K - K (K + diag(sigma2))^-1 K
    M = jax.scipy.linalg.cho_solve((L, True), K)
    V = K - K @ M
    V = 0.5 * (V + V.T)
    q_var = jnp.diag(V)
    return q_mean, V, q_var


def _per_point_grad_hess(
    log_prob_per_point: Callable[
        [Float[Array, " N"], Float[Array, " N"]], Float[Array, " N"]
    ],
    f: Float[Array, " N"],
    y: Float[Array, " N"],
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    r"""Per-point ``g_n = d log p_n / df_n`` and ``h_n = d^2 log p_n / df_n^2``.

    ``log_prob_per_point`` must be vectorized over ``f`` and return one
    log-density per observation (no sum). For scalar likelihoods the
    Hessian is diagonal in ``f`` so per-point second derivatives suffice.
    """

    def per_n(f_n: Float[Array, ""], y_n: Float[Array, ""]) -> Float[Array, ""]:
        return log_prob_per_point(f_n[None], y_n[None])[0]

    g = jax.vmap(jax.grad(per_n))(f, y)
    h = jax.vmap(jax.grad(jax.grad(per_n)))(f, y)
    return g, h


def _stable_cholesky(
    M: Float[Array, "N N"], floor: float = 1e-3
) -> Float[Array, "N N"]:
    """Cholesky with one-shot adaptive jitter fallback.

    Try the symmetrized matrix as-is; if any entry of the Cholesky is
    ``NaN`` (matrix is not numerically PD in the requested precision),
    retry with ``floor * I`` added. Float32 + densely-packed kernel
    inputs is the realistic failure case this guards against.
    """
    M = 0.5 * (M + M.T)
    L = jnp.linalg.cholesky(M)
    return jax.lax.cond(
        jnp.any(jnp.isnan(L)),
        lambda: jnp.linalg.cholesky(M + floor * jnp.eye(M.shape[0], dtype=M.dtype)),
        lambda: L,
    )


def _ep_tilted_moments(
    lp_per_n: Callable[[Float[Array, ""], Float[Array, ""]], Float[Array, ""]],
    y: Float[Array, " N"],
    cav_mean: Float[Array, " N"],
    cav_var: Float[Array, " N"],
    deg: int = 20,
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    r"""Numerically stable tilted moments for EP.

    Returns ``(tilted_mean, tilted_var)`` per site, where the tilted
    distribution is :math:`q_{\rm cav}(f) p(y \mid f)`. Computes
    :math:`\log p` at the cavity-rescaled Gauss-Hermite nodes once,
    subtracts the per-site max to avoid overflow, then forms moment
    ratios that are independent of the per-site shift.
    """
    from gaussx import gauss_hermite_points

    nodes, weights = gauss_hermite_points(deg, dim=1)
    x = nodes[:, 0]
    log_w = jnp.log(weights / jnp.sqrt(2.0 * jnp.pi))
    std = jnp.sqrt(cav_var)
    f_grid = cav_mean[None, :] + std[None, :] * x[:, None]
    log_p = jax.vmap(lambda f_row: jax.vmap(lp_per_n)(f_row, y))(f_grid)
    log_unnorm = log_p + log_w[:, None]
    shift = jnp.max(log_unnorm, axis=0, keepdims=True)
    w_unnorm = jnp.exp(log_unnorm - shift)
    Z_unnorm = jnp.sum(w_unnorm, axis=0)
    Z_safe = jnp.maximum(Z_unnorm, 1e-30)
    tilted_mean = jnp.sum(w_unnorm * f_grid, axis=0) / Z_safe
    tilted_sq = jnp.sum(w_unnorm * f_grid * f_grid, axis=0) / Z_safe
    tilted_var = jnp.maximum(tilted_sq - tilted_mean**2, 1e-12)
    return tilted_mean, tilted_var


def _laplace_log_marginal(
    log_prob_per_point: Callable[
        [Float[Array, " N"], Float[Array, " N"]], Float[Array, " N"]
    ],
    f: Float[Array, " N"],
    y: Float[Array, " N"],
    prior_mean: Float[Array, " N"],
    K: Float[Array, "N N"],
    Lam: Float[Array, " N"],
) -> Float[Array, ""]:
    r"""Standard Laplace log-marginal-likelihood approximation.

    Computes
    :math:`\log p(y) \approx \log p(y \mid \hat f)
    - \tfrac12 (\hat f - \mu)^\top K^{-1} (\hat f - \mu)
    - \tfrac12 \log |I + K \Lambda|`

    with both Cholesky factorizations gated through :func:`_stable_cholesky`
    to survive near-singular ``K`` under float32 + dense data.
    """
    ll = log_prob_per_point(f, y).sum()
    residual = f - prior_mean
    L_K = _stable_cholesky(K)
    alpha = jax.scipy.linalg.cho_solve((L_K, True), residual)
    quad = 0.5 * jnp.dot(residual, alpha)
    sqrt_lam = jnp.sqrt(Lam)
    B = jnp.eye(K.shape[0], dtype=K.dtype) + sqrt_lam[:, None] * K * sqrt_lam[None, :]
    L_B = _stable_cholesky(B)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L_B)))
    return ll - quad - 0.5 * logdet


def _log_prob_per_point_factory(
    lik: Likelihood,
) -> Callable[[Float[Array, " N"], Float[Array, " N"]], Float[Array, " N"]]:
    """Wrap ``lik.log_prob`` so it returns per-point log-densities.

    pyrox's ``Likelihood.log_prob`` is summed over points by convention.
    For per-site curvature we need the unsummed per-point quantity,
    which we compute via ``vmap`` over the scalar log-density.
    """

    def per_one(f_n: Float[Array, ""], y_n: Float[Array, ""]) -> Float[Array, ""]:
        return lik.log_prob(f_n[None], y_n[None])

    def per_point(f: Float[Array, " N"], y: Float[Array, " N"]) -> Float[Array, " N"]:
        return jax.vmap(per_one)(f, y)

    return per_point


# --- Laplace -------------------------------------------------------------


class LaplaceInference(eqx.Module):
    r"""Laplace approximation via Newton iteration on the log posterior.

    Iterates the standard GP-Laplace fixed-point loop (Rasmussen &
    Williams Algorithm 3.1): at each iteration evaluate per-point
    gradient ``g`` and Hessian-diag ``h`` of ``log p(y | f)``, form the
    Newton update with site precision :math:`\Lambda = -h` (clipped to
    a small positive floor for numerical safety), and recompute ``f``
    as the mean of the implied Gaussian posterior.

    The reported ``log_marginal_approx`` is the standard Laplace
    log-marginal-likelihood approximation
    :math:`\log p(y) \approx \log p(y | \hat f) - \tfrac12 \hat f^\top
    K^{-1} \hat f - \tfrac12 \log |I + K \Lambda|`.

    Args:
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
    """

    max_iter: int = eqx.field(static=True, default=20)
    tol: float = eqx.field(static=True, default=1e-6)
    damping: float = eqx.field(static=True, default=1.0)
    precision_floor: float = eqx.field(static=True, default=1e-6)

    def fit(
        self,
        prior: GPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
    ) -> NonGaussConditionedGP:
        _check_scalar_latent(likelihood)
        K = _prior_K(prior)
        prior_mean = prior.mean(prior.X)
        log_prob_per_point = _log_prob_per_point_factory(likelihood)

        f = jnp.asarray(prior_mean)
        converged = False
        n_iter = 0
        for it in range(self.max_iter):
            g, h = _per_point_grad_hess(log_prob_per_point, f, y)
            # Site precision Λ = -h (positive for log-concave likelihoods).
            Lam = jnp.maximum(-h, self.precision_floor)
            nat1 = g + Lam * f
            f_newton, _, _ = _posterior_from_diag_sites(K, nat1, Lam, prior_mean)
            f_new = (1.0 - self.damping) * f + self.damping * f_newton
            delta = jnp.max(jnp.abs(f_new - f))
            f = f_new
            n_iter = it + 1
            if delta < self.tol:
                converged = True
                break

        # Final site naturals at convergence.
        g, h = _per_point_grad_hess(log_prob_per_point, f, y)
        Lam = jnp.maximum(-h, self.precision_floor)
        nat1 = g + Lam * f
        q_mean, _, q_var = _posterior_from_diag_sites(K, nat1, Lam, prior_mean)

        log_marg = _laplace_log_marginal(log_prob_per_point, f, y, prior_mean, K, Lam)

        return NonGaussConditionedGP(  # ty: ignore[invalid-return-type]
            prior=prior,
            y=y,
            site_nat1=nat1,
            site_nat2=Lam,
            q_mean=q_mean,
            q_var=q_var,
            log_marginal_approx=log_marg,
            n_iter=n_iter,
            converged=converged,
        )


# --- Gauss-Newton --------------------------------------------------------


class GaussNewtonInference(eqx.Module):
    r"""Gauss-Newton inference: Newton loop with PSD-projected curvature.

    Identical to :class:`LaplaceInference` for log-concave likelihoods,
    where ``-d^2 log p / df^2`` is already positive. For non-log-concave
    likelihoods (e.g. :class:`StudentTLikelihood`, where the Hessian
    becomes positive in the tails) GN aggressively floors the curvature
    to a strictly-positive value via ``precision_floor``, guaranteeing a
    PSD site precision and stable Newton steps. Laplace uses the same
    floor but typically with a smaller default.

    For Bernoulli / Poisson the Fisher information equals the negative
    Hessian, so GGN ≡ Laplace; for StudentT the floor matters.

    Args:
        max_iter: Iterations. Default ``20``.
        tol: ``inf``-norm convergence tolerance on ``f``. Default ``1e-6``.
        damping: Step-size in (0, 1] applied to each Newton update.
            Default ``1.0`` (full Newton step). Drop below 1 for
            non-log-concave likelihoods where pure Newton oscillates.
        precision_floor: Lower bound on the diagonal precision. Default
            ``1e-3`` — larger than :class:`LaplaceInference` to ensure
            stable updates on non-log-concave likelihoods.
    """

    max_iter: int = eqx.field(static=True, default=20)
    tol: float = eqx.field(static=True, default=1e-6)
    damping: float = eqx.field(static=True, default=1.0)
    precision_floor: float = eqx.field(static=True, default=1e-3)

    def fit(
        self,
        prior: GPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
    ) -> NonGaussConditionedGP:
        _check_scalar_latent(likelihood)
        K = _prior_K(prior)
        prior_mean = prior.mean(prior.X)
        log_prob_per_point = _log_prob_per_point_factory(likelihood)

        f = jnp.asarray(prior_mean)
        converged = False
        n_iter = 0
        for it in range(self.max_iter):
            g, h = _per_point_grad_hess(log_prob_per_point, f, y)
            # PSD curvature: clip the negative Hessian to a strictly
            # positive floor so the Newton step is always well-defined
            # even when ``-h`` goes negative (StudentT tails).
            Lam = jnp.maximum(-h, self.precision_floor)
            nat1 = g + Lam * f
            f_newton, _, _ = _posterior_from_diag_sites(K, nat1, Lam, prior_mean)
            f_new = (1.0 - self.damping) * f + self.damping * f_newton
            delta = jnp.max(jnp.abs(f_new - f))
            f = f_new
            n_iter = it + 1
            if delta < self.tol:
                converged = True
                break

        g, h = _per_point_grad_hess(log_prob_per_point, f, y)
        Lam = jnp.maximum(-h, self.precision_floor)
        nat1 = g + Lam * f
        q_mean, _, q_var = _posterior_from_diag_sites(K, nat1, Lam, prior_mean)

        log_marg = _laplace_log_marginal(log_prob_per_point, f, y, prior_mean, K, Lam)

        return NonGaussConditionedGP(  # ty: ignore[invalid-return-type]
            prior=prior,
            y=y,
            site_nat1=nat1,
            site_nat2=Lam,
            q_mean=q_mean,
            q_var=q_var,
            log_marginal_approx=log_marg,
            n_iter=n_iter,
            converged=converged,
        )


# --- Posterior linearization --------------------------------------------


class PosteriorLinearization(eqx.Module):
    r"""Iterated statistical-linearization site updates.

    At each iteration: form the cavity ``q_n^\\(¬n) = N(m^c_n, v^c_n)``
    by removing the current site, then under the cavity compute
    statistical-linearization moments
    :math:`\bar g = E_{\rm cav}[\partial_f \log p]` and
    :math:`\bar h = E_{\rm cav}[\partial^2_f \log p]` via the integrator,
    update the site naturals via :func:`gaussx.blr_diag_update` with
    a damping factor, recompute the global posterior, repeat. This is
    PL/CVI in the sense of Adam/Garcia-Fernandez/Sarkka — equivalent
    in the Gaussian-cavity limit to taking one EP-style step but using
    derivative expectations instead of moment matching.

    Args:
        integrator: Cavity integrator. Default ``GaussHermite(deg=20)``.
        max_iter: Iterations. Default ``20``.
        damping: Step size in (0, 1]. Default ``0.5``.
        tol: ``inf``-norm convergence on the posterior mean.
        precision_floor: Lower bound on the diagonal precision.
    """

    integrator: Integrator = eqx.field(default_factory=lambda: GaussHermite(deg=20))
    max_iter: int = eqx.field(static=True, default=20)
    damping: float = eqx.field(static=True, default=0.5)
    tol: float = eqx.field(static=True, default=1e-6)
    precision_floor: float = eqx.field(static=True, default=1e-6)

    def fit(
        self,
        prior: GPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
    ) -> NonGaussConditionedGP:
        _check_scalar_latent(likelihood)
        K = _prior_K(prior)
        prior_mean = prior.mean(prior.X)
        log_prob_per_point = _log_prob_per_point_factory(likelihood)

        N = K.shape[0]
        # Initialize sites at zero — posterior == prior.
        nat1 = jnp.zeros(N, dtype=K.dtype)
        nat2 = jnp.full((N,), self.precision_floor, dtype=K.dtype)
        q_mean = jnp.asarray(prior_mean)
        q_var = jnp.diag(K)

        def grad_at(f_n: Float[Array, ""], y_n: Float[Array, ""]) -> Float[Array, ""]:
            return jax.grad(lambda f: log_prob_per_point(f[None], y_n[None])[0])(f_n)

        def hess_at(f_n: Float[Array, ""], y_n: Float[Array, ""]) -> Float[Array, ""]:
            return jax.grad(
                jax.grad(lambda f: log_prob_per_point(f[None], y_n[None])[0])
            )(f_n)

        converged = False
        n_iter = 0
        for it in range(self.max_iter):
            # Cavity for diagonal sites: q_n / site_n.
            cav_prec = jnp.maximum(jnp.reciprocal(q_var) - nat2, self.precision_floor)
            cav_var = jnp.reciprocal(cav_prec)
            cav_mean = cav_var * (q_mean / q_var - nat1)

            # Statistical-linearization moments under the cavity.
            grad_per = lambda f: jax.vmap(grad_at)(f, y)
            hess_per = lambda f: jax.vmap(hess_at)(f, y)
            E_grad = self.integrator.integrate(grad_per, cav_mean, cav_var)
            E_hess = self.integrator.integrate(hess_per, cav_mean, cav_var)

            # Site update via BLR (diag): nat1_new = grad - H mu, nat2_new = -H,
            # damped.
            H = jnp.maximum(-E_hess, self.precision_floor)
            nat1_target = E_grad + H * cav_mean
            nat2_target = H
            nat1 = (1.0 - self.damping) * nat1 + self.damping * nat1_target
            nat2 = (1.0 - self.damping) * nat2 + self.damping * nat2_target
            nat2 = jnp.maximum(nat2, self.precision_floor)

            q_mean_new, _, q_var = _posterior_from_diag_sites(K, nat1, nat2, prior_mean)
            delta = jnp.max(jnp.abs(q_mean_new - q_mean))
            q_mean = q_mean_new
            n_iter = it + 1
            if delta < self.tol:
                converged = True
                break

        # Approximate log marginal via the same Laplace-style identity
        # at the converged ``q_mean`` (acceptable for small N; not
        # guaranteed PL-consistent — strategies expose their own marginal
        # when the user needs strategy-specific values).
        log_marg = _laplace_log_marginal(
            log_prob_per_point, q_mean, y, prior_mean, K, nat2
        )

        return NonGaussConditionedGP(  # ty: ignore[invalid-return-type]
            prior=prior,
            y=y,
            site_nat1=nat1,
            site_nat2=nat2,
            q_mean=q_mean,
            q_var=q_var,
            log_marginal_approx=log_marg,
            n_iter=n_iter,
            converged=converged,
        )


# --- Expectation propagation --------------------------------------------


class ExpectationPropagation(eqx.Module):
    r"""Parallel expectation propagation (Minka 2001) with damping.

    Per outer iteration: for every site simultaneously, form the
    cavity, compute the *tilted* moments
    ``E_{q^c(f) p(y|f)}[1, f, f^2]`` via the integrator, match them to
    a Gaussian, and update site naturals (damped). Parallel-EP is
    embarrassingly vectorizable and converges for well-behaved
    log-concave likelihoods; for problematic models reduce ``damping``.

    Args:
        integrator: Cavity integrator. Default ``GaussHermite(deg=20)``.
        max_iter: Iterations. Default ``40``.
        damping: Damping in (0, 1]. Default ``0.5``.
        tol: ``inf``-norm convergence on the posterior mean.
        precision_floor: Lower bound on the diagonal precision.
    """

    integrator: Integrator = eqx.field(default_factory=lambda: GaussHermite(deg=20))
    max_iter: int = eqx.field(static=True, default=40)
    damping: float = eqx.field(static=True, default=0.5)
    tol: float = eqx.field(static=True, default=1e-6)
    precision_floor: float = eqx.field(static=True, default=1e-6)

    def fit(
        self,
        prior: GPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
    ) -> NonGaussConditionedGP:
        _check_scalar_latent(likelihood)
        K = _prior_K(prior)
        prior_mean = prior.mean(prior.X)
        log_prob_per_point = _log_prob_per_point_factory(likelihood)

        N = K.shape[0]
        nat1 = jnp.zeros(N, dtype=K.dtype)
        nat2 = jnp.full((N,), self.precision_floor, dtype=K.dtype)
        q_mean = jnp.asarray(prior_mean)
        q_var = jnp.diag(K)

        def lp(f_n: Float[Array, ""], y_n: Float[Array, ""]) -> Float[Array, ""]:
            return log_prob_per_point(f_n[None], y_n[None])[0]

        converged = False
        n_iter = 0
        for it in range(self.max_iter):
            cav_prec = jnp.maximum(jnp.reciprocal(q_var) - nat2, self.precision_floor)
            cav_var = jnp.reciprocal(cav_prec)
            cav_mean = cav_var * (q_mean / q_var - nat1)

            # Tilted moments computed in log-space for numerical
            # stability. With Gauss-Hermite nodes ``x_i`` and weights
            # ``w_i' = w_i / sqrt(2*pi)``, the tilted moments per site
            # are weighted sums of ``exp(log p_n(m_n + s_n x_i))``. We
            # subtract the per-site max log-density before exponentiating
            # (the standard log-sum-exp trick) so neither overflow nor
            # underflow can corrupt the moment ratios. Z itself does not
            # enter the marginal-likelihood approximation below — only
            # the moment ratios matter — so the per-site shifts cancel.
            tilted_mean, tilted_var = _ep_tilted_moments(
                lp, y, cav_mean, cav_var, deg=getattr(self.integrator, "deg", 20)
            )

            # New site naturals from matched moments minus the cavity.
            new_prec = jnp.reciprocal(tilted_var) - cav_prec
            new_prec = jnp.maximum(new_prec, self.precision_floor)
            new_nat1 = tilted_mean / tilted_var - cav_mean / cav_var
            # Damping in natural-parameter space.
            nat1 = (1.0 - self.damping) * nat1 + self.damping * new_nat1
            nat2 = (1.0 - self.damping) * nat2 + self.damping * new_prec
            nat2 = jnp.maximum(nat2, self.precision_floor)

            q_mean_new, _, q_var = _posterior_from_diag_sites(K, nat1, nat2, prior_mean)
            delta = jnp.max(jnp.abs(q_mean_new - q_mean))
            q_mean = q_mean_new
            n_iter = it + 1
            if delta < self.tol:
                converged = True
                break

        # Approximate marginal via Laplace identity at the EP mean.
        log_marg = _laplace_log_marginal(
            log_prob_per_point, q_mean, y, prior_mean, K, nat2
        )

        return NonGaussConditionedGP(  # ty: ignore[invalid-return-type]
            prior=prior,
            y=y,
            site_nat1=nat1,
            site_nat2=nat2,
            q_mean=q_mean,
            q_var=q_var,
            log_marginal_approx=log_marg,
            n_iter=n_iter,
            converged=converged,
        )


# --- Quasi-Newton --------------------------------------------------------


class QuasiNewtonInference(eqx.Module):
    r"""MAP optimization via L-BFGS, Laplace covariance at convergence.

    Optimizes the unnormalized log posterior
    :math:`\log p(y | f) - \tfrac12 (f - \mu)^\top K^{-1} (f - \mu)`
    with ``optax.lbfgs``, then forms a Laplace-style Gaussian
    approximation centered at the optimum using the exact per-point
    Hessian. The optimization is the cheap path for very high N where
    you cannot afford dense Hessians per iteration; the final Laplace
    covariance is computed once.

    For full low-rank posterior covariance from L-BFGS history (without
    the final Hessian solve) see the follow-up issue — this class
    delivers the simpler "QN optimization, Laplace covariance"
    contract.

    Args:
        max_iter: L-BFGS iterations. Default ``50``.
        tol: Gradient-norm tolerance. Default ``1e-6``.
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
    ) -> NonGaussConditionedGP:
        import optax

        _check_scalar_latent(likelihood)
        K = _prior_K(prior)
        prior_mean = prior.mean(prior.X)
        log_prob_per_point = _log_prob_per_point_factory(likelihood)
        L_K = _stable_cholesky(K)

        def neg_log_post(f: Float[Array, " N"]) -> Float[Array, ""]:
            ll = log_prob_per_point(f, y).sum()
            r = f - prior_mean
            alpha = jax.scipy.linalg.cho_solve((L_K, True), r)
            return -(ll - 0.5 * jnp.dot(r, alpha))

        opt = optax.lbfgs()
        f = jnp.asarray(prior_mean)
        opt_state = opt.init(f)
        value_and_grad = optax.value_and_grad_from_state(neg_log_post)
        n_iter = 0
        converged = False
        for it in range(self.max_iter):
            v, g = value_and_grad(f, state=opt_state)
            updates, opt_state = opt.update(
                g, opt_state, f, value=v, grad=g, value_fn=neg_log_post
            )
            f = optax.apply_updates(f, updates)
            n_iter = it + 1
            if jnp.linalg.norm(g) < self.tol:
                converged = True
                break

        # Laplace covariance at the optimum. Cast ``f`` from optax's
        # generic ``Array`` to a typed JAX array so downstream typing
        # narrows correctly.
        f_opt = jnp.asarray(f)
        g, h = _per_point_grad_hess(log_prob_per_point, f_opt, y)
        Lam = jnp.maximum(-h, self.precision_floor)
        nat1 = g + Lam * f_opt
        q_mean, _, q_var = _posterior_from_diag_sites(K, nat1, Lam, prior_mean)

        log_marg = _laplace_log_marginal(
            log_prob_per_point, f_opt, y, prior_mean, K, Lam
        )

        return NonGaussConditionedGP(  # ty: ignore[invalid-return-type]
            prior=prior,
            y=y,
            site_nat1=nat1,
            site_nat2=Lam,
            q_mean=q_mean,
            q_var=q_var,
            log_marginal_approx=log_marg,
            n_iter=n_iter,
            converged=converged,
        )


# --- module-level conveniences ------------------------------------------
