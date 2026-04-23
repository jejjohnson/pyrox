"""Pathwise GP posterior samplers via Matheron's rule.

Provides callable posterior function draws for exact and sparse GP
surfaces. Each sampled path can be evaluated repeatedly at arbitrary
inputs without refactorizing a test-set covariance.

The zero-mean prior path is drawn via the shared random-Fourier-feature
primitives in :mod:`pyrox._basis._rff` (:func:`draw_rff_cosine_basis` /
:func:`evaluate_rff_cosine_paths`); this module then adds the posterior
correction and the optional prior mean.

Scope: RBF and Matern kernels; point-inducing :class:`SparseGPPrior`.
Inducing-feature priors (:class:`pyrox.gp.FourierInducingFeatures`,
:class:`pyrox.gp.SphericalHarmonicInducingFeatures`,
:class:`pyrox.gp.LaplacianInducingFeatures`) are not yet covered —
:class:`DecoupledPathwiseSampler` raises on construction when
``prior.Z is None`` so the limitation surfaces before a long sample
loop.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from gaussx import cholesky, unwhiten
from jaxtyping import Array, Float

from pyrox._basis._rff import (
    draw_rff_cosine_basis,
    evaluate_rff_cosine_paths,
)
from pyrox.gp._context import _kernel_context
from pyrox.gp._guides import WhitenedGuide
from pyrox.gp._models import ConditionedGP
from pyrox.gp._protocols import Guide, Kernel
from pyrox.gp._sparse import SparseGPPrior


def _solve_with_cholesky(
    chol: lx.AbstractLinearOperator,
    rhs: Float[Array, "S R"],
) -> Float[Array, "S R"]:
    """Solve ``L L^T alpha^T = rhs^T`` column-wise; return ``alpha`` shape ``(S, R)``.

    Each row of ``rhs`` is a per-path right-hand side; the output row
    is ``(L L^T)^{-1}`` applied to that row, so downstream callers can
    contract ``K_cross[n, r] * alpha[s, r]`` without extra transposes.
    """
    left = jax.vmap(
        lambda col: lx.linear_solve(chol, col).value,
        in_axes=1,
        out_axes=1,
    )(rhs.T)
    solved = jax.vmap(
        lambda col: lx.linear_solve(chol.T, col).value,
        in_axes=1,
        out_axes=1,
    )(left)
    return solved.T


def _broadcast_mean(
    mean_fn: Callable[[Float[Array, "N D"]], Float[Array, " N"]] | None,
    X: Float[Array, "N D"],
) -> Float[Array, " N"]:
    """Evaluate the deterministic mean function or return zeros."""
    if mean_fn is None:
        return jnp.zeros(X.shape[0], dtype=X.dtype)
    return mean_fn(X)


class PathwiseFunction(eqx.Module):
    """Callable posterior function draw(s) produced by a pathwise sampler.

    Carries the random-feature prior basis (``omega``, ``phase``,
    ``feature_weights``) and the posterior correction weights evaluated
    against either the training inputs (exact) or the inducing inputs
    (sparse). Calling the instance on test points ``X_star`` evaluates

    .. math::

        f_{\\text{post}}(x_*) =
            \\tilde{f}(x_*)
            + K(x_*,\\, X_{\\mathrm{corr}})\\,\\alpha
            + \\mu(x_*),

    where :math:`\\tilde f` is the stored RFF prior draw and
    :math:`X_{\\mathrm{corr}}` is either the training set (exact) or
    the inducing set (sparse).

    Example:
        >>> prior = GPPrior(kernel=RBF(), X=X)
        >>> posterior = prior.condition(y, noise_var=jnp.array(0.05))
        >>> sampler = PathwiseSampler(posterior, n_features=512)
        >>> paths = sampler.sample_paths(key, n_paths=8)
        >>> samples = paths(X_star)

    Example:
        >>> sparse_prior = SparseGPPrior(kernel=RBF(), Z=Z)
        >>> guide = FullRankGuide.init(Z.shape[0])
        >>> paths = DecoupledPathwiseSampler(sparse_prior, guide).sample_paths(key)
        >>> thompson_values = paths(X_candidates)
    """

    kernel: Kernel
    correction_points: Float[Array, "R D"]
    correction_weights: Float[Array, "S R"]
    omega: Float[Array, "S D F"]
    phase: Float[Array, "S F"]
    feature_weights: Float[Array, "S F"]
    variance: Float[Array, ""]
    lengthscale: Float[Array, ""]
    mean_fn: Callable[[Float[Array, "N D"]], Float[Array, " N"]] | None = None

    def __call__(self, X_star: Float[Array, "N D"]) -> Float[Array, "S N"]:
        """Evaluate the sampled function(s) at arbitrary inputs ``X_star``."""
        prior = evaluate_rff_cosine_paths(
            X_star,
            variance=self.variance,
            lengthscale=self.lengthscale,
            omega=self.omega,
            phase=self.phase,
            weights=self.feature_weights,
        )
        with _kernel_context(self.kernel):
            K_cross = self.kernel(X_star, self.correction_points)
        update = jnp.einsum("nr,sr->sn", K_cross, self.correction_weights)
        mean = _broadcast_mean(self.mean_fn, X_star)
        return prior + update + mean[None, :]


class PathwiseSampler(eqx.Module):
    """Exact-GP pathwise posterior sampler using Matheron's rule.

    Given a :class:`ConditionedGP`, draws a zero-mean RFF prior path
    ``f_tilde`` and an iid noise draw ``eps_tilde`` at the training
    inputs, forms the residual ``y - mu(X) - f_tilde(X) - eps_tilde``,
    solves it against the cached noisy operator ``K + (jitter + sigma^2)I``,
    and stores the result as posterior correction weights. The returned
    :class:`PathwiseFunction` is callable at any ``X_*`` in
    :math:`\\mathcal{O}(F + N \\cdot N_*)` per path.

    Example:
        >>> posterior = GPPrior(kernel=RBF(), X=X).condition(y, jnp.array(0.05))
        >>> sampler = PathwiseSampler(posterior, n_features=512)
        >>> paths = sampler.sample_paths(key, n_paths=32)
        >>> draws = paths(X_star)

    Example:
        >>> sampler = PathwiseSampler(posterior, n_features=1024)
        >>> thompson = sampler.sample_paths(key, n_paths=1)
        >>> values = thompson(X_candidates)
    """

    conditioned_gp: ConditionedGP
    n_features: int = eqx.field(static=True, default=512)

    def sample_paths(self, key: Array, n_paths: int = 1) -> PathwiseFunction:
        """Sample callable posterior paths.

        ``key`` is split into three subkeys: one for the RFF basis,
        one for the iid training-noise draw, and one reserved for
        future extensions.
        """
        rff_key, noise_key, _reserved = jax.random.split(key, 3)

        X = self.conditioned_gp.prior.X
        variance, lengthscale, omega, phase, feature_weights = draw_rff_cosine_basis(
            self.conditioned_gp.prior.kernel,
            rff_key,
            n_paths=n_paths,
            n_features=self.n_features,
            in_features=X.shape[1],
            dtype=X.dtype,
        )
        prior_train = evaluate_rff_cosine_paths(
            X,
            variance=variance,
            lengthscale=lengthscale,
            omega=omega,
            phase=phase,
            weights=feature_weights,
        )
        mean_train = _broadcast_mean(self.conditioned_gp.prior.mean_fn, X)
        noise = jnp.sqrt(jnp.asarray(self.conditioned_gp.noise_var, dtype=X.dtype)) * (
            jax.random.normal(noise_key, shape=(n_paths, X.shape[0]), dtype=X.dtype)
        )
        residual = (
            self.conditioned_gp.y[None, :] - (mean_train[None, :] + prior_train) - noise
        )
        correction_weights = _solve_with_cholesky(
            cholesky(self.conditioned_gp.operator), residual
        )
        return PathwiseFunction(  # ty: ignore[invalid-return-type]
            kernel=self.conditioned_gp.prior.kernel,
            correction_points=X,
            correction_weights=correction_weights,
            omega=omega,
            phase=phase,
            feature_weights=feature_weights,
            variance=variance,
            lengthscale=lengthscale,
            mean_fn=self.conditioned_gp.prior.mean_fn,
        )

    def __call__(
        self,
        key: Array,
        X_star: Float[Array, "N D"],
        n_paths: int = 1,
    ) -> Float[Array, "S N"]:
        """Convenience wrapper for ``sample_paths(key, n_paths)(X_star)``."""
        return self.sample_paths(key, n_paths=n_paths)(X_star)


class DecoupledPathwiseSampler(eqx.Module):
    """Sparse/decoupled pathwise sampler with RFF prior + inducing update.

    The prior draw uses random features while the correction is represented in
    the inducing-point basis, so each sampled path stays callable at arbitrary
    inputs after a one-time inducing solve.

    Supported for point-inducing :class:`SparseGPPrior` (``Z=...``);
    inducing-feature priors (``inducing=...``) are rejected at
    construction with a clear error.

    Handles :class:`WhitenedGuide` automatically: whitened guide draws
    ``v ~ q(v)`` are unwhitened to inducing values ``u = L_ZZ v`` via
    :func:`gaussx.unwhiten` before forming the inducing-space residual.

    Example:
        >>> prior = SparseGPPrior(kernel=RBF(), Z=Z)
        >>> guide = FullRankGuide.init(Z.shape[0])
        >>> sampler = DecoupledPathwiseSampler(prior, guide, n_features=512)
        >>> paths = sampler.sample_paths(key, n_paths=16)
        >>> draws = paths(X_star)
    """

    prior: SparseGPPrior
    guide: Guide
    n_features: int = eqx.field(static=True, default=512)

    def __check_init__(self) -> None:
        if self.prior.Z is None:
            raise ValueError(
                "DecoupledPathwiseSampler currently requires a point-inducing "
                "SparseGPPrior constructed with `Z=...`. Inducing-feature "
                "priors (FourierInducingFeatures, SphericalHarmonicInducingFeatures, "
                "LaplacianInducingFeatures) are not yet supported."
            )

    def sample_paths(self, key: Array, n_paths: int = 1) -> PathwiseFunction:
        """Sample callable sparse posterior paths.

        ``key`` is split into two subkeys: one for the RFF basis and
        one for ``n_paths`` independent guide draws.
        """
        rff_key, guide_key = jax.random.split(key, 2)

        Z = self.prior.Z
        assert Z is not None  # __check_init__ guarantees
        variance, lengthscale, omega, phase, feature_weights = draw_rff_cosine_basis(
            self.prior.kernel,
            rff_key,
            n_paths=n_paths,
            n_features=self.n_features,
            in_features=Z.shape[1],
            dtype=Z.dtype,
        )
        prior_inducing = evaluate_rff_cosine_paths(
            Z,
            variance=variance,
            lengthscale=lengthscale,
            omega=omega,
            phase=phase,
            weights=feature_weights,
        )

        guide_keys = jax.random.split(guide_key, n_paths)
        guide_samples = jax.vmap(self.guide.sample)(guide_keys)
        inducing_chol = cholesky(self.prior.inducing_operator())
        if isinstance(self.guide, WhitenedGuide):
            inducing_samples = jax.vmap(lambda sample: unwhiten(sample, inducing_chol))(
                guide_samples
            )
        else:
            inducing_samples = guide_samples

        correction_weights = _solve_with_cholesky(
            inducing_chol,
            inducing_samples - prior_inducing,
        )
        return PathwiseFunction(  # ty: ignore[invalid-return-type]
            kernel=self.prior.kernel,
            correction_points=Z,
            correction_weights=correction_weights,
            omega=omega,
            phase=phase,
            feature_weights=feature_weights,
            variance=variance,
            lengthscale=lengthscale,
            mean_fn=self.prior.mean_fn,
        )

    def __call__(
        self,
        key: Array,
        X_star: Float[Array, "N D"],
        n_paths: int = 1,
    ) -> Float[Array, "S N"]:
        """Convenience wrapper for ``sample_paths(key, n_paths)(X_star)``."""
        return self.sample_paths(key, n_paths=n_paths)(X_star)
