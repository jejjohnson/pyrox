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
from pyrox.gp._kernels import RBF, Matern
from pyrox.gp._models import ConditionedGP
from pyrox.gp._protocols import Guide, Kernel
from pyrox.gp._sparse import SparseGPPrior
from pyrox.gp._src import kernels as _kernel_fns


def _frozen_kernel_fn(
    kernel: Kernel,
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""],
) -> Callable[[Float[Array, "N1 D"], Float[Array, "N2 D"]], Float[Array, "N1 N2"]]:
    """Return ``(X1, X2) -> K(X1, X2)`` with frozen hyperparameters.

    Pattern B/C kernels read ``variance`` / ``lengthscale`` via
    ``get_param`` under a ``_kernel_context``, which resamples on every
    call for kernels that registered priors. The pathwise sampler must
    instead reuse the *same* hyperparameter draw that produced the RFF
    prior basis, so the posterior correction term ``K(x_*, X_corr)``
    uses values consistent with ``omega / lengthscale`` in the RFF draw.

    This helper closes over the captured scalars and calls the pure math
    primitive directly, bypassing the ``pyrox_sample`` layer entirely.
    Only RBF and Matern are supported — same scope as
    :func:`pyrox._basis._rff.draw_rff_cosine_basis`.
    """
    if isinstance(kernel, RBF):

        def rbf_fn(X1, X2):
            return _kernel_fns.rbf_kernel(X1, X2, variance, lengthscale)

        return rbf_fn
    if isinstance(kernel, Matern):
        nu = kernel.nu

        def matern_fn(X1, X2):
            return _kernel_fns.matern_kernel(X1, X2, variance, lengthscale, nu)

        return matern_fn
    raise NotImplementedError(
        "Pathwise sampling currently supports RBF and Matern kernels; "
        f"got {type(kernel).__name__}."
    )


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

    The kernel enters only as a frozen ``(X1, X2) -> K`` callable with
    the sample-time ``variance`` and ``lengthscale`` baked in, so
    repeated evaluations stay consistent with the original RFF draw
    even for Pattern B/C kernels that register hyperparameter priors.

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

    kernel_fn: Callable[
        [Float[Array, "N1 D"], Float[Array, "N2 D"]], Float[Array, "N1 N2"]
    ]
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
        K_cross = self.kernel_fn(X_star, self.correction_points)
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
    :math:`\\mathcal{O}(N_* \\cdot F \\cdot D + N_* \\cdot N)` per path,
    where ``N`` is the number of training (correction) points: the RFF
    prior term recomputes features over ``X_*`` each call
    (``N_* · F · D``), and the correction term forms a fresh
    ``K(X_*, X)`` block (``N_* · N``).

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
        kernel = self.conditioned_gp.prior.kernel
        # Reuse the resolved (variance, lengthscale) captured by
        # GPPrior.condition under its kernel context. For Pattern B/C
        # kernels the cached operator was built with these exact
        # values; resampling here would put the RFF basis in a
        # different posterior than the cached training solve.
        cached = self.conditioned_gp.resolved_hyperparams
        cached_variance, cached_lengthscale = (
            cached if cached is not None else (None, None)
        )
        variance, lengthscale, omega, phase, feature_weights = draw_rff_cosine_basis(
            kernel,
            rff_key,
            n_paths=n_paths,
            n_features=self.n_features,
            in_features=X.shape[1],
            dtype=X.dtype,
            variance=cached_variance,
            lengthscale=cached_lengthscale,
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
        # Matheron requires Cov(eps_tilde) to match the diagonal added to
        # the cached operator. _noisy_operator uses (noise_var + jitter) I,
        # so eps_tilde must have the same variance — otherwise the
        # correction solve is inconsistent and paths are under-dispersed
        # (pronounced when jitter is bumped up for stability).
        noise_var = jnp.asarray(self.conditioned_gp.noise_var, dtype=X.dtype)
        jitter = jnp.asarray(self.conditioned_gp.prior.jitter, dtype=X.dtype)
        eps_var = noise_var + jitter
        noise = jnp.sqrt(eps_var) * jax.random.normal(
            noise_key, shape=(n_paths, X.shape[0]), dtype=X.dtype
        )
        residual = (
            self.conditioned_gp.y[None, :] - (mean_train[None, :] + prior_train) - noise
        )
        correction_weights = _solve_with_cholesky(
            cholesky(self.conditioned_gp.operator), residual
        )
        return PathwiseFunction(  # ty: ignore[invalid-return-type]
            kernel_fn=_frozen_kernel_fn(kernel, variance, lengthscale),
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

        ``key`` is split into three subkeys: one for the RFF basis, one
        for ``n_paths`` independent guide draws, and one for the
        jitter-augmentation of the prior inducing draw. The RFF basis
        draw and the :math:`K_{zz}` assembly share a single
        ``_kernel_context`` so kernels with hyperparameter priors
        (Pattern B / C) sample ``(variance, lengthscale)`` once.

        The Matheron correction needs ``Cov(u_tilde) = K_{zz} + \
        \\text{jitter}\\,I`` so it matches the operator that the
        correction is solved against. The bare RFF draw at ``Z``
        produces only the ``K_{zz}`` part; we add an iid Gaussian
        with variance ``jitter`` per inducing index to close the gap —
        without this, paths are under-dispersed when jitter is bumped
        up for stability.
        """
        rff_key, guide_key, jitter_key = jax.random.split(key, 3)

        Z = self.prior.Z
        assert Z is not None  # __check_init__ guarantees
        with _kernel_context(self.prior.kernel):
            basis = draw_rff_cosine_basis(
                self.prior.kernel,
                rff_key,
                n_paths=n_paths,
                n_features=self.n_features,
                in_features=Z.shape[1],
                dtype=Z.dtype,
            )
            variance, lengthscale, omega, phase, feature_weights = basis
            prior_inducing = evaluate_rff_cosine_paths(
                Z,
                variance=variance,
                lengthscale=lengthscale,
                omega=omega,
                phase=phase,
                weights=feature_weights,
            )
            inducing_chol = cholesky(self.prior.inducing_operator())

        # See docstring: u_tilde must have covariance K_zz + jitter I.
        jitter = jnp.asarray(self.prior.jitter, dtype=Z.dtype)
        prior_inducing = prior_inducing + jnp.sqrt(jitter) * jax.random.normal(
            jitter_key, shape=prior_inducing.shape, dtype=Z.dtype
        )

        guide_keys = jax.random.split(guide_key, n_paths)
        guide_samples = jax.vmap(self.guide.sample)(guide_keys)
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
            kernel_fn=_frozen_kernel_fn(self.prior.kernel, variance, lengthscale),
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
