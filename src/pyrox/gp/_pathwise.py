"""Pathwise GP posterior samplers via Matheron's rule.

Provides callable posterior function draws for exact and sparse GP surfaces.
Each sampled path can be evaluated repeatedly at arbitrary inputs without
refactorizing a test-set covariance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpyro.distributions as dist
from gaussx import cholesky, unwhiten
from jaxtyping import Array, Float

from pyrox.gp._context import _kernel_context
from pyrox.gp._guides import WhitenedGuide
from pyrox.gp._kernels import RBF, Matern
from pyrox.gp._models import ConditionedGP
from pyrox.gp._protocols import Guide, Kernel
from pyrox.gp._sparse import SparseGPPrior


def _solve_with_cholesky(
    chol: lx.AbstractLinearOperator,
    rhs: Float[Array, "N M"],
) -> Float[Array, "N M"]:
    """Solve ``L L^T x = rhs`` for batched row right-hand sides."""
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


def _sample_rff_parameters(
    kernel: Kernel,
    key: Array,
    *,
    n_paths: int,
    n_features: int,
    in_features: int,
    dtype: jnp.dtype,
) -> tuple[
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, "S D F"],
    Float[Array, "S F"],
    Float[Array, "S F"],
]:
    """Sample a cosine-bias RFF prior basis for supported stationary kernels."""
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}.")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}.")

    with _kernel_context(kernel):
        if isinstance(kernel, RBF):
            variance = jnp.asarray(kernel.get_param("variance"), dtype=dtype)
            lengthscale = jnp.asarray(kernel.get_param("lengthscale"), dtype=dtype)
            w_key, b_key, weight_key = jax.random.split(key, 3)
            omega = jax.random.normal(
                w_key,
                shape=(n_paths, in_features, n_features),
                dtype=dtype,
            )
        elif isinstance(kernel, Matern):
            variance = jnp.asarray(kernel.get_param("variance"), dtype=dtype)
            lengthscale = jnp.asarray(kernel.get_param("lengthscale"), dtype=dtype)
            w_key, b_key, weight_key = jax.random.split(key, 3)
            omega = jnp.asarray(
                dist.StudentT(df=2.0 * kernel.nu).sample(
                    w_key,
                    sample_shape=(n_paths, in_features, n_features),
                ),
                dtype=dtype,
            )
        else:
            raise ValueError(
                "Pathwise samplers currently support RBF and Matern kernels for "
                f"their random-feature prior draws; got {type(kernel).__name__}."
            )

    phase = jax.random.uniform(
        b_key,
        shape=(n_paths, n_features),
        minval=jnp.array(0.0, dtype=dtype),
        maxval=jnp.array(2.0 * jnp.pi, dtype=dtype),
        dtype=dtype,
    )
    weights = jax.random.normal(
        weight_key,
        shape=(n_paths, n_features),
        dtype=dtype,
    )
    return variance, lengthscale, omega, phase, weights


def _evaluate_rff_paths(
    X: Float[Array, "N D"],
    *,
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""],
    omega: Float[Array, "S D F"],
    phase: Float[Array, "S F"],
    weights: Float[Array, "S F"],
) -> Float[Array, "S N"]:
    """Evaluate the sampled zero-mean RFF prior paths at ``X``."""
    angles = jnp.einsum("nd,sdf->snf", X, omega) / lengthscale + phase[:, None, :]
    features = jnp.sqrt(2.0 * variance / omega.shape[-1]) * jnp.cos(angles)
    return jnp.sum(features * weights[:, None, :], axis=-1)


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
    ref_points: Float[Array, "R D"]
    correction_weights: Float[Array, "S R"]
    omega: Float[Array, "S D F"]
    phase: Float[Array, "S F"]
    feature_weights: Float[Array, "S F"]
    variance: Float[Array, ""]
    lengthscale: Float[Array, ""]
    mean_fn: Callable[[Float[Array, "N D"]], Float[Array, " N"]] | None = None

    def __call__(self, X_star: Float[Array, "N D"]) -> Float[Array, "S N"]:
        """Evaluate the sampled function(s) at arbitrary inputs ``X_star``."""
        prior = _evaluate_rff_paths(
            X_star,
            variance=self.variance,
            lengthscale=self.lengthscale,
            omega=self.omega,
            phase=self.phase,
            weights=self.feature_weights,
        )
        with _kernel_context(self.kernel):
            K_cross = self.kernel(X_star, self.ref_points)
        update = jnp.einsum("nr,sr->sn", K_cross, self.correction_weights)
        mean = _broadcast_mean(self.mean_fn, X_star)
        return prior + update + mean[None, :]


class PathwiseSampler(eqx.Module):
    """Exact-GP pathwise posterior sampler using Matheron's rule.

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
        """Sample callable posterior paths evaluated in ``O(n_features + N)``."""
        X = self.conditioned_gp.prior.X
        variance, lengthscale, omega, phase, feature_weights = _sample_rff_parameters(
            self.conditioned_gp.prior.kernel,
            key,
            n_paths=n_paths,
            n_features=self.n_features,
            in_features=X.shape[1],
            dtype=X.dtype,
        )
        prior_train = _evaluate_rff_paths(
            X,
            variance=variance,
            lengthscale=lengthscale,
            omega=omega,
            phase=phase,
            weights=feature_weights,
        )
        mean_train = _broadcast_mean(self.conditioned_gp.prior.mean_fn, X)
        noise_key = jax.random.fold_in(key, 1)
        noise = jnp.sqrt(jnp.asarray(self.conditioned_gp.noise_var, dtype=X.dtype)) * (
            jax.random.normal(noise_key, shape=(n_paths, X.shape[0]), dtype=X.dtype)
        )
        residual = (
            self.conditioned_gp.y[None, :] - (mean_train[None, :] + prior_train) - noise
        )
        correction_weights = _solve_with_cholesky(
            cholesky(self.conditioned_gp.operator), residual
        )
        return cast(
            PathwiseFunction,
            PathwiseFunction(
                kernel=self.conditioned_gp.prior.kernel,
                ref_points=X,
                correction_weights=correction_weights,
                omega=omega,
                phase=phase,
                feature_weights=feature_weights,
                variance=variance,
                lengthscale=lengthscale,
                mean_fn=self.conditioned_gp.prior.mean_fn,
            ),
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

    def sample_paths(self, key: Array, n_paths: int = 1) -> PathwiseFunction:
        """Sample callable sparse posterior paths."""
        if self.prior.Z is None:
            raise ValueError(
                "DecoupledPathwiseSampler currently requires a point-inducing "
                "SparseGPPrior with `Z=...`."
            )

        Z = self.prior.Z
        variance, lengthscale, omega, phase, feature_weights = _sample_rff_parameters(
            self.prior.kernel,
            key,
            n_paths=n_paths,
            n_features=self.n_features,
            in_features=Z.shape[1],
            dtype=Z.dtype,
        )
        prior_inducing = _evaluate_rff_paths(
            Z,
            variance=variance,
            lengthscale=lengthscale,
            omega=omega,
            phase=phase,
            weights=feature_weights,
        )

        guide_keys = jax.random.split(jax.random.fold_in(key, 1), n_paths)
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
        return cast(
            PathwiseFunction,
            PathwiseFunction(
                kernel=self.prior.kernel,
                ref_points=Z,
                correction_weights=correction_weights,
                omega=omega,
                phase=phase,
                feature_weights=feature_weights,
                variance=variance,
                lengthscale=lengthscale,
                mean_fn=self.prior.mean_fn,
            ),
        )

    def __call__(
        self,
        key: Array,
        X_star: Float[Array, "N D"],
        n_paths: int = 1,
    ) -> Float[Array, "S N"]:
        """Convenience wrapper for ``sample_paths(key, n_paths)(X_star)``."""
        return self.sample_paths(key, n_paths=n_paths)(X_star)
