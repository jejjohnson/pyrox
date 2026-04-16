"""Model-facing GP entry points — ``GPPrior``, ``ConditionedGP``, and NumPyro hooks.

This module is the NumPyro-aware shell on top of ``gaussx``. pyrox holds
the kernel math and the model surface; ``gaussx`` holds every piece of
linear algebra. In particular the solver strategy is pluggable: every
entry here accepts any ``gaussx.AbstractSolverStrategy`` (default
``gaussx.DenseSolver()``) or separate solve / logdet strategies composed
via ``gaussx.ComposedSolver``.

Wave 2 scope: dense, finite-dimensional workflows only. Sparse variational
guides, pathwise samplers, state-space, and multi-output extensions land
in later waves.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpyro
from einops import einsum
from gaussx import (
    AbstractSolverStrategy,
    DenseSolver,
    MultivariateNormal,
    PredictionCache,
    build_prediction_cache,
    log_marginal_likelihood,
    predict_mean,
    predict_variance,
)
from jaxtyping import Array, Float

from pyrox.gp._protocols import Kernel


def _psd_operator(K: Float[Array, "N N"]) -> lx.AbstractLinearOperator:
    """Wrap a Gram matrix as a PSD ``lineax`` operator.

    The positive-semidefinite tag lets gaussx route the matrix to
    Cholesky-based solvers / logdets when using ``AutoSolver``.
    """
    return lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)  # ty: ignore[invalid-return-type]


class GPPrior(eqx.Module):
    """Finite-dimensional GP prior over a fixed training input set.

    Holds a kernel, training inputs ``X``, an optional mean function, an
    optional solver strategy, and a small diagonal jitter for numerical
    stability on otherwise-singular prior covariances.

    Attributes:
        kernel: Any :class:`pyrox.gp.Kernel` — evaluated on ``X``.
        X: Training inputs of shape ``(N, D)``.
        mean_fn: Callable ``X -> (N,)`` or ``None`` for the zero mean.
        solver: Any ``gaussx.AbstractSolverStrategy``. Defaults to
            ``gaussx.DenseSolver()`` — swap for ``CGSolver``,
            ``BBMMSolver``, ``ComposedSolver(solve=..., logdet=...)``, etc.
        jitter: Diagonal regularization added to the prior covariance
            for numerical stability. Not a noise model — use
            ``noise_var`` on :meth:`condition` for that.
    """

    kernel: Kernel
    X: Float[Array, "N D"]
    mean_fn: Callable[[Float[Array, "N D"]], Float[Array, " N"]] | None = None
    solver: AbstractSolverStrategy | None = None
    jitter: float = 1e-6

    def mean(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        """Evaluate the mean function at ``X``; zero by default."""
        if self.mean_fn is None:
            return jnp.zeros(X.shape[0], dtype=X.dtype)
        return self.mean_fn(X)

    def _prior_operator(self) -> lx.AbstractLinearOperator:
        K = self.kernel(self.X, self.X)
        K = K + self.jitter * jnp.eye(K.shape[0], dtype=K.dtype)
        return _psd_operator(K)

    def _noisy_operator(self, noise_var: Float[Array, ""]) -> lx.AbstractLinearOperator:
        K = self.kernel(self.X, self.X)
        reg = (self.jitter + noise_var) * jnp.eye(K.shape[0], dtype=K.dtype)
        return _psd_operator(K + reg)

    def _resolved_solver(self) -> AbstractSolverStrategy:
        return DenseSolver() if self.solver is None else self.solver  # ty: ignore[invalid-return-type]

    def log_prob(self, f: Float[Array, " N"]) -> Float[Array, ""]:
        r"""Marginal log-density of ``f`` under the GP prior.

        Computes :math:`\log \mathcal{N}(f \mid \mu(X), K(X, X) + \text{jitter}\,I)`
        using :func:`gaussx.log_marginal_likelihood`, so any solver strategy
        on this prior applies.
        """
        return log_marginal_likelihood(
            self.mean(self.X),
            self._prior_operator(),
            f,
            solver=self._resolved_solver(),
        )

    def condition(
        self,
        y: Float[Array, " N"],
        noise_var: Float[Array, ""],
    ) -> ConditionedGP:
        """Condition on Gaussian-likelihood observations ``y``.

        Precomputes ``alpha = (K + noise_var * I)^{-1} (y - mu(X))`` and
        caches it in the returned :class:`ConditionedGP`.
        """
        operator = self._noisy_operator(noise_var)
        residual = y - self.mean(self.X)
        cache = build_prediction_cache(
            operator, residual, solver=self._resolved_solver()
        )
        return ConditionedGP(  # ty: ignore[invalid-return-type]
            prior=self,
            y=y,
            noise_var=noise_var,
            cache=cache,
            operator=operator,
        )


class ConditionedGP(eqx.Module):
    """GP conditioned on Gaussian-likelihood training observations.

    Holds the precomputed training solve ``alpha`` (via
    :class:`gaussx.PredictionCache`) and the noisy covariance operator so
    predictions at multiple test sets reuse the training solve.
    """

    prior: GPPrior
    y: Float[Array, " N"]
    noise_var: Float[Array, ""]
    cache: PredictionCache
    operator: lx.AbstractLinearOperator

    def predict_mean(self, X_star: Float[Array, "M D"]) -> Float[Array, " M"]:
        r""":math:`\mu_* = \mu(X_*) + K_{*f}\,\alpha`."""
        K_cross = self.prior.kernel(X_star, self.prior.X)
        return self.prior.mean(X_star) + predict_mean(self.cache, K_cross)

    def predict_var(self, X_star: Float[Array, "M D"]) -> Float[Array, " M"]:
        r"""Diagonal predictive variance at ``X_*``.

        .. math::
            \sigma^2_{*,i} = k(x_{*,i}, x_{*,i})
                - K_{*f}[i,:] \cdot (K + \sigma^2 I)^{-1} K_{f*}[:,i]
        """
        K_cross = self.prior.kernel(X_star, self.prior.X)
        K_diag = self.prior.kernel.diag(X_star)
        return predict_variance(
            K_cross,
            K_diag,
            self.operator,
            solver=self.prior._resolved_solver(),
        )

    def predict(
        self, X_star: Float[Array, "M D"]
    ) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
        """Return ``(mean, variance)`` at ``X_*`` as a tuple."""
        return self.predict_mean(X_star), self.predict_var(X_star)

    def sample(
        self,
        key: Array,
        X_star: Float[Array, "M D"],
        n_samples: int = 1,
    ) -> Float[Array, "S M"]:
        """Sample from the diagonal predictive ``N(mean, diag(var))``.

        Returns samples independently per test point; correlated joint
        samples from the full predictive covariance are not covered by
        the Wave 2 dense surface. For correlated samples, build the full
        predictive covariance explicitly and draw from
        :class:`gaussx.MultivariateNormal`.
        """
        mean = self.predict_mean(X_star)
        var = self.predict_var(X_star)
        std = jnp.sqrt(jnp.clip(var, min=0.0))
        eps = jax.random.normal(key, (n_samples, X_star.shape[0]), dtype=mean.dtype)
        return einsum(std, eps, "m, s m -> s m") + mean


def gp_factor(
    name: str,
    prior: GPPrior,
    y: Float[Array, " N"],
    noise_var: Float[Array, ""],
) -> None:
    """Register the collapsed GP log marginal likelihood with NumPyro.

    Adds ``log p(y | X, theta) = log N(y | mu, K + sigma^2 I)`` to the
    NumPyro trace as ``numpyro.factor(name, ...)``. Use this inside a
    NumPyro model when the likelihood is Gaussian and you want the
    latent function marginalized analytically.
    """
    logp = log_marginal_likelihood(
        prior.mean(prior.X),
        prior._noisy_operator(noise_var),
        y,
        solver=prior._resolved_solver(),
    )
    numpyro.factor(name, logp)


def gp_sample(
    name: str,
    prior: GPPrior,
    *,
    guide: object | None = None,
) -> Float[Array, " N"]:
    """Sample a latent function ``f`` at the prior's training inputs.

    When ``guide`` is ``None`` (Wave 2 default) the sample site draws
    from the prior ``MVN(mu(X), K(X, X) + jitter I)`` via
    :class:`gaussx.MultivariateNormal`. A non-``None`` ``guide`` is the
    extension point for Wave 3 variational families — its
    ``sample(name, prior)`` method is invoked to register the posterior
    site instead.

    Use this inside a NumPyro model for non-conjugate likelihoods, where
    the latent function cannot be marginalized analytically.
    """
    if guide is not None:
        return guide.sample(name, prior)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    return numpyro.sample(  # ty: ignore[invalid-return-type]
        name,
        MultivariateNormal(
            prior.mean(prior.X),
            prior._prior_operator(),
            solver=prior._resolved_solver(),
        ),
    )
