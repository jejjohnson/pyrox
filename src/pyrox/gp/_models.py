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
from typing import Any

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
    cholesky,
    log_marginal_likelihood,
    predict_mean,
    predict_variance,
    unwhiten,
)
from jaxtyping import Array, Float
from numpyro import distributions as dist

from pyrox.gp._context import _kernel_context
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

    def sample(self, key: Array) -> Float[Array, " N"]:
        r"""Draw ``f \sim p(f) = \mathcal{N}(\mu(X), K + \text{jitter}\,I)``.

        Wraps the prior in a :class:`gaussx.MultivariateNormal` with
        the configured :attr:`solver`. This is the non-NumPyro analogue
        of :func:`gp_sample` — useful for tests, diagnostics, and
        prior-sample initialization without registering a sample site.
        """
        op = self._prior_operator()
        loc = self.mean(self.X)
        mvn = MultivariateNormal(loc, op, solver=self._resolved_solver())
        return mvn.sample(key)

    def condition(
        self,
        y: Float[Array, " N"],
        noise_var: Float[Array, ""],
    ) -> ConditionedGP:
        """Condition on Gaussian-likelihood observations ``y``.

        Precomputes
        ``alpha = (K + (jitter + noise_var) * I)^{-1} (y - mu(X))`` and
        caches it in the returned :class:`ConditionedGP`. The same
        ``jitter`` regularization configured on this prior is included
        alongside ``noise_var`` in the conditioned operator and solve, so
        every downstream predict / sample call sees the regularized
        covariance.

        The operator construction and any subsequent hyperparameter
        capture share one :func:`_kernel_context`, so for Pattern B/C
        kernels with priors the cached operator and the resolved
        hyperparameters on the returned :class:`ConditionedGP` come from
        the same draw. Downstream consumers (notably
        :class:`pyrox.gp.PathwiseSampler`) reuse those values to stay
        consistent with the cached operator.
        """
        with _kernel_context(self.kernel):
            operator = self._noisy_operator(noise_var)
            resolved_hyperparams = _resolve_kernel_hyperparams(self.kernel)
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
            resolved_hyperparams=resolved_hyperparams,
        )

    def condition_nongauss(
        self,
        likelihood: Any,
        y: Float[Array, " N"],
        *,
        strategy: Any,
    ) -> Any:
        """Condition on a non-Gaussian likelihood via a site-based strategy.

        Convenience that forwards to ``strategy.fit(self, likelihood, y)``.
        Pick any of the site-based strategies in
        :mod:`pyrox.gp._inference_nongauss`:
        :class:`pyrox.gp.LaplaceInference`,
        :class:`pyrox.gp.GaussNewtonInference`,
        :class:`pyrox.gp.PosteriorLinearization`,
        :class:`pyrox.gp.ExpectationPropagation`, or
        :class:`pyrox.gp.QuasiNewtonInference`. Returns a
        :class:`pyrox.gp.NonGaussConditionedGP` with the same
        ``predict`` / ``predict_mean`` / ``predict_var`` API as the
        Gaussian-likelihood :class:`ConditionedGP`.

        Example::

            from pyrox.gp import (
                BernoulliLikelihood,
                ExpectationPropagation,
                GPPrior,
                RBF,
            )

            prior = GPPrior(kernel=RBF(), X=X)
            cond = prior.condition_nongauss(
                BernoulliLikelihood(), y,
                strategy=ExpectationPropagation(),
            )
            mean, var = cond.predict(X_star)
        """
        return strategy.fit(self, likelihood, y)


def _resolve_kernel_hyperparams(
    kernel: Kernel,
) -> tuple[Float[Array, ""], Float[Array, ""]] | None:
    """Capture ``(variance, lengthscale)`` for kernels that expose them.

    Must be called inside the same :func:`_kernel_context` as the
    kernel evaluation that downstream code wants to stay consistent
    with — Pattern B/C kernels register their priors per call, and a
    fresh outer context would resample.

    Returns ``None`` for kernels without those names so callers fall
    back to whatever default the consumer expects (e.g.
    :func:`pyrox._basis.draw_rff_cosine_basis` will read them itself).
    """
    get_param = getattr(kernel, "get_param", None)
    if get_param is None:
        return None
    try:
        variance = jnp.asarray(get_param("variance"))
        lengthscale = jnp.asarray(get_param("lengthscale"))
    except (KeyError, AttributeError, ValueError):
        return None
    return variance, lengthscale


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
    resolved_hyperparams: tuple[Float[Array, ""], Float[Array, ""]] | None = None

    def predict_mean(self, X_star: Float[Array, "M D"]) -> Float[Array, " M"]:
        r""":math:`\mu_* = \mu(X_*) + K_{*f}\,\alpha`."""
        with _kernel_context(self.prior.kernel):
            K_cross = self.prior.kernel(X_star, self.prior.X)
        return self.prior.mean(X_star) + predict_mean(self.cache, K_cross)

    def predict_var(self, X_star: Float[Array, "M D"]) -> Float[Array, " M"]:
        r"""Diagonal predictive variance at ``X_*``.

        .. math::
            \sigma^2_{*,i} = k(x_{*,i}, x_{*,i})
                - K_{*f}[i,:] \cdot (K + \sigma^2 I)^{-1} K_{f*}[:,i]

        ``K_cross`` and ``K_diag`` are computed under one shared kernel
        context so Pattern B / C kernels with prior'd hyperparameters
        register their NumPyro sites once and reuse them across both
        kernel calls (and the cached training solve).
        """
        with _kernel_context(self.prior.kernel):
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
        """Return ``(mean, variance)`` at ``X_*`` as a tuple.

        Both kernel evaluations share a single kernel context; see
        :meth:`predict_var`.
        """
        with _kernel_context(self.prior.kernel):
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
        with _kernel_context(self.prior.kernel):
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

    Adds
    ``log p(y | X, theta) = log N(y | mu, K + (jitter + sigma^2) I)``
    to the NumPyro trace as ``numpyro.factor(name, ...)``. The prior's
    ``jitter`` is included in addition to the observation noise variance
    so the covariance matches what :meth:`GPPrior.condition` builds. Use
    this inside a NumPyro model when the likelihood is Gaussian and you
    want the latent function marginalized analytically.
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
    whitened: bool = False,
    guide: object | None = None,
) -> Float[Array, " N"]:
    r"""Sample a latent function ``f`` at the prior's training inputs.

    Three mutually exclusive modes:

    * ``whitened=False``, ``guide=None`` (default) — register a single
      ``numpyro.sample(name, MVN(mu, K + jitter I))`` site. The latent
      function is sampled directly from the prior.
    * ``whitened=True``, ``guide=None`` — register a unit-normal latent
      site ``f"{name}_u"`` with shape ``(N,)`` and return the
      deterministic value ``f = mu(X) + L u`` where ``L`` is the
      Cholesky factor of ``K + jitter I``. This reparameterization is the
      standard fix for mean-field SVI on GP-correlated latents
      (Murray & Adams, 2010): a NumPyro auto-guide such as
      :class:`numpyro.infer.autoguide.AutoNormal` then approximates the
      well-conditioned isotropic posterior over ``u`` instead of the
      ill-conditioned correlated posterior over ``f``.
    * ``guide`` provided — delegate to ``guide.register(name, prior)``.
      Concrete variational guides (Wave 3) own their own
      parameterization, so combining ``whitened=True`` with ``guide`` is
      rejected.

    Use this inside a NumPyro model for non-conjugate likelihoods, where
    the latent function cannot be marginalized analytically.
    """
    if guide is not None:
        if whitened:
            raise ValueError(
                "gp_sample: cannot combine `whitened=True` with `guide=...`. "
                "Provide one or the other; concrete guides own their own "
                "parameterization."
            )
        return guide.register(name, prior)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]

    if whitened:
        L = cholesky(prior._prior_operator())
        n = prior.X.shape[0]
        dtype = prior.X.dtype
        u = numpyro.sample(
            f"{name}_u",
            dist.Normal(jnp.zeros(n, dtype=dtype), jnp.ones((), dtype=dtype)).to_event(
                1
            ),
        )
        f = prior.mean(prior.X) + unwhiten(jnp.asarray(u), L)
        return numpyro.deterministic(name, f)  # ty: ignore[invalid-return-type]

    return numpyro.sample(  # ty: ignore[invalid-return-type]
        name,
        MultivariateNormal(
            prior.mean(prior.X),
            prior._prior_operator(),
            solver=prior._resolved_solver(),
        ),
    )
