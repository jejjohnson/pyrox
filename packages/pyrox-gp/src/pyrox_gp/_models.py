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
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import kernellib as kl
import lineax as lx
import numpyro
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
    solve_columns,
    unwhiten,
)
from jaxtyping import Array, Float
from numpyro import distributions as dist

from pyrox_gp._context import _kernel_context
from pyrox_gp._kernel_operator import freeze_kernel, matrix_free_operator
from pyrox_gp._protocols import Kernel, Likelihood


if TYPE_CHECKING:
    from pyrox_gp._inference_nongauss import NonGaussConditionedGP


class _NonGaussStrategy(Protocol):
    """Structural protocol for non-Gaussian inference strategies."""

    def fit(
        self,
        prior: GPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
    ) -> NonGaussConditionedGP: ...


def _psd_operator(K: Float[Array, "N N"]) -> lx.AbstractLinearOperator:
    """Wrap a Gram matrix as a PSD ``lineax`` operator.

    The positive-semidefinite tag lets gaussx route the matrix to
    Cholesky-based solvers / logdets when using ``AutoSolver``.
    """
    return lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)


class GPPrior(eqx.Module):
    """Finite-dimensional GP prior over a fixed training input set.

    Holds a kernel, training inputs ``X``, an optional mean function, an
    optional solver strategy, and a small diagonal jitter for numerical
    stability on otherwise-singular prior covariances.

    Attributes:
        kernel: Any `pyrox_gp.Kernel` — evaluated on ``X``.
        X: Training inputs of shape ``(N, D)``.
        mean_fn: Callable ``X -> (N,)`` or ``None`` for the zero mean.
        solver: Any ``gaussx.AbstractSolverStrategy``. Defaults to
            ``gaussx.DenseSolver()`` — swap for ``CGSolver``,
            ``BBMMSolver``, ``ComposedSolver(solve=..., logdet=...)``, etc.
        jitter: Diagonal regularization added to the prior covariance
            for numerical stability. Not a noise model — use
            ``noise_var`` on `condition` for that.
        matrix_free: If ``True``, never form ``K``. The covariance is the
            `lineax.AddLinearOperator` of a matrix-free kernel operator
            (row blocks of ``K`` evaluated per matvec, ``O(block_size N)``
            memory, gradients included) and ``(jitter + noise_var) I``. Pair
            it with an iterative solver, normally
            `pyrox_gp.preconditioned_cg_solver`, which reads the noise from
            the sum; a dense solver materialises ``K`` again. NumPyro-aware
            kernels are resolved once per call through ``frozen()``.
        block_size: Rows of ``K`` per matvec step when ``matrix_free``.

    Examples:
        Exact GP regression on $10^5$ stations, hyperparameters by SVI:

        ```python
        import jax.random as jr
        import numpyro
        import numpyro.distributions as dist
        import pyrox_gp as px

        # X: (100_000, 2) station coordinates, y: (100_000,) readings
        solver = px.preconditioned_cg_solver(rank=500, key=jr.key(0))

        def model(X, y):
            kernel = px.Matern(nu=1.5, init_lengthscale=0.1)
            prior = px.GPPrior(kernel, X, solver=solver, matrix_free=True)
            noise_var = numpyro.param(
                "noise_var", 0.1, constraint=dist.constraints.positive
            )
            px.gp_factor("y", prior, y, noise_var)

        # numpyro.infer.SVI(model, lambda X, y: None, optax.adam(1e-2),
        #                   numpyro.infer.Trace_ELBO()).run(key, 500, X, y)
        ```
    """

    kernel: Kernel
    X: Float[Array, "N D"]
    mean_fn: Callable[[Float[Array, "N D"]], Float[Array, " N"]] | None = None
    solver: AbstractSolverStrategy | None = None
    jitter: float = 1e-6
    matrix_free: bool = eqx.field(static=True, default=False)
    block_size: int = eqx.field(static=True, default=256)

    def mean(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        """Evaluate the mean function at ``X``; zero by default."""
        if self.mean_fn is None:
            return jnp.zeros(X.shape[0], dtype=X.dtype)
        return self.mean_fn(X)

    def _matrix_free_operator(
        self, diagonal: Float[Array, ""]
    ) -> lx.AbstractLinearOperator:
        return matrix_free_operator(
            freeze_kernel(self.kernel), self.X, diagonal, block_size=self.block_size
        )

    def _prior_operator(self) -> lx.AbstractLinearOperator:
        if self.matrix_free:
            return self._matrix_free_operator(jnp.asarray(self.jitter))
        K = self.kernel(self.X, self.X)
        K = K.at[jnp.diag_indices_from(K)].add(self.jitter)
        return _psd_operator(K)

    def _noisy_operator(self, noise_var: Float[Array, ""]) -> lx.AbstractLinearOperator:
        if self.matrix_free:
            return self._matrix_free_operator(self.jitter + noise_var)
        K = self.kernel(self.X, self.X)
        K = K.at[jnp.diag_indices_from(K)].add(self.jitter + noise_var)
        return _psd_operator(K)

    def _resolved_solver(self) -> AbstractSolverStrategy:
        return DenseSolver() if self.solver is None else self.solver

    def log_prob(self, f: Float[Array, " N"]) -> Float[Array, ""]:
        r"""Marginal log-density of ``f`` under the GP prior.

        Computes $\log \mathcal{N}(f \mid \mu(X), K(X, X) + \text{jitter}\,I)$
        using `gaussx.log_marginal_likelihood`, so any solver strategy
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

        Wraps the prior in a `gaussx.MultivariateNormal` with
        the configured `solver`. This is the non-NumPyro analogue
        of `gp_sample` — useful for tests, diagnostics, and
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
        caches it in the returned `ConditionedGP`. The same
        ``jitter`` regularization configured on this prior is included
        alongside ``noise_var`` in the conditioned operator and solve, so
        every downstream predict / sample call sees the regularized
        covariance.

        The operator construction and any subsequent hyperparameter
        capture share one `_kernel_context`, so for Pattern B/C
        kernels with priors the cached operator and the resolved
        hyperparameters on the returned `ConditionedGP` come from
        the same draw. Downstream consumers (notably
        `pyrox_gp.PathwiseSampler`) reuse those values to stay
        consistent with the cached operator.
        """
        with _kernel_context(self.kernel):
            operator = self._noisy_operator(noise_var)
            resolved_hyperparams = _resolve_kernel_hyperparams(self.kernel)
        residual = y - self.mean(self.X)
        cache = build_prediction_cache(
            operator, residual, solver=self._resolved_solver()
        )
        return ConditionedGP(
            prior=self,
            y=y,
            noise_var=noise_var,
            cache=cache,
            operator=operator,
            resolved_hyperparams=resolved_hyperparams,
        )

    def condition_nongauss(
        self,
        likelihood: Likelihood,
        y: Float[Array, " N"],
        *,
        strategy: _NonGaussStrategy,
    ) -> NonGaussConditionedGP:
        """Condition on a non-Gaussian likelihood via a site-based strategy.

        Convenience that forwards to ``strategy.fit(self, likelihood, y)``.
        Pick any of the site-based strategies in
        `pyrox_gp._inference_nongauss`:
        `pyrox_gp.LaplaceInference`,
        `pyrox_gp.GaussNewtonInference`,
        `pyrox_gp.PosteriorLinearization`,
        `pyrox_gp.ExpectationPropagation`, or
        `pyrox_gp.QuasiNewtonInference`. Returns a
        `pyrox_gp.NonGaussConditionedGP` with the same
        ``predict`` / ``predict_mean`` / ``predict_var`` API as the
        Gaussian-likelihood `ConditionedGP`.

        Examples:

            from pyrox_gp import (
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

    Must be called inside the same `_kernel_context` as the
    kernel evaluation that downstream code wants to stay consistent
    with — Pattern B/C kernels register their priors per call, and a
    fresh outer context would resample.

    Returns ``None`` for kernels without those names so callers fall
    back to whatever default the consumer expects (e.g.
    `pyrox_gp._basis.draw_rff_cosine_basis` will read them itself).
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


class GradientPosterior(NamedTuple):
    r"""Gaussian posterior over $\nabla f$ at $M$ test points (point-major).

    Returned by `ConditionedGP.predict_gradient`. A ``NamedTuple``, so it is
    a pytree and passes through ``jit`` / ``vmap``.

    Attributes:
        mean: $\mathbb{E}[\nabla f(x_{*,m}) \mid y]$, shape ``(M, D)``.
        cov: The posterior covariance of the gradient, or ``None``.
            With ``cov="marginal"`` the per-point blocks, shape
            ``(M, D, D)``: ``cov[m]`` is
            $\operatorname{Cov}[\nabla f(x_{*,m}) \mid y]$. With
            ``cov="full"`` the joint, shape ``(M, D, M, D)``:
            ``cov[a, i, b, j]`` is
            $\operatorname{Cov}[\partial_i f(x_{*,a}), \partial_j f(x_{*,b}) \mid y]$;
            flatten it with
            ``einx.id("a i b j -> (a i) (b j)", cov)``. ``None``
            with ``cov="none"``.

    Examples:
        ```python
        post = cond.predict_gradient(X_star)  # X_star: (M, D)
        post.mean.shape  # (M, D)
        post.cov.shape  # (M, D, D)
        ```
    """

    mean: Float[Array, "M D"]
    cov: Float[Array, "M D D"] | Float[Array, "M D M D"] | None


def _derivative_kernel(kernel: Kernel) -> kl.AbstractKernel:
    """Freeze ``kernel`` to a plain kernellib kernel for differentiation.

    Must run inside the caller's `_kernel_context`, so the frozen
    hyperparameters are the draw every other kernel read in the call sees.
    A `pyrox_gp` kernel without a kernellib counterpart cannot be frozen;
    its ``NotImplementedError`` is re-raised with what to do about it.
    """
    try:
        frozen = freeze_kernel(kernel)
    except NotImplementedError as err:
        raise NotImplementedError(
            f"{err} The gradient posterior differentiates the kernel with "
            "kernellib's Derivative kernels, which need a plain kernellib "
            "kernel: give the class `_frozen_cls` / `_frozen_params` so "
            "`frozen()` works, or build the GPPrior on a kernellib kernel."
        ) from err
    return frozen


def _cross_gradient(
    kernel: kl.AbstractKernel,
    X_star: Float[Array, "M D"],
    X: Float[Array, "N D"],
) -> Float[Array, "DM N"]:
    r"""$G = \partial_1 K(X_*, X)$ in kind-major rows ``(d m)``.

    Row ``d * M + m`` holds $\partial k(x_{*,m}, x_n) / \partial x_{*,d}$,
    the ordering of ``kernellib.derivative_inputs(X_star, values=False)``.
    Each block is one ``kernellib.Derivative(kernel, dx=d)`` Gram, which
    also rejects kernels whose GP is not mean-square differentiable.
    """
    blocks = [kl.Derivative(kernel, dx=d)(X_star, X) for d in range(X_star.shape[-1])]
    return einx.id("d m n -> (d m) n", jnp.stack(blocks))


def _whitened_cross(
    cache: PredictionCache,
    A: Float[Array, "K N"],
    B: Float[Array, "L N"],
    *,
    solver: AbstractSolverStrategy,
) -> tuple[Float[Array, "N K"], Float[Array, "N L"]]:
    r"""Split $A\,\hat K^{-1} B^\top = P^\top Q$ using the cached solve.

    The two branches of `gaussx.predict_variance`, with no refactorisation
    of $\hat K$. With the cached Cholesky factor $\hat K = L L^\top$,
    $P = L^{-1} A^\top$ and $Q = L^{-1} B^\top$ (triangular solves only;
    $Q = P$ when ``B is A``, so the downdate $P^\top P$ is PSD). Without
    a factor (matrix-free / iterative solvers), $P = A^\top$ and
    $Q = \hat K^{-1} B^\top$, solved column by column with ``solver`` on
    the cached operator. Callers contract ``P`` and ``Q`` over ``n``.
    """
    if cache.factor is not None:
        P = solve_columns(cache.factor, einx.id("k n -> n k", A))
        if B is A:
            return P, P
        return P, solve_columns(cache.factor, einx.id("l n -> n l", B))
    if cache.operator is None:
        raise ValueError(
            "This PredictionCache holds only alpha; the gradient posterior "
            "needs the training operator (build it with GPPrior.condition)."
        )
    Q = solve_columns(cache.operator, einx.id("l n -> n l", B), solver=solver)
    return einx.id("k n -> n k", A), Q


def _mean_fn_gradient(
    mean_fn: Callable[[Float[Array, "N D"]], Float[Array, " N"]] | None,
    X_star: Float[Array, "M D"],
) -> Float[Array, "M D"]:
    r"""$\nabla m(x_{*,m})$ per test point by autodiff; zero without a mean."""
    if mean_fn is None:
        return jnp.zeros_like(X_star)
    return jax.vmap(jax.grad(lambda x: mean_fn(x[None])[0]))(X_star)


class ConditionedGP(eqx.Module):
    """GP conditioned on Gaussian-likelihood training observations.

    Holds the precomputed training solve ``alpha`` (via
    `gaussx.PredictionCache`) and the noisy covariance operator so
    predictions at multiple test sets reuse the training solve.
    """

    prior: GPPrior
    y: Float[Array, " N"]
    noise_var: Float[Array, ""]
    cache: PredictionCache
    operator: lx.AbstractLinearOperator
    resolved_hyperparams: tuple[Float[Array, ""], Float[Array, ""]] | None = None

    def predict_mean(self, X_star: Float[Array, "M D"]) -> Float[Array, " M"]:
        r"""$\mu_* = \mu(X_*) + K_{*f}\,\alpha$."""
        with _kernel_context(self.prior.kernel):
            K_cross = self.prior.kernel(X_star, self.prior.X)
        return self.prior.mean(X_star) + predict_mean(self.cache, K_cross)

    def predict_var(self, X_star: Float[Array, "M D"]) -> Float[Array, " M"]:
        r"""Diagonal predictive variance at ``X_*``.

        $$
        \sigma^2_{*,i} = k(x_{*,i}, x_{*,i})
            - K_{*f}[i,:] \cdot (K + \sigma^2 I)^{-1} K_{f*}[:,i]
        $$

        ``K_cross`` and ``K_diag`` are computed under one shared kernel
        context so Pattern B / C kernels with prior'd hyperparameters
        register their NumPyro sites once and reuse them across both
        kernel calls (and the cached training solve).
        """
        with _kernel_context(self.prior.kernel):
            K_cross = self.prior.kernel(X_star, self.prior.X)
            K_diag = self.prior.kernel.diag(X_star)
        return predict_variance(
            self.cache,
            K_cross,
            K_diag,
            solver=self.prior._resolved_solver(),
        )

    def predict(
        self, X_star: Float[Array, "M D"]
    ) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
        """Return ``(mean, variance)`` at ``X_*`` as a tuple.

        Both kernel evaluations share a single kernel context; see
        `predict_var`.
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
        `gaussx.MultivariateNormal`.
        """
        with _kernel_context(self.prior.kernel):
            mean = self.predict_mean(X_star)
            var = self.predict_var(X_star)
        std = jnp.sqrt(jnp.clip(var, min=0.0))
        eps = jax.random.normal(key, (n_samples, X_star.shape[0]), dtype=mean.dtype)
        # Scale per-point std across the S sample rows: (M,) ⊙ (S, M) → (S, M).
        return einx.multiply("m, s m -> s m", std, eps) + mean

    def predict_gradient(
        self,
        X_star: Float[Array, "M D"],
        *,
        cov: Literal["marginal", "full", "none"] = "marginal",
    ) -> GradientPosterior:
        r"""Gaussian posterior over the gradient $\nabla f(x_*)$ at ``X_*``.

        Differentiation is linear, so $\nabla f$ is jointly Gaussian with
        $y$ and conditioning commutes with it (Solak et al. 2003; Rasmussen
        & Williams 2006, §9.4). With $\hat K = K(X, X) + (\text{jitter} +
        \sigma^2) I$, $\alpha = \hat K^{-1}(y - m(X))$ (the cached
        training solve) and the cross-covariance
        $G(x_*) = \partial_1 K(x_*, X) \in \mathbb{R}^{D \times N}$,
        $G_{dn} = \partial k(x_*, x_n) / \partial x_{*,d}$:

        $$
        \mathbb{E}[\nabla f(x_*) \mid y] = \nabla m(x_*) + G(x_*)\,\alpha ,
        $$

        which equals $\nabla_{x_*} \mu_*(x_*)$ exactly (Johnson et al.
        2020, §3, with a mean function), and

        $$
        \operatorname{Cov}[\partial_i f(x_a), \partial_j f(x_b) \mid y]
        = \partial_{1,i} \partial_{2,j} k(x_a, x_b)
          - \partial_{1,i} K(x_a, X)\,\hat K^{-1}\,
            \partial_{1,j} K(x_b, X)^\top .
        $$

        With the cached factor $\hat K = L L^\top$ the downdate is
        $V^\top V$, $V = L^{-1} G^\top$ (triangular solves only); without
        one (``matrix_free=True`` / iterative solvers) it is
        $G\,\hat K^{-1} G^\top$ through the prior's ``solver``. The
        covariance is symmetrised but not clipped, like `predict_var`.

        ```text
        k  = freeze(kernel)                          # same draw as the call
        G  = [Derivative(k, dx=d)(X*, X) for d]      # (D*M, N), rows (d m)
        mean = rearrange(G @ alpha, "(d m) -> m d") + grad m(X*)
        P, Q = whitened_cross(cache, G, G)           # G K^-1 G^T = P^T Q
        marginal: H_m = d1 d2 k(x_m, x_m)            # (M, D, D)
                  cov = H_m - P_m^T Q_m
        full:     H = DerivativeIndexed(k)(Xt, Xt)   # (D*M, D*M)
                  cov = rearrange(H - P^T Q, "(i a) (j b) -> a i b j")
        ```

        The hyperparameters are read once, under one kernel context, so a
        kernel with priors registers each site once per call (here
        ``RBF.variance`` / ``RBF.lengthscale`` for `pyrox_gp.RBF`); the
        cached $\alpha$ comes from the draw taken in `GPPrior.condition`.

        References: Johnson, Laparra, Pérez-Suay, Mahecha & Camps-Valls
        (2020), PLOS ONE 15(10): e0235885, doi:10.1371/journal.pone.0235885;
        Solak et al. (2003), NeurIPS 15; Rasmussen & Williams (2006), §9.4.

        Args:
            X_star: Test inputs, shape ``(M, D)``.
            cov: ``"marginal"`` (default) for the per-point ``(M, D, D)``
                blocks, ``"full"`` for the joint ``(M, D, M, D)`` and
                ``"none"`` for the mean only (skips the ``N x DM`` solve).

        Returns:
            A `GradientPosterior` ``(mean, cov)`` in point-major layout.

        Raises:
            ValueError: For an unknown ``cov``, or a kernel whose GP is not
                mean-square differentiable (``Matern(nu=0.5)``, ``White``;
                kernellib's check).
            TypeError: For a Gram-only kernellib kernel.
            NotImplementedError: For a `pyrox_gp` kernel without a
                kernellib counterpart (no ``frozen()``).

        Examples:
            Sensitivity map with error bars (Johnson et al. 2020):

            ```python
            import jax.numpy as jnp
            import pyrox_gp as px

            # X: (N, 2), y: (N,)
            cond = px.GPPrior(px.RBF(init_lengthscale=0.5), X).condition(
                y, jnp.array(0.01)
            )
            post = cond.predict_gradient(X_star)  # X_star: (M, 2)
            post.mean  # (M, 2)   E[grad f]
            post.cov  # (M, 2, 2) Cov[grad f] per point
            sensitivity = jnp.sqrt(jnp.sum(post.mean**2, axis=-1))  # (M,)
            joint = cond.predict_gradient(X_star, cov="full").cov  # (M, 2, M, 2)
            ```
        """
        if cov not in ("marginal", "full", "none"):
            raise ValueError(f"cov must be 'marginal', 'full' or 'none', got {cov!r}.")
        D = X_star.shape[-1]
        with _kernel_context(self.prior.kernel):
            k = _derivative_kernel(self.prior.kernel)
            G = _cross_gradient(k, X_star, self.prior.X)
        mean = einx.id(
            "(d m) -> m d", einx.dot("k n, n -> k", G, self.cache.alpha), d=D
        ) + _mean_fn_gradient(self.prior.mean_fn, X_star)
        if cov == "none":
            return GradientPosterior(mean, None)
        P, Q = _whitened_cross(self.cache, G, G, solver=self.prior._resolved_solver())
        if cov == "marginal":
            H = jax.vmap(lambda x: jax.jacfwd(jax.grad(k.pairwise, 0), 1)(x, x))(X_star)
            Pm = einx.id("n (d m) -> m n d", P, d=D)
            Qm = einx.id("n (d m) -> m n d", Q, d=D)
            S = H - einx.dot("m n i, m n j -> m i j", Pm, Qm)
            return GradientPosterior(mean, 0.5 * (S + einx.id("m i j -> m j i", S)))
        Xt = kl.derivative_inputs(X_star, values=False)
        S = kl.DerivativeIndexed(k)(Xt, Xt) - einx.dot("n k, n l -> k l", P, Q)
        S = 0.5 * (S + einx.id("k l -> l k", S))
        return GradientPosterior(mean, einx.id("(i a) (j b) -> a i b j", S, i=D, j=D))

    def predict_var_gradient(self, X_star: Float[Array, "M D"]) -> Float[Array, "M D"]:
        r"""Gradient of the predictive variance, $\nabla_x \sigma^2_*(x)$.

        The acquisition gradient for variance-reduction / uncertainty-sampling
        active learning. From
        $\sigma^2_*(x) = k(x, x) - k(x, X)\,\hat K^{-1} k(X, x)$,

        $$
        \nabla_x \sigma^2_*(x) = \big[\partial_1 k + \partial_2 k\big](x, x)
            - 2\,G(x)\,\hat K^{-1} k(X, x),
        $$

        with $G(x) = \partial_1 K(x, X)$ as in `predict_gradient`. The first
        term vanishes for a stationary kernel. With the cached factor the
        second term is $-2\,(L^{-1} G^\top)^\top (L^{-1} k(X, x))$; it
        equals ``jax.grad`` of `predict_var` but reuses the training solve.
        The latent variance is differentiated (no observation noise), as in
        `predict_var`.

        ```text
        k = freeze(kernel)
        G = d1 K(X*, X)                     # (D*M, N)
        P, Q = whitened_cross(cache, G, K(X*, X))
        grad var[m] = grad_x k(x, x)|x_m - 2 P_m^T Q[:, m]
        ```

        Args:
            X_star: Test inputs, shape ``(M, D)``.

        Returns:
            $\nabla_x \sigma^2_*$ at each test point, shape ``(M, D)``.

        Raises:
            ValueError: For a kernel whose GP is not mean-square
                differentiable (see `predict_gradient`).
            NotImplementedError: For a `pyrox_gp` kernel without
                ``frozen()``.

        Examples:
            ```python
            # One gradient-ascent step on the predictive variance
            # (uncertainty sampling), X_cand: (M, D)
            X_next = X_cand + 0.1 * cond.predict_var_gradient(X_cand)
            ```
        """
        D = X_star.shape[-1]
        with _kernel_context(self.prior.kernel):
            k = _derivative_kernel(self.prior.kernel)
            G = _cross_gradient(k, X_star, self.prior.X)
            K_cross = k(X_star, self.prior.X)
        prior_term = jax.vmap(jax.grad(lambda x: k.pairwise(x, x)))(X_star)
        P, Q = _whitened_cross(
            self.cache, G, K_cross, solver=self.prior._resolved_solver()
        )
        Pm = einx.id("n (d m) -> m n d", P, d=D)
        return prior_term - 2.0 * einx.dot("m n d, n m -> m d", Pm, Q)

    def predict_mean_hessian(
        self, X_star: Float[Array, "M D"]
    ) -> Float[Array, "M D D"]:
        r"""Hessian of the predictive mean, $\nabla \nabla^\top \mu_*(x)$.

        $$
        \nabla \nabla^\top \mu_*(x) = \nabla^2 m(x)
            + \sum_n \alpha_n\, \nabla_x^2 k(x, x_n),
        $$

        by forward-over-reverse autodiff (``jax.hessian``) of
        $x \mapsto m(x) + k(x, X)\,\alpha$ with the frozen kernel and the
        cached $\alpha$, $O(N D^2)$ per point. It needs $k \in C^2$ in its
        first argument: fine for `pyrox_gp.RBF`,
        `pyrox_gp.RationalQuadratic` and ``Matern(nu=2.5)``; with
        ``Matern(nu=1.5)`` it is only defined away from the training
        inputs (the Hessian jumps at $x = x_n$).

        Args:
            X_star: Test inputs, shape ``(M, D)``.

        Returns:
            The symmetric ``(M, D, D)`` Hessians.

        Raises:
            ValueError: For a kernel whose GP is not mean-square
                differentiable (see `predict_gradient`).
            NotImplementedError: For a `pyrox_gp` kernel without
                ``frozen()``.

        Examples:
            ```python
            hess = cond.predict_mean_hessian(X_star)  # (M, D, D)
            curvature = jnp.trace(hess, axis1=-2, axis2=-1)  # Laplacian of mu
            ```
        """
        with _kernel_context(self.prior.kernel):
            k = _derivative_kernel(self.prior.kernel)
        kl.DerivativeIndexed(k)  # kernellib's mean-square differentiability check
        X, alpha, mean_fn = self.prior.X, self.cache.alpha, self.prior.mean_fn

        def mu(x: Float[Array, " D"]) -> Float[Array, ""]:
            value = einx.dot("n, n ->", k(x[None], X)[0], alpha)
            if mean_fn is not None:
                value = value + mean_fn(x[None])[0]
            return value

        return jax.vmap(jax.hessian(mu))(X_star)


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
    so the covariance matches what `GPPrior.condition` builds. Use
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
      `numpyro.infer.autoguide.AutoNormal` then approximates the
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
