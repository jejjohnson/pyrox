"""Multi-output GP model entry points — exact and sparse variational.

The model surface on top of the multi-output kernels in
`pyrox_gp._multi_output`. Two workflows:

* **Exact (dense)** — `MultiOutputGPPrior` /
  `MultiOutputConditionedGP` mirror the single-output
  `pyrox_gp.GPPrior` / `pyrox_gp.ConditionedGP` pair for
  vector-valued observations ``Y`` of shape ``(N, P)`` under a Gaussian
  likelihood with scalar or per-output noise. `mo_gp_factor` is
  the collapsed NumPyro hook.
* **Sparse variational (inducing inputs)** —
  `MultiOutputSparseGPPrior` assembles the SVGP blocks
  (block-diagonal ``K_uu`` over the ``Q`` latent processes, the mixed
  cross-covariance ``K_fu``, and the marginal prior diagonal) in the
  exact shapes the existing single-output guides
  (`pyrox_gp.FullRankGuide`, `MeanFieldGuide`,
  `WhitenedGuide`, `NaturalGuide`, `DeltaGuide`)
  consume, so one guide over the stacked ``Q * M`` inducing values
  serves the whole multi-output model. `mo_svgp_elbo` /
  `mo_svgp_factor` are the ELBO entry points.
* **OILMM (projected exact)** — `OILMMGPPrior` /
  `ConditionedOILMMGP` exploit the orthogonal mixing of an
  `pyrox_gp.OILMMKernel` to condition ``Q`` independent scalar
  GPs instead of one dense multi-output solve, matching the exact
  posterior under orthonormal mixing and isotropic noise.

All flattened quantities use the ``vec`` ordering of the Kronecker
operators returned by the kernels — output-major ``(p n)``, i.e. entry
``p * N + n`` is output ``p`` at input ``n``. `MultiOutputGPPrior`
and `MultiOutputSparseGPPrior` keep that convention internal:
user-facing observations, means, and predictions all carry the natural
``(N, P)`` layout.

As with the single-output entry points, all linear algebra delegates to
``gaussx`` and every entry accepts any ``gaussx.AbstractSolverStrategy``
(default ``gaussx.DenseSolver()``).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpyro
from gaussx import (
    AbstractIntegrator as GaussxIntegrator,
    AbstractSolverStrategy,
    BlockDiag,
    DenseSolver,
    MultivariateNormal,
    PredictionCache,
    build_prediction_cache,
    log_marginal_likelihood,
    predict_mean,
    predict_variance,
    variational_elbo_gaussian,
)
from jaxtyping import Array, Float

from pyrox_gp._context import _kernel_contexts
from pyrox_gp._inference import _ell_numerical
from pyrox_gp._likelihoods import GaussianLikelihood
from pyrox_gp._models import ConditionedGP, GPPrior
from pyrox_gp._multi_output import (
    ICMKernel,
    LMCKernel,
    MultiOutputInducingVariables,
    OILMMKernel,
    SharedInducingPoints,
)
from pyrox_gp._protocols import Guide, Kernel, Likelihood


MultiOutputKernel = LMCKernel | ICMKernel | OILMMKernel
"""Union of the multi-output kernels accepted by the exact model layer."""


def _latent_kernels(kernel: MultiOutputKernel) -> tuple[Kernel, ...]:
    """Return one latent kernel per latent process ``q``.

    `ICMKernel` shares a single kernel instance across its ``Q``
    latents, so the same instance is repeated — downstream
    `_kernel_contexts` deduplicates by identity, and the inducing
    builders accept repeated instances as intentional tying.
    """
    if isinstance(kernel, ICMKernel):
        return (kernel.kernel,) * kernel.num_latents
    return kernel.kernels


@contextlib.contextmanager
def _mo_kernel_context(kernel: MultiOutputKernel) -> Iterator[None]:
    """Scope every latent kernel of ``kernel`` under one shared context.

    The multi-output kernels open their own per-call contexts inside
    each evaluation, but a model entry point that makes *several*
    kernel calls (e.g. ``K_ff`` at condition time, then ``K_*f`` and the
    prior diagonal at predict time within one trace) must hold one outer
    context so Pattern B / C latent kernels with priors register their
    NumPyro sites once and share a single hyperparameter draw. The inner
    per-call contexts are reentrant and become no-ops.
    """
    with _kernel_contexts(_latent_kernels(kernel)):
        yield


def _flatten_outputs(Y: Float[Array, "N P"]) -> Float[Array, " PN"]:
    """Flatten ``(N, P)`` values to the output-major ``vec`` layout ``(p n)``."""
    return einx.id("n p -> (p n)", Y)


def _unflatten_outputs(v: Float[Array, " PN"], num_outputs: int) -> Float[Array, "N P"]:
    """Invert `_flatten_outputs` back to the ``(N, P)`` layout."""
    return einx.id("(p n) -> n p", v, p=num_outputs)


def _validate_targets(Y: Float[Array, "N P"], num_outputs: int) -> None:
    if Y.ndim != 2 or Y.shape[1] != num_outputs:
        raise ValueError(
            f"Y must have shape (N, {num_outputs}) to match the kernel's "
            f"output channels; got {Y.shape}."
        )


def _flat_noise(
    noise_var: Float[Array, ""] | Float[Array, " P"],
    num_points: int,
    num_outputs: int,
) -> Float[Array, " PN"]:
    """Expand scalar or per-output noise to the flattened ``(p n)`` diagonal."""
    noise = jnp.asarray(noise_var)
    if noise.ndim == 0:
        return jnp.full(num_outputs * num_points, noise)
    if noise.shape == (num_outputs,):
        # Per-output noise repeats along the trailing input axis of the
        # output-major layout: entry p * N + n gets noise[p].
        return jnp.repeat(noise, num_points)
    raise ValueError(
        f"noise_var must be a scalar or have shape ({num_outputs},) for "
        f"per-output noise; got shape {noise.shape}."
    )


def _psd_operator(K: Float[Array, "N N"]) -> lx.AbstractLinearOperator:
    """Wrap a Gram matrix as a PSD ``lineax`` operator."""
    return lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)


class MultiOutputGPPrior(eqx.Module):
    """Finite-dimensional multi-output GP prior over fixed training inputs.

    The vector-valued analogue of `pyrox_gp.GPPrior`: holds a
    multi-output kernel (`pyrox_gp.LMCKernel`,
    `pyrox_gp.ICMKernel`, or `pyrox_gp.OILMMKernel`),
    training inputs ``X``, an optional mean function over the ``P``
    output channels, a solver strategy, and a diagonal jitter.

    The prior covariance is the kernel's full ``(P*N, P*N)`` Gram over
    the isotopic observation set — every output observed at every input.
    The dense workflow accepts all three kernel families; structured
    (Kronecker-exact / projected) fast paths can layer on later without
    changing this surface.

    Attributes:
        kernel: A multi-output kernel exposing ``full_covariance``,
            ``cross_covariance``, and ``diag``.
        X: Training inputs of shape ``(N, D)``.
        mean_fn: Callable ``X -> (N, P)`` or ``None`` for the zero mean.
        solver: Any ``gaussx.AbstractSolverStrategy``. Defaults to
            ``gaussx.DenseSolver()``.
        jitter: Diagonal regularization added to the prior covariance.
            Not a noise model — use ``noise_var`` on `condition`
            for that.
    """

    kernel: MultiOutputKernel
    X: Float[Array, "N D"]
    mean_fn: Callable[[Float[Array, "N D"]], Float[Array, "N P"]] | None = None
    solver: AbstractSolverStrategy | None = None
    jitter: float = 1e-6

    @property
    def num_outputs(self) -> int:
        """Number of observed output channels ``P``."""
        return self.kernel.num_outputs

    def mean(self, X: Float[Array, "N D"]) -> Float[Array, "N P"]:
        """Evaluate the mean function at ``X``; zero by default."""
        if self.mean_fn is None:
            return jnp.zeros((X.shape[0], self.num_outputs), dtype=X.dtype)
        return self.mean_fn(X)

    def _flat_mean(self, X: Float[Array, "N D"]) -> Float[Array, " PN"]:
        return _flatten_outputs(self.mean(X))

    def _prior_operator(self) -> lx.AbstractLinearOperator:
        K = self.kernel.full_covariance(self.X)
        K = K.at[jnp.diag_indices_from(K)].add(self.jitter)
        return _psd_operator(K)

    def _noisy_operator(
        self, noise_var: Float[Array, ""] | Float[Array, " P"]
    ) -> lx.AbstractLinearOperator:
        K = self.kernel.full_covariance(self.X)
        noise = _flat_noise(noise_var, self.X.shape[0], self.num_outputs)
        K = K.at[jnp.diag_indices_from(K)].add(self.jitter + noise)
        return _psd_operator(K)

    def _resolved_solver(self) -> AbstractSolverStrategy:
        return DenseSolver() if self.solver is None else self.solver

    def log_prob(self, F: Float[Array, "N P"]) -> Float[Array, ""]:
        r"""Marginal log-density of the latent functions ``F`` under the prior.

        Computes
        $\log \mathcal{N}(\mathrm{vec}(F) \mid \mathrm{vec}(\mu(X)),
        K_{ff} + \text{jitter}\,I)$ over the flattened isotopic
        observation set.
        """
        _validate_targets(F, self.num_outputs)
        with _mo_kernel_context(self.kernel):
            operator = self._prior_operator()
        return log_marginal_likelihood(
            self._flat_mean(self.X),
            operator,
            _flatten_outputs(F),
            solver=self._resolved_solver(),
        )

    def sample(self, key: Array) -> Float[Array, "N P"]:
        """Draw one prior function sample over all outputs, shape ``(N, P)``."""
        with _mo_kernel_context(self.kernel):
            operator = self._prior_operator()
        mvn = MultivariateNormal(
            self._flat_mean(self.X), operator, solver=self._resolved_solver()
        )
        return _unflatten_outputs(mvn.sample(key), self.num_outputs)

    def condition(
        self,
        Y: Float[Array, "N P"],
        noise_var: Float[Array, ""] | Float[Array, " P"],
    ) -> MultiOutputConditionedGP:
        """Condition on Gaussian-likelihood observations ``Y`` of shape ``(N, P)``.

        ``noise_var`` is either a scalar (shared across outputs) or a
        ``(P,)`` vector of per-output noise variances. Precomputes the
        flattened training solve
        ``alpha = (K_ff + (jitter + noise) I)^{-1} vec(Y - mu(X))`` and
        caches it in the returned `MultiOutputConditionedGP`, so
        predictions at multiple test sets reuse the factorization.
        """
        _validate_targets(Y, self.num_outputs)
        with _mo_kernel_context(self.kernel):
            operator = self._noisy_operator(noise_var)
        residual = _flatten_outputs(Y - self.mean(self.X))
        cache = build_prediction_cache(
            operator, residual, solver=self._resolved_solver()
        )
        return MultiOutputConditionedGP(
            prior=self,
            Y=Y,
            noise_var=jnp.asarray(noise_var),
            cache=cache,
            operator=operator,
        )


class MultiOutputConditionedGP(eqx.Module):
    """Multi-output GP conditioned on Gaussian-likelihood observations.

    Holds the precomputed flattened training solve (via
    `gaussx.PredictionCache`) and the noisy covariance operator;
    predictions return per-output moments in the ``(N, P)`` layout.
    """

    prior: MultiOutputGPPrior
    Y: Float[Array, "N P"]
    noise_var: Float[Array, ""] | Float[Array, " P"]
    cache: PredictionCache
    operator: lx.AbstractLinearOperator

    def predict_mean(self, X_star: Float[Array, "M D"]) -> Float[Array, "M P"]:
        r"""Predictive mean $\mu(X_*) + K_{*f}\,\alpha$ per output channel."""
        with _mo_kernel_context(self.prior.kernel):
            K_cross = self.prior.kernel.cross_covariance(X_star, self.prior.X)
        flat = predict_mean(self.cache, K_cross)
        return self.prior.mean(X_star) + _unflatten_outputs(
            flat, self.prior.num_outputs
        )

    def predict_var(self, X_star: Float[Array, "M D"]) -> Float[Array, "M P"]:
        r"""Marginal predictive variance per input and output, shape ``(M, P)``.

        $$
        \sigma^2_{*}[i, p] = k_{pp}(x_i, x_i)
            - K_{*f}[(p,i),:]\, (K_{ff} + \Sigma)^{-1}\, K_{f*}[:,(p,i)]
        $$

        Both kernel evaluations share one context — see
        `pyrox_gp.ConditionedGP.predict_var` for the rationale.
        """
        with _mo_kernel_context(self.prior.kernel):
            K_cross = self.prior.kernel.cross_covariance(X_star, self.prior.X)
            K_diag = _flatten_outputs(self.prior.kernel.diag(X_star))
        flat = predict_variance(
            K_cross,
            K_diag,
            self.operator,
            solver=self.prior._resolved_solver(),
        )
        return _unflatten_outputs(flat, self.prior.num_outputs)

    def predict(
        self, X_star: Float[Array, "M D"]
    ) -> tuple[Float[Array, "M P"], Float[Array, "M P"]]:
        """Return per-output ``(mean, variance)`` at ``X_*`` as ``(M, P)`` pairs."""
        with _mo_kernel_context(self.prior.kernel):
            return self.predict_mean(X_star), self.predict_var(X_star)

    def sample(
        self,
        key: Array,
        X_star: Float[Array, "M D"],
        n_samples: int = 1,
    ) -> Float[Array, "S M P"]:
        """Sample from the diagonal predictive ``N(mean, diag(var))``.

        Samples are independent per input and output channel, matching
        the single-output `pyrox_gp.ConditionedGP.sample`
        convention. For correlated joint samples, build the full
        predictive covariance explicitly and draw from
        `gaussx.MultivariateNormal`.
        """
        with _mo_kernel_context(self.prior.kernel):
            mean = self.predict_mean(X_star)
            var = self.predict_var(X_star)
        std = jnp.sqrt(jnp.clip(var, min=0.0))
        eps = jax.random.normal(key, (n_samples, *mean.shape), dtype=mean.dtype)
        # Scale per-point, per-output std across the S sample rows.
        return einx.multiply("m p, s m p -> s m p", std, eps) + mean


def mo_gp_factor(
    name: str,
    prior: MultiOutputGPPrior,
    Y: Float[Array, "N P"],
    noise_var: Float[Array, ""] | Float[Array, " P"],
) -> None:
    """Register the collapsed multi-output GP marginal likelihood with NumPyro.

    Adds ``log N(vec(Y) | vec(mu), K_ff + (jitter + noise) I)`` to the
    NumPyro trace as ``numpyro.factor(name, ...)`` — the multi-output
    analogue of `pyrox_gp.gp_factor`. ``noise_var`` is a scalar or
    a ``(P,)`` per-output vector.
    """
    _validate_targets(Y, prior.num_outputs)
    with _mo_kernel_context(prior.kernel):
        operator = prior._noisy_operator(noise_var)
    logp = log_marginal_likelihood(
        prior._flat_mean(prior.X),
        operator,
        _flatten_outputs(Y),
        solver=prior._resolved_solver(),
    )
    numpyro.factor(name, logp)


class MultiOutputSparseGPPrior(eqx.Module):
    r"""Sparse multi-output GP prior over shared inducing inputs.

    The multi-output analogue of `pyrox_gp.SparseGPPrior`: the
    ``Q`` latent processes of an `pyrox_gp.LMCKernel` (or the
    shared-kernel latents of an `pyrox_gp.ICMKernel`) each carry
    inducing values at the same inducing locations ``Z``, giving the
    stacked prior

    $$
    p(u) = \mathcal{N}\!\bigl(0,\,
        \mathrm{blockdiag}(K^{(1)}_{ZZ}, \dots, K^{(Q)}_{ZZ})
        + \text{jitter}\,I\bigr)
    $$

    over ``Q * M`` inducing values. `predictive_blocks` returns
    ``(K_uu_op, K_fu, K_ff_diag)`` in exactly the shapes the
    single-output guides consume — a guide initialized with
    ``num_inducing = prior.num_inducing`` works unchanged, and the
    block-diagonal ``K_uu`` keeps its structure through
    ``gaussx.cholesky`` / ``solve`` / ``logdet`` so the ``(Q M)^3``
    cost decomposes into ``Q`` independent ``M^3`` factorizations.

    `OILMMKernel` is rejected (its efficiency comes from
    orthogonal projection, not from this inducing decomposition), and so
    is `ICMKernel` with a non-zero ``kappa`` — see
    `pyrox_gp.MultiOutputInducingVariables.from_kernel`.

    Attributes:
        kernel: `pyrox_gp.LMCKernel` or `pyrox_gp.ICMKernel`.
        inducing: `pyrox_gp.SharedInducingPoints` holding ``Z``.
        mean_fn: Callable ``X -> (N, P)`` or ``None`` for the zero mean.
            Added onto the flattened predictive mean by
            `mo_svgp_elbo` and `predict`; not part of the
            inducing prior (standard SVGP convention).
        solver: Any ``gaussx.AbstractSolverStrategy``. Defaults to
            ``gaussx.DenseSolver()``.
        jitter: Diagonal regularization added to every per-latent
            ``K_zz`` block.
    """

    kernel: LMCKernel | ICMKernel
    inducing: SharedInducingPoints
    mean_fn: Callable[[Float[Array, "N D"]], Float[Array, "N P"]] | None = None
    solver: AbstractSolverStrategy | None = None
    jitter: float = 1e-6

    def __check_init__(self) -> None:
        # Delegates kernel-family and ICM-kappa validation; constructing
        # the inducing variables is free of kernel evaluations.
        MultiOutputInducingVariables.from_kernel(self.kernel, self.inducing)

    @property
    def num_outputs(self) -> int:
        """Number of observed output channels ``P``."""
        return self.kernel.num_outputs

    @property
    def num_latents(self) -> int:
        """Number of latent scalar GPs ``Q``."""
        return self.kernel.num_latents

    @property
    def num_inducing(self) -> int:
        """Total stacked inducing values ``Q * M`` — size a guide with this."""
        return self.num_latents * self.inducing.num_inducing

    def mean(self, X: Float[Array, "N D"]) -> Float[Array, "N P"]:
        """Evaluate the mean function at ``X``; zero by default."""
        if self.mean_fn is None:
            return jnp.zeros((X.shape[0], self.num_outputs), dtype=X.dtype)
        return self.mean_fn(X)

    def _resolved_solver(self) -> AbstractSolverStrategy:
        return DenseSolver() if self.solver is None else self.solver

    def predictive_blocks(
        self, X: Float[Array, "N D"]
    ) -> tuple[
        lx.AbstractLinearOperator,
        Float[Array, "PN QM"],
        Float[Array, " PN"],
    ]:
        r"""Return ``(K_uu_op, K_fu, K_ff_diag)`` under one shared kernel context.

        ``K_uu_op`` is a ``gaussx.BlockDiag`` of the ``Q`` per-latent
        ``K_zz + \text{jitter}\,I`` blocks (jitter folded into each block
        so the block-diagonal dispatch survives); ``K_fu`` is the
        ``(P*N, Q*M)`` cross-covariance in the output-major ``(p n)``
        row layout; ``K_ff_diag`` is the flattened marginal prior
        variance. All kernel evaluations share one context per unique
        latent kernel instance, so Pattern B / C kernels with priors
        register their NumPyro sites once across all three blocks.
        """
        iv = MultiOutputInducingVariables.from_kernel(self.kernel, self.inducing)
        kernels = _latent_kernels(self.kernel)
        with _kernel_contexts(kernels):
            K_uu_blocks, K_uf_blocks = self.inducing.inducing_blocks(X, kernels)
            K_ff_diag = _flatten_outputs(self.kernel.diag(X))
        jittered = tuple(
            B.at[jnp.diag_indices_from(B)].add(self.jitter) for B in K_uu_blocks
        )
        K_uu_op = BlockDiag(*(_psd_operator(B) for B in jittered))
        K_uf = iv._assemble_K_uf(K_uf_blocks)
        K_fu = einx.id("u f -> f u", K_uf)
        return K_uu_op, K_fu, K_ff_diag

    def predict(
        self,
        guide: Guide,
        X_star: Float[Array, "M D"],
    ) -> tuple[Float[Array, "M P"], Float[Array, "M P"]]:
        """Per-output predictive ``(mean, variance)`` under ``q(u)`` from ``guide``.

        Assembles `predictive_blocks` at ``X_star``, routes them
        through ``guide.predict``, adds the mean function, and
        unflattens to the ``(M, P)`` layout. Handles the ``vec``
        bookkeeping so callers never touch the flattened layout.
        """
        K_uu_op, K_fu, K_ff_diag = self.predictive_blocks(X_star)
        f_loc, f_var = guide.predict(K_fu, K_uu_op, K_ff_diag)  # ty: ignore[unresolved-attribute]
        f_loc = f_loc + _flatten_outputs(self.mean(X_star))
        return (
            _unflatten_outputs(f_loc, self.num_outputs),
            _unflatten_outputs(f_var, self.num_outputs),
        )


def mo_svgp_elbo(
    prior: MultiOutputSparseGPPrior,
    guide: Guide,
    likelihood: Likelihood,
    X: Float[Array, "N D"],
    Y: Float[Array, "N P"],
    *,
    integrator: GaussxIntegrator | None = None,
) -> Float[Array, ""]:
    r"""Structured multi-output SVGP ELBO as a differentiable scalar.

    The multi-output analogue of `pyrox_gp.svgp_elbo`:

    $$
    \mathcal{L} = \sum_{n,p} \mathbb{E}_{q(f_{p}(x_n))}
                  [\log p(y_{n,p} \mid f_{p}(x_n))]
                - \mathrm{KL}[q(u) \| p(u)]
    $$

    with ``q(u)`` a single guide over the stacked ``Q * M`` inducing
    values and the KL evaluated against the block-diagonal inducing
    prior. Observations ``Y`` carry the natural ``(N, P)`` layout;
    the flattening into the guide's vectorized convention is internal.

    For `pyrox_gp.GaussianLikelihood` the expected
    log-likelihood is closed-form; every scalar point-wise likelihood
    works through a gaussx integrator, applied independently per
    ``(n, p)`` observation.

    Args:
        prior: Sparse multi-output GP prior.
        guide: Variational guide sized ``num_inducing = prior.num_inducing``.
        likelihood: Observation model applied point-wise per ``(n, p)``.
        X: Training inputs, shape ``(N, D)``.
        Y: Training targets, shape ``(N, P)``.
        integrator: gaussx integrator for the per-point ELL. Required
            for non-conjugate likelihoods; ignored for
            `GaussianLikelihood`.

    Returns:
        Scalar ELBO value (higher is better).

    Raises:
        ValueError: If ``Y`` has the wrong shape, or a non-conjugate
            likelihood is used without an integrator.
    """
    _validate_targets(Y, prior.num_outputs)
    K_uu_op, K_fu, K_ff_diag = prior.predictive_blocks(X)

    f_loc, f_var = guide.predict(K_fu, K_uu_op, K_ff_diag)  # ty: ignore[unresolved-attribute]
    f_loc = f_loc + _flatten_outputs(prior.mean(X))

    kl = guide.kl_divergence(K_uu_op)  # ty: ignore[unresolved-attribute]
    y = _flatten_outputs(Y)

    if isinstance(likelihood, GaussianLikelihood):
        return variational_elbo_gaussian(
            y,
            f_loc,
            f_var,
            likelihood.noise_var,  # ty: ignore[invalid-argument-type]
            kl,
        )

    if integrator is None:
        raise ValueError(
            "Non-conjugate likelihoods require an integrator "
            "(e.g. gaussx.GaussHermiteIntegrator). "
            "Pass integrator=GaussHermiteIntegrator(order=20)."
        )
    ell = _ell_numerical(likelihood, y, f_loc, f_var, integrator)
    return ell - kl


def mo_svgp_factor(
    name: str,
    prior: MultiOutputSparseGPPrior,
    guide: Guide,
    likelihood: Likelihood,
    X: Float[Array, "N D"],
    Y: Float[Array, "N P"],
    *,
    integrator: GaussxIntegrator | None = None,
) -> None:
    """Register the multi-output SVGP ELBO as a NumPyro factor site.

    Wraps `mo_svgp_elbo` in ``numpyro.factor`` so it plugs into
    ``numpyro.infer.SVI`` + ``Trace_ELBO``, mirroring
    `pyrox_gp.svgp_factor`.
    """
    numpyro.factor(
        name,
        mo_svgp_elbo(prior, guide, likelihood, X, Y, integrator=integrator),
    )


def _per_output_noise(
    noise_var: Float[Array, ""] | Float[Array, " P"],
    num_outputs: int,
) -> Float[Array, " P"]:
    """Expand scalar noise to a per-output ``(P,)`` vector; validate shape."""
    noise = jnp.asarray(noise_var)
    if noise.ndim == 0:
        return jnp.full(num_outputs, noise)
    if noise.shape == (num_outputs,):
        return noise
    raise ValueError(
        f"noise_var must be a scalar or have shape ({num_outputs},) for "
        f"per-output noise; got shape {noise.shape}."
    )


class OILMMGPPrior(eqx.Module):
    r"""Orthogonal-mixing multi-output GP prior with a projected exact posterior.

    The OILMM workflow (Bruinsma et al., 2020): with a semi-orthogonal
    mixing matrix (``W^T W = I``) the multi-output regression problem
    projects into ``Q`` *independent scalar GP problems* — conditioning
    costs ``Q`` factorizations of ``(N, N)`` latent Grams instead of one
    ``(P*N, P*N)`` dense solve, while giving the same posterior over the
    latent processes as the exact dense model.

    Exactness holds when the mixing is orthonormal and the observation
    noise is isotropic (scalar ``noise_var``). Per-output noise vectors
    are supported through the standard OILMM projected-noise
    approximation ``s_latent = (W ** 2)^T s`` — see
    `pyrox_gp.OILMMKernel.project`. For exact inference under
    general noise, use the dense `MultiOutputGPPrior` instead.
    That dense model is also the collapsed NumPyro path
    (`mo_gp_factor`) for `OILMMKernel`.

    Attributes:
        kernel: `pyrox_gp.OILMMKernel` holding the latent kernels
            and the semi-orthogonal mixing matrix. Orthogonality is
            assumed, not enforced here — construct the kernel with
            ``check_orthogonal=True`` to verify.
        X: Training inputs of shape ``(N, D)``.
        mean_fn: Callable ``X -> (N, P)`` or ``None`` for the zero mean.
            Subtracted in output space before projection and added back
            onto predictions.
        solver: Any ``gaussx.AbstractSolverStrategy``, forwarded to the
            per-latent `pyrox_gp.GPPrior` instances.
        jitter: Diagonal regularization forwarded to each latent prior.
    """

    kernel: OILMMKernel
    X: Float[Array, "N D"]
    mean_fn: Callable[[Float[Array, "N D"]], Float[Array, "N P"]] | None = None
    solver: AbstractSolverStrategy | None = None
    jitter: float = 1e-6

    @property
    def num_outputs(self) -> int:
        """Number of observed output channels ``P``."""
        return self.kernel.num_outputs

    @property
    def num_latents(self) -> int:
        """Number of latent scalar GPs ``Q``."""
        return self.kernel.num_latents

    def mean(self, X: Float[Array, "N D"]) -> Float[Array, "N P"]:
        """Evaluate the mean function at ``X``; zero by default."""
        if self.mean_fn is None:
            return jnp.zeros((X.shape[0], self.num_outputs), dtype=X.dtype)
        return self.mean_fn(X)

    def latent_priors(self) -> tuple[GPPrior, ...]:
        """Return one scalar `pyrox_gp.GPPrior` per latent process."""
        return tuple(
            GPPrior(kernel=kernel, X=self.X, solver=self.solver, jitter=self.jitter)
            for kernel in self.kernel.kernels
        )

    def sample(self, key: Array) -> Float[Array, "N P"]:
        """Draw one prior function sample over all outputs, shape ``(N, P)``.

        Draws each latent process from its scalar prior and mixes
        through ``W`` — equivalent to (and cheaper than) sampling the
        full ``(P*N, P*N)`` OILMM covariance.
        """
        keys = jax.random.split(key, self.num_latents)
        with _kernel_contexts(self.kernel.kernels):
            latents = jnp.stack(
                [
                    prior.sample(k)
                    for prior, k in zip(self.latent_priors(), keys, strict=True)
                ],
                axis=-1,
            )
        # Mix latent draws into output space: f = g W^T.
        return self.mean(self.X) + einx.dot(
            "n q, p q -> n p", latents, self.kernel.mixing
        )

    def condition(
        self,
        Y: Float[Array, "N P"],
        noise_var: Float[Array, ""] | Float[Array, " P"],
    ) -> ConditionedOILMMGP:
        """Condition on observations ``Y`` via the orthogonal projection.

        Projects the mean-subtracted observations and the noise into
        latent space (`pyrox_gp.OILMMKernel.project`) and
        conditions each scalar latent `pyrox_gp.GPPrior`
        independently. All latent kernel evaluations share one context
        per unique kernel instance, so a priored kernel reused across
        latents registers its NumPyro sites once.
        """
        _validate_targets(Y, self.num_outputs)
        noise = _per_output_noise(noise_var, self.num_outputs)
        residual = Y - self.mean(self.X)
        Y_latent, noise_latent = self.kernel.project(residual, noise)
        with _kernel_contexts(self.kernel.kernels):
            latents = tuple(
                prior.condition(Y_latent[:, q], noise_latent[q])
                for q, prior in enumerate(self.latent_priors())
            )
        return ConditionedOILMMGP(
            prior=self,
            Y=Y,
            noise_var=noise,
            latents=latents,
        )


class ConditionedOILMMGP(eqx.Module):
    """OILMM posterior — ``Q`` conditioned scalar GPs plus the mixing.

    Holds one `pyrox_gp.ConditionedGP` per latent process; every
    prediction runs the scalar predictives and mixes back to output
    space via `pyrox_gp.OILMMKernel.back_project` (means through
    ``W``, variances through ``W ** 2``).
    """

    prior: OILMMGPPrior
    Y: Float[Array, "N P"]
    noise_var: Float[Array, " P"]
    latents: tuple[ConditionedGP, ...]

    def _latent_means(self, X_star: Float[Array, "M D"]) -> Float[Array, "M Q"]:
        return jnp.stack([cond.predict_mean(X_star) for cond in self.latents], axis=-1)

    def _latent_vars(self, X_star: Float[Array, "M D"]) -> Float[Array, "M Q"]:
        return jnp.stack([cond.predict_var(X_star) for cond in self.latents], axis=-1)

    def predict_mean(self, X_star: Float[Array, "M D"]) -> Float[Array, "M P"]:
        """Per-output predictive mean, shape ``(M, P)``."""
        with _kernel_contexts(self.prior.kernel.kernels):
            latent_means = self._latent_means(X_star)
        # Mix latent means into output space: f = g W^T.
        f_mean = einx.dot("m q, p q -> m p", latent_means, self.prior.kernel.mixing)
        return self.prior.mean(X_star) + f_mean

    def predict_var(self, X_star: Float[Array, "M D"]) -> Float[Array, "M P"]:
        """Per-output marginal predictive variance, shape ``(M, P)``."""
        with _kernel_contexts(self.prior.kernel.kernels):
            latent_vars = self._latent_vars(X_star)
        # Independent latents: output variance mixes through W ** 2.
        return einx.dot(
            "m q, p q -> m p", latent_vars, jnp.square(self.prior.kernel.mixing)
        )

    def predict(
        self, X_star: Float[Array, "M D"]
    ) -> tuple[Float[Array, "M P"], Float[Array, "M P"]]:
        """Return per-output ``(mean, variance)`` at ``X_*`` as ``(M, P)`` pairs.

        Both latent sweeps share one kernel context; the back-projection
        delegates to `pyrox_gp.OILMMKernel.back_project`.
        """
        with _kernel_contexts(self.prior.kernel.kernels):
            latent_means = self._latent_means(X_star)
            latent_vars = self._latent_vars(X_star)
        f_mean, f_var = self.prior.kernel.back_project(latent_means, latent_vars)
        return self.prior.mean(X_star) + f_mean, f_var

    def sample(
        self,
        key: Array,
        X_star: Float[Array, "M D"],
        n_samples: int = 1,
    ) -> Float[Array, "S M P"]:
        """Sample from the diagonal predictive ``N(mean, diag(var))``.

        Matches the `MultiOutputConditionedGP.sample` convention —
        independent per input and output channel.
        """
        with _kernel_contexts(self.prior.kernel.kernels):
            mean, var = self.predict(X_star)
        std = jnp.sqrt(jnp.clip(var, min=0.0))
        eps = jax.random.normal(key, (n_samples, *mean.shape), dtype=mean.dtype)
        # Scale per-point, per-output std across the S sample rows.
        return einx.multiply("m p, s m p -> s m p", std, eps) + mean
