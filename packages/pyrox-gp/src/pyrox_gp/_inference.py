"""Sparse variational GP inference entry points.

Three paths, all sharing the same likelihood + guide + prior surface:

* `svgp_elbo` — pure-function ELBO returning a differentiable
  scalar. Use with ``optax`` / ``equinox`` for optimization outside
  NumPyro.
* `svgp_factor` — wraps `svgp_elbo` in
  ``numpyro.factor`` so it plugs into ``numpyro.infer.SVI`` +
  ``Trace_ELBO``. NumPyro sees one factor site; the *actual* ELBO uses
  the efficient structured computation.
* `ConjugateVI` — natural-gradient / CVI update loop. Operates
  in natural-parameter space via
  `NaturalGuide.natural_update`; not NumPyro-coupled.

All Gaussian linear algebra delegates to ``gaussx``.
"""

from __future__ import annotations

from collections.abc import Callable

import einx
import jax
import jax.numpy as jnp
import lineax as lx
import numpyro
from gaussx import (
    AbstractIntegrator as GaussxIntegrator,
    GaussianState,
    mean_expectation,
    solve_matrix,
    symmetrize,
    variational_elbo_gaussian,
)
from jaxtyping import Array, Float, PyTree
from pyrox._core import call_context

from pyrox_gp._guides import NaturalGuide
from pyrox_gp._likelihoods import GaussianLikelihood
from pyrox_gp._protocols import Guide, Likelihood, likelihood_accepts_inputs
from pyrox_gp._sparse import SparseGPPrior


def _site_expectation(
    integrator: GaussxIntegrator,
    fn: Callable[[Float[Array, ""], PyTree], Float[Array, ""]],
    mean_n: Float[Array, ""],
    var_n: Float[Array, ""],
    y_n: PyTree,
) -> Float[Array, ""]:
    r"""Gaussian expectation at one site, $\mathbb{E}_{f_n}[\text{fn}(f_n, y_n)]$.

    With $f_n \sim \mathcal{N}(\text{mean}_n, \text{var}_n)$. Point rules take
    the weighted sum alone through `gaussx.mean_expectation`, with no output
    covariance built; a rule with no points, such as Taylor, falls back to
    ``integrate``. Any `gaussx.AbstractIntegrator` works.

    Args:
        integrator: The rule for the 1-D Gaussian.
        fn: The integrand, from the site's scalar $f_n$ and its $y_n$.
        mean_n: The site mean.
        var_n: The site variance.
        y_n: The site's data.

    Returns:
        The expectation, a scalar.
    """
    # q(fₙ) = N(mₙ, vₙ), a 1-D GaussianState
    cov = lx.MatrixLinearOperator(var_n[None, None], lx.positive_semidefinite_tag)
    state = GaussianState(mean=mean_n[None], cov=cov)
    # 𝔼[fn(fₙ, yₙ)] = ∑ₚ wₚ fn(χₚ, yₙ)
    expectation = mean_expectation(
        lambda f: jnp.atleast_1d(fn(f[0], y_n)), state, integrator
    )
    return expectation[0]


def _per_site_expectation(
    integrator: GaussxIntegrator,
    fn: Callable[[Float[Array, ""], PyTree], Float[Array, ""]],
    mean: Float[Array, " N"],
    var: Float[Array, " N"],
    y: PyTree[Float[Array, " N ..."]],
) -> Float[Array, " N"]:
    """`_site_expectation` at every site.

    Args:
        integrator: The rule for each 1-D Gaussian.
        fn: The integrand, from one site's scalar $f_n$ and its $y_n$.
        mean: The site means.
        var: The site variances.
        y: The per-site data, a pytree with a leading site axis on each leaf.

    Returns:
        The expectation at each site.
    """
    return jax.vmap(lambda m, v, d: _site_expectation(integrator, fn, m, v, d))(
        mean, var, y
    )


def _ell_numerical(
    lik: Likelihood,
    y: Float[Array, " N"],
    f_loc: Float[Array, " N"],
    f_var: Float[Array, " N"],
    integrator: GaussxIntegrator,
    X: Float[Array, "N D"] | None = None,
) -> Float[Array, ""]:
    r"""Per-point numerical ELL for non-conjugate likelihoods.

    Integrates $\mathbb{E}_{q(f_n)}[\log p(y_n \mid f_n)]$
    for each data point via the gaussx integrator, then sums.

    ``X`` is consumed only by likelihoods whose parameters depend on the
    input (see `pyrox_gp.WarpedGaussianLikelihood` with a conditional
    warp); scalar observation models ignore it.

    A `Likelihood` written against the pre-gh-204 ``log_prob(f, y)``
    signature is never handed the third argument -- it could not consume
    it, and passing it would raise ``TypeError`` at the first quadrature
    evaluation.
    """
    if X is not None and likelihood_accepts_inputs(lik):
        expectations = _per_site_expectation(
            integrator,
            lambda f_n, site: lik.log_prob(f_n[None], site[0][None], site[1]),
            f_loc,
            f_var,
            (y, X),
        )
    else:
        expectations = _per_site_expectation(
            integrator,
            lambda f_n, y_n: lik.log_prob(f_n[None], y_n[None]),
            f_loc,
            f_var,
            y,
        )
    return expectations.sum()


def svgp_elbo(
    prior: SparseGPPrior,
    guide: Guide,
    likelihood: Likelihood,
    X: Float[Array, "N D"],
    y: Float[Array, " N"],
    *,
    integrator: GaussxIntegrator | None = None,
) -> Float[Array, ""]:
    r"""Structured SVGP ELBO as a differentiable scalar.

    $$
    \mathcal{L} = \sum_n \mathbb{E}_{q(f_n)}
                  [\log p(y_n \mid f_n)]
                - \mathrm{KL}[q(u) \| p(u)]
    $$

    For `GaussianLikelihood` the expected log-likelihood has a
    closed form and no integrator is needed:

    ```python
    loss = svgp_elbo(prior, guide, GaussianLikelihood(0.1), X, y)
    grad = eqx.filter_grad(lambda g: -svgp_elbo(prior, g, ...))(guide)
    ```

    For non-conjugate likelihoods supply a gaussx integrator:

    ```python
    from gaussx import GaussHermiteIntegrator
    loss = svgp_elbo(
        prior, guide,
        DistLikelihood(lambda f: dist.Bernoulli(logits=f)),
        X, y, integrator=GaussHermiteIntegrator(order=20),
    )
    ```

    Args:
        prior: Sparse GP prior (kernel + inducing inputs).
        guide: Variational guide over inducing values.
        likelihood: Observation model.
        X: Training inputs, shape ``(N, D)``.
        y: Training targets, shape ``(N,)``.
        integrator: gaussx integrator for the per-point ELL. Required
            for non-conjugate likelihoods; ignored for
            `GaussianLikelihood`.

    Returns:
        Scalar ELBO value (higher is better).

    Raises:
        ValueError: If a non-conjugate likelihood is used without an
            integrator.
    """
    with call_context(prior.kernel):
        K_zz_op, K_xz, K_xx_diag = prior.predictive_blocks(X)

    f_loc, f_var = guide.predict(K_xz, K_zz_op, K_xx_diag)  # ty: ignore[unresolved-attribute]
    f_loc = f_loc + prior.mean(X)

    kl = guide.kl_divergence(K_zz_op)  # ty: ignore[unresolved-attribute]

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
    ell = _ell_numerical(likelihood, y, f_loc, f_var, integrator, X)
    return ell - kl


def svgp_factor(
    name: str,
    prior: SparseGPPrior,
    guide: Guide,
    likelihood: Likelihood,
    X: Float[Array, "N D"],
    y: Float[Array, " N"],
    *,
    integrator: GaussxIntegrator | None = None,
) -> None:
    """Register the structured SVGP ELBO as a NumPyro factor site.

    Wraps `svgp_elbo` in ``numpyro.factor`` so it plugs into
    ``numpyro.infer.SVI`` + ``Trace_ELBO``. NumPyro sees one
    deterministic factor site — the *actual* ELBO uses the efficient
    closed-form KL + structured ELL computation.

    ```python
    def model(X, y):
        svgp_factor("elbo", prior, guide, lik, X, y)

    svi = SVI(model, lambda X, y: None, Adam(1e-3), Trace_ELBO())
    ```
    """
    numpyro.factor(
        name,
        svgp_elbo(prior, guide, likelihood, X, y, integrator=integrator),
    )


class ConjugateVI:
    r"""Natural-gradient / CVI update for sparse variational GPs.

    Operates in natural-parameter space: each `step` computes the
    per-point expected gradients and Hessians of the log-likelihood
    under $q(f_n)$, projects them into the inducing-value natural
    parameters, and applies a damped update via
    `NaturalGuide.natural_update`.

    For `GaussianLikelihood` the gradients and Hessians are
    analytical. For non-conjugate likelihoods an integrator is required
    to evaluate the expectations numerically.

    ```python
    guide = NaturalGuide.init(num_inducing=M)
    cvi = ConjugateVI(damping=0.5)

    for epoch in range(100):
        guide = cvi.step(prior, guide, GaussianLikelihood(0.1), X, y)
    ```

    Attributes:
        damping: Learning rate / damping factor $\rho \in (0, 1]$.
            ``1.0`` replaces the natural parameters with the target;
            ``< 1`` interpolates for stability.
        integrator: gaussx integrator for non-conjugate expected
            gradients. ``None`` is fine for `GaussianLikelihood`.
    """

    damping: float
    integrator: GaussxIntegrator | None

    def __init__(
        self,
        damping: float = 1.0,
        integrator: GaussxIntegrator | None = None,
    ) -> None:
        if not 0.0 <= damping <= 1.0:
            raise ValueError(f"damping must be in [0, 1], got {damping}.")
        self.damping = damping
        self.integrator = integrator

    def step(
        self,
        prior: SparseGPPrior,
        guide: NaturalGuide,
        likelihood: Likelihood,
        X: Float[Array, "N D"],
        y: Float[Array, " N"],
    ) -> NaturalGuide:
        r"""One CVI step: compute sites, project, update.

        1. Predict ``(f_loc, f_var) = guide.predict(...)``
        2. Compute per-point natural-gradient targets
           $(\lambda_n^{(1)}, \Lambda_n^{(2)})$.
        3. Project into inducing space:

        $$
        \begin{aligned}
        \hat{\eta}_1 &= \eta_1^{\text{prior}}
            + K_{ZZ}^{-1} K_{ZX}\, \lambda^{(1)}, \\
        \hat{\eta}_2 &= \eta_2^{\text{prior}}
            + K_{ZZ}^{-1} K_{ZX}\,
              \mathrm{diag}(\Lambda^{(2)})\, K_{XZ} K_{ZZ}^{-1}.
        \end{aligned}
        $$

        4. Damped update via `NaturalGuide.natural_update`.

        Args:
            prior: Sparse GP prior.
            guide: Current natural-parameter guide.
            likelihood: Observation model.
            X: Training inputs, shape ``(N, D)``.
            y: Training targets, shape ``(N,)``.

        Returns:
            Updated `NaturalGuide`.
        """
        with call_context(prior.kernel):
            K_zz_op, K_xz, K_xx_diag = prior.predictive_blocks(X)

        f_loc, f_var = guide.predict(K_xz, K_zz_op, K_xx_diag)
        f_loc = f_loc + prior.mean(X)

        grad1, grad2 = self._site_gradients(likelihood, y, f_loc, f_var)

        # B = K_zz^{-1} K_zx via gaussx.solve_matrix: one Cholesky
        # factorization shared across the matrix RHS for dense PSD
        # K_zz, structured dispatch (O(M) diagonal) otherwise.
        K_zx = einx.id("n m -> m n", K_xz)  # (N, M) → (M, N)
        B = solve_matrix(K_zz_op, K_zx)

        # Prior natural parameters: p(u) = N(0, K_zz)
        # => eta1_prior = 0, eta2_prior = -0.5 K_zz^{-1}
        # K_zz^{-1} must be dense here (it seeds the dense nat2), so
        # solve against I and re-symmetrize.
        M = K_zz_op.in_size()
        nat1_prior = jnp.zeros_like(guide.nat1)
        eye = jnp.eye(M, dtype=guide.nat2.dtype)
        nat2_prior = -0.5 * symmetrize(solve_matrix(K_zz_op, eye))

        # Project per-point site natural parameters into inducing
        # space. The site λ₁ = grad1 - f_loc * grad2 (not just grad1);
        # omitting the f_loc correction makes the update depend on the
        # current guide mean instead of being a fixed-point update.
        site_lambda1 = grad1 - f_loc * grad2
        # η̂₁ = η₁ᵖʳⁱᵒʳ + B λ⁽¹⁾,  with B: (M, N), λ⁽¹⁾: (N,) → (M,).
        nat1_hat = nat1_prior + einx.dot("m n, n -> m", B, site_lambda1)
        # η̂₂ = η₂ᵖʳⁱᵒʳ + B diag(½ grad2) Bᵀ. Fold the per-point scale into
        # B then contract the shared n axis: (M, N)·(K, N) → (M, K).
        scaled_B = einx.multiply("n, m n -> m n", 0.5 * grad2, B)
        nat2_hat = nat2_prior + einx.dot("m n, k n -> m k", scaled_B, B)

        return guide.natural_update(nat1_hat, nat2_hat, rho=self.damping)

    def _site_gradients(
        self,
        likelihood: Likelihood,
        y: Float[Array, " N"],
        f_loc: Float[Array, " N"],
        f_var: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        r"""Per-point ELL gradients w.r.t. the marginal mean.

        Returns ``(grad1, grad2)`` where ``grad1[n]`` is
        $\partial / \partial \mu_n \,
        \mathbb{E}_{q(f_n)}[\log p(y_n \mid f_n)]$ and ``grad2[n]``
        is the second derivative.

        For `GaussianLikelihood` these are analytical. For
        non-conjugate likelihoods they are computed via JAX autodiff
        through the per-point ELL.
        """
        if isinstance(likelihood, GaussianLikelihood):
            noise_var = likelihood.noise_var
            grad1 = (y - f_loc) / noise_var
            grad2 = jnp.full_like(f_loc, -1.0 / noise_var)
            return grad1, grad2

        if self.integrator is None:
            raise ValueError(
                "Non-conjugate likelihoods require an integrator for "
                "CVI. Pass integrator=GaussHermiteIntegrator(order=20)."
            )

        def _per_point_ell(
            mu_n: Float[Array, ""],
            var_n: Float[Array, ""],
            y_n: Float[Array, ""],
        ) -> Float[Array, ""]:
            return _site_expectation(
                self.integrator,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                lambda f_n, y: lik.log_prob(f_n[None], y[None]),
                mu_n,
                var_n,
                y_n,
            )

        lik = likelihood
        grad1 = jax.vmap(jax.grad(lambda mu, v, yn: _per_point_ell(mu, v, yn)))(
            f_loc, f_var, y
        )
        grad2 = jax.vmap(
            jax.grad(jax.grad(lambda mu, v, yn: _per_point_ell(mu, v, yn)))
        )(f_loc, f_var, y)
        return grad1, grad2
