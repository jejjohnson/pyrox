"""Sparse variational GP inference entry points.

Three paths, all sharing the same likelihood + guide + prior surface:

* :func:`svgp_elbo` — pure-function ELBO returning a differentiable
  scalar. Use with ``optax`` / ``equinox`` for optimization outside
  NumPyro.
* :func:`svgp_factor` — wraps :func:`svgp_elbo` in
  ``numpyro.factor`` so it plugs into ``numpyro.infer.SVI`` +
  ``Trace_ELBO``. NumPyro sees one factor site; the *actual* ELBO uses
  the efficient structured computation.
* :class:`ConjugateVI` — natural-gradient / CVI update loop. Operates
  in natural-parameter space via
  :meth:`NaturalGuide.natural_update`; not NumPyro-coupled.

All Gaussian linear algebra delegates to ``gaussx``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpyro
from gaussx import (
    AbstractIntegrator as GaussxIntegrator,
    GaussianState,
    cholesky,
    inv,
    log_likelihood_expectation,
    variational_elbo_gaussian,
)
from jaxtyping import Array, Float

from pyrox.gp._context import _kernel_context
from pyrox.gp._guides import NaturalGuide
from pyrox.gp._likelihoods import GaussianLikelihood
from pyrox.gp._protocols import Guide, Likelihood
from pyrox.gp._sparse import SparseGPPrior


def _ell_numerical(
    lik: Likelihood,
    y: Float[Array, " N"],
    f_loc: Float[Array, " N"],
    f_var: Float[Array, " N"],
    integrator: GaussxIntegrator,
) -> Float[Array, ""]:
    r"""Per-point numerical ELL for non-conjugate likelihoods.

    Integrates :math:`\mathbb{E}_{q(f_n)}[\log p(y_n \mid f_n)]`
    for each data point via the gaussx integrator, then sums.
    """

    def _ell_one(
        mu_n: Float[Array, ""],
        var_n: Float[Array, ""],
        y_n: Float[Array, ""],
    ) -> Float[Array, ""]:
        state = GaussianState(
            mean=mu_n[None],
            cov=lx.MatrixLinearOperator(
                var_n[None, None], lx.positive_semidefinite_tag
            ),
        )
        return log_likelihood_expectation(
            lambda f: lik.log_prob(f, y_n[None]),
            state,  # ty: ignore[invalid-argument-type]
            integrator,
        )

    return jax.vmap(_ell_one)(f_loc, f_var, y).sum()


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

    .. math::

        \mathcal{L} = \sum_n \mathbb{E}_{q(f_n)}
                      [\log p(y_n \mid f_n)]
                    - \mathrm{KL}[q(u) \| p(u)]

    For :class:`GaussianLikelihood` the expected log-likelihood has a
    closed form and no integrator is needed:

    .. code-block:: python

        loss = svgp_elbo(prior, guide, GaussianLikelihood(0.1), X, y)
        grad = eqx.filter_grad(lambda g: -svgp_elbo(prior, g, ...))(guide)

    For non-conjugate likelihoods supply a gaussx integrator:

    .. code-block:: python

        from gaussx import GaussHermiteIntegrator
        loss = svgp_elbo(
            prior, guide,
            DistLikelihood(lambda f: dist.Bernoulli(logits=f)),
            X, y, integrator=GaussHermiteIntegrator(order=20),
        )

    Args:
        prior: Sparse GP prior (kernel + inducing inputs).
        guide: Variational guide over inducing values.
        likelihood: Observation model.
        X: Training inputs, shape ``(N, D)``.
        y: Training targets, shape ``(N,)``.
        integrator: gaussx integrator for the per-point ELL. Required
            for non-conjugate likelihoods; ignored for
            :class:`GaussianLikelihood`.

    Returns:
        Scalar ELBO value (higher is better).

    Raises:
        ValueError: If a non-conjugate likelihood is used without an
            integrator.
    """
    with _kernel_context(prior.kernel):
        K_zz_op, K_xz, K_xx_diag = prior.predictive_blocks(X)

    f_loc, f_var = guide.predict(K_xz, K_zz_op, K_xx_diag)  # ty: ignore[unresolved-attribute]
    f_loc = f_loc + prior.mean(X)

    kl = guide.kl_divergence(K_zz_op)  # ty: ignore[unresolved-attribute]

    if isinstance(likelihood, GaussianLikelihood):
        return variational_elbo_gaussian(
            y,
            f_loc,
            f_var,
            likelihood.noise_var,
            kl,  # ty: ignore[invalid-argument-type]
        )

    if integrator is None:
        raise ValueError(
            "Non-conjugate likelihoods require an integrator "
            "(e.g. gaussx.GaussHermiteIntegrator). "
            "Pass integrator=GaussHermiteIntegrator(order=20)."
        )
    ell = _ell_numerical(likelihood, y, f_loc, f_var, integrator)
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

    Wraps :func:`svgp_elbo` in ``numpyro.factor`` so it plugs into
    ``numpyro.infer.SVI`` + ``Trace_ELBO``. NumPyro sees one
    deterministic factor site — the *actual* ELBO uses the efficient
    closed-form KL + structured ELL computation.

    .. code-block:: python

        def model(X, y):
            svgp_factor("elbo", prior, guide, lik, X, y)

        svi = SVI(model, lambda X, y: None, Adam(1e-3), Trace_ELBO())
    """
    numpyro.factor(
        name,
        svgp_elbo(prior, guide, likelihood, X, y, integrator=integrator),
    )


class ConjugateVI:
    r"""Natural-gradient / CVI update for sparse variational GPs.

    Operates in natural-parameter space: each :meth:`step` computes the
    per-point expected gradients and Hessians of the log-likelihood
    under :math:`q(f_n)`, projects them into the inducing-value natural
    parameters, and applies a damped update via
    :meth:`NaturalGuide.natural_update`.

    For :class:`GaussianLikelihood` the gradients and Hessians are
    analytical. For non-conjugate likelihoods an integrator is required
    to evaluate the expectations numerically.

    .. code-block:: python

        guide = NaturalGuide.init(num_inducing=M)
        cvi = ConjugateVI(damping=0.5)

        for epoch in range(100):
            guide = cvi.step(prior, guide, GaussianLikelihood(0.1), X, y)

    Attributes:
        damping: Learning rate / damping factor :math:`\rho \in (0, 1]`.
            ``1.0`` replaces the natural parameters with the target;
            ``< 1`` interpolates for stability.
        integrator: gaussx integrator for non-conjugate expected
            gradients. ``None`` is fine for :class:`GaussianLikelihood`.
    """

    damping: float
    integrator: GaussxIntegrator | None

    def __init__(
        self,
        damping: float = 1.0,
        integrator: GaussxIntegrator | None = None,
    ) -> None:
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
           :math:`(\lambda_n^{(1)}, \Lambda_n^{(2)})`.
        3. Project into inducing space:

        .. math::

            \hat{\eta}_1 &= \eta_1^{\text{prior}}
                + K_{ZZ}^{-1} K_{ZX}\, \lambda^{(1)}, \\
            \hat{\eta}_2 &= \eta_2^{\text{prior}}
                + K_{ZZ}^{-1} K_{ZX}\,
                  \mathrm{diag}(\Lambda^{(2)})\, K_{XZ} K_{ZZ}^{-1}.

        4. Damped update via :meth:`NaturalGuide.natural_update`.

        Args:
            prior: Sparse GP prior.
            guide: Current natural-parameter guide.
            likelihood: Observation model.
            X: Training inputs, shape ``(N, D)``.
            y: Training targets, shape ``(N,)``.

        Returns:
            Updated :class:`NaturalGuide`.
        """
        with _kernel_context(prior.kernel):
            K_zz_op, K_xz, K_xx_diag = prior.predictive_blocks(X)

        f_loc, f_var = guide.predict(K_xz, K_zz_op, K_xx_diag)
        f_loc = f_loc + prior.mean(X)

        grad1, grad2 = self._site_gradients(likelihood, y, f_loc, f_var)

        # B = K_zz^{-1} K_zx, shape (M, N)
        # Via two triangular solves: A = L^{-1} K_zx, B = L^{-T} A.
        L_zz = cholesky(K_zz_op)
        A = jax.vmap(
            lambda col: lx.linear_solve(L_zz, col).value,
            in_axes=1,
            out_axes=1,
        )(K_xz.T)
        B = jax.vmap(
            lambda col: lx.linear_solve(L_zz.T, col).value,
            in_axes=1,
            out_axes=1,
        )(A)

        # Prior natural parameters: p(u) = N(0, K_zz)
        # => eta1_prior = 0, eta2_prior = -0.5 K_zz^{-1}
        M = K_zz_op.in_size()
        nat1_prior = jnp.zeros(M)
        K_zz_inv = inv(K_zz_op)
        nat2_prior = -0.5 * K_zz_inv.as_matrix()

        # Project per-point sites into inducing space.
        # grad1 is ∂ELL/∂μ_n; the nat2 site contribution uses
        # ∂ELL/∂σ²_n = 0.5 * ∂²ELL/∂μ²_n (the variance gradient,
        # not the mean Hessian).
        nat1_hat = nat1_prior + B @ grad1
        nat2_hat = nat2_prior + B @ (0.5 * grad2[:, None] * B.T)

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
        :math:`\partial / \partial \mu_n \,
        \mathbb{E}_{q(f_n)}[\log p(y_n \mid f_n)]` and ``grad2[n]``
        is the second derivative.

        For :class:`GaussianLikelihood` these are analytical. For
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
            state = GaussianState(
                mean=mu_n[None],
                cov=lx.MatrixLinearOperator(
                    var_n[None, None], lx.positive_semidefinite_tag
                ),
            )
            return log_likelihood_expectation(
                lambda f: lik.log_prob(f, y_n[None]),
                state,  # ty: ignore[invalid-argument-type]
                self.integrator,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            )

        lik = likelihood
        grad1 = jax.vmap(jax.grad(lambda mu, v, yn: _per_point_ell(mu, v, yn)))(
            f_loc, f_var, y
        )
        grad2 = jax.vmap(
            jax.grad(jax.grad(lambda mu, v, yn: _per_point_ell(mu, v, yn)))
        )(f_loc, f_var, y)
        return grad1, grad2
