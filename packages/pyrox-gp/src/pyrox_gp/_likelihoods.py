"""Observation models for GP inference.

Three kinds of likelihood ship in this module:

* **Concrete analytic** — `GaussianLikelihood` carries closed-form
  expected log-likelihood, so `svgp_elbo` can skip numerical
  integration entirely and use
  `gaussx.variational_elbo_gaussian`.
* **Generic wrapper** — `DistLikelihood` turns *any*
  ``numpyro.distributions.Distribution`` into a `Likelihood` via
  a user-supplied link function ``f -> dist``. Non-conjugate ELBO paths
  integrate the wrapped ``log_prob`` numerically through a gaussx
  integrator.
* **Concrete non-Gaussian** — `BernoulliLikelihood`,
  `PoissonLikelihood`, `StudentTLikelihood` are scalar
  observation models for the advanced inference strategies (Laplace, GN,
  EP, posterior linearization). `SoftmaxLikelihood` and
  `HeteroscedasticGaussianLikelihood` are multi-latent
  (``latent_dim > 1``); their ``log_prob`` works but the scalar-latent
  advanced inference paths reject them with a clear error.

All likelihoods satisfy the `pyrox_gp.Likelihood` protocol:
``log_prob(f, y) -> scalar``. Multi-latent observation models declare
their per-observation latent count via the ``latent_dim`` static field
(default ``1`` for scalar likelihoods).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as nd
from jaxtyping import Array, Float, Int

from pyrox_gp._protocols import Likelihood


class GaussianLikelihood(Likelihood):
    r"""Gaussian observation model $p(y \mid f) = N(y \mid f, \sigma^2)$.

    The only likelihood with a closed-form expected log-likelihood,
    enabling the analytical Titsias ELBO via
    `gaussx.variational_elbo_gaussian`.

    Attributes:
        noise_var: Observation noise variance $\sigma^2$.
    """

    noise_var: float | Float[Array, ""]

    def log_prob(
        self,
        f: Float[Array, " ..."],
        y: Float[Array, " ..."],
    ) -> Float[Array, ""]:
        r"""Sum of per-point Gaussian log-densities."""
        return nd.Normal(f, jnp.sqrt(self.noise_var)).log_prob(y).sum()


class DistLikelihood(Likelihood):
    r"""Generic likelihood wrapping any ``numpyro.distributions.Distribution``.

    The user supplies a *link function* that maps the latent function
    value ``f`` to a numpyro distribution over observations:

    ```python
    # Bernoulli with logit link
    lik = DistLikelihood(lambda f: dist.Bernoulli(logits=f))

    # Poisson with log link
    lik = DistLikelihood(lambda f: dist.Poisson(rate=jnp.exp(f)))

    # Student-t noise
    lik = DistLikelihood(lambda f: dist.StudentT(df=3, loc=f, scale=0.5))
    ```

    !!! warning "The link function cannot carry trainable parameters"
        ``dist_fn`` is a **static** field, so anything the callable closes
        over is invisible to ``eqx.filter_grad``. A link function holding
        learnable parameters will silently never train — there is no error,
        and the loss still decreases because the kernel and guide keep
        fitting.

        ```python
        # WRONG -- `warp` is frozen at its initial value forever
        lik = DistLikelihood(lambda f: dist.Normal(warp(f), sigma))
        ```

        For a parameterized link, write a `Likelihood` subclass and
        hold the parameters as child modules, so they are ordinary
        trainable leaves.

    The resulting object satisfies the `Likelihood` protocol and
    can be passed to `svgp_elbo`. Because no closed-form expected
    log-likelihood is available, the ELBO uses numerical integration
    (``GaussHermiteIntegrator`` or ``MonteCarloIntegrator`` from gaussx).

    Attributes:
        dist_fn: Callable mapping ``f`` to a
            `numpyro.distributions.Distribution`.
    """

    dist_fn: Callable[..., Any] = eqx.field(static=True)

    def log_prob(
        self,
        f: Float[Array, " ..."],
        y: Float[Array, " ..."],
    ) -> Float[Array, ""]:
        r"""Sum of per-point log-densities under the wrapped distribution."""
        return self.dist_fn(f).log_prob(y).sum()


class BernoulliLikelihood(Likelihood):
    r"""Binary classification likelihood with logit link.

    ``p(y \mid f) = \mathrm{Bernoulli}(\sigma(f))`` where
    $\sigma$ is the logistic function. Targets ``y`` are
    ``{0, 1}`` valued. Scalar latent (``latent_dim = 1``).
    """

    def log_prob(
        self,
        f: Float[Array, " ..."],
        y: Float[Array, " ..."],
    ) -> Float[Array, ""]:
        return nd.Bernoulli(logits=f).log_prob(y).sum()


class PoissonLikelihood(Likelihood):
    r"""Count likelihood with log-link.

    ``p(y \mid f) = \mathrm{Poisson}(\exp(f))``. Targets ``y`` are
    non-negative integers. Scalar latent (``latent_dim = 1``).
    """

    def log_prob(
        self,
        f: Float[Array, " ..."],
        y: Float[Array, " ..."],
    ) -> Float[Array, ""]:
        return nd.Poisson(rate=jnp.exp(f)).log_prob(y).sum()


class StudentTLikelihood(Likelihood):
    r"""Heavy-tailed regression: ``p(y | f) = StudentT(nu, f, sigma)``.

    Robust to outliers — the heavier-than-Gaussian tails downweight
    observations that are far from the latent. Scalar latent
    (``latent_dim = 1``). Both ``df`` and ``scale`` are positive.

    Attributes:
        df: Degrees of freedom $\nu > 0$. Smaller values give
            heavier tails. ``df -> infinity`` recovers the Gaussian.
        scale: Scale parameter $\sigma > 0$.
    """

    df: float | Float[Array, ""]
    scale: float | Float[Array, ""]

    def log_prob(
        self,
        f: Float[Array, " ..."],
        y: Float[Array, " ..."],
    ) -> Float[Array, ""]:
        return nd.StudentT(df=self.df, loc=f, scale=self.scale).log_prob(y).sum()


class SoftmaxLikelihood(Likelihood):
    r"""Multi-class classification likelihood with softmax link.

    Each observation has one latent function value per class:
    ``f`` has shape ``(N, num_classes)`` and
    ``p(y_n \mid f_n) = \mathrm{Categorical}(\mathrm{softmax}(f_n))``.
    Targets ``y`` are integer class indices in ``[0, num_classes)``.

    Multi-latent (``latent_dim = num_classes``). The scalar-latent
    advanced inference strategies in `pyrox_gp._inference_nongauss`
    reject this likelihood with a clear error; use SVGP / MAP for now,
    or wait for the multi-latent inference follow-up.

    Attributes:
        num_classes: Number of output classes $C \geq 2$.
    """

    num_classes: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    def __init__(self, num_classes: int) -> None:
        if num_classes < 2:
            msg = f"num_classes must be >= 2, got {num_classes}"
            raise ValueError(msg)
        self.num_classes = num_classes
        self.latent_dim = num_classes

    def log_prob(
        self,
        f: Float[Array, "N C"],
        y: Int[Array, " N"],
    ) -> Float[Array, ""]:
        return nd.Categorical(logits=f).log_prob(y).sum()


class HeteroscedasticGaussianLikelihood(Likelihood):
    r"""Gaussian regression with input-dependent noise.

    Each observation consumes two latents — the mean and the
    log-noise-standard-deviation:
    $p(y_n | f_n^{(0)}, f_n^{(1)}) = N(y_n | f_n^{(0)}, e^{2 f_n^{(1)}})$.

    Multi-latent (``latent_dim = 2``). The scalar-latent advanced
    inference strategies reject this likelihood; use SVGP for now.
    """

    latent_dim: int = eqx.field(static=True, default=2)

    def log_prob(
        self,
        f: Float[Array, "N 2"],
        y: Float[Array, " N"],
    ) -> Float[Array, ""]:
        loc = f[..., 0]
        log_scale = f[..., 1]
        return nd.Normal(loc=loc, scale=jnp.exp(log_scale)).log_prob(y).sum()
