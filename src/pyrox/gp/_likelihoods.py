"""Observation models for sparse variational GP inference.

Two kinds of likelihood ship in this module:

* **Concrete analytic** — :class:`GaussianLikelihood` carries closed-form
  expected log-likelihood, so :func:`svgp_elbo` can skip numerical
  integration entirely and use
  :func:`gaussx.variational_elbo_gaussian`.
* **Generic wrapper** — :class:`DistLikelihood` turns *any*
  ``numpyro.distributions.Distribution`` into a :class:`Likelihood` via
  a user-supplied link function ``f -> dist``. Non-conjugate ELBO paths
  integrate the wrapped ``log_prob`` numerically through a gaussx
  integrator.

All likelihoods satisfy the :class:`pyrox.gp.Likelihood` protocol:
``log_prob(f, y) -> scalar``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as nd
from jaxtyping import Array, Float

from pyrox.gp._protocols import Likelihood


class GaussianLikelihood(Likelihood):
    r"""Gaussian observation model :math:`p(y \mid f) = N(y \mid f, \sigma^2)`.

    The only likelihood with a closed-form expected log-likelihood,
    enabling the analytical Titsias ELBO via
    :func:`gaussx.variational_elbo_gaussian`.

    Attributes:
        noise_var: Observation noise variance :math:`\sigma^2`.
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

    .. code-block:: python

        # Bernoulli with logit link
        lik = DistLikelihood(lambda f: dist.Bernoulli(logits=f))

        # Poisson with log link
        lik = DistLikelihood(lambda f: dist.Poisson(rate=jnp.exp(f)))

        # Student-t noise
        lik = DistLikelihood(lambda f: dist.StudentT(df=3, loc=f, scale=0.5))

    The resulting object satisfies the :class:`Likelihood` protocol and
    can be passed to :func:`svgp_elbo`. Because no closed-form expected
    log-likelihood is available, the ELBO uses numerical integration
    (``GaussHermiteIntegrator`` or ``MonteCarloIntegrator`` from gaussx).

    Attributes:
        dist_fn: Callable mapping ``f`` to a
            :class:`numpyro.distributions.Distribution`.
    """

    dist_fn: Callable[..., Any] = eqx.field(static=True)

    def log_prob(
        self,
        f: Float[Array, " ..."],
        y: Float[Array, " ..."],
    ) -> Float[Array, ""]:
        r"""Sum of per-point log-densities under the wrapped distribution."""
        return self.dist_fn(f).log_prob(y).sum()
