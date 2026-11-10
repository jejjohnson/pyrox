r"""Observation-model and variational-guide protocols shared by the pyrox packages.

`Likelihood` is the observation model $p(y \mid f)$ and `Guide` the
variational posterior $q(f)$; every likelihood and guide in pyrox-gp (and in
any package built on the core) subclasses one of them. `GaussianLikelihood`
is the one concrete observation model here, because the closed-form paths of
several packages dispatch on it. `likelihood_accepts_inputs` tells whether a
likelihood takes the optional inputs ``X``.

The module carries no linear algebra and imports no gaussx: the numerical
work on a likelihood (expectations, per-site derivatives) happens in the
packages that use it.
"""

from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as nd
from jaxtyping import Array, Float


class Guide(eqx.Module):
    """Abstract base for variational posterior families.

    Concrete guides (``DeltaGuide``, ``MeanFieldGuide``, ``LowRankGuide``,
    ``FullRankGuide``, etc.) live in pyrox-gp. The whitening principle keeps
    optimization geometry well-conditioned — sample from a unit-scale latent
    and unwhiten with the prior Cholesky.

    Two distinct entry points:

    * `sample` / `log_prob` — pure variational draws and
      densities. ``sample(self, key)`` returns a draw from ``q(f)``;
      ``log_prob(self, f)`` evaluates ``log q(f)``. Neither touches the
      NumPyro trace.
    * ``register(name, prior)`` (optional) — the NumPyro-integration hook
      invoked by `pyrox_gp.gp_sample` when a guide is supplied. Use
      it to register a sample / param site (or compose one out of guide
      state) under ``name`` and return the latent function value. Concrete
      guides that participate in `gp_sample` should implement this;
      the protocol leaves it unspecified so guides usable purely outside
      NumPyro stay valid.

    Examples:
        A point-mass guide at a fixed vector:

        >>> import jax.numpy as jnp
        >>> from pyrox.likelihoods import Guide
        >>> class PointGuide(Guide):
        ...     loc: jnp.ndarray
        ...     def sample(self, key):
        ...         return self.loc
        ...     def log_prob(self, f):
        ...         return jnp.asarray(0.0)
        >>> PointGuide(jnp.zeros(3)).sample(None).shape
        (3,)
    """

    @abstractmethod
    def sample(self, key: Any) -> Float[Array, " ..."]:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, f: Float[Array, " ..."]) -> Float[Array, ""]:
        raise NotImplementedError


class Likelihood(eqx.Module):
    """Abstract base for observation models.

    Implements the conditional ``p(y | f)``, summed over points. The advanced
    inference strategies in pyrox-gp integrate ``log p(y | f)`` against a
    Gaussian cavity via any `gaussx.AbstractIntegrator`. Concrete
    scalar-latent likelihoods (`GaussianLikelihood` here;
    `BernoulliLikelihood`, `PoissonLikelihood`, `StudentTLikelihood` in
    pyrox-gp) and multi-latent ones (`SoftmaxLikelihood`,
    `HeteroscedasticGaussianLikelihood`) subclass it.

    Multi-latent likelihoods declare ``latent_dim: int`` as a static
    field (e.g. ``latent_dim = num_classes`` for softmax). Scalar
    likelihoods may omit the field; consumers should read
    ``getattr(lik, "latent_dim", 1)``.

    Examples:
        A Laplace observation model with unit scale:

        >>> import jax.numpy as jnp
        >>> from pyrox.likelihoods import Likelihood
        >>> class LaplaceLikelihood(Likelihood):
        ...     def log_prob(self, f, y, X=None):
        ...         return jnp.sum(-jnp.abs(y - f) - jnp.log(2.0))
        >>> float(LaplaceLikelihood().log_prob(jnp.zeros(2), jnp.zeros(2)))
        -1.386...
    """

    @abstractmethod
    def log_prob(
        self,
        f: Float[Array, " ..."],
        y: Float[Array, " ..."],
        X: Float[Array, " ..."] | None = None,
    ) -> Float[Array, ""]:
        """Conditional ``p(y | f)``, optionally conditioned on inputs.

        ``X`` is consumed only by likelihoods whose parameters are
        functions of the input (see `pyrox_gp.WarpedGaussianLikelihood`
        with a conditional warp). Scalar observation models ignore it.
        """
        raise NotImplementedError


class GaussianLikelihood(Likelihood):
    r"""Gaussian observation model $p(y \mid f) = N(y \mid f, \sigma^2)$.

    The only likelihood with a closed-form expected log-likelihood,
    enabling the analytical Titsias ELBO via
    `gaussx.variational_elbo_gaussian`.

    Attributes:
        noise_var: Observation noise variance $\sigma^2$.

    Examples:
        >>> import jax.numpy as jnp
        >>> from pyrox.likelihoods import GaussianLikelihood
        >>> lik = GaussianLikelihood(noise_var=1.0)
        >>> float(lik.log_prob(jnp.zeros(1), jnp.zeros(1)))
        -0.918...
    """

    noise_var: float | Float[Array, ""]

    def log_prob(
        self,
        f: Float[Array, " ..."],
        y: Float[Array, " ..."],
        X: Float[Array, " ..."] | None = None,
    ) -> Float[Array, ""]:
        r"""Sum of per-point Gaussian log-densities."""
        return nd.Normal(f, jnp.sqrt(self.noise_var)).log_prob(y).sum()


def likelihood_accepts_inputs(lik: Likelihood) -> bool:
    """Whether ``lik.log_prob`` takes the optional trailing ``X``.

    The protocol widened in gh-204 to carry the inputs an
    input-dependent observation model conditions on. Third-party
    `Likelihood` subclasses written against the original
    ``log_prob(self, f, y)`` are still concrete and valid, and handing
    one a third positional argument raises ``TypeError`` at the first
    quadrature evaluation -- so probe rather than assume.

    Static Python on the unbound signature, evaluated once per trace, so
    it costs nothing under ``jit``.

    Args:
        lik: The likelihood to probe.

    Returns:
        ``True`` if ``lik.log_prob`` takes a third positional argument.

    Examples:
        >>> from pyrox.likelihoods import GaussianLikelihood, likelihood_accepts_inputs
        >>> likelihood_accepts_inputs(GaussianLikelihood(noise_var=1.0))
        True
    """
    try:
        params = list(inspect.signature(lik.log_prob).parameters.values())
    except (TypeError, ValueError):  # unintrospectable callable
        return False
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return True
    positional = [
        p
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    return len(positional) >= 3


__all__ = [
    "GaussianLikelihood",
    "Guide",
    "Likelihood",
    "likelihood_accepts_inputs",
]
