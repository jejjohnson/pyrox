r"""Observation-model and variational-guide protocols shared by the pyrox packages.

`Likelihood` is the observation model $p(y \mid f)$ and `Guide` the
variational posterior $q(f)$; every likelihood and guide in pyrox-gp (and in
any package built on the core) subclasses one of them. `GaussianLikelihood`
is the one concrete observation model here, because the closed-form paths of
several packages dispatch on it. `likelihood_accepts_inputs` tells whether a
likelihood takes the optional inputs ``X``. `AbstractObservation` is the
observation-model base that declares its own hyperparameters and builds a
gaussx likelihood holding the data; pyrox-lgm's observation models subclass
it.

The module carries no linear algebra and imports no gaussx: the numerical
work on a likelihood (expectations, per-site derivatives) happens in the
packages that use it.
"""

from __future__ import annotations

import inspect
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as nd
from jaxtyping import Array, ArrayLike, Float
from numpyro.distributions.transforms import Transform


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


class AbstractObservation(eqx.Module):
    r"""Observation model $p(y \mid \eta, \theta)$ with its own hyperparameters.

    An observation model declares its hyperparameters $\theta$ (`theta_spec`)
    and builds, for data ``y`` at a given $\theta$, the gaussx likelihood
    that inference runs on (`build`). Per-site data other than ``y``
    (offsets, trial counts, inputs) passes through ``data``. The per-site
    methods (`site_distribution`, `site_log_prob`, `site_cdf`) evaluate
    $p(y_i \mid \eta_i, \theta)$ directly, for diagnostics. pyrox-lgm's
    observation models (`pyrox_lgm.Gaussian`, `pyrox_lgm.Poisson`, …)
    subclass it, and `pyrox_lgm.AbstractObservation` is this class.

    The linear predictor ``eta`` has shape ``(N,)`` when `latent_dim` is 1
    (the default) and ``(N, latent_dim)`` otherwise: one row of latents per
    site, as for a softmax over $K$ classes (``latent_dim = K``) or a
    heteroscedastic Gaussian (``latent_dim = 2``). A subclass changes it by
    overriding the property, or with a static field that has a default
    (``latent_dim: int = eqx.field(static=True, default=2)``).

    The class registers no NumPyro sites, and the core imports no gaussx:
    `build` is annotated ``Any``, and a subclass returns a
    `gaussx.AbstractLikelihood`.

    Attributes:
        name: Prefix of the hyperparameter names, which join $\theta$ as
            ``f"{name}.{key}"`` (pyrox-lgm uses ``"lik"``).

    Examples:
        A heteroscedastic Gaussian with two latents per site, the mean and
        the noise scale before a softplus (gaussx's likelihood takes the two
        latent columns concatenated, ``eta.T.reshape(-1)``):

        >>> import equinox as eqx
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import numpyro.distributions as dist
        >>> from pyrox.likelihoods import AbstractObservation
        >>> class Heteroscedastic(AbstractObservation):
        ...     name: str = eqx.field(static=True, default="lik")
        ...     @property
        ...     def latent_dim(self):
        ...         return 2
        ...     def build(self, y, theta, data):
        ...         import gaussx as gx
        ...         return gx.HeteroscedasticGaussianLikelihood(y)
        ...     def site_distribution(self, eta, theta, data):
        ...         return dist.Normal(eta[:, 0], jax.nn.softplus(eta[:, 1]))
        >>> obs = Heteroscedastic()
        >>> eta = jnp.zeros((3, obs.latent_dim))
        >>> obs.site_log_prob(jnp.zeros(3), eta, {}, {}).shape
        (3,)
    """

    name: eqx.AbstractVar[str]

    @property
    def latent_dim(self) -> int:
        """Latents per site: the trailing size of ``eta``; 1 by default."""
        return 1

    def theta_spec(self) -> dict[str, tuple[nd.Distribution, Transform]]:
        """Hyperparameters of the likelihood; none by default.

        Returns:
            ``{key: (prior, transform)}``, where ``transform`` maps the
            unconstrained value to the prior's support.
        """
        return {}

    @abstractmethod
    def build(
        self, y: Array, theta: dict[str, Array], data: Mapping[str, ArrayLike]
    ) -> Any:
        """The gaussx likelihood holding ``y`` at hyperparameters ``theta``.

        Args:
            y: Observations, ``(N,)``.
            theta: Hyperparameters, keyed as in `theta_spec`.
            data: Per-site data other than ``y`` (offsets, trial counts,
                inputs).

        Returns:
            A `gaussx.AbstractLikelihood` over the ``N`` sites (annotated
            ``Any``, because the core imports no gaussx).
        """

    def site_distribution(
        self,
        eta: Float[Array, " N"] | Float[Array, "N L"],
        theta: dict[str, Array],
        data: Mapping[str, ArrayLike],
    ) -> nd.Distribution:
        """``p(y_i | eta_i)`` per site as a NumPyro distribution (batched in eta).

        ``eta`` is ``(N,)`` for one latent per site and ``(N, latent_dim)``
        otherwise. Diagnostics may evaluate it on quadrature nodes, so for
        one latent ``eta`` may carry trailing axes after the sites'
        (broadcast per-site data against them). Only diagnostics need it
        (`pyrox_lgm.diagnostics`), so an observation model that does not
        define it still fits.

        Raises:
            NotImplementedError: Unless a subclass defines it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} defines no site_distribution, which the "
            "diagnostics need"
        )

    def site_log_prob(
        self,
        y: Array,
        eta: Float[Array, " N"] | Float[Array, "N L"],
        theta: dict[str, Array],
        data: Mapping[str, ArrayLike],
    ) -> Array:
        """``log p(y_i | eta_i)`` per site, from `site_distribution`."""
        return jnp.asarray(self.site_distribution(eta, theta, data).log_prob(y))

    def site_cdf(
        self,
        y: Array,
        eta: Float[Array, " N"] | Float[Array, "N L"],
        theta: dict[str, Array],
        data: Mapping[str, ArrayLike],
    ) -> Array:
        """``P(Y_i <= y_i | eta_i)`` per site, from `site_distribution`."""
        return jnp.asarray(self.site_distribution(eta, theta, data).cdf(y))


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
    "AbstractObservation",
    "GaussianLikelihood",
    "Guide",
    "Likelihood",
    "likelihood_accepts_inputs",
]
