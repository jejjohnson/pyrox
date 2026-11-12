r"""Observation-model and variational-guide protocols shared by the pyrox packages.

`Likelihood` is the observation model $p(y \mid f)$ and `Guide` the
variational posterior $q(f)$; every likelihood and guide in pyrox-gp (and in
any package built on the core) subclasses one of them. `GaussianLikelihood`
is the one concrete observation model here, because the closed-form paths of
several packages dispatch on it. `likelihood_accepts_inputs` tells whether a
likelihood takes the optional inputs ``X``. `AbstractObservation` is the
observation-model base that declares its own hyperparameters and builds a
gaussx likelihood holding the data; pyrox-lgm's observation models subclass
it. `DistLikelihood` is an observation model over any NumPyro distribution:
its `build` returns the per-site log density and data that
`gaussx.CallableLikelihood` takes, which the gaussx-backed packages wrap.

The module carries no linear algebra and imports no gaussx: the numerical
work on a likelihood (expectations, per-site derivatives) happens in the
packages that use it.
"""

from __future__ import annotations

import inspect
import warnings
from abc import abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as nd
from jaxtyping import Array, ArrayLike, Float, PyTree
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
    `gaussx.AbstractLikelihood`. The one exception is the core's own
    `DistLikelihood`, whose `build` returns the two arguments of
    `gaussx.CallableLikelihood` for the gaussx-backed packages to wrap.

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
            ``Any``, because the core imports no gaussx); for
            `DistLikelihood`, the ``(log_density, data)`` pair that
            `gaussx.CallableLikelihood` takes.
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


def _takes_latent_only(fn: Callable[..., nd.Distribution]) -> bool:
    """Whether ``fn`` is the old one-argument link ``f -> Distribution``.

    Raises:
        TypeError: If ``fn`` takes neither one nor three positional arguments.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # unintrospectable callable: trust it
        return False
    try:
        signature.bind(None, None, None)
    except TypeError:
        pass
    else:
        return False
    try:
        signature.bind(None)
    except TypeError:
        msg = (
            "DistLikelihood's make_dist takes (eta_i, theta, data_i) and returns "
            f"a numpyro Distribution; {fn!r} takes neither that nor one argument"
        )
        raise TypeError(msg) from None
    return True


class _LatentOnly(eqx.Module):
    """The old one-argument link ``f -> Distribution`` as a ``make_dist``."""

    dist_fn: Callable[[Array], nd.Distribution] = eqx.field(static=True)

    def __call__(
        self, eta: Array, theta: Mapping[str, Array], data: Mapping[str, Array]
    ) -> nd.Distribution:
        return self.dist_fn(eta)


class _DistSiteLogDensity(eqx.Module):
    r"""One site's $\ell(\eta_i, d_i) = \log p(y_i \mid \eta_i, \theta, d_i)$.

    An `equinox.Module` rather than a closure, so $\theta$ stays a leaf of
    the pytree that holds it (a `gaussx.CallableLikelihood` downstream).
    Site $i$'s data is ``{"y": y_i, "data": data_i}``.
    """

    make_dist: Callable[..., nd.Distribution] = eqx.field(static=True)
    theta: dict[str, Array]

    def __call__(self, eta_i: Float[Array, ""], site: PyTree) -> Float[Array, ""]:
        # log p(yᵢ ∣ ηᵢ, θ, dᵢ)
        log_prob = jnp.asarray(
            self.make_dist(eta_i, self.theta, site["data"]).log_prob(site["y"])
        )
        if log_prob.shape != ():
            msg = (
                "DistLikelihood's make_dist must return one site's distribution "
                "for one site's eta_i and data_i, so that log_prob(y_i) is a "
                f"scalar; got shape {log_prob.shape}. Pass per-site arrays "
                "through data rather than closing over them."
            )
            raise ValueError(msg)
        return log_prob


class DistLikelihood(AbstractObservation, Likelihood):
    r"""Observation model over any NumPyro distribution, $p(y_i \mid \eta_i, \theta)$.

    The user's ``make_dist(eta_i, theta, data_i)`` returns site $i$'s
    distribution, and its ``log_prob`` is the per-site log density

    $$
    \ell(\eta_i, d_i) = \log p(y_i \mid \eta_i, \theta, \mathrm{data}_i),
    \qquad
    \log p(y \mid \eta) = \sum_i \ell(\eta_i, d_i),
    $$

    so a likelihood gaussx has no class for (zero-inflated Poisson, Beta, a
    censored Gaussian) gets Laplace, EP and the Markov strategies.

    The core imports no gaussx, so `build` does not return a gaussx
    likelihood, unlike other observation models: it returns the pair
    ``(log_density, data)`` that `gaussx.CallableLikelihood` takes, where
    ``log_density(eta_i, d_i)`` is $\ell$ and ``data`` is
    ``{"y": y, "data": data}``, every leaf with a leading site axis of
    length $N$. pyrox-gp's inference and pyrox-ssm's strategies recognise a
    `DistLikelihood` and wrap the pair as
    ``gaussx.CallableLikelihood(*lik.build(y, theta, data))``, which gives
    the per-site derivatives and expectations. ``log_prob(f, y, X)`` and the
    per-site diagnostics (`site_distribution`) call ``make_dist`` on every
    site at once, so ``make_dist`` must broadcast over a leading site axis,
    as NumPyro distributions do; index inputs from the end
    (``data_i["X"][..., 0]``).

    The hyperparameters $\theta$ that ``make_dist`` reads are the ``theta``
    field, updated key by key by the ``theta`` passed to `build` (pyrox-gp
    and pyrox-ssm pass none). They are array leaves of the module, so they
    receive gradients and may be NumPyro sample values. pyrox-gp and
    pyrox-ssm pass the inputs as ``data_i = {"X": x_i}``. `pyrox_lgm.inla`
    does not take a `DistLikelihood` yet: it uses what `build` returns as a
    gaussx likelihood. The class registers no NumPyro sites.

    The old one-argument link ``DistLikelihood(lambda f: dist.Bernoulli(logits=f))``
    still works: a callable that takes one positional argument is wrapped to
    ignore ``theta`` and ``data_i``. ``DistLikelihood(dist_fn=...)`` warns
    and does the same.

    !!! warning "The callable cannot carry trainable parameters"
        ``make_dist`` is a **static** field, so anything it closes over is
        invisible to ``eqx.filter_grad`` and silently never trains. Put
        scalar hyperparameters in ``theta``; for a parameterized link (a
        warp, a network), write a `Likelihood` subclass holding it as a
        child module.

    Attributes:
        make_dist: ``(eta_i, theta, data_i) -> numpyro.distributions.Distribution``,
            site $i$'s distribution of $y_i$.
        theta: Hyperparameter values ``make_dist`` reads, as arrays.
        name: Prefix of the hyperparameter names (``"dist"``).

    Examples:
        A zero-inflated Poisson with gate $\pi$, and the gaussx likelihood a
        gaussx-backed package wraps it into:

        >>> import gaussx as gx
        >>> import jax.numpy as jnp
        >>> import numpyro.distributions as dist
        >>> from pyrox.likelihoods import DistLikelihood
        >>> zip_lik = DistLikelihood(
        ...     lambda eta, th, d: dist.ZeroInflatedPoisson(
        ...         gate=th["pi"], rate=jnp.exp(eta)
        ...     ),
        ...     theta={"pi": 0.2},
        ... )
        >>> y = jnp.array([0.0, 2.0, 1.0])
        >>> site_lik = gx.CallableLikelihood(*zip_lik.build(y, {}, {}))
        >>> g, h = site_lik.site_derivatives(jnp.zeros(3))
        >>> g.shape, h.shape
        ((3,), (3,))

        The old one-argument link:

        >>> lik = DistLikelihood(lambda f: dist.Bernoulli(logits=f))
        >>> float(lik.log_prob(jnp.zeros(2), jnp.array([1.0, 0.0])))
        -1.386...
    """

    # (eta_i, theta, data_i) -> Distribution
    make_dist: Callable[..., nd.Distribution] = eqx.field(static=True)
    theta: dict[str, Array]
    name: str = eqx.field(static=True)

    def __init__(
        self,
        make_dist: Callable[..., nd.Distribution] | None = None,
        theta: Mapping[str, ArrayLike] | None = None,
        *,
        name: str = "dist",
        dist_fn: Callable[[Array], nd.Distribution] | None = None,
    ) -> None:
        if dist_fn is not None:
            if make_dist is not None:
                msg = "pass make_dist or the deprecated dist_fn, not both"
                raise TypeError(msg)
            warnings.warn(
                "DistLikelihood(dist_fn=...) is deprecated; pass the callable "
                "as make_dist: DistLikelihood(make_dist=...) or positionally.",
                DeprecationWarning,
                stacklevel=2,
                # equinox's metaclass calls __init__; point at the caller
                skip_file_prefixes=(str(Path(eqx.__file__).parent),),
            )
            make_dist = dist_fn
        if make_dist is None:
            msg = "DistLikelihood needs make_dist(eta_i, theta, data_i)"
            raise TypeError(msg)
        if _takes_latent_only(make_dist):
            make_dist = _LatentOnly(make_dist)
        self.make_dist = make_dist
        self.theta = {key: jnp.asarray(value) for key, value in (theta or {}).items()}
        self.name = name

    def _theta(self, theta: Mapping[str, Array]) -> dict[str, Array]:
        """The ``theta`` field, updated by the caller's ``theta``."""
        return {**self.theta, **theta}

    def build(
        self, y: Array, theta: dict[str, Array], data: Mapping[str, ArrayLike]
    ) -> tuple[Callable[[Float[Array, ""], PyTree], Float[Array, ""]], PyTree]:
        r"""The per-site log density and data that `gaussx.CallableLikelihood` takes.

        Args:
            y: Observations, ``(N,)``.
            theta: Hyperparameters, overriding the ``theta`` field key by key.
            data: Per-site data other than ``y``, every leaf with a leading
                site axis of length ``N``.

        Returns:
            ``(log_density, site_data)``: ``log_density(eta_i, d_i)`` is
            $\log p(y_i \mid \eta_i, \theta, \mathrm{data}_i)$ for a scalar
            ``eta_i`` and one site's slice ``d_i`` of
            ``site_data = {"y": y, "data": data}``.
        """
        log_density = _DistSiteLogDensity(self.make_dist, self._theta(theta))
        site_data = {
            "y": jnp.asarray(y),
            "data": {key: jnp.asarray(value) for key, value in data.items()},
        }
        return log_density, site_data

    def site_distribution(
        self,
        eta: Float[Array, " N"],
        theta: dict[str, Array],
        data: Mapping[str, ArrayLike],
    ) -> nd.Distribution:
        """``make_dist`` on every site at once (batched in eta)."""
        return self.make_dist(eta, self._theta(theta), data)

    def log_prob(
        self,
        f: Float[Array, " ..."],
        y: Float[Array, " ..."],
        X: Float[Array, " ..."] | None = None,
    ) -> Float[Array, ""]:
        r"""$\sum_i \log p(y_i \mid f_i)$, with the inputs in ``data["X"]``."""
        data = {} if X is None else {"X": X}
        return self.site_log_prob(y, f, {}, data).sum()


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
    "DistLikelihood",
    "GaussianLikelihood",
    "Guide",
    "Likelihood",
    "likelihood_accepts_inputs",
]
