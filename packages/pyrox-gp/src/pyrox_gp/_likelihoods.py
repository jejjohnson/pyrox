"""Observation models for GP inference.

Three kinds of likelihood ship in this module:

* **Observation models over gaussx** — `GaussianLikelihood`,
  `BernoulliLikelihood`, `PoissonLikelihood`, `StudentTLikelihood`,
  `SoftmaxLikelihood` and `HeteroscedasticGaussianLikelihood` are
  `pyrox.likelihoods.AbstractObservation` subclasses: ``build(y, theta,
  data)`` returns the gaussx likelihood of the same name holding ``y``, and
  the inference strategies take their log densities, derivatives and
  expectations from it. Inputs pass through ``data`` (``{"X": X}``). Their
  hyperparameters (``noise_var``, ``df``, ``scale``) are fields, so they
  declare no `theta_spec` and `build` ignores ``theta``.
  `GaussianLikelihood` subclasses `pyrox.likelihoods.GaussianLikelihood`, so
  the closed-form ELBO paths that dispatch on the core class keep it.
* **Generic wrapper** — `DistLikelihood` is `pyrox.likelihoods.DistLikelihood`,
  the core observation model over any ``numpyro.distributions.Distribution``
  (``make_dist(eta_i, theta, data_i)``, or the old one-argument ``f -> dist``).
  Its `build` returns the arguments of `gaussx.CallableLikelihood`, since the
  core imports no gaussx; `_gaussx_likelihood` wraps them, so the inference
  strategies take its derivatives and expectations from gaussx too.
* `SoftmaxLikelihood` and `HeteroscedasticGaussianLikelihood` are
  multi-latent (``latent_dim > 1``); their ``log_prob`` works but the
  scalar-latent advanced inference paths reject them with a clear error.

All of them satisfy the `pyrox_gp.Likelihood` protocol,
``log_prob(f, y, X=None) -> scalar``, summed over points; for the
observation models over gaussx it is ``build(y, {}, {"X": X}).log_prob(f)``.
"""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import gaussx as gx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Int
from numpyro.distributions.transforms import SoftplusTransform
from pyrox.likelihoods import (
    AbstractObservation,
    DistLikelihood,
    GaussianLikelihood as _CoreGaussianLikelihood,
    _observation_site_log_prob,
)

from pyrox_gp._protocols import Likelihood


class _GPObservation(AbstractObservation, Likelihood):
    r"""Base of pyrox-gp's observation models: `build` and the summed ``log_prob``.

    A subclass defines `build` (and, if it has more than one latent per site,
    `_gaussx_latents`); ``log_prob(f, y, X)`` then evaluates the gaussx
    likelihood that `build` returns, with ``X`` in ``data["X"]``. The latents
    ``f`` and the observations ``y`` broadcast against each other over the
    sites (``f`` carries a trailing ``latent_dim`` axis when it is above 1),
    and the result is summed over every broadcast site, as numpyro's
    ``log_prob(...).sum()`` was. ``X`` is passed as given.
    """

    def _gaussx_latents(
        self, f: Float[Array, " N"] | Float[Array, "N L"]
    ) -> Float[Array, " NL"]:
        """Site latents ``(N,)`` or ``(N, latent_dim)`` in `build`'s layout."""
        return f

    def log_prob(
        self,
        f: Float[Array, " ..."],
        y: Float[Array, " ..."],
        X: Float[Array, " ..."] | None = None,
    ) -> Float[Array, ""]:
        r"""$\log p(y \mid f) = \sum_i \log p(y_i \mid f_i)$, through `build`."""
        latent_dim = self.latent_dim
        f = jnp.asarray(f)
        site_shape = f.shape if latent_dim == 1 else f.shape[:-1]
        shape = jnp.broadcast_shapes(site_shape, jnp.shape(y))
        y_sites = jnp.reshape(jnp.broadcast_to(y, shape), (-1,))
        if latent_dim == 1:
            f_sites = jnp.reshape(jnp.broadcast_to(f, shape), (-1,))
        else:
            f_sites = jnp.reshape(
                jnp.broadcast_to(f, (*shape, latent_dim)), (-1, latent_dim)
            )
        data = {} if X is None else {"X": X}
        return self.build(y_sites, {}, data).log_prob(self._gaussx_latents(f_sites))


def _gaussx_likelihood(
    observation: AbstractObservation,
    y: Array,
    theta: dict[str, Array],
    data: Mapping[str, ArrayLike],
) -> gx.AbstractLikelihood:
    """The gaussx likelihood of ``observation`` holding ``y``.

    `build` returns it, except for the core `DistLikelihood`, whose `build`
    returns the ``(log_density, data)`` pair of `gaussx.CallableLikelihood`
    (the core imports no gaussx), wrapped here.
    """
    built = observation.build(y, theta, data)
    if isinstance(observation, DistLikelihood):
        return gx.CallableLikelihood(*built)
    return built


def _site_log_prob(
    observation: AbstractObservation,
    f_i: Float[Array, ""],
    y_i: Array,
    data_i: Mapping[str, Array],
) -> Float[Array, ""]:
    r"""$\log p(y_i \mid f_i)$ at one site, from the gaussx likelihood `build` gives.

    `pyrox.likelihoods._observation_site_log_prob`, the adapter pyrox-ssm
    shares, with `_gaussx_likelihood` as its ``build`` so a `DistLikelihood`
    is wrapped into `gaussx.CallableLikelihood`.
    """
    return _observation_site_log_prob(
        observation,
        f_i,
        y_i,
        data_i,
        build=lambda y, theta, data: _gaussx_likelihood(observation, y, theta, data),
    )


class GaussianLikelihood(_GPObservation, _CoreGaussianLikelihood):
    r"""Gaussian observation model $p(y \mid f) = N(y \mid f, \sigma^2)$, over
    `gaussx.GaussianLikelihood`.

    The only likelihood with a closed-form expected log-likelihood, so
    `svgp_elbo` and `ConjugateVI` take the analytical Titsias path for it. It
    subclasses `pyrox.likelihoods.GaussianLikelihood`, so every closed-form
    path that dispatches on the core class takes it too.

    Attributes:
        noise_var: Observation noise variance $\sigma^2$.

    Examples:
        >>> import jax.numpy as jnp
        >>> from pyrox_gp import GaussianLikelihood
        >>> lik = GaussianLikelihood(noise_var=1.0)
        >>> type(lik.build(jnp.zeros(2), {}, {})).__name__
        'GaussianLikelihood'
        >>> float(lik.log_prob(jnp.zeros(1), jnp.zeros(1)))
        -0.918...
    """

    name: str = eqx.field(static=True, default="gaussian", repr=False, kw_only=True)

    def build(
        self, y: Array, theta: dict[str, Array], data: Mapping[str, ArrayLike]
    ) -> gx.GaussianLikelihood:
        """`gaussx.GaussianLikelihood` holding ``y`` at ``noise_var``."""
        return gx.GaussianLikelihood(y, self.noise_var)  # ty: ignore[invalid-argument-type]


class BernoulliLikelihood(_GPObservation):
    r"""Binary classification likelihood with logit link, over
    `gaussx.BernoulliLikelihood`.

    $p(y \mid f) = \mathrm{Bernoulli}(\sigma(f))$ where $\sigma$ is the
    logistic function. Targets ``y`` are ``{0, 1}`` valued. Scalar latent
    (``latent_dim = 1``).

    Examples:
        >>> import jax.numpy as jnp
        >>> from pyrox_gp import BernoulliLikelihood
        >>> lik = BernoulliLikelihood()
        >>> site = lik.build(jnp.array([1.0, 0.0]), {}, {})
        >>> bool(jnp.allclose(site.log_prob(jnp.zeros(2)), 2.0 * jnp.log(0.5)))
        True
    """

    name: str = eqx.field(static=True, default="bernoulli", repr=False, kw_only=True)

    def build(
        self, y: Array, theta: dict[str, Array], data: Mapping[str, ArrayLike]
    ) -> gx.BernoulliLikelihood:
        """`gaussx.BernoulliLikelihood` holding ``y``."""
        return gx.BernoulliLikelihood(y)


class PoissonLikelihood(_GPObservation):
    r"""Count likelihood with log link, over `gaussx.PoissonLikelihood`.

    $p(y \mid f) = \mathrm{Poisson}(\exp(f))$. Targets ``y`` are non-negative
    integers. Scalar latent (``latent_dim = 1``).

    Examples:
        >>> import jax.numpy as jnp
        >>> from pyrox_gp import PoissonLikelihood
        >>> float(PoissonLikelihood().log_prob(jnp.zeros(3), jnp.zeros(3)))
        -3.0
    """

    name: str = eqx.field(static=True, default="poisson", repr=False, kw_only=True)

    def build(
        self, y: Array, theta: dict[str, Array], data: Mapping[str, ArrayLike]
    ) -> gx.PoissonLikelihood:
        """`gaussx.PoissonLikelihood` holding ``y``."""
        return gx.PoissonLikelihood(y)


class StudentTLikelihood(_GPObservation):
    r"""Heavy-tailed regression $p(y \mid f) = \mathrm{StudentT}(\nu, f, \sigma)$, over
    `gaussx.StudentTLikelihood`.

    Robust to outliers — the heavier-than-Gaussian tails downweight
    observations that are far from the latent. Scalar latent
    (``latent_dim = 1``). Both ``df`` and ``scale`` are positive.

    Attributes:
        df: Degrees of freedom $\nu > 0$. Smaller values give
            heavier tails. ``df -> infinity`` recovers the Gaussian.
        scale: Scale parameter $\sigma > 0$.

    Examples:
        >>> import jax.numpy as jnp
        >>> from pyrox_gp import StudentTLikelihood
        >>> lik = StudentTLikelihood(df=1.0, scale=1.0)
        >>> lp = lik.log_prob(jnp.zeros(1), jnp.zeros(1))
        >>> bool(jnp.allclose(lp, -jnp.log(jnp.pi)))
        True
    """

    df: float | Float[Array, ""]
    scale: float | Float[Array, ""]
    name: str = eqx.field(static=True, default="student_t", repr=False, kw_only=True)

    def build(
        self, y: Array, theta: dict[str, Array], data: Mapping[str, ArrayLike]
    ) -> gx.StudentTLikelihood:
        """`gaussx.StudentTLikelihood` holding ``y`` at ``df`` and ``scale``."""
        return gx.StudentTLikelihood(y, self.df, self.scale)  # ty: ignore[invalid-argument-type]


class SoftmaxLikelihood(_GPObservation):
    r"""Multi-class classification likelihood with softmax link, over
    `gaussx.SoftmaxLikelihood`.

    Each observation has one latent function value per class:
    ``f`` has shape ``(N, num_classes)`` and
    $p(y_n \mid f_n) = \mathrm{Categorical}(\mathrm{softmax}(f_n))$.
    Targets ``y`` are integer class indices in ``[0, num_classes)``. The
    gaussx likelihood takes the latents flattened site by site
    (``f.reshape(-1)``).

    Multi-latent (``latent_dim = num_classes``). The scalar-latent
    advanced inference strategies in `pyrox_gp._inference_nongauss`
    reject this likelihood with a clear error; use SVGP / MAP for now,
    or wait for the multi-latent inference follow-up.

    Attributes:
        num_classes: Number of output classes $C \geq 2$.

    Examples:
        >>> import jax.numpy as jnp
        >>> from pyrox_gp import SoftmaxLikelihood
        >>> lik = SoftmaxLikelihood(num_classes=3)
        >>> lik.latent_dim
        3
        >>> lp = lik.log_prob(jnp.zeros((2, 3)), jnp.array([0, 2]))
        >>> bool(jnp.allclose(lp, 2.0 * jnp.log(1.0 / 3.0)))
        True
    """

    num_classes: int = eqx.field(static=True)
    name: str = eqx.field(static=True, default="softmax", repr=False, kw_only=True)

    def __init__(self, num_classes: int) -> None:
        if num_classes < 2:
            msg = f"num_classes must be >= 2, got {num_classes}"
            raise ValueError(msg)
        self.num_classes = num_classes

    @property
    def latent_dim(self) -> int:
        """Latents per site, ``num_classes``."""
        return self.num_classes

    def _gaussx_latents(self, f: Float[Array, "N C"]) -> Float[Array, " NC"]:
        # (N, C) → (N C), site by site
        return jnp.reshape(f, (-1,))

    def build(
        self,
        y: Int[Array, " N"],
        theta: dict[str, Array],
        data: Mapping[str, ArrayLike],
    ) -> gx.SoftmaxLikelihood:
        """`gaussx.SoftmaxLikelihood` holding ``y`` over ``num_classes`` classes."""
        return gx.SoftmaxLikelihood(y, self.num_classes)


class HeteroscedasticGaussianLikelihood(_GPObservation):
    r"""Gaussian regression with input-dependent noise, over
    `gaussx.HeteroscedasticGaussianLikelihood`.

    Each observation consumes two latents — the mean and the
    log-noise-standard-deviation:
    $p(y_n | f_n^{(0)}, f_n^{(1)}) = N(y_n | f_n^{(0)}, e^{2 f_n^{(1)}})$.

    gaussx's likelihood takes the noise standard deviation as
    $\mathrm{softplus}(g_n)$ and the two latent columns concatenated, so the
    second column enters it as $g_n = \mathrm{softplus}^{-1}(e^{f_n^{(1)}})$:
    the same density, with gaussx's numerics.

    Multi-latent (``latent_dim = 2``). The scalar-latent advanced
    inference strategies reject this likelihood; use SVGP for now.

    Examples:
        >>> import jax.numpy as jnp
        >>> from pyrox_gp import HeteroscedasticGaussianLikelihood
        >>> lik = HeteroscedasticGaussianLikelihood()
        >>> lp = lik.log_prob(jnp.zeros((1, 2)), jnp.zeros(1))
        >>> bool(jnp.allclose(lp, -0.5 * jnp.log(2.0 * jnp.pi)))
        True
    """

    latent_dim: int = eqx.field(static=True, default=2)
    name: str = eqx.field(
        static=True, default="heteroscedastic_gaussian", repr=False, kw_only=True
    )

    def _gaussx_latents(self, f: Float[Array, "N 2"]) -> Float[Array, " 2N"]:
        # σₙ = exp(fₙ⁽¹⁾) = softplus(gₙ), so gₙ = softplus⁻¹(exp(fₙ⁽¹⁾))
        noise_latent = SoftplusTransform().inv(jnp.exp(f[:, 1]))
        # (f⁽⁰⁾, g), the two columns concatenated
        return jnp.concatenate([f[:, 0], noise_latent])

    def build(
        self, y: Array, theta: dict[str, Array], data: Mapping[str, ArrayLike]
    ) -> gx.HeteroscedasticGaussianLikelihood:
        """`gaussx.HeteroscedasticGaussianLikelihood` holding ``y``."""
        return gx.HeteroscedasticGaussianLikelihood(y)
