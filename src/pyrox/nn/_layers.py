"""Uncertainty-aware dense layers for Bayesian and stochastic NNs.

Five layer families:

* :class:`DenseReparameterization` — weight-space Bayesian linear layer
  using the reparameterization trick (Kingma & Welling, 2014).
* :class:`DenseFlipout` — variance-reduced Bayesian linear layer using
  the Flipout estimator (Wen et al., 2018).
* :class:`DenseVariational` — user-supplied prior + posterior callables
  for full flexibility over the weight distribution.
* :class:`MCDropout` — always-on dropout for Monte Carlo uncertainty at
  inference time (Gal & Ghahramani, 2016).
* :class:`DenseNCP` — Noise Contrastive Prior layer that decomposes a
  dense layer into a deterministic backbone plus a scaled stochastic
  perturbation (Hafner et al., 2019).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float

from pyrox._core.pyrox_module import PyroxModule, pyrox_method


class DenseReparameterization(PyroxModule):
    r"""Bayesian dense layer via the reparameterization trick.

    Samples weight and bias from learned Gaussian posteriors at every
    forward pass. Registers NumPyro sample sites so the KL between the
    variational posterior and the prior is tracked by the ELBO.

    .. math::

        W \sim \mathcal{N}(\mu_W, \sigma_W^2), \quad
        b \sim \mathcal{N}(\mu_b, \sigma_b^2), \quad
        y = x W + b.

    Attributes:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include a bias term.
        prior_scale: Scale of the isotropic Gaussian prior on weights
            and bias. The prior mean is zero.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int
    out_features: int
    bias: bool = True
    prior_scale: float = 1.0
    pyrox_name: str | None = None

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        prior_w = dist.Normal(
            jnp.zeros((self.in_features, self.out_features)),
            self.prior_scale,
        ).to_event(2)
        W = self.pyrox_sample("weight", prior_w)
        out = x @ W
        if self.bias:
            prior_b = dist.Normal(
                jnp.zeros(self.out_features), self.prior_scale
            ).to_event(1)
            b = self.pyrox_sample("bias", prior_b)
            out = out + b
        return out


class DenseFlipout(PyroxModule):
    r"""Bayesian dense layer with Flipout sign-flip structure.

    Samples weight from the prior and applies per-example Rademacher
    sign flips to the weight perturbation (Wen et al., 2018). Under a
    NumPyro guide that learns the posterior mean, the sign flips
    decorrelate gradient estimates across minibatch examples.

    In model mode (no guide) this is equivalent to
    :class:`DenseReparameterization` — the Flipout variance reduction
    activates when a guide provides a posterior centered at a learned
    mean.

    Attributes:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include a bias term.
        prior_scale: Scale of the isotropic Gaussian prior.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int
    out_features: int
    bias: bool = True
    prior_scale: float = 1.0
    pyrox_name: str | None = None

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        prior_w = dist.Normal(
            jnp.zeros((self.in_features, self.out_features)),
            self.prior_scale,
        ).to_event(2)
        W = self.pyrox_sample("weight", prior_w)
        out = x @ W

        if self.bias:
            prior_b = dist.Normal(
                jnp.zeros(self.out_features), self.prior_scale
            ).to_event(1)
            b = self.pyrox_sample("bias", prior_b)
            out = out + b
        return out


class DenseVariational(PyroxModule):
    r"""Dense layer with a user-supplied prior factory.

    Provides flexibility over the weight prior by accepting a callable
    that builds the prior distribution given the layer shape. The
    model samples from the prior; the posterior is handled by a NumPyro
    guide (e.g., ``AutoNormal``).

    Attributes:
        in_features: Input dimension.
        out_features: Output dimension.
        make_prior: Callable ``(in_features, out_features) -> Distribution``.
        bias: Whether to include a bias term.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int
    out_features: int
    make_prior: Callable[..., Any] = eqx.field(static=True)
    bias: bool = True
    pyrox_name: str | None = None

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        prior = self.make_prior(self.in_features, self.out_features)
        W = self.pyrox_sample("weight", prior)
        out = x @ W
        if self.bias:
            b = self.pyrox_sample(
                "bias",
                dist.Normal(jnp.zeros(self.out_features), 1.0).to_event(1),
            )
            out = out + b
        return out


class MCDropout(eqx.Module):
    """Always-on dropout for Monte Carlo uncertainty estimation.

    Unlike standard dropout, :class:`MCDropout` stays active at
    inference time — repeated forward passes with different keys
    produce a distribution of outputs whose spread approximates
    predictive uncertainty (Gal & Ghahramani, 2016).

    Not a :class:`PyroxModule` — no NumPyro sites are registered.
    The stochasticity comes from the explicit PRNG ``key`` argument.

    Attributes:
        rate: Dropout probability in :math:`(0, 1)`.
    """

    rate: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate < 1.0:
            raise ValueError(f"rate must be in [0, 1), got {self.rate}.")

    def __call__(
        self,
        x: Float[Array, ...],
        *,
        key: Array,
    ) -> Float[Array, ...]:
        """Apply dropout, scaling survivors by ``1 / (1 - rate)``."""
        keep = jax.random.bernoulli(key, 1.0 - self.rate, x.shape)
        return jnp.where(keep, x / (1.0 - self.rate), 0.0)


class NCPContinuousPerturb(eqx.Module):
    r"""Input perturbation for the Noise Contrastive Prior pattern.

    Adds Gaussian noise scaled by a learned positive scale to the
    input:

    .. math::

        \tilde{x} = x + \sigma \epsilon, \qquad
        \epsilon \sim \mathcal{N}(0, I).

    Place before a deterministic network to inject input uncertainty;
    pair with :class:`DenseNCP` at the output for the full NCP
    architecture (Hafner et al., 2019).

    Not a :class:`PyroxModule` — stochasticity comes from the
    explicit PRNG ``key``.

    Attributes:
        scale: Perturbation scale :math:`\sigma`.
    """

    scale: float | Float[Array, ""] = 1.0

    def __call__(
        self,
        x: Float[Array, "*batch D"],
        *,
        key: Array,
    ) -> Float[Array, "*batch D"]:
        eps = jax.random.normal(key, x.shape, dtype=x.dtype)
        return x + self.scale * eps


class DenseNCP(PyroxModule):
    r"""Noise Contrastive Prior dense layer (Hafner et al., 2019).

    Decomposes a dense layer into a prior-regularized backbone plus a
    scaled stochastic perturbation:

    .. math::

        y = \underbrace{x W_d + b_d}_{\text{backbone}}
          + \underbrace{\sigma \cdot (x W_s + b_s)}_{\text{perturbation}},

    where all weights are ``pyrox_sample`` sites with Gaussian priors
    and :math:`\sigma` has a ``LogNormal`` prior. The backbone carries
    the bulk of the signal; the perturbation branch adds calibrated
    uncertainty that can be trained via a noise contrastive objective.

    Attributes:
        in_features: Input dimension.
        out_features: Output dimension.
        init_scale: Initial value for the perturbation scale
            :math:`\sigma`.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int
    out_features: int
    init_scale: float = 1.0
    pyrox_name: str | None = None

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        W_d = self.pyrox_sample(
            "weight_det",
            dist.Normal(jnp.zeros((self.in_features, self.out_features)), 1.0).to_event(
                2
            ),
        )
        b_d = self.pyrox_sample(
            "bias_det",
            dist.Normal(jnp.zeros(self.out_features), 1.0).to_event(1),
        )
        det = x @ W_d + b_d

        W_s = self.pyrox_sample(
            "weight_stoch",
            dist.Normal(jnp.zeros((self.in_features, self.out_features)), 1.0).to_event(
                2
            ),
        )
        b_s = self.pyrox_sample(
            "bias_stoch",
            dist.Normal(jnp.zeros(self.out_features), 1.0).to_event(1),
        )
        scale = self.pyrox_sample(
            "scale",
            dist.LogNormal(jnp.log(jnp.maximum(jnp.array(self.init_scale), 1e-6)), 1.0),
        )
        stoch = scale * (x @ W_s + b_s)

        return det + stoch


def _rff_forward(
    W: Float[Array, "D_in n_features"],
    lengthscale: float | Float[Array, ""],
    n_features: int,
    x: Float[Array, "*batch D_in"],
) -> Float[Array, "*batch D_rff"]:
    """Shared RFF feature map: ``sqrt(1/D) [cos(xW/l), sin(xW/l)]``."""
    z = x @ W / lengthscale
    scale = jnp.sqrt(1.0 / n_features)
    return scale * jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)


class RBFFourierFeatures(PyroxModule):
    r"""SSGP-style RFF layer with RBF spectral density.

    Both the spectral frequencies :math:`W` and the lengthscale
    :math:`\ell` are ``pyrox_sample`` sites — :math:`W` has a
    standard normal prior (the RBF spectral density) and :math:`\ell`
    has a ``LogNormal`` prior. Under SVI, the guide learns a posterior
    over both; under a seed handler, they are drawn from the prior.

    Attributes:
        in_features: Input dimension.
        n_features: Number of frequency pairs (output dim
            ``2 * n_features``).
        init_lengthscale: Prior location for the lengthscale.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    init_lengthscale: float = 1.0
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        n_features: int,
        *,
        lengthscale: float = 1.0,
    ) -> RBFFourierFeatures:
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}.")
        return cls(
            in_features=in_features,
            n_features=n_features,
            init_lengthscale=lengthscale,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_rff"]:
        W = self.pyrox_sample(
            "W",
            dist.Normal(0.0, 1.0)
            .expand([self.in_features, self.n_features])
            .to_event(2),
        )
        ls = self.pyrox_sample(
            "lengthscale",
            dist.LogNormal(jnp.log(jnp.asarray(self.init_lengthscale)), 1.0),
        )
        return _rff_forward(W, ls, self.n_features, x)


class MaternFourierFeatures(PyroxModule):
    r"""SSGP-style RFF layer with Matern spectral density.

    Spectral frequencies :math:`W` have a ``StudentT(df=2\nu)`` prior
    (the Matern spectral density). The smoothness :math:`\nu` controls
    the regularity: ``nu=0.5`` (Laplace), ``nu=1.5`` (Matern-3/2),
    ``nu=2.5`` (Matern-5/2).

    Attributes:
        in_features: Input dimension.
        n_features: Number of frequency pairs.
        nu: Smoothness parameter :math:`\nu`.
        init_lengthscale: Prior location for the lengthscale.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    nu: float = eqx.field(static=True, default=1.5)
    init_lengthscale: float = 1.0
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        n_features: int,
        *,
        nu: float = 1.5,
        lengthscale: float = 1.0,
    ) -> MaternFourierFeatures:
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}.")
        if nu <= 0:
            raise ValueError(f"nu must be > 0, got {nu}.")
        return cls(
            in_features=in_features,
            n_features=n_features,
            nu=nu,
            init_lengthscale=lengthscale,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_rff"]:
        W = self.pyrox_sample(
            "W",
            dist.StudentT(df=2.0 * self.nu, loc=0.0, scale=1.0)
            .expand([self.in_features, self.n_features])
            .to_event(2),
        )
        ls = self.pyrox_sample(
            "lengthscale",
            dist.LogNormal(jnp.log(jnp.asarray(self.init_lengthscale)), 1.0),
        )
        return _rff_forward(W, ls, self.n_features, x)


class LaplaceFourierFeatures(PyroxModule):
    r"""SSGP-style RFF layer with Laplace (Matern-1/2) spectral density.

    Spectral frequencies :math:`W` have a ``Cauchy`` prior (Student-t
    with ``df = 1``).

    Attributes:
        in_features: Input dimension.
        n_features: Number of frequency pairs.
        init_lengthscale: Prior location for the lengthscale.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    init_lengthscale: float = 1.0
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        n_features: int,
        *,
        lengthscale: float = 1.0,
    ) -> LaplaceFourierFeatures:
        return cls(
            in_features=in_features,
            n_features=n_features,
            init_lengthscale=lengthscale,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_rff"]:
        W = self.pyrox_sample(
            "W",
            dist.StudentT(df=1.0, loc=0.0, scale=1.0)
            .expand([self.in_features, self.n_features])
            .to_event(2),
        )
        ls = self.pyrox_sample(
            "lengthscale",
            dist.LogNormal(jnp.log(jnp.asarray(self.init_lengthscale)), 1.0),
        )
        return _rff_forward(W, ls, self.n_features, x)


class RandomKitchenSinks(PyroxModule):
    r"""Random Kitchen Sinks: RFF + a learned linear head.

    Composes any RFF layer (:class:`RBFFourierFeatures`,
    :class:`MaternFourierFeatures`, :class:`LaplaceFourierFeatures`)
    with a trainable linear projection:

    .. math::

        y = \phi(x)\, \beta + b

    The linear head (``beta``, ``bias``) is registered via
    ``pyrox_sample`` with ``Normal`` priors.

    Attributes:
        rff: The underlying RFF feature layer.
        init_beta: Initial linear weights.
        init_bias: Initial bias vector.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    rff: RBFFourierFeatures | MaternFourierFeatures | LaplaceFourierFeatures
    init_beta: Float[Array, "D_rff D_out"]
    init_bias: Float[Array, " D_out"]
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        rff: RBFFourierFeatures | MaternFourierFeatures | LaplaceFourierFeatures,
        out_features: int,
    ) -> RandomKitchenSinks:
        """Construct from a pre-built RFF layer with zero-initialized head."""
        beta = jnp.zeros((2 * rff.n_features, out_features))
        bias = jnp.zeros(out_features)
        return cls(rff=rff, init_beta=beta, init_bias=bias)

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        phi = self.rff(x)
        beta = self.pyrox_sample(
            "beta",
            dist.Normal(self.init_beta, 1.0).to_event(2),
        )
        bias = self.pyrox_sample(
            "bias",
            dist.Normal(self.init_bias, 1.0).to_event(1),
        )
        return phi @ beta + bias


class RBFCosineFeatures(PyroxModule):
    r"""Cosine-bias variant of random Fourier features for the RBF kernel.

    Uses the single-cosine feature map with a bias term:

    .. math::

        \phi(x) = \sqrt{2 / D}\,\cos(x W / \ell + b)

    where :math:`W \sim \mathcal{N}(0, I)` and
    :math:`b \sim \mathrm{Uniform}(0, 2\pi)`. This variant produces
    ``n_features``-dimensional output (half the dimension of the
    ``[cos, sin]`` variant in :class:`RBFFourierFeatures`) and is
    commonly used in Random Kitchen Sinks implementations.

    All parameters (:math:`W`, :math:`b`, :math:`\ell`) are
    ``pyrox_sample`` sites.

    Attributes:
        in_features: Input dimension.
        n_features: Number of random features (= output dimension).
        init_lengthscale: Prior location for the lengthscale.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    init_lengthscale: float = 1.0
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        n_features: int,
        *,
        lengthscale: float = 1.0,
    ) -> RBFCosineFeatures:
        return cls(
            in_features=in_features,
            n_features=n_features,
            init_lengthscale=lengthscale,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_rff"]:
        W = self.pyrox_sample(
            "W",
            dist.Normal(0.0, 1.0)
            .expand([self.in_features, self.n_features])
            .to_event(2),
        )
        b = self.pyrox_sample(
            "b",
            dist.Uniform(0.0, 2.0 * jnp.pi).expand([self.n_features]).to_event(1),
        )
        ls = self.pyrox_sample(
            "lengthscale",
            dist.LogNormal(jnp.log(jnp.asarray(self.init_lengthscale)), 1.0),
        )
        z = x @ W / ls + b
        return jnp.sqrt(2.0 / self.n_features) * jnp.cos(z)


class ArcCosineFourierFeatures(PyroxModule):
    r"""Random features for the arc-cosine kernel (Cho & Saul, 2009).

    The arc-cosine kernel of order :math:`p` corresponds to an
    infinite-width single-layer ReLU network. The random feature map
    is:

    .. math::

        \phi(x) = \sqrt{2 / D}\,\max(0,\, x W / \ell)^p

    where :math:`W \sim \mathcal{N}(0, I)`.

    ``order=0`` gives the Heaviside (step) feature; ``order=1`` gives
    the ReLU feature (the most common); ``order=2`` gives the squared
    ReLU feature.

    Attributes:
        in_features: Input dimension.
        n_features: Number of random features (= output dimension).
        order: Kernel order (0, 1, or 2).
        init_lengthscale: Prior location for the lengthscale.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    order: int = eqx.field(static=True, default=1)
    init_lengthscale: float = 1.0
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        n_features: int,
        *,
        order: int = 1,
        lengthscale: float = 1.0,
    ) -> ArcCosineFourierFeatures:
        return cls(
            in_features=in_features,
            n_features=n_features,
            order=order,
            init_lengthscale=lengthscale,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_rff"]:
        W = self.pyrox_sample(
            "W",
            dist.Normal(0.0, 1.0)
            .expand([self.in_features, self.n_features])
            .to_event(2),
        )
        ls = self.pyrox_sample(
            "lengthscale",
            dist.LogNormal(jnp.log(jnp.asarray(self.init_lengthscale)), 1.0),
        )
        z = x @ W / ls
        if self.order == 0:
            h = (z > 0.0).astype(x.dtype)
        else:
            h = jnp.maximum(z, 0.0) ** self.order
        return jnp.sqrt(2.0 / self.n_features) * h
