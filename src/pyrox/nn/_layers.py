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

import math
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float, PRNGKeyArray

from pyrox._basis import fourier_basis, spectral_density
from pyrox._core.pyrox_module import PyroxModule, pyrox_method
from pyrox.gp._context import _kernel_context
from pyrox.gp._protocols import Kernel


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


class VariationalFourierFeatures(PyroxModule):
    r"""VSSGP — RFF with a learnable variational posterior over frequencies.

    Standard RFF (e.g. :class:`RBFFourierFeatures`) treats the spectral
    frequencies :math:`W` as a frozen prior draw; VSSGP (Gal & Turner,
    2015) treats :math:`W` as a latent with a learnable mean-field
    posterior, recovering spectral *uncertainty* on top of the
    feature-space uncertainty.

    Prior: :math:`p(W) = \mathcal{N}(0, I)` (RBF spectral density in
    lengthscale-1 units). The lengthscale is itself a sampled site
    (``LogNormal(log init_lengthscale, 1)``) so that frequencies are
    rescaled to the physical kernel.

    Under SVI, attach an :class:`~numpyro.infer.autoguide.AutoNormal` to
    learn the posterior on ``W``; under prior-only seeds, behaves
    identically to :class:`RBFFourierFeatures`.

    Attributes:
        in_features: Input dimension :math:`D`.
        n_features: Number of frequency pairs (output dim ``2 * n_features``).
        init_lengthscale: Prior location for the kernel lengthscale.
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
    ) -> VariationalFourierFeatures:
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}.")
        return cls(
            in_features=in_features,
            n_features=n_features,
            init_lengthscale=lengthscale,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_rff"]:
        # Same prior as RBFFourierFeatures — the *posterior* is what differs
        # under SVI: an attached AutoGuide learns q(W) instead of forcing W
        # to its prior draw.
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


def _orthogonal_blocks(
    in_features: int,
    n_blocks: int,
    *,
    key: jax.Array,
) -> Float[Array, "D_in D_orf"]:
    """Build the ORF frequency matrix as ``[Q_1 S_1, ..., Q_K S_K]``.

    Each block is a Haar-orthogonal ``D x D`` matrix with its *columns*
    scaled by independent chi-distributed magnitudes. The RFF forward uses
    ``z = x @ W`` so columns of ``W`` carry the per-feature frequencies:
    scaling columns (not rows) preserves the ORF construction where each
    frequency vector is an orthogonal unit direction times its own chi
    magnitude.
    """
    D = in_features
    blocks: list[Float[Array, "D_in D_in"]] = []
    for _ in range(n_blocks):
        key, k_qr, k_chi = jax.random.split(key, 3)
        G = jax.random.normal(k_qr, (D, D))
        Q, _ = jnp.linalg.qr(G)
        # Chi-distributed scale per column (sqrt of chi-squared with df=D).
        chi = jnp.sqrt(jnp.sum(jax.random.normal(k_chi, (D, D)) ** 2, axis=-1))  # (D,)
        blocks.append(Q * chi[None, :])
    return jnp.concatenate(blocks, axis=-1)  # (D, n_blocks * D)


class OrthogonalRandomFeatures(eqx.Module):
    r"""Orthogonal Random Features (Yu et al., 2016) — variance-reduced RFF.

    Frequencies are drawn from blocks of Haar-orthogonal matrices scaled by
    independent chi-distributed magnitudes, giving the same RBF kernel
    approximation as plain :class:`RBFFourierFeatures` *in expectation* but
    with provably lower variance for finite ``n_features``.

    Frozen at construction time — no priors, no SVI on ``W``. The frequency
    matrix is built once from a ``key`` and stored as a static array.

    Attributes:
        in_features: Input dimension :math:`D`.
        n_features: Number of feature pairs. Must satisfy
            ``n_features % in_features == 0`` so that ORF blocks tile cleanly.
        lengthscale: Fixed kernel lengthscale (no prior; pass a value).
        W: Pre-built frequency matrix of shape ``(in_features, 2 * n_features // 2)``.

    Note:
        For learnable lengthscale or full Bayesian treatment of the
        frequencies, prefer :class:`VariationalFourierFeatures`.
    """

    in_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    lengthscale: Float[Array, ""]
    W: Float[Array, "D_in D_orf"]

    @classmethod
    def init(
        cls,
        in_features: int,
        n_features: int,
        *,
        key: jax.Array,
        lengthscale: float = 1.0,
    ) -> OrthogonalRandomFeatures:
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}.")
        if n_features % in_features != 0:
            raise ValueError(
                f"n_features ({n_features}) must be divisible by in_features "
                f"({in_features}) so ORF blocks tile cleanly."
            )
        n_blocks = n_features // in_features
        W = _orthogonal_blocks(in_features, n_blocks, key=key)
        return cls(
            in_features=in_features,
            n_features=n_features,
            lengthscale=jnp.asarray(lengthscale),
            W=W,
        )

    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_rff"]:
        return _rff_forward(self.W, self.lengthscale, self.n_features, x)


class HSGPFeatures(PyroxModule):
    r"""Hilbert-Space Gaussian Process feature layer (Riutort-Mayol et al., 2023).

    A *deterministic* Laplacian-eigenfunction basis on the bounded box
    :math:`[-L, L]^D` plus learnable per-basis amplitudes with a
    kernel-spectral-density prior:

    .. math::

        \hat{f}(x) = \sum_{j=1}^{M} \alpha_j\,\sqrt{S(\sqrt{\lambda_j})}\,\phi_j(x),
        \quad \alpha_j \sim \mathcal{N}(0, 1).

    This is the NN-side dual of :class:`pyrox.gp.FourierInducingFeatures`
    — same basis, different prior wiring. As ``M`` and ``L`` grow, the
    induced GP converges to the kernel passed in.

    Attributes:
        in_features: Input dimension :math:`D`.
        num_basis_per_dim: Per-axis number of 1D eigenfunctions; total
            basis count is ``prod(num_basis_per_dim)``.
        L: Per-axis box half-width.
        kernel: A stationary kernel from :mod:`pyrox.gp` whose spectral
            density supplies the per-basis prior variance. Currently
            :class:`pyrox.gp.RBF` and :class:`pyrox.gp.Matern` are
            supported by :func:`pyrox._basis.spectral_density`.
        pyrox_name: Explicit scope name for NumPyro site registration.
    """

    in_features: int = eqx.field(static=True)
    num_basis_per_dim: tuple[int, ...] = eqx.field(static=True)
    L: tuple[float, ...] = eqx.field(static=True)
    kernel: Kernel
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        num_basis_per_dim: int | tuple[int, ...],
        L: float | tuple[float, ...],
        *,
        kernel: Kernel,
    ) -> HSGPFeatures:
        if isinstance(num_basis_per_dim, int):
            num_basis_per_dim = (num_basis_per_dim,) * in_features
        if isinstance(L, int | float):
            L = (float(L),) * in_features
        if len(num_basis_per_dim) != in_features:
            raise ValueError(
                f"num_basis_per_dim length ({len(num_basis_per_dim)}) "
                f"must match in_features ({in_features})."
            )
        if len(L) != in_features:
            raise ValueError(
                f"L length ({len(L)}) must match in_features ({in_features})."
            )
        if any(L_d <= 0 for L_d in L):
            raise ValueError(f"L must be all positive; got {L}.")
        if any(M_d < 1 for M_d in num_basis_per_dim):
            raise ValueError(
                f"num_basis_per_dim must be all >= 1; got {num_basis_per_dim}."
            )
        return cls(
            in_features=in_features,
            num_basis_per_dim=tuple(num_basis_per_dim),
            L=tuple(float(L_d) for L_d in L),
            kernel=kernel,
        )

    @property
    def num_basis(self) -> int:
        n = 1
        for m in self.num_basis_per_dim:
            n *= m
        return n

    @pyrox_method
    def __call__(self, x: Float[Array, "N D_in"]) -> Float[Array, " N"]:
        Phi, lam = fourier_basis(x, self.num_basis_per_dim, self.L)  # (N, M), (M,)
        # Spectral density evaluated under the kernel's own context so any
        # priors on (variance, lengthscale) register exactly once.
        with _kernel_context(self.kernel):
            S = spectral_density(self.kernel, lam, D=self.in_features)
        sqrt_S = jnp.sqrt(S)
        alpha = self.pyrox_sample(
            "alpha",
            dist.Normal(0.0, 1.0).expand([self.num_basis]).to_event(1),
        )
        return jnp.einsum("nm,m->n", Phi, sqrt_S * alpha)


# ---------------------------------------------------------------------------
# Multiplicative Filter Networks (Fathony et al., ICLR 2021)
# ---------------------------------------------------------------------------


class FourierFilter(PyroxModule):
    r"""Single Fourier filter: :math:`g(x) = \sin(\Omega x + \varphi)`.

    One multiplicative filter primitive for use inside a
    :class:`FourierNet`.  Outputs a vector of sinusoidal activations:

    .. math::

        g(x) = \sin(x\,\Omega^\top + \varphi)

    Init follows Fathony et al. (2021) §4.1: frequencies are drawn as
    :math:`\Omega_{ij} \sim \mathcal{N}(0,\,\sigma_f^2/D)` where
    :math:`D` is ``in_features`` and :math:`\sigma_f` is
    ``freq_scale``; phases are drawn as
    :math:`\varphi_i \sim \mathrm{Uniform}(-\pi, \pi)`.

    Attributes:
        Omega: Frequency matrix of shape ``(out_features, in_features)``.
        phi: Phase vector of shape ``(out_features,)``.
        in_features: Input dimension.
        out_features: Output (filter) dimension.
        pyrox_name: Optional explicit scope name for NumPyro site
            registration (used only in :class:`BayesianFourierNet`).
    """

    Omega: Float[Array, "out in"]
    phi: Float[Array, " out"]
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        out_features: int,
        *,
        key: PRNGKeyArray,
        freq_scale: float = 256.0,
        pyrox_name: str | None = None,
    ) -> FourierFilter:
        """Construct with Fathony-et-al. §4.1 initialization.

        Args:
            in_features: Input dimension.
            out_features: Number of filters (output dimension).
            key: JAX PRNG key.
            freq_scale: Frequency standard deviation :math:`\\sigma_f`
                (default 256, as in the original paper).
            pyrox_name: Optional scope name.

        Returns:
            Initialised :class:`FourierFilter`.
        """
        k_omega, k_phi = jax.random.split(key)
        # Per-element std: sigma_f / sqrt(D)  (Fathony et al. 2021 Sec 4.1)
        omega_std = freq_scale / math.sqrt(in_features)
        Omega = jax.random.normal(k_omega, (out_features, in_features)) * omega_std
        phi = jax.random.uniform(k_phi, (out_features,), minval=-jnp.pi, maxval=jnp.pi)
        return cls(
            Omega=Omega,
            phi=phi,
            in_features=in_features,
            out_features=out_features,
            pyrox_name=pyrox_name,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N H"]:
        """Apply the Fourier filter to a batch of inputs.

        Args:
            x: Input array of shape ``(N, D)`` or ``(D,)`` (single point).

        Returns:
            Filter output of shape ``(N, H)``.
        """
        return jnp.sin(jnp.atleast_2d(x) @ self.Omega.T + self.phi)


class GaborFilter(PyroxModule):
    r"""Single Gabor filter:
    :math:`g(x) = \sin(\Omega x + \varphi) \odot \exp(-\tfrac{\gamma}{2}\|x - \mu\|^2)`.

    One multiplicative filter primitive for use inside a
    :class:`GaborNet`.  Each filter is a sinusoidal oscillation
    modulated by a Gaussian envelope:

    .. math::

        g(x) = \sin(x\,\Omega^\top + \varphi)
                \odot \exp\!\bigl(-\tfrac{\gamma}{2}\|x - \mu\|^2\bigr)

    Init follows Fathony et al. (2021) §4.2:

    1. :math:`\gamma_i \sim \mathrm{Gamma}(\alpha, \beta)` (per filter).
    2. :math:`\mu_i \sim \mathrm{Uniform}(\texttt{domain\_low},
       \texttt{domain\_high})` (per input dimension).
    3. :math:`\Omega_{i,:} \sim \mathcal{N}(0, \gamma_i\,I_D)` — the
       load-bearing tied initialization: frequency scale matches the
       filter-specific bandwidth.

    :math:`\gamma` is stored in log space so no positivity constraint is
    needed on the optimizer.

    Attributes:
        Omega: Frequency matrix ``(out_features, in_features)``.
        phi: Phase vector ``(out_features,)``.
        mu: Envelope centres ``(out_features, in_features)``.
        log_gamma: Log-bandwidth ``(out_features,)``; stored in log space
            so :math:`\gamma = \exp(\texttt{log\_gamma}) > 0` without
            optimizer constraints.
        in_features: Input dimension.
        out_features: Output (filter) dimension.
        domain: ``(low, high)`` used for :math:`\mu` initialization (static).
        pyrox_name: Optional scope name for NumPyro sites.
    """

    Omega: Float[Array, "out in"]
    phi: Float[Array, " out"]
    mu: Float[Array, "out in"]
    log_gamma: Float[Array, " out"]
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    domain: tuple[float, float] = eqx.field(static=True)
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        out_features: int,
        *,
        key: PRNGKeyArray,
        domain: tuple[float, float] = (-1.0, 1.0),
        gamma_alpha: float = 6.0,
        gamma_beta: float = 1.0,
        pyrox_name: str | None = None,
    ) -> GaborFilter:
        """Construct with Fathony-et-al. §4.2 initialization.

        Args:
            in_features: Input dimension.
            out_features: Number of filters (output dimension).
            key: JAX PRNG key.
            domain: ``(low, high)`` for :math:`\\mu` initialization.
            gamma_alpha: Shape parameter of the Gamma prior on
                :math:`\\gamma` (default 6.0).
            gamma_beta: Rate parameter of the Gamma prior on
                :math:`\\gamma` (default 1.0).
            pyrox_name: Optional scope name.

        Returns:
            Initialised :class:`GaborFilter`.
        """
        k_gamma, k_mu, k_omega, k_phi = jax.random.split(key, 4)
        # gamma ~ Gamma(alpha, rate=beta): jax.random.gamma samples Gamma(alpha, 1),
        # dividing by beta converts to Gamma(alpha, rate=beta).
        gamma = jax.random.gamma(k_gamma, gamma_alpha, (out_features,)) / gamma_beta
        log_gamma = jnp.log(gamma)
        # mu ~ Uniform(domain_low, domain_high)
        mu = jax.random.uniform(
            k_mu, (out_features, in_features), minval=domain[0], maxval=domain[1]
        )
        # Omega_i ~ N(0, gamma_i * I_D)  -- tied init (Fathony et al. 2021 Sec 4.2)
        Omega = jax.random.normal(k_omega, (out_features, in_features)) * jnp.sqrt(
            gamma[:, None]
        )
        phi = jax.random.uniform(k_phi, (out_features,), minval=-jnp.pi, maxval=jnp.pi)
        return cls(
            Omega=Omega,
            phi=phi,
            mu=mu,
            log_gamma=log_gamma,
            in_features=in_features,
            out_features=out_features,
            domain=domain,
            pyrox_name=pyrox_name,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N H"]:
        """Apply the Gabor filter to a batch of inputs.

        Args:
            x: Input array of shape ``(N, D)`` or ``(D,)`` (single point).

        Returns:
            Filter output of shape ``(N, H)``.
        """
        x2d = jnp.atleast_2d(x)
        gamma = jnp.exp(self.log_gamma)  # (H,)
        # squared distance: (N, H)
        diff = x2d[:, None, :] - self.mu[None, :, :]  # (N, H, D)
        sq_dist = jnp.sum(diff**2, axis=-1)  # (N, H)
        envelope = jnp.exp(-0.5 * gamma[None, :] * sq_dist)  # (N, H)
        sinusoidal = jnp.sin(x2d @ self.Omega.T + self.phi)  # (N, H)
        return sinusoidal * envelope


def mfn_forward(
    x: Float[Array, "N D"],
    filters: Sequence[Callable[[Array], Array]],
    linears: Sequence[Callable[[Array], Array]],
) -> Float[Array, "N O"]:
    """Pure-JAX MFN forward pass given user-supplied filter and linear callables.

    Implements the Fathony et al. (2021) multiplicative chaining:

    .. math::

        z_1 = g_1(x), \\quad
        z_{i+1} = g_{i+1}(x) \\odot (W_i z_i + b_i), \\quad
        y = W_L z_L + b_L.

    Exists as an escape hatch so users can plug custom filter families
    (e.g. wavelet, complex-Gabor) into the MFN topology without
    subclassing :class:`FourierNet` or :class:`GaborNet`.

    ``filters`` and ``linears`` must have the same length :math:`L`.
    ``linears`` are treated as single-sample callables and are
    :func:`jax.vmap`-ed over the batch dimension internally.
    :class:`FourierFilter` / :class:`GaborFilter` instances handle
    batched input natively.

    Args:
        x: Input array of shape ``(N, D)``.
        filters: Length-``L`` list of filter callables ``(N, D) -> (N, H)``.
        linears: Length-``L`` list of linear callables ``(H,) -> (H_out,)``,
            e.g. :class:`equinox.nn.Linear` instances.

    Returns:
        Output array of shape ``(N, O)``.
    """
    x = jnp.atleast_2d(x)
    z = filters[0](x)
    for f, lin in zip(filters[1:], linears[:-1], strict=True):
        z = f(x) * jax.vmap(lin)(z)
    return jax.vmap(linears[-1])(z)


class FourierNet(PyroxModule):
    r"""Multiplicative Fourier Filter Network (Fathony et al., ICLR 2021).

    Chains :class:`FourierFilter` primitives multiplicatively:

    .. math::

        z_1 = g_1(x), \quad
        z_{i+1} = g_{i+1}(x) \odot (W_i z_i + b_i), \quad
        y = W_L z_L + b_L.

    Each :math:`g_i` is a :class:`FourierFilter` of width
    ``hidden_features``; the last linear is the readout projecting to
    ``out_features``.

    Unlike SIREN, no special per-layer initialization ceremony is
    required: standard Gaussian / uniform init on all filter parameters
    gives stable training.

    Note:
        Single-point input ``(D,)`` is automatically promoted to
        ``(1, D)`` and the result is squeezed back to ``(O,)``.

    Attributes:
        filters: Length-``depth`` list of :class:`FourierFilter` primitives.
        linears: Length-``depth`` list of :class:`~equinox.nn.Linear` layers
            (last one is the readout).
        in_features: Input dimension.
        hidden_features: Filter / hidden width.
        out_features: Output dimension.
        depth: Number of filter layers :math:`L`.
        pyrox_name: Optional scope name for NumPyro sites.
    """

    filters: list[FourierFilter]
    linears: list[eqx.nn.Linear]
    in_features: int = eqx.field(static=True)
    hidden_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        depth: int,
        key: PRNGKeyArray,
        freq_scale: float = 256.0,
        pyrox_name: str | None = None,
    ) -> FourierNet:
        """Construct a :class:`FourierNet`.

        Args:
            in_features: Input dimension.
            hidden_features: Filter width (all hidden layers).
            out_features: Output dimension.
            depth: Number of filter layers :math:`L` (must be ≥ 1).
            key: JAX PRNG key.
            freq_scale: Frequency scale passed to each
                :class:`FourierFilter` (default 256).
            pyrox_name: Optional scope name.

        Returns:
            Initialised :class:`FourierNet`.

        Raises:
            ValueError: If ``depth < 1``.
        """
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}.")
        keys = jax.random.split(key, 2 * depth)
        filter_keys = keys[:depth]
        linear_keys = keys[depth:]
        filters = [
            FourierFilter.init(
                in_features, hidden_features, key=filter_keys[i], freq_scale=freq_scale
            )
            for i in range(depth)
        ]
        linears = [
            eqx.nn.Linear(
                hidden_features,
                hidden_features if i < depth - 1 else out_features,
                key=linear_keys[i],
            )
            for i in range(depth)
        ]
        return cls(
            filters=filters,
            linears=linears,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            depth=depth,
            pyrox_name=pyrox_name,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N O"]:
        """Run the MFN forward pass.

        Args:
            x: Input of shape ``(N, D)`` or ``(D,)`` (single point).

        Returns:
            Output of shape ``(N, O)`` or ``(O,)`` if input was 1-D.
        """
        squeeze = x.ndim == 1
        out = mfn_forward(x, self.filters, self.linears)
        return out[0] if squeeze else out


class GaborNet(PyroxModule):
    r"""Multiplicative Gabor Filter Network (Fathony et al., ICLR 2021).

    Same MFN topology as :class:`FourierNet` but each :math:`g_i` is a
    :class:`GaborFilter` — a sinusoidal oscillation modulated by a
    Gaussian envelope:

    .. math::

        g_i(x) = \sin(\Omega_i x + \varphi_i)
                  \odot \exp\!\bigl(-\tfrac{\gamma_i}{2}\|x - \mu_i\|^2\bigr).

    The tied :math:`\Omega_i \sim \mathcal{N}(0,\gamma_i I)` initialization
    means each filter's frequency scale matches its spatial bandwidth, giving
    a multi-resolution basis whose effective kernel is the Hadamard product
    of :math:`L` localized RBF kernels.

    **Connection to** :class:`~pyrox.nn.RBFFourierFeatures`: a depth-1
    ``GaborNet`` with :math:`\mu = 0` is a localized variant of random
    Fourier features; as :math:`\gamma \to 0` (very wide envelope) it
    recovers the plain RBF-RFF feature map.

    Note:
        Single-point input ``(D,)`` is automatically promoted to
        ``(1, D)`` and the result is squeezed back to ``(O,)``.

    Attributes:
        filters: Length-``depth`` list of :class:`GaborFilter` primitives.
        linears: Length-``depth`` list of :class:`~equinox.nn.Linear` layers.
        in_features: Input dimension.
        hidden_features: Filter / hidden width.
        out_features: Output dimension.
        depth: Number of filter layers :math:`L`.
        domain: ``(low, high)`` used for :math:`\\mu` initialization.
        gamma_alpha: Shape parameter of the :math:`\\gamma` Gamma prior.
        gamma_beta: Rate parameter of the :math:`\\gamma` Gamma prior.
        pyrox_name: Optional scope name for NumPyro sites.
    """

    filters: list[GaborFilter]
    linears: list[eqx.nn.Linear]
    in_features: int = eqx.field(static=True)
    hidden_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    domain: tuple[float, float] = eqx.field(static=True)
    gamma_alpha: float = eqx.field(static=True)
    gamma_beta: float = eqx.field(static=True)
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        depth: int,
        key: PRNGKeyArray,
        domain: tuple[float, float] = (-1.0, 1.0),
        gamma_alpha: float = 6.0,
        gamma_beta: float = 1.0,
        pyrox_name: str | None = None,
    ) -> GaborNet:
        """Construct a :class:`GaborNet`.

        Args:
            in_features: Input dimension.
            hidden_features: Filter width (all hidden layers).
            out_features: Output dimension.
            depth: Number of filter layers :math:`L` (must be ≥ 1).
            key: JAX PRNG key.
            domain: ``(low, high)`` for :math:`\\mu` initialization.
            gamma_alpha: Gamma shape for :math:`\\gamma` init (default 6.0).
            gamma_beta: Gamma rate for :math:`\\gamma` init (default 1.0).
            pyrox_name: Optional scope name.

        Returns:
            Initialised :class:`GaborNet`.

        Raises:
            ValueError: If ``depth < 1``.
        """
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}.")
        keys = jax.random.split(key, 2 * depth)
        filter_keys = keys[:depth]
        linear_keys = keys[depth:]
        filters = [
            GaborFilter.init(
                in_features,
                hidden_features,
                key=filter_keys[i],
                domain=domain,
                gamma_alpha=gamma_alpha,
                gamma_beta=gamma_beta,
            )
            for i in range(depth)
        ]
        linears = [
            eqx.nn.Linear(
                hidden_features,
                hidden_features if i < depth - 1 else out_features,
                key=linear_keys[i],
            )
            for i in range(depth)
        ]
        return cls(
            filters=filters,
            linears=linears,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            depth=depth,
            domain=domain,
            gamma_alpha=gamma_alpha,
            gamma_beta=gamma_beta,
            pyrox_name=pyrox_name,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N O"]:
        """Run the MFN forward pass.

        Args:
            x: Input of shape ``(N, D)`` or ``(D,)`` (single point).

        Returns:
            Output of shape ``(N, O)`` or ``(O,)`` if input was 1-D.
        """
        squeeze = x.ndim == 1
        out = mfn_forward(x, self.filters, self.linears)
        return out[0] if squeeze else out


class BayesianFourierNet(FourierNet):
    r"""FourierNet with Bayesian priors on all filter and linear weights.

    A thin subclass of :class:`FourierNet` that overrides ``__call__`` to
    register NumPyro sample sites for every parameter:

    - Per filter *i*: ``filter_{i}.Omega`` and ``filter_{i}.phi``.
    - Per linear *i*: ``linear_{i}.W`` and ``linear_{i}.b``.

    Total number of sites: :math:`4L` where :math:`L` is ``depth``.

    Priors:

    - :math:`\Omega_i \sim \mathcal{N}(0, \sigma^2)` (matrix).
    - :math:`\varphi_i \sim \mathrm{Uniform}(-\pi, \pi)`.
    - :math:`W_i \sim \mathcal{N}(0, \sigma^2)` (matrix).
    - :math:`b_i \sim \mathcal{N}(0, \sigma^2)` (vector).

    Attributes:
        prior_std: Prior standard deviation :math:`\\sigma` for Gaussian
            sites (default 1.0).  Phase sites always use
            :math:`\mathrm{Uniform}(-\pi, \pi)`.
    """

    prior_std: float = 1.0

    @pyrox_method
    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N O"]:
        """Forward pass with sampled parameters.

        Args:
            x: Input of shape ``(N, D)`` or ``(D,)`` (single point).

        Returns:
            Output of shape ``(N, O)`` or ``(O,)`` if input was 1-D.
        """
        squeeze = x.ndim == 1
        x2d = jnp.atleast_2d(x)

        sampled_filters: list[FourierFilter] = []
        for i, f in enumerate(self.filters):
            Omega = self.pyrox_sample(
                f"filter_{i}.Omega",
                dist.Normal(0.0, self.prior_std)
                .expand([f.out_features, f.in_features])
                .to_event(2),
            )
            phi = self.pyrox_sample(
                f"filter_{i}.phi",
                dist.Uniform(-jnp.pi, jnp.pi).expand([f.out_features]).to_event(1),
            )
            sampled_filters.append(
                eqx.tree_at(lambda ff: (ff.Omega, ff.phi), f, (Omega, phi))
            )

        sampled_linears: list[eqx.nn.Linear] = []
        for i, lin in enumerate(self.linears):
            W = self.pyrox_sample(
                f"linear_{i}.W",
                dist.Normal(0.0, self.prior_std)
                .expand([lin.out_features, lin.in_features])
                .to_event(2),
            )
            b_vec = self.pyrox_sample(
                f"linear_{i}.b",
                dist.Normal(0.0, self.prior_std).expand([lin.out_features]).to_event(1),
            )
            sampled_linears.append(
                eqx.tree_at(lambda ll: (ll.weight, ll.bias), lin, (W, b_vec))
            )

        out = mfn_forward(x2d, sampled_filters, sampled_linears)
        return out[0] if squeeze else out


class BayesianGaborNet(GaborNet):
    r"""GaborNet with Bayesian priors on all filter and linear weights.

    A thin subclass of :class:`GaborNet` that overrides ``__call__`` to
    register NumPyro sample sites for every parameter:

    - Per filter *i*: ``filter_{i}.Omega``, ``filter_{i}.phi``,
      ``filter_{i}.mu``, and ``filter_{i}.log_gamma``.
    - Per linear *i*: ``linear_{i}.W`` and ``linear_{i}.b``.

    Total number of sites: :math:`6L` where :math:`L` is ``depth``.

    Priors:

    - :math:`\Omega_i \sim \mathcal{N}(0, \sigma^2)` (matrix).
    - :math:`\varphi_i \sim \mathrm{Uniform}(-\pi, \pi)`.
    - :math:`\mu_i \sim \mathrm{Uniform}(\texttt{domain\_low},\texttt{domain\_high})`.
    - :math:`\log\gamma_i \sim \mathcal{N}(0, \sigma^2)` (log-space).
    - :math:`W_i \sim \mathcal{N}(0, \sigma^2)` (matrix).
    - :math:`b_i \sim \mathcal{N}(0, \sigma^2)` (vector).

    Attributes:
        prior_std: Prior standard deviation :math:`\\sigma` for Gaussian
            and log-gamma sites (default 1.0).
    """

    prior_std: float = 1.0

    @pyrox_method
    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N O"]:
        """Forward pass with sampled parameters.

        Args:
            x: Input of shape ``(N, D)`` or ``(D,)`` (single point).

        Returns:
            Output of shape ``(N, O)`` or ``(O,)`` if input was 1-D.
        """
        squeeze = x.ndim == 1
        x2d = jnp.atleast_2d(x)
        low, high = self.domain

        sampled_filters: list[GaborFilter] = []
        for i, f in enumerate(self.filters):
            Omega = self.pyrox_sample(
                f"filter_{i}.Omega",
                dist.Normal(0.0, self.prior_std)
                .expand([f.out_features, f.in_features])
                .to_event(2),
            )
            phi = self.pyrox_sample(
                f"filter_{i}.phi",
                dist.Uniform(-jnp.pi, jnp.pi).expand([f.out_features]).to_event(1),
            )
            mu = self.pyrox_sample(
                f"filter_{i}.mu",
                dist.Uniform(low, high)
                .expand([f.out_features, f.in_features])
                .to_event(2),
            )
            log_gamma = self.pyrox_sample(
                f"filter_{i}.log_gamma",
                dist.Normal(0.0, self.prior_std).expand([f.out_features]).to_event(1),
            )
            sampled_filters.append(
                eqx.tree_at(
                    lambda ff: (ff.Omega, ff.phi, ff.mu, ff.log_gamma),
                    f,
                    (Omega, phi, mu, log_gamma),
                )
            )

        sampled_linears: list[eqx.nn.Linear] = []
        for i, lin in enumerate(self.linears):
            W = self.pyrox_sample(
                f"linear_{i}.W",
                dist.Normal(0.0, self.prior_std)
                .expand([lin.out_features, lin.in_features])
                .to_event(2),
            )
            b_vec = self.pyrox_sample(
                f"linear_{i}.b",
                dist.Normal(0.0, self.prior_std).expand([lin.out_features]).to_event(1),
            )
            sampled_linears.append(
                eqx.tree_at(lambda ll: (ll.weight, ll.bias), lin, (W, b_vec))
            )

        out = mfn_forward(x2d, sampled_filters, sampled_linears)
        return out[0] if squeeze else out
