"""Coordinate encoders and uncertainty-aware dense layers.

Deterministic coordinate encoders (pure ``equinox.Module`` wrappers
around the helpers in :mod:`pyrox.nn._geo`):

* :class:`Deg2Rad` — element-wise degrees-to-radians conversion.
* :class:`LonLatScale` — affine lon/lat scaling.
* :class:`Cartesian3DEncoder` — lon/lat lift to unit Cartesian
  coordinates on :math:`S^2`.
* :class:`CyclicEncoder` — periodic ``(cos, sin)`` feature map.
* :class:`SphericalHarmonicEncoder` — real spherical-harmonic features
  via :func:`pyrox._basis.real_spherical_harmonics`.

Uncertainty-aware dense / random-feature layers:

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
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float

from pyrox._basis import fourier_basis, real_spherical_harmonics, spectral_density
from pyrox._core.pyrox_module import PyroxModule, pyrox_method
from pyrox.gp._context import _kernel_context
from pyrox.gp._protocols import Kernel
from pyrox.nn._geo import (
    _validate_input_unit,
    _validate_range,
    cyclic_encode,
    deg2rad,
    lonlat_scale,
    lonlat_to_cartesian3d,
)


class Deg2Rad(eqx.Module):
    """Element-wise degrees-to-radians conversion.

    Stateless ``equinox.Module`` wrapper around :func:`deg2rad` — no
    learnable parameters and no sample sites.

    Example:
        >>> import jax.numpy as jnp
        >>> Deg2Rad()(jnp.array([0.0, 90.0, 180.0]))
        Array([0.       , 1.5707964, 3.1415927], dtype=float32)
        >>> # Composes with other encoders in eqx.nn.Sequential.
        >>> import equinox as eqx
        >>> pipeline = eqx.nn.Sequential([Deg2Rad(), Cartesian3DEncoder()])
    """

    def __call__(self, x: Float[Array, ...]) -> Float[Array, ...]:
        return deg2rad(x)


class LonLatScale(eqx.Module):
    """Affine-rescale lon/lat columns.

    Values inside the given ranges map into ``[-1, 1]``; out-of-range
    values are *not* clipped. The default ranges assume ``lonlat`` is
    in degrees.

    Attributes:
        lon_range: ``(min, max)`` longitude domain (must satisfy
            ``min < max``).
        lat_range: ``(min, max)`` latitude domain (must satisfy
            ``min < max``).

    Example:
        >>> import jax.numpy as jnp
        >>> LonLatScale()(jnp.array([[0.0, 0.0]]))
        Array([[0., 0.]], dtype=float32)
        >>> # Regional grid in degrees:
        >>> LonLatScale(lon_range=(-10.0, 10.0), lat_range=(40.0, 60.0))(
        ...     jnp.array([[0.0, 50.0]])
        ... )
        Array([[0., 0.]], dtype=float32)
    """

    lon_range: tuple[float, float] = eqx.field(static=True, default=(-180.0, 180.0))
    lat_range: tuple[float, float] = eqx.field(static=True, default=(-90.0, 90.0))

    def __post_init__(self) -> None:
        _validate_range(self.lon_range, name="lon_range")
        _validate_range(self.lat_range, name="lat_range")

    def __call__(self, lonlat: Float[Array, "N 2"]) -> Float[Array, "N 2"]:
        return lonlat_scale(
            lonlat,
            lon_range=self.lon_range,
            lat_range=self.lat_range,
        )


class Cartesian3DEncoder(eqx.Module):
    """Lift lon/lat coordinates onto the unit sphere :math:`S^2`.

    Stateless wrapper around :func:`lonlat_to_cartesian3d`. Uses the
    same axis convention as
    :class:`pyrox.gp.SphericalHarmonicInducingFeatures`.

    Attributes:
        input_unit: Whether the input is in ``"degrees"`` or
            ``"radians"``.

    Example:
        >>> import jax.numpy as jnp
        >>> # Prime meridian / equator → +x
        >>> Cartesian3DEncoder()(jnp.array([[0.0, 0.0]]))
        Array([[1., 0., 0.]], dtype=float32)
        >>> # Degrees input:
        >>> Cartesian3DEncoder(input_unit="degrees")(jnp.array([[0.0, 90.0]]))[:, 2]
        Array([1.], dtype=float32)
    """

    input_unit: Literal["degrees", "radians"] = eqx.field(
        static=True, default="radians"
    )

    def __post_init__(self) -> None:
        _validate_input_unit(self.input_unit)

    def __call__(self, lonlat: Float[Array, "N 2"]) -> Float[Array, "N 3"]:
        return lonlat_to_cartesian3d(lonlat, input_unit=self.input_unit)


class CyclicEncoder(eqx.Module):
    """Encode periodic inputs as concatenated cos/sin features.

    Stateless wrapper around :func:`cyclic_encode`.

    Example:
        >>> import jax.numpy as jnp
        >>> CyclicEncoder()(jnp.array([0.0, jnp.pi]))[:, 0]
        Array([ 1., -1.], dtype=float32)
        >>> # 2-D input: each column encoded independently.
        >>> CyclicEncoder()(jnp.zeros((3, 2))).shape
        (3, 4)
    """

    def __call__(
        self,
        angles: Float[Array, " N"] | Float[Array, "N D"],
    ) -> Float[Array, "N F"]:
        return cyclic_encode(angles)


class SphericalHarmonicEncoder(eqx.Module):
    """Real spherical-harmonic features on the unit sphere.

    Stateless wrapper that evaluates
    :func:`pyrox._basis.real_spherical_harmonics` on either already-
    cartesian inputs (``input_mode='cartesian'``) or lon/lat pairs
    (``input_mode='lonlat'``, assumed in radians).

    Attributes:
        l_max: Maximum harmonic degree (must be ``>= 0``). The output
            has ``(l_max + 1) ** 2`` features.
        input_mode: ``"cartesian"`` for ``(N, 3)`` unit-sphere inputs
            or ``"lonlat"`` for ``(N, 2)`` lon/lat pairs in radians.

    Example:
        >>> import jax.numpy as jnp
        >>> xyz = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> SphericalHarmonicEncoder(l_max=3)(xyz).shape
        (2, 16)
        >>> # Lon/lat input mode (radians):
        >>> lonlat = jnp.array([[0.0, 0.0], [0.5 * jnp.pi, 0.0]])
        >>> SphericalHarmonicEncoder(l_max=3, input_mode="lonlat")(lonlat).shape
        (2, 16)
    """

    l_max: int = eqx.field(static=True)
    input_mode: Literal["cartesian", "lonlat"] = eqx.field(
        static=True, default="cartesian"
    )

    def __post_init__(self) -> None:
        if self.l_max < 0:
            raise ValueError(f"l_max must be >= 0; got {self.l_max}.")
        if self.input_mode not in {"cartesian", "lonlat"}:
            raise ValueError(
                f"input_mode must be 'cartesian' or 'lonlat'; got {self.input_mode!r}."
            )

    @property
    def num_features(self) -> int:
        return (self.l_max + 1) ** 2

    def __call__(
        self,
        x: Float[Array, "N 3"] | Float[Array, "N 2"],
    ) -> Float[Array, "N M"]:
        if self.input_mode == "cartesian":
            if x.ndim != 2 or x.shape[-1] != 3:
                raise ValueError(
                    "x must be (N, 3) when input_mode='cartesian'; "
                    f"got shape {x.shape}."
                )
            unit_xyz = x
        else:
            if x.ndim != 2 or x.shape[-1] != 2:
                raise ValueError(
                    f"x must be (N, 2) when input_mode='lonlat'; got shape {x.shape}."
                )
            unit_xyz = lonlat_to_cartesian3d(x, input_unit="radians")
        return real_spherical_harmonics(unit_xyz, l_max=self.l_max)


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
