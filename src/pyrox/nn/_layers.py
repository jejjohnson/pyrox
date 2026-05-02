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
* :class:`DenseVariationalDropout` — sparse variational dropout with
  per-weight learnable dropout rates (Molchanov et al., 2017).
* :class:`MCDropout` — always-on dropout for Monte Carlo uncertainty at
  inference time (Gal & Ghahramani, 2016).
* :class:`DenseNCP` — Noise Contrastive Prior layer that decomposes a
  dense layer into a deterministic backbone plus a scaled stochastic
  perturbation (Hafner et al., 2019).
* :class:`NCPNormalOutput` — output-side Noise Contrastive Prior
  regulariser; KL between the noisy-batch predictive distribution and
  a fixed Gaussian prior (Hafner et al., 2018).
* :class:`SirenDense` — single sine-activated dense layer with
  Sitzmann-regime init (Sitzmann et al., NeurIPS 2020).
* :class:`SIREN` — multi-layer sinusoidal representation network.
* :class:`BayesianSIREN` — SIREN with regime-scaled Normal priors.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array as JaxArray
from jaxtyping import Array, Float, Num

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

    def __call__(self, lonlat: Num[Array, "N 2"]) -> Float[Array, "N 2"]:
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


class NCPNormalOutput(PyroxModule):
    r"""Output-side Noise Contrastive Prior layer (Hafner et al., 2018).

    Completes the NCP pattern in ``pyrox.nn``: pair with
    :class:`NCPContinuousPerturb` at the input and a heteroscedastic
    network (e.g. an MLP terminating in a mean head and a positive-std
    head — a softplus or ``exp`` of a learned log-scale) so the
    network produces predictions for both the *clean* batch and the
    input-perturbed *noisy* batch. Given the noisy batch's predictive
    distribution :math:`\mathcal{N}(\hat{y}_n, \hat{\sigma}_n^2)`,
    this layer adds the analytic NCP regulariser

    .. math::

        \mathcal{L}_\mathrm{NCP} =
        \sum_{n} \mathrm{KL}\!\bigl[\mathcal{N}(\hat{y}_n, \hat{\sigma}_n^2)
            \;\big\|\; \mathcal{N}(\mu_\mathrm{prior}, \sigma_\mathrm{prior}^2)\bigr]

    to the model log density via :func:`numpyro.factor`. Pulling the
    noisy-input predictive distribution toward the fixed prior away
    from the training distribution gives the network calibrated
    out-of-distribution uncertainty, which is the central claim of NCP.

    The closed-form Gaussian KL used here is

    .. math::

        \mathrm{KL}\bigl[\mathcal{N}(\mu, \sigma^2)
            \,\|\, \mathcal{N}(\mu_p, \sigma_p^2)\bigr]
        = \log\frac{\sigma_p}{\sigma} +
          \frac{\sigma^2 + (\mu - \mu_p)^2}{2\sigma_p^2} - \tfrac{1}{2}.

    Plate semantics:
        Unlike pyrox's *weight-prior* KL terms, the NCP KL is
        **data-dependent** — every input row contributes its own
        :math:`\mathrm{KL}_n` term. Internally the layer emits the
        :func:`numpyro.factor` site as a *per-example* vector
        (shape ``(*batch,)``) rather than a pre-summed scalar; that
        lets NumPyro's plate machinery sum over the batch axis and
        apply the subsample scaling automatically.

        The canonical training pattern is to emit the layer **inside**
        ``numpyro.plate("data", N, subsample_size=B)``::

            def model(x_clean, y_clean, x_noisy):
                clean_mean, _clean_std = network(x_clean)
                noisy_mean, noisy_std = network(x_noisy)
                ncp_out = NCPNormalOutput(prior_std=1.0)
                with numpyro.plate("data", N, subsample_size=B):
                    ncp_out(noisy_mean, noisy_std)              # scaled to N
                    numpyro.sample("obs",
                        dist.Normal(clean_mean, ...), obs=y_clean)

        Inside the plate, NumPyro sums the per-example log-densities
        over the batch dim and multiplies by ``scale = N / B``,
        producing the standard unbiased estimate of the full-dataset
        NCP KL ``Σ_{n=1}^N KL_n``. Outside any plate the layer's
        contribution is just ``Σ_{n in batch} KL_n`` (i.e. the raw
        batch sum), which is the correct full-dataset value when
        you train on the whole dataset at once.

    Attributes:
        prior_mean: Prior predictive mean :math:`\mu_\mathrm{prior}`.
        prior_std: Prior predictive std :math:`\sigma_\mathrm{prior}`
            (must be positive).
        pyrox_name: Explicit scope name for NumPyro site registration.

    Example:
        >>> import jax.numpy as jnp
        >>> from numpyro import handlers
        >>> ncp = NCPNormalOutput(
        ...     prior_mean=0.0, prior_std=1.0, pyrox_name="ncp_out"
        ... )
        >>> noisy_mean = jnp.zeros((4, 1))
        >>> noisy_std = 0.5 * jnp.ones((4, 1))
        >>> with handlers.seed(rng_seed=0):
        ...     kl = ncp(noisy_mean, noisy_std)
        >>> kl.shape
        ()

    References:
        Hafner, D., Tran, D., Lillicrap, T., Irpan, A., & Davidson, J.
        (2018). *Noise Contrastive Priors for Functional Uncertainty.*
        UAI.
    """

    prior_mean: float = eqx.field(static=True, default=0.0)
    prior_std: float = eqx.field(static=True, default=1.0)
    pyrox_name: str | None = eqx.field(static=True, default=None)

    def __post_init__(self) -> None:
        if self.prior_std <= 0:
            raise ValueError(f"prior_std must be > 0; got {self.prior_std}.")

    @pyrox_method
    def __call__(
        self,
        noisy_mean: Float[Array, "*batch D"],
        noisy_std: Float[Array, "*batch D"],
    ) -> Float[Array, ""]:
        # `noisy_std` is a *standard deviation* — the caller is responsible
        # for ensuring it is non-negative (e.g. via softplus/exp on a
        # learned log-scale head). Negative inputs would silently give a
        # finite-but-wrong KL after squaring; an explicit zero produces
        # `+inf` from `log(0)` which surfaces the bug. We do not floor
        # `noisy_var` because doing so asymmetrically (without a matching
        # floor on `prior_var`) would break the `noisy_std == prior_std`
        # → KL = 0 invariant for tiny `prior_std`.
        prior_var = jnp.asarray(self.prior_std) ** 2
        noisy_var = noisy_std**2
        kl_per_elem = (
            jnp.log(self.prior_std)
            - 0.5 * jnp.log(noisy_var)
            + (noisy_var + (noisy_mean - self.prior_mean) ** 2) / (2.0 * prior_var)
            - 0.5
        )
        # Sum only over the trailing feature axis. Keeping the leading
        # batch axis intact is what makes NumPyro's plate machinery do
        # the right thing under `plate("data", N, subsample_size=B)`:
        # the plate handler sums log_probs over the batch dim and then
        # multiplies by `N/B`, giving the unbiased full-dataset estimate
        # `(N/B) * sum_{n in batch} kl_n`. Emitting an already-summed
        # scalar instead would let NumPyro broadcast it across the
        # plate dim and over-count by a factor of B.
        kl_per_example = jnp.sum(kl_per_elem, axis=-1)
        # Add -kl_per_example to the model log density site-by-site.
        # Outside any plate this sums to -total_KL; inside a plate the
        # plate handler scales it correctly.
        numpyro.factor(self._pyrox_fullname("kl"), -kl_per_example)
        return jnp.sum(kl_per_example)


# --- DenseVariationalDropout ------------------------------------------------

# Constants for the Molchanov et al. (2017) KL approximation between
# q(W | theta, alpha) = N(theta, alpha * theta^2) and the improper
# log-uniform prior on log|W|. log_alpha is clamped to a finite interval
# to keep the approximation numerically well-behaved at extreme values.
_VD_K1 = 0.63576
_VD_K2 = 1.87320
_VD_K3 = 1.48695
_VD_LOG_ALPHA_MIN = -10.0
_VD_LOG_ALPHA_MAX = 10.0


def _vd_neg_kl(log_alpha: Float[Array, ...]) -> Float[Array, ...]:
    log_alpha = jnp.clip(log_alpha, _VD_LOG_ALPHA_MIN, _VD_LOG_ALPHA_MAX)
    alpha = jnp.exp(log_alpha)
    return (
        _VD_K1 * jax.nn.sigmoid(_VD_K2 + _VD_K3 * log_alpha)
        - 0.5 * jnp.log1p(1.0 / alpha)
        - _VD_K1
    )


class DenseVariationalDropout(PyroxModule):
    r"""Sparse variational dropout dense layer.

    Implements variational dropout (Kingma et al., 2015) extended by
    Molchanov et al. (2017) to a log-uniform prior that enables
    automatic sparsification via per-weight learnable dropout rates.
    The variational posterior on weights is

    .. math::

        q(W_{ij} \mid \theta_{ij}, \alpha_{ij}) =
        \mathcal{N}\!\bigl(\theta_{ij},\; \alpha_{ij}\,\theta_{ij}^2\bigr).

    Forward passes use the *local reparameterization trick* — the
    pre-activation distribution is closed-form and the noise is sampled
    once per output unit per batch element rather than once per weight:

    .. math::

        \gamma = X\theta, \quad
        \delta = X^{\circ 2}\,(\alpha \circ \theta^{\circ 2}), \quad
        Y = \gamma + \sqrt{\delta} \circ \epsilon, \quad
        \epsilon \sim \mathcal{N}(0, I).

    The KL between the posterior and the log-uniform prior is
    approximated analytically (Molchanov et al., 2017) and added to the
    NumPyro trace via :func:`numpyro.factor`. SVI then optimizes

    .. math::

        \mathcal{L} = \mathbb{E}_q[\log p(y \mid f)] - \mathrm{KL}\bigl[q\,\|\,p\bigr].

    Weights with ``log_alpha > threshold`` (default 3.0, dropout rate
    ~0.95) are effectively pruned; inspect the trained pattern via
    :meth:`sparsity`.

    Plate semantics:
        The KL contribution is registered via :func:`numpyro.factor`,
        which is itself a sample-type site and therefore subject to
        ``numpyro.plate`` scaling. To keep the per-layer KL counted
        once (not scaled by the data-plate's subsample ratio), call
        the layer **outside** any ``plate("data", ..., subsample_size=...)``
        block — the standard pyrox / NumPyro convention for global
        Bayesian parameters. Plate only the observation likelihood.

        Correct (forward outside the data plate)::

            def model(x, y=None):
                layer = DenseVariationalDropout(in_features=D, out_features=1)
                f = layer(x).squeeze(-1)               # KL emitted here
                with numpyro.plate("data", x.shape[0]):
                    numpyro.sample("obs", dist.Normal(f, 0.5), obs=y)

        Incorrect (forward inside a subsampled data plate scales KL by
        ``N / subsample_size``)::

            def model(x, y=None):
                with numpyro.plate("data", N, subsample_size=B) as idx:
                    f = layer(x[idx]).squeeze(-1)      # ⚠ scales KL
                    numpyro.sample("obs", ...)

    Attributes:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include a bias term.
        log_alpha_init: Initial value for ``log_alpha`` (typically a
            small negative number, e.g., ``-5.0``).
        threshold: ``log_alpha`` threshold for declaring a weight pruned.
        pyrox_name: Explicit scope name for NumPyro site registration.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from numpyro import handlers
        >>> layer = DenseVariationalDropout(
        ...     in_features=4, out_features=2, pyrox_name="vd"
        ... )
        >>> x = jnp.ones((3, 4))
        >>> with handlers.seed(rng_seed=0):
        ...     y = layer(x)
        >>> y.shape
        (3, 2)
    """

    in_features: int
    out_features: int
    bias: bool = True
    log_alpha_init: float = -5.0
    threshold: float = 3.0
    pyrox_name: str | None = None

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        theta = self.pyrox_param(
            "theta",
            jnp.zeros((self.in_features, self.out_features)),
        )
        log_alpha = self.pyrox_param(
            "log_alpha",
            jnp.full(
                (self.in_features, self.out_features),
                float(self.log_alpha_init),
            ),
        )
        log_alpha_clamped = jnp.clip(log_alpha, _VD_LOG_ALPHA_MIN, _VD_LOG_ALPHA_MAX)
        alpha = jnp.exp(log_alpha_clamped)

        gamma = x @ theta
        delta = (x**2) @ (alpha * theta**2)
        # Floor to a tiny positive value: keeps the sqrt gradient finite at
        # delta = 0 without injecting visible noise (sqrt(1e-30) ≈ 1e-15).
        std = jnp.sqrt(jnp.maximum(delta, 1e-30))
        # numpyro.prng_key returns Array | None at the type level, but is
        # always an Array inside a `seed` handler — which is required for
        # the pyrox_param/factor calls above to succeed in any case.
        key = cast(JaxArray, numpyro.prng_key())
        eps = jax.random.normal(key, gamma.shape, dtype=gamma.dtype)
        out = gamma + std * eps

        if self.bias:
            b = self.pyrox_param("bias", jnp.zeros(self.out_features))
            out = out + b

        numpyro.factor(
            self._pyrox_fullname("kl"),
            jnp.sum(_vd_neg_kl(log_alpha)),
        )
        return out

    def sparsity(self, log_alpha: Float[Array, "D_in D_out"]) -> Float[Array, ""]:
        """Fraction of weights with ``log_alpha > threshold``.

        Pass the trained ``log_alpha`` parameter, typically retrieved
        from the SVI param store under ``f"{pyrox_name}.log_alpha"``.
        """
        return jnp.mean((log_alpha > self.threshold).astype(log_alpha.dtype))


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


def _rff_cosine_forward(
    W: Float[Array, "D_in n_features"],
    b: Float[Array, " n_features"],
    lengthscale: float | Float[Array, ""],
    n_features: int,
    x: Float[Array, "*batch D_in"],
) -> Float[Array, "*batch n_features"]:
    """Shared single-cosine RFF feature map: ``sqrt(2/D) cos(xW/l + b)``."""
    return jnp.sqrt(2.0 / n_features) * jnp.cos(x @ W / lengthscale + b)


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
        return _rff_cosine_forward(W, b, ls, self.n_features, x)


class MaternCosineFeatures(PyroxModule):
    r"""Cosine-bias variant of random Fourier features for the Matern kernel.

    Single-cosine analogue of :class:`MaternFourierFeatures`:

    .. math::

        \phi(x) = \sqrt{2 / D}\,\cos(x W / \ell + b)

    where :math:`W \sim \mathrm{StudentT}(2\nu)` (the Matern spectral
    density) and :math:`b \sim \mathrm{Uniform}(0, 2\pi)`. Output dim is
    ``n_features`` (vs ``2 * n_features`` for the ``[cos, sin]``
    variant). Approximates the same kernel as
    :class:`MaternFourierFeatures` in expectation but with higher
    variance per draw — see Sutherland & Schneider (2015).

    All parameters (:math:`W`, :math:`b`, :math:`\ell`) are
    ``pyrox_sample`` sites.

    Attributes:
        in_features: Input dimension.
        n_features: Number of random features (= output dimension).
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
    ) -> MaternCosineFeatures:
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
        b = self.pyrox_sample(
            "b",
            dist.Uniform(0.0, 2.0 * jnp.pi).expand([self.n_features]).to_event(1),
        )
        ls = self.pyrox_sample(
            "lengthscale",
            dist.LogNormal(jnp.log(jnp.asarray(self.init_lengthscale)), 1.0),
        )
        return _rff_cosine_forward(W, b, ls, self.n_features, x)


class LaplaceCosineFeatures(PyroxModule):
    r"""Cosine-bias variant of random Fourier features for the Laplace kernel.

    Single-cosine analogue of :class:`LaplaceFourierFeatures` (the
    Matern-1/2 kernel):

    .. math::

        \phi(x) = \sqrt{2 / D}\,\cos(x W / \ell + b)

    where :math:`W \sim \mathrm{Cauchy}(0, 1)` (Student-t with
    ``df = 1``) and :math:`b \sim \mathrm{Uniform}(0, 2\pi)`. Output
    dim is ``n_features``.

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
    ) -> LaplaceCosineFeatures:
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
            dist.StudentT(df=1.0, loc=0.0, scale=1.0)
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
        return _rff_cosine_forward(W, b, ls, self.n_features, x)


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
# SIREN — Sinusoidal Representation Networks (Sitzmann et al., NeurIPS 2020)
# ---------------------------------------------------------------------------


_SirenLayerType = Literal["first", "hidden", "last"]


class _SirenLayerSpec(NamedTuple):
    """Static per-layer metadata shared by deterministic and Bayesian SIRENs."""

    layer_type: _SirenLayerType
    in_features: int
    out_features: int
    omega: float
    c: float


def _siren_W_limit(
    layer_type: _SirenLayerType,
    in_features: int,
    omega: float,
    c: float = 6.0,
) -> float:
    """Return the half-width ``a`` of the ``U(-a, a)`` weight init for a SIREN layer.

    Implements Sitzmann et al. (2020) Theorem 1's three-regime prescription:

    - ``"first"``:  ``a = 1 / in_features``
    - ``"hidden"``: ``a = sqrt(c / in_features) / omega``
    - ``"last"``:   ``a = sqrt(c / in_features)``

    Args:
        layer_type: One of ``"first"``, ``"hidden"``, ``"last"``.
        in_features: Input dimension of the layer.
        omega: Frequency multiplier (only affects hidden-layer limit).
        c: Constant from Theorem 1; defaults to 6.0.

    Returns:
        Half-width ``a`` such that ``W ~ U(-a, a)``.

    Raises:
        ValueError: If ``layer_type`` is not one of the three valid values.
    """
    if layer_type == "first":
        return 1.0 / in_features
    if layer_type == "hidden":
        return math.sqrt(c / in_features) / omega
    if layer_type == "last":
        return math.sqrt(c / in_features)
    raise ValueError(
        f"layer_type must be 'first', 'hidden', or 'last', got {layer_type!r}"
    )


def _require_positive(**values: float) -> None:
    """Raise ``ValueError`` if any keyword value is non-positive."""
    for name, v in values.items():
        if v <= 0:
            raise ValueError(f"{name} must be > 0, got {v}.")


def _build_siren_specs(
    in_features: int,
    hidden_features: int,
    out_features: int,
    depth: int,
    first_omega: float,
    hidden_omega: float,
    c: float,
) -> tuple[_SirenLayerSpec, ...]:
    """Produce per-layer specs for a depth-``depth`` SIREN, first + hidden… + last."""
    specs: list[_SirenLayerSpec] = []
    for i in range(depth):
        if i == 0:
            specs.append(
                _SirenLayerSpec("first", in_features, hidden_features, first_omega, c)
            )
        elif i == depth - 1:
            specs.append(
                _SirenLayerSpec("last", hidden_features, out_features, hidden_omega, c)
            )
        else:
            specs.append(
                _SirenLayerSpec(
                    "hidden", hidden_features, hidden_features, hidden_omega, c
                )
            )
    return tuple(specs)


class SirenDense(eqx.Module):
    r"""Sine-activated dense layer: ``y = sin(ω · (W x + b))`` or ``y = W x + b``.

    Single primitive of a SIREN network.  Three init regimes
    (Sitzmann et al. 2020, Theorem 1):

    +----------+-------------------------------------------+-------------+
    | Regime   | ``W`` init                                | Activation  |
    +==========+===========================================+=============+
    | first    | ``U(-1/d_in, 1/d_in)``                    | ``sin(ω··)``|
    +----------+-------------------------------------------+-------------+
    | hidden   | ``U(-√(c/d_in)/ω, √(c/d_in)/ω)``         | ``sin(ω··)``|
    +----------+-------------------------------------------+-------------+
    | last     | ``U(-√(c/d_in), √(c/d_in))``              | none        |
    +----------+-------------------------------------------+-------------+

    Bias ``b`` is initialised ``U(-1/√d_in, 1/√d_in)`` for every regime.

    Attributes:
        W: Weight matrix of shape ``(in_features, out_features)``.
        b: Bias vector of shape ``(out_features,)``.
        omega: Frequency multiplier applied inside the sine.
        in_features: Input dimension.
        out_features: Output dimension.
        layer_type: One of ``"first"``, ``"hidden"``, ``"last"``.
        c: Constant from Theorem 1 (default 6.0).

    Example:
        >>> import jax.random as jr
        >>> layer = SirenDense.init(3, 16, key=jr.PRNGKey(0), layer_type="first")
        >>> import jax.numpy as jnp
        >>> y = layer(jnp.ones((5, 3)))
        >>> y.shape
        (5, 16)
    """

    W: Float[Array, "in_features out_features"]
    b: Float[Array, " out_features"]
    omega: float = eqx.field(static=True)
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    layer_type: _SirenLayerType = eqx.field(static=True)
    c: float = eqx.field(static=True, default=6.0)

    @classmethod
    def init(
        cls,
        in_features: int,
        out_features: int,
        *,
        key: Array,
        omega: float = 30.0,
        layer_type: _SirenLayerType = "hidden",
        c: float = 6.0,
    ) -> SirenDense:
        """Construct a ``SirenDense`` with Sitzmann-regime weight initialisation.

        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            key: JAX PRNG key for weight and bias sampling.
            omega: Frequency multiplier (default 30.0, as in Sitzmann et al.).
            layer_type: Init regime — ``"first"``, ``"hidden"``, or ``"last"``.
            c: Theorem-1 constant (default 6.0).

        Returns:
            Initialised :class:`SirenDense`.

        Raises:
            ValueError: If ``layer_type`` is not a valid regime name, or if any of
                ``in_features``, ``out_features``, ``omega``, or ``c`` is
                non-positive.
        """
        _require_positive(
            in_features=in_features,
            out_features=out_features,
            omega=omega,
            c=c,
        )
        k_w, k_b = jax.random.split(key)
        # _siren_W_limit validates layer_type.
        w_limit = _siren_W_limit(layer_type, in_features, omega, c)
        W = jax.random.uniform(
            k_w, (in_features, out_features), minval=-w_limit, maxval=w_limit
        )
        b_limit = 1.0 / math.sqrt(in_features)
        b = jax.random.uniform(k_b, (out_features,), minval=-b_limit, maxval=b_limit)
        return cls(
            W=W,
            b=b,
            omega=omega,
            in_features=in_features,
            out_features=out_features,
            layer_type=layer_type,
            c=c,
        )

    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        """Apply the sine-activated linear transform.

        Args:
            x: Input tensor of shape ``(*batch, in_features)``.

        Returns:
            ``sin(ω · (x W + b))`` for ``layer_type`` in ``{"first", "hidden"}``,
            or ``x W + b`` for ``layer_type == "last"``.
        """
        pre = x @ self.W + self.b
        if self.layer_type == "last":
            return pre
        return jnp.sin(self.omega * pre)


class SIREN(eqx.Module):
    r"""Multi-layer sinusoidal representation network (Sitzmann et al., NeurIPS 2020).

    Topology:

    .. math::

        z_1 &= \sin(\omega_0 (W_0 x + b_0)), \\
        z_{i+1} &= \sin(\omega (W_i z_i + b_i)), \quad i = 1 \ldots L-1, \\
        y &= W_L z_L + b_L.

    Each layer uses the corresponding Sitzmann Theorem 1 init regime
    (:class:`SirenDense`):  ``"first"`` for layer 0, ``"hidden"`` for
    intermediate layers, and ``"last"`` for the readout.

    ``depth`` counts *all* layers including the readout; ``depth=2`` gives
    one first-layer + one last-layer (no hidden layers); ``depth=5`` gives
    first + 3 hidden + last.  Must be ≥ 2.

    Attributes:
        layers: List of :class:`SirenDense` of length ``depth``.
        in_features: Input dimension.
        hidden_features: Hidden dimension (all intermediate layers).
        out_features: Output dimension.
        depth: Total number of layers (including readout).  Must be ≥ 2.
        first_omega: Frequency multiplier for the first layer (default 30.0).
        hidden_omega: Frequency multiplier for hidden layers (default 30.0).

    Example:
        >>> import jax.random as jr, jax.numpy as jnp
        >>> net = SIREN.init(2, 64, 1, depth=5, key=jr.PRNGKey(0))
        >>> net(jnp.zeros((10, 2))).shape
        (10, 1)
    """

    layers: list[SirenDense]
    in_features: int = eqx.field(static=True)
    hidden_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    first_omega: float = eqx.field(static=True)
    hidden_omega: float = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        depth: int,
        key: Array,
        first_omega: float = 30.0,
        hidden_omega: float = 30.0,
        c: float = 6.0,
    ) -> SIREN:
        """Construct a SIREN with the correct per-layer init regimes.

        Args:
            in_features: Input dimension.
            hidden_features: Hidden dimension for all intermediate layers.
            out_features: Output dimension.
            depth: Total layers including readout.  Must be ≥ 2.
            key: JAX PRNG key.
            first_omega: Frequency for the first layer (default 30.0).
            hidden_omega: Frequency for hidden layers (default 30.0).
            c: Theorem-1 constant passed to each :class:`SirenDense`.

        Returns:
            Initialised :class:`SIREN`.

        Raises:
            ValueError: If ``depth < 2`` or any of the feature dimensions,
                omegas, or ``c`` is non-positive.
        """
        if depth < 2:
            raise ValueError(f"depth must be >= 2 (first + last); got depth={depth}")
        _require_positive(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            first_omega=first_omega,
            hidden_omega=hidden_omega,
            c=c,
        )
        specs = _build_siren_specs(
            in_features,
            hidden_features,
            out_features,
            depth,
            first_omega,
            hidden_omega,
            c,
        )
        keys = jax.random.split(key, depth)
        layers = [
            SirenDense.init(
                spec.in_features,
                spec.out_features,
                key=k,
                omega=spec.omega,
                layer_type=spec.layer_type,
                c=spec.c,
            )
            for spec, k in zip(specs, keys, strict=True)
        ]
        return cls(
            layers=layers,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            depth=depth,
            first_omega=first_omega,
            hidden_omega=hidden_omega,
        )

    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        """Run the forward pass through all SIREN layers.

        Args:
            x: Input tensor of shape ``(*batch, in_features)``.

        Returns:
            Output tensor of shape ``(*batch, out_features)``.
        """
        z = x
        for layer in self.layers:
            z = layer(z)
        return z


class BayesianSIREN(PyroxModule):
    r"""SIREN with regime-scaled Normal priors on all layer weights.

    Replaces the deterministic weight matrices of :class:`SIREN` with NumPyro
    sample sites.  For layer :math:`i` with Sitzmann Theorem 1 half-width
    :math:`a_i` (the uniform bound used by :class:`SirenDense`):

    .. math::

        W_i \sim \mathcal{N}\!\left(0,\, \sigma_0 \cdot \frac{a_i}{\sqrt{3}}\right),
        \qquad
        b_i \sim \mathcal{N}\!\left(0,\,
            \sigma_0 \cdot \frac{1}{\sqrt{3 \, d_i}}\right),

    where :math:`\sigma_0` is ``prior_std`` and :math:`d_i` is the input
    dimension of layer :math:`i`.  The :math:`a_i / \sqrt{3}` factor makes
    :math:`\operatorname{Var}(W_i)` equal to the variance of Sitzmann's
    :math:`\mathcal{U}(-a_i, a_i)` init exactly, so the Bayesian prior
    preserves the activation variance prescribed by Theorem 1 — avoiding
    the saturated-sine pathology that a flat :math:`\mathcal{N}(0, 1)`
    prior would cause.

    Registered sites: ``{scope}.layer_0.W``, ``{scope}.layer_0.b``, …,
    ``{scope}.layer_{depth-1}.W``, ``{scope}.layer_{depth-1}.b``
    — exactly ``2 · depth`` sites per forward call.

    Attributes:
        specs: Tuple of per-layer specs (static).  Holds each layer's
            ``layer_type``, ``in_features``, ``out_features``, ``omega``,
            and ``c`` — i.e. everything needed to scale the priors.
        in_features: Input dimension.
        hidden_features: Hidden dimension.
        out_features: Output dimension.
        depth: Total layers including readout.  Must be ≥ 2.
        first_omega: Frequency multiplier for the first layer.
        hidden_omega: Frequency multiplier for hidden layers.
        prior_std: Scale factor for the regime-scaled Normal prior (default 1.0).
        pyrox_name: Explicit scope name for NumPyro site registration.

    Example:
        >>> import jax.random as jr, jax.numpy as jnp
        >>> from numpyro import handlers
        >>> net = BayesianSIREN.init(2, 32, 1, depth=3)
        >>> with handlers.seed(rng_seed=0):
        ...     y = net(jnp.zeros((4, 2)))
        >>> y.shape
        (4, 1)
    """

    specs: tuple[_SirenLayerSpec, ...] = eqx.field(static=True)
    in_features: int = eqx.field(static=True)
    hidden_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    first_omega: float = eqx.field(static=True)
    hidden_omega: float = eqx.field(static=True)
    prior_std: float = eqx.field(static=True, default=1.0)
    pyrox_name: str | None = eqx.field(static=True, default=None)

    @classmethod
    def init(
        cls,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        depth: int,
        first_omega: float = 30.0,
        hidden_omega: float = 30.0,
        c: float = 6.0,
        prior_std: float = 1.0,
        pyrox_name: str | None = None,
    ) -> BayesianSIREN:
        """Construct a :class:`BayesianSIREN`.

        All weights come from the prior, so no PRNG key is needed at
        construction time — the key enters when sampling inside a
        ``numpyro`` handler (``handlers.seed``, SVI, etc.).

        Args:
            in_features: Input dimension.
            hidden_features: Hidden dimension.
            out_features: Output dimension.
            depth: Total layers including readout.  Must be ≥ 2.
            first_omega: Frequency for the first layer.
            hidden_omega: Frequency for hidden layers.
            c: Theorem-1 constant.
            prior_std: Scale factor for the Normal priors (default 1.0, must be > 0).
            pyrox_name: Optional explicit scope name for NumPyro.

        Returns:
            Initialised :class:`BayesianSIREN`.

        Raises:
            ValueError: If ``depth < 2``, or any of the feature dimensions,
                omegas, ``c``, or ``prior_std`` is non-positive.
        """
        if depth < 2:
            raise ValueError(f"depth must be >= 2 (first + last); got depth={depth}")
        _require_positive(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            first_omega=first_omega,
            hidden_omega=hidden_omega,
            c=c,
            prior_std=prior_std,
        )
        specs = _build_siren_specs(
            in_features,
            hidden_features,
            out_features,
            depth,
            first_omega,
            hidden_omega,
            c,
        )
        return cls(
            specs=specs,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            depth=depth,
            first_omega=first_omega,
            hidden_omega=hidden_omega,
            prior_std=prior_std,
            pyrox_name=pyrox_name,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        """Sample weights from regime-scaled priors and run the forward pass.

        Registers ``layer_{i}.W`` and ``layer_{i}.b`` NumPyro sample sites
        for each layer ``i`` in ``[0, depth)``.

        Args:
            x: Input tensor of shape ``(*batch, in_features)``.

        Returns:
            Output tensor of shape ``(*batch, out_features)``.
        """
        # Normal stddev = (Uniform half-width) / √3 so Var(W) matches
        # Sitzmann's U(-a, a) init exactly.
        inv_sqrt3 = 1.0 / math.sqrt(3.0)
        z = x
        for i, spec in enumerate(self.specs):
            a = _siren_W_limit(spec.layer_type, spec.in_features, spec.omega, spec.c)
            w_scale = self.prior_std * a * inv_sqrt3
            b_scale = self.prior_std * inv_sqrt3 / math.sqrt(spec.in_features)
            W = self.pyrox_sample(
                f"layer_{i}.W",
                dist.Normal(0.0, w_scale)
                .expand([spec.in_features, spec.out_features])
                .to_event(2),
            )
            b = self.pyrox_sample(
                f"layer_{i}.b",
                dist.Normal(0.0, b_scale).expand([spec.out_features]).to_event(1),
            )
            pre = z @ W + b
            z = pre if spec.layer_type == "last" else jnp.sin(spec.omega * pre)
        return z


class DeepVSSGP(PyroxModule):
    r"""Deep Random Feature Expansion for Variational SSGP (Cutajar et al. 2017).

    A stack of :math:`L` variational SSGP layers, each with random
    spectral frequencies :math:`\Omega_l` and random projection weights
    :math:`W_l`:

    .. math::

        F_0 &= X, \\
        F_{l+1} &= \Phi_l(F_l;\, \Omega_l, \ell_l)\, W_l,
            \quad l = 0, \ldots, L-1, \\
        \Phi_l(F;\, \Omega_l, \ell_l) &=
            \sqrt{1/M}\,
            [\cos(F\,\Omega_l/\ell_l), \sin(F\,\Omega_l/\ell_l)].

    Each layer registers three sample sites:

    - ``layer_{l}.W_freq`` — RFF frequencies, prior :math:`\mathcal{N}(0, 1)`
      (RBF spectral density in lengthscale-1 units).
    - ``layer_{l}.lengthscale`` — kernel lengthscale, prior
      :math:`\mathrm{LogNormal}(\log \ell_{\mathrm{init}}, 1)`.
    - ``layer_{l}.W_proj`` — projection weights, prior
      :math:`\mathcal{N}(0, \sigma_W^2)`.

    Under SVI an
    :class:`~numpyro.infer.autoguide.AutoNormal` learns mean-field
    Gaussian posteriors over all :math:`3L` sites — one MC sample per
    forward pass gives the doubly-stochastic reparameterised ELBO of
    Cutajar et al. (2017).

    At ``depth=1`` this reduces to a single VSSGP layer mapping
    ``in_features -> out_features`` via the RFF basis (same model class
    as :class:`VariationalFourierFeatures` followed by a
    :class:`DenseReparameterization` head). Stacking adds
    non-stationarity at the cost of a non-Gaussian aggregate likelihood
    — the layer-wise marginalisation that makes single-layer SSGP
    closed-form is no longer available, hence the variational
    treatment.

    Attributes:
        in_features: Input dimension :math:`D_{\mathrm{in}}`.
        hidden_features: Inter-layer dimension :math:`D_h` (constant
            across hidden layers).
        out_features: Output dimension :math:`D_{\mathrm{out}}`.
        n_features: Per-layer Fourier-feature pair count :math:`M` (so
            each layer's hidden state is :math:`2M`-dim before
            projection).
        depth: Total number of stacked SSGP layers :math:`L`. Must
            be :math:`\ge 1`.
        init_lengthscale: Prior location for each layer's lengthscale.
        prior_std: Standard deviation of the per-layer projection
            prior :math:`\mathcal{N}(0, \sigma_W^2)`.
        pyrox_name: Explicit scope name for NumPyro site registration.

    Example:
        >>> import jax.random as jr, jax.numpy as jnp
        >>> from numpyro import handlers
        >>> net = DeepVSSGP.init(in_features=2, hidden_features=4,
        ...                       out_features=1, depth=3, n_features=16)
        >>> with handlers.seed(rng_seed=0):
        ...     y = net(jnp.zeros((8, 2)))
        >>> y.shape
        (8, 1)
    """

    in_features: int = eqx.field(static=True)
    hidden_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    init_lengthscale: float = 1.0
    prior_std: float = 1.0
    pyrox_name: str | None = None

    @classmethod
    def init(
        cls,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        depth: int,
        n_features: int = 64,
        lengthscale: float = 1.0,
        prior_std: float = 1.0,
        pyrox_name: str | None = None,
    ) -> DeepVSSGP:
        """Construct a :class:`DeepVSSGP`.

        Args:
            in_features: Input dimension. Must be :math:`\\ge 1`.
            hidden_features: Hidden dimension. Must be :math:`\\ge 1`.
            out_features: Output dimension. Must be :math:`\\ge 1`.
            depth: Total stacked SSGP layers (including readout).
                Must be :math:`\\ge 1`.
            n_features: Per-layer Fourier-feature pair count. Must be
                :math:`\\ge 1`.
            lengthscale: Prior location for each layer's lengthscale.
                Must be :math:`> 0`.
            prior_std: Per-layer projection prior standard deviation.
                Must be :math:`> 0`.
            pyrox_name: Optional explicit scope name for NumPyro site
                registration.

        Returns:
            Initialised :class:`DeepVSSGP`.

        Raises:
            ValueError: If ``depth``, any feature dimension, or
                ``n_features`` is :math:`< 1`, or if ``lengthscale`` /
                ``prior_std`` is :math:`\\le 0`.
        """
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}.")
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}.")
        if lengthscale <= 0:
            raise ValueError(f"lengthscale must be > 0, got {lengthscale}.")
        if prior_std <= 0:
            raise ValueError(f"prior_std must be > 0, got {prior_std}.")
        for name, dim in (
            ("in_features", in_features),
            ("hidden_features", hidden_features),
            ("out_features", out_features),
        ):
            if dim < 1:
                raise ValueError(f"{name} must be >= 1, got {dim}.")
        return cls(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            n_features=n_features,
            depth=depth,
            init_lengthscale=lengthscale,
            prior_std=prior_std,
            pyrox_name=pyrox_name,
        )

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D_in"]) -> Float[Array, "*batch D_out"]:
        z = x
        for layer_idx in range(self.depth):
            in_dim = self.in_features if layer_idx == 0 else self.hidden_features
            out_dim = (
                self.out_features
                if layer_idx == self.depth - 1
                else self.hidden_features
            )
            W_freq = self.pyrox_sample(
                f"layer_{layer_idx}.W_freq",
                dist.Normal(0.0, 1.0).expand([in_dim, self.n_features]).to_event(2),
            )
            ls = self.pyrox_sample(
                f"layer_{layer_idx}.lengthscale",
                dist.LogNormal(jnp.log(jnp.asarray(self.init_lengthscale)), 1.0),
            )
            W_proj = self.pyrox_sample(
                f"layer_{layer_idx}.W_proj",
                dist.Normal(0.0, self.prior_std)
                .expand([2 * self.n_features, out_dim])
                .to_event(2),
            )
            phi = _rff_forward(W_freq, ls, self.n_features, z)
            z = phi @ W_proj
        return z
