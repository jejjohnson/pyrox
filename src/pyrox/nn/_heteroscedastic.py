"""Heteroscedastic output layers (Collier et al., 2021).

* :class:`MCSoftmaxDenseFA` — multi-class output with input-dependent
  low-rank-plus-diagonal logit noise, MC-averaged softmax probabilities.
* :class:`MCSigmoidDenseFA` — same noise model, sigmoid output for
  multi-label / binary classification.

Both share the same heteroscedastic logit-noise model — given an
input-dependent low-rank factor :math:`V(x) \\in \\mathbb{R}^{C \\times r}`
and diagonal :math:`\\sigma(x) \\in \\mathbb{R}^C`,

.. math::

    \\eta(x) = W_\\mu x + b_\\mu + \\epsilon, \\qquad
    \\Sigma(x) = V(x) V(x)^\\top + \\mathrm{diag}\\!\\bigl(\\sigma^2(x)\\bigr),
    \\;\\; \\epsilon \\sim \\mathcal{N}(0, \\Sigma(x)).

Predictions average a small number of Monte Carlo softmax / sigmoid
samples.
"""

from __future__ import annotations

import math
from typing import Self, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
from jax import Array as JaxArray
from jaxtyping import Array, Float, PRNGKeyArray

from pyrox._core.pyrox_module import PyroxModule, pyrox_method


def _glorot_uniform(
    key: PRNGKeyArray, in_features: int, out_features: int
) -> Float[Array, "D_in D_out"]:
    lim = math.sqrt(6.0 / (in_features + out_features))
    return jr.uniform(
        key,
        (in_features, out_features),
        minval=-lim,
        maxval=lim,
    )


def _hetero_noisy_logits(
    layer: _HeteroscedasticBase,
    x: Float[Array, "N D_in"],
) -> Float[Array, "S N C"]:
    """Sample ``num_mc_samples`` heteroscedastic logits in one shot.

    Computes mean logits, low-rank factor, and diagonal scale via three
    deterministic linear maps (registered as ``pyrox_param``), then
    draws Monte Carlo logit perturbations.
    """
    N = x.shape[0]
    C = layer.num_classes
    r = layer.rank
    S = layer.num_mc_samples

    W_loc = layer.pyrox_param("W_loc", layer.W_loc_init)
    b_loc = layer.pyrox_param("b_loc", jnp.zeros(C, dtype=x.dtype))
    mu = x @ W_loc + b_loc

    W_scale = layer.pyrox_param("W_scale", layer.W_scale_init)
    b_scale = layer.pyrox_param("b_scale", jnp.zeros(C * r, dtype=x.dtype))
    V = (x @ W_scale + b_scale).reshape(N, C, r)

    W_diag = layer.pyrox_param("W_diag", layer.W_diag_init)
    b_diag = layer.pyrox_param(
        "b_diag", jnp.full((C,), float(layer.diag_init_bias), dtype=x.dtype)
    )
    sigma = jnp.exp(x @ W_diag + b_diag)

    key = cast(JaxArray, numpyro.prng_key())
    kz, ku = jr.split(key)
    z = jr.normal(kz, (S, N, r), dtype=x.dtype)
    u = jr.normal(ku, (S, N, C), dtype=x.dtype)
    eps = jnp.einsum("ncr,snr->snc", V, z) + sigma[None, :, :] * u
    return mu[None, :, :] + eps


class _HeteroscedasticBase(PyroxModule):
    """Shared state and init for the FA-noise heteroscedastic layers."""

    in_features: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    num_mc_samples: int = eqx.field(static=True, default=10)
    diag_init_bias: float = eqx.field(static=True, default=-3.0)
    pyrox_name: str | None = eqx.field(static=True, default=None)
    W_loc_init: Float[Array, "D_in C"] | None = eqx.field(default=None)
    W_scale_init: Float[Array, "D_in Cr"] | None = eqx.field(default=None)
    W_diag_init: Float[Array, "D_in C"] | None = eqx.field(default=None)

    @classmethod
    def init(
        cls,
        key: PRNGKeyArray,
        in_features: int,
        num_classes: int,
        rank: int,
        *,
        num_mc_samples: int = 10,
        diag_init_bias: float = -3.0,
        scale_init_factor: float = 0.1,
        pyrox_name: str | None = None,
    ) -> Self:
        """Construct the layer with Glorot-init linear factors."""
        if in_features <= 0 or num_classes <= 0 or rank <= 0:
            raise ValueError(
                "in_features, num_classes, rank must all be > 0; "
                f"got {in_features=}, {num_classes=}, {rank=}."
            )
        if num_mc_samples <= 0:
            raise ValueError(f"num_mc_samples must be > 0; got {num_mc_samples}.")
        kl, ks = jr.split(key)
        W_loc_init = _glorot_uniform(kl, in_features, num_classes)
        # Small init for the low-rank factor so it does not dominate the
        # mean logits before training.
        W_scale_init = scale_init_factor * _glorot_uniform(
            ks, in_features, num_classes * rank
        )
        W_diag_init = jnp.zeros((in_features, num_classes))
        return cls(
            in_features=in_features,
            num_classes=num_classes,
            rank=rank,
            num_mc_samples=num_mc_samples,
            diag_init_bias=diag_init_bias,
            pyrox_name=pyrox_name,
            W_loc_init=W_loc_init,
            W_scale_init=W_scale_init,
            W_diag_init=W_diag_init,
        )

    def __post_init__(self) -> None:
        if (
            self.W_loc_init is None
            or self.W_scale_init is None
            or self.W_diag_init is None
        ):
            raise ValueError(
                f"{type(self).__name__} requires W_loc_init / W_scale_init / "
                "W_diag_init. Use the .init(key, ...) classmethod."
            )
        d_in, C, r = self.in_features, self.num_classes, self.rank
        if self.W_loc_init.shape != (d_in, C):
            raise ValueError(
                f"W_loc_init shape {self.W_loc_init.shape} != ({d_in}, {C})."
            )
        if self.W_scale_init.shape != (d_in, C * r):
            raise ValueError(
                f"W_scale_init shape {self.W_scale_init.shape} != ({d_in}, {C * r})."
            )
        if self.W_diag_init.shape != (d_in, C):
            raise ValueError(
                f"W_diag_init shape {self.W_diag_init.shape} != ({d_in}, {C})."
            )


class MCSoftmaxDenseFA(_HeteroscedasticBase):
    r"""Heteroscedastic multi-class output layer (FA noise + softmax).

    Implements Collier et al. (2021): the logit covariance is
    input-dependent low-rank-plus-diagonal,

    .. math::

        \eta(x) = W_\mu x + b_\mu + \epsilon, \qquad
        \Sigma(x) = V(x) V(x)^\top + \operatorname{diag}\!\bigl(\sigma^2(x)\bigr),
        \;\; \epsilon \sim \mathcal{N}(0, \Sigma(x)),

    where :math:`V(x) = \mathrm{reshape}(W_V x + b_V, [C, r])` and
    :math:`\sigma(x) = \exp(W_\sigma x + b_\sigma)`. Output is the
    Monte Carlo average of softmaxed perturbed logits

    .. math::

        \hat{p}(y = k \mid x) \approx
        \frac{1}{S}\sum_{s=1}^{S}
        \mathrm{softmax}_k\!\bigl(\eta(x) + \epsilon_s\bigr).

    All linear factors are deterministic ``pyrox_param`` sites — the
    layer is heteroscedastic but not Bayesian over its weights. Use it
    as a drop-in head for classification when label noise is known to
    be input-dependent (label disagreement, fine-grained categories).

    Plate semantics:
        Same as other ``pyrox.nn`` Bayesian dense layers — call
        outside ``numpyro.plate("data", ..., subsample_size=...)`` so
        the parameter sites are unscaled. The MC noise is drawn from
        ``numpyro.prng_key()``.

    Attributes:
        in_features: Input dimension :math:`D_\mathrm{in}`.
        num_classes: Number of classes :math:`C`.
        rank: Rank :math:`r` of the low-rank factor :math:`V(x)`.
        num_mc_samples: Number of MC softmax samples :math:`S` per
            forward call.
        diag_init_bias: Initial value for the diagonal-scale bias
            ``b_diag`` (a small negative number keeps initial noise
            small).
        pyrox_name: Explicit scope name for NumPyro site registration.

    Example:
        >>> import jax.random as jr
        >>> import jax.numpy as jnp
        >>> from numpyro import handlers
        >>> layer = MCSoftmaxDenseFA.init(
        ...     jr.PRNGKey(0), in_features=4, num_classes=3, rank=2,
        ... )
        >>> x = jnp.ones((5, 4))
        >>> with handlers.seed(rng_seed=0):
        ...     probs = layer(x)
        >>> probs.shape
        (5, 3)
        >>> bool(jnp.allclose(probs.sum(axis=-1), 1.0))
        True

    References:
        Collier, M., Mustafa, B., Kokiopoulou, E., Jenatton, R., &
        Berent, J. (2021). *Correlated Input-Dependent Label Noise in
        Large-Scale Image Classification.* CVPR.
    """

    @pyrox_method
    def __call__(self, x: Float[Array, "N D_in"]) -> Float[Array, "N C"]:
        logits = _hetero_noisy_logits(self, x)
        return jnp.mean(jax.nn.softmax(logits, axis=-1), axis=0)


class MCSigmoidDenseFA(_HeteroscedasticBase):
    r"""Heteroscedastic multi-label output layer (FA noise + sigmoid).

    Identical low-rank-plus-diagonal logit-noise model as
    :class:`MCSoftmaxDenseFA`, but the per-class outputs are
    independent Bernoullis — final probabilities are the MC average of
    *element-wise* sigmoids, not a softmax. Use this for multi-label
    classification or independent binary heads.

    .. math::

        \hat{p}(y_k = 1 \mid x) \approx
        \frac{1}{S}\sum_{s=1}^{S}
        \sigma\!\bigl(\eta(x) + \epsilon_s\bigr)_k.

    See :class:`MCSoftmaxDenseFA` for the noise model, plate semantics,
    init API, and references.
    """

    @pyrox_method
    def __call__(self, x: Float[Array, "N D_in"]) -> Float[Array, "N C"]:
        logits = _hetero_noisy_logits(self, x)
        return jnp.mean(jax.nn.sigmoid(logits), axis=0)
