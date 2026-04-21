r"""Bayesian Neural Field (BNF) layers — port of Google's bayesnf.

Five layers wrapping the pure-JAX feature helpers in
:mod:`pyrox.nn._features` behind the :class:`PyroxModule` PyTree
contract:

* :class:`Standardization` — affine normalization with fixed mean/std.
* :class:`FourierFeatures` — dyadic-frequency cos/sin basis per input.
* :class:`SeasonalFeatures` — period-and-harmonic cos/sin basis on a
  scalar time axis.
* :class:`InteractionFeatures` — element-wise products on selected
  pairs of input columns.
* :class:`BayesianNeuralField` — the full BNF MLP: input rescaling +
  feature concatenation + gain-modulated MLP with mixed
  ``elu`` / ``tanh`` activation. Every learnable leaf carries a
  :math:`\mathrm{Logistic}(0, 1)` prior registered via
  :meth:`PyroxModule.pyrox_sample`.

Reference
---------
Saad, F. A. *et al.* (2024) *Scalable spatiotemporal prediction with
Bayesian neural fields.* Nat. Commun. 15, 7942.
[bayesnf source.](https://github.com/google/bayesnf)
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float

from pyrox._core.pyrox_module import PyroxModule, pyrox_method
from pyrox.nn._features import (
    fourier_features,
    interaction_features,
    seasonal_features,
)


class Standardization(PyroxModule):
    r"""Apply a fixed-coefficient affine standardization.

    .. math::

        \tilde x \;=\; \frac{x - \mu}{\sigma}.

    Both ``mu`` and ``std`` are static (fit-time) constants, not
    learned. Use :func:`pyrox.preprocessing.fit_standardization` to
    construct from a pandas DataFrame.

    Attributes:
        mu: Per-feature mean, shape ``(D,)``.
        std: Per-feature standard deviation, shape ``(D,)``. Must be
            strictly positive — guard upstream.
        pyrox_name: Optional override for the per-instance scope name.
    """

    mu: Float[Array, " D"]
    std: Float[Array, " D"]
    pyrox_name: str | None = None

    @pyrox_method
    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N D"]:
        return (x - self.mu) / self.std


class FourierFeatures(PyroxModule):
    r"""Per-input dyadic-frequency Fourier basis.

    For each input column, evaluates ``2 * degree`` Fourier features at
    frequencies :math:`2\pi \cdot 2^d` for :math:`d \in \{0, \dots,
    \text{degree} - 1\}`. Concatenated across all columns.

    Wraps :func:`pyrox.nn._features.fourier_features` per input
    dimension.

    Attributes:
        degrees: Number of dyadic frequencies per input column, as a
            Python ``tuple[int, ...]``. A column with ``degree = 0``
            contributes no features. Marked ``static`` so the loop
            over columns unrolls at trace time.
        rescale: If ``True``, divide each ``(cos_d, sin_d)`` pair by
            ``d + 1`` to bias the prior toward lower frequencies.
        pyrox_name: Optional scope-name override.
    """

    degrees: tuple[int, ...] = eqx.field(static=True)
    rescale: bool = eqx.field(static=True, default=False)
    pyrox_name: str | None = None

    @pyrox_method
    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N F"]:
        feats = []
        for col_idx, d in enumerate(self.degrees):
            if d <= 0:
                continue
            feats.append(fourier_features(x[:, col_idx], d, rescale=self.rescale))
        if not feats:
            return jnp.zeros((x.shape[0], 0), dtype=x.dtype)
        return jnp.concatenate(feats, axis=-1)


class SeasonalFeatures(PyroxModule):
    r"""Period-and-harmonic cos/sin basis on a scalar time axis.

    For each period :math:`\tau_p` with :math:`H_p` harmonics, emits
    ``2 * H_p`` cos/sin columns. Total output width is :math:`2 \sum_p
    H_p`.

    Wraps :func:`pyrox.nn._features.seasonal_features`. Periods and
    harmonics are kept as Python tuples (static) so the inner shape
    structure is known at trace time.

    Attributes:
        periods: Period values, ``tuple[float, ...]``.
        harmonics: Harmonics per period, ``tuple[int, ...]``.
        rescale: If ``True``, divide each ``(cos, sin)`` pair by its
            within-period harmonic index.
        pyrox_name: Optional scope-name override.
    """

    periods: tuple[float, ...] = eqx.field(static=True)
    harmonics: tuple[int, ...] = eqx.field(static=True)
    rescale: bool = eqx.field(static=True, default=False)
    pyrox_name: str | None = None

    @pyrox_method
    def __call__(self, t: Float[Array, " N"]) -> Float[Array, "N F"]:
        return seasonal_features(t, self.periods, self.harmonics, rescale=self.rescale)


class InteractionFeatures(PyroxModule):
    r"""Element-wise products on selected pairs of input columns.

    Wraps :func:`pyrox.nn._features.interaction_features`.

    Attributes:
        pairs: Index pairs, ``tuple[tuple[int, int], ...]``. Empty
            tuple produces an ``(N, 0)`` output. Static so the count
            ``K`` is known at trace time.
        pyrox_name: Optional scope-name override.
    """

    pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    pyrox_name: str | None = None

    @pyrox_method
    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N K"]:
        if not self.pairs:
            return jnp.zeros((x.shape[0], 0), dtype=x.dtype)
        return interaction_features(x, jnp.asarray(self.pairs, dtype=jnp.int32))


class BayesianNeuralField(PyroxModule):
    r"""The full Bayesian Neural Field architecture.

    A spatiotemporal MLP with:

    1. A learned per-input log-scale adjustment (Logistic(0, 1) prior).
    2. Four feature blocks concatenated into ``h_0``: rescaled inputs,
       Fourier features, seasonal features, interaction products.
    3. Per-block ``softplus(feature_gain)`` modulation.
    4. A depth-``L`` MLP whose layers are
       :math:`h_{\ell+1} = \sigma_\alpha\bigl(g_\ell \cdot W_\ell\, h_\ell
       / \sqrt{\lvert h_\ell \rvert}\bigr)`, where :math:`\sigma_\alpha
       = \mathrm{sig}(\beta) \cdot \mathrm{elu} + (1 - \mathrm{sig}(\beta))
       \cdot \mathrm{tanh}` is a learned mixed activation.
    5. A final linear layer scaled by ``softplus(output_gain)``.

    All weights, biases, gains, scales, and the activation logit carry
    independent :math:`\mathrm{Logistic}(0, 1)` priors registered via
    :meth:`PyroxModule.pyrox_sample`.

    The :math:`1/\sqrt{\text{fan-in}}` pre-normalization is the
    standard NTK-scaling trick — it makes the layer-wise prior
    predictive a fan-in-independent Gaussian process in the
    infinite-width limit (Lee et al., 2018).

    Attributes:
        input_scales: Per-input fixed scale (typically training-data
            inter-quartile range). Static ``tuple[float, ...]``.
        fourier_degrees: Per-input number of dyadic Fourier
            frequencies. Static ``tuple[int, ...]``; use ``0`` to skip
            a column.
        interactions: Pair-index list for interaction features. Static
            ``tuple[tuple[int, int], ...]``; empty for none.
        seasonality_periods: Periods for seasonal features. Static
            ``tuple[float, ...]``. The time variable is taken from
            input column ``time_col``.
        num_seasonal_harmonics: Harmonics per period. Static
            ``tuple[int, ...]``.
        width: Hidden layer width.
        depth: Number of hidden MLP layers.
        time_col: Index of the time column inside ``x`` used for
            seasonal features (default 0).
        pyrox_name: Optional scope-name override.
    """

    input_scales: tuple[float, ...] = eqx.field(static=True)
    fourier_degrees: tuple[int, ...] = eqx.field(static=True)
    interactions: tuple[tuple[int, int], ...] = eqx.field(static=True)
    seasonality_periods: tuple[float, ...] = eqx.field(static=True)
    num_seasonal_harmonics: tuple[int, ...] = eqx.field(static=True)
    width: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    time_col: int = eqx.field(static=True, default=0)
    pyrox_name: str | None = None

    @staticmethod
    def _logistic_prior(shape: tuple[int, ...]) -> dist.Distribution:
        """Independent Logistic(0, 1) prior over an array of given shape."""
        if not shape:
            return dist.Logistic(0.0, 1.0)
        return dist.Logistic(jnp.zeros(shape), 1.0).to_event(len(shape))

    @pyrox_method
    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, " N"]:
        d_in = len(self.input_scales)
        input_scales = jnp.asarray(self.input_scales, dtype=jnp.float32)

        # 1. Input rescaling: x / (input_scales * exp(log_scale_adjustment)).
        log_scale_adjustment = self.pyrox_sample(
            "log_scale_adjustment",
            self._logistic_prior((d_in,)),
        )
        scaled_x = x / (input_scales * jnp.exp(log_scale_adjustment))

        # 2. Build the four feature blocks.
        feature_blocks: list[Float[Array, "N F_block"]] = [scaled_x]

        # Fourier per input dim (only for degrees > 0).
        for col_idx, d in enumerate(self.fourier_degrees):
            if d > 0:
                feature_blocks.append(
                    fourier_features(scaled_x[:, col_idx], d, rescale=True)
                )

        # Seasonal on the time column.
        if self.seasonality_periods and any(self.num_seasonal_harmonics):
            feature_blocks.append(
                seasonal_features(
                    x[:, self.time_col],
                    self.seasonality_periods,
                    self.num_seasonal_harmonics,
                    rescale=True,
                )
            )

        # Interaction products.
        if self.interactions:
            feature_blocks.append(
                interaction_features(
                    scaled_x, jnp.asarray(self.interactions, dtype=jnp.int32)
                )
            )

        # 3. Per-block softplus(feature_gain) modulation.
        gated_blocks: list[Float[Array, "N F_block"]] = []
        for b_idx, block in enumerate(feature_blocks):
            if block.shape[-1] == 0:
                continue
            gain = self.pyrox_sample(
                f"feature_gain_{b_idx}",
                self._logistic_prior(()),
            )
            gated_blocks.append(block * jax.nn.softplus(gain))
        h = jnp.concatenate(gated_blocks, axis=-1)

        # 4. Mixed elu/tanh activation, learned mix weight.
        logit_activation_weight = self.pyrox_sample(
            "logit_activation_weight",
            self._logistic_prior(()),
        )
        alpha = jax.nn.sigmoid(logit_activation_weight)

        def activation(z: Float[Array, ...]) -> Float[Array, ...]:
            return alpha * jax.nn.elu(z) + (1.0 - alpha) * jnp.tanh(z)

        # 5. Hidden MLP layers.
        for layer_idx in range(self.depth):
            fan_in = h.shape[-1]
            W = self.pyrox_sample(
                f"layer_{layer_idx}_W",
                self._logistic_prior((fan_in, self.width)),
            )
            b = self.pyrox_sample(
                f"layer_{layer_idx}_b",
                self._logistic_prior((self.width,)),
            )
            layer_gain = self.pyrox_sample(
                f"layer_{layer_idx}_gain",
                self._logistic_prior(()),
            )
            h = h / jnp.sqrt(fan_in)
            h = activation(jax.nn.softplus(layer_gain) * (h @ W + b))

        # 6. Output linear, scaled by softplus(output_gain).
        fan_in = h.shape[-1]
        W_out = self.pyrox_sample(
            "output_W",
            self._logistic_prior((fan_in, 1)),
        )
        b_out = self.pyrox_sample(
            "output_b",
            self._logistic_prior((1,)),
        )
        output_gain = self.pyrox_sample(
            "output_gain",
            self._logistic_prior(()),
        )
        h = h / jnp.sqrt(fan_in)
        return (jax.nn.softplus(output_gain) * (h @ W_out + b_out)).squeeze(-1)
