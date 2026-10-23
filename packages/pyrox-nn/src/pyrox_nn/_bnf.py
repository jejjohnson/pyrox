r"""Bayesian Neural Field (BNF) layers — port of Google's bayesnf.

Five layers wrapping the pure-JAX feature helpers in
`pyrox_nn._features` behind the `PyroxModule` PyTree
contract:

* `Standardization` — affine normalization with fixed mean/std.
* `FourierFeatures` — dyadic-frequency cos/sin basis per input.
* `SeasonalFeatures` — period-and-harmonic cos/sin basis on a
  scalar time axis.
* `InteractionFeatures` — element-wise products on selected
  pairs of input columns.
* `BayesianNeuralField` — the full BNF MLP, a wrapper around
  `geonnax.NeuralFieldMLP` (input rescaling + gated feature blocks +
  gain-modulated MLP with mixed ``elu`` / ``tanh`` activation, plus an
  optional encoder block). Every parameter of the core carries a
  $\mathrm{Logistic}(0, 1)$ prior registered via
  `PyroxModule.pyrox_sample`.

Reference
---------
Saad, F. A. *et al.* (2024) *Scalable spatiotemporal prediction with
Bayesian neural fields.* Nat. Commun. 15, 7942.
[bayesnf source.](https://github.com/google/bayesnf)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from geonnax import NeuralFieldMLP
from geonnax.basis import (
    fourier_features,
    interaction_features,
    seasonal_features,
)
from jaxtyping import Array, Float
from pyrox._core.pyrox_module import PyroxModule, pyrox_method

from pyrox_nn._batching import vmap_over_flat_batch


class Standardization(PyroxModule):
    r"""Apply a fixed-coefficient affine standardization.

    $$
    \tilde x \;=\; \frac{x - \mu}{\sigma}.
    $$

    Both ``mu`` and ``std`` are static (fit-time) constants, not
    learned. Use `pyrox_nn.preprocessing.fit_standardization` to
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
    frequencies $2\pi \cdot 2^d$ for $d \in \{0, \dots,
    \text{degree} - 1\}$. Concatenated across all columns.

    Wraps `pyrox_nn._features.fourier_features` per input
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

    For each period $\tau_p$ with $H_p$ harmonics, emits
    ``2 * H_p`` cos/sin columns. Total output width is $2 \sum_p
    H_p$.

    Wraps `pyrox_nn._features.seasonal_features`. Periods and
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

    Wraps `pyrox_nn._features.interaction_features`.

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


def _core_columns(
    d_in: int,
    encoder_cols: tuple[int, ...],
    fourier_degrees: tuple[int, ...],
    interactions: tuple[tuple[int, int], ...],
    seasonality_periods: tuple[float, ...],
    num_seasonal_harmonics: tuple[int, ...],
    time_col: int,
) -> tuple[int, ...]:
    """Input columns the core sees: all but the encoder's, which must be unused."""
    # Index checks the core would make, needed before the columns are remapped.
    if fourier_degrees and len(fourier_degrees) != d_in:
        raise ValueError(
            f"fourier_degrees needs one entry per input ({d_in}); "
            f"got {len(fourier_degrees)}."
        )
    if any(not (0 <= c < d_in) for pair in interactions for c in pair):
        raise ValueError(f"interactions index outside [0, {d_in}); got {interactions}.")
    if not 0 <= time_col < d_in:
        raise ValueError(f"time_col must be in [0, {d_in}); got {time_col}.")
    enc = set(encoder_cols)
    clashes = []
    if fourier_degrees:
        clashes += [c for c in enc if fourier_degrees[c] > 0]
    clashes += [c for pair in interactions for c in pair if c in enc]
    if seasonality_periods and any(num_seasonal_harmonics) and time_col in enc:
        clashes.append(time_col)
    if clashes:
        raise ValueError(
            "encoder_cols feed only the encoder, so they cannot also have Fourier "
            "features, interactions or seasonal features; got clashes on columns "
            f"{sorted(set(clashes))}."
        )
    core_cols = tuple(c for c in range(d_in) if c not in enc)
    if not core_cols:
        raise ValueError(
            "at least one input column must stay outside encoder_cols; got "
            f"encoder_cols={encoder_cols} for {d_in} inputs."
        )
    return core_cols


class BayesianNeuralField(PyroxModule):
    r"""The full Bayesian Neural Field architecture, wrapping `geonnax.NeuralFieldMLP`.

    The deterministic network is `geonnax.NeuralFieldMLP`; this layer puts
    an independent $\mathrm{Logistic}(0, 1)$ prior on every one of its
    parameters, registers them with `PyroxModule.pyrox_sample`, splices the
    draws into the core with `eqx.tree_at` and runs it over the batch. For
    one input $x \in \mathbb{R}^D$ with fixed scales $s$:

    1. The input is rescaled, $\tilde x = x \oslash (s \odot e^{a})$, with a
       learned log-scale adjustment $a$.
    2. Feature blocks $F_k$ are built: $\tilde x$, dyadic Fourier features of
       each column with a positive degree, seasonal features of the raw time
       column, interaction products of $\tilde x$, and, when an ``encoder``
       is given, the encoder's features of ``x[encoder_cols]``. Each block is
       gated by $\mathrm{softplus}(g_k)$; the concatenation is $h_0$. The
       encoder's columns enter **only** through the encoder: they are left
       out of $\tilde x$ (and so of the scales, Fourier and interaction
       features). A spherical-harmonic encoder on lon/lat therefore gives a
       field that is continuous across the dateline and regular at the
       poles by construction.
    3. Hidden layers
       $h_{\ell+1} = \sigma_\alpha\bigl(\mathrm{softplus}(c_\ell)\,
       (W_\ell^\top h_\ell / \sqrt{\lvert h_\ell \rvert} + b_\ell)\bigr)$,
       with $\sigma_\alpha = \alpha\,\mathrm{elu} + (1 - \alpha)\tanh$ and
       $\alpha = \mathrm{sigmoid}(\beta)$.
    4. A readout
       $f(x) = \mathrm{softplus}(c_{\text{out}})\,
       (W_{\text{out}}^\top h_L / \sqrt{\lvert h_L \rvert} + b_{\text{out}})$.

    The $1/\sqrt{\text{fan-in}}$ factor is the NTK scaling of BayesNF: it
    makes the layer-wise prior predictive a fan-in-independent Gaussian
    process in the infinite-width limit (Lee et al., 2018).

    Sites (scope ``pyrox_name``, else ``BayesianNeuralField``), each with a
    $\mathrm{Logistic}(0, 1)$ prior at full shape, in registration order:
    ``log_scale_adjustment`` $(D,)$ ($D$ minus the encoder's columns with
    an encoder); ``feature_gain_{b}`` (scalar, one per
    feature block, the encoder block last); ``logit_activation_weight``
    (scalar); ``layer_{l}_W`` $(I_\ell, \text{width})$, ``layer_{l}_b``
    $(\text{width},)$ and ``layer_{l}_gain`` (scalar) for each hidden layer;
    ``output_W`` $(\text{width}, 1)$; ``output_b`` $(1,)$; ``output_gain``
    (scalar). The names and shapes are those of the original inline
    implementation, so saved posteriors still load.

    The fixed scales follow the input's dtype (the inline implementation
    cast them to float32), so float64 inputs get float64 scales.

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
        encoder: Optional single-example encoder ``(C,) -> (E,)`` applied
            to ``x[encoder_cols]``, whose output enters as one extra gated
            feature block (for example
            ``geonnax.SphericalHarmonicEncoder(l_max=10, input_mode="lonlat")``
            on a lon/lat pair in radians). It is deterministic: its arrays,
            if any, are not sample sites.
        encoder_cols: Input columns fed to ``encoder``, in order; required
            exactly when ``encoder`` is given. They feed only the encoder,
            so they may not have Fourier degrees, appear in
            ``interactions`` or be the seasonal time column, and at least
            one column must stay outside them.
        core: The wrapped `geonnax.NeuralFieldMLP`, built from the fields
            above. Its arrays are placeholders: every one is replaced by a
            sample site on each call.

    Raises:
        ValueError: If ``encoder`` and ``encoder_cols`` are not given
            together, an encoder column is out of range or also used by
            another feature block, no column is left outside them, the
            encoder does not return a vector, or the core rejects the
            configuration.

    Examples:
        >>> import jax.numpy as jnp
        >>> from numpyro import handlers
        >>> from geonnax import SphericalHarmonicEncoder
        >>> bnf = BayesianNeuralField(
        ...     input_scales=(1.0, 1.0, 1.0), fourier_degrees=(0, 0, 0),
        ...     interactions=(), seasonality_periods=(365.25,),
        ...     num_seasonal_harmonics=(2,), width=16, depth=2,
        ...     encoder=SphericalHarmonicEncoder(l_max=3, input_mode="lonlat"),
        ...     encoder_cols=(1, 2), pyrox_name="bnf",
        ... )
        >>> with handlers.seed(rng_seed=0):
        ...     f = bnf(jnp.zeros((5, 3)))
        >>> f.shape
        (5,)

    References:
        Saad, F. A. *et al.* (2024). Scalable spatiotemporal prediction with
        Bayesian neural fields. *Nature Communications* 15, 7942.
        Lee, J. *et al.* (2018). Deep neural networks as Gaussian processes.
        *ICLR*.
    """

    input_scales: tuple[float, ...] = eqx.field(static=True)
    fourier_degrees: tuple[int, ...] = eqx.field(static=True)
    interactions: tuple[tuple[int, int], ...] = eqx.field(static=True)
    seasonality_periods: tuple[float, ...] = eqx.field(static=True)
    num_seasonal_harmonics: tuple[int, ...] = eqx.field(static=True)
    width: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    time_col: int = eqx.field(static=True)
    pyrox_name: str | None
    encoder: Callable[[Float[Array, " C"]], Float[Array, " E"]] | None
    encoder_cols: tuple[int, ...] | None = eqx.field(static=True)
    core: NeuralFieldMLP

    def __init__(
        self,
        input_scales: tuple[float, ...],
        fourier_degrees: tuple[int, ...],
        interactions: tuple[tuple[int, int], ...],
        seasonality_periods: tuple[float, ...],
        num_seasonal_harmonics: tuple[int, ...],
        width: int,
        depth: int,
        time_col: int = 0,
        pyrox_name: str | None = None,
        encoder: Callable[[Float[Array, " C"]], Float[Array, " E"]] | None = None,
        encoder_cols: tuple[int, ...] | None = None,
    ) -> None:
        d_in = len(input_scales)
        extra_features = 0
        if (encoder is None) != (encoder_cols is None):
            raise ValueError(
                "encoder and encoder_cols must be given together; got "
                f"encoder={encoder!r}, encoder_cols={encoder_cols!r}."
            )
        if encoder is not None and encoder_cols is not None:
            encoder_cols = tuple(int(c) for c in encoder_cols)
            if not encoder_cols or any(not 0 <= c < d_in for c in encoder_cols):
                raise ValueError(
                    f"encoder_cols must be a non-empty tuple of indices in "
                    f"[0, {d_in}); got {encoder_cols!r}."
                )
            out = jax.eval_shape(
                encoder, jax.ShapeDtypeStruct((len(encoder_cols),), jnp.float32)
            )
            if len(out.shape) != 1:
                raise ValueError(
                    "encoder must map a (C,) vector to an (E,) vector; got output "
                    f"shape {out.shape}."
                )
            extra_features = out.shape[0]
        core_cols = _core_columns(
            d_in,
            encoder_cols or (),
            fourier_degrees,
            interactions,
            seasonality_periods,
            num_seasonal_harmonics,
            time_col,
        )
        remap = {c: i for i, c in enumerate(core_cols)}
        self.input_scales = input_scales
        self.fourier_degrees = fourier_degrees
        self.interactions = interactions
        self.seasonality_periods = seasonality_periods
        self.num_seasonal_harmonics = num_seasonal_harmonics
        self.width = width
        self.depth = depth
        self.time_col = time_col
        self.pyrox_name = pyrox_name
        self.encoder = encoder
        self.encoder_cols = encoder_cols
        # The core sees only the columns the encoder does not consume. Its
        # arrays are placeholders: `__call__` replaces every one of them
        # (all nine parameter fields) with a sample site before running the
        # core, so their values never reach an output and a fixed key is
        # harmless. Real arrays (rather than `eqx.filter_eval_shape`
        # structs) keep the module an ordinary pytree of arrays under
        # `jit`, `vmap` and serialisation.
        self.core = NeuralFieldMLP.init(
            tuple(input_scales[c] for c in core_cols),
            width=width,
            depth=depth,
            fourier_degrees=tuple(fourier_degrees[c] for c in core_cols)
            if fourier_degrees
            else (),
            interactions=tuple((remap[i], remap[j]) for i, j in interactions),
            seasonality_periods=seasonality_periods,
            num_seasonal_harmonics=num_seasonal_harmonics,
            time_col=remap.get(time_col, 0),
            extra_features=extra_features,
            key=jr.PRNGKey(0),
        )

    @staticmethod
    def _logistic_prior(shape: tuple[int, ...]) -> dist.Distribution:
        """Independent Logistic(0, 1) prior over an array of given shape."""
        if not shape:
            return dist.Logistic(0.0, 1.0)
        return dist.Logistic(jnp.zeros(shape), 1.0).to_event(len(shape))

    @pyrox_method
    def __call__(self, x: Float[Array, "*batch D"]) -> Float[Array, " *batch"]:
        core = self.core
        prior = self._logistic_prior
        # Sites are registered in the order of the original inline
        # implementation, so seeded draws are unchanged too.
        log_scale_adjustment = self.pyrox_sample(
            "log_scale_adjustment", prior(core.log_scale_adjustment.shape)
        )
        feature_gain = jnp.stack(
            [
                self.pyrox_sample(f"feature_gain_{b}", prior(()))
                for b in range(core.feature_gain.shape[0])
            ]
        )
        logit_activation_weight = self.pyrox_sample(
            "logit_activation_weight", prior(())
        )
        layers_W, layers_b, layers_gain = [], [], []
        for layer_idx in range(self.depth):
            layers_W.append(
                self.pyrox_sample(
                    f"layer_{layer_idx}_W", prior(core.layers_W[layer_idx].shape)
                )
            )
            layers_b.append(
                self.pyrox_sample(
                    f"layer_{layer_idx}_b", prior(core.layers_b[layer_idx].shape)
                )
            )
            layers_gain.append(self.pyrox_sample(f"layer_{layer_idx}_gain", prior(())))
        output_W = self.pyrox_sample("output_W", prior(core.output_W.shape))
        output_b = self.pyrox_sample("output_b", prior(core.output_b.shape))
        output_gain = self.pyrox_sample("output_gain", prior(()))

        # With depth 0 the layer fields are empty (an empty tuple has no
        # identity `eqx.tree_at` can find), so they are only swapped when
        # there are hidden layers.
        def where(c: NeuralFieldMLP) -> tuple[Any, ...]:
            nodes = (
                c.log_scale_adjustment,
                c.feature_gain,
                c.logit_activation_weight,
                c.output_W,
                c.output_b,
                c.output_gain,
            )
            if self.depth > 0:
                nodes = (*nodes, c.layers_W, c.layers_b, c.layers_gain)
            return nodes

        values: tuple[Any, ...] = (
            log_scale_adjustment,
            feature_gain,
            logit_activation_weight,
            output_W,
            output_b,
            output_gain,
        )
        if self.depth > 0:
            values = (
                *values,
                tuple(layers_W),
                tuple(layers_b),
                jnp.stack(layers_gain),
            )
        core = eqx.tree_at(where, core, values)

        encoder, encoder_cols = self.encoder, self.encoder_cols
        if encoder is None or encoder_cols is None:
            fn = core
        else:
            enc_idx = jnp.asarray(encoder_cols, dtype=jnp.int32)
            core_idx = jnp.asarray(
                [c for c in range(len(self.input_scales)) if c not in encoder_cols],
                dtype=jnp.int32,
            )

            def fn(xi: Float[Array, " D"]) -> Float[Array, " 1"]:
                return core(xi[core_idx], encoder(xi[enc_idx]))

        return vmap_over_flat_batch(fn, x)[..., 0]
