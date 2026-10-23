"""``BNFEstimator`` family — sklearn-style facade over the BNF stack.

Three configurations:

* `BNFEstimator` — MAP fit (``prior_weight=1``).
* `BNFEstimatorMLE` — MLE fit (``prior_weight=0``); identical
  otherwise.
* `BNFEstimatorVI` — mean-field VI via ``ensemble_vi`` over
  ``AutoNormal``.

All three return a `FittedBNF` whose ``predict`` produces a
posterior-predictive mean (Gaussian-mixture closed form for
``observation_model="NORMAL"``) and, optionally, per-row quantiles
(MC-sampled from the mixture).

Fitting and prediction are those of `pyrox_nn.api.NeuralFieldEstimator`:
a BNF estimator is that estimator on a `BayesianNeuralField` built from
pandas preprocessing, with the same log joint.

Currently only ``observation_model="NORMAL"`` is implemented end-to-end.
``"NB"`` and ``"ZINB"`` are reserved for the upcoming gaussx mixture-
quantile follow-up; instantiating them raises a clear
``NotImplementedError`` pointing at the tracking issue.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import pandas as pd
from jaxtyping import Array, Float, PRNGKeyArray

from pyrox_nn._bnf import BayesianNeuralField
from pyrox_nn.api._estimator import EstimatorBase, FittedEstimator
from pyrox_nn.api._field import (
    NeuralFieldEstimator,
    _field_init_fn,
    _field_log_joint,
    _field_model,
    # Re-exported: the BNF tests (and earlier callers) import them from here.
    _gaussian_mixture_quantiles as _gaussian_mixture_quantiles,
    _predict,
    _predict_map_ensemble as _predict_map_ensemble,
)
from pyrox_nn.preprocessing import (
    SpatiotemporalFit,
    encode_time_column,
    fit_spatiotemporal,
)


ObservationModel = Literal["NORMAL", "NB", "ZINB"]
InferenceKind = Literal["map", "vi"]


def _df_to_design(
    df: pd.DataFrame,
    fit: SpatiotemporalFit,
    *,
    timetype: Literal["int", "datetime"] = "int",
    freq: str | None = None,
) -> Float[Array, "N D"]:
    """Pack a DataFrame into the standardized design matrix.

    The time column is encoded with the *training* ``time_min`` /
    ``time_scale`` so test-set predictions align with the basis the
    BNF was fit on.
    """
    cols = list(fit.feature_cols)
    arrs: list[Float[Array, " N"]] = []
    for col_idx, col in enumerate(cols):
        if col_idx == 0:
            t, _, _ = encode_time_column(
                df[col], timetype=timetype, freq=freq, time_min=fit.time_min
            )
            arrs.append(t)
        else:
            arrs.append(jnp.asarray(df[col].to_numpy(), dtype=jnp.float32))
    x = jnp.stack(arrs, axis=-1)
    return fit.standardize_layer(x)


def _build_bnf(
    fit: SpatiotemporalFit,
    width: int,
    depth: int,
    encoder: Callable[[Array], Array] | None = None,
    encoder_cols: tuple[int, ...] | None = None,
) -> BayesianNeuralField:
    """Construct the `BayesianNeuralField` layer from a fit bundle.

    The design matrix is standardized, but an encoder expects coordinates in
    their own units (a lon/lat pair in radians, say), so the encoder is
    composed with the inverse standardization of its columns.
    """
    d_in = len(fit.feature_cols)
    if encoder is not None and encoder_cols is not None:
        cols = jnp.asarray(encoder_cols, dtype=jnp.int32)
        mu = fit.standardize_layer.mu[cols]
        std = fit.standardize_layer.std[cols]
        user_encoder = encoder

        def encoder(z: Array) -> Array:
            return user_encoder(z * std + mu)

    return BayesianNeuralField(
        input_scales=tuple([1.0] * d_in),  # already standardized
        fourier_degrees=fit.fourier_layer.degrees,
        interactions=fit.interaction_layer.pairs,
        seasonality_periods=fit.seasonal_layer.periods,
        num_seasonal_harmonics=fit.seasonal_layer.harmonics,
        width=width,
        depth=depth,
        time_col=0,
        pyrox_name="bnf",
        encoder=encoder,
        encoder_cols=encoder_cols,
    )


NoiseModel = Literal["fixed", "learned"]


# The BNF names of the shared init and log joint (unchanged signatures).
_bnf_init_fn = _field_init_fn
_bnf_log_joint = _field_log_joint


def _bnf_model(
    bnf_template: BayesianNeuralField,
    sigma_obs: float,
    *,
    noise: NoiseModel = "fixed",
    has_sigma_col: bool = False,
) -> Callable[[Array, Array | None], None]:
    r"""The BNF regression model: `pyrox_nn.api._field._field_model` on the BNF.

    $$
    y_i \sim \mathcal N\bigl(f_\theta(x_i),\ \sigma_i^2 + \sigma_0^2\bigr),
    $$

    with $\sigma_i = 0$ without a ``sigma_col``; with it the observations are
    packed as ``y = [y_i, σ_i]`` of shape ``(N, 2)``. The latent field is the
    deterministic site ``f``.
    """
    return _field_model(bnf_template, sigma_obs, noise=noise, has_sigma=has_sigma_col)


class BNFEstimator(EstimatorBase):
    r"""MAP-fit Bayesian Neural Field estimator.

    Composes the BNF layer (`pyrox_nn.BayesianNeuralField`) with
    pandas-side preprocessing (`pyrox_nn.preprocessing.fit_spatiotemporal`)
    and an ensemble-MAP inference loop
    (`pyrox.inference.ensemble_map`).

    Attributes:
        feature_cols: Input columns (first one is time).
        target_col: Target column.
        width: Hidden MLP width.
        depth: Hidden MLP depth.
        observation_model: ``"NORMAL"`` (currently only this is fully
            implemented). ``"NB"`` and ``"ZINB"`` raise
            ``NotImplementedError`` until ``gaussx.mixture_quantile``
            ships.
        seasonality_periods: Periods for seasonal features. Empty ⇒
            no seasonal features.
        num_seasonal_harmonics: Harmonics per period.
        fourier_degrees: Per-input dyadic Fourier degrees. Length must
            match ``feature_cols`` (or empty ⇒ no Fourier features).
        interactions: Pair-index list for interaction features.
        encoder: Optional single-example encoder ``(C,) -> (E,)`` whose
            output enters the network as one extra gated feature block (see
            `pyrox_nn.BayesianNeuralField`), for example
            ``geonnax.SphericalHarmonicEncoder(l_max=10, input_mode="lonlat")``.
            It receives ``encoder_cols`` in the DataFrame's own units: the
            estimator undoes its standardization of those columns first (the
            time column arrives encoded, as in the design matrix).
        encoder_cols: Indices into ``feature_cols`` fed to ``encoder``, in
            order; required exactly when ``encoder`` is given.
        timetype: ``"int"`` or ``"datetime"``.
        freq: Optional unit string for ``datetime`` time columns.
        noise: ``"fixed"`` (default) keeps the noise scale $\sigma_0$ at
            ``sigma_obs``; ``"learned"`` infers it through the sample site
            ``log_sigma_obs`` $= \log \sigma_0 \sim \mathcal N(\log
            \sigma_{\text{obs}}, 1)$, which the MAP and VI ensembles treat
            like any other latent. With ``prior_weight=0`` (MLE) and no
            ``sigma_col`` nothing bounds $\sigma_0$ below, so that
            combination raises ``ValueError``.
        sigma_obs: Observation noise std $\sigma_0$ when ``noise="fixed"``
            (with ``sigma_col``, a floor for representativeness error), or
            the prior median of $\sigma_0$ when ``noise="learned"``.
        sigma_col: Optional column of per-row noise stds $\sigma_i > 0$, in
            target units. Each row's likelihood is
            $\mathcal N(f(x_i), \sigma_i^2 + \sigma_0^2)$. The target is not
            rescaled by the estimator, so $\sigma_i$ is used as given.
        ensemble_size: Number of MAP ensemble members.
        num_epochs: SGD-MAP iterations per member.
        learning_rate: Adam learning rate.
        prior_weight: ``1`` ⇒ MAP, ``0`` ⇒ MLE, fractional ⇒ tempered.
    """

    width: int = eqx.field(static=True, default=64)
    depth: int = eqx.field(static=True, default=4)
    observation_model: ObservationModel = eqx.field(static=True, default="NORMAL")
    seasonality_periods: tuple[float, ...] = eqx.field(static=True, default=())
    num_seasonal_harmonics: tuple[int, ...] = eqx.field(static=True, default=())
    fourier_degrees: tuple[int, ...] = eqx.field(static=True, default=())
    interactions: tuple[tuple[int, int], ...] = eqx.field(static=True, default=())
    encoder: Callable[[Array], Array] | None = None
    encoder_cols: tuple[int, ...] | None = eqx.field(static=True, default=None)
    timetype: Literal["int", "datetime"] = eqx.field(static=True, default="int")
    freq: str | None = eqx.field(static=True, default=None)
    noise: NoiseModel = eqx.field(static=True, default="fixed")
    sigma_obs: float = eqx.field(static=True, default=0.1)
    sigma_col: str | None = eqx.field(static=True, default=None)
    ensemble_size: int = eqx.field(static=True, default=16)
    num_epochs: int = eqx.field(static=True, default=2000)
    learning_rate: float = eqx.field(static=True, default=5e-3)
    prior_weight: float = eqx.field(static=True, default=1.0)
    inference_kind: InferenceKind = eqx.field(static=True, default="map")
    standardize_columns: tuple[str, ...] | None = eqx.field(static=True, default=None)
    # Optional guide factory for the VI path. Takes the numpyro model
    # function and returns a guide. Defaults to AutoNormal at fit-time
    # (mean-field). Pass `AutoLowRankMultivariateNormal`, `AutoIAFNormal`,
    # `AutoBNAFNormal`, or any other AutoGuide constructor for richer
    # variational families that can represent posterior weight
    # correlations the mean-field guide cannot.
    guide_factory: Callable[[Callable], Any] | None = eqx.field(
        static=True, default=None
    )

    def fit(self, df: pd.DataFrame, *, seed: PRNGKeyArray | int) -> FittedBNF:
        if self.observation_model != "NORMAL":
            raise NotImplementedError(
                f"observation_model={self.observation_model!r} requires "
                "gaussx.mixture_quantile (tracked as gaussx#121). Use "
                "observation_model='NORMAL' for now."
            )
        if self.noise not in ("fixed", "learned"):
            # A BayesianNeuralField returns the field alone, so it cannot be
            # heteroscedastic.
            raise ValueError(f"noise must be 'fixed' or 'learned'; got {self.noise!r}.")
        # sigma_obs, the inference kind and learned noise under MLE are
        # validated by NeuralFieldEstimator.fit, after the preprocessing.
        # 1. Pandas-side preprocessing → SpatiotemporalFit bundle.
        fit_bundle = fit_spatiotemporal(
            df,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            timetype=self.timetype,
            freq=self.freq,
            seasonality_periods=self.seasonality_periods,
            num_seasonal_harmonics=self.num_seasonal_harmonics,
            fourier_degrees=self.fourier_degrees,
            interactions=self.interactions,
            standardize=self.standardize_columns,
        )
        # 2. Build the BNF layer template + design matrix.
        bnf = _build_bnf(
            fit_bundle, self.width, self.depth, self.encoder, self.encoder_cols
        )
        x_train = _df_to_design(df, fit_bundle, timetype=self.timetype, freq=self.freq)
        y_train = jnp.asarray(df[self.target_col].to_numpy(), dtype=jnp.float32)
        sigma_i = None
        if self.sigma_col is not None:
            sigma_i = _sigma_column(df, self.sigma_col)
        # 3. Run ensemble inference: a NeuralFieldEstimator on the BNF.
        fitted = NeuralFieldEstimator(
            model_fn=lambda: bnf,
            noise=self.noise,
            sigma_obs=self.sigma_obs,
            ensemble_size=self.ensemble_size,
            inference=self.inference_kind,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            prior_weight=self.prior_weight,
            guide_factory=self.guide_factory,
        ).fit(x_train, y_train, sigma=sigma_i, seed=seed)
        return FittedBNF(
            config=self,
            fit_bundle=fit_bundle,
            params=fitted.params,
            losses=fitted.losses,
            bnf_template=bnf,
        )


class BNFEstimatorMLE(BNFEstimator):
    """BNF estimator with `prior_weight=0` (MLE)."""

    prior_weight: float = eqx.field(static=True, default=0.0)


class BNFEstimatorVI(BNFEstimator):
    """BNF estimator using mean-field VI via `ensemble_vi`."""

    inference_kind: InferenceKind = eqx.field(static=True, default="vi")


class FittedBNF(FittedEstimator):
    """Output of `BNFEstimator.fit`.

    Attributes:
        config: The `BNFEstimator` that produced this fit.
        fit_bundle: Pandas-side preprocessing record.
        params: Stacked per-ensemble-member parameter dict (each leaf
            has a leading ``(E,)`` axis for MAP, or an
            ``AutoNormal``-shaped variational-parameter dict for VI).
        losses: Per-member loss history, shape ``(E, num_epochs)``.
        bnf_template: The `BayesianNeuralField` layer instance
            used for inference (identical across members).
    """

    fit_bundle: SpatiotemporalFit
    params: Any
    losses: Float[Array, "E T"]
    bnf_template: BayesianNeuralField

    def predict(
        self,
        df: pd.DataFrame,
        *,
        quantiles: Sequence[float] | None = None,
        predict_seed: int = 0,
        num_posterior_samples: int = 32,
    ) -> Float[Array, " N"] | tuple[Float[Array, " N"], tuple[Float[Array, " N"], ...]]:
        r"""Posterior-predictive mean (and optional quantiles).

        For MAP fits the per-member predictions are point estimates;
        for VI fits we draw ``num_posterior_samples`` weight samples
        per member from the variational guide.

        The quantiles are those of the latent field plus the noise scale
        $\sigma_0$ (the fixed ``sigma_obs``, or each member's or draw's
        learned value): an equal-weight mixture of
        $\mathcal N(f_m(x), \sigma_{0,m}^2)$. Per-row $\sigma_i$ from
        ``sigma_col`` are not known at new points, so they are not added;
        add $\sigma_i^2$ yourself where you know it.

        Args:
            df: Inputs.
            quantiles: Optional levels. Returns ``(mean, (q_1, ..., q_K))``.
            predict_seed: PRNG seed for VI posterior sampling and
                quantile MC.
            num_posterior_samples: Per-member posterior draws used for
                VI predictive moments and quantile MC.
        """
        cfg = self.config
        assert isinstance(cfg, BNFEstimator)
        x_pred = _df_to_design(
            df, self.fit_bundle, timetype=cfg.timetype, freq=cfg.freq
        )
        return _predict(
            self.params,
            self.bnf_template,
            x_pred,
            encoder=None,
            noise=cfg.noise,
            sigma_obs=cfg.sigma_obs,
            has_sigma=cfg.sigma_col is not None,
            inference=cfg.inference_kind,
            guide_factory=cfg.guide_factory,
            quantiles=quantiles,
            predict_seed=predict_seed,
            num_posterior_samples=num_posterior_samples,
        )


def _sigma_column(df: pd.DataFrame, col: str) -> Float[Array, " N"]:
    """Read and validate the per-row noise std column (finite, > 0)."""
    if col not in df.columns:
        raise ValueError(f"sigma_col {col!r} is not a column of the DataFrame.")
    sigma = df[col].to_numpy(dtype=float)
    if not ((sigma > 0) & (sigma < float("inf"))).all():
        raise ValueError(
            f"sigma_col {col!r} must be finite and > 0 on every row; got "
            f"min {sigma.min()} and {int((~(sigma > 0)).sum())} non-positive or NaN."
        )
    return jnp.asarray(sigma, dtype=jnp.float32)
