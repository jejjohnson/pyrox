"""``BNFEstimator`` family — sklearn-style facade over the BNF stack.

Three configurations:

* :class:`BNFEstimator` — MAP fit (``prior_weight=1``).
* :class:`BNFEstimatorMLE` — MLE fit (``prior_weight=0``); identical
  otherwise.
* :class:`BNFEstimatorVI` — mean-field VI via ``ensemble_vi`` over
  ``AutoNormal``.

All three return a :class:`FittedBNF` whose ``predict`` produces a
posterior-predictive mean (Gaussian-mixture closed form for
``observation_model="NORMAL"``) and, optionally, per-row quantiles
(MC-sampled from the mixture).

Currently only ``observation_model="NORMAL"`` is implemented end-to-end.
``"NB"`` and ``"ZINB"`` are reserved for the upcoming gaussx mixture-
quantile follow-up; instantiating them raises a clear
``NotImplementedError`` pointing at the tracking issue.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jaxtyping import Array, Float, PRNGKeyArray

from pyrox.api._estimator import EstimatorBase, FittedEstimator
from pyrox.nn._bnf import BayesianNeuralField
from pyrox.preprocessing import (
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


def _build_bnf(fit: SpatiotemporalFit, width: int, depth: int) -> BayesianNeuralField:
    """Construct the :class:`BayesianNeuralField` layer from a fit bundle."""
    d_in = len(fit.feature_cols)
    return BayesianNeuralField(  # ty: ignore[invalid-return-type]
        input_scales=tuple([1.0] * d_in),  # already standardized
        fourier_degrees=fit.fourier_layer.degrees,
        interactions=fit.interaction_layer.pairs,
        seasonality_periods=fit.seasonal_layer.periods,
        num_seasonal_harmonics=fit.seasonal_layer.harmonics,
        width=width,
        depth=depth,
        time_col=0,
        pyrox_name="bnf",
    )


def _bnf_init_fn(
    bnf_template: BayesianNeuralField,
    seed_key: PRNGKeyArray,
) -> dict[str, Array]:
    """Sample initial parameter values from the BNF prior at one PRNG key.

    Returns a flat dict ``{site_name: value}`` matching the sites
    registered by ``BayesianNeuralField.__call__``. The dict shape
    becomes the per-ensemble-member parameter PyTree carried by the
    ensemble runner.
    """
    from numpyro.handlers import seed, trace

    # Provide a tiny dummy input — the only thing we need is for the
    # layer to reach every `pyrox_sample` call site. Width 1 along the
    # batch axis is enough.
    d_in = len(bnf_template.input_scales)
    dummy_x = jnp.zeros((1, d_in))
    seeded = seed(bnf_template, seed_key)
    tr = trace(seeded).get_trace(dummy_x)
    # Extract the name → value map for sample sites.
    return {
        name: site["value"]
        for name, site in tr.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }


def _bnf_log_joint(
    bnf_template: BayesianNeuralField,
    sigma_obs: float,
) -> Callable[
    [dict[str, Array], Array, Array], tuple[Float[Array, ""], Float[Array, ""]]
]:
    """Build a (loglik, logprior) callable that takes a flat param dict."""
    from numpyro.handlers import substitute, trace

    def model_fn(x: Array, y: Array) -> None:
        f = bnf_template(x)
        numpyro.sample("y", dist.Normal(f, sigma_obs), obs=y)

    def log_joint(
        params: dict[str, Array], xb: Array, yb: Array
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        seeded = substitute(model_fn, params)
        tr = trace(seeded).get_trace(xb, yb)
        ll = jnp.zeros(())
        lp = jnp.zeros(())
        for site in tr.values():
            if site["type"] != "sample":
                continue
            contrib = site["fn"].log_prob(site["value"])
            scale_meta = site.get("scale")
            contrib = (
                contrib.sum() if scale_meta is None else (contrib.sum() * scale_meta)
            )
            if site.get("is_observed", False):
                ll = ll + contrib
            else:
                lp = lp + contrib
        return ll, lp

    return log_joint


class BNFEstimator(EstimatorBase):
    """MAP-fit Bayesian Neural Field estimator.

    Composes the BNF layer (:class:`pyrox.nn.BayesianNeuralField`) with
    pandas-side preprocessing (:func:`pyrox.preprocessing.fit_spatiotemporal`)
    and an ensemble-MAP inference loop
    (:func:`pyrox.inference.ensemble_map`).

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
        timetype: ``"int"`` or ``"datetime"``.
        freq: Optional unit string for ``datetime`` time columns.
        sigma_obs: Observation noise std (fixed in this MVP).
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
    timetype: Literal["int", "datetime"] = eqx.field(static=True, default="int")
    freq: str | None = eqx.field(static=True, default=None)
    sigma_obs: float = eqx.field(static=True, default=0.1)
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
        if isinstance(seed, int):
            seed = jr.PRNGKey(seed)
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
        bnf = _build_bnf(fit_bundle, self.width, self.depth)
        x_train = _df_to_design(df, fit_bundle, timetype=self.timetype, freq=self.freq)
        y_train = jnp.asarray(df[self.target_col].to_numpy(), dtype=jnp.float32)
        # 3. Run ensemble inference.
        if self.inference_kind == "map":
            from pyrox.inference import ensemble_map

            init_fn = lambda k: _bnf_init_fn(bnf, k)
            log_joint = _bnf_log_joint(bnf, self.sigma_obs)
            params, losses = ensemble_map(
                log_joint,
                init_fn,
                ensemble_size=self.ensemble_size,
                num_epochs=self.num_epochs,
                data=(x_train, y_train),
                seed=seed,
                learning_rate=self.learning_rate,
                prior_weight=self.prior_weight,
            )
        elif self.inference_kind == "vi":
            from numpyro.infer.autoguide import AutoNormal

            from pyrox.inference import ensemble_vi

            def model_fn(x: Array, y: Array | None = None) -> None:
                f = bnf(x)
                numpyro.sample("y", dist.Normal(f, self.sigma_obs), obs=y)

            guide_factory = self.guide_factory or AutoNormal
            guide = guide_factory(model_fn)
            params, losses = ensemble_vi(
                model_fn,
                guide,
                ensemble_size=self.ensemble_size,
                num_epochs=self.num_epochs,
                data=(x_train, y_train),
                seed=seed,
                learning_rate=self.learning_rate,
            )
        else:
            raise ValueError(f"Unknown inference_kind {self.inference_kind!r}")
        return FittedBNF(  # ty: ignore[invalid-return-type]
            config=self,
            fit_bundle=fit_bundle,
            params=params,
            losses=losses,
            bnf_template=bnf,
        )


class BNFEstimatorMLE(BNFEstimator):
    """BNF estimator with `prior_weight=0` (MLE)."""

    prior_weight: float = eqx.field(static=True, default=0.0)


class BNFEstimatorVI(BNFEstimator):
    """BNF estimator using mean-field VI via :func:`ensemble_vi`."""

    inference_kind: InferenceKind = eqx.field(static=True, default="vi")


class FittedBNF(FittedEstimator):
    """Output of :meth:`BNFEstimator.fit`.

    Attributes:
        config: The :class:`BNFEstimator` that produced this fit.
        fit_bundle: Pandas-side preprocessing record.
        params: Stacked per-ensemble-member parameter dict (each leaf
            has a leading ``(E,)`` axis for MAP, or an
            ``AutoNormal``-shaped variational-parameter dict for VI).
        losses: Per-member loss history, shape ``(E, num_epochs)``.
        bnf_template: The :class:`BayesianNeuralField` layer instance
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
        """Posterior-predictive mean (and optional quantiles).

        For MAP fits the per-member predictions are point estimates;
        for VI fits we draw ``num_posterior_samples`` weight samples
        per member from the variational guide.

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

        if cfg.inference_kind == "map":
            preds = _predict_map_ensemble(
                self.params, self.bnf_template, x_pred
            )  # (E, N)
            sigma_obs = float(cfg.sigma_obs)
            mean = preds.mean(axis=0)
            if quantiles is None:
                return mean
            qs = _gaussian_mixture_quantiles(
                preds,
                sigma_obs,
                quantiles=tuple(quantiles),
                seed=predict_seed,
                num_samples=max(num_posterior_samples * preds.shape[0], 200),
            )
            return mean, qs
        # VI path
        preds = _predict_vi_ensemble(
            self.params,
            self.bnf_template,
            x_pred,
            sigma_obs=float(cfg.sigma_obs),
            num_posterior_samples=num_posterior_samples,
            seed=predict_seed,
            guide_factory=cfg.guide_factory,
        )  # (E*S, N)
        mean = preds.mean(axis=0)
        if quantiles is None:
            return mean
        qs = _gaussian_mixture_quantiles(
            preds,
            float(cfg.sigma_obs),
            quantiles=tuple(quantiles),
            seed=predict_seed + 1,
            num_samples=max(num_posterior_samples * preds.shape[0], 200),
        )
        return mean, qs


def _predict_map_ensemble(
    params: dict[str, Array],
    bnf_template: BayesianNeuralField,
    x: Float[Array, "N D"],
) -> Float[Array, "E N"]:
    """vmapped deterministic forward pass over MAP-stacked params."""
    from numpyro.handlers import substitute

    def _per_member(member_params: dict[str, Array]) -> Float[Array, " N"]:
        seeded = substitute(bnf_template, member_params)
        return seeded(x)

    return jax.vmap(_per_member)(params)


def _predict_vi_ensemble(
    guide_params: Any,
    bnf_template: BayesianNeuralField,
    x: Float[Array, "N D"],
    *,
    sigma_obs: float,
    num_posterior_samples: int,
    seed: int,
    guide_factory: Callable[[Callable], Any] | None = None,
) -> Float[Array, "ES N"]:
    """Per-member: draw ``S`` samples from the variational guide, push through.

    Returns the stacked predictions with leading axis ``E * S``. The
    ``guide_factory`` must match what was passed to ``BNFEstimator(VI)``;
    otherwise the guide's expected param shapes won't line up.
    """
    from numpyro.infer import Predictive
    from numpyro.infer.autoguide import AutoNormal

    def _model(x: Array, y: Array | None = None) -> None:
        f = bnf_template(x)
        numpyro.sample("y", dist.Normal(f, sigma_obs), obs=y)

    factory = guide_factory or AutoNormal
    guide = factory(_model)

    def _per_member(member_params: Any, key: PRNGKeyArray) -> Float[Array, "S N"]:
        pred = Predictive(
            _model,
            guide=guide,
            params=member_params,
            num_samples=num_posterior_samples,
        )
        out = pred(key, x, None)
        return out["y"]  # (S, N)

    e = jax.tree.leaves(guide_params)[0].shape[0]
    keys = jr.split(jr.PRNGKey(seed), e)
    preds = jax.vmap(_per_member)(guide_params, keys)  # (E, S, N)
    return preds.reshape(-1, x.shape[0])


def _gaussian_mixture_quantiles(
    means: Float[Array, "M N"],
    sigma: float,
    *,
    quantiles: tuple[float, ...],
    seed: int,
    num_samples: int,
) -> tuple[Float[Array, " N"], ...]:
    """MC-estimate per-row quantiles of an equal-weight Gaussian mixture.

    Each component has mean ``means[m, n]`` and standard deviation
    ``sigma``. Drawing ``num_samples`` mixture samples per row is
    equivalent to first picking a uniform component index and then
    sampling from that component's Gaussian.
    """
    m, n = means.shape
    rk = jr.PRNGKey(seed)
    k_idx, k_eps = jr.split(rk)
    component_idx = jr.randint(k_idx, (num_samples,), 0, m)
    eps = jr.normal(k_eps, (num_samples, n))
    samples = means[component_idx] + sigma * eps  # (S, N)
    qs = tuple(jnp.quantile(samples, q, axis=0) for q in quantiles)
    return qs
