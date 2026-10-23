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

from pyrox_nn._bnf import BayesianNeuralField
from pyrox_nn.api._estimator import EstimatorBase, FittedEstimator
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


def _bnf_model(
    bnf_template: BayesianNeuralField,
    sigma_obs: float,
    *,
    noise: NoiseModel = "fixed",
    has_sigma_col: bool = False,
) -> Callable[[Array, Array | None], None]:
    r"""The BNF regression model: field, observation noise, likelihood.

    $$
    y_i \sim \mathcal N\bigl(f_\theta(x_i),\ \sigma_i^2 + \sigma_0^2\bigr),
    $$

    with $\sigma_i = 0$ without a ``sigma_col`` and $\sigma_0$ either fixed
    at ``sigma_obs`` (``noise="fixed"``) or learned through the site
    ``sigma_obs`` $= \log \sigma_0 \sim \mathcal N(\log \sigma_{\text{obs}}, 1)$
    (``noise="learned"``). With ``has_sigma_col`` the observations are packed
    as ``y = [y_i, σ_i]`` of shape ``(N, 2)``, so the per-row noise travels
    with its row through the ensemble runners' mini-batching. The latent
    field is exposed as the deterministic site ``f``.
    """

    def model_fn(x: Array, y: Array | None = None) -> None:
        f = bnf_template(x)
        numpyro.deterministic("f", f)
        if noise == "learned":
            log_sigma0 = numpyro.sample(
                "sigma_obs", dist.Normal(jnp.log(sigma_obs), 1.0)
            )
            sigma0 = jnp.exp(log_sigma0)
        else:
            sigma0 = sigma_obs
        obs, sigma_i = y, None
        if has_sigma_col and y is not None:
            obs, sigma_i = y[..., 0], y[..., 1]
        scale = sigma0 if sigma_i is None else jnp.sqrt(sigma_i**2 + sigma0**2)
        numpyro.sample("y", dist.Normal(f, scale), obs=obs)

    return model_fn


def _bnf_init_fn(
    model_fn: Callable[[Array, Array | None], None],
    d_in: int,
    y_shape: tuple[int, ...],
    seed_key: PRNGKeyArray,
) -> dict[str, Array]:
    """Sample initial parameter values from the model's prior at one PRNG key.

    Returns a flat dict ``{site_name: value}`` of the unobserved sample sites
    (the BNF weights, plus ``sigma_obs`` when the noise is learned). The dict
    shape becomes the per-ensemble-member parameter PyTree carried by the
    ensemble runner.
    """
    from numpyro.handlers import seed, trace

    # One dummy row reaches every sample site; the observation is given so
    # that "y" is observed (and consumes no PRNG key).
    dummy_x = jnp.zeros((1, d_in))
    dummy_y = jnp.ones((1, *y_shape))
    tr = trace(seed(model_fn, seed_key)).get_trace(dummy_x, dummy_y)
    return {
        name: site["value"]
        for name, site in tr.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }


def _bnf_log_joint(
    model_fn: Callable[[Array, Array | None], None],
) -> Callable[
    [dict[str, Array], Array, Array], tuple[Float[Array, ""], Float[Array, ""]]
]:
    """Build a (loglik, logprior) callable that takes a flat param dict."""
    from numpyro.handlers import substitute, trace

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
            ``sigma_obs`` $= \log \sigma_0 \sim \mathcal N(\log
            \sigma_{\text{obs}}, 1)$, which the MAP and VI ensembles treat
            like any other latent.
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
            raise ValueError(f"noise must be 'fixed' or 'learned'; got {self.noise!r}.")
        if not self.sigma_obs > 0:
            raise ValueError(f"sigma_obs must be > 0; got {self.sigma_obs}.")
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
        bnf = _build_bnf(
            fit_bundle, self.width, self.depth, self.encoder, self.encoder_cols
        )
        x_train = _df_to_design(df, fit_bundle, timetype=self.timetype, freq=self.freq)
        y_train = jnp.asarray(df[self.target_col].to_numpy(), dtype=jnp.float32)
        if self.sigma_col is not None:
            sigma_i = _sigma_column(df, self.sigma_col)
            y_train = jnp.stack([y_train, sigma_i], axis=-1)  # (N, 2)
        model_fn = _bnf_model(
            bnf,
            self.sigma_obs,
            noise=self.noise,
            has_sigma_col=self.sigma_col is not None,
        )
        # 3. Run ensemble inference.
        if self.inference_kind == "map":
            from pyrox.inference import ensemble_map

            d_in = len(self.feature_cols)
            y_shape = y_train.shape[1:]
            init_fn = lambda k: _bnf_init_fn(model_fn, d_in, y_shape, k)
            log_joint = _bnf_log_joint(model_fn)
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
        return FittedBNF(
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
        model_fn = _bnf_model(
            self.bnf_template,
            cfg.sigma_obs,
            noise=cfg.noise,
            has_sigma_col=cfg.sigma_col is not None,
        )

        if cfg.inference_kind == "map":
            preds = _predict_map_ensemble(
                self.params, self.bnf_template, x_pred
            )  # (E, N)
            if cfg.noise == "learned":
                sigma0 = jnp.exp(self.params["sigma_obs"])  # (E,)
            else:
                sigma0 = jnp.full((preds.shape[0],), float(cfg.sigma_obs))
            seed = predict_seed
        else:
            preds, sigma0 = _predict_vi_ensemble(
                self.params,
                model_fn,
                x_pred,
                learned_noise=cfg.noise == "learned",
                sigma_obs=float(cfg.sigma_obs),
                num_posterior_samples=num_posterior_samples,
                seed=predict_seed,
                guide_factory=cfg.guide_factory,
            )  # (E*S, N), (E*S,)
            seed = predict_seed + 1
        mean = preds.mean(axis=0)
        if quantiles is None:
            return mean
        qs = _gaussian_mixture_quantiles(
            preds,
            sigma0,
            quantiles=tuple(quantiles),
            seed=seed,
            num_samples=max(num_posterior_samples * preds.shape[0], 200),
        )
        return mean, qs


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
    model_fn: Callable[[Array, Array | None], None],
    x: Float[Array, "N D"],
    *,
    learned_noise: bool,
    sigma_obs: float,
    num_posterior_samples: int,
    seed: int,
    guide_factory: Callable[[Callable], Any] | None = None,
) -> tuple[Float[Array, "ES N"], Float[Array, " ES"]]:
    r"""Per-member: draw ``S`` samples from the variational guide, push through.

    Returns the latent predictions with leading axis ``E * S`` and each
    draw's noise scale $\sigma_0$. The ``guide_factory`` must match what was
    passed to ``BNFEstimator(VI)``; otherwise the guide's expected param
    shapes won't line up.
    """
    from numpyro.infer import Predictive
    from numpyro.infer.autoguide import AutoNormal

    # `model_fn` exposes the latent as the deterministic site "f", so
    # `Predictive` returns noise-free ensemble samples; `out["y"]` would
    # already include one noise draw and `_gaussian_mixture_quantiles` —
    # which adds sigma_0 itself — would double-count observation noise.
    factory = guide_factory or AutoNormal
    guide = factory(model_fn)
    sites = ("f", "sigma_obs") if learned_noise else ("f",)

    def _per_member(
        member_params: Any, key: PRNGKeyArray
    ) -> tuple[Float[Array, "S N"], Float[Array, " S"]]:
        pred = Predictive(
            model_fn,
            guide=guide,
            params=member_params,
            num_samples=num_posterior_samples,
            return_sites=sites,
        )
        out = pred(key, x, None)
        if learned_noise:
            sigma0 = jnp.exp(out["sigma_obs"])
        else:
            sigma0 = jnp.full((num_posterior_samples,), sigma_obs)
        return out["f"], sigma0

    e = jax.tree.leaves(guide_params)[0].shape[0]
    keys = jr.split(jr.PRNGKey(seed), e)
    preds, sigma0 = jax.vmap(_per_member)(guide_params, keys)  # (E, S, N), (E, S)
    return preds.reshape(-1, x.shape[0]), sigma0.reshape(-1)


def _gaussian_mixture_quantiles(
    means: Float[Array, "M N"],
    sigma: float | Float[Array, " M"],
    *,
    quantiles: tuple[float, ...],
    seed: int,
    num_samples: int,
) -> tuple[Float[Array, " N"], ...]:
    """MC-estimate per-row quantiles of an equal-weight Gaussian mixture.

    Component ``m`` has mean ``means[m, n]`` and standard deviation
    ``sigma`` (a scalar, or ``sigma[m]`` per component). Drawing
    ``num_samples`` mixture samples per row is equivalent to first picking
    a uniform component index and then sampling from that component's
    Gaussian.
    """
    m, n = means.shape
    sigma_m = jnp.broadcast_to(jnp.asarray(sigma), (m,))
    rk = jr.PRNGKey(seed)
    k_idx, k_eps = jr.split(rk)
    component_idx = jr.randint(k_idx, (num_samples,), 0, m)
    eps = jr.normal(k_eps, (num_samples, n))
    samples = means[component_idx] + sigma_m[component_idx][:, None] * eps  # (S, N)
    qs = tuple(jnp.quantile(samples, q, axis=0) for q in quantiles)
    return qs
