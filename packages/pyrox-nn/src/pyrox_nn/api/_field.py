r"""``NeuralFieldEstimator`` — fit any Bayesian neural field with ensembles.

A neural field maps coordinates to a scalar field. This module fits one
given as a NumPyro-site-carrying callable (a `PyroxModule` such as
`pyrox_nn.BayesianSIREN`, `pyrox_nn.SphericalHarmonicFeatures` or
`pyrox_nn.BayesianNeuralField`) to observations with the ensemble runners
of `pyrox.inference`, on plain arrays (with an optional validity mask for
padded batches) or on an xarray Dataset.

The model, per ensemble member with latent sites $\theta$, is

$$
y_i \sim \mathcal N\bigl(f_\theta(\gamma(x_i)),\ \sigma_i^2 + \sigma_0^2\bigr),
$$

with $\gamma$ an optional deterministic encoder, $\sigma_i$ optional
per-row noise and $\sigma_0$ fixed, learned, or input-dependent. The MAP
objective is

$$
\log p(\theta, y) = \sum_i m_i \log \mathcal N\bigl(y_i \mid
f_\theta(\gamma(x_i)),\ \sigma_i^2 + \sigma_0^2\bigr) + w \log p(\theta),
$$

with the validity mask $m_i \in \{0, 1\}$ and $w$ = ``prior_weight``.

`pyrox_nn.api.BNFEstimator` is this estimator configured with a
`BayesianNeuralField` and pandas preprocessing. This module imports
neither pandas nor xarray at module scope; ``fit_xarray`` imports xarray
when called (``pip install pyrox-nn[xarray]``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray
from numpyro.primitives import Messenger


NoiseModel = Literal["fixed", "learned", "heteroscedastic"]
InferenceKind = Literal["map", "vi"]

FieldOutput = Float[Array, " N"] | tuple[Float[Array, " N"], Float[Array, " N"]]
Prediction = (
    Float[Array, " N"] | tuple[Float[Array, " N"], tuple[Float[Array, " N"], ...]]
)


# -- the model ----------------------------------------------------------------


def _as_target(out: Array) -> Array:
    """Accept a field output of shape ``(N,)`` or ``(N, 1)``; return ``(N,)``."""
    if out.ndim == 1:
        return out
    if out.ndim == 2 and out.shape[-1] == 1:
        return out[:, 0]
    raise ValueError(
        "a neural field for NeuralFieldEstimator must return one target per "
        f"row, of shape (N,) or (N, 1); got {out.shape}."
    )


def _make_forward(
    field: Callable[[Array], Any],
    encoder: Callable[[Array], Array] | None,
) -> Callable[[Array], FieldOutput]:
    """Compose the per-row encoder, the field and the target-shape check."""

    def forward(x: Float[Array, "N D"]) -> FieldOutput:
        z = x if encoder is None else jax.vmap(encoder)(x)
        out = field(z)
        if isinstance(out, tuple):
            if len(out) != 2:
                raise ValueError(
                    "a heteroscedastic field must return (mu, log_sigma); got a "
                    f"tuple of {len(out)}."
                )
            return _as_target(out[0]), _as_target(out[1])
        return _as_target(out)

    return forward


def _field_model(
    forward: Callable[[Array], Any],
    sigma_obs: float,
    *,
    noise: NoiseModel = "fixed",
    has_sigma: bool = False,
    has_valid: bool = False,
) -> Callable[[Array, Array | None], None]:
    r"""The regression model: field, observation noise, masked likelihood.

    $$
    y_i \sim \mathcal N\bigl(f_\theta(x_i),\ \sigma_i^2 + \sigma_0^2\bigr),
    $$

    with $\sigma_i = 0$ without per-row noise and $\sigma_0$ fixed at
    ``sigma_obs`` (``noise="fixed"``), learned through the site
    ``log_sigma_obs`` $= \log \sigma_0 \sim \mathcal N(\log
    \sigma_{\text{obs}}, 1)$ (``noise="learned"``), or the field's own
    $\sigma(x_i)$ when ``forward`` returns ``(mu, log_sigma)``
    (``noise="heteroscedastic"``). Per-row noise and the validity mask travel
    with their rows as extra columns of ``y``: ``[y, σ]`` with
    ``has_sigma``, then ``m`` with ``has_valid``. Masked rows contribute
    $\log p = 0$ through ``Distribution.mask``. The latent field is the
    deterministic site ``f`` (and ``log_sigma_f`` when heteroscedastic).
    """

    def model_fn(x: Array, y: Array | None = None) -> None:
        out = forward(x)
        if (noise == "heteroscedastic") != isinstance(out, tuple):
            raise ValueError(
                "noise='heteroscedastic' needs a field returning (mu, log_sigma), "
                "and only then; got noise="
                f"{noise!r} with a field returning a {type(out).__name__}."
            )
        if noise == "heteroscedastic":
            f, log_sigma_f = out
            numpyro.deterministic("f", f)
            numpyro.deterministic("log_sigma_f", log_sigma_f)
            sigma0 = jnp.exp(log_sigma_f)
        else:
            f = out
            numpyro.deterministic("f", f)
            if noise == "learned":
                log_sigma0 = numpyro.sample(
                    "log_sigma_obs", dist.Normal(jnp.log(sigma_obs), 1.0)
                )
                sigma0 = jnp.exp(log_sigma0)
            else:
                sigma0 = sigma_obs
        obs, sigma_i, valid = y, None, None
        if y is not None and (has_sigma or has_valid):
            obs, col = y[..., 0], 1
            if has_sigma:
                sigma_i, col = y[..., col], col + 1
            if has_valid:
                valid = y[..., col] > 0.5
        scale = sigma0 if sigma_i is None else jnp.sqrt(sigma_i**2 + sigma0**2)
        likelihood = dist.Normal(f, scale)
        if valid is not None:
            likelihood = likelihood.mask(valid)
        numpyro.sample("y", likelihood, obs=obs)

    return model_fn


def _field_init_fn(
    model_fn: Callable[[Array, Array | None], None],
    d_in: int,
    y_shape: tuple[int, ...],
    seed_key: PRNGKeyArray,
    param_constraints: dict[str, Any] | None = None,
) -> dict[str, Array]:
    """Sample initial parameter values from the model's prior at one PRNG key.

    Returns a flat dict ``{site_name: value}`` of the unobserved sample sites
    (the field's weights, plus ``log_sigma_obs`` when the noise is learned)
    and the param sites; each name in ``param_constraints`` (param sites and
    constrained latents) holds its initial value mapped to unconstrained
    space. The dict becomes the
    per-ensemble-member parameter PyTree carried by the ensemble runner.
    """
    from numpyro.distributions.transforms import biject_to
    from numpyro.handlers import seed, trace

    # One dummy row reaches every sample site; the observation is given so
    # that "y" is observed (and consumes no PRNG key).
    dummy_x = jnp.zeros((1, d_in))
    dummy_y = jnp.ones((1, *y_shape))
    tr = trace(seed(model_fn, seed_key)).get_trace(dummy_x, dummy_y)
    params = {
        name: site["value"]
        for name, site in tr.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }
    for name, constraint in (param_constraints or {}).items():
        params[name] = biject_to(constraint).inv(tr[name]["value"])
    return params


def _field_log_joint(
    model_fn: Callable[[Array, Array | None], None],
    param_constraints: dict[str, Any] | None = None,
) -> Callable[
    [dict[str, Array], Array, Array], tuple[Float[Array, ""], Float[Array, ""]]
]:
    """Build a ``(loglik, logprior)`` callable that takes a flat param dict.

    Names in ``param_constraints`` (param sites and constrained latents) are
    held in unconstrained space: they are mapped through their constraint's
    bijection and substituted. The prior is evaluated on the constrained
    value with no log-det-Jacobian, so the optimum is the constrained-space
    MAP; a param site has no prior and adds nothing to ``logprior``.
    """
    from numpyro.distributions.transforms import biject_to
    from numpyro.handlers import substitute, trace

    constraints = param_constraints or {}

    def log_joint(
        params: dict[str, Array], xb: Array, yb: Array
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        if constraints:
            params = {
                name: biject_to(constraints[name])(value)
                if name in constraints
                else value
                for name, value in params.items()
            }
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


class _PrngKeyProbe(Messenger):
    """Records whether the wrapped model draws ``numpyro.prng_key()``."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.used = False
        super().__init__(fn)

    def process_message(self, msg: dict[str, Any]) -> None:
        if msg["type"] == "prng_key":
            self.used = True


def _map_param_constraints(
    model_fn: Callable[[Array, Array | None], None],
    d_in: int,
    y_shape: tuple[int, ...],
) -> dict[str, Any]:
    r"""Check a model for the MAP path; return the sites to reparameterise.

    MAP optimises the param sites (``pyrox_param``) and the latent sample
    sites together, each in unconstrained space through
    $T = $ ``biject_to(constraint)`` (the param's constraint, the prior's
    support), so Adam's additive steps never leave the support. The
    objective stays the *constrained-space* density $\log p(T(u))$, with no
    log-det-Jacobian: its maximiser maps to the usual MAP point, which a
    Jacobian term would move. Sample sites with real support (an identity
    $T$) are left out of the map, so their values pass through untouched.
    A field that draws ``numpyro.prng_key()`` (Monte Carlo layers such as
    `pyrox_nn.DenseVariationalDropout`) has a random log joint, which is not
    a MAP objective, so it is rejected; so is a discrete latent.
    """
    from numpyro.distributions import constraints
    from numpyro.distributions.transforms import (
        IdentityTransform,
        IndependentTransform,
        biject_to,
    )
    from numpyro.handlers import seed, trace

    def is_identity(t: Any) -> bool:
        # biject_to(real.to_event(k)) wraps the identity in IndependentTransform.
        while isinstance(t, IndependentTransform):
            t = t.base_transform
        return isinstance(t, IdentityTransform)

    probe = _PrngKeyProbe(seed(model_fn, jr.PRNGKey(0)))
    tr = trace(probe).get_trace(jnp.zeros((1, d_in)), jnp.ones((1, *y_shape)))
    if probe.used:
        raise ValueError(
            "this field draws numpyro.prng_key() (a Monte-Carlo layer such as "
            "DenseVariationalDropout), so its log joint is random and has no MAP "
            "point; use inference='vi', or a field without internal sampling."
        )
    out: dict[str, Any] = {}
    for name, site in tr.items():
        if site["type"] == "param":
            out[name] = site["kwargs"].get("constraint", constraints.real)
        elif site["type"] == "sample" and not site.get("is_observed", False):
            support = site["fn"].support
            if support.is_discrete:
                raise ValueError(
                    f"the latent site {name!r} is discrete; MAP needs continuous "
                    "latents."
                )
            if not is_identity(biject_to(support)):
                out[name] = support
    return out


# -- fitting --------------------------------------------------------------------


def _check_config(
    *,
    noise: str,
    sigma_obs: float,
    inference: str,
    prior_weight: float,
    has_sigma: bool,
) -> None:
    """Validate the estimator's configuration (concrete Python values)."""
    if noise not in ("fixed", "learned", "heteroscedastic"):
        raise ValueError(
            f"noise must be 'fixed', 'learned' or 'heteroscedastic'; got {noise!r}."
        )
    if not sigma_obs > 0:
        raise ValueError(f"sigma_obs must be > 0; got {sigma_obs}.")
    if inference not in ("map", "vi"):
        raise ValueError(f"Unknown inference kind {inference!r}; use 'map' or 'vi'.")
    if (
        noise == "learned"
        and not has_sigma
        and inference == "map"
        and (prior_weight == 0)
    ):
        # Without the prior on log sigma_0 and without per-row sigma_i,
        # nothing bounds sigma_0 below: a network that interpolates the
        # data drives it to 0 and the likelihood to +inf.
        raise ValueError(
            "noise='learned' needs the prior on log_sigma_obs (prior_weight "
            "> 0) or a sigma_col to bound sigma_0 below; with "
            "prior_weight=0 (MLE) use noise='fixed'."
        )


def _floating(a: ArrayLike) -> Array:
    """``jnp.asarray`` keeping a floating dtype; integers become float32."""
    a = jnp.asarray(a)
    return a if jnp.issubdtype(a.dtype, jnp.floating) else a.astype(jnp.float32)


def _pack_data(
    X: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike | None,
    valid: ArrayLike | None,
) -> tuple[Float[Array, "N D"], Float[Array, "N ..."]]:
    """Validate the arrays and pack ``y`` with its per-row columns.

    Returns ``x`` of shape ``(N, D)`` and ``y`` of shape ``(N,)``, or
    ``(N, K)`` with the columns ``[y, σ?, m?]``. Masked rows are overwritten
    with the first valid row, so junk values (NaN, inf) in padding never
    reach the field or its gradients.
    """
    x = _floating(X)
    if x.ndim != 2:
        raise ValueError(f"X must have shape (N, D); got {x.shape}.")
    n = x.shape[0]
    y_arr = jnp.asarray(y, dtype=x.dtype)
    if y_arr.shape != (n,):
        raise ValueError(f"y must have shape ({n},) to match X; got {y_arr.shape}.")
    sigma_arr = None if sigma is None else jnp.asarray(sigma, dtype=x.dtype)
    if sigma_arr is not None and sigma_arr.shape != (n,):
        raise ValueError(
            f"sigma must have shape ({n},) to match X; got {sigma_arr.shape}."
        )
    m = None
    if valid is not None:
        m = np.asarray(valid)
        if m.shape != (n,) or m.dtype != np.bool_:
            raise ValueError(
                f"valid must be a boolean array of shape ({n},); got "
                f"{m.dtype} {m.shape}."
            )
        if not m.any():
            raise ValueError("valid masks out every row; nothing to fit.")
    rows = np.ones(n, dtype=bool) if m is None else m
    if not np.isfinite(np.asarray(x)[rows]).all():
        raise ValueError("X must be finite on every valid row.")
    if not np.isfinite(np.asarray(y_arr)[rows]).all():
        raise ValueError("y must be finite on every valid row.")
    if sigma_arr is not None:
        s = np.asarray(sigma_arr)[rows]
        if not ((s > 0) & np.isfinite(s)).all():
            raise ValueError(
                "sigma must be finite and > 0 on every valid row; got "
                f"{int((~((s > 0) & np.isfinite(s))).sum())} bad values."
            )
    cols = [y_arr]
    if m is not None:
        first = int(np.argmax(m))
        mask = jnp.asarray(m)
        x = jnp.where(mask[:, None], x, x[first])
        cols = [jnp.where(mask, y_arr, y_arr[first])]
        if sigma_arr is not None:
            sigma_arr = jnp.where(mask, sigma_arr, sigma_arr[first])
    if sigma_arr is not None:
        cols.append(sigma_arr)
    if m is not None:
        cols.append(jnp.asarray(m, dtype=x.dtype))
    y_packed = cols[0] if len(cols) == 1 else jnp.stack(cols, axis=-1)
    return x, y_packed


def _fit_ensemble(
    model_fn: Callable[[Array, Array | None], None],
    x: Float[Array, "N D"],
    y: Float[Array, "N ..."],
    *,
    inference: InferenceKind,
    ensemble_size: int,
    num_epochs: int,
    learning_rate: float,
    prior_weight: float,
    guide_factory: Callable[[Callable], Any] | None,
    seed: PRNGKeyArray,
) -> tuple[Any, Float[Array, "E T"]]:
    """Run ``ensemble_map`` or ``ensemble_vi`` on the packed data."""
    if inference == "map":
        from numpyro.distributions.transforms import biject_to
        from pyrox.inference import ensemble_map

        d_in, y_shape = x.shape[1], y.shape[1:]
        param_constraints = _map_param_constraints(model_fn, d_in, y_shape)
        params, losses = ensemble_map(
            _field_log_joint(model_fn, param_constraints),
            lambda k: _field_init_fn(model_fn, d_in, y_shape, k, param_constraints),
            ensemble_size=ensemble_size,
            num_epochs=num_epochs,
            data=(x, y),
            seed=seed,
            learning_rate=learning_rate,
            prior_weight=prior_weight,
        )
        # Hand back the reparameterised sites in constrained space, per
        # member, so the fitted dict substitutes directly into the field.
        for name, constraint in param_constraints.items():
            params[name] = jax.vmap(biject_to(constraint))(params[name])
        return params, losses
    from numpyro.infer.autoguide import AutoNormal
    from pyrox.inference import ensemble_vi

    guide = (guide_factory or AutoNormal)(model_fn)
    return ensemble_vi(
        model_fn,
        guide,
        ensemble_size=ensemble_size,
        num_epochs=num_epochs,
        data=(x, y),
        seed=seed,
        learning_rate=learning_rate,
    )


# -- prediction -------------------------------------------------------------------


def _predict_map_ensemble(
    params: dict[str, Array],
    forward: Callable[[Array], Any],
    x: Float[Array, "N D"],
) -> Any:
    """vmapped deterministic forward pass over MAP-stacked params.

    Returns ``(E, N)``, or ``((E, N), (E, N))`` for a heteroscedastic field.
    """
    from numpyro.handlers import substitute

    def _per_member(member_params: dict[str, Array]) -> Any:
        return substitute(forward, member_params)(x)

    return jax.vmap(_per_member)(params)


def _predict_vi_ensemble(
    guide_params: Any,
    model_fn: Callable[[Array, Array | None], None],
    x: Float[Array, "N D"],
    *,
    noise: NoiseModel,
    sigma_obs: float,
    num_posterior_samples: int,
    seed: int,
    guide_factory: Callable[[Callable], Any] | None = None,
) -> tuple[Float[Array, "ES N"], Float[Array, " ES"] | Float[Array, "ES N"]]:
    r"""Per-member: draw ``S`` samples from the variational guide, push through.

    Returns the latent predictions with leading axis ``E * S`` and each
    draw's noise scale $\sigma_0$ (``(E * S,)``, or ``(E * S, N)`` for a
    heteroscedastic field). The ``guide_factory`` must match the one used to
    fit; otherwise the guide's expected param shapes won't line up.
    """
    from numpyro.infer import Predictive
    from numpyro.infer.autoguide import AutoNormal

    # `model_fn` exposes the latent as the deterministic site "f", so
    # `Predictive` returns noise-free ensemble samples; `out["y"]` would
    # already include one noise draw and `_gaussian_mixture_quantiles` —
    # which adds sigma_0 itself — would double-count observation noise.
    factory = guide_factory or AutoNormal
    guide = factory(model_fn)
    noise_site = {"learned": "log_sigma_obs", "heteroscedastic": "log_sigma_f"}.get(
        noise
    )
    sites = ("f",) if noise_site is None else ("f", noise_site)

    def _per_member(member_params: Any, key: PRNGKeyArray) -> tuple[Array, Array]:
        pred = Predictive(
            model_fn,
            guide=guide,
            params=member_params,
            num_samples=num_posterior_samples,
            return_sites=sites,
        )
        out = pred(key, x, None)
        if noise_site is None:
            sigma0 = jnp.full((num_posterior_samples,), sigma_obs)
        else:
            sigma0 = jnp.exp(out[noise_site])
        return out["f"], sigma0

    e = jax.tree.leaves(guide_params)[0].shape[0]
    keys = jr.split(jr.PRNGKey(seed), e)
    preds, sigma0 = jax.vmap(_per_member)(guide_params, keys)  # (E, S, N), (E, S…)
    return preds.reshape(-1, x.shape[0]), sigma0.reshape(-1, *sigma0.shape[2:])


def _gaussian_mixture_quantiles(
    means: Float[Array, "M N"],
    sigma: float | Float[Array, " M"] | Float[Array, "M N"],
    *,
    quantiles: tuple[float, ...],
    seed: int,
    num_samples: int,
) -> tuple[Float[Array, " N"], ...]:
    """MC-estimate per-row quantiles of an equal-weight Gaussian mixture.

    Component ``m`` has mean ``means[m, n]`` and standard deviation
    ``sigma`` (a scalar, ``sigma[m]`` per component, or ``sigma[m, n]`` per
    component and row). Drawing ``num_samples`` mixture samples per row is
    equivalent to first picking a uniform component index and then sampling
    from that component's Gaussian.
    """
    m, n = means.shape
    sigma_arr = jnp.asarray(sigma)
    if sigma_arr.ndim < 2:
        sigma_arr = jnp.broadcast_to(sigma_arr, (m,))[:, None]  # (M, 1)
    rk = jr.PRNGKey(seed)
    k_idx, k_eps = jr.split(rk)
    component_idx = jr.randint(k_idx, (num_samples,), 0, m)
    eps = jr.normal(k_eps, (num_samples, n))
    samples = means[component_idx] + sigma_arr[component_idx] * eps  # (S, N)
    return tuple(jnp.quantile(samples, q, axis=0) for q in quantiles)


def _predict(
    params: Any,
    field: Callable[[Array], Any],
    x: Float[Array, "N D"],
    *,
    encoder: Callable[[Array], Array] | None,
    noise: NoiseModel,
    sigma_obs: float,
    has_sigma: bool,
    inference: InferenceKind,
    guide_factory: Callable[[Callable], Any] | None,
    quantiles: Sequence[float] | None,
    predict_seed: int,
    num_posterior_samples: int,
    min_quantile_samples: int = 2000,
) -> Prediction:
    """Posterior-predictive mean and optional mixture quantiles (shared).

    The quantiles use at least ``min_quantile_samples`` mixture draws per
    row: the Monte-Carlo sd of a 5 % quantile from $S$ draws is
    $\\sqrt{0.05 \\cdot 0.95 / S} / \\phi(1.645) \\approx 2.1/\\sqrt{S}$ in
    units of $\\sigma$, 0.15σ at $S = 200$ and 0.05σ at the default 2000.
    """
    forward = _make_forward(field, encoder)
    if inference == "map":
        out = _predict_map_ensemble(params, forward, x)
        if noise == "heteroscedastic":
            preds, log_sigma_f = out
            sigma0 = jnp.exp(log_sigma_f)  # (E, N)
        else:
            preds = out
            if noise == "learned":
                sigma0 = jnp.exp(params["log_sigma_obs"])  # (E,)
            else:
                sigma0 = jnp.full((preds.shape[0],), float(sigma_obs))
        seed = predict_seed
    else:
        model_fn = _field_model(forward, sigma_obs, noise=noise, has_sigma=has_sigma)
        preds, sigma0 = _predict_vi_ensemble(
            params,
            model_fn,
            x,
            noise=noise,
            sigma_obs=float(sigma_obs),
            num_posterior_samples=num_posterior_samples,
            seed=predict_seed,
            guide_factory=guide_factory,
        )  # (E*S, N), (E*S,) or (E*S, N)
        seed = predict_seed + 1
    mean = preds.mean(axis=0)
    if quantiles is None:
        return mean
    qs = _gaussian_mixture_quantiles(
        preds,
        sigma0,
        quantiles=tuple(quantiles),
        seed=seed,
        num_samples=max(num_posterior_samples * preds.shape[0], min_quantile_samples),
    )
    return mean, qs


# -- the estimator ----------------------------------------------------------------


class NeuralFieldEstimator(eqx.Module):
    r"""Fit any Bayesian neural field with ensemble MAP or VI.

    The field is built by ``model_fn()``: a callable mapping a batch of
    (encoded) coordinates ``(N, D_enc)`` to the field ``(N,)`` or
    ``(N, 1)`` while registering its weights as NumPyro sample sites,
    usually a `PyroxModule` such as `pyrox_nn.BayesianSIREN`,
    `pyrox_nn.SphericalHarmonicFeatures` or
    `pyrox_nn.BayesianNeuralField`. Each ensemble member maximises (MAP) or
    approximates (VI) the posterior of

    $$
    y_i \sim \mathcal N\bigl(f_\theta(\gamma(x_i)),\
        \sigma_i^2 + \sigma_0^2\bigr),
    $$

    where $\gamma$ is ``encoder`` (identity when ``None``), $\sigma_i$ the
    optional per-row noise passed to ``fit`` and $\sigma_0$ set by
    ``noise``. Rows with ``valid[i] = False`` contribute nothing to the log
    joint or its gradient, so padded batches fit exactly like their valid
    rows alone.

    Sites: the field's own, plus ``log_sigma_obs`` with
    ``noise="learned"``; the likelihood site ``y`` and the deterministic
    sites ``f`` (and ``log_sigma_f`` with ``noise="heteroscedastic"``).

    Attributes:
        model_fn: Zero-argument factory returning the field. Called once per
            ``fit``; the returned module's sites must not depend on the data.
        encoder: Optional deterministic single-row encoder
            $\gamma: (D,) \to (D_\mathrm{enc},)$ applied to every row before
            the field (vmapped), e.g.
            ``geonnax.SphericalHarmonicEncoder(l_max=20, input_mode="lonlat")``
            (lon/lat in radians).
        noise: ``"fixed"`` keeps $\sigma_0$ at ``sigma_obs``; ``"learned"``
            infers it through the site ``log_sigma_obs`` $= \log\sigma_0 \sim
            \mathcal N(\log\sigma_\mathrm{obs}, 1)$; ``"heteroscedastic"``
            uses the field's own input-dependent scale: the field must
            return ``(mu, log_sigma)``, e.g. a trunk followed by
            `pyrox_nn.GaussianRegressionHead`, and $\sigma_0 = \sigma(x_i)$.
            MAP members start from prior draws, and the head clips
            $\log\sigma$ to $[-7, 3]$ with zero gradient outside, so keep its
            ``log_scale_prior_std`` small (the default 0.1): with 1.0 and a
            SIREN trunk, members start past the clip and never leave it.
            With ``prior_weight=0`` (MLE), ``"learned"`` and no ``sigma``,
            nothing bounds $\sigma_0$ below, so ``fit`` raises.
        sigma_obs: $\sigma_0$ for ``noise="fixed"``, or the prior median of
            $\sigma_0$ for ``noise="learned"``; unused when heteroscedastic.
        ensemble_size: Number of ensemble members $E$.
        inference: ``"map"`` (``pyrox.inference.ensemble_map``) or ``"vi"``
            (``pyrox.inference.ensemble_vi`` with ``guide_factory``).
        num_epochs: Optimiser steps per member (full batch).
        learning_rate: Adam learning rate.
        prior_weight: ``1`` ⇒ MAP, ``0`` ⇒ MLE, fractional ⇒ tempered
            (MAP only).
        guide_factory: VI guide constructor taking the model function;
            ``None`` ⇒ ``AutoNormal``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import geonnax
        >>> from pyrox_nn import BayesianSIREN
        >>> from pyrox_nn.api import NeuralFieldEstimator
        >>> enc = geonnax.SphericalHarmonicEncoder(l_max=3, input_mode="lonlat")
        >>> est = NeuralFieldEstimator(
        ...     model_fn=lambda: BayesianSIREN.init(16, 32, 1, depth=3),
        ...     encoder=enc, ensemble_size=2, num_epochs=10,
        ... )
        >>> lonlat = jnp.stack([jnp.linspace(-3, 3, 20), jnp.zeros(20)], -1)
        >>> fitted = est.fit(lonlat, jnp.sin(lonlat[:, 0]), seed=0)
        >>> mean, (lo, hi) = fitted.predict(lonlat, quantiles=(0.05, 0.95))
        >>> mean.shape, lo.shape
        ((20,), (20,))
    """

    model_fn: Callable[[], Callable[[Array], Any]] = eqx.field(static=True)
    encoder: Callable[[Array], Array] | None = None
    noise: NoiseModel = eqx.field(static=True, default="fixed")
    sigma_obs: float = eqx.field(static=True, default=0.1)
    ensemble_size: int = eqx.field(static=True, default=16)
    inference: InferenceKind = eqx.field(static=True, default="map")
    num_epochs: int = eqx.field(static=True, default=2000)
    learning_rate: float = eqx.field(static=True, default=5e-3)
    prior_weight: float = eqx.field(static=True, default=1.0)
    guide_factory: Callable[[Callable], Any] | None = eqx.field(
        static=True, default=None
    )

    def fit(
        self,
        X: Float[ArrayLike, "N D"],
        y: Float[ArrayLike, " N"],
        *,
        sigma: Float[ArrayLike, " N"] | None = None,
        valid: ArrayLike | None = None,
        seed: PRNGKeyArray | int,
    ) -> FittedNeuralField:
        r"""Fit the ensemble on arrays.

        Args:
            X: Coordinates of shape ``(N, D)`` (the encoder's input units).
            y: Targets of shape ``(N,)``.
            sigma: Optional per-row noise stds $\sigma_i$ of shape ``(N,)``,
                finite and > 0 on valid rows, in target units.
            valid: Optional boolean mask of shape ``(N,)``; rows with
                ``False`` (padding) are ignored, whatever their values.
            seed: PRNG key or integer seed.

        Returns:
            The fitted ensemble.

        Raises:
            ValueError: On a bad configuration, mismatched shapes, a mask
                with no valid row, or non-finite values (or $\sigma_i \le 0$)
                on a valid row.
        """
        _check_config(
            noise=self.noise,
            sigma_obs=self.sigma_obs,
            inference=self.inference,
            prior_weight=self.prior_weight,
            has_sigma=sigma is not None,
        )
        x_train, y_train = _pack_data(X, y, sigma, valid)
        if isinstance(seed, int):
            seed = jr.PRNGKey(seed)
        field = self.model_fn()
        model_fn = _field_model(
            _make_forward(field, self.encoder),
            self.sigma_obs,
            noise=self.noise,
            has_sigma=sigma is not None,
            has_valid=valid is not None,
        )
        params, losses = _fit_ensemble(
            model_fn,
            x_train,
            y_train,
            inference=self.inference,
            ensemble_size=self.ensemble_size,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            prior_weight=self.prior_weight,
            guide_factory=self.guide_factory,
            seed=seed,
        )
        return FittedNeuralField(
            config=self,
            field=field,
            params=params,
            losses=losses,
            has_sigma=sigma is not None,
        )

    def fit_xarray(
        self,
        ds: Any,
        *,
        coords: Sequence[str],
        target: str,
        sigma: str | None = None,
        time_origin: Any = None,
        seed: PRNGKeyArray | int,
    ) -> FittedNeuralField:
        r"""Fit on an xarray Dataset (``pip install pyrox-nn[xarray]``).

        Every variable named in ``coords``, ``target`` and ``sigma`` (a
        dimension, coordinate or data variable) is broadcast against the
        others and flattened, in the target's dimension order, to one row
        per grid point; ``coords`` become the columns of ``X`` in the order
        given. Rows where any of them is missing (NaN, NaT) are masked out,
        so a gappy grid needs no reshaping.

        A datetime coordinate becomes float days since ``time_origin``
        (default: its earliest timestamp), so the column starts near 0
        instead of at ~2·10⁴ days since 1970, which float32 would resolve
        only to minutes and a SIREN would see as a huge frequency. A
        timedelta coordinate becomes float days. The origins are stored on
        the fit; `FittedNeuralField.predict_xarray` applies the same
        mapping, and ``predict`` on arrays expects the same units.

        Args:
            ds: An ``xarray.Dataset``.
            coords: Names of the input variables, e.g. ``("lon", "lat", "time")``.
            target: Name of the target variable.
            sigma: Optional name of the per-row noise std variable.
            time_origin: Optional reference time for every datetime
                coordinate (anything ``numpy.datetime64`` accepts).
            seed: PRNG key or integer seed.

        Returns:
            The fitted ensemble, which remembers ``coords`` and the time
            origins.

        Raises:
            ImportError: If xarray is not installed.
            ValueError: If a name is not a variable of ``ds``, or as `fit`.
        """
        xr = _require_xarray("fit_xarray")
        names = [*coords, target, *([sigma] if sigma is not None else [])]
        _check_names(ds, names)
        # Flatten in the target's dimension order (C order), whatever the
        # order of ``coords``.
        arrays = xr.broadcast(*(ds[name] for name in names))
        dims = (
            *ds[target].dims,
            *(d for d in arrays[0].dims if d not in ds[target].dims),
        )
        arrays = [a.transpose(*dims) for a in arrays]
        origins = tuple(_time_origin(ds[name].values, time_origin) for name in coords)
        cols = [
            _numeric_column(a.values.ravel(), origin)
            for a, origin in zip(
                arrays, (*origins, *([None] * (len(names) - len(coords)))), strict=True
            )
        ]
        X = np.stack(cols[: len(coords)], axis=-1)
        valid = np.isfinite(np.stack(cols, axis=-1)).all(axis=-1)
        fitted = self.fit(
            X,
            cols[len(coords)],
            sigma=None if sigma is None else cols[-1],
            valid=valid,
            seed=seed,
        )
        return FittedNeuralField(
            config=fitted.config,
            field=fitted.field,
            params=fitted.params,
            losses=fitted.losses,
            has_sigma=fitted.has_sigma,
            xarray_coords=tuple(coords),
            time_origins=origins,
        )


def _require_xarray(caller: str) -> Any:
    try:
        import xarray as xr
    except ImportError as err:  # pragma: no cover - exercised without xarray
        raise ImportError(
            f"{caller} needs xarray: pip install 'pyrox-nn[xarray]'."
        ) from err
    return xr


def _check_names(ds: Any, names: Sequence[str]) -> None:
    missing = [name for name in names if name not in ds.variables]
    if missing:
        raise ValueError(f"{missing} are not variables of the Dataset.")


def _time_origin(values: np.ndarray, time_origin: Any) -> str | None:
    """The reference time of a datetime column as an ISO string (ns), or
    ``None`` for other columns. A string, not a ``datetime64``, so it can be
    a static field of `FittedNeuralField`."""
    if not np.issubdtype(values.dtype, np.datetime64):
        return None
    if time_origin is not None:
        return str(np.datetime64(time_origin, "ns"))
    finite = values[~np.isnat(values)]
    if finite.size == 0:
        raise ValueError("a datetime coordinate has no valid timestamp.")
    return str(finite.min().astype("datetime64[ns]"))


def _numeric_column(values: np.ndarray, origin: str | None = None) -> np.ndarray:
    """Datetimes → float days since ``origin``, timedeltas → float days.

    Missing values (NaT) become NaN so the validity mask catches them.
    """
    if np.issubdtype(values.dtype, np.datetime64):
        if origin is None:  # pragma: no cover - origins come from _time_origin
            raise ValueError("a datetime column needs a time origin.")
        values = values.astype("datetime64[ns]") - np.datetime64(origin, "ns")
    if np.issubdtype(values.dtype, np.timedelta64):
        days = values / np.timedelta64(1, "D")
        return np.where(np.isnat(values), np.nan, days)
    return np.asarray(values, dtype=float)


class FittedNeuralField(eqx.Module):
    """Output of `NeuralFieldEstimator.fit`.

    Attributes:
        config: The `NeuralFieldEstimator` that produced this fit.
        field: The field returned by ``model_fn()`` (identical across
            members; the weights live in ``params``).
        params: Stacked per-member parameters: a dict of the field's sites
            with a leading ``(E,)`` axis for MAP, or the guide's
            variational parameters for VI.
        losses: Per-member loss history, shape ``(E, num_epochs)``.
        has_sigma: Whether per-row $\\sigma_i$ were given at fit time.
        xarray_coords: The ``coords`` of ``fit_xarray`` (``None`` after
            ``fit``).
        time_origins: Per coordinate, the reference time of a datetime
            coordinate as an ISO string (``None`` for the others), from
            ``fit_xarray``.
    """

    config: NeuralFieldEstimator
    field: Any
    params: Any
    losses: Float[Array, "E T"]
    has_sigma: bool = eqx.field(static=True, default=False)
    xarray_coords: tuple[str, ...] | None = eqx.field(static=True, default=None)
    time_origins: tuple[str | None, ...] | None = eqx.field(static=True, default=None)

    def predict(
        self,
        X: Float[ArrayLike, "N D"],
        *,
        quantiles: Sequence[float] | None = None,
        predict_seed: int = 0,
        num_posterior_samples: int = 32,
    ) -> Prediction:
        r"""Posterior-predictive mean (and optional quantiles).

        MAP members give one prediction each; VI members give
        ``num_posterior_samples`` guide draws each. The quantiles are those
        of the equal-weight mixture of $\mathcal N(f_m(x), \sigma_{0,m}^2)$
        over members (or draws), estimated by Monte Carlo, with
        $\sigma_{0,m}$ the fixed ``sigma_obs``, the member's learned value,
        or its $\sigma_m(x)$ when heteroscedastic. Per-row $\sigma_i$ are not
        known at new points, so they are not added.

        Args:
            X: Coordinates of shape ``(N, D)``, in the units used to fit.
            quantiles: Optional levels; then returns
                ``(mean, (q_1, ..., q_K))``.
            predict_seed: PRNG seed for VI posterior draws and quantile MC.
            num_posterior_samples: Per-member draws for VI predictions and
                quantile MC.

        Returns:
            The mean of shape ``(N,)``, or ``(mean, quantiles)``.
        """
        cfg = self.config
        return _predict(
            self.params,
            self.field,
            _floating(X),
            encoder=cfg.encoder,
            noise=cfg.noise,
            sigma_obs=cfg.sigma_obs,
            has_sigma=self.has_sigma,
            inference=cfg.inference,
            guide_factory=cfg.guide_factory,
            quantiles=quantiles,
            predict_seed=predict_seed,
            num_posterior_samples=num_posterior_samples,
        )

    def predict_xarray(
        self,
        ds: Any,
        *,
        quantiles: Sequence[float] | None = None,
        predict_seed: int = 0,
        num_posterior_samples: int = 32,
    ) -> Any:
        """Predict on the grid of an xarray Dataset, as fitted by ``fit_xarray``.

        The fitted ``coords`` are read from ``ds``, broadcast against each
        other and mapped to the fit's units (datetimes as days since the
        stored origins); the result has their broadcast shape and
        coordinates. A grid point with a missing coordinate predicts NaN.

        Args:
            ds: An ``xarray.Dataset`` holding the fitted ``coords``.
            quantiles: Optional levels, as in `predict`.
            predict_seed: As in `predict`.
            num_posterior_samples: As in `predict`.

        Returns:
            The mean as an ``xarray.DataArray``, or ``(mean, quantiles)``
            with one ``DataArray`` per level.

        Raises:
            ValueError: If this fit did not come from ``fit_xarray``, or a
                coordinate is not a variable of ``ds``.
        """
        if self.xarray_coords is None or self.time_origins is None:
            raise ValueError("predict_xarray needs a fit from fit_xarray.")
        xr = _require_xarray("predict_xarray")
        _check_names(ds, self.xarray_coords)
        arrays = xr.broadcast(*(ds[name] for name in self.xarray_coords))
        template = arrays[0]
        cols = [
            _numeric_column(a.transpose(*template.dims).values.ravel(), origin)
            for a, origin in zip(arrays, self.time_origins, strict=True)
        ]
        X = np.stack(cols, axis=-1)
        rows = np.asarray(np.isfinite(X).all(axis=-1))
        out = self.predict(
            np.where(rows[:, None], X, 0.0),
            quantiles=quantiles,
            predict_seed=predict_seed,
            num_posterior_samples=num_posterior_samples,
        )

        def as_grid(values: Any) -> Any:
            grid = np.where(rows, np.asarray(values), np.nan).reshape(template.shape)
            return xr.DataArray(grid, coords=template.coords, dims=template.dims)

        if quantiles is None:
            return as_grid(out)
        mean, qs = out
        return as_grid(mean), tuple(as_grid(q) for q in qs)
