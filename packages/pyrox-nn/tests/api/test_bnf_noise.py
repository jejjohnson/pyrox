"""Observation noise in the BNF estimator: learned σ₀ and per-row σᵢ."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pytest
from numpyro import handlers
from pyrox_nn import BayesianNeuralField
from pyrox_nn.api import BNFEstimator
from pyrox_nn.api._bnf import (
    _bnf_init_fn,
    _bnf_log_joint,
    _bnf_model,
    _gaussian_mixture_quantiles,
)


def _bnf() -> BayesianNeuralField:
    return BayesianNeuralField(
        input_scales=(1.0, 1.0),
        fourier_degrees=(2, 0),
        interactions=(),
        seasonality_periods=(),
        num_seasonal_harmonics=(),
        width=8,
        depth=2,
        pyrox_name="bnf",
    )


def _previous_log_joint(bnf_template, sigma_obs):
    """`_bnf_log_joint` before observation noise was configurable, verbatim."""
    from numpyro.handlers import substitute, trace

    def model_fn(x, y):
        f = bnf_template(x)
        numpyro.sample("y", dist.Normal(f, sigma_obs), obs=y)

    def log_joint(params, xb, yb):
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


def _xy(n: int = 12, seed: int = 0):
    kx, ky, ks = jr.split(jr.PRNGKey(seed), 3)
    x = jr.normal(kx, (n, 2))
    y = jr.normal(ky, (n,))
    sigma = jr.uniform(ks, (n,), minval=0.05, maxval=0.5)
    return x, y, sigma


def _params(model_fn, y_shape, seed):
    return _bnf_init_fn(model_fn, 2, y_shape, jr.PRNGKey(seed))


# -- the default configuration is unchanged -----------------------------------


def test_default_log_joint_equals_previous_exactly():
    bnf = _bnf()
    model_fn = _bnf_model(bnf, 0.1)
    new, old = _bnf_log_joint(model_fn), _previous_log_joint(bnf, 0.1)
    x, y, _ = _xy()
    for seed in range(5):
        params = _params(model_fn, (), seed)
        assert set(params) == {k for k in params if k.startswith("bnf.")}
        ll_new, lp_new = new(params, x, y)
        ll_old, lp_old = old(params, x, y)
        # Same distributions on the same values: bitwise equal.
        assert ll_new == ll_old and lp_new == lp_old


def test_default_init_draws_equal_previous():
    """The init trace now includes the likelihood; the draws do not change."""
    from numpyro.handlers import seed, trace

    bnf = _bnf()
    params = _params(_bnf_model(bnf, 0.1), (), 3)
    tr = trace(seed(bnf, jr.PRNGKey(3))).get_trace(jnp.zeros((1, 2)))
    assert params.keys() == tr.keys()
    for name, value in params.items():
        assert jnp.array_equal(value, tr[name]["value"])


def test_estimator_defaults_are_backwards_compatible():
    est = BNFEstimator(feature_cols=("t",), target_col="z")
    assert est.noise == "fixed" and est.sigma_obs == 0.1 and est.sigma_col is None


# -- per-row σᵢ ---------------------------------------------------------------


def test_sigma_column_log_likelihood_is_the_variance_sum():
    bnf = _bnf()
    x, y, sigma = _xy()
    packed = jnp.stack([y, sigma], axis=-1)
    for noise in ("fixed", "learned"):
        model_fn = _bnf_model(bnf, 0.1, noise=noise, has_sigma_col=True)
        params = _params(model_fn, (2,), 0)
        ll, _ = _bnf_log_joint(model_fn)(params, x, packed)
        f = handlers.substitute(bnf, data=params)(x)
        sigma0 = jnp.exp(params["sigma_obs"]) if noise == "learned" else 0.1
        ref = dist.Normal(f, jnp.sqrt(sigma**2 + sigma0**2)).log_prob(y).sum()
        # Float64 round-off (the root conftest enables x64).
        assert jnp.allclose(ll, ref, rtol=1e-12)


def test_zero_sigma_column_reduces_to_homoscedastic():
    """σᵢ = 0 packed with y gives the plain (N,) likelihood: σ₀ is the floor."""
    bnf = _bnf()
    x, y, _ = _xy()
    plain = _bnf_model(bnf, 0.1)
    packed = _bnf_model(bnf, 0.1, has_sigma_col=True)
    params = _params(plain, (), 1)
    ll_plain, lp_plain = _bnf_log_joint(plain)(params, x, y)
    ll_packed, lp_packed = _bnf_log_joint(packed)(
        params, x, jnp.stack([y, jnp.zeros_like(y)], axis=-1)
    )
    # sqrt(0 + 0.1²) vs 0.1: one rounding in float64.
    assert jnp.allclose(ll_packed, ll_plain, rtol=1e-14)
    assert lp_packed == lp_plain


def test_sigma_travels_with_its_row():
    """Permuting rows (as mini-batching does) leaves the likelihood unchanged,
    and swapping σᵢ between rows changes it: σᵢ is per row, not per position."""
    bnf = _bnf()
    x, y, sigma = _xy()
    model_fn = _bnf_model(bnf, 0.1, has_sigma_col=True)
    log_joint = _bnf_log_joint(model_fn)
    params = _params(model_fn, (2,), 2)
    packed = jnp.stack([y, sigma], axis=-1)
    perm = jr.permutation(jr.PRNGKey(9), x.shape[0])
    ll, _ = log_joint(params, x, packed)
    ll_perm, _ = log_joint(params, x[perm], packed[perm])
    assert jnp.allclose(ll, ll_perm, rtol=1e-12)
    swapped = jnp.stack([y, sigma[perm]], axis=-1)
    assert not jnp.allclose(log_joint(params, x, swapped)[0], ll)


def test_learned_noise_registers_log_space_site():
    bnf = _bnf()
    model_fn = _bnf_model(bnf, 0.2, noise="learned")
    x, y, _ = _xy()
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model_fn(x, y)
    site = tr["sigma_obs"]
    assert not site["is_observed"]
    assert isinstance(site["fn"], dist.Normal)
    assert jnp.allclose(site["fn"].loc, jnp.log(0.2))
    assert site["fn"].scale == 1.0
    assert "sigma_obs" in _params(model_fn, (), 0)


def _hetero_df(seed: int, n: int):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 10.0, n))
    sigma = rng.uniform(0.05, 0.5, n)
    y = np.sin(t) + sigma * rng.normal(size=n)
    return pd.DataFrame({"t": t, "z": y, "z_sd": sigma}), np.sin(t)


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ({"z_sd": 0.0}, "finite and > 0"),
        ({"z_sd": -1.0}, "finite and > 0"),
        ({"z_sd": np.nan}, "finite and > 0"),
        ({"z_sd": np.inf}, "finite and > 0"),
    ],
)
def test_sigma_column_validation(bad, match):
    df, _ = _hetero_df(0, 20)
    df.loc[3, "z_sd"] = bad["z_sd"]
    est = BNFEstimator(
        feature_cols=("t",), target_col="z", sigma_col="z_sd", num_epochs=2
    )
    with pytest.raises(ValueError, match=match):
        est.fit(df, seed=0)


def test_config_validation():
    df, _ = _hetero_df(0, 20)
    with pytest.raises(ValueError, match="not a column"):
        BNFEstimator(
            feature_cols=("t",), target_col="z", sigma_col="nope", num_epochs=2
        ).fit(df, seed=0)
    with pytest.raises(ValueError, match="noise must be"):
        BNFEstimator(
            feature_cols=("t",), target_col="z", noise="robust", num_epochs=2
        ).fit(df, seed=0)
    with pytest.raises(ValueError, match="sigma_obs must be > 0"):
        BNFEstimator(
            feature_cols=("t",), target_col="z", sigma_obs=0.0, num_epochs=2
        ).fit(df, seed=0)


@pytest.mark.parametrize("kind", ["map", "vi"])
def test_learned_noise_with_sigma_column_fits_and_predicts(kind):
    df, _ = _hetero_df(0, 40)
    est = BNFEstimator(
        feature_cols=("t",),
        target_col="z",
        width=8,
        depth=1,
        fourier_degrees=(2,),
        noise="learned",
        sigma_col="z_sd",
        ensemble_size=2,
        num_epochs=10,
        inference_kind=kind,
    )
    fitted = est.fit(df, seed=0)
    mean, (lo, hi) = fitted.predict(
        df[["t"]], quantiles=(0.05, 0.95), num_posterior_samples=4
    )
    assert mean.shape == lo.shape == hi.shape == (40,)
    assert jnp.all(lo < hi)


def test_mixture_quantiles_take_per_component_scales():
    """A zero-scale component contributes its mean exactly; the scalar form
    is unchanged."""
    means = jnp.array([[0.0], [10.0]])
    q_scalar = _gaussian_mixture_quantiles(
        means, 0.5, quantiles=(0.5,), seed=0, num_samples=64
    )
    q_vector = _gaussian_mixture_quantiles(
        means, jnp.array([0.5, 0.5]), quantiles=(0.5,), seed=0, num_samples=64
    )
    assert jnp.array_equal(q_scalar[0], q_vector[0])
    (q,) = _gaussian_mixture_quantiles(
        jnp.array([[1.0], [1.0]]),
        jnp.array([0.0, 0.0]),
        quantiles=(0.9,),
        seed=0,
        num_samples=64,
    )
    assert q == 1.0


# -- slow: calibration and recovery -------------------------------------------


@pytest.mark.slow
def test_heteroscedastic_coverage_with_sigma_column():
    """With known σᵢ in the likelihood, 90% intervals are calibrated.

    Test rows have known σᵢ (5x spread), so the interval for a new
    observation is the predictive mixture widened by σᵢ:
    f̂ ± 1.645·sqrt(v̂ + σᵢ²), where v̂ is the predictive variance of the
    latent plus σ₀ recovered from `predict`'s 5% / 95% quantiles. Coverage
    over n = 1000 test rows is Binomial(n, 0.9)/n under calibration, with
    standard error sqrt(0.9·0.1/1000) = 0.0095, so [0.85, 0.95] is a ±5.3 SE
    band: a calibrated fit fails it with probability < 1e-6.
    """
    df_train, _ = _hetero_df(0, 600)
    df_test, _ = _hetero_df(1, 1000)
    est = BNFEstimator(
        feature_cols=("t",),
        target_col="z",
        width=16,
        depth=2,
        fourier_degrees=(3,),
        noise="fixed",
        sigma_obs=0.01,
        sigma_col="z_sd",
        ensemble_size=4,
        num_epochs=3000,
    )
    fitted = est.fit(df_train, seed=0)
    mean, (lo, hi) = fitted.predict(df_test[["t"]], quantiles=(0.05, 0.95))
    v = ((np.asarray(hi) - np.asarray(lo)) / (2 * 1.645)) ** 2
    half = 1.645 * np.sqrt(v + df_test["z_sd"].to_numpy() ** 2)
    z = df_test["z"].to_numpy()
    covered = np.abs(z - np.asarray(mean)) <= half
    coverage = float(covered.mean())
    assert 0.85 <= coverage <= 0.95, f"90% coverage {coverage:.3f}"


@pytest.mark.slow
def test_learned_noise_recovers_homoscedastic_sigma():
    """noise='learned' recovers σ₀ = 0.3 from a prior median of 0.1.

    The MAP estimate is the ensemble member with the lowest final loss (the
    members are independent restarts, and some stop in poor local optima of
    the network weights, where σ₀ absorbs the misfit). Its ``sigma_obs`` site
    is log σ̂₀. With the mean well fit, the posterior of log σ from N
    Gaussian residuals has sd 1/sqrt(2N) = 0.016 for N = 2000 (Fisher
    information 2N; the Normal(log 0.1, 1) prior is negligible next to it),
    so the bound is |log σ̂₀ − log 0.3| ≤ 3/sqrt(2N).
    """
    n = 2000
    rng = np.random.default_rng(0)
    t = rng.uniform(0.0, 10.0, n)
    df = pd.DataFrame({"t": t, "z": np.sin(t) + 0.3 * rng.normal(size=n)})
    est = BNFEstimator(
        feature_cols=("t",),
        target_col="z",
        width=16,
        depth=2,
        fourier_degrees=(3,),
        noise="learned",
        sigma_obs=0.1,
        ensemble_size=3,
        num_epochs=2000,
    )
    fitted = est.fit(df, seed=0)
    best = int(jnp.argmin(fitted.losses[:, -1]))
    log_sigma0 = float(fitted.params["sigma_obs"][best])
    bound = 3.0 / np.sqrt(2 * n)
    assert abs(log_sigma0 - np.log(0.3)) <= bound, (
        f"sigma_0 = {np.exp(log_sigma0):.4f}; truth 0.3, log bound {bound:.3f}"
    )
