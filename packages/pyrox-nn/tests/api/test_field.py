"""Tests for `pyrox_nn.api.NeuralFieldEstimator` (arrays, masks, xarray)."""

from __future__ import annotations

import equinox as eqx
import geonnax
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro.distributions as dist
import pytest
from geonnax.basis import real_spherical_harmonics
from geonnax.geo import lonlat_to_cartesian3d
from numpyro import handlers
from numpyro.distributions import constraints
from pyrox._core import PyroxModule, pyrox_method
from pyrox_nn import BayesianSIREN, DenseVariationalDropout, GaussianRegressionHead
from pyrox_nn.api import FittedNeuralField, NeuralFieldEstimator
from pyrox_nn.api._field import (
    _field_init_fn,
    _field_log_joint,
    _field_model,
    _make_forward,
    _pack_data,
)


def _siren(d_in: int = 1) -> BayesianSIREN:
    return BayesianSIREN.init(d_in, 8, 1, depth=2, first_omega=3.0)


def _data(seed: int = 0, n: int = 24):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 1))
    y = np.sin(3 * x[:, 0]) + 0.1 * rng.normal(size=n)
    sigma = rng.uniform(0.05, 0.3, n)
    return x, y, sigma


class _HeteroField(eqx.Module):
    """A SIREN trunk followed by a Gaussian regression head: (mu, log_sigma)."""

    trunk: BayesianSIREN
    head: GaussianRegressionHead

    def __call__(self, x):
        return self.head(self.trunk(x))


def _hetero_field(hidden: int = 16) -> _HeteroField:
    return _HeteroField(
        BayesianSIREN.init(1, 32, hidden, depth=2, first_omega=3.0, pyrox_name="trunk"),
        GaussianRegressionHead.init(hidden, 1, sigma_init=0.3, pyrox_name="head"),
    )


# -- the model and its log joint ---------------------------------------------


def test_log_joint_matches_closed_form_with_sigma_and_mask():
    """loglik = Σ mᵢ log N(yᵢ | f(xᵢ), σᵢ² + σ₀²); logprior = the field's prior."""
    field = _siren()
    forward = _make_forward(field, None)
    model_fn = _field_model(forward, 0.2, has_sigma=True, has_valid=True)
    x, y, sigma = _data(0, 10)
    valid = np.arange(10) % 3 != 0
    xb, yb = _pack_data(x, y, sigma, valid)
    params = _field_init_fn(model_fn, 1, yb.shape[1:], jr.key(0))
    ll, lp = _field_log_joint(model_fn)(params, xb, yb)

    f = handlers.substitute(forward, params)(jnp.asarray(x))
    scale = jnp.sqrt(jnp.asarray(sigma) ** 2 + 0.2**2)
    expected_ll = jnp.sum(jnp.where(valid, dist.Normal(f, scale).log_prob(y), 0.0))
    tr = handlers.trace(handlers.substitute(field, params)).get_trace(jnp.asarray(x))
    expected_lp = sum(s["fn"].log_prob(s["value"]).sum() for s in tr.values())
    # Same terms summed in a different order: float64 round-off.
    np.testing.assert_allclose(ll, expected_ll, rtol=1e-12)
    np.testing.assert_allclose(lp, expected_lp, rtol=1e-12)


def test_masked_junk_rows_do_not_change_the_log_joint_or_its_gradient():
    field = _siren()
    model_fn = _field_model(_make_forward(field, None), 0.1, has_valid=True)
    x, y, _ = _data(1, 12)
    xb, yb = _pack_data(x, y, None, np.ones(12, dtype=bool))
    x_pad = np.concatenate([x, np.full((5, 1), np.nan)])
    y_pad = np.concatenate([y, [np.inf, np.nan, 1e30, -3.0, 0.0]])
    valid = np.arange(17) < 12
    xp, yp = _pack_data(x_pad, y_pad, None, valid)
    log_joint = _field_log_joint(model_fn)
    params = _field_init_fn(model_fn, 1, yb.shape[1:], jr.key(2))

    def total(p, xx, yy):
        ll, lp = log_joint(p, xx, yy)
        return ll + lp

    v0, g0 = jax.value_and_grad(total)(params, xb, yb)
    v1, g1 = jax.value_and_grad(total)(params, xp, yp)
    # Masked rows add exact zeros (and zero cotangents) to the same sums.
    np.testing.assert_allclose(v1, v0, rtol=1e-12)
    for name in g0:
        assert jnp.all(jnp.isfinite(g1[name]))
        np.testing.assert_allclose(g1[name], g0[name], rtol=1e-10, atol=1e-12)


def test_heteroscedastic_scale_is_the_variance_sum():
    field = _hetero_field(4)
    forward = _make_forward(field, None)
    model_fn = _field_model(forward, 0.1, noise="heteroscedastic", has_sigma=True)
    x, y, sigma = _data(2, 8)
    xb, yb = _pack_data(x, y, sigma, None)
    params = _field_init_fn(model_fn, 1, yb.shape[1:], jr.key(0))
    assert not any(k == "log_sigma_obs" for k in params)
    ll, _ = _field_log_joint(model_fn)(params, xb, yb)
    mu, log_s = handlers.substitute(forward, params)(jnp.asarray(x))
    scale = jnp.sqrt(jnp.asarray(sigma) ** 2 + jnp.exp(2 * log_s))
    np.testing.assert_allclose(ll, dist.Normal(mu, scale).log_prob(y).sum(), rtol=1e-12)


def test_model_sites():
    model_fn = _field_model(_make_forward(_siren(), None), 0.1, noise="learned")
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model_fn(jnp.zeros((3, 1)), jnp.zeros(3))
    assert set(tr) == {
        "BayesianSIREN.layer_0.W",
        "BayesianSIREN.layer_0.b",
        "BayesianSIREN.layer_1.W",
        "BayesianSIREN.layer_1.b",
        "f",
        "log_sigma_obs",
        "y",
    }
    assert tr["f"]["value"].shape == (3,)  # (N, 1) output squeezed to (N,)


@pytest.mark.parametrize(
    ("make", "noise"), [(_siren, "heteroscedastic"), (_hetero_field, "fixed")]
)
def test_noise_must_match_the_field_output(make, noise):
    model_fn = _field_model(_make_forward(make(), None), 0.1, noise=noise)
    with handlers.seed(rng_seed=0), pytest.raises(ValueError, match="only then"):
        model_fn(jnp.zeros((3, 1)), jnp.zeros(3))


def test_multi_output_field_is_rejected():
    forward = _make_forward(BayesianSIREN.init(1, 4, 2, depth=2), None)
    with handlers.seed(rng_seed=0), pytest.raises(ValueError, match=r"\(N,\) or"):
        forward(jnp.zeros((3, 1)))


# -- fitting on arrays ----------------------------------------------------------


@pytest.mark.parametrize("inference", ["map", "vi"])
def test_fit_and_predict_shapes(inference):
    x, y, sigma = _data(0, 20)
    est = NeuralFieldEstimator(
        model_fn=_siren,
        noise="learned",
        ensemble_size=2,
        num_epochs=5,
        inference=inference,
    )
    fitted = est.fit(x, y, sigma=sigma, seed=0)
    assert isinstance(fitted, FittedNeuralField)
    assert fitted.losses.shape == (2, 5)
    mean, (lo, hi) = fitted.predict(
        x[:7], quantiles=(0.05, 0.95), num_posterior_samples=4
    )
    assert mean.shape == lo.shape == hi.shape == (7,)
    assert jnp.all(lo < hi)
    assert fitted.predict(x[:7]).shape == (7,)


def test_padding_invariance_of_the_fit():
    """Appending masked junk rows leaves the fitted ensemble unchanged."""
    x, y, sigma = _data(3, 20)
    est = NeuralFieldEstimator(model_fn=_siren, ensemble_size=2, num_epochs=50)
    plain = est.fit(x, y, sigma=sigma, seed=0)
    x_pad = np.concatenate([x, np.full((6, 1), np.nan)])
    y_pad = np.concatenate([y, np.full(6, 1e6)])
    s_pad = np.concatenate([sigma, np.zeros(6)])  # invalid σ on padding is fine
    padded = est.fit(x_pad, y_pad, sigma=s_pad, valid=np.arange(26) < 20, seed=0)
    # The padded rows add exact zeros, but a longer reduction may round
    # differently (~1e-16 relative per step); 50 Adam steps keep it far
    # below 1e-9.
    for name in plain.params:
        np.testing.assert_allclose(
            padded.params[name], plain.params[name], rtol=1e-9, atol=1e-12
        )
    np.testing.assert_allclose(padded.losses, plain.losses, rtol=1e-9)


class _ParamField(PyroxModule):
    """``w·x + b + s`` with a Normal weight, a real param ``b`` and a
    positive param ``s``."""

    pyrox_name: str | None = eqx.field(static=True, default="P")

    @pyrox_method
    def __call__(self, z):
        w = self.pyrox_sample("w", dist.Normal(0.0, 1.0))
        b = self.pyrox_param("b", jnp.asarray(0.0))
        s = self.pyrox_param("s", jnp.asarray(0.5), constraint=constraints.positive)
        return w * z[:, 0] + b + s


def test_map_optimises_param_sites_in_unconstrained_space():
    """pyrox_param sites are fitted, not left at their initial values."""
    x, y, _ = _data(0, 30)
    est = NeuralFieldEstimator(
        model_fn=_ParamField, ensemble_size=2, num_epochs=300, learning_rate=5e-2
    )
    fitted = est.fit(x, y + 5.0, seed=0)
    assert set(fitted.params) == {"P.w", "P.b", "P.s"}
    # Stored in constrained space, per member.
    assert fitted.params["P.s"].shape == (2,)
    assert jnp.all(fitted.params["P.s"] > 0)
    offset = fitted.params["P.b"] + fitted.params["P.s"]
    # The least-squares offset of sin(3x) + 0.1·noise + 5 on 30 points is
    # 5 ± ~0.3 (the slope absorbs part of the sine); with the params frozen
    # at their init the offset stays 0.5 and the mean prediction near 0.5.
    np.testing.assert_allclose(offset, 5.0, atol=0.5)
    assert abs(float(jnp.mean(fitted.predict(x))) - float(np.mean(y + 5.0))) < 0.2


class _PositiveField(PyroxModule):
    """``-a`` with ``a ~ LogNormal(log 1e-4, 1)``: the data pull ``a`` to 0."""

    pyrox_name: str | None = eqx.field(static=True, default="Pos")

    @pyrox_method
    def __call__(self, z):
        a = self.pyrox_sample("a", dist.LogNormal(jnp.log(1e-4), 1.0))
        return -a * jnp.ones(z.shape[0], dtype=z.dtype)


def test_map_keeps_constrained_latents_in_their_support():
    """Adam steps on a raw LogNormal draw near 1e-4 overshoot below 0 (NaN
    losses); optimising log a keeps a > 0 and the fit finite."""
    x, _, _ = _data(0, 20)
    est = NeuralFieldEstimator(model_fn=_PositiveField, ensemble_size=2, num_epochs=50)
    fitted = est.fit(x, -np.ones(20), seed=0)
    assert jnp.all(jnp.isfinite(fitted.losses))
    assert jnp.all(fitted.params["Pos.a"] > 0)


def test_constrained_latent_log_joint_is_the_constrained_density():
    """No Jacobian: logprior is the LogNormal density at a = exp(u)."""
    from pyrox_nn.api._field import _map_param_constraints

    model_fn = _field_model(_make_forward(_PositiveField(), None), 0.1)
    constraints_ = _map_param_constraints(model_fn, 1, ())
    assert set(constraints_) == {"Pos.a"}
    u = jnp.asarray(-8.0)
    _, lp = _field_log_joint(model_fn, constraints_)(
        {"Pos.a": u}, jnp.zeros((2, 1)), jnp.zeros(2)
    )
    expected = dist.LogNormal(jnp.log(1e-4), 1.0).log_prob(jnp.exp(u))
    np.testing.assert_allclose(lp, expected, rtol=1e-12)


def test_map_fits_kernel_hyperparameter_param_sites():
    """SphericalHarmonicFeatures with a plain RBF: the kernel's variance and
    lengthscale are pyrox_param sites (no priors) and must move from 1."""
    import pyrox_gp as pgp
    from pyrox_nn import SphericalHarmonicFeatures

    rng = np.random.default_rng(0)
    lonlat = np.stack([rng.uniform(-3, 3, 40), rng.uniform(-1.4, 1.4, 40)], -1)
    y = 3.0 * np.cos(lonlat[:, 1])
    est = NeuralFieldEstimator(
        model_fn=lambda: SphericalHarmonicFeatures.init(
            l_max=3, kernel=pgp.RBF(), input_mode="lonlat", input_unit="radians"
        ),
        ensemble_size=2,
        num_epochs=100,
        learning_rate=5e-2,
    )
    fitted = est.fit(lonlat, y, seed=0)
    for name in ("RBF.variance", "RBF.lengthscale"):
        assert fitted.params[name].shape == (2,)
        assert jnp.all(fitted.params[name] > 0)
        # Initialised at 1.0; 100 steps at lr 0.05 move them by O(1)
        # (measured 1.4-2.7), far from a frozen value.
        assert jnp.all(jnp.abs(fitted.params[name] - 1.0) > 0.1)


def test_map_rejects_fields_that_draw_prng_keys():
    x, y, _ = _data(0, 10)
    make = lambda: DenseVariationalDropout(in_features=1, out_features=1)
    with pytest.raises(ValueError, match="prng_key"):
        NeuralFieldEstimator(model_fn=make, ensemble_size=1, num_epochs=2).fit(
            x, y, seed=0
        )
    # VI runs under SVI's seed, so the same field fits there.
    fitted = NeuralFieldEstimator(
        model_fn=make, inference="vi", ensemble_size=1, num_epochs=2
    ).fit(x, y, seed=0)
    assert fitted.predict(x).shape == (10,)


def test_vi_padding_invariance_of_the_fit():
    x, y, _ = _data(5, 20)
    est = NeuralFieldEstimator(
        model_fn=_siren, inference="vi", ensemble_size=2, num_epochs=30
    )
    plain = est.fit(x, y, seed=0)
    x_pad = np.concatenate([x, np.full((5, 1), np.nan)])
    y_pad = np.concatenate([y, np.full(5, np.nan)])
    padded = est.fit(x_pad, y_pad, valid=np.arange(25) < 20, seed=0)
    # Same ELBO up to reduction-order round-off (measured 2e-12 on 30 steps).
    np.testing.assert_allclose(padded.losses, plain.losses, rtol=1e-9)
    for a, b in zip(
        jax.tree.leaves(plain.params), jax.tree.leaves(padded.params), strict=True
    ):
        np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-12)


def test_encoder_is_applied_per_row():
    enc = geonnax.SphericalHarmonicEncoder(l_max=1, input_mode="lonlat")
    forward = _make_forward(lambda z: z[:, :1], enc)
    lonlat = jnp.asarray([[0.3, -0.2], [1.0, 0.5]])
    np.testing.assert_allclose(forward(lonlat), jax.vmap(enc)(lonlat)[:, 0])


def test_heteroscedastic_fit_and_predict_smoke():
    x, y, _ = _data(4, 20)
    est = NeuralFieldEstimator(
        model_fn=lambda: _hetero_field(4),
        noise="heteroscedastic",
        ensemble_size=2,
        num_epochs=5,
    )
    fitted = est.fit(x, y, seed=0)
    mean, (lo, hi) = fitted.predict(x, quantiles=(0.1, 0.9))
    assert mean.shape == lo.shape == (20,)
    assert jnp.all(lo < hi)


@pytest.mark.parametrize(
    ("kwargs", "fit_kwargs", "match"),
    [
        ({"noise": "robust"}, {}, "noise must be"),
        ({"sigma_obs": 0.0}, {}, "sigma_obs must be > 0"),
        ({"inference": "nuts"}, {}, "Unknown inference"),
        ({"noise": "learned", "prior_weight": 0.0}, {}, "bound sigma_0 below"),
        ({}, {"valid": np.zeros(20, dtype=bool)}, "every row"),
        ({}, {"valid": np.ones(20)}, "boolean array"),
        ({}, {"sigma": np.zeros(20)}, "finite and > 0"),
        ({}, {"sigma": np.ones(3)}, "sigma must have shape"),
    ],
)
def test_validation(kwargs, fit_kwargs, match):
    x, y, _ = _data(0, 20)
    est = NeuralFieldEstimator(model_fn=_siren, num_epochs=2, **kwargs)
    with pytest.raises(ValueError, match=match):
        est.fit(x, y, seed=0, **fit_kwargs)


def test_non_finite_values_on_valid_rows_are_rejected():
    x, y, _ = _data(0, 20)
    est = NeuralFieldEstimator(model_fn=_siren, num_epochs=2)
    y_bad = y.copy()
    y_bad[3] = np.nan
    with pytest.raises(ValueError, match="y must be finite"):
        est.fit(x, y_bad, seed=0)
    with pytest.raises(ValueError, match="X must have shape"):
        est.fit(x[:, 0], y, seed=0)
    # The same NaN on a masked row is fine.
    est.fit(x, y_bad, valid=np.arange(20) != 3, seed=0)


# -- xarray -----------------------------------------------------------------------


def _grid_dataset():
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(0)
    lon = np.linspace(-2.5, 2.5, 5)
    lat = np.linspace(-1.0, 1.0, 4)
    time = np.array(["2020-01-01", "2020-01-03", "2020-01-04"], dtype="datetime64[ns]")
    sla = rng.normal(size=(3, 4, 5))
    sla[0, 1, 2] = np.nan  # a gap (e.g. land or cloud)
    err = rng.uniform(0.05, 0.2, size=(3, 4, 5))
    return xr.Dataset(
        {
            "sla": (("time", "lat", "lon"), sla),
            "sla_err": (("time", "lat", "lon"), err),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


def test_fit_xarray_equals_fit_on_the_flattened_grid():
    """Flattening order, datetime units and the NaN mask match a hand-built fit."""
    ds = _grid_dataset()
    est = NeuralFieldEstimator(
        model_fn=lambda: BayesianSIREN.init(3, 8, 1, depth=2, first_omega=1.0),
        ensemble_size=2,
        num_epochs=10,
    )
    fitted = est.fit_xarray(
        ds, coords=("lon", "lat", "time"), target="sla", sigma="sla_err", seed=0
    )
    tt, la, lo = np.meshgrid(
        np.array([0.0, 2.0, 3.0]),  # days since the first timestamp
        ds["lat"].values,
        ds["lon"].values,
        indexing="ij",
    )
    X = np.stack([lo.ravel(), la.ravel(), tt.ravel()], axis=-1)
    y = ds["sla"].values.ravel()
    valid = np.isfinite(y)
    assert valid.sum() == 59
    direct = est.fit(X, y, sigma=ds["sla_err"].values.ravel(), valid=valid, seed=0)
    # Identical inputs, identical computation: bitwise equal.
    for name in direct.params:
        assert jnp.array_equal(fitted.params[name], direct.params[name])
    assert fitted.time_origins == (None, None, "2020-01-01T00:00:00.000000000")

    # predict_xarray applies the same mapping and returns the grid.
    mean = fitted.predict_xarray(ds)
    assert mean.dims == ("lon", "lat", "time")
    # Same rows in another order: the batched matmuls may round differently.
    np.testing.assert_allclose(
        mean.transpose("time", "lat", "lon").values.ravel(),
        fitted.predict(X),
        rtol=1e-12,
    )


def test_fit_xarray_time_origin_and_sub_minute_resolution():
    xr = pytest.importorskip("xarray")
    time = np.datetime64("2024-03-01T00:00:00", "ns") + np.arange(6) * np.timedelta64(
        10, "s"
    )
    ds = xr.Dataset({"z": (("time",), np.arange(6.0))}, coords={"time": time})
    est = NeuralFieldEstimator(model_fn=_siren, ensemble_size=1, num_epochs=2)
    fitted = est.fit_xarray(
        ds, coords=("time",), target="z", time_origin="2024-02-29", seed=0
    )
    assert fitted.time_origins == ("2024-02-29T00:00:00.000000000",)
    from pyrox_nn.api._field import _numeric_column

    days = _numeric_column(time, fitted.time_origins[0])
    # 10 s steps stay distinct in float32 near 1 day (spacing ~6e-8 day =
    # 5 ms); since 1970 (~2e4 days, spacing ~2e-3 day = 3 min) they merge.
    assert np.unique(days.astype(np.float32)).size == 6
    # diff of values ~1 in float64: absolute error ~1e-16.
    np.testing.assert_allclose(np.diff(days), 10 / 86400, rtol=0, atol=1e-14)
    assert fitted.predict_xarray(ds).shape == (6,)


def test_predict_xarray_needs_an_xarray_fit():
    x, y, _ = _data(0, 10)
    est = NeuralFieldEstimator(model_fn=_siren, ensemble_size=1, num_epochs=2)
    with pytest.raises(ValueError, match="fit_xarray"):
        est.fit(x, y, seed=0).predict_xarray(_grid_dataset())


def test_fit_xarray_rejects_unknown_names():
    ds = _grid_dataset()
    est = NeuralFieldEstimator(model_fn=_siren, num_epochs=2)
    with pytest.raises(ValueError, match="not variables"):
        est.fit_xarray(ds, coords=("lon", "depth"), target="sla", seed=0)


# -- end to end (slow) ----------------------------------------------------------------


def _sphere(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-np.pi, np.pi, n)
    lat = np.arcsin(rng.uniform(-1.0, 1.0, n))  # uniform on the sphere
    return np.stack([lon, lat], axis=-1)


def _degree3_field(lonlat: np.ndarray) -> np.ndarray:
    coeffs = np.random.default_rng(100).normal(size=16)  # all (l, m), l ≤ 3
    xyz = lonlat_to_cartesian3d(jnp.asarray(lonlat))
    return np.asarray(real_spherical_harmonics(xyz, 3)) @ coeffs


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_siren_with_sh_encoder_fits_a_degree3_spherical_harmonic_field(seed):
    x_train, x_test = _sphere(seed, 300), _sphere(seed + 50, 300)
    noise = 0.05 * np.random.default_rng(seed + 7).normal(size=300)
    y_train = _degree3_field(x_train) + noise
    y_test = _degree3_field(x_test)
    est = NeuralFieldEstimator(
        model_fn=lambda: BayesianSIREN.init(16, 32, 1, depth=2, first_omega=1.0),
        encoder=geonnax.SphericalHarmonicEncoder(l_max=3, input_mode="lonlat"),
        sigma_obs=0.05,
        ensemble_size=4,
        num_epochs=1000,
        learning_rate=1e-2,
    )
    mean = est.fit(x_train, y_train, seed=seed).predict(x_test)
    rel_rmse = np.sqrt(np.mean((np.asarray(mean) - y_test) ** 2)) / y_test.std()
    # Calibrated on seeds 0, 1, 2: 0.012, 0.008, 0.012 (the noise is 0.05
    # against a field sd of ~1.1). 0.05 leaves 4x headroom; a field that
    # missed the degree-3 structure scores ~1.
    assert rel_rmse < 0.05


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_heteroscedastic_noise_recovers_input_dependent_scale(seed):
    """σ(x) = 0.1 + 0.5|x| (the #320 check): wider intervals at the edges,
    calibrated 90 % coverage."""

    def data(s, n):
        rng = np.random.default_rng(s)
        x = rng.uniform(-1, 1, n)
        y = np.sin(3 * x) + (0.1 + 0.5 * np.abs(x)) * rng.normal(size=n)
        return x[:, None], y

    x, y = data(seed, 400)
    x_test, y_test = data(seed + 50, 2000)
    est = NeuralFieldEstimator(
        model_fn=_hetero_field,
        noise="heteroscedastic",
        ensemble_size=4,
        num_epochs=2000,
    )
    _, (lo, hi) = est.fit(x, y, seed=seed).predict(x_test, quantiles=(0.05, 0.95))
    lo, hi = np.asarray(lo), np.asarray(hi)
    coverage = np.mean((y_test >= lo) & (y_test <= hi))
    # 2000 test points: binomial sd sqrt(0.9·0.1/2000) = 0.0067. Calibrated on
    # seeds 0, 1, 2: 0.883, 0.911, 0.886; the bound 0.04 is 6 sd, and also
    # absorbs the 200-sample MC error of the quantiles.
    assert abs(coverage - 0.9) < 0.04
    width = hi - lo
    edge = width[np.abs(x_test[:, 0]) > 0.8].mean()
    centre = width[np.abs(x_test[:, 0]) < 0.2].mean()
    # True σ ratio between the bands is ~0.55 / 0.15 ≈ 3.7; seeds 0, 1, 2
    # give 3.2, 3.4, 3.2. A homoscedastic fit gives ~1.
    assert edge / centre > 2.0
