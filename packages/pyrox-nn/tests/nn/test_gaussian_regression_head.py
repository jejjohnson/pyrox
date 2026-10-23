"""Tests for ``pyrox_nn.GaussianRegressionHead`` (heteroscedastic regression)."""

from __future__ import annotations

import math

import geonnax
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
import pytest
from numpyro import handlers
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_median
from numpyro.infer.util import log_density
from pyrox._core.pyrox_module import PyroxModule
from pyrox_nn import GaussianRegressionHead


_SITES = ("W_loc", "b_loc", "W_log_scale", "b_log_scale")


def _core_values(head: GaussianRegressionHead) -> dict[str, jax.Array]:
    scope = head.pyrox_name or "GaussianRegressionHead"
    return {f"{scope}.{n}": getattr(head.core, n) for n in _SITES}


def test_is_pyrox_module_wrapping_the_geonnax_core():
    head = GaussianRegressionHead.init(4, 2)
    assert isinstance(head, PyroxModule)
    assert isinstance(head.core, geonnax.GaussianRegressionHead)
    assert (head.in_features, head.out_features) == (4, 2)


def test_registers_exactly_the_four_weight_sites():
    head = GaussianRegressionHead.init(4, 2, pyrox_name="noise")
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        head(jnp.ones((3, 4)))
    sample_sites = {k for k, v in tr.items() if v["type"] == "sample"}
    assert sample_sites == {f"noise.{n}" for n in _SITES}
    assert tr["noise.W_loc"]["value"].shape == (4, 2)
    assert tr["noise.b_loc"]["value"].shape == (2,)
    assert tr["noise.W_log_scale"]["value"].shape == (4, 2)
    assert tr["noise.b_log_scale"]["value"].shape == (2,)


def test_priors_are_centred_on_the_homoscedastic_head():
    head = GaussianRegressionHead.init(4, 1, prior_std=0.3, sigma_init=0.2)
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        head(jnp.ones(4))
    for name in _SITES:
        fn = tr[f"GaussianRegressionHead.{name}"]["fn"]
        base = fn.base_dist
        expected = math.log(0.2) if name == "b_log_scale" else 0.0
        np.testing.assert_allclose(base.loc, expected, atol=1e-7)
        np.testing.assert_allclose(base.scale, 0.3)


@pytest.mark.parametrize(
    ("shape", "out"), [((4,), (2,)), ((5, 4), (5, 2)), ((3, 5, 4), (3, 5, 2))]
)
def test_output_shapes(shape, out):
    head = GaussianRegressionHead.init(4, 2)
    with handlers.seed(rng_seed=0):
        mu, log_sigma = head(jnp.ones(shape))
    assert mu.shape == out
    assert log_sigma.shape == out


def test_weights_are_sampled_once_per_call():
    """Every batch row sees the same weight draw: equal inputs, equal outputs."""
    head = GaussianRegressionHead.init(4, 1)
    with handlers.seed(rng_seed=0):
        mu, log_sigma = head(jnp.ones((6, 4)))
    assert jnp.all(mu == mu[0])
    assert jnp.all(log_sigma == log_sigma[0])


@pytest.mark.parametrize("sigma_init", [0.05, 1.0, 3.0])
def test_head_at_prior_location_is_homoscedastic_at_sigma_init(sigma_init):
    head = GaussianRegressionHead.init(6, 2, sigma_init=sigma_init)
    h = 5.0 * jr.normal(jr.PRNGKey(0), (40, 6))
    with handlers.substitute(data=_core_values(head)):
        mu, log_sigma = head(h)
    # b_s = log σ_init stored in float64 (x64 suite): exact to round-off.
    np.testing.assert_allclose(log_sigma, math.log(sigma_init), atol=1e-12)
    np.testing.assert_allclose(mu, 0.0, atol=1e-12)


def test_substituted_weights_match_the_deterministic_core():
    head = GaussianRegressionHead.init(6, 2, pyrox_name="h")
    core = geonnax.GaussianRegressionHead.init(
        6, 2, key=jr.PRNGKey(1), scale_init_factor=0.5
    )
    data = {f"h.{n}": getattr(core, n) for n in _SITES}
    x = jr.normal(jr.PRNGKey(2), (7, 6))
    with handlers.substitute(data=data):
        mu, log_sigma = head(x)
    mu_ref, log_sigma_ref = jax.vmap(core)(x)
    # Same float ops on the same arrays: exact up to round-off.
    np.testing.assert_allclose(mu, mu_ref, atol=1e-12)
    np.testing.assert_allclose(log_sigma, log_sigma_ref, atol=1e-12)


def test_log_sigma_is_clamped():
    head = GaussianRegressionHead.init(2, 2, pyrox_name="h")
    data = _core_values(head) | {"h.b_log_scale": jnp.array([-50.0, 50.0])}
    with handlers.substitute(data=data):
        _, log_sigma = head(jnp.ones(2))
    np.testing.assert_allclose(log_sigma, [-7.0, 3.0])


def test_named_siblings_get_distinct_sites_and_default_names_collide():
    a = GaussianRegressionHead.init(3, pyrox_name="a")
    b = GaussianRegressionHead.init(3, pyrox_name="b")
    x = jnp.ones((2, 3))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        a(x)
        b(x)
    assert "a.W_loc" in tr and "b.W_loc" in tr

    c, d = GaussianRegressionHead.init(3), GaussianRegressionHead.init(3)
    with pytest.raises(AssertionError), handlers.trace(), handlers.seed(rng_seed=0):
        c(x)
        d(x)


def test_log_density_jit_and_grad():
    head = GaussianRegressionHead.init(3, 1, pyrox_name="h")
    h = jr.normal(jr.PRNGKey(0), (16, 3))
    y = jr.normal(jr.PRNGKey(1), (16,))

    def model():
        mu, log_sigma = head(h)
        numpyro.sample("y", dist.Normal(mu[:, 0], jnp.exp(log_sigma[:, 0])), obs=y)

    @jax.jit
    def lp(params):
        return log_density(model, (), {}, params)[0]

    params = _core_values(head)
    grads = jax.grad(lp)(params)
    for g in grads.values():
        assert jnp.all(jnp.isfinite(g))
    # A zero W_s still gets gradient: d log σ / d W_s = h.
    assert jnp.any(grads["h.W_log_scale"] != 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prior_std": 0.0},
        {"in_features": 0},
        {"out_features": 0},
        {"sigma_init": 1e-4},  # log σ_init < -7
        {"sigma_init": 100.0},  # log σ_init > 3
    ],
)
def test_init_validation(kwargs):
    args = {"in_features": 3, "out_features": 1} | kwargs
    with pytest.raises(ValueError):
        GaussianRegressionHead.init(**args)


def test_svi_and_predictive_smoke():
    head = GaussianRegressionHead.init(3, 1)
    h = jr.normal(jr.PRNGKey(0), (20, 3))
    y = jr.normal(jr.PRNGKey(1), (20,))

    def model(h, y=None):
        mu, log_sigma = head(h)
        with numpyro.plate("data", h.shape[0]):
            numpyro.sample("y", dist.Normal(mu[:, 0], jnp.exp(log_sigma[:, 0])), obs=y)

    guide = AutoNormal(model, init_loc_fn=init_to_median)
    svi = SVI(model, guide, optax.adam(1e-2), Trace_ELBO())
    res = svi.run(jr.PRNGKey(2), 20, h, y, progress_bar=False)
    assert jnp.all(jnp.isfinite(res.losses))
    pred = Predictive(model, guide=guide, params=res.params, num_samples=4)
    assert pred(jr.PRNGKey(3), h)["y"].shape == (4, 20)


@pytest.mark.slow
def test_recovers_known_noise_scale_on_synthetic_1d_data():
    r"""Fit σ(x) = 0.1 + 0.5·|x| and check the predictive intervals.

    Data: x ~ U(-1, 1), y = sin(πx) + σ(x) ε with N = 2000; the features are
    15 Gaussian RBFs (spacing = width = 1/7), so both branches of the head
    are linear in the same basis. Mean-field VI from the homoscedastic prior
    median (``init_to_median``), then the 90 % posterior-predictive interval
    from 400 draws per test point.
    """
    m = 15
    centers = jnp.linspace(-1.0, 1.0, m)[:, None]
    widths = jnp.full((m,), 2.0 / (m - 1))

    def feats(x):
        return geonnax.basis.rbf_basis(x[:, None], centers, widths)

    def sigma(x):
        return 0.1 + 0.5 * jnp.abs(x)

    def draw(key, n):
        kx, ke = jr.split(key)
        x = jr.uniform(kx, (n,), minval=-1.0, maxval=1.0)
        return x, jnp.sin(jnp.pi * x) + sigma(x) * jr.normal(ke, (n,))

    head = GaussianRegressionHead.init(m, 1, prior_std=1.0, sigma_init=0.5)

    def model(h, y=None):
        mu, log_sigma = head(h)
        with numpyro.plate("data", h.shape[0]):
            numpyro.sample("y", dist.Normal(mu[:, 0], jnp.exp(log_sigma[:, 0])), obs=y)

    x, y = draw(jr.PRNGKey(0), 2000)
    guide = AutoNormal(model, init_loc_fn=init_to_median, init_scale=0.01)
    svi = SVI(model, guide, optax.adam(1e-2), Trace_ELBO())
    res = svi.run(jr.PRNGKey(1), 3000, feats(x), y, progress_bar=False)

    # Learned σ(x) at the posterior median against the truth. Each RBF sees
    # about n_eff ≈ 2000 · (2/7) / 2 ≈ 290 points, and the MLE of log σ from
    # n points has sd 1 / sqrt(2n) ≈ 0.04; 0.2 is about 5 sd. Within one RBF
    # width of x = 0 the smooth basis cannot follow the kink of |x|, so that
    # band is excluded (measured 0.25 there, 0.14 elsewhere).
    grid = jnp.linspace(-0.95, 0.95, 39)
    with handlers.substitute(data=guide.median(res.params)):
        _, log_sigma = head(feats(grid))
    err = jnp.abs(log_sigma[:, 0] - jnp.log(sigma(grid)))
    assert float(jnp.max(err[jnp.abs(grid) > 1.0 / 7.0])) < 0.2
    # The intervals widen away from the coast-like centre (truth: 5.5×).
    assert float(jnp.exp(log_sigma[-1, 0] - log_sigma[19, 0])) > 3.0

    # Coverage of the 90 % interval on fresh test points. For a calibrated
    # interval the hit count is Binomial(n, 0.9), sd sqrt(0.09 / n); allow
    # 4 sd: 0.019 for n = 4000, about 0.033 in each third-of-range bin
    # (n ≈ 1330). A homoscedastic fit (σ ≈ 0.38 everywhere) would cover
    # ≈ 1.0 in the inner bin and ≈ 0.75 in the outer one, far outside.
    xt, yt = draw(jr.PRNGKey(2), 4000)
    pred = Predictive(model, guide=guide, params=res.params, num_samples=400)
    ys = pred(jr.PRNGKey(3), feats(xt))["y"]
    lo, hi = jnp.quantile(ys, 0.05, axis=0), jnp.quantile(ys, 0.95, axis=0)
    hit = (yt >= lo) & (yt <= hi)
    for mask in (
        jnp.ones_like(hit),
        jnp.abs(xt) < 1.0 / 3.0,
        jnp.abs(xt) > 2.0 / 3.0,
    ):
        n = int(mask.sum())
        bound = 4.0 * math.sqrt(0.9 * 0.1 / n)
        assert abs(float(hit[mask].mean()) - 0.9) < bound, (n, hit[mask].mean())
