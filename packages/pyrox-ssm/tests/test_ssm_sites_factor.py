"""`markov_gp_sites_factor`: the summed per-site Markov-GP log marginal as one factor.

The reference is the sum of per-site `markov_gp_factor` log densities on
ragged sites padded by `pad_sites`; both run the same masked Kalman filter,
so they agree to ``rtol=1e-12`` under x64.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density
from pyrox_ssm import (
    MarkovGPPrior,
    MaternSDE,
    markov_gp_factor,
    markov_gp_sites_factor,
    pad_sites,
)


jax.config.update("jax_enable_x64", True)


def _ragged():
    times = [
        jnp.linspace(0.0, 4.0, 9),
        jnp.array([0.3, 1.0, 2.2, 2.9]),
        jnp.array([1.5]),
    ]
    ys = [jnp.sin(t) for t in times]
    return times, ys


def _sites_log_density(kernel, t, y, noise, mask):
    def model():
        markov_gp_sites_factor("y", kernel, t, y, noise, mask=mask)

    return log_density(model, (), {}, {})[0]


def _sum_of_site_factors(ells, t, y, noises, mask):
    total = 0.0
    for s in range(t.shape[0]):

        def model(s=s):
            prior = MarkovGPPrior(MaternSDE(0.7, ells[s], order=1), t[s])
            markov_gp_factor("y", prior, y[s], noises[s], mask=mask[s])

        total = total + log_density(model, (), {}, {})[0]
    return total


def test_equals_the_sum_of_per_site_factors() -> None:
    t, y, mask = pad_sites(*_ragged())
    ells = jnp.array([0.5, 1.2, 2.0])
    noises = jnp.array([0.05, 0.1, 0.2])
    kernel = MaternSDE(0.7, ells, order=1)  # variance shared, lengthscale per site
    got = _sites_log_density(kernel, t, y, noises, mask)
    want = _sum_of_site_factors(ells, t, y, noises, mask)
    assert jnp.allclose(got, want, rtol=1e-12)


def test_a_shared_noise_variance_broadcasts() -> None:
    t, y, mask = pad_sites(*_ragged())
    ells = jnp.array([0.5, 1.2, 2.0])
    kernel = MaternSDE(0.7, ells, order=1)
    got = _sites_log_density(kernel, t, y, jnp.asarray(0.1), mask)
    want = _sum_of_site_factors(ells, t, y, jnp.full(3, 0.1), mask)
    assert jnp.allclose(got, want, rtol=1e-12)


def test_registers_one_factor_site() -> None:
    t, y, mask = pad_sites(*_ragged())

    def model():
        with numpyro.plate("site", 3):
            ell = numpyro.sample("lengthscale", dist.LogNormal(0.0, 0.5))
        markov_gp_sites_factor(
            "y", MaternSDE(1.0, ell, order=1), t, y, jnp.asarray(0.1), mask=mask
        )

    tr = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace()
    assert [k for k, v in tr.items() if v["type"] == "sample"] == ["lengthscale", "y"]
    assert tr["y"]["fn"].log_prob(tr["y"]["value"]).shape == ()


def test_runs_under_nuts_smoke() -> None:
    t, y, mask = pad_sites(*_ragged())

    def model():
        with numpyro.plate("site", 3):
            ell = numpyro.sample("lengthscale", dist.LogNormal(0.0, 0.5))
        markov_gp_sites_factor(
            "y", MaternSDE(1.0, ell, order=1), t, y, jnp.asarray(0.1), mask=mask
        )

    mcmc = MCMC(NUTS(model), num_warmup=5, num_samples=5, progress_bar=False)
    mcmc.run(jr.key(0))
    assert mcmc.get_samples()["lengthscale"].shape == (5, 3)


def test_rejects_mismatched_shapes() -> None:
    t, y, mask = pad_sites(*_ragged())
    kernel = MaternSDE(1.0, jnp.ones(3), order=1)
    with pytest.raises(ValueError, match="leading axis of length S"):
        _sites_log_density(MaternSDE(1.0, jnp.ones(2), order=1), t, y, 0.1, mask)
    with pytest.raises(ValueError, match="noise_var"):
        _sites_log_density(kernel, t, y, jnp.ones(2), mask)
    with pytest.raises(ValueError, match="y has shape"):
        _sites_log_density(kernel, t, y[:, :-1], 0.1, mask)
    with pytest.raises(ValueError, match=r"times must be \(S, N\)"):
        _sites_log_density(kernel, t[0], y[0], 0.1, None)
