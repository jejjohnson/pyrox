"""`noncentred_log_normal`: a per-site log-normal parameter, non-centred.

log θ_n = m + τ z_n with z_n ~ N(0, 1) has the distribution of the centred
log θ_n ~ N(m, τ²); the sampled site is z_n and θ_n is a deterministic site.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive
from pyrox_ssm import (
    MaternSDE,
    markov_gp_sites_factor,
    noncentred_log_normal,
    pad_sites,
)


jax.config.update("jax_enable_x64", True)


def test_site_names_under_a_plate() -> None:
    def model():
        with numpyro.plate("site", 4):
            noncentred_log_normal("lengthscale", jnp.log(7.0), 0.5)

    tr = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace()
    assert tr["lengthscale_z"]["type"] == "sample"
    assert tr["lengthscale_z"]["value"].shape == (4,)
    assert tr["lengthscale"]["type"] == "deterministic"
    assert jnp.allclose(
        tr["lengthscale"]["value"],
        jnp.exp(jnp.log(7.0) + 0.5 * tr["lengthscale_z"]["value"]),
    )


def test_matches_the_centred_log_normal_in_distribution() -> None:
    """The mean and standard deviation of log θ are m and τ, to five standard errors."""
    n, m, tau = 20_000, 0.7, 0.3

    def model():
        with numpyro.plate("site", n):
            noncentred_log_normal("theta", m, tau)

    draws = Predictive(model, num_samples=1)(jr.key(0))["theta"][0]
    log_theta = jnp.log(draws)
    # SE of the mean τ/√n, of the standard deviation about τ/√(2n)
    assert abs(float(log_theta.mean()) - m) < 5 * tau / jnp.sqrt(n)
    assert abs(float(log_theta.std()) - tau) < 5 * tau / jnp.sqrt(2 * n)
    centred = dist.LogNormal(m, tau)
    assert abs(float(draws.mean()) - float(centred.mean)) < 5 * float(
        jnp.sqrt(centred.variance / n)
    )


def test_pooled_sites_model_runs_under_nuts_smoke() -> None:
    t, y, mask = pad_sites(
        [jnp.arange(8.0), jnp.arange(5.0)], [jnp.sin(jnp.arange(8.0)), jnp.zeros(5)]
    )

    def model():
        loc = numpyro.sample("lengthscale_loc", dist.Normal(0.0, 1.0))
        scale = numpyro.sample("lengthscale_scale", dist.HalfNormal(0.5))
        with numpyro.plate("site", 2):
            ell = noncentred_log_normal("lengthscale", loc, scale)
        kernel = MaternSDE(1.0, ell, order=1)
        markov_gp_sites_factor("y", kernel, t, y, jnp.asarray(0.1), mask=mask)

    mcmc = MCMC(NUTS(model), num_warmup=5, num_samples=5, progress_bar=False)
    mcmc.run(jr.key(1))
    samples = mcmc.get_samples()
    assert samples["lengthscale"].shape == (5, 2)
    assert jnp.all(samples["lengthscale"] > 0)
