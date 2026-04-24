"""Tests for NumPyro wrappers around the temporal PP operators."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random
from numpyro.infer import MCMC, NUTS

from xtremax.point_processes.distributions import (
    HomogeneousPoissonProcess,
    InhomogeneousPoissonProcess,
)


class TestHppDistribution:
    def test_arg_constraints(self):
        hpp = HomogeneousPoissonProcess(rate=1.0, observation_window=5.0)
        assert "rate" in hpp.arg_constraints
        assert "observation_window" in hpp.arg_constraints

    def test_sample_returns_times_and_mask(self):
        # NumPyro contract: log_prob(sample(...)) must round-trip, so sample
        # must return exactly what log_prob accepts.
        hpp = HomogeneousPoissonProcess(rate=1.0, observation_window=5.0, max_events=64)
        value = hpp.sample(random.PRNGKey(0))
        assert len(value) == 2
        times, mask = value
        assert times.shape == (64,)
        assert mask.shape == (64,)

    def test_log_prob_round_trips_on_sample_output(self):
        # Guards the regression reported in PR review.
        hpp = HomogeneousPoissonProcess(rate=1.5, observation_window=3.0, max_events=64)
        value = hpp.sample(random.PRNGKey(0))
        log_p = hpp.log_prob(value)
        assert jnp.isfinite(log_p)

    def test_log_prob_with_mask(self):
        hpp = HomogeneousPoissonProcess(rate=2.0, observation_window=5.0, max_events=64)
        times = jnp.zeros(64)
        mask = jnp.concatenate(
            [jnp.ones(3, dtype=jnp.bool_), jnp.zeros(61, dtype=jnp.bool_)]
        )
        log_p = hpp.log_prob((times, mask))
        expected = 3 * jnp.log(2.0) - 2.0 * 5.0
        assert jnp.allclose(log_p, expected)

    def test_log_prob_with_count(self):
        hpp = HomogeneousPoissonProcess(rate=2.0, observation_window=5.0)
        n = jnp.asarray(3)
        log_p = hpp.log_prob((jnp.zeros(10), n))
        expected = 3 * jnp.log(2.0) - 2.0 * 5.0
        assert jnp.allclose(log_p, expected)

    def test_mcmc_recovers_rate(self):
        # Place a weak prior on λ, condition on a known count, recover via NUTS.
        observed_n = jnp.asarray(25)
        T = 5.0

        def model():
            rate = numpyro.sample("rate", dist.LogNormal(0.0, 1.0))
            numpyro.factor(
                "lik",
                HomogeneousPoissonProcess(rate, T).log_prob(
                    (jnp.zeros(10), observed_n)
                ),
            )

        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=200, num_samples=400, progress_bar=False)
        mcmc.run(random.PRNGKey(0))
        rate_post = mcmc.get_samples()["rate"]
        # Posterior mean should be close to 25 / 5 = 5.0.
        assert jnp.abs(jnp.mean(rate_post) - 5.0) < 0.7


class TestIppDistribution:
    def test_log_prob_matches_operator(self):
        rate = 2.0
        T = 5.0

        def fn(t):
            return jnp.full_like(t, jnp.log(rate))

        ipp = InhomogeneousPoissonProcess(
            log_intensity_fn=fn,
            observation_window=T,
            integrated_intensity=rate * T,
            lambda_max=rate,
            max_candidates=128,
        )
        times = jnp.array([1.0, 2.0, 3.0])
        mask = jnp.ones_like(times, dtype=jnp.bool_)
        log_p = ipp.log_prob((times, mask))
        expected = 3 * jnp.log(rate) - rate * T
        assert jnp.allclose(log_p, expected)

    def test_sample_without_lambda_max_errors(self):
        def fn(t):
            return jnp.zeros_like(t)

        ipp = InhomogeneousPoissonProcess(log_intensity_fn=fn, observation_window=1.0)
        with pytest.raises(ValueError):
            ipp.sample(random.PRNGKey(0))

    def test_sample_shape_nonempty_errors(self):
        def fn(t):
            return jnp.zeros_like(t)

        ipp = InhomogeneousPoissonProcess(
            log_intensity_fn=fn, observation_window=1.0, lambda_max=1.0
        )
        with pytest.raises(NotImplementedError):
            ipp.sample(random.PRNGKey(0), sample_shape=(3,))

    def test_reject_non_prng_key(self):
        ipp = InhomogeneousPoissonProcess(
            log_intensity_fn=lambda t: jnp.zeros_like(t),
            observation_window=1.0,
            lambda_max=1.0,
        )
        with pytest.raises(TypeError):
            ipp.sample(jnp.array([1.0, 2.0]))


class TestDistributionRegressionsFromPrReview:
    def test_support_is_dependent_for_pytree_samples(self):
        # PR review: ``constraints.real_vector`` was misleading since
        # samples are a PyTree of (times, mask). ``constraints.dependent``
        # is the sentinel NumPyro uses for non-standard supports.
        from numpyro.distributions import constraints as c

        hpp = HomogeneousPoissonProcess(rate=1.0, observation_window=2.0)
        assert hpp.support is c.dependent

        def fn(t):
            return jnp.zeros_like(t)

        ipp = InhomogeneousPoissonProcess(log_intensity_fn=fn, observation_window=1.0)
        assert ipp.support is c.dependent

    def test_ipp_sample_log_prob_round_trip(self):
        # Sample → log_prob contract for the IPP wrapper.
        def fn(t):
            return jnp.full_like(t, jnp.log(2.0))

        ipp = InhomogeneousPoissonProcess(
            log_intensity_fn=fn,
            observation_window=3.0,
            integrated_intensity=6.0,
            lambda_max=2.0,
            max_candidates=128,
        )
        value = ipp.sample(random.PRNGKey(0))
        assert len(value) == 2
        log_p = ipp.log_prob(value)
        assert jnp.isfinite(log_p)
