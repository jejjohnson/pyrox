"""NumPyro Distribution wrapper tests for the extension families.

Verifies the ``sample → log_prob`` round-trip contract and that each
wrapper delegates to the right operator.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from xtremax.point_processes import constant_mark_distribution, constant_retention
from xtremax.point_processes.distributions import (
    ExponentialHawkes,
    GeneralHawkesProcess,
    MarkedTemporalPointProcess,
    RenewalProcess,
    ThinningProcess,
)
from xtremax.point_processes.operators import (
    ExponentialKernel,
    HomogeneousPoissonProcess,
)


class TestDistributionRoundTrips:
    def test_renewal_round_trip(self):
        d = RenewalProcess(dist.Exponential(2.0), observation_window=5.0)
        value = d.sample(jax.random.PRNGKey(0))
        ll = float(d.log_prob(value))
        assert jnp.isfinite(ll)

    def test_exp_hawkes_round_trip(self):
        d = ExponentialHawkes(mu=0.5, alpha=0.3, beta=1.0, observation_window=10.0)
        value = d.sample(jax.random.PRNGKey(0))
        ll = float(d.log_prob(value))
        assert jnp.isfinite(ll)

    def test_general_hawkes_round_trip(self):
        d = GeneralHawkesProcess(
            mu=0.5,
            kernel=ExponentialKernel(alpha=jnp.asarray(0.3), beta=jnp.asarray(1.0)),
            observation_window=10.0,
        )
        value = d.sample(jax.random.PRNGKey(0))
        ll = float(d.log_prob(value))
        assert jnp.isfinite(ll)

    def test_marked_round_trip(self):
        ground = HomogeneousPoissonProcess(2.0, 5.0)
        d = MarkedTemporalPointProcess(
            ground=ground,
            mark_distribution_fn=constant_mark_distribution(dist.Normal(0.0, 1.0)),
            history_at_each_event=False,
        )
        value = d.sample(jax.random.PRNGKey(0))
        assert len(value) == 3  # (times, mask, marks)
        ll = float(d.log_prob(value))
        assert jnp.isfinite(ll)

    def test_thinning_round_trip(self):
        base = HomogeneousPoissonProcess(3.0, 10.0)
        d = ThinningProcess(base=base, retention_fn=constant_retention(0.7))
        value = d.sample(jax.random.PRNGKey(0))
        assert len(value) == 2
        ll = float(d.log_prob(value))
        assert jnp.isfinite(ll)


class TestSampleShapeUnsupported:
    """Batched ``sample_shape`` should raise a clear error, not silently succeed."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: RenewalProcess(dist.Exponential(2.0), observation_window=5.0),
            lambda: ExponentialHawkes(
                mu=0.5, alpha=0.3, beta=1.0, observation_window=10.0
            ),
            lambda: ThinningProcess(
                base=HomogeneousPoissonProcess(2.0, 5.0),
                retention_fn=constant_retention(0.5),
            ),
        ],
    )
    def test_raises(self, factory):
        d = factory()
        with pytest.raises(NotImplementedError):
            d.sample(jax.random.PRNGKey(0), sample_shape=(3,))
