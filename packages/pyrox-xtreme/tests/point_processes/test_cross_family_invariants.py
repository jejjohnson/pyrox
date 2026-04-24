"""Cross-family invariants: key degeneracies collapse across TPP families.

Captures the load-bearing algebraic identities between families so that
refactors cannot break them silently:

* Renewal(Exponential(λ)) == HPP(λ) on any sequence.
* ExponentialHawkes(μ, α=0) == HPP(μ).
* ExponentialHawkes(μ, α, β) == GeneralHawkes with ExponentialKernel(α, β).
* Marked with uniform marks f(m) == ground-process log-prob +
  Σ log f(m_i).
* ThinningProcess(base, p=1) == base.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from xtremax.point_processes import constant_mark_distribution, constant_retention
from xtremax.point_processes.operators import (
    ExponentialHawkes,
    ExponentialKernel,
    GeneralHawkesProcess,
    HomogeneousPoissonProcess,
    MarkedTemporalPointProcess,
    RenewalProcess,
    ThinningProcess,
)


@pytest.fixture
def ev():
    times = jnp.array([0.3, 1.1, 2.0, 2.8, 3.9])
    mask = jnp.array([True, True, True, True, True])
    return times, mask


class TestCrossFamilyInvariants:
    def test_renewal_exp_equals_hpp(self, ev):
        times, mask = ev
        T = 5.0
        rate = 1.3
        ll_r = float(RenewalProcess(dist.Exponential(rate), T).log_prob(times, mask))
        ll_h = float(HomogeneousPoissonProcess(rate, T).log_prob(jnp.sum(mask)))
        assert ll_r == pytest.approx(ll_h, rel=1e-5)

    def test_exp_hawkes_alpha_zero_equals_hpp(self, ev):
        times, mask = ev
        T = 5.0
        mu = 0.7
        ll_hawkes = float(
            ExponentialHawkes(
                mu=mu, alpha=0.0, beta=1.0, observation_window=T
            ).log_prob(times, mask)
        )
        ll_hpp = float(HomogeneousPoissonProcess(mu, T).log_prob(jnp.sum(mask)))
        assert ll_hawkes == pytest.approx(ll_hpp, rel=1e-5)

    def test_exp_hawkes_equals_general_with_exp_kernel(self, ev):
        times, mask = ev
        T = 5.0
        mu, alpha, beta = 0.5, 0.3, 1.0
        ll_exp = float(
            ExponentialHawkes(
                mu=mu, alpha=alpha, beta=beta, observation_window=T
            ).log_prob(times, mask)
        )
        ll_gen = float(
            GeneralHawkesProcess(
                mu=mu,
                kernel=ExponentialKernel(
                    alpha=jnp.asarray(alpha), beta=jnp.asarray(beta)
                ),
                observation_window=T,
            ).log_prob(times, mask)
        )
        assert ll_exp == pytest.approx(ll_gen, rel=1e-4)

    def test_marked_with_uniform_marks_decomposes(self, ev):
        times, mask = ev
        T = 5.0
        rate = 1.3
        hpp = HomogeneousPoissonProcess(rate, T)
        marks = jnp.array([0.1, 0.2, -0.3, 0.7, -0.5])
        mtpp = MarkedTemporalPointProcess(
            ground=hpp,
            mark_distribution_fn=constant_mark_distribution(dist.Normal(0.0, 1.0)),
            mark_dim=None,
            history_at_each_event=False,
        )
        ll_joint = float(mtpp.log_prob(times, mask, marks))
        ll_ground = float(hpp.log_prob(jnp.sum(mask)))
        ll_marks = float(jnp.sum(dist.Normal(0.0, 1.0).log_prob(marks[mask])))
        assert ll_joint == pytest.approx(ll_ground + ll_marks, rel=1e-4)

    def test_thinning_full_retention_equals_base(self, ev):
        times, mask = ev
        T = 5.0
        hpp = HomogeneousPoissonProcess(1.3, T)
        thin = ThinningProcess(base=hpp, retention_fn=constant_retention(1.0))
        ll_thin = float(thin.log_prob(times, mask))
        ll_base = float(hpp.log_prob(jnp.sum(mask)))
        assert ll_thin == pytest.approx(ll_base, rel=1e-5)
