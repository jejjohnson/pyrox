"""Tests for the renewal, Hawkes, marked, and thinning operators.

Covers the same "live-value" invariants as the IPP operator PR — updates
to PyTree leaves must flow into log-prob, sampling, and intensity
without requiring reconstruction of the operator.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from xtremax.point_processes import (
    EventHistory,
    constant_mark_distribution,
    constant_retention,
)
from xtremax.point_processes.operators import (
    ExponentialHawkes,
    ExponentialKernel,
    GeneralHawkesProcess,
    HomogeneousPoissonProcess,
    MarkedTemporalPointProcess,
    RenewalProcess,
    ThinningProcess,
)


class TestRenewalOperator:
    def test_exponential_renewal_matches_hpp(self):
        event_times = jnp.array([0.5, 1.8, 2.7, 3.9, 4.5])
        mask = jnp.array([True, True, True, True, True])
        T = 5.0
        rate = 2.0

        renewal = RenewalProcess(dist.Exponential(rate), observation_window=T)
        ll_r = float(renewal.log_prob(event_times, mask))

        hpp = HomogeneousPoissonProcess(rate, T)
        ll_h = float(hpp.log_prob(jnp.sum(mask)))
        assert ll_r == pytest.approx(ll_h, rel=1e-5)

    def test_jit_log_prob(self):
        op = RenewalProcess(dist.Exponential(2.0), observation_window=5.0)
        t, m, _ = op.sample(jax.random.PRNGKey(0), max_events=64)
        jit_ll = eqx.filter_jit(RenewalProcess.log_prob)
        assert float(abs(jit_ll(op, t, m) - op.log_prob(t, m))) < 1e-5


class TestExponentialHawkes:
    def test_alpha_zero_matches_hpp(self):
        event_times = jnp.array([0.3, 1.1, 2.0, 2.8])
        mask = jnp.array([True, True, True, True])
        T = 4.0
        mu = 0.6

        hawkes = ExponentialHawkes(mu=mu, alpha=0.0, beta=1.0, observation_window=T)
        hpp = HomogeneousPoissonProcess(mu, T)
        assert float(hawkes.log_prob(event_times, mask)) == pytest.approx(
            float(hpp.log_prob(jnp.sum(mask))), rel=1e-5
        )

    def test_matches_general_hawkes_with_exponential_kernel(self):
        event_times = jnp.array([0.3, 1.1, 2.0, 2.8])
        mask = jnp.array([True, True, True, True])
        T = 4.0
        mu, alpha, beta = 0.5, 0.3, 1.0

        exp_h = ExponentialHawkes(mu=mu, alpha=alpha, beta=beta, observation_window=T)
        gen_h = GeneralHawkesProcess(
            mu=mu,
            kernel=ExponentialKernel(alpha=jnp.asarray(alpha), beta=jnp.asarray(beta)),
            observation_window=T,
        )
        ll_exp = float(exp_h.log_prob(event_times, mask))
        ll_gen = float(gen_h.log_prob(event_times, mask))
        assert ll_exp == pytest.approx(ll_gen, rel=1e-4)

    def test_live_lambda_max_tracks_alpha(self):
        """After mutating α via eqx.tree_at, the thinning envelope must update."""
        original = ExponentialHawkes(
            mu=0.5, alpha=0.3, beta=1.0, observation_window=10.0
        )
        history = EventHistory.empty(max_events=8)
        # Append a few "events" so N(t) > 0.
        h = history
        for t in (0.5, 1.0, 2.0):
            h = h.append(jnp.asarray(t))
        bound_orig = float(original.effective_lambda_max(h))
        updated = eqx.tree_at(lambda op: op.alpha, original, jnp.asarray(0.8))
        bound_new = float(updated.effective_lambda_max(h))
        # With three prior events: bound = μ + α·3. Old 0.5+0.9=1.4, new 0.5+2.4=2.9.
        assert bound_orig == pytest.approx(1.4, rel=1e-4)
        assert bound_new == pytest.approx(2.9, rel=1e-4)


class TestGeneralHawkesOperator:
    def test_sample_with_exponential_kernel_in_window(self):
        kernel = ExponentialKernel(alpha=jnp.asarray(0.3), beta=jnp.asarray(1.0))
        op = GeneralHawkesProcess(mu=0.5, kernel=kernel, observation_window=10.0)
        t, m, _ = op.sample(jax.random.PRNGKey(0), max_events=256)
        real = t[m]
        assert jnp.all(real >= 0.0) and jnp.all(real <= 10.0)


class TestThinningOperator:
    def test_full_retention_is_base_log_prob(self):
        hpp = HomogeneousPoissonProcess(2.0, 10.0)
        thin = ThinningProcess(base=hpp, retention_fn=constant_retention(1.0))
        t, m, _ = thin.sample(jax.random.PRNGKey(0), max_events=128)
        ll_thin = float(thin.log_prob(t, m))
        ll_base = float(hpp.log_prob(jnp.sum(m)))
        assert ll_thin == pytest.approx(ll_base, rel=1e-5)

    def test_half_retention_halves_expected_count(self):
        rate = 3.0
        T = 10.0
        hpp = HomogeneousPoissonProcess(rate, T)
        thin = ThinningProcess(base=hpp, retention_fn=constant_retention(0.4))
        keys = jax.random.split(jax.random.PRNGKey(0), 300)
        counts = jax.vmap(lambda k: jnp.sum(thin.sample(k, max_events=128)[1]))(keys)
        assert float(jnp.mean(counts)) == pytest.approx(0.4 * rate * T, abs=0.5)

    def test_live_retention_parameters_flow_into_log_prob(self):
        """Mutating a retention-fn leaf via tree_at must change log_prob."""

        class ConstantRetention(eqx.Module):
            p: jnp.ndarray

            def __call__(self, t, history, proposed_mark=None):
                return self.p

        hpp = HomogeneousPoissonProcess(2.0, 10.0)
        thin = ThinningProcess(
            base=hpp, retention_fn=ConstantRetention(p=jnp.asarray(0.7))
        )
        t, m, _ = hpp.sample(jax.random.PRNGKey(0), max_events=64)
        ll0 = float(thin.log_prob(t, m))
        thin_low = eqx.tree_at(lambda op: op.retention_fn.p, thin, jnp.asarray(0.3))
        ll1 = float(thin_low.log_prob(t, m))
        assert ll0 != ll1


class TestMarkedOperator:
    def test_ground_alone_matches_mark_uniform(self):
        """Marked with a constant Normal(0,1) should equal ground + Σ log N(m; 0, 1)."""
        hpp = HomogeneousPoissonProcess(2.0, 5.0)
        mtpp = MarkedTemporalPointProcess(
            ground=hpp,
            mark_distribution_fn=constant_mark_distribution(dist.Normal(0.0, 1.0)),
            mark_dim=None,
            history_at_each_event=False,
        )
        t, m, marks = mtpp.sample(jax.random.PRNGKey(0), max_events=64)
        ll_joint = float(mtpp.log_prob(t, m, marks))
        ll_ground = float(hpp.log_prob(jnp.sum(m)))
        ll_marks = float(jnp.sum(dist.Normal(0.0, 1.0).log_prob(marks[m])))
        assert ll_joint == pytest.approx(ll_ground + ll_marks, rel=1e-4)
