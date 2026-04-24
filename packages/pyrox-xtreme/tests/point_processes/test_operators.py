"""Tests for the ``equinox``-based temporal point process operators."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import random

from xtremax.point_processes.operators import (
    GoodnessOfFit,
    HomogeneousPoissonProcess,
    InhomogeneousPoissonProcess,
)
from xtremax.point_processes.primitives import (
    ipp_sample_inversion as ipp_sample_inversion_op,
)


class TestHppOperator:
    def test_log_prob_matches_closed_form(self):
        op = HomogeneousPoissonProcess(rate=2.0, observation_window=5.0)
        n = jnp.asarray(4)
        expected = 4 * jnp.log(2.0) - 2.0 * 5.0
        assert jnp.allclose(op.log_prob(n), expected)

    def test_sample_returns_padded_triple(self):
        op = HomogeneousPoissonProcess(rate=3.0, observation_window=4.0)
        times, mask, n = op.sample(random.PRNGKey(0), max_events=64)
        assert times.shape == (64,)
        assert mask.shape == (64,)
        assert mask.sum() <= n

    def test_operator_is_pytree(self):
        op = HomogeneousPoissonProcess(rate=1.0, observation_window=2.0)
        leaves = jax.tree_util.tree_leaves(op)
        assert len(leaves) == 2

    def test_is_optimisable_via_grad(self):
        # Fit rate to maximise log_prob of a known n_events over window T.
        op = HomogeneousPoissonProcess(
            rate=jnp.asarray(1.0), observation_window=jnp.asarray(5.0)
        )
        n_observed = jnp.asarray(20)

        def neg_log_lik(op):
            return -op.log_prob(n_observed)

        # Gradient w.r.t. the PyTree leaves (rate, observation_window).
        grad = eqx.filter_grad(neg_log_lik)(op)
        # ∂(-log L)/∂λ = -n/λ + T = -20 + 5 = -15. We take grad wrt op —
        # the rate leaf should hold that value.
        assert jnp.allclose(grad.rate, -15.0)

    def test_survival_at_given_time_is_one(self):
        op = HomogeneousPoissonProcess(rate=2.0, observation_window=10.0)
        assert jnp.allclose(op.survival(3.0, given_time=3.0), 1.0)

    def test_return_period_round_trip(self):
        op = HomogeneousPoissonProcess(rate=1.5, observation_window=10.0)
        tau = op.return_period(0.8)
        assert jnp.allclose(1.0 - jnp.exp(-1.5 * tau), 0.8)

    def test_goodness_of_fit_on_own_samples(self):
        op = HomogeneousPoissonProcess(rate=2.0, observation_window=20.0)
        times, mask, _ = op.sample(random.PRNGKey(0), max_events=128)
        gof = op.goodness_of_fit(times, mask)
        assert isinstance(gof, GoodnessOfFit)
        assert gof.residuals.shape == times.shape
        assert jnp.isfinite(gof.ks_statistic)


class TestIppOperator:
    def test_piecewise_constant_factory(self):
        op = InhomogeneousPoissonProcess.from_piecewise_constant(
            bin_edges=jnp.array([0.0, 2.0, 5.0]),
            rates=jnp.array([1.0, 4.0]),
        )
        # The factory now defers Λ(T) / λ_max to the intensity module
        # so gradient updates to log_rates stay consistent; the fields
        # themselves are None and the live values come from accessors.
        assert op.integrated_intensity is None
        assert op.lambda_max is None
        # Λ(T) = 1*2 + 4*3 = 14 via the live accessor.
        assert jnp.allclose(op.effective_integrated_intensity(), 14.0)
        # λ_max = 4 via the live accessor.
        assert jnp.allclose(op.effective_lambda_max(), 4.0)
        # Intensity at t=1 is 1, at t=3 is 4.
        assert jnp.allclose(op.intensity(jnp.asarray(1.0)), 1.0)
        assert jnp.allclose(op.intensity(jnp.asarray(3.0)), 4.0)

    def test_ipp_with_constant_matches_hpp(self):
        rate = 2.5
        T = 5.0

        def fn(t):
            return jnp.full_like(t, jnp.log(rate))

        ipp = InhomogeneousPoissonProcess(
            log_intensity_fn=fn,
            observation_window=T,
            integrated_intensity=rate * T,
            lambda_max=rate,
            n_integration_points=200,
        )
        hpp = HomogeneousPoissonProcess(rate=rate, observation_window=T)

        times = jnp.array([0.5, 1.0, 2.0, 3.5, 4.2])
        mask = jnp.ones_like(times, dtype=jnp.bool_)
        assert jnp.allclose(
            ipp.log_prob(times, mask),
            hpp.log_prob(jnp.asarray(5)),
        )

    def test_sample_requires_lambda_max(self):
        def fn(t):
            return jnp.full_like(t, 0.0)

        ipp = InhomogeneousPoissonProcess(log_intensity_fn=fn, observation_window=1.0)
        with pytest.raises(ValueError, match="lambda_max"):
            ipp.sample(random.PRNGKey(0), max_candidates=32)

    def test_integrated_intensity_computed_when_none(self):
        # If integrated_intensity is omitted, log_prob should still be finite.
        def fn(t):
            return jnp.log(1.0 + 0.5 * t)

        ipp = InhomogeneousPoissonProcess(
            log_intensity_fn=fn,
            observation_window=4.0,
            n_integration_points=400,
        )
        times = jnp.array([1.0, 2.0, 3.0])
        mask = jnp.ones_like(times, dtype=jnp.bool_)
        log_L = ipp.log_prob(times, mask)
        assert jnp.isfinite(log_L)

    def test_piecewise_sampling_is_jittable(self):
        op = InhomogeneousPoissonProcess.from_piecewise_constant(
            bin_edges=jnp.array([0.0, 2.0, 5.0]),
            rates=jnp.array([1.0, 4.0]),
        )
        jitted = jax.jit(lambda k: op.sample(k, max_candidates=256))
        times, _mask, _ = jitted(random.PRNGKey(0))
        assert times.shape == (256,)

    def test_inversion_sampling(self):
        rate = 2.0
        T = 5.0

        def fn(t):
            return jnp.full_like(t, jnp.log(rate))

        def inv_cum(y):
            return y / rate

        ipp = InhomogeneousPoissonProcess(
            log_intensity_fn=fn,
            observation_window=T,
            integrated_intensity=rate * T,
        )
        times, mask, _n = ipp.sample_inversion(
            random.PRNGKey(0), inv_cum, max_events=128
        )
        valid = times[mask]
        assert jnp.all(jnp.diff(valid) >= -1e-7)

    def test_goodness_of_fit_on_piecewise_sample(self):
        op = InhomogeneousPoissonProcess.from_piecewise_constant(
            bin_edges=jnp.array([0.0, 5.0, 10.0]),
            rates=jnp.array([2.0, 5.0]),
            n_integration_points=400,
        )

        # Use inversion sampling for exactness with piecewise constant.
        # Λ(t) piecewise linear. Inverse:
        # y in [0, 10] → t = y/2; y in [10, 35] → t = 5 + (y-10)/5.
        def inv_cum(y):
            t_bin1 = y / 2.0
            t_bin2 = 5.0 + (y - 10.0) / 5.0
            return jnp.where(y <= 10.0, t_bin1, t_bin2)

        times, mask, _ = op.sample_inversion(random.PRNGKey(2), inv_cum, max_events=512)
        gof = op.goodness_of_fit(times, mask)
        assert jnp.isfinite(gof.ks_statistic)


class TestRegressionsFromPrReview:
    """Regression tests guarding specific PR-review findings on PR #10."""

    def test_neural_intensity_module_is_trainable(self):
        # PR review: declaring log_intensity_fn with eqx.field(static=True)
        # removed it from the PyTree, so an `eqx.Module` intensity's
        # parameters couldn't be optimised. Check that grads flow into
        # the intensity module's array leaves.
        import equinox as eqx

        class LinearLogIntensity(eqx.Module):
            a: jnp.ndarray
            b: jnp.ndarray

            def __call__(self, t):
                return self.a * t + self.b

        fn = LinearLogIntensity(a=jnp.asarray(0.2), b=jnp.asarray(0.5))
        ipp = InhomogeneousPoissonProcess(
            log_intensity_fn=fn,
            observation_window=jnp.asarray(5.0),
            n_integration_points=200,
        )
        times = jnp.array([0.5, 1.0, 2.5, 3.8])
        mask = jnp.ones_like(times, dtype=jnp.bool_)

        grads = eqx.filter_grad(lambda m: -m.log_prob(times, mask))(ipp)
        # The intensity module's `a` and `b` leaves should receive
        # finite non-zero gradients; if `log_intensity_fn` were static,
        # filter_grad would return None for these leaves.
        assert grads.log_intensity_fn.a is not None
        assert grads.log_intensity_fn.b is not None
        assert jnp.isfinite(grads.log_intensity_fn.a)
        assert jnp.isfinite(grads.log_intensity_fn.b)

    def test_piecewise_log_intensity_is_pytree_with_leaves(self):
        # PR review: from_piecewise_constant used to build a Python
        # closure, hiding `bin_edges`/`rates` from the PyTree. They
        # should now appear as leaves.
        import equinox as eqx

        from xtremax.point_processes.operators import PiecewiseConstantLogIntensity

        op = InhomogeneousPoissonProcess.from_piecewise_constant(
            bin_edges=jnp.array([0.0, 2.0, 5.0]),
            rates=jnp.array([1.0, 4.0]),
        )
        assert isinstance(op.log_intensity_fn, PiecewiseConstantLogIntensity)
        leaves = jax.tree_util.tree_leaves(op.log_intensity_fn)
        # bin_edges (shape (3,)) + log_rates (shape (2,)) = 2 array leaves.
        assert len(leaves) == 2

        # Gradients should flow through log_rates.
        times = jnp.array([1.0, 3.0, 4.0])
        mask = jnp.ones_like(times, dtype=jnp.bool_)

        grads = eqx.filter_grad(lambda m: -m.log_prob(times, mask))(op)
        assert grads.log_intensity_fn.log_rates is not None
        assert jnp.all(jnp.isfinite(grads.log_intensity_fn.log_rates))

    def test_inversion_sampler_mean_not_biased_early(self):
        # PR review: ipp_sample_inversion had the same order-statistic
        # bias that was fixed in thinning. With constant rate on [0, T],
        # accepted event times should be ~ Uniform(0, T) — check the
        # empirical mean averaged across seeds.
        rate = 2.0
        T = 10.0
        Lambda_T = rate * T

        def inv_cum(y):
            return y / rate

        def once(k):
            times, mask, _ = ipp_sample_inversion_op(
                k, inv_cum, Lambda_T, max_events=512
            )
            return jnp.sum(jnp.where(mask, times, 0.0)), jnp.sum(mask)

        keys = random.split(random.PRNGKey(0), 400)
        sums, counts = jax.vmap(once)(keys)
        mean_time = jnp.sum(sums) / jnp.sum(counts)
        # Unbiased mean is T/2 = 5.0; order-statistic bias would pull
        # this well below 5.0 (toward ~n/(max_events+1) * T).
        assert jnp.abs(mean_time - 0.5 * T) < 0.3


class TestLiveLambdaAndIntegrated:
    """Guards PR #10 second-round review: cached Λ(T) / λ_max must
    not go stale when the intensity parameters are updated."""

    def test_log_prob_tracks_live_log_rates(self):
        # After mutating log_rates via eqx.tree_at, log_prob's
        # -Λ(T) term must reflect the new rates — otherwise gradients
        # would miss that term entirely.
        import equinox as eqx

        op = InhomogeneousPoissonProcess.from_piecewise_constant(
            bin_edges=jnp.array([0.0, 5.0, 10.0]),
            rates=jnp.array([1.0, 1.0]),  # uniform λ=1, Λ(T)=10
        )
        times = jnp.array([1.0, 2.0, 3.0])
        mask = jnp.ones_like(times, dtype=jnp.bool_)

        before = op.log_prob(times, mask)
        # Λ(T) = 10 and sum log λ = 0, so log_prob = -10.
        assert jnp.allclose(before, -10.0, atol=1e-4)

        # Double the rates.
        new_log_rates = op.log_intensity_fn.log_rates + jnp.log(2.0)
        op2 = eqx.tree_at(lambda m: m.log_intensity_fn.log_rates, op, new_log_rates)
        after = op2.log_prob(times, mask)
        # Now λ=2 everywhere, Λ(T)=20, sum log λ = 3 log 2.
        expected = 3 * jnp.log(2.0) - 20.0
        assert jnp.allclose(after, expected, atol=1e-4)

    def test_grad_through_log_rates_flows_into_compensator_term(self):
        # ∂log L / ∂log_rates[b] = #events_in_bin(b) - rate(b) * width(b).
        # Without the stale-cache fix the second term drops out.
        import equinox as eqx

        op = InhomogeneousPoissonProcess.from_piecewise_constant(
            bin_edges=jnp.array([0.0, 5.0]),
            rates=jnp.array([1.0]),
        )
        # Put 3 events inside the single bin of width 5.
        times = jnp.array([0.5, 1.5, 3.2])
        mask = jnp.ones_like(times, dtype=jnp.bool_)

        def neg_log_lik(m):
            return -m.log_prob(times, mask)

        grads = eqx.filter_grad(neg_log_lik)(op)
        # d/d(log r) [n log r - r * w] at r=1, n=3, w=5 ⇒ -(3 - 5) = 2.
        # A stale cache would yield -(3 - 0) = -3 here.
        grad_log_rates = grads.log_intensity_fn.log_rates
        assert jnp.allclose(grad_log_rates, jnp.asarray([2.0]), atol=1e-4)

    def test_effective_lambda_max_tracks_log_rates(self):
        import equinox as eqx

        op = InhomogeneousPoissonProcess.from_piecewise_constant(
            bin_edges=jnp.array([0.0, 1.0, 2.0]),
            rates=jnp.array([1.0, 2.0]),
        )
        assert jnp.allclose(op.effective_lambda_max(), 2.0)

        # Quadruple the second bin's rate.
        new_log_rates = jnp.array([jnp.log(1.0), jnp.log(8.0)])
        op2 = eqx.tree_at(lambda m: m.log_intensity_fn.log_rates, op, new_log_rates)
        assert jnp.allclose(op2.effective_lambda_max(), 8.0)

    def test_pinned_integrated_intensity_is_still_respected(self):
        # If the user pins a fixed Λ(T), it takes precedence (this is
        # the explicit "I know, don't touch" path).
        def fn(t):
            return jnp.zeros_like(t)

        op = InhomogeneousPoissonProcess(
            log_intensity_fn=fn,
            observation_window=1.0,
            integrated_intensity=123.0,
        )
        assert jnp.allclose(op.effective_integrated_intensity(), 123.0)

    def test_sample_uses_live_lambda_max(self):
        # After bumping log_rates, thinning should draw the new number
        # of candidates (≈ λ_max_new · T instead of the old bound).
        import equinox as eqx

        op = InhomogeneousPoissonProcess.from_piecewise_constant(
            bin_edges=jnp.array([0.0, 10.0]),
            rates=jnp.array([1.0]),
        )
        new_log_rates = jnp.full((1,), jnp.log(5.0))
        op2 = eqx.tree_at(lambda m: m.log_intensity_fn.log_rates, op, new_log_rates)
        # With T=10 and λ=5, expected n_candidates ≈ 50.
        keys = random.split(random.PRNGKey(0), 200)
        counts = jax.vmap(lambda k: op2.sample(k, max_candidates=256)[2])(keys)
        mean_count = jnp.mean(counts.astype(jnp.float32))
        assert jnp.abs(mean_count - 50.0) < 3.0
