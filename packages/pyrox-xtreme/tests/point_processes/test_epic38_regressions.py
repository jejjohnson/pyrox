"""Regression tests for the point-process correctness epic (#38).

One test class per fixed issue:

* #55 — ThinningProcess.log_prob compensator sign.
* #56 — GeneralHawkes vector-t cumulative_intensity / GOF parity.
* #57 — temporal marks_log_prob padding NaN gradients.
* #58 — temporal Hawkes dt clipping + timestamp causality.
* #59 — contiguous-mask invariant / mask-robust residuals.
* #60 — renewal_expected_count truncation.
* #61 — ThinningProcess.sample with an IPP base.
* #62 — IPP integral-method consistency.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from xtremax.point_processes.operators import (
    ExponentialHawkes,
    ExponentialKernel,
    GeneralHawkesProcess,
    InhomogeneousPoissonProcess,
    ThinningProcess,
)
from xtremax.point_processes.primitives import (
    exp_hawkes_cumulative_intensity,
    exp_hawkes_intensity,
    general_hawkes_log_prob,
    ipp_sample_thinning,
    marks_log_prob,
    renewal_expected_count,
    time_rescaling_residuals,
)


def _constant_ipp(rate: float, T: float) -> InhomogeneousPoissonProcess:
    return InhomogeneousPoissonProcess.from_piecewise_constant(
        bin_edges=jnp.array([0.0, T]), rates=jnp.array([rate])
    )


class TestThinningCompensatorSign:
    """#55 — the retention correction ∫(1-p)λ must be ADDED."""

    @pytest.mark.slow
    @pytest.mark.parametrize("p", [0.3, 0.7])
    def test_thinned_hpp_matches_closed_form(self, p):
        rate, T = 2.0, 10.0
        base = _constant_ipp(rate, T)
        thin = ThinningProcess(
            base=base,
            retention_fn=lambda t, h, m=None: jnp.full_like(
                jnp.asarray(t, dtype=jnp.float32), p
            ),
        )
        times = jnp.array([1.0, 4.0, 7.0, T, T])
        mask = jnp.array([True, True, True, False, False])
        lp = thin.log_prob(times, mask)
        exact = 3 * jnp.log(p * rate) - p * rate * T
        assert float(lp) == pytest.approx(float(exact), rel=1e-4)


class TestGeneralHawkesVectorCompensator:
    """#56 — vector-t compensator must keep the excitation terms."""

    def _pair(self):
        mu, alpha, beta, T = 0.15, 0.5, 1.0, 10.0
        exp_op = ExponentialHawkes(mu=mu, alpha=alpha, beta=beta, observation_window=T)
        gen_op = GeneralHawkesProcess(
            mu=mu,
            kernel=ExponentialKernel(alpha=alpha, beta=beta),
            observation_window=T,
        )
        times = jnp.array([1.0, 2.5, 4.0, 6.0, T, T])
        mask = jnp.array([True, True, True, True, False, False])
        return exp_op, gen_op, times, mask

    @pytest.mark.slow
    def test_vector_t_cumulative_intensity_matches_exponential(self):
        exp_op, gen_op, times, mask = self._pair()
        lam_exp = exp_op.cumulative_intensity(times, times, mask)
        lam_gen = gen_op.cumulative_intensity(times, times, mask)
        assert jnp.allclose(lam_exp, lam_gen, rtol=1e-4)
        # And the excitation genuinely contributes (not just μ·t).
        assert float(lam_gen[2]) > 0.15 * float(times[2]) + 1e-3

    def test_general_hawkes_residuals_match_exponential(self):
        exp_op, gen_op, times, mask = self._pair()
        res_exp, _ = exp_op.residuals(times, mask)
        res_gen, _ = gen_op.residuals(times, mask)
        assert jnp.allclose(res_exp, res_gen, rtol=1e-4, atol=1e-6)

    @pytest.mark.slow
    def test_goodness_of_fit_matches(self):
        exp_op, gen_op, times, mask = self._pair()
        ks_exp = exp_op.goodness_of_fit(times, mask).ks_statistic
        ks_gen = gen_op.goodness_of_fit(times, mask).ks_statistic
        assert float(ks_exp) == pytest.approx(float(ks_gen), rel=1e-4)


class TestTemporalMarksPaddingGradients:
    """#57 — bounded-support mark laws must give finite gradients."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "make_dist",
        [
            lambda rate: dist.Gamma(concentration=rate, rate=1.0),
            lambda rate: dist.LogNormal(loc=rate, scale=1.0),
        ],
    )
    @pytest.mark.parametrize("history_flag", [True, False])
    def test_temporal_marks_padding_grad_finite(self, make_dist, history_flag):
        times = jnp.array([1.0, 2.0, 5.0])
        marks = jnp.array([2.5, 1.5, 0.0])  # zero padding out of support
        mask = jnp.array([True, True, False])

        def loss(rate):
            return marks_log_prob(
                times,
                marks,
                mask,
                lambda t, h: make_dist(rate),
                history_at_each_event=history_flag,
            )

        assert jnp.isfinite(loss(2.0))
        g = jax.grad(loss)(2.0)
        assert jnp.isfinite(g)


class TestTemporalHawkesSTPPBackports:
    """#58 — dt clipping and timestamp causality (ported from the STPP twin)."""

    def test_intensity_beta_grad_finite_with_padding(self):
        # Padding at T = 200 sits far after t = 1; unclipped dt = -199
        # overflowed exp(β·199) and gave a NaN β-gradient.
        times = jnp.array([0.5, 200.0, 200.0])
        mask = jnp.array([True, False, False])

        def lam(beta):
            return exp_hawkes_intensity(1.0, times, mask, 0.5, 0.8, beta)

        g = jax.grad(lam)(1.0)
        assert jnp.isfinite(g)

    def test_cumulative_intensity_beta_grad_finite_with_padding(self):
        times = jnp.array([0.5, 200.0, 200.0])
        mask = jnp.array([True, False, False])

        def lam(beta):
            return exp_hawkes_cumulative_intensity(1.0, times, mask, 0.5, 0.8, beta)

        g = jax.grad(lam)(1.0)
        assert jnp.isfinite(g)

    @pytest.mark.slow
    def test_general_log_prob_permutation_invariant(self):
        kernel = ExponentialKernel(alpha=0.5, beta=1.0)
        times = jnp.array([1.0, 2.5, 4.0, 10.0])
        mask = jnp.array([True, True, True, False])
        perm = jnp.array([2, 0, 1, 3])
        lp_sorted = general_hawkes_log_prob(
            times, mask, 10.0, 0.3, kernel.kernel, kernel.kernel_integral
        )
        lp_perm = general_hawkes_log_prob(
            times[perm], mask[perm], 10.0, 0.3, kernel.kernel, kernel.kernel_integral
        )
        assert float(lp_sorted) == pytest.approx(float(lp_perm), rel=1e-5)


class TestMaskContiguityInvariant:
    """#59 — samplers emit contiguous-prefix masks; residuals are
    mask-robust regardless."""

    @pytest.mark.slow
    def test_ipp_thinning_sampler_emits_contiguous_mask(self):
        log_fn = lambda t: jnp.log(2.0) * jnp.ones_like(jnp.asarray(t))
        times, mask, _ = ipp_sample_thinning(
            jax.random.PRNGKey(0), log_fn, 10.0, 2.5, 64
        )
        n = int(jnp.sum(mask))
        assert bool(jnp.all(mask[:n])) and bool(jnp.all(~mask[n:]))
        # Accepted times sorted; padding at T.
        assert bool(jnp.all(jnp.diff(times[:n]) >= 0))
        assert bool(jnp.all(times[n:] == 10.0))

    def test_residuals_robust_to_hole_masks(self):
        # Λ(t) = 2t. Hole at slot 1: residual at slot 2 must span from
        # the previous VALID event (0.5), not the masked slot (0.7).
        times = jnp.array([0.5, 0.7, 1.0])
        mask = jnp.array([True, False, True])
        res, _ = time_rescaling_residuals(times, mask, lambda t: 2.0 * t)
        assert float(res[0]) == pytest.approx(1.0)
        assert float(res[2]) == pytest.approx(1.0)  # 2·(1.0-0.5)

    @pytest.mark.slow
    def test_residuals_match_hand_compacted(self):
        log_fn = lambda t: jnp.log(3.0) * jnp.ones_like(jnp.asarray(t))
        times, mask, _ = ipp_sample_thinning(
            jax.random.PRNGKey(1), log_fn, 10.0, 3.5, 128
        )
        cum = lambda t: 3.0 * t
        res, _ = time_rescaling_residuals(times, mask, cum)
        # Hand-compact and recompute.
        n = int(jnp.sum(mask))
        compact_times = jnp.concatenate(
            [times[mask], jnp.full((len(times) - n,), 10.0)]
        )
        compact_mask = jnp.arange(len(times)) < n
        res_ref, _ = time_rescaling_residuals(compact_times, compact_mask, cum)
        assert jnp.allclose(res, res_ref, atol=1e-5)


class TestRenewalExpectedCount:
    """#60 — the renewal equation is solved exactly, not truncated at ~9."""

    @pytest.mark.parametrize("T", [2.0, 10.0, 20.0])
    def test_expected_count_matches_poisson(self, T):
        m = renewal_expected_count(T, dist.Exponential(1.0), n_points=200)
        assert float(m) == pytest.approx(T, rel=0.02)


class TestThinningSampleBases:
    """#61 — ThinningProcess.sample works for every documented base family."""

    @pytest.mark.slow
    def test_thinning_sample_ipp_base(self):
        base = _constant_ipp(2.0, 10.0)
        thin = ThinningProcess(
            base=base,
            retention_fn=lambda t, h, m=None: jnp.asarray(0.5),
        )
        times, mask, n = thin.sample(
            jax.random.PRNGKey(0), max_events=64, max_candidates=128
        )
        assert times.shape == (64,)
        assert int(n) == int(jnp.sum(mask))

    @pytest.mark.slow
    def test_thinning_sample_hawkes_base_still_works(self):
        base = ExponentialHawkes(mu=0.5, alpha=0.3, beta=1.0, observation_window=10.0)
        thin = ThinningProcess(
            base=base,
            retention_fn=lambda t, h, m=None: jnp.asarray(0.7),
        )
        times, mask, n = thin.sample(
            jax.random.PRNGKey(0), max_events=64, max_candidates=256
        )
        assert times.shape == (64,)
        assert int(n) == int(jnp.sum(mask))


class TestIppIntegralConsistency:
    """#62 — every integral-consuming method uses the same integrator."""

    def _op(self):
        return InhomogeneousPoissonProcess.from_piecewise_constant(
            bin_edges=jnp.array([0.0, 2.0, 5.0, 10.0]),
            rates=jnp.array([1.0, 4.0, 2.0]),
        )

    def test_predict_count_matches_closed_form(self):
        op = self._op()
        # Λ(5) - Λ(0) = 1·2 + 4·3 = 14 exactly.
        assert float(op.predict_count(0.0, 5.0)) == pytest.approx(14.0, abs=1e-5)
        assert float(op.effective_integrated_intensity(0.0, 5.0)) == pytest.approx(
            14.0, abs=1e-5
        )

    def test_survival_and_hazard_consistent(self):
        op = self._op()
        # Λ(3) = 2 + 4 = 6.
        assert float(op.cumulative_hazard(3.0)) == pytest.approx(6.0, abs=1e-5)
        assert float(op.survival(3.0)) == pytest.approx(float(jnp.exp(-6.0)), rel=1e-5)

    def test_inter_event_log_prob_consistent(self):
        op = self._op()
        # log f(τ=2 | s=1) = log λ(3) - [Λ(3) - Λ(1)] = log 4 - (6 - 1).
        got = float(op.inter_event_log_prob(2.0, 1.0))
        assert got == pytest.approx(float(jnp.log(4.0) - 5.0), rel=1e-5)

    def test_piecewise_integrate_batched_limits(self):
        op = self._op()
        integrate = op.log_intensity_fn.integrate
        vals = integrate(jnp.zeros(2), jnp.array([2.0, 5.0]))
        assert vals.shape == (2,)
        assert jnp.allclose(vals, jnp.array([2.0, 14.0]), atol=1e-5)


class TestBatchedHawkesCompensator:
    """Round-1 review: batched (B,) T against (B, n) histories must give
    one compensator per history, not a (B, B) cross-product."""

    @pytest.mark.slow
    def test_batched_histories_match_per_row(self):
        kernel = ExponentialKernel(alpha=0.5, beta=1.0)
        gen_op = GeneralHawkesProcess(mu=0.3, kernel=kernel, observation_window=10.0)
        times = jnp.array([[1.0, 2.0, 10.0], [0.5, 3.0, 4.0]])
        mask = jnp.array([[True, True, False], [True, True, True]])
        T = jnp.array([10.0, 8.0])
        batched = jax.vmap(gen_op.cumulative_intensity)(T, times, mask)
        assert batched.shape == (2,)
        for k in range(2):
            single = gen_op.cumulative_intensity(T[k], times[k], mask[k])
            assert float(batched[k]) == pytest.approx(float(single), rel=1e-6)


class TestPinnedIntensityFullWindow:
    """Round-1 review: an explicit full-window query must honour a pinned
    integrated_intensity, matching log_prob's Λ(T)."""

    @pytest.mark.slow
    def test_explicit_full_window_uses_pin(self):
        op = InhomogeneousPoissonProcess(
            log_intensity_fn=lambda t: jnp.zeros_like(jnp.asarray(t)),
            observation_window=5.0,
            integrated_intensity=7.0,  # deliberately != quadrature (5.0)
        )
        assert float(op.cumulative_hazard(op.observation_window)) == pytest.approx(7.0)
        assert float(op.survival(op.observation_window)) == pytest.approx(
            float(jnp.exp(-7.0))
        )
        # Sub-interval queries still use the live integrator.
        assert float(op.cumulative_hazard(2.0)) == pytest.approx(2.0, rel=1e-3)

    def test_pin_selection_is_batched_and_jit_safe(self):
        op = InhomogeneousPoissonProcess(
            log_intensity_fn=lambda t: jnp.zeros_like(jnp.asarray(t)),
            observation_window=5.0,
            integrated_intensity=7.0,
        )
        # Batched endpoints: full-window entries take the pin, others the
        # live integrator, with the batch shape preserved.
        t = jnp.array([5.0, 2.0, 5.0])
        out = op.cumulative_hazard(t)
        assert out.shape == (3,)
        assert jnp.allclose(out, jnp.array([7.0, 2.0, 7.0]), rtol=1e-3)
        # And the same holds under jit (filter_jit: the operator holds a
        # plain-callable leaf, per the documented PyTree contract).
        import equinox as eqx

        out_jit = eqx.filter_jit(op.cumulative_hazard)(t)
        assert jnp.allclose(out_jit, out, rtol=1e-5)
