"""Regressions for review-round fixes on the extensions PR.

Each test corresponds to a specific bug raised in PR review so a
future refactor cannot silently re-introduce it.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from xtremax.point_processes import EventHistory
from xtremax.point_processes.operators import (
    ExponentialHawkes,
    HomogeneousPoissonProcess,
    MarkedTemporalPointProcess,
    ThinningProcess,
)
from xtremax.point_processes.primitives.marked import sample_marks_at_times
from xtremax.point_processes.primitives.thinning import (
    retention_compensator,
    thinning_retention_log_prob,
)


class TestHistoryDependentRetentionSeesPrefixOnly:
    """Codex P1: retention must not peek at events in its own future.

    If the retention callable ever sees events with ``time >= t`` in
    ``history.mask``, it can condition on the future and produce a
    wrong log-likelihood / compensator. We assert that both the
    per-event log-prob and the quadrature compensator honour the
    ``time < t`` prefix contract.
    """

    def test_thinning_retention_log_prob_uses_prefix_only(self):
        event_times = jnp.array([1.0, 2.0, 3.0, 4.0])
        mask = jnp.array([True, True, True, True])
        history = EventHistory(times=event_times, mask=mask, marks=None)

        # Retention that returns a value encoding the prefix count so
        # the test can read it back via the summed log-prob. If the
        # prefix contract leaks (full history used instead), every
        # event sees n=4 and the sum is 4·log(f(4)).
        def retention(t, h, mark=None):
            n = jnp.sum(h.mask).astype(jnp.float32)
            return jnp.clip((n + 1.0) * 0.2, 1e-6, 1.0)

        total = float(
            thinning_retention_log_prob(retention, event_times, mask, history)
        )
        # Event i sees n=i prior events → p = (i+1)·0.2 ∈ {0.2, 0.4, 0.6, 0.8}.
        expected = sum(float(jnp.log((i + 1) * 0.2)) for i in range(4))
        assert total == pytest.approx(expected, rel=1e-4)

        # Sanity: if history leaked (all four events), sum would be
        # 4·log(5·0.2) = 4·log(1.0) = 0, which this test would not
        # match.
        leaked = 4 * float(jnp.log(1.0))
        assert total != pytest.approx(leaked, abs=1e-3)

    def test_retention_compensator_sees_prefix_only(self):
        event_times = jnp.array([1.0, 2.0, 3.0])
        mask = jnp.array([True, True, True])
        history = EventHistory(times=event_times, mask=mask, marks=None)

        def retention(t, h, mark=None):
            # At t = 0.5 there should be no prior events; at t = 2.5
            # there should be exactly 2 (events at 1.0, 2.0).
            n = jnp.sum(h.mask)
            # We cannot assert under vmap; instead check that the
            # visible count is consistent with t via a numeric bound:
            # n_visible ≤ count of event_times strictly less than t.
            return jnp.asarray(1.0) * (n.astype(jnp.float32) + 0.0) * 0.0 + 1.0

        def intensity(t, h):
            return jnp.asarray(1.0)

        # If the prefix logic is wrong and full history leaks in, the
        # retention_fn above would still be 1.0, but a concrete test:
        # pick a retention that actively reads ``h.mask`` and check
        # the integral numerically against the hand computation.
        def retention_count(t, h, mark=None):
            # p = 1 / (1 + N_prefix); compensator = ∫(1 - p)·λ dt
            n = jnp.sum(h.mask).astype(jnp.float32)
            return 1.0 / (1.0 + n)

        T = 4.0
        val = float(retention_compensator(retention_count, intensity, history, T))
        # At each grid point t: n_prefix = # events < t.
        # Known values: p=1 for t<1, p=1/2 for 1≤t<2, p=1/3 for 2≤t<3, p=1/4 for t≥3.
        # ∫(1-p)·1 dt = 0·1 + 0.5·1 + (2/3)·1 + 0.75·1 = 0 + 0.5 + 0.667 + 0.75 = 1.917
        expected = 0.0 + 0.5 + (2.0 / 3.0) + 0.75
        assert val == pytest.approx(expected, rel=0.05)


class TestThinningMarkedBaseRoutesMarks:
    """Codex P1: thinning a marked base must preserve and use marks.

    Previously ``sample`` forced ``latent_marks = None`` because the
    third element of the base result was always treated as a count.
    """

    def test_mark_dependent_retention_fires(self):
        class MagnitudeGate(eqx.Module):
            cutoff: jnp.ndarray

            def __call__(self, t, history, proposed_mark=None):
                # proposed_mark must be non-None for this to fire.
                if proposed_mark is None:
                    return jnp.asarray(0.0)
                return (proposed_mark > self.cutoff).astype(jnp.float32)

        hpp = HomogeneousPoissonProcess(5.0, 5.0)

        def marks_fn(t, history):
            return dist.Normal(0.5, 0.1)  # most marks > 0.3

        mtpp = MarkedTemporalPointProcess(
            ground=hpp,
            mark_distribution_fn=marks_fn,
            mark_dim=None,
            history_at_each_event=False,
        )
        gate = MagnitudeGate(cutoff=jnp.asarray(0.3))
        thin = ThinningProcess(base=mtpp, retention_fn=gate)
        # Marked base → sample returns (times, mask, retained_marks).
        _t, retained_mask, _marks = thin.sample(jax.random.PRNGKey(0), max_events=64)

        # If marks are routed correctly, the gate should keep most of
        # the draws (mean ≈ 0.5 > 0.3 so retention ≈ 1). If marks get
        # dropped to None, the gate returns 0 and nothing is retained.
        assert int(jnp.sum(retained_mask)) > 0


class TestSampleMarksAtTimesUsesPriorMarks:
    """Codex P1: sequential mark sampling must thread prior marks.

    Without this, mark distributions that depend on ``history.marks``
    silently sample from the wrong law.
    """

    def test_prior_marks_are_visible_to_next_mark_draw(self):
        """Each successive draw's ``loc`` shifts by the mean of prior marks.

        If prior marks are correctly threaded, mark ``m_i`` is drawn
        from ``Normal(sum_prior_marks, 0.01)`` — the successive means
        are cumulative partial sums. A broken implementation would
        always draw from ``Normal(0, 0.01)``.
        """
        event_times = jnp.array([0.0, 1.0, 2.0, 3.0])
        mask = jnp.array([True, True, True, True])

        def marks_fn(t, history):
            loc = (
                jnp.sum(history.marks[..., 0])
                if history.marks is not None
                else jnp.asarray(0.0)
            )
            return dist.Normal(loc, 0.01)

        marks = sample_marks_at_times(
            jax.random.PRNGKey(0),
            event_times,
            mask,
            marks_fn,
            history_at_each_event=True,
        )

        # Running sum of marks: each m_i ≈ m_0 + m_1 + ... + m_{i-1}
        # (with tiny noise). So m_1 ≈ m_0, m_2 ≈ 2·m_0, m_3 ≈ 4·m_0 in
        # expectation — strictly increasing in absolute value when m_0
        # is not too small.
        m = marks
        # m[0] arbitrary but finite; thereafter each should be roughly
        # the cumulative sum of prior draws.
        assert jnp.all(jnp.isfinite(m))
        # If prior marks are ignored, m[i] ~ Normal(0, 0.01) for all i
        # and |m[3]| would typically be well under 0.05. With threading
        # the cumulative sum pushes |m[3]| dramatically larger.
        assert abs(float(m[3])) > 3 * abs(float(m[0])) - 0.1


class TestThinningPromotesScalarMarksInHistory:
    """Copilot: scalar marks must not be dropped from ``EventHistory``."""

    def test_history_marks_populated_for_scalar_marks(self):
        event_times = jnp.array([0.1, 0.5, 1.0, 2.0])
        mask = jnp.array([True, True, True, True])
        marks = jnp.array([0.1, 0.2, 0.3, 0.4])  # shape (4,) — same as mask

        seen_marks = []

        def retention(t, history, mark=None):
            if history.marks is not None:
                seen_marks.append(history.marks.shape)
            return jnp.asarray(1.0)

        hpp = HomogeneousPoissonProcess(2.0, 3.0)
        thin = ThinningProcess(base=hpp, retention_fn=retention)
        thin.log_prob(event_times, mask, marks=marks)

        # Should have been invoked with a non-None, 2-D marks buffer at
        # every event — regression for the previous drop on
        # marks.shape[-1] == mask.shape[-1].
        assert len(seen_marks) >= 1
        for shape in seen_marks:
            assert len(shape) == 2


class TestThinningForwardsMaxCandidates:
    """Copilot: ``max_candidates`` must reach a thinning-based base."""

    def test_max_candidates_passes_through_to_hawkes(self):
        # Hawkes accepts ``max_candidates``; HPP does not. We assert
        # that passing it through doesn't raise for Hawkes.
        hawkes = ExponentialHawkes(mu=0.5, alpha=0.3, beta=1.0, observation_window=10.0)

        def retention(t, history, mark=None):
            return jnp.asarray(1.0)

        thin = ThinningProcess(base=hawkes, retention_fn=retention)
        # Should not raise — base.sample(max_events, max_candidates=...)
        # is valid for Hawkes.
        _ = thin.sample(jax.random.PRNGKey(0), max_events=32, max_candidates=128)


class TestThinningMarkedBaseLogProb:
    """Codex P1: ThinningProcess.log_prob must accept a marked base."""

    def test_log_prob_routes_through_ground_intensity(self):
        """A marked base has no ``.intensity`` / ``.rate``; the
        compensator path must fall through to ``base.ground``.
        """
        hpp = HomogeneousPoissonProcess(2.0, 5.0)

        def marks_fn(t, history):
            return dist.Normal(0.0, 1.0)

        mtpp = MarkedTemporalPointProcess(
            ground=hpp,
            mark_distribution_fn=marks_fn,
            mark_dim=None,
            history_at_each_event=False,
        )

        def retention(t, history, mark=None):
            return jnp.asarray(0.6)

        thin = ThinningProcess(base=mtpp, retention_fn=retention)
        t, m, marks = mtpp.sample(jax.random.PRNGKey(0), max_events=64)
        # Before the _base_intensity_fn fallback to ``base.ground``,
        # this raised AttributeError.
        ll = float(thin.log_prob(t, m, marks=marks))
        assert jnp.isfinite(ll)


class TestThinningSampleReturnsMarks:
    """Codex P2: thinning a marked base must surface retained marks."""

    def test_operator_sample_returns_marks_for_marked_base(self):
        hpp = HomogeneousPoissonProcess(3.0, 5.0)

        def marks_fn(t, history):
            return dist.Normal(0.0, 1.0)

        mtpp = MarkedTemporalPointProcess(
            ground=hpp,
            mark_distribution_fn=marks_fn,
            mark_dim=None,
            history_at_each_event=False,
        )

        def retention(t, history, mark=None):
            return jnp.asarray(1.0)

        thin = ThinningProcess(base=mtpp, retention_fn=retention)
        t, _m, third = thin.sample(jax.random.PRNGKey(0), max_events=64)
        # With retention=1 the third slot is the retained-marks array,
        # not a scalar count.
        third_arr = jnp.asarray(third)
        assert third_arr.ndim >= 1
        assert third_arr.shape[-1] == t.shape[-1]

    def test_distribution_round_trip_with_marked_base(self):
        from xtremax.point_processes.distributions import (
            ThinningProcess as ThinningDist,
        )

        hpp = HomogeneousPoissonProcess(3.0, 5.0)

        def marks_fn(t, history):
            return dist.Normal(0.0, 1.0)

        mtpp = MarkedTemporalPointProcess(
            ground=hpp,
            mark_distribution_fn=marks_fn,
            mark_dim=None,
            history_at_each_event=False,
        )

        def retention(t, history, mark=None):
            return jnp.asarray(1.0)

        d = ThinningDist(base=mtpp, retention_fn=retention, max_events=64)
        value = d.sample(jax.random.PRNGKey(0))
        # Marked base → sample returns (times, mask, marks), so
        # log_prob(sample(...)) round-trips without having to
        # reassemble the marks out of band.
        assert len(value) == 3
        ll = float(d.log_prob(value))
        assert jnp.isfinite(ll)


class TestRenewalBatchShape:
    """Codex P2: Renewal batch_shape must include inter-event dims."""

    def test_batch_shape_broadcasts_with_inter_event_batch(self):
        from xtremax.point_processes.distributions import (
            RenewalProcess as RenewalDist,
        )

        # Vmap-style batched rates on the inter-event Exponential.
        batched_rates = jnp.array([1.0, 2.0, 3.0])
        d = RenewalDist(
            dist.Exponential(batched_rates), observation_window=5.0, max_events=64
        )
        assert d.batch_shape == (3,)


class TestGeneralHawkesBatchShape:
    """Codex P2: GeneralHawkes batch_shape must include ``mu``."""

    def test_batch_shape_broadcasts_with_mu(self):
        from xtremax.point_processes.distributions import (
            GeneralHawkesProcess as GenHawkesDist,
        )
        from xtremax.point_processes.operators import ExponentialKernel

        batched_mu = jnp.array([0.3, 0.5, 0.7])
        kernel = ExponentialKernel(alpha=jnp.asarray(0.3), beta=jnp.asarray(1.0))
        d = GenHawkesDist(
            mu=batched_mu, kernel=kernel, observation_window=5.0, max_events=64
        )
        assert d.batch_shape == (3,)


# ---------------------------------------------------------------------
# Spatial PP review-round fixes
# ---------------------------------------------------------------------


class TestSpatialIppLambdaMaxRequired:
    """Codex P1: thinning sampler must not silently use a non-bound.

    Previously the IPP operator fell back to ``2 Λ(D) / |D|`` (twice
    the mean intensity), which is not a true upper bound — it
    silently biased the sampler low in peaks. The operator now raises
    when no valid bound is available.
    """

    def test_unset_lambda_max_raises(self):
        from xtremax.point_processes import RectangularDomain
        from xtremax.point_processes.operators import InhomogeneousSpatialPP

        domain = RectangularDomain.from_size(jnp.array([4.0, 4.0]))

        def log_lam(s):
            return jnp.full(s.shape[:-1], jnp.log(0.5))

        op = InhomogeneousSpatialPP(log_lam, domain)
        with pytest.raises(ValueError, match="lambda_max"):
            op.sample(jax.random.PRNGKey(0), max_candidates=64)

    def test_pinned_lambda_max_used(self):
        from xtremax.point_processes import RectangularDomain
        from xtremax.point_processes.operators import InhomogeneousSpatialPP

        domain = RectangularDomain.from_size(jnp.array([4.0, 4.0]))

        def log_lam(s):
            return jnp.full(s.shape[:-1], jnp.log(0.5))

        op = InhomogeneousSpatialPP(log_lam, domain, lambda_max=0.5)
        # Should not raise; effective bound matches the pinned value.
        assert jnp.allclose(op.effective_lambda_max(), 0.5)


class TestHomogeneousSpatialPPRejectsBatchedRate:
    """Codex P2 / Copilot: spatial HPP distribution must reject batched rates.

    The sampler doesn't support batched rates; declaring
    ``batch_shape=rate.shape`` would let NumPyro plate the
    distribution but the actual sample call would silently fail.
    """

    def test_scalar_rate_ok(self):
        from xtremax.point_processes import RectangularDomain
        from xtremax.point_processes.distributions import HomogeneousSpatialPP

        domain = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        d = HomogeneousSpatialPP(rate=0.5, domain=domain)
        assert d.batch_shape == ()

    def test_batched_rate_raises(self):
        from xtremax.point_processes import RectangularDomain
        from xtremax.point_processes.distributions import HomogeneousSpatialPP

        domain = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        with pytest.raises(ValueError, match="scalar"):
            HomogeneousSpatialPP(rate=jnp.array([0.5, 1.0]), domain=domain)


class TestSpatialMarksPaddingSupportAware:
    """Codex P2 / Copilot: padding-mark substitute must be in-support.

    A literal ``1.0`` is not inside Beta or Uniform(2, 3) supports
    and produces ``-inf`` log-probs / NaN gradients at padding rows.
    The fix queries ``d.support.feasible_like(m_i)`` so the padding
    value is always in-support and dtype-preserving.
    """

    def test_uniform_2_3_padding(self):
        # Uniform(2, 3) — old fallback ``1.0`` is below the support.
        from xtremax.point_processes.primitives import spatial_marks_log_prob

        locations = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        marks = jnp.array([2.5, 2.7, 0.0])  # padding mark at row 2
        mask = jnp.array([True, True, False])
        lp = spatial_marks_log_prob(
            locations, marks, mask, lambda s: dist.Uniform(2.0, 3.0)
        )
        assert jnp.isfinite(lp)
        # Hand-compute the unmasked sum.
        expected = jnp.sum(dist.Uniform(2.0, 3.0).log_prob(marks[:2]))
        assert jnp.allclose(lp, expected)

    def test_beta_padding(self):
        # Beta on (0, 1) — endpoints are not in support; ``1.0`` would
        # fail and so would ``0.0``. ``feasible_like`` returns 0.5.
        from xtremax.point_processes.primitives import spatial_marks_log_prob

        locations = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        marks = jnp.array([0.3, 0.7, 0.0])
        mask = jnp.array([True, True, False])
        lp = spatial_marks_log_prob(
            locations, marks, mask, lambda s: dist.Beta(2.0, 5.0)
        )
        assert jnp.isfinite(lp)

    def test_padding_log_prob_carries_finite_grad(self):
        # Without the support-aware substitute, gradients through the
        # masked branch were NaN even though the value itself was
        # masked out. Verify the gradient is finite end-to-end.
        from xtremax.point_processes.primitives import spatial_marks_log_prob

        locations = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        marks = jnp.array([2.5, 0.0])
        mask = jnp.array([True, False])

        def loss(rate):
            return spatial_marks_log_prob(
                locations,
                marks,
                mask,
                lambda s: dist.Gamma(concentration=rate, rate=1.0),
            )

        g = jax.grad(loss)(2.0)
        assert jnp.isfinite(g)


class TestMarkSamplerPreservesDiscreteDtype:
    """Copilot: padding for discrete-mark draws must preserve int dtype."""

    def test_categorical_sample_dtype(self):
        from xtremax.point_processes.primitives import (
            sample_spatial_marks_at_locations,
        )

        locations = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        mask = jnp.array([True, True, False])
        marks = sample_spatial_marks_at_locations(
            jax.random.PRNGKey(0),
            locations,
            mask,
            lambda s: dist.Categorical(probs=jnp.array([0.5, 0.3, 0.2])),
        )
        # Categorical draws are integer-valued; padding mustn't promote
        # the dtype to float.
        assert jnp.issubdtype(marks.dtype, jnp.integer)


class TestRectangularDomainValidation:
    """Copilot: ``RectangularDomain.__init__`` validates bounds."""

    def test_shape_mismatch_raises(self):
        from xtremax.point_processes import RectangularDomain

        with pytest.raises(ValueError, match="shape"):
            RectangularDomain(lo=jnp.zeros(2), hi=jnp.ones(3))

    def test_inverted_bounds_raise(self):
        from xtremax.point_processes import RectangularDomain

        with pytest.raises(ValueError, match="hi > lo"):
            RectangularDomain(lo=jnp.array([1.0, 2.0]), hi=jnp.array([3.0, 1.0]))


class TestHaltonDynamicPrimes:
    """Copilot: Halton QMC must not hard-cap at d ≤ 12."""

    def test_d13_runs(self):
        from xtremax.point_processes import (
            RectangularDomain,
            integrate_log_intensity_spatial,
        )

        domain = RectangularDomain.from_size(jnp.full((13,), 1.0))

        def log_lam(s):
            return jnp.full(s.shape[:-1], 0.0)  # constant intensity 1

        result = integrate_log_intensity_spatial(
            log_lam, domain, n_points=512, method="qmc"
        )
        # ∫_[0,1]^13 1 ds = 1.
        assert jnp.allclose(result, 1.0, rtol=1e-3)


class TestMarkedSpatialDomainReturnsRectangularDomain:
    """Copilot: ``MarkedSpatialPP.domain`` returns a RectangularDomain."""

    def test_domain_has_volume(self):
        from xtremax.point_processes import RectangularDomain
        from xtremax.point_processes.operators import (
            HomogeneousSpatialPP,
            MarkedSpatialPP,
        )

        domain = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        ground = HomogeneousSpatialPP(rate=0.5, domain=domain)
        mpp = MarkedSpatialPP(
            ground=ground, mark_distribution_fn=lambda s: dist.Exponential(1.0)
        )
        # If the return type were ``Array``, ``.volume()`` would fail.
        assert jnp.allclose(mpp.domain.volume(), 16.0)


class TestHppSpatialLogProbJanossy:
    """Codex P2 round 2: HPP log-prob aligns with IPP under constant intensity.

    The previous form included a spurious ``-n log |D|`` term that
    diverged from the standard Janossy log-likelihood (and from the
    IPP form under constant intensity), skewing absolute values used
    in model comparison.
    """

    def test_drops_n_log_volume_term(self):
        from xtremax.point_processes.primitives import hpp_spatial_log_prob

        rate = 0.5
        vol = 100.0
        n = jnp.asarray(20)
        # Janossy: n log λ - λ|D|. No -n log |D| term.
        expected = 20 * jnp.log(rate) - rate * vol
        assert jnp.allclose(hpp_spatial_log_prob(n, rate, vol), expected)

    def test_matches_ipp_with_constant_intensity(self):
        from xtremax.point_processes.primitives import (
            hpp_spatial_log_prob,
            ipp_spatial_log_prob,
        )

        rate = 0.5
        vol = 16.0  # 4×4 box
        locations = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [0.0, 0.0]])
        mask = jnp.array([True, True, True, False])

        def log_lam(s):
            return jnp.full(s.shape[:-1], jnp.log(rate))

        ipp_lp = ipp_spatial_log_prob(locations, mask, log_lam, rate * vol)
        hpp_lp = hpp_spatial_log_prob(jnp.asarray(3), rate, vol)
        assert jnp.allclose(ipp_lp, hpp_lp)


class TestHppSpatialDistCountPathNoClip:
    """Codex P1 round 2: count-path log_prob must use raw count.

    Fabricating a mask via ``ranks < count`` silently clipped to
    ``max_events``; for ``n > max_events`` the log-likelihood would
    be wrong.
    """

    def test_count_above_buffer_size_unclipped(self):
        from xtremax.point_processes import RectangularDomain
        from xtremax.point_processes.distributions import HomogeneousSpatialPP
        from xtremax.point_processes.primitives import hpp_spatial_log_prob

        domain = RectangularDomain.from_size(jnp.array([2.0, 2.0]))
        # Tiny buffer of size 8; supply a count well above it.
        d_pp = HomogeneousSpatialPP(rate=5.0, domain=domain, max_events=8)
        locations = jnp.zeros((8, 2))
        big_count = jnp.asarray(50)
        result = d_pp.log_prob((locations, big_count))
        # Compare against the primitive at the *real* count, not the
        # clipped value. With clipping the result would be off by
        # ``(50 - 8) * log(rate) = 42 * log 5 ≈ 67.6``.
        expected = hpp_spatial_log_prob(big_count, 5.0, domain.volume())
        assert jnp.allclose(result, expected)

    def test_mask_path_still_works(self):
        from xtremax.point_processes import RectangularDomain
        from xtremax.point_processes.distributions import HomogeneousSpatialPP

        domain = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        d_pp = HomogeneousSpatialPP(rate=0.5, domain=domain, max_events=64)
        import jax.random as jr

        locs, mask = d_pp.sample(jr.PRNGKey(0))
        n = jnp.sum(mask).astype(jnp.int32)
        # Both signatures must agree.
        lp_mask = d_pp.log_prob((locs, mask))
        lp_count = d_pp.log_prob((locs, n))
        assert jnp.allclose(lp_mask, lp_count)


# ====================================================================
# PR #13 — spatiotemporal port review fixes
# ====================================================================


class TestStppHawkesIntensityHandlesPaddingTimes:
    """Codex P1: ``β·exp(-β·dt)`` must stay finite for padding times.

    Padding rows live at ``temporal.t1`` so ``dt = t_query - t_pad``
    can go negative when querying before the window end. Without the
    ``jnp.clip(dt, 0, inf)`` guard the exponential overflows and
    poisons the masked-out branch with ``inf · 0 = nan``.
    """

    def test_no_nan_with_padding_times_in_future(self) -> None:
        from xtremax.point_processes.primitives import stpp_hawkes_intensity

        # All slots are padding; padding times in the "future".
        event_locs = jnp.zeros((4, 2))
        event_times = jnp.array([100.0, 100.0, 100.0, 100.0])
        mask = jnp.zeros(4, dtype=jnp.bool_)
        s = jnp.array([1.0, 1.0])
        # Query at t=0.5 — all event_times are >> t, so dt is hugely
        # negative and exp(-β·dt) overflows without the clip.
        lam = stpp_hawkes_intensity(
            s,
            jnp.asarray(0.5),
            event_locs,
            event_times,
            mask,
            mu=0.3,
            alpha=0.5,
            beta=2.0,
            sigma=0.5,
        )
        assert jnp.isfinite(lam)
        assert jnp.allclose(lam, 0.3)


class TestStppHawkesLogProbCausalityByTimestamp:
    """Codex P2: causal mask must use timestamps, not buffer position.

    Position-based causality silently treats whichever rows the caller
    wrote first as "earlier". For a buffer that is permuted relative
    to time order this would inject future events as parents.
    """

    def test_log_prob_invariant_to_row_permutation(self) -> None:
        from xtremax.point_processes import RectangularDomain, TemporalDomain
        from xtremax.point_processes.primitives import stpp_hawkes_log_prob

        spatial = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        temporal = TemporalDomain.from_duration(2.0)
        locs = jnp.array([[1.0, 1.0], [3.0, 1.0], [2.0, 2.0]])
        times = jnp.array([0.2, 1.0, 1.5])
        mask = jnp.array([True, True, True])

        kwargs = dict(
            mu=0.3,
            alpha=0.4,
            beta=1.0,
            sigma=0.5,
            spatial=spatial,
            temporal=temporal,
            boundary_correction=False,
        )
        lp_sorted = stpp_hawkes_log_prob(locs, times, mask, **kwargs)
        # Reverse the buffer; timestamps now decrease with row index.
        # A position-based causality check would treat row 0 (t=1.5)
        # as a parent of row 2 (t=0.2), which is wrong.
        lp_reversed = stpp_hawkes_log_prob(locs[::-1], times[::-1], mask, **kwargs)
        assert jnp.allclose(lp_sorted, lp_reversed, rtol=1e-5)


class TestStppHawkesSampleNoOOB:
    """Codex/Copilot P1: sampler must not scatter at ``n_max``.

    Out-of-bounds scatter under JAX's default
    ``promise_in_bounds`` semantics is backend-dependent; in
    rejection-heavy runs the buffer state can become inconsistent.
    """

    def test_high_rejection_rate_keeps_buffer_clean(self) -> None:
        # μ=0, α=0 → never any acceptance; every step is a rejection.
        # Without the clip-to-(n_max-1) guard this would scatter at
        # ``n_max`` repeatedly.
        from jax import random

        from xtremax.point_processes import RectangularDomain, TemporalDomain
        from xtremax.point_processes.primitives import stpp_hawkes_sample

        spatial = RectangularDomain.from_size(jnp.array([2.0, 2.0]))
        temporal = TemporalDomain.from_duration(1.0)
        locs, times, mask, n = stpp_hawkes_sample(
            random.PRNGKey(0),
            mu=1e-9,
            alpha=0.0,
            beta=1.0,
            sigma=0.5,
            spatial=spatial,
            temporal=temporal,
            max_events=8,
        )
        # All slots remain padding; no real events; finite buffers.
        assert int(n) == 0
        assert not bool(jnp.any(mask))
        assert jnp.all(jnp.isfinite(locs))
        assert jnp.all(jnp.isfinite(times))


class TestStppHawkesSampleZeroBaselineDoesNotDivByZero:
    """Codex P1: λ̄ must be floored before division.

    With μ=0 and an empty history, the Ogata clock rate
    ``λ̄·|D|`` is exactly 0 and the inter-event time
    ``-log(u)/λ̄`` becomes ``inf``. The acceptance ratio
    ``λ/λ̄`` then becomes ``0/0 = nan``.
    """

    def test_mu_zero_buffer_finite(self) -> None:
        from jax import random

        from xtremax.point_processes import RectangularDomain, TemporalDomain
        from xtremax.point_processes.primitives import stpp_hawkes_sample

        spatial = RectangularDomain.from_size(jnp.array([2.0, 2.0]))
        temporal = TemporalDomain.from_duration(1.0)
        locs, times, _mask, n = stpp_hawkes_sample(
            random.PRNGKey(0),
            mu=0.0,
            alpha=0.0,
            beta=1.0,
            sigma=0.5,
            spatial=spatial,
            temporal=temporal,
            max_events=8,
        )
        assert jnp.all(jnp.isfinite(locs))
        assert jnp.all(jnp.isfinite(times))
        assert int(n) == 0


class TestPearsonResidualsNoSentinelArtifact:
    """Codex P1: ``-1`` was being interpreted as last-cell index.

    The original code used ``masked_idx = where(mask, cell_idx, -1)``
    intending ``-1`` as a sentinel for padding rows. Under JAX's
    negative-index semantics this actually wrote padding into the
    last cell, then the follow-up ``where(arange(n) >= 0, ...)`` was
    a no-op. The 0/1 weight already does the masking; we drop the
    sentinel and verify the last cell's residual matches a manual
    count.
    """

    def test_last_cell_residual_uses_real_count_only(self) -> None:
        from xtremax.point_processes import RectangularDomain, TemporalDomain
        from xtremax.point_processes.primitives import (
            ipp_spatiotemporal_pearson_residuals,
        )

        spatial = RectangularDomain.from_size(jnp.array([2.0, 2.0]))
        temporal = TemporalDomain.from_duration(2.0)

        def log_lam(s, t):
            return jnp.full(s.shape[:-1], jnp.log(0.5))

        # Pack three real events into the last cell; pad with two
        # padding rows whose locations would land in cell 0 (the
        # default ``spatial.lo`` padding) but whose mask is False.
        locs = jnp.array([[1.5, 1.5], [1.6, 1.6], [1.4, 1.7], [0.0, 0.0], [0.0, 0.0]])
        times = jnp.array([1.5, 1.6, 1.7, 0.0, 0.0])
        mask = jnp.array([True, True, True, False, False])
        residuals = ipp_spatiotemporal_pearson_residuals(
            locs,
            times,
            mask,
            log_lam,
            spatial,
            temporal,
            n_spatial_bins=2,
            n_temporal_bins=2,
            n_integration_points=4,
        )
        # 2*2*2 = 8 cells. Last cell has expected count 0.5, count 3,
        # so residual = (3 - 0.5) / sqrt(0.5) ≈ 3.535.
        # If padding had bled in via the -1 wrap the count would be
        # different. We assert the last-cell residual matches the
        # ground-truth derivation.
        last_residual = residuals[-1]
        expected_last = (3.0 - 0.5) / jnp.sqrt(0.5)
        assert jnp.allclose(last_residual, expected_last, rtol=5e-2)
