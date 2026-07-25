"""Point-process property suite (#74): non-degenerate coverage.

The pre-existing suite validated mostly degenerate parameter points
(p = 1 thinning, α = 0 Hawkes, scalar T) where this wave's bugs were
invisible. Each test group here pins one of those gaps with closed
forms, cross-implementation equalities, or statistical validation.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from xtremax.point_processes import SampleResult
from xtremax.point_processes.distributions.temporal import (
    HomogeneousPoissonProcess as HppDist,
)
from xtremax.point_processes.operators import (
    ExponentialHawkes,
    ExponentialKernel,
    GeneralHawkesProcess,
    HomogeneousPoissonProcess,
    InhomogeneousPoissonProcess,
    MarkedTemporalPointProcess,
    RenewalProcess,
    ThinningProcess,
)
from xtremax.point_processes.primitives.hawkes import general_hawkes_log_prob


def _ipp(rate: float = 2.0, T: float = 10.0) -> InhomogeneousPoissonProcess:
    return InhomogeneousPoissonProcess.from_piecewise_constant(
        bin_edges=jnp.array([0.0, T]), rates=jnp.array([rate])
    )


def _padded(times: list[float], max_events: int, T: float):
    n = len(times)
    padded = jnp.asarray(times + [T] * (max_events - n))
    mask = jnp.arange(max_events) < n
    return padded, mask


class TestThinnedClosedForm:
    """Group 1 (pins #55): thinning an HPP(λ) with constant retention p
    IS an HPP(pλ) — the likelihoods must agree away from p = 1, where
    the previously wrong-signed compensator term vanished."""

    @pytest.mark.parametrize("p", [0.3, 0.7])
    def test_matches_thinned_hpp(self, p):
        lam, T = 2.0, 10.0
        base = HomogeneousPoissonProcess(lam, T)
        thin = ThinningProcess(
            base=base, retention_fn=lambda t, h, m=None: jnp.asarray(p)
        )
        times, mask = _padded([0.7, 2.1, 4.4, 6.0, 9.2], 16, T)

        got = float(thin.log_prob(times, mask))
        ref = float(HomogeneousPoissonProcess(p * lam, T).log_prob(jnp.sum(mask)))
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


class TestGeneralHawkesVectorDiagnostics:
    """Group 2 (pins #56): the general-kernel operator's diagnostics
    (which evaluate the compensator at vector t) must equal the
    closed-form exponential operator's."""

    def _pair(self):
        mu, alpha, beta, T = 0.4, 0.3, 1.2, 20.0
        exp_op = ExponentialHawkes(mu=mu, alpha=alpha, beta=beta, observation_window=T)
        gen_op = GeneralHawkesProcess(
            mu=mu, kernel=ExponentialKernel(alpha, beta), observation_window=T
        )
        return exp_op, gen_op

    def test_residuals_and_gof_agree(self):
        exp_op, gen_op = self._pair()
        times, mask = _padded([1.0, 2.5, 3.0, 7.7, 12.0, 15.5], 12, 20.0)

        res_e, mask_e = exp_op.residuals(times, mask)
        res_g, mask_g = gen_op.residuals(times, mask)
        np.testing.assert_array_equal(np.asarray(mask_e), np.asarray(mask_g))
        np.testing.assert_allclose(
            np.asarray(res_e), np.asarray(res_g), rtol=1e-4, atol=1e-5
        )

        gof_e = exp_op.goodness_of_fit(times, mask)
        gof_g = gen_op.goodness_of_fit(times, mask)
        np.testing.assert_allclose(
            float(gof_e.ks_statistic), float(gof_g.ks_statistic), rtol=1e-4, atol=1e-5
        )

    def test_log_prob_agrees(self):
        exp_op, gen_op = self._pair()
        times, mask = _padded([1.0, 2.5, 3.0, 7.7, 12.0, 15.5], 12, 20.0)
        np.testing.assert_allclose(
            float(exp_op.log_prob(times, mask)),
            float(gen_op.log_prob(times, mask)),
            rtol=1e-5,
            atol=1e-5,
        )


class TestMarkedGradients:
    """Group 3 (pins #57): gradients through a temporal marked process
    with bounded-support (Gamma) marks must be finite."""

    def test_filter_grad_finite(self):
        marked = MarkedTemporalPointProcess(
            ground=ExponentialHawkes(
                mu=0.5, alpha=0.2, beta=1.0, observation_window=10.0
            ),
            mark_distribution_fn=lambda t, h: dist.Gamma(2.0, 1.0),
        )
        times, mask = _padded([1.0, 3.0, 6.5], 8, 10.0)
        marks = jnp.where(mask, 1.3, 0.0)

        def loss(op):
            return op.log_prob(times, mask, marks)

        grads = eqx.filter_grad(loss)(marked)
        leaves = [
            leaf
            for leaf in jax.tree_util.tree_leaves(grads)
            if eqx.is_inexact_array(leaf)
        ]
        assert leaves, "no differentiable leaves found"
        for leaf in leaves:
            assert bool(jnp.all(jnp.isfinite(leaf))), leaf


class TestRenewalHorizon:
    """Group 4 (pins #60): an Exp(1) renewal process observed over
    T = 20 is a unit-rate Poisson process — E[N] must be ≈ 20, not
    truncated short of the horizon."""

    def test_expected_count(self):
        op = RenewalProcess(dist.Exponential(1.0), 20.0)
        keys = jax.random.split(jax.random.PRNGKey(0), 128)
        counts = jnp.stack([jnp.sum(op.sample(k, 64).mask) for k in keys]).astype(
            jnp.float32
        )
        # SE of the mean ≈ sqrt(20/128) ≈ 0.4; allow 3.5 σ.
        assert abs(float(counts.mean()) - 20.0) < 1.5


class TestPermutationInvariance:
    """Group 5 (pins #58): timestamp-based causality makes the general
    Hawkes likelihood invariant to the buffer order of the same events."""

    def test_row_shuffle(self):
        mu, alpha, beta, T = 0.4, 0.3, 1.2, 10.0

        def kernel_fn(dt):
            return alpha * jnp.exp(-beta * dt)

        def kernel_integral_fn(a, b):
            return (alpha / beta) * (jnp.exp(-beta * a) - jnp.exp(-beta * b))

        times = jnp.array([0.5, 1.2, 2.0, 3.1, 5.9, 8.4])
        mask = jnp.ones_like(times, dtype=bool)
        perm = jnp.array([3, 0, 5, 2, 4, 1])

        lp = general_hawkes_log_prob(times, mask, T, mu, kernel_fn, kernel_integral_fn)
        lp_perm = general_hawkes_log_prob(
            times[perm], mask, T, mu, kernel_fn, kernel_integral_fn
        )
        np.testing.assert_allclose(float(lp), float(lp_perm), rtol=1e-5, atol=1e-6)


class TestSamplerMaskInvariant:
    """Group 6 (pins #59): thinning-based samplers emit the documented
    contiguous-prefix mask with right-edge padding, so the diagnostics
    (which assume it) equal a hand-compacted version."""

    def test_thinning_output_compacted(self):
        base = _ipp(rate=2.0, T=10.0)
        thin = ThinningProcess(
            base=base, retention_fn=lambda t, h, m=None: jnp.asarray(0.6)
        )
        out = thin.sample(jax.random.PRNGKey(3), max_events=64, max_candidates=128)
        assert isinstance(out, SampleResult)
        times, mask = np.asarray(out.times), np.asarray(out.mask)

        n = int(mask.sum())
        # Contiguous prefix…
        np.testing.assert_array_equal(mask, np.arange(mask.size) < n)
        # …with right-edge padding and sorted valid times.
        np.testing.assert_array_equal(times[n:], 10.0)
        assert np.all(np.diff(times[:n]) >= 0)

        # Diagnostics equal a hand-compacted rebuild of the same events.
        rebuilt = np.full_like(times, 10.0)
        rebuilt[:n] = np.sort(times[:n])
        gof = base.goodness_of_fit(out.times, out.mask)
        gof_rebuilt = base.goodness_of_fit(
            jnp.asarray(rebuilt), jnp.asarray(np.arange(mask.size) < n)
        )
        np.testing.assert_allclose(
            float(gof.ks_statistic), float(gof_rebuilt.ks_statistic), rtol=1e-6
        )


class TestHawkesSampleStatistics:
    """Group 7: the sampler's output must pass its own compensator's
    time-rescaling test (KS against Exp(1))."""

    def test_time_rescaling_ks(self):
        op = ExponentialHawkes(mu=0.7, alpha=0.3, beta=1.5, observation_window=400.0)
        times, mask, _ = op.sample(jax.random.PRNGKey(0), 1024, max_candidates=4096)
        n = int(jnp.sum(mask))
        assert n > 150  # enough events for the KS to have power
        gof = op.goodness_of_fit(times, mask)
        # 95% KS band for n≈280 is ~0.081; a wrong compensator or a
        # biased sampler blows straight past it.
        assert float(gof.ks_statistic) < 0.09


class TestDistributionWrappers:
    """Group 8: NumPyro-facing wrappers — batched parameter shapes and
    the documented (times, n_events) tuple path."""

    def test_batched_rate_log_prob_shape(self):
        d = HppDist(rate=jnp.array([1.0, 2.0]), observation_window=10.0)
        assert d.batch_shape == (2,)
        times, mask = _padded([1.0, 4.0, 7.0], 16, 10.0)
        lp = d.log_prob((times, mask))
        assert lp.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(lp)))

    def test_times_n_events_tuple_path(self):
        d = HppDist(rate=1.5, observation_window=10.0)
        times, mask = _padded([1.0, 4.0, 7.0], 16, 10.0)
        lp_mask = d.log_prob((times, mask))
        lp_count = d.log_prob((times, jnp.asarray(3)))
        np.testing.assert_allclose(float(lp_mask), float(lp_count), rtol=1e-6)

    def test_sample_log_prob_roundtrip(self):
        d = HppDist(rate=1.5, observation_window=10.0)
        value = d.sample(jax.random.PRNGKey(0))
        lp = d.log_prob(value)
        assert bool(jnp.isfinite(lp))


class TestIPPClosedForms:
    """Group 9 (pins #62): every integral-consuming IPP method against
    the homogeneous closed forms (a constant-rate IPP is an HPP)."""

    def test_against_constant_rate(self):
        lam, T = 2.0, 10.0
        op = _ipp(rate=lam, T=T)
        t = jnp.array([1.0, 3.5, 7.0])

        np.testing.assert_allclose(
            np.asarray(op.survival(t)), np.exp(-lam * np.asarray(t)), rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(op.cumulative_hazard(t)), lam * np.asarray(t), rtol=1e-5
        )
        tau = jnp.array([0.5, 2.0])
        np.testing.assert_allclose(
            np.asarray(op.inter_event_log_prob(tau)),
            np.log(lam) - lam * np.asarray(tau),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(op.predict_count(2.0, 5.0)), lam * 3.0, rtol=1e-5
        )


class TestSampleShapeThroughWrappers:
    """Group 10 (informs #65): `sample_shape` batching exists only on
    the temporal HPP operator; wrapper operators must fail loudly, not
    return silently mis-shaped buffers."""

    def test_marked_wrapper_raises(self):
        marked = MarkedTemporalPointProcess(
            ground=_ipp(), mark_distribution_fn=lambda t, h: dist.Gamma(2.0, 1.0)
        )
        with pytest.raises(TypeError):
            marked.sample(jax.random.PRNGKey(0), 16, sample_shape=(3,))

    def test_thinning_wrapper_raises(self):
        thin = ThinningProcess(
            base=_ipp(), retention_fn=lambda t, h, m=None: jnp.asarray(0.5)
        )
        with pytest.raises(TypeError):
            thin.sample(jax.random.PRNGKey(0), 16, sample_shape=(3,))
