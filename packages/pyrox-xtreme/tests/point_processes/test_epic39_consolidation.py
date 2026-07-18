"""Tests for the operator-consolidation epic (#39).

* #64 — mixin-hosted diagnostics behave identically across families and
  the renewal family (which previously lacked a GOF test) is covered.
* #65 — samplers return typed results; wrapper operators dispatch on
  the result type instead of shape/dtype heuristics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from xtremax.point_processes import (
    MarkedSampleResult,
    SampleResult,
    SpatialSampleResult,
    SpatiotemporalSampleResult,
)
from xtremax.point_processes._domain import RectangularDomain, TemporalDomain
from xtremax.point_processes.operators import (
    ExponentialHawkes,
    HomogeneousPoissonProcess,
    InhomogeneousPoissonProcess,
    MarkedTemporalPointProcess,
    RenewalProcess,
    ThinningProcess,
)
from xtremax.point_processes.operators.hpp_spatial import HomogeneousSpatialPP
from xtremax.point_processes.operators.hpp_spatiotemporal import (
    HomogeneousSpatioTemporalPP,
)


def _ipp(rate: float = 2.0, T: float = 10.0) -> InhomogeneousPoissonProcess:
    return InhomogeneousPoissonProcess.from_piecewise_constant(
        bin_edges=jnp.array([0.0, T]), rates=jnp.array([rate])
    )


class TestTypedSampleResults:
    """#65 — every sampler returns its typed NamedTuple, and positional
    unpacking is unchanged."""

    def test_temporal_samplers_return_sample_result(self):
        key = jax.random.PRNGKey(0)
        assert isinstance(
            HomogeneousPoissonProcess(2.0, 10.0).sample(key, 64), SampleResult
        )
        assert isinstance(_ipp().sample(key, 64), SampleResult)
        assert isinstance(
            ExponentialHawkes(
                mu=0.5, alpha=0.3, beta=1.0, observation_window=10.0
            ).sample(key, 64),
            SampleResult,
        )
        assert isinstance(
            RenewalProcess(dist.Exponential(1.0), 10.0).sample(key, 64),
            SampleResult,
        )

    def test_positional_unpacking_unchanged(self):
        times, mask, n = _ipp().sample(jax.random.PRNGKey(0), 64)
        assert times.shape == (64,)
        assert mask.shape == (64,)
        assert n.ndim == 0

    def test_spatial_and_spatiotemporal_results(self):
        key = jax.random.PRNGKey(0)
        dom = RectangularDomain.from_size(jnp.array([2.0, 2.0]))
        assert isinstance(
            HomogeneousSpatialPP(rate=1.0, domain=dom).sample(key, 64),
            SpatialSampleResult,
        )
        st = HomogeneousSpatioTemporalPP(
            rate=0.5, spatial=dom, temporal=TemporalDomain.from_duration(2.0)
        )
        assert isinstance(st.sample(key, 64), SpatiotemporalSampleResult)

    def test_marked_sampler_returns_marked_result(self):
        marked = MarkedTemporalPointProcess(
            ground=_ipp(),
            mark_distribution_fn=lambda t, h: dist.Gamma(2.0, 1.0),
        )
        out = marked.sample(jax.random.PRNGKey(0), 64)
        assert isinstance(out, MarkedSampleResult)
        times, _mask, marks = out
        assert marks.shape == times.shape

    def test_sample_result_is_pytree(self):
        result = _ipp().sample(jax.random.PRNGKey(0), 32)
        doubled = jax.tree_util.tree_map(lambda x: x, result)
        assert isinstance(doubled, SampleResult)


class TestThinningTypeDispatch:
    """#65 — ThinningProcess dispatches on the result type, so marked and
    unmarked bases compose without shape heuristics."""

    def test_unmarked_base_returns_count(self):
        thin = ThinningProcess(
            base=_ipp(), retention_fn=lambda t, h, m=None: jnp.asarray(0.5)
        )
        out = thin.sample(jax.random.PRNGKey(0), max_events=64, max_candidates=128)
        assert isinstance(out, SampleResult)
        assert int(out.n_events) == int(jnp.sum(out.mask))

    def test_marked_base_returns_marks(self):
        marked = MarkedTemporalPointProcess(
            ground=_ipp(),
            mark_distribution_fn=lambda t, h: dist.Gamma(2.0, 1.0),
        )
        thin = ThinningProcess(
            base=marked, retention_fn=lambda t, h, m=None: jnp.asarray(0.7)
        )
        out = thin.sample(jax.random.PRNGKey(0), max_events=64, max_candidates=128)
        assert isinstance(out, MarkedSampleResult)
        assert out.marks.shape == out.times.shape

    def test_plain_tuple_base_still_works(self):
        """User-supplied bases returning bare tuples keep the legacy
        heuristic fallback."""

        class TupleBase:
            observation_window = jnp.asarray(10.0)
            rate = jnp.asarray(2.0)

            def sample(self, key, max_events):
                times, mask, n = _ipp().sample(key, max_events)
                return tuple((times, mask, n))  # strip the NamedTuple type

            def log_prob(self, times, mask):
                return _ipp().log_prob(times, mask)

            def intensity(self, t):
                return _ipp().intensity(t)

        thin = ThinningProcess(
            base=TupleBase(), retention_fn=lambda t, h, m=None: jnp.asarray(0.5)
        )
        out = thin.sample(jax.random.PRNGKey(0), max_events=64)
        assert isinstance(out, SampleResult)


class TestMixinDiagnostics:
    """#64 — the mixin-hosted GOF surface is present and consistent on
    every temporal family, including renewal (previously untested)."""

    def _times_mask(self):
        times = jnp.array([0.8, 1.9, 3.1, 4.4, 10.0, 10.0])
        mask = jnp.array([True, True, True, True, False, False])
        return times, mask

    @pytest.mark.parametrize(
        "make",
        [
            lambda: HomogeneousPoissonProcess(0.5, 10.0),
            lambda: _ipp(0.5),
            lambda: ExponentialHawkes(
                mu=0.3, alpha=0.2, beta=1.0, observation_window=10.0
            ),
            lambda: RenewalProcess(dist.Exponential(0.5), 10.0),
        ],
    )
    def test_goodness_of_fit_finite(self, make):
        op = make()
        times, mask = self._times_mask()
        gof = op.goodness_of_fit(times, mask)
        assert jnp.isfinite(gof.ks_statistic)
        assert jnp.all(jnp.isfinite(jnp.where(gof.mask, gof.residuals, 0.0)))
        # compensator_curve shares the same hook.
        ts, lam = op.compensator_curve(times, mask)
        assert ts.shape == lam.shape

    def test_renewal_gof_exponential_gaps_are_exp1(self):
        """For Exp(λ) gaps the rescaled residuals are exactly Exp(1), so
        the KS statistic on a large sample must be small."""
        op = RenewalProcess(dist.Exponential(2.0), 200.0)
        times, mask, _ = op.sample(jax.random.PRNGKey(0), 1024)
        gof = op.goodness_of_fit(times, mask)
        assert float(gof.ks_statistic) < 0.06
