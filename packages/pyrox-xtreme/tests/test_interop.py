"""Structural pipekit-interop tests (#76).

xtremax must stay pipekit-free (see ``docs/interop.md``), so
conformance is asserted against a local mirror of the relevant
runtime-checkable Protocols; a real-pipekit `isinstance` check runs
only when pipekit happens to be installed.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtremax.distributions import (
    GeneralizedExtremeValueDistribution,
    GeneralizedParetoDistribution,
)


@runtime_checkable
class _ObservationNoise(Protocol):
    """Local mirror of ``pipekit_cycle.protocols.ObservationNoise``."""

    def covariance(self) -> Any: ...

    def sample(self, key: Any, shape: tuple[int, ...]) -> Any: ...


@runtime_checkable
class _Predictor(Protocol):
    """Local mirror of ``pipekit.protocols.Predictor``."""

    def predict(self, x: Any) -> Any: ...


class TestCovariance:
    def test_scalar_equals_variance(self):
        d = GeneralizedExtremeValueDistribution(1.0, 2.0, concentration=0.1)
        np.testing.assert_allclose(np.asarray(d.covariance()), np.asarray(d.variance))
        g = GeneralizedParetoDistribution(2.0, concentration=0.1)
        np.testing.assert_allclose(np.asarray(g.covariance()), np.asarray(g.variance))

    def test_batched_broadcasts_to_batch_shape(self):
        d = GeneralizedExtremeValueDistribution(
            loc=jnp.array([0.0, 1.0, 2.0]),
            scale=2.0,
            concentration=jnp.array([0.1, -0.2, 0.3]),
        )
        cov = d.covariance()
        assert cov.shape == d.batch_shape == (3,)
        np.testing.assert_allclose(
            np.asarray(cov), np.asarray(jnp.broadcast_to(d.variance, (3,)))
        )

    def test_infinite_variance_propagates(self):
        """ξ >= 1/2: the moment does not exist; the seam reports +inf
        rather than a silently wrong finite number."""
        d = GeneralizedParetoDistribution(1.0, concentration=0.7)
        assert np.isposinf(float(d.covariance()))


class TestStructuralConformance:
    @pytest.mark.parametrize(
        "d",
        [
            GeneralizedExtremeValueDistribution(0.0, 1.0, concentration=0.1),
            GeneralizedParetoDistribution(1.0, concentration=0.1),
        ],
        ids=lambda d: type(d).__name__,
    )
    def test_observation_noise_protocol(self, d):
        assert isinstance(d, _ObservationNoise)
        # And the members actually work with the protocol's call shape —
        # including the protocol's *keyword* spelling `shape=`, which a
        # caller typed against the Protocol may legitimately use (the
        # runtime isinstance check only verifies member presence).
        cov = d.covariance()
        draws = d.sample(jax.random.PRNGKey(0), (5,))
        draws_kw = d.sample(jax.random.PRNGKey(0), shape=(5,))
        assert np.asarray(cov).shape == ()
        assert draws.shape == (5,)
        np.testing.assert_array_equal(np.asarray(draws), np.asarray(draws_kw))

    def test_batched_noise_draw_semantics(self):
        """`shape` is the *leading draw batch* (the protocol's "batch
        shape"), prepended to the distribution's own batch shape — the
        NumPyro `sample_shape` convention. A d-element heteroscedastic
        noise model asked for n draws returns (n, d): n error vectors of
        dimension d, which is what perturbed-observation ensemble
        methods consume. `shape` is NOT the full output shape."""
        d = GeneralizedExtremeValueDistribution(
            loc=jnp.array([0.0, 1.0, 2.0]),
            scale=1.0,
            concentration=0.1,
        )
        assert d.batch_shape == (3,)
        draws = d.sample(jax.random.PRNGKey(0), shape=(2,))
        assert draws.shape == (2, 3)

    @pytest.mark.parametrize(
        "d",
        [
            GeneralizedExtremeValueDistribution(0.0, 1.0, concentration=0.1),
            GeneralizedParetoDistribution(1.0, concentration=0.1),
        ],
        ids=lambda d: type(d).__name__,
    )
    def test_both_shape_spellings_raise(self, d):
        """Passing both spellings must fail loudly — a silent override
        would produce valid-looking draws with wrong leading dims."""
        with pytest.raises(ValueError, match="only one"):
            d.sample(jax.random.PRNGKey(0), (100,), shape=(10,))

    def test_quantile_regressor_is_predictor(self):
        sklearn_present = importlib.util.find_spec("sklearn") is not None
        if not sklearn_present:
            pytest.skip("sklearn not installed (optional [threshold] extra)")
        from xtremax.extraction.quantile_regression import XarrayQuantileRegressor

        assert isinstance(XarrayQuantileRegressor(quantile=0.9), _Predictor)

    @pytest.mark.skipif(
        importlib.util.find_spec("pipekit_cycle") is None,
        reason="pipekit-cycle not installed (interop is structural-only)",
    )
    def test_real_pipekit_protocol(self):
        from pipekit_cycle.protocols import ObservationNoise

        d = GeneralizedExtremeValueDistribution(0.0, 1.0, concentration=0.1)
        assert isinstance(d, ObservationNoise)
