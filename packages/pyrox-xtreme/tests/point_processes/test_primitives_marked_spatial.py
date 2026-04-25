"""Tests for :mod:`xtremax.point_processes.primitives.marked_spatial`."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from jax import random

from xtremax.point_processes.primitives import (
    sample_spatial_marks_at_locations,
    spatial_marks_log_prob,
)


@pytest.fixture
def locations_and_mask():
    locs = jnp.array([[1.0, 1.0], [2.0, 3.0], [4.0, 5.0], [0.0, 0.0]])
    mask = jnp.array([True, True, True, False])
    return locs, mask


class TestSpatialMarksLogProb:
    def test_independent_scalar_marks(self, locations_and_mask) -> None:
        locs, mask = locations_and_mask
        d = dist.Gamma(concentration=5.0, rate=2.0)
        marks = jnp.array([1.0, 2.0, 0.5, 0.0])

        def fn(s):
            return d  # location-independent

        result = spatial_marks_log_prob(locs, marks, mask, fn)
        # Hand-compute: ∑ d.log_prob(mᵢ) over mask=True.
        expected = jnp.sum(d.log_prob(marks[:3]))
        assert jnp.allclose(result, expected)

    def test_location_dependent_marks(self, locations_and_mask) -> None:
        locs, mask = locations_and_mask
        # Mark distribution rate decays with x-coordinate.
        marks = jnp.array([1.0, 1.5, 2.0, 0.0])

        def fn(s):
            return dist.Exponential(rate=1.0 / (1.0 + s[0]))

        result = spatial_marks_log_prob(locs, marks, mask, fn)
        # Hand-compute on the unmasked events.
        contributions = []
        for s, m in zip(locs[:3], marks[:3], strict=False):
            d_i = dist.Exponential(rate=1.0 / (1.0 + s[0]))
            contributions.append(d_i.log_prob(m))
        expected = sum(contributions)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_vector_marks(self, locations_and_mask) -> None:
        locs, mask = locations_and_mask
        d = dist.Normal(jnp.zeros(2), jnp.ones(2)).to_event(1)
        marks = jnp.array([[0.5, -0.5], [1.0, 1.0], [-1.0, 0.5], [0.0, 0.0]])

        def fn(s):
            return d

        result = spatial_marks_log_prob(locs, marks, mask, fn)
        expected = jnp.sum(d.log_prob(marks[:3]))
        assert jnp.allclose(result, expected)


class TestSampleSpatialMarks:
    def test_independent_scalar_marks_padding_zeroed(self, locations_and_mask) -> None:
        locs, mask = locations_and_mask
        d = dist.Gamma(concentration=5.0, rate=2.0)

        def fn(s):
            return d

        marks = sample_spatial_marks_at_locations(random.PRNGKey(0), locs, mask, fn)
        assert marks.shape == (4,)
        # Real events: positive Gamma draws. Padding: zeros.
        assert jnp.all(marks[:3] > 0.0)
        assert marks[3] == 0.0

    def test_vector_marks(self, locations_and_mask) -> None:
        locs, mask = locations_and_mask
        d = dist.MultivariateNormal(jnp.zeros(3), jnp.eye(3))

        def fn(s):
            return d

        marks = sample_spatial_marks_at_locations(random.PRNGKey(0), locs, mask, fn)
        assert marks.shape == (4, 3)
        # Padding zero across all components.
        assert jnp.all(marks[3] == 0.0)
