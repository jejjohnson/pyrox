"""Tests for :mod:`xtremax.point_processes._domain.RectangularDomain`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import random

from xtremax.point_processes import RectangularDomain


class TestRectangularDomain:
    @pytest.mark.parametrize("d", [1, 2, 3, 5])
    def test_from_size_volume(self, d: int) -> None:
        size = jnp.arange(1.0, d + 1.0)
        domain = RectangularDomain.from_size(size)
        assert domain.n_dims == d
        # |D| = ∏ size_i.
        assert jnp.allclose(domain.volume(), jnp.prod(size))

    def test_lo_hi_construction(self) -> None:
        lo = jnp.array([1.0, -2.0])
        hi = jnp.array([3.0, 5.0])
        domain = RectangularDomain(lo, hi)
        assert domain.n_dims == 2
        assert jnp.allclose(domain.side_lengths, jnp.array([2.0, 7.0]))
        assert jnp.allclose(domain.volume(), 14.0)

    def test_contains(self) -> None:
        domain = RectangularDomain.from_size(jnp.array([10.0, 10.0]))
        x = jnp.array([[5.0, 5.0], [-1.0, 5.0], [5.0, 11.0]])
        assert jnp.array_equal(domain.contains(x), jnp.array([True, False, False]))

    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_sample_uniform_in_domain(self, d: int) -> None:
        domain = RectangularDomain.from_size(jnp.full((d,), 5.0))
        key = random.PRNGKey(0)
        pts = domain.sample_uniform(key, shape=(1024,))
        assert pts.shape == (1024, d)
        # All points in [0, 5)^d.
        assert jnp.all(pts >= 0.0)
        assert jnp.all(pts < 5.0 + 1e-6)
