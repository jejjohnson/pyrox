"""Tests for :mod:`xtremax.point_processes.primitives.hpp_spatial`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import random

from xtremax.point_processes import RectangularDomain
from xtremax.point_processes.primitives import (
    csr_l_function,
    csr_pair_correlation,
    csr_ripleys_k,
    hpp_spatial_count_log_prob,
    hpp_spatial_log_prob,
    hpp_spatial_nearest_neighbor_distance,
    hpp_spatial_predict_count,
    hpp_spatial_sample,
)


class TestHppSpatialLogProb:
    def test_matches_closed_form(self) -> None:
        # Janossy form: log L = n log λ - λ|D|. (No -n log|D| term —
        # see Codex P2 in PR #12 review for the rationale.)
        rate = 0.5
        vol = 100.0
        n = jnp.asarray(20)
        expected = 20 * jnp.log(rate) - rate * vol
        assert jnp.allclose(hpp_spatial_log_prob(n, rate, vol), expected)

    def test_matches_ipp_for_constant_intensity(self) -> None:
        # The HPP log-prob must equal the IPP log-prob under constant
        # intensity. Codex P2 regression.
        from xtremax.point_processes.primitives import ipp_spatial_log_prob

        rate = 0.7
        vol = 9.0
        locations = jnp.array(
            [[1.0, 1.0], [1.5, 2.0], [2.5, 0.5], [0.0, 0.0], [0.0, 0.0]]
        )
        mask = jnp.array([True, True, True, False, False])

        def log_lam(s):
            return jnp.full(s.shape[:-1], jnp.log(rate))

        ipp_lp = ipp_spatial_log_prob(locations, mask, log_lam, rate * vol)
        hpp_lp = hpp_spatial_log_prob(jnp.asarray(3), rate, vol)
        assert jnp.allclose(ipp_lp, hpp_lp)

    def test_grad_through_rate(self) -> None:
        # d(log L)/dλ = n/λ - |D|.
        n = jnp.asarray(15.0)
        vol = jnp.asarray(50.0)
        rate = 0.4
        grad = jax.grad(lambda r: hpp_spatial_log_prob(n, r, vol))(rate)
        expected = 15.0 / rate - 50.0
        assert jnp.allclose(grad, expected)

    def test_count_log_prob_normalises(self) -> None:
        # ∑_n exp(log P(N=n)) ≈ 1 for a finite truncation around the mean.
        rate = 1.5
        vol = 4.0
        ns = jnp.arange(50)
        log_pmf = jax.vmap(hpp_spatial_count_log_prob, in_axes=(0, None, None))(
            ns, rate, vol
        )
        assert jnp.isclose(jnp.sum(jnp.exp(log_pmf)), 1.0, atol=1e-6)


class TestHppSpatialSample:
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_shapes_and_mask(self, d: int) -> None:
        domain = RectangularDomain.from_size(jnp.full((d,), 5.0))
        key = random.PRNGKey(0)
        locs, mask, n = hpp_spatial_sample(key, rate=0.3, domain=domain, max_events=64)
        assert locs.shape == (64, d)
        assert mask.shape == (64,)
        # ``n`` is uncapped; ``mask.sum`` is capped at max_events.
        assert mask.sum() == jnp.minimum(n, 64)
        # All real points lie inside the domain.
        for i in range(d):
            assert jnp.all(locs[mask, i] >= domain.lo[i])
            assert jnp.all(locs[mask, i] < domain.hi[i] + 1e-6)

    def test_empirical_count_matches_lambda_volume(self) -> None:
        domain = RectangularDomain.from_size(jnp.array([4.0, 4.0]))
        rate = 0.5  # E[N] = 0.5 * 16 = 8.
        keys = random.split(random.PRNGKey(42), 2000)

        def draw(k):
            return hpp_spatial_sample(k, rate, domain, max_events=128)[2]

        counts = jax.vmap(draw)(keys)
        # SE ≈ √(8/2000) ≈ 0.063, so a 0.3 tolerance is comfortable.
        assert jnp.abs(jnp.mean(counts.astype(jnp.float32)) - 8.0) < 0.3

    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_padding_locations_in_domain(self, d: int) -> None:
        # Padding rows are set to ``domain.lo`` so downstream callers
        # that evaluate ``log_intensity_fn`` on the whole buffer don't
        # see ``+inf`` arguments.
        domain = RectangularDomain.from_size(jnp.full((d,), 5.0))
        locs, mask, _ = hpp_spatial_sample(
            random.PRNGKey(0), rate=0.01, domain=domain, max_events=32
        )
        # With λ small, almost all rows are padding.
        padding_rows = locs[~mask]
        assert jnp.all(jnp.isfinite(padding_rows))
        assert jnp.all(padding_rows == domain.lo)


class TestHppSpatialDiagnostics:
    @pytest.mark.parametrize(
        ("d", "expected_K"),
        [
            (1, lambda r: 2.0 * r),
            (2, lambda r: jnp.pi * r**2),
            (3, lambda r: (4.0 / 3.0) * jnp.pi * r**3),
        ],
    )
    def test_ripleys_k_closed_form(self, d, expected_K):
        r = jnp.linspace(0.1, 2.0, 5)
        assert jnp.allclose(csr_ripleys_k(r, d), expected_K(r))

    @pytest.mark.parametrize("d", [1, 2, 3, 5])
    def test_l_function_equals_r_under_csr(self, d):
        r = jnp.linspace(0.1, 1.0, 4)
        # Under CSR, L(r) = r identically.
        assert jnp.allclose(csr_l_function(r, d), r, atol=1e-4)

    def test_pair_correlation_under_csr(self):
        r = jnp.linspace(0.1, 1.0, 5)
        assert jnp.allclose(csr_pair_correlation(r, 2), jnp.ones_like(r))

    def test_predict_count(self):
        mean, var = hpp_spatial_predict_count(rate=0.5, subdomain_volume=8.0)
        assert jnp.allclose(mean, 4.0)
        assert jnp.allclose(var, 4.0)

    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_nearest_neighbor_finite_positive(self, d):
        # Just check we get a finite, positive number for reasonable rates.
        r_nn = hpp_spatial_nearest_neighbor_distance(rate=1.0, n_dims=d)
        assert jnp.isfinite(r_nn)
        assert r_nn > 0.0
