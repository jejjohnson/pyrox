"""`pad_sites`: ragged per-site series padded to one shape with a mask.

The padded times continue each site at its median spacing, so they stay
strictly increasing; the mask is ``True`` at exactly the ``N_s`` observed
steps; and a padded, masked site has the unpadded site's log marginal to
``rtol=1e-12`` under x64 (the padded steps are predict-only and add exactly
zero).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from pyrox_ssm import MarkovGPPrior, MaternSDE, pad_sites


jax.config.update("jax_enable_x64", True)

NOISE = jnp.asarray(0.05)


def _sites():
    times = [
        jnp.array([0.0, 0.4, 1.1, 1.5, 2.6, 3.0]),
        jnp.array([0.2, 0.9, 1.3]),
        jnp.array([2.0]),
    ]
    ys = [jnp.sin(t) for t in times]
    return times, ys


def test_padded_times_increase_and_mask_counts_match() -> None:
    times, ys = _sites()
    t, y, mask = pad_sites(times, ys)
    assert t.shape == y.shape == mask.shape == (3, 6)
    assert mask.dtype == jnp.bool_
    assert jnp.all(jnp.diff(t, axis=1) > 0)
    assert mask.sum(axis=1).tolist() == [6, 3, 1]
    for s, (t_s, y_s) in enumerate(zip(times, ys, strict=True)):
        n_s = t_s.shape[0]
        assert jnp.array_equal(t[s, :n_s], t_s)
        assert jnp.array_equal(y[s, :n_s], y_s)
        assert jnp.all(y[s, n_s:] == 0.0)
        assert jnp.all(mask[s, :n_s]) and not jnp.any(mask[s, n_s:])


def test_padding_steps_at_the_median_spacing() -> None:
    times, ys = _sites()
    t, _, _ = pad_sites(times, ys)
    # site 1: diffs (0.7, 0.4), median 0.55
    assert jnp.allclose(t[1, 3:], 1.3 + 0.55 * jnp.arange(1, 4))
    # site 2 has one observation: unit steps
    assert jnp.allclose(t[2, 1:], 2.0 + jnp.arange(1, 6))


def test_length_pads_beyond_the_longest_site() -> None:
    times, ys = _sites()
    t, _, mask = pad_sites(times, ys, length=9)
    assert t.shape == (3, 9)
    assert mask.sum(axis=1).tolist() == [6, 3, 1]
    assert jnp.all(jnp.diff(t, axis=1) > 0)


@pytest.mark.parametrize("order", [0, 1], ids=["matern12", "matern32"])
def test_padded_masked_log_marginal_equals_the_unpadded_site(order: int) -> None:
    times, ys = _sites()
    t, y, mask = pad_sites(times, ys, length=8)
    sde = MaternSDE(variance=1.3, lengthscale=0.6, order=order)
    for s, (t_s, y_s) in enumerate(zip(times, ys, strict=True)):
        padded = MarkovGPPrior(sde, t[s]).log_marginal(y[s], NOISE, mask=mask[s])
        unpadded = MarkovGPPrior(sde, t_s).log_marginal(y_s, NOISE)
        assert jnp.allclose(padded, unpadded, rtol=1e-12, atol=0)


def test_ragged_inputs_are_validated() -> None:
    times, ys = _sites()
    with pytest.raises(ValueError, match="sites but ys has"):
        pad_sites(times, ys[:2])
    with pytest.raises(ValueError, match="at least one site"):
        pad_sites([], [])
    with pytest.raises(ValueError, match="shorter than the longest site"):
        pad_sites(times, ys, length=4)
    with pytest.raises(ValueError, match="strictly increasing"):
        pad_sites([jnp.array([0.0, 2.0, 1.0])], [jnp.zeros(3)])
    with pytest.raises(ValueError, match="each observation needs one time"):
        pad_sites([jnp.array([0.0, 1.0])], [jnp.zeros(3)])
    with pytest.raises(ValueError, match="non-empty 1-D"):
        pad_sites([jnp.zeros(0)], [jnp.zeros(0)])
