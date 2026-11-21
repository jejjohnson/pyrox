"""Tests for the `lfr_factor` call order (pyrox#520).

The new form ``lfr_factor(name, prior, Y, Z, noise_var, *, beta=None)``
reads the warp from the prior; the old form ``lfr_factor(Y, Z, noise_var,
*, warp=None, beta=None, name=...)`` warns and registers the same factor.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.bijections import Affine
from numpyro import handlers
from pyrox_gp import RBF, LatentFactorGPPrior, lfr_factor


N, P, Q = 6, 4, 2
X = jr.uniform(jr.key(0), (N, 1))
Y = jr.normal(jr.key(1), (N, P))
Z = jr.normal(jr.key(2), (N, Q))
NOISE = jnp.asarray(0.3)
WARP = Affine(loc=jnp.full(P, 0.2), scale=jnp.full(P, 1.5))


def _prior(warp=None) -> LatentFactorGPPrior:
    kernels = tuple(RBF(pyrox_name=f"RBF_q{q}") for q in range(Q))
    return LatentFactorGPPrior(kernels=kernels, X=X, warp=warp)


def _site(model, name):
    site = handlers.trace(handlers.seed(model, 0)).get_trace()[name]
    return site["fn"].log_prob(site["value"]), site["scale"]


@pytest.mark.parametrize("warp", [None, WARP])
@pytest.mark.parametrize("beta", [None, 0.5])
def test_new_and_old_forms_register_the_same_factor(warp, beta):
    prior = _prior(warp)

    def new():
        lfr_factor("y", prior, Y, Z, NOISE, beta=beta)

    def old():
        lfr_factor(Y, Z, NOISE, warp=warp, beta=beta, name="y")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new_value, new_scale = _site(new, "y")
    with pytest.warns(DeprecationWarning, match=r"lfr_factor\(name, prior"):
        old_value, old_scale = _site(old, "y")
    assert jnp.array_equal(new_value, old_value)
    assert new_scale == old_scale == (Q / P if beta is None else beta)


def test_new_form_accepts_keywords():
    prior = _prior()

    def model():
        lfr_factor(name="y", prior=prior, Y=Y, Z=Z, noise_var=NOISE)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        value, _ = _site(model, "y")
    assert jnp.isfinite(value)


def test_old_form_default_site_name_still_collapsed_lfr():
    def model():
        lfr_factor(Y, Z, NOISE)

    with pytest.warns(DeprecationWarning):
        trace = handlers.trace(handlers.seed(model, 0)).get_trace()
    assert list(trace) == ["collapsed_lfr"]


def test_new_form_rejects_shapes_that_do_not_match_the_prior():
    prior = _prior()
    with pytest.raises(ValueError, match=r"rows to match prior\.X"):
        lfr_factor("y", prior, Y[:-1], Z, NOISE)
    with pytest.raises(ValueError, match=r"Z must have shape \(6, 2\)"):
        lfr_factor("y", prior, Y, Z[:, :1], NOISE)
