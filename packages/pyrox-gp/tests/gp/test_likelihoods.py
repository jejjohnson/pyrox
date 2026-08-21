"""Tests for ``pyrox_gp`` likelihood families.

Covers :class:`GaussianLikelihood` and :class:`DistLikelihood`, verifying
that ``log_prob`` matches reference numpyro distributions and that the
``DistLikelihood`` wrapper handles arbitrary observation models.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as nd
from pyrox_gp import DistLikelihood, GaussianLikelihood
from pyrox_gp._protocols import Likelihood


# --- GaussianLikelihood ---------------------------------------------------


def test_gaussian_log_prob_matches_numpyro_normal():
    f = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.1, 1.9, 3.2])
    noise_var = 0.25
    lik = GaussianLikelihood(noise_var=noise_var)
    ref = nd.Normal(f, jnp.sqrt(noise_var)).log_prob(y).sum()
    assert jnp.allclose(lik.log_prob(f, y), ref, atol=1e-5)


def test_gaussian_log_prob_zero_residual_is_maximum():
    f = jnp.ones(4)
    lik = GaussianLikelihood(noise_var=0.1)
    lp_zero = lik.log_prob(f, f)
    lp_shift = lik.log_prob(f, f + 0.5)
    assert lp_zero > lp_shift


def test_gaussian_log_prob_dtype_preserved():
    f = jnp.array([1.0, 2.0], dtype=jnp.float32)
    y = jnp.array([1.1, 2.1], dtype=jnp.float32)
    lik = GaussianLikelihood(noise_var=0.1)
    assert lik.log_prob(f, y).dtype == f.dtype


# --- DistLikelihood -------------------------------------------------------


def test_dist_bernoulli_log_prob_matches_numpyro():
    f = jnp.array([0.5, -0.5, 1.0])
    y = jnp.array([1.0, 0.0, 1.0])
    lik = DistLikelihood(dist_fn=lambda f: nd.Bernoulli(logits=f))
    ref = nd.Bernoulli(logits=f).log_prob(y).sum()
    assert jnp.allclose(lik.log_prob(f, y), ref, atol=1e-5)


def test_dist_poisson_log_prob_matches_numpyro():
    f = jnp.array([0.5, 1.0, -0.5])
    y = jnp.array([1, 3, 0], dtype=jnp.int32)
    lik = DistLikelihood(dist_fn=lambda f: nd.Poisson(rate=jnp.exp(f)))
    ref = nd.Poisson(rate=jnp.exp(f)).log_prob(y).sum()
    assert jnp.allclose(lik.log_prob(f, y), ref, atol=1e-5)


def test_dist_student_t_log_prob_matches_numpyro():
    f = jnp.array([0.0, 1.0, -1.0])
    y = jnp.array([0.1, 0.9, -1.2])
    lik = DistLikelihood(dist_fn=lambda f: nd.StudentT(df=3.0, loc=f, scale=0.5))
    ref = nd.StudentT(df=3.0, loc=f, scale=0.5).log_prob(y).sum()
    assert jnp.allclose(lik.log_prob(f, y), ref, atol=1e-5)


def test_dist_likelihood_is_a_pyrox_likelihood():
    lik = DistLikelihood(dist_fn=lambda f: nd.Normal(f, 1.0))
    assert isinstance(lik, Likelihood)


def test_gaussian_likelihood_is_a_pyrox_likelihood():
    lik = GaussianLikelihood(noise_var=0.1)
    assert isinstance(lik, Likelihood)


# --- DistLikelihood static dist_fn trap -----------------------------------


class _Warp(eqx.Module):
    """Warp with a learnable parameter ``a``."""

    a: jax.Array

    def __call__(self, f: jax.Array) -> jax.Array:
        return jnp.sinh(self.a * jnp.arcsinh(f))


class _WarpedNormalLikelihood(Likelihood):
    """Same warped-Normal model, with the warp held as a module field."""

    warp: _Warp

    def log_prob(self, f: jax.Array, y: jax.Array) -> jax.Array:
        return nd.Normal(self.warp(f), 0.5).log_prob(y).sum()


def test_dist_fn_is_static_and_freezes_closures():
    """``dist_fn`` closures are invisible to ``eqx.filter_grad``.

    Pins the trap documented in the ``DistLikelihood`` class-docstring
    warning: ``dist_fn`` is a static field, so parameters the callable
    closes over never appear in the gradient pytree and silently never
    train. This test documents the behaviour; it does not assert it is
    fixed.
    """
    warp = _Warp(a=jnp.asarray(1.3))
    lik = DistLikelihood(dist_fn=lambda f: nd.Normal(warp(f), 0.5))
    f = jnp.array([0.0, 1.0, -1.0])
    y = jnp.array([0.1, 0.9, -1.2])
    grad = eqx.filter_grad(lambda lik_: -lik_.log_prob(f, y))(lik)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grad, eqx.is_inexact_array))
    assert leaves == []


def test_module_field_link_parameters_receive_gradients():
    """Holding the warp as a module field makes ``a`` a trainable leaf."""
    lik = _WarpedNormalLikelihood(warp=_Warp(a=jnp.asarray(1.3)))
    f = jnp.array([0.0, 1.0, -1.0])
    y = jnp.array([0.1, 0.9, -1.2])
    grad = eqx.filter_grad(lambda lik_: -lik_.log_prob(f, y))(lik)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grad, eqx.is_inexact_array))
    assert len(leaves) > 0
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
