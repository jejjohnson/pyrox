"""Names that moved out of pyrox-gp still work, and warn with the replacement."""

from __future__ import annotations

import importlib

import jax.numpy as jnp
import kernellib as kl
import numpy as np
import pytest
from numpyro import handlers
from pyrox._core import call_context, freeze
from pyrox_gp import RBF, Matern, RationalQuadratic, funk_hecke_coefficients
from pyrox_gp._basis import _FUNCTAX_BASIS_FAMILIES, spectral_density
from pyrox_gp._context import _kernel_context, _kernel_contexts


def test_kernel_context_alias_warns_and_shares_one_draw():
    kernel = RBF(init_lengthscale=0.5)
    kernel.set_prior("lengthscale", _lognormal())
    with pytest.warns(DeprecationWarning, match=r"pyrox\._core\.call_context"):
        scope = _kernel_context(kernel)
    with handlers.seed(rng_seed=0), scope:
        assert kernel.frozen().lengthscale == kernel.frozen().lengthscale


def test_kernel_contexts_alias_warns_and_deduplicates():
    kernel = RBF(init_lengthscale=0.5)
    kernel.set_prior("lengthscale", _lognormal())
    with pytest.warns(DeprecationWarning, match=r"pyrox\._core\.call_contexts"):
        scope = _kernel_contexts([kernel, kernel])
    with handlers.seed(rng_seed=0), scope:
        assert kernel.frozen().lengthscale == kernel.frozen().lengthscale


def test_spectral_density_alias_warns_and_matches_kernellib():
    eigenvalues = jnp.array([0.0, 0.5, 2.0])
    with pytest.warns(DeprecationWarning, match="unit_spectral_density"):
        got = spectral_density(
            RBF(init_lengthscale=0.7, init_variance=1.5), eigenvalues
        )
    plain = kl.RBF(lengthscale=0.7, variance=1.5)
    # The 1-D frequency vectors the kernellib method takes.
    expected = plain.spectral_density(jnp.sqrt(eigenvalues)[:, None])
    np.testing.assert_allclose(got, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "kernel",
    [
        RBF(init_lengthscale=0.7, init_variance=1.5),
        Matern(nu=1.5, init_lengthscale=0.7),
        RationalQuadratic(init_lengthscale=0.7, init_alpha=2.0),
    ],
)
def test_funk_hecke_alias_warns_and_matches_a_direct_quadrature(kernel):
    with pytest.warns(DeprecationWarning, match="kernellib.funk_hecke_coefficients"):
        got = funk_hecke_coefficients(kernel, 6, num_quadrature=256)
    # a_l = 2π ∫ κ(t) P_l(t) dt, the convention pyrox-gp has always used.
    t, w = np.polynomial.legendre.leggauss(256)
    pole = jnp.array([[0.0, 0.0, 1.0]])
    meridian = jnp.stack([jnp.sqrt(1.0 - t**2), jnp.zeros_like(t), t], axis=-1)
    kappa = np.asarray(freeze(kernel)(pole, meridian))[0]
    expected = [
        2.0 * np.pi * np.sum(w * kappa * np.polynomial.legendre.legval(t, np.eye(7)[l]))
        for l in range(7)
    ]
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_funk_hecke_alias_resolves_a_prior_once():
    kernel = RBF(init_lengthscale=0.7)
    kernel.set_prior("lengthscale", _lognormal())
    with (
        handlers.trace() as tr,
        handlers.seed(rng_seed=0),
        call_context(kernel),
        pytest.warns(DeprecationWarning),
    ):
        funk_hecke_coefficients(kernel, 3)
    samples = [name for name, site in tr.items() if site["type"] == "sample"]
    assert samples == ["RBF.lengthscale"]


@pytest.mark.parametrize(
    ("name", "family"),
    sorted(_FUNCTAX_BASIS_FAMILIES.items()),
)
def test_basis_reexport_warns_and_returns_the_functax_object(name, family):
    import pyrox_gp._basis as basis

    with pytest.warns(DeprecationWarning, match=rf"functax\.basis\.{family}"):
        got = getattr(basis, name)
    assert got is getattr(importlib.import_module(f"functax.basis.{family}"), name)


def test_basis_rejects_an_unknown_name():
    import pyrox_gp._basis as basis

    with pytest.raises(AttributeError, match="no_such_basis"):
        basis.no_such_basis  # noqa: B018


def _lognormal():
    import numpyro.distributions as dist

    return dist.LogNormal(0.0, 1.0)
