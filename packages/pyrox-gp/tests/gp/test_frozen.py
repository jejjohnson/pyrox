"""`_ParameterizedKernel.frozen` and the kernellib delegation in `_basis`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import kernellib as kl
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from pyrox_gp import (
    RBF,
    Constant,
    Cosine,
    Linear,
    Matern,
    Periodic,
    Polynomial,
    RationalQuadratic,
    White,
)
from pyrox_gp._basis import draw_rff_cosine_basis, spectral_density
from pyrox_gp._context import _kernel_context


X = jnp.linspace(-1.0, 1.0, 5)[:, None]


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        (RBF(init_lengthscale=0.4, init_variance=1.3), kl.RBF),
        (Matern(nu=1.5), kl.Matern),
        (Periodic(), kl.Periodic),
        (Linear(), kl.Linear),
        (RationalQuadratic(), kl.RationalQuadratic),
        (Polynomial(degree=3), kl.Polynomial),
        (Cosine(), kl.Cosine),
        (White(), kl.White),
        (Constant(), kl.Constant),
    ],
    ids=lambda v: getattr(v, "__name__", type(v).__name__),
)
def test_frozen_matches_the_parameterized_kernel(kernel, expected):
    frozen = kernel.frozen()
    assert type(frozen) is expected
    assert jnp.allclose(frozen(X, X), kernel(X, X))


def test_frozen_copies_static_fields():
    assert Matern(nu=0.5).frozen().nu == 0.5
    assert Polynomial(degree=4).frozen().degree == 4


def _prior_rbf(name):
    k = RBF(pyrox_name=name)
    k.set_prior("lengthscale", dist.LogNormal(0.0, 1.0))
    return k


def test_frozen_registers_one_site_per_prior():
    # Design risk 8: freezing inside NumPyro tracing must not duplicate
    # sites. Two freezes in one context share one draw.
    k = _prior_rbf("RBF_frozen_trace")

    def model():
        with _kernel_context(k):
            return k.frozen(), k.frozen()

    tr = handlers.trace(handlers.seed(model, 0)).get_trace()
    assert [n for n in tr if tr[n]["type"] == "sample"] == [
        "RBF_frozen_trace.lengthscale"
    ]
    a, b = handlers.seed(model, 0)()
    assert jnp.array_equal(a.lengthscale, b.lengthscale)


def test_frozen_shares_the_draw_with_kernel_evaluation():
    k = _prior_rbf("RBF_frozen_shared")

    def model():
        with _kernel_context(k):
            return k.frozen()(X, X), k(X, X)

    frozen_gram, gram = handlers.seed(model, 1)()
    assert jnp.allclose(frozen_gram, gram)


def test_overrides_are_not_resolved():
    k = _prior_rbf("RBF_frozen_override")
    tr = handlers.trace(
        handlers.seed(lambda: k.frozen(lengthscale=2.0, variance=0.5), 0)
    ).get_trace()
    assert "RBF_frozen_override.lengthscale" not in tr
    frozen = handlers.seed(lambda: k.frozen(lengthscale=2.0), 0)()
    assert frozen.lengthscale == 2.0


def test_unknown_override_raises():
    with pytest.raises(ValueError, match="period"):
        RBF().frozen(period=1.0)


def test_spectral_density_accepts_kernellib_kernels():
    lam = jnp.linspace(0.0, 4.0, 5)
    pyrox = spectral_density(Matern(nu=1.5, init_lengthscale=0.6), lam, D=2)
    plain = spectral_density(kl.Matern(nu=1.5, lengthscale=0.6), lam, D=2)
    assert jnp.allclose(pyrox, plain)


def test_spectral_density_rational_quadratic_has_no_closed_form():
    with pytest.raises(NotImplementedError, match="RationalQuadratic"):
        spectral_density(RationalQuadratic(), jnp.zeros(3), D=1)


def test_rff_draw_accepts_kernellib_kernels_and_overrides():
    key = jax.random.PRNGKey(0)
    kwargs = {"n_paths": 2, "n_features": 4, "in_features": 1, "dtype": jnp.float32}
    v, ell, omega, *_ = draw_rff_cosine_basis(kl.RBF(lengthscale=0.3), key, **kwargs)
    assert ell == pytest.approx(0.3)
    v, ell, omega2, *_ = draw_rff_cosine_basis(
        kl.RBF(lengthscale=0.3), key, variance=2.0, lengthscale=0.7, **kwargs
    )
    assert (v, ell) == (pytest.approx(2.0), pytest.approx(0.7))
    # Unit-lengthscale frequencies: the overrides do not change the draw.
    assert jnp.array_equal(omega, omega2)


def test_rff_draw_now_covers_rational_quadratic():
    omega = draw_rff_cosine_basis(
        RationalQuadratic(),
        jax.random.PRNGKey(0),
        n_paths=2,
        n_features=4,
        in_features=3,
        dtype=jnp.float32,
    )[2]
    assert omega.shape == (2, 3, 4)
