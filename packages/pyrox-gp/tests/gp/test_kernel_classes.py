"""Tests for concrete Parameterized kernel classes in pyrox_gp.

Verify (1) that each class evaluates to the same thing as its pure-math
counterpart, (2) that Parameterized state (priors, mode switching) composes
correctly, and (3) site-name scoping — sibling instances share a site
by design when ``pyrox_name`` is the same, while distinct ``pyrox_name``
values keep their sites separate so stacked kernels compose cleanly.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from pyrox_gp import (
    RBF,
    Constant,
    Cosine,
    Kernel,
    Linear,
    Matern,
    Periodic,
    Polynomial,
    RationalQuadratic,
    White,
)
from pyrox_gp._src import kernels as _k


X = jnp.array([[0.0], [0.5], [1.0], [1.5]])


# --- Kernel protocol conformance ------------------------------------------


@pytest.mark.parametrize(
    "kernel",
    [
        RBF(),
        Matern(nu=0.5),
        Matern(nu=1.5),
        Matern(nu=2.5),
        Periodic(),
        Linear(),
        RationalQuadratic(),
        Polynomial(degree=2),
        Cosine(),
        White(),
        Constant(),
    ],
)
def test_kernel_is_instance_of_protocol(kernel):
    assert isinstance(kernel, Kernel)


# --- Numerical agreement with primitives ----------------------------------


def test_rbf_matches_primitive():
    k = RBF(init_variance=1.3, init_lengthscale=0.7)
    with handlers.seed(rng_seed=0):
        K = k(X, X)
    expected = _k.rbf_kernel(X, X, jnp.array(1.3), jnp.array(0.7))
    assert jnp.allclose(K, expected)


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_matern_matches_primitive(nu):
    k = Matern(init_variance=0.9, init_lengthscale=0.4, nu=nu)
    with handlers.seed(rng_seed=0):
        K = k(X, X)
    expected = _k.matern_kernel(X, X, jnp.array(0.9), jnp.array(0.4), nu)
    assert jnp.allclose(K, expected)


def test_periodic_matches_primitive():
    k = Periodic(init_variance=1.1, init_lengthscale=0.6, init_period=1.3)
    with handlers.seed(rng_seed=0):
        K = k(X, X)
    expected = _k.periodic_kernel(X, X, jnp.array(1.1), jnp.array(0.6), jnp.array(1.3))
    assert jnp.allclose(K, expected)


def test_linear_matches_primitive():
    k = Linear(init_variance=0.5, init_bias=0.2)
    with handlers.seed(rng_seed=0):
        K = k(X, X)
    expected = _k.linear_kernel(X, X, jnp.array(0.5), jnp.array(0.2))
    assert jnp.allclose(K, expected)


def test_rational_quadratic_matches_primitive():
    k = RationalQuadratic(init_variance=1.4, init_lengthscale=0.5, init_alpha=2.0)
    with handlers.seed(rng_seed=0):
        K = k(X, X)
    expected = _k.rational_quadratic_kernel(
        X, X, jnp.array(1.4), jnp.array(0.5), jnp.array(2.0)
    )
    assert jnp.allclose(K, expected)


def test_polynomial_matches_primitive():
    k = Polynomial(init_variance=0.8, init_bias=0.3, degree=3)
    with handlers.seed(rng_seed=0):
        K = k(X, X)
    expected = _k.polynomial_kernel(X, X, jnp.array(0.8), jnp.array(0.3), 3)
    assert jnp.allclose(K, expected)


def test_cosine_matches_primitive():
    k = Cosine(init_variance=1.2, init_period=2.0)
    with handlers.seed(rng_seed=0):
        K = k(X, X)
    expected = _k.cosine_kernel(X, X, jnp.array(1.2), jnp.array(2.0))
    assert jnp.allclose(K, expected)


def test_white_matches_primitive():
    k = White(init_variance=0.9)
    with handlers.seed(rng_seed=0):
        K = k(X, X)
    expected = _k.white_kernel(X, X, jnp.array(0.9))
    assert jnp.allclose(K, expected)


def test_constant_matches_primitive():
    k = Constant(init_variance=1.5)
    with handlers.seed(rng_seed=0):
        K = k(X, X)
    expected = _k.constant_kernel(X, X, jnp.array(1.5))
    assert jnp.allclose(K, expected)


# --- diag() overrides are consistent with the full Gram ------------------


@pytest.mark.parametrize(
    "kernel",
    [
        RBF(),
        Matern(),
        Periodic(),
        Linear(),
        RationalQuadratic(),
        Polynomial(),
        Cosine(),
        Constant(),
    ],
)
def test_diag_matches_gram_diagonal(kernel):
    with handlers.seed(rng_seed=0):
        K = kernel(X, X)
        d = kernel.diag(X)
    assert jnp.allclose(d, jnp.diag(K), atol=1e-5)


# --- Parameterized composition --------------------------------------------


def test_kernel_param_sites_registered_in_trace():
    k = RBF()
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        _ = k(X, X)
    assert "RBF.variance" in tr
    assert "RBF.lengthscale" in tr
    assert tr["RBF.variance"]["type"] == "param"


def test_kernel_with_prior_samples_in_model_mode():
    k = RBF()
    k.set_prior("variance", dist.LogNormal(0.0, 1.0))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        _ = k(X, X)
    assert tr["RBF.variance"]["type"] == "sample"
    assert tr["RBF.lengthscale"]["type"] == "param"


def test_kernel_with_prior_delta_guide_registers_param_backed_delta():
    """The `delta` guide exposes the latent as a param-backed ``Delta``
    *sample* site (``{name}_loc`` param + ``{name}`` sample) so NumPyro's
    ``replay`` handler conditions the model on it under SVI. See pyrox #182.
    """
    k = RBF()
    k.set_prior("variance", dist.LogNormal(0.0, 1.0))
    k.set_mode("guide")
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        _ = k(X, X)
    assert tr["RBF.variance"]["type"] == "sample"
    assert isinstance(tr["RBF.variance"]["fn"], dist.Delta)
    assert tr["RBF.variance_loc"]["type"] == "param"


# --- Sibling instances must not collide ----------------------------------


def test_two_rbf_instances_same_pyrox_name_collide_by_design():
    """Without user override, two RBFs share ``pyrox_name='RBF'``; their
    sites are the same on purpose (users who want distinct sites should
    override ``pyrox_name``)."""
    k1 = RBF()
    k2 = RBF()
    assert k1._pyrox_scope_name() == k2._pyrox_scope_name() == "RBF"


def test_distinct_pyrox_names_prevent_collision():
    """Subclasses / instances can override pyrox_name for stacking."""

    class RBF_A(RBF):
        pyrox_name: str = "RBF_A"

    class RBF_B(RBF):
        pyrox_name: str = "RBF_B"

    k1 = RBF_A()
    k2 = RBF_B()

    def model():
        k1(X, X)
        k2(X, X)

    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model()
    assert "RBF_A.variance" in tr
    assert "RBF_B.variance" in tr


# --- ARD opt-in via input_dim ---------------------------------------------


X3 = jnp.array(
    [
        [0.0, 1.0, 2.0],
        [0.5, 0.0, 1.0],
        [1.0, 2.0, 0.0],
        [1.5, 0.5, 0.5],
    ]
)


@pytest.mark.parametrize(
    "cls", [RBF, Matern, RationalQuadratic], ids=lambda c: c.__name__
)
def test_input_dim_registers_per_dimension_lengthscale(cls):
    assert cls(input_dim=3).get_param("lengthscale").shape == (3,)
    assert cls().get_param("lengthscale").shape == ()


def test_rbf_ard_matches_primitive():
    k = RBF(input_dim=3, init_variance=1.3, init_lengthscale=0.7)
    with handlers.seed(rng_seed=0):
        K = k(X3, X3)
    expected = _k.rbf_kernel(X3, X3, jnp.array(1.3), jnp.full((3,), 0.7))
    assert jnp.allclose(K, expected)


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_matern_ard_matches_primitive(nu):
    k = Matern(input_dim=3, init_variance=0.9, init_lengthscale=0.4, nu=nu)
    with handlers.seed(rng_seed=0):
        K = k(X3, X3)
    expected = _k.matern_kernel(X3, X3, jnp.array(0.9), jnp.full((3,), 0.4), nu)
    assert jnp.allclose(K, expected)


def test_rational_quadratic_ard_matches_primitive():
    k = RationalQuadratic(input_dim=3, init_lengthscale=0.5, init_alpha=2.0)
    with handlers.seed(rng_seed=0):
        K = k(X3, X3)
    expected = _k.rational_quadratic_kernel(
        X3, X3, jnp.array(1.0), jnp.full((3,), 0.5), jnp.array(2.0)
    )
    assert jnp.allclose(K, expected)


def test_ard_lengthscale_prior_produces_event_shaped_site():
    """`set_prior` on an ARD kernel: the ``(D,)`` lengthscale needs
    ``.to_event(1)`` semantics from the caller and must produce a
    correctly-shaped sample site."""
    k = RBF(input_dim=3)
    k.set_prior("lengthscale", dist.LogNormal(0.0, 1.0).expand([3]).to_event(1))
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        K = k(X3, X3)
    site = tr["RBF.lengthscale"]
    assert site["type"] == "sample"
    assert site["value"].shape == (3,)
    assert site["fn"].event_shape == (3,)
    assert K.shape == (4, 4)
    assert jnp.all(jnp.isfinite(K))


def test_ard_kernel_works_with_rff_paths():
    """ARD lengthscales must flow through the RFF/pathwise API: the
    lengthscale applies along the input axis, not to the projection."""
    import jax.random as _jr
    from pyrox_gp._basis import draw_rff_cosine_basis, evaluate_rff_cosine_paths

    kernel = RBF(pyrox_name="RBF_ard_rff", input_dim=3, init_lengthscale=0.7)
    with kernel._get_context():
        variance, lengthscale, omega, phase, weights = draw_rff_cosine_basis(
            kernel,
            _jr.PRNGKey(0),
            n_paths=2,
            n_features=16,
            in_features=3,
            dtype=jnp.float64,
        )
    assert lengthscale.shape == (3,)
    X = _jr.normal(_jr.PRNGKey(1), (5, 3))
    paths = evaluate_rff_cosine_paths(
        X,
        variance=variance,
        lengthscale=lengthscale,
        omega=omega,
        phase=phase,
        weights=weights,
    )
    assert paths.shape[-1] == 5
    assert jnp.all(jnp.isfinite(paths))


def test_spectral_density_accepts_singleton_ard_lengthscale():
    """``input_dim=1`` registers a ``(1,)`` lengthscale, which is a scalar
    in disguise — the 1-D closed forms stay valid, so it must not be
    rejected (FourierInducingFeatures relies on this)."""
    from pyrox_gp._basis import spectral_density

    ard = RBF(pyrox_name="RBF_ard_1d", input_dim=1, init_lengthscale=0.7)
    iso = RBF(pyrox_name="RBF_iso_1d", init_lengthscale=0.7)
    eigvals = jnp.asarray([0.1, 0.5, 1.0])
    with ard._get_context():
        got = spectral_density(ard, eigvals, D=1)
    with iso._get_context():
        expected = spectral_density(iso, eigvals, D=1)
    assert jnp.allclose(got, expected, atol=1e-12)


def test_spectral_density_rejects_ard_kernels():
    """The registered closed forms are isotropic; an ARD lengthscale must
    raise rather than silently pair dimensions with unrelated frequencies."""
    from pyrox_gp._basis import spectral_density

    kernel = RBF(pyrox_name="RBF_ard_sd", input_dim=3)
    with kernel._get_context(), pytest.raises(NotImplementedError, match="isotropic"):
        spectral_density(kernel, jnp.asarray([0.1, 0.5, 1.0]), D=3)


def test_spectral_density_singleton_ard_still_rejected_for_multidim_domain():
    """A ``(1,)`` lengthscale is only a scalar in disguise on a 1-D domain.
    With ``D > 1`` the kernel would reject the inputs, so the density must
    not silently describe a domain it cannot evaluate."""
    from pyrox_gp._basis import spectral_density

    kernel = RBF(pyrox_name="RBF_ard_1d_2ddomain", input_dim=1)
    with kernel._get_context(), pytest.raises(NotImplementedError, match="isotropic"):
        spectral_density(kernel, jnp.asarray([0.1, 0.5]), D=2)
