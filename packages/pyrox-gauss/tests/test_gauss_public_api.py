"""pyrox_gauss's public surface: ``__all__`` is the contract."""

from __future__ import annotations

import numpyro.distributions as dist
import pyrox_gauss


def test_version():
    assert isinstance(pyrox_gauss.__version__, str)
    assert pyrox_gauss.__version__


def test_all_names_exist():
    for name in pyrox_gauss.__all__:
        assert hasattr(pyrox_gauss, name), name


def test_all_has_no_duplicates():
    assert len(pyrox_gauss.__all__) == len(set(pyrox_gauss.__all__))


def test_every_class_is_a_numpyro_distribution():
    for name in pyrox_gauss.__all__:
        if name == "LGSSMFactory":  # a callable mask -> MaskedLGSSM
            continue
        assert issubclass(getattr(pyrox_gauss, name), dist.Distribution), name


def test_kl_registration_is_active_on_import():
    import jax.numpy as jnp
    import lineax as lx

    cov = lx.MatrixLinearOperator(jnp.eye(2), lx.positive_semidefinite_tag)
    p = pyrox_gauss.MultivariateNormal(jnp.zeros(2), cov)
    q = pyrox_gauss.MultivariateNormal(jnp.ones(2), cov)
    # KL[p ‖ q] = ½ ‖μ_p − μ_q‖² for equal identity covariances
    assert jnp.allclose(dist.kl_divergence(p, q), 1.0)
