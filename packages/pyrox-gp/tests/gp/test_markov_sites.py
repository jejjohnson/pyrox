"""The Markov track's per-site numerics on gaussx, and its import boundary.

The four Markov modules are moving to pyrox-ssm, which may import only the
pyrox core and gaussx. Here: an AST scan that they (and `_markov_sites`, which
moves with them) import nothing else from pyrox-gp, and the equivalence of
the gaussx adapter with the pyrox-gp helpers it replaced. Every equivalence is
the same arithmetic in a different order, so the tolerances are float64
round-off.
"""

from __future__ import annotations

import ast
from pathlib import Path

import gaussx
import jax
import jax.numpy as jnp
import lineax as lx
import pyrox.likelihoods
import pyrox_gp
import pytest
from pyrox_gp import (
    BernoulliLikelihood,
    GaussianLikelihood,
    HeteroscedasticGaussianLikelihood,
    PoissonLikelihood,
    SoftmaxLikelihood,
    StudentTLikelihood,
)
from pyrox_gp._inference import _ell_numerical, _per_site_expectation
from pyrox_gp._inference_nongauss import (
    _log_prob_per_point_factory,
    _per_point_grad_hess,
)
from pyrox_gp._markov_sites import (
    _cavity_site_derivatives,
    _pushforward_moments,
    as_gaussx_likelihood,
)
from pyrox_gp._protocols import Likelihood
from pyrox_gp._warped import _pushforward_moments as _warped_pushforward_moments


SRC = Path(pyrox_gp.__file__).parent
# The modules that move to pyrox-ssm, and the helpers that move with them.
MARKOV_MODULES = (
    "_markov",
    "_sparse_markov",
    "_inference_nongauss_markov",
    "_markov_flow",
    "_markov_sites",
)


def _pyrox_gp_imports(path: Path) -> set[str]:
    """Every ``pyrox_gp`` module the file imports, at any depth (lazy included)."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.startswith("pyrox_gp"))
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.level:
                found.add(f"pyrox_gp.{node.module}")
            elif node.module.split(".")[0] == "pyrox_gp":
                found.add(node.module)
    return found


@pytest.mark.parametrize("module", MARKOV_MODULES)
def test_markov_module_imports_only_the_markov_track(module) -> None:
    allowed = {f"pyrox_gp.{name}" for name in MARKOV_MODULES}
    imported = _pyrox_gp_imports(SRC / f"{module}.py")
    assert imported <= allowed, sorted(imported - allowed)


def test_the_import_scan_sees_a_borrowed_helper(tmp_path) -> None:
    source = tmp_path / "borrowed.py"
    source.write_text(
        "def f():\n    from pyrox_gp._inference import _ell_numerical\n"
        "from pyrox_gp import RBF\n"
    )
    assert _pyrox_gp_imports(source) == {"pyrox_gp._inference", "pyrox_gp"}


# --- the moved names --------------------------------------------------------


@pytest.mark.parametrize("name", ["Likelihood", "Guide", "GaussianLikelihood"])
def test_pyrox_gp_reexports_the_core_object(name) -> None:
    assert getattr(pyrox_gp, name) is getattr(pyrox.likelihoods, name)


# --- the adapter against the helpers it replaces ------------------------------


SCALAR_LIKELIHOODS = {
    "bernoulli": (BernoulliLikelihood(), jnp.array([1.0, 0.0, 1.0, 1.0, 0.0])),
    "poisson": (PoissonLikelihood(), jnp.array([0.0, 2.0, 1.0, 5.0, 3.0])),
    "student_t": (
        StudentTLikelihood(df=3.0, scale=0.5),
        jnp.array([0.3, -1.2, 2.0, 0.0, -0.4]),
    ),
    "gaussian": (
        GaussianLikelihood(noise_var=0.3),
        jnp.array([0.3, -1.2, 2.0, 0.0, -0.4]),
    ),
}
F = jnp.array([-0.8, 0.1, 0.6, 1.3, -0.2])
VAR = jnp.array([0.2, 0.5, 1.0, 1.5, 0.05])


@pytest.mark.parametrize("name", sorted(SCALAR_LIKELIHOODS))
def test_site_derivatives_match_per_point_grad_hess(name) -> None:
    lik, y = SCALAR_LIKELIHOODS[name]
    g_ref, h_ref = _per_point_grad_hess(_log_prob_per_point_factory(lik), F, y)
    g, h = as_gaussx_likelihood(lik, y).site_derivatives(F)
    assert jnp.allclose(g, g_ref, rtol=1e-6, atol=1e-12)
    assert jnp.allclose(h, h_ref, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("name", sorted(SCALAR_LIKELIHOODS))
def test_log_prob_matches_the_summed_likelihood(name) -> None:
    lik, y = SCALAR_LIKELIHOODS[name]
    got = as_gaussx_likelihood(lik, y).log_prob(F)
    assert jnp.allclose(got, lik.log_prob(F, y), rtol=1e-12)


@pytest.mark.parametrize("name", sorted(SCALAR_LIKELIHOODS))
def test_expected_log_likelihood_matches_ell_numerical(name) -> None:
    lik, y = SCALAR_LIKELIHOODS[name]
    rule = gaussx.GaussHermiteIntegrator(order=20)
    state = gaussx.GaussianState(F, lx.DiagonalLinearOperator(VAR))
    got = gaussx.expected_log_likelihood(as_gaussx_likelihood(lik, y), state, rule)
    assert jnp.allclose(got, _ell_numerical(lik, y, F, VAR, rule), rtol=1e-6)


class _InputScaledGaussian(Likelihood):
    """A Gaussian whose noise variance is the input's squared norm."""

    def log_prob(self, f, y, X=None):
        var = jnp.sum(X**2) if X is not None else 1.0
        return jnp.sum(-0.5 * (y - f) ** 2 / var - 0.5 * jnp.log(2 * jnp.pi * var))


class _LegacyGaussian(Likelihood):
    """A likelihood written against the pre-gh-204 ``log_prob(f, y)``."""

    def log_prob(self, f, y):
        return jnp.sum(-0.5 * (y - f) ** 2 - 0.5 * jnp.log(2 * jnp.pi))


@pytest.mark.parametrize("lik", [_InputScaledGaussian(), _LegacyGaussian()])
def test_inputs_reach_only_a_likelihood_that_takes_them(lik) -> None:
    y = jnp.array([0.3, -1.2, 2.0, 0.0, -0.4])
    X = jnp.linspace(0.5, 2.0, 10).reshape(5, 2)
    rule = gaussx.GaussHermiteIntegrator(order=20)
    state = gaussx.GaussianState(F, lx.DiagonalLinearOperator(VAR))
    got = gaussx.expected_log_likelihood(as_gaussx_likelihood(lik, y, X), state, rule)
    assert jnp.allclose(got, _ell_numerical(lik, y, F, VAR, rule, X=X), rtol=1e-6)


@pytest.mark.parametrize(
    "lik", [SoftmaxLikelihood(num_classes=3), HeteroscedasticGaussianLikelihood()]
)
def test_adapter_rejects_a_multi_latent_likelihood(lik) -> None:
    with pytest.raises(ValueError, match="latent_dim"):
        as_gaussx_likelihood(lik, jnp.zeros(4))


@pytest.mark.parametrize(
    "rule",
    [gaussx.GaussHermiteIntegrator(order=20), gaussx.UnscentedIntegrator(alpha=1.0)],
    ids=["hermite", "unscented"],
)
@pytest.mark.parametrize("name", ["bernoulli", "poisson", "student_t"])
def test_cavity_site_derivatives_match_the_per_site_expectations(name, rule) -> None:
    """Bonnet and Price on the ELL equal integrating the site derivatives directly."""
    lik, y = SCALAR_LIKELIHOODS[name]
    per_point = _log_prob_per_point_factory(lik)

    def lp(f_n, y_n):
        return per_point(f_n[None], y_n[None])[0]

    grad_lp = jax.grad(lp)
    g_ref = _per_site_expectation(rule, grad_lp, F, VAR, y)
    h_ref = _per_site_expectation(rule, jax.grad(grad_lp), F, VAR, y)
    g, h = _cavity_site_derivatives(as_gaussx_likelihood(lik, y), F, VAR, rule)
    assert jnp.allclose(g, g_ref, rtol=1e-6, atol=1e-12)
    assert jnp.allclose(h, h_ref, rtol=1e-6, atol=1e-12)


def test_pushforward_moments_copy_matches_the_warped_original() -> None:
    mean = jnp.array([[-0.5, 0.2], [1.0, 3.0]])
    var = jnp.array([[0.3, 1.0], [0.01, 2.0]])
    got = _pushforward_moments(jnp.exp, mean, var, order=30)
    expected = _warped_pushforward_moments(jnp.exp, mean, var, order=30)
    assert jnp.allclose(got[0], expected[0], rtol=1e-12)
    assert jnp.allclose(got[1], expected[1], rtol=1e-12)
