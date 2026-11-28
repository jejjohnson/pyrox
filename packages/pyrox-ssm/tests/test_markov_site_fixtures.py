"""The Markov strategies on argminax's site loop and gaussx's site evidence (#553).

``fixtures/markov_site_fixtures.npz`` froze every strategy's fit, with and
without a mask that holds a ``NaN``, while the loop was pyrox-ssm's own
``_solve_sites`` and the evidence its own ``_site_corrected_log_marginal``.
The shared `argminax.solve.site_fixed_point` and
`gaussx.site_evidence_correction` must reproduce them, and must also run on
float32 sites under x64, where the private loop raised.
"""

from __future__ import annotations

import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyrox_ssm import (
    ExpectationPropagationMarkov,
    GaussNewtonMarkovInference,
    LaplaceMarkovInference,
    MarkovGPPrior,
    MaternSDE,
    PosteriorLinearizationMarkov,
)
from ssm_reference import BernoulliLikelihood


_FIXTURES = pathlib.Path(__file__).parent / "fixtures"
_spec = importlib.util.spec_from_file_location(
    "make_markov_site_fixtures", _FIXTURES / "make_markov_site_fixtures.py"
)
assert _spec is not None and _spec.loader is not None
_make = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_make)

_FROZEN = np.load(_FIXTURES / "markov_site_fixtures.npz")

# CI runners and this machine differ by about 3e-10 in these fits, so 1e-9
# rather than the 1e-10 of #553; anything the refactor changed would be far
# larger (a different stopping test moves n_iter, and the sites by ~1e-7).
_RTOL = 1e-9
_ATOL = 1e-12

_CASES = [
    (strategy, likelihood, masked)
    for strategy, likelihood in _make.CASES
    for masked in (False, True)
]


def _ids(case) -> str:
    return _make.case_key(*case)


# one case per strategy, masked and not, in the fast tier
_FAST = [_CASES[i] for i in (1, 6, 9, 14)]


@pytest.mark.parametrize("case", _FAST, ids=_ids)
def test_fit_matches_the_frozen_private_loop(case) -> None:
    """Values, ``n_iter`` and ``converged`` of the private loop, exactly."""
    key = _make.case_key(*case)
    cond = _make.fit(*case, 0.7)
    for field in _make.FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(cond, field)),
            _FROZEN[f"{key}__{field}"],
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=f"{key}: {field}",
        )
    assert int(cond.n_iter) == int(_FROZEN[f"{key}__n_iter"])
    assert bool(cond.converged) == bool(_FROZEN[f"{key}__converged"])


@pytest.mark.slow
@pytest.mark.parametrize("case", _CASES, ids=_ids)
def test_fit_and_gradient_match_the_frozen_private_loop(case) -> None:
    """Every case, and the gradient of the evidence in the lengthscale."""
    key = _make.case_key(*case)
    cond = _make.fit(*case, 0.7)
    for field in _make.FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(cond, field)),
            _FROZEN[f"{key}__{field}"],
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=f"{key}: {field}",
        )
    assert int(cond.n_iter) == int(_FROZEN[f"{key}__n_iter"])
    assert bool(cond.converged) == bool(_FROZEN[f"{key}__converged"])
    grad = jax.grad(lambda ell: _make.fit(*case, ell).log_marginal_approx)(0.7)
    # the implicit adjoint's GMRES runs at 1e-10, so the gradient keeps ~1e-9
    np.testing.assert_allclose(
        np.asarray(grad), _FROZEN[f"{key}__grad_lengthscale"], rtol=1e-8
    )


_STRATEGIES = [
    LaplaceMarkovInference,
    GaussNewtonMarkovInference,
    PosteriorLinearizationMarkov,
    ExpectationPropagationMarkov,
]


@pytest.mark.parametrize("strategy", _STRATEGIES, ids=lambda s: s.__name__)
def test_float32_sites_fit_under_x64(strategy) -> None:
    """Float32 data under ``jax_enable_x64`` fit and stay float32.

    Before #553 every strategy raised ``ValueError: body_fun must have the
    same input and output structure``: optimistix started the loop's relative
    error as a float64 and the float32 step returned a float32 one.
    """
    times = jnp.linspace(0.0, 5.0, 16, dtype=jnp.float32)
    y = (jnp.sin(2.0 * times) > 0.0).astype(jnp.float32)
    mask = jnp.ones(16, dtype=bool).at[3].set(False)
    y = y.at[3].set(jnp.nan)
    sde = MaternSDE(jnp.float32(1.0), jnp.float32(0.7), order=1)
    cond = strategy().fit(
        MarkovGPPrior(sde, times), BernoulliLikelihood(), y, mask=mask
    )
    assert cond.q_mean.dtype == jnp.float32
    assert cond.site_nat1.dtype == jnp.float32
    assert bool(cond.converged)
    assert bool(jnp.all(jnp.isfinite(cond.q_mean)))
    assert bool(jnp.isfinite(cond.log_marginal_approx))
    # the float64 fit of the same problem, to float32 accuracy
    sde64 = MaternSDE(1.0, 0.7, order=1)
    ref = strategy().fit(
        MarkovGPPrior(sde64, times.astype(jnp.float64)),
        BernoulliLikelihood(),
        y.astype(jnp.float64),
        mask=mask,
    )
    np.testing.assert_allclose(cond.q_mean, ref.q_mean, rtol=1e-3, atol=1e-4)


def test_one_iteration_budget_runs_only_the_first_update() -> None:
    """``max_iter=1`` is the prior-cavity update alone, as before."""
    times = jnp.linspace(0.0, 5.0, 16)
    y = (jnp.sin(2.0 * times) > 0.0).astype(jnp.float64)
    prior = MarkovGPPrior(MaternSDE(1.0, 0.7, order=1), times)
    cond = PosteriorLinearizationMarkov(max_iter=1).fit(prior, BernoulliLikelihood(), y)
    assert int(cond.n_iter) == 1
    assert not bool(cond.converged)
