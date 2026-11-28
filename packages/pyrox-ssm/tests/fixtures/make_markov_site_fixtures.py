"""Freeze the fits of the four Markov strategies, for the site-loop refactor (#553).

Run from the repo root, on the commit whose behaviour the fixtures pin:

    uv run python packages/pyrox-ssm/tests/fixtures/make_markov_site_fixtures.py

It writes ``markov_site_fixtures.npz`` beside this file: for every case in
``CASES`` (strategy, likelihood, with or without a mask that holds a ``NaN``),
the fitted site naturals, the posterior marginals, ``log_marginal_approx``,
``n_iter``, ``converged``, and the gradient of ``log_marginal_approx`` in the
prior's lengthscale. ``test_markov_site_fixtures.py`` reads the same ``CASES``,
so the two cannot drift apart. The fixtures were made at pyrox ``94ce05f``,
whose pyrox-ssm sources match ``7d81421``, the commit #553 names.
"""

from __future__ import annotations

import pathlib
import sys

import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from pyrox_ssm import (  # noqa: E402
    ExpectationPropagationMarkov,
    GaussNewtonMarkovInference,
    LaplaceMarkovInference,
    MarkovGPPrior,
    MaternSDE,
    PosteriorLinearizationMarkov,
)


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ssm_reference import (  # noqa: E402
    BernoulliLikelihood,
    PoissonLikelihood,
    StudentTLikelihood,
)


FIXTURE = pathlib.Path(__file__).with_name("markov_site_fixtures.npz")

STRATEGIES = {
    "laplace": LaplaceMarkovInference,
    "gauss_newton": GaussNewtonMarkovInference,
    "pl": PosteriorLinearizationMarkov,
    "ep": ExpectationPropagationMarkov,
}

# (strategy, likelihood) pairs; each runs with and without a mask
CASES = [
    ("laplace", "bernoulli"),
    ("laplace", "poisson"),
    ("gauss_newton", "bernoulli"),
    ("gauss_newton", "student_t"),
    ("pl", "bernoulli"),
    ("pl", "poisson"),
    ("ep", "bernoulli"),
    ("ep", "student_t"),
]

FIELDS = ("site_nat1", "site_nat2", "q_mean", "q_var", "log_marginal_approx")


def problem(likelihood: str, masked: bool):
    """Times, targets, mask and likelihood of one case (16 times, x64)."""
    times = jnp.linspace(0.0, 5.0, 16)
    signal = 1.2 * jnp.sin(1.7 * times)
    if likelihood == "bernoulli":
        y, lik = (signal > 0.0).astype(jnp.float64), BernoulliLikelihood()
    elif likelihood == "poisson":
        y, lik = jnp.round(jnp.exp(0.5 * signal + 0.5)), PoissonLikelihood()
    else:
        y = signal + 0.3 * jnp.cos(5.0 * times)
        y = y.at[4].set(4.0)  # one outlier, where the Student-t matters
        lik = StudentTLikelihood(df=3.0, scale=0.4)
    mask = None
    if masked:
        mask = jnp.ones(16, dtype=bool).at[jnp.array([3, 9])].set(False)
        y = y.at[3].set(jnp.nan)
    return times, y, mask, lik


def fit(strategy: str, likelihood: str, masked: bool, lengthscale):
    """The fit of one case at ``lengthscale``."""
    times, y, mask, lik = problem(likelihood, masked)
    prior = MarkovGPPrior(MaternSDE(1.0, lengthscale, order=1), times)
    return STRATEGIES[strategy]().fit(prior, lik, y, mask=mask)


def case_key(strategy: str, likelihood: str, masked: bool) -> str:
    """The fixture-key prefix of one case."""
    return f"{strategy}__{likelihood}__{'masked' if masked else 'full'}"


def main() -> None:
    """Fit every case and write the fixture file."""
    arrays: dict[str, np.ndarray] = {}
    for strategy, likelihood in CASES:
        for masked in (False, True):
            key = case_key(strategy, likelihood, masked)
            cond = fit(strategy, likelihood, masked, 0.7)
            for field in FIELDS:
                arrays[f"{key}__{field}"] = np.asarray(getattr(cond, field))
            arrays[f"{key}__n_iter"] = np.asarray(cond.n_iter)
            arrays[f"{key}__converged"] = np.asarray(cond.converged)
            grad = jax.grad(
                lambda ell, s=strategy, li=likelihood, m=masked: fit(
                    s, li, m, ell
                ).log_marginal_approx
            )(0.7)
            arrays[f"{key}__grad_lengthscale"] = np.asarray(grad)
    np.savez(FIXTURE, **arrays)


if __name__ == "__main__":
    main()
