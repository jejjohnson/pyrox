"""Freeze the eager fits of PosteriorLinearization and EP, for the site loop (#552).

Run from the repo root, on the commit whose behaviour the fixtures pin:

    uv run python packages/pyrox-gp/tests/gp/data/make_eager_cavity_fixtures.py

It writes ``eager_cavity_fits.npz`` beside this file: for every case in
``CASES`` (strategy and likelihood, at the strategy's defaults), the site
naturals, the posterior marginals, ``log_marginal_approx``, ``n_iter`` and
``converged`` of the eager Python loops those two strategies ran before #552.
``test_inference_nongauss_sites.py`` reads the same ``CASES``. The fixtures
were made at pyrox ``062804c``, whose ``_inference_nongauss.py`` matches
``7d81421``, the commit #552 names, in both classes.
"""

from __future__ import annotations

import pathlib

import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pyrox_gp


FIXTURE = pathlib.Path(__file__).with_name("eager_cavity_fits.npz")

CASES = [
    ("pl", "bernoulli"),
    ("pl", "poisson"),
    ("pl", "student_t"),
    ("ep", "bernoulli"),
    ("ep", "poisson"),
    ("ep", "student_t"),
]

FIELDS = ("site_nat1", "site_nat2", "q_mean", "q_var", "log_marginal_approx")


def strategy_class(name: str) -> type:
    """The strategy class of a case."""
    if name == "pl":
        return pyrox_gp.PosteriorLinearization
    return pyrox_gp.ExpectationPropagation


def problem(likelihood: str, lengthscale=1.0):
    """The prior, likelihood and targets of one case (16 points, x64)."""
    X = jnp.linspace(-3.0, 3.0, 16)[:, None]
    signal = 1.3 * jnp.sin(1.2 * X[:, 0])
    if likelihood == "bernoulli":
        y, lik = (signal > 0.0).astype(jnp.float64), pyrox_gp.BernoulliLikelihood()
    elif likelihood == "poisson":
        y, lik = jnp.round(jnp.exp(0.5 * signal + 0.5)), pyrox_gp.PoissonLikelihood()
    else:
        # one outlier, where the Student-t matters
        y = (signal + 0.2 * jnp.cos(4.0 * X[:, 0])).at[5].set(4.0)
        lik = pyrox_gp.StudentTLikelihood(df=3.0, scale=0.4)
    kernel = pyrox_gp.RBF(init_lengthscale=lengthscale, init_variance=1.0)
    return pyrox_gp.GPPrior(kernel=kernel, X=X), lik, y


def main() -> None:
    """Fit every case and write the fixture file."""
    arrays: dict[str, np.ndarray] = {}
    for strategy, likelihood in CASES:
        prior, lik, y = problem(likelihood)
        cond = strategy_class(strategy)().fit(prior, lik, y)
        key = f"{strategy}__{likelihood}"
        for field in FIELDS:
            arrays[f"{key}__{field}"] = np.asarray(getattr(cond, field))
        arrays[f"{key}__n_iter"] = np.asarray(cond.n_iter)
        arrays[f"{key}__converged"] = np.asarray(cond.converged)
    np.savez(FIXTURE, **arrays)


if __name__ == "__main__":
    main()
