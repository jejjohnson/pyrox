"""Observation models of an LGM (P8).

Each wraps a gaussx likelihood and declares its own hyperparameters, which
join $\\theta$ under the names ``f"{name}.{k}"`` (``"lik.prec"`` for the
Gaussian noise precision). Exposures and offsets go in ``data["offset"]``
(the log exposure for a Poisson model), trial counts in
``data["n_trials"]``.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping

import equinox as eqx
import gaussx as gx
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array
from numpyro.distributions.transforms import Transform

from pyrox_lgm._components._base import default_transform
from pyrox_lgm._priors import PCPrecision


class AbstractObservation(eqx.Module):
    """``y | eta`` for an LGM, with its own hyperparameters."""

    name: eqx.AbstractVar[str]

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        """Hyperparameters of the likelihood; none by default."""
        return {}

    @abc.abstractmethod
    def build(
        self, y: Array, theta: dict[str, Array], data: Mapping[str, Array]
    ) -> gx.AbstractLikelihood:
        """The gaussx likelihood holding ``y`` at hyperparameters ``theta``."""


class Gaussian(AbstractObservation):
    """``y ~ N(eta, 1/prec)``; ``prec`` under ``PCPrecision(1, 0.01)``."""

    name: str = eqx.field(static=True, default="lik")
    prec_prior: dist.Distribution = eqx.field(
        default_factory=lambda: PCPrecision(1.0, 0.01)
    )

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        return {"prec": (self.prec_prior, default_transform(self.prec_prior))}

    def build(self, y, theta, data):
        return gx.GaussianLikelihood(y, 1.0 / theta["prec"])


class Poisson(AbstractObservation):
    """``y ~ Poisson(exp(eta))``; put the log exposure in ``data["offset"]``."""

    name: str = eqx.field(static=True, default="lik")

    def build(self, y, theta, data):
        return gx.PoissonLikelihood(y)


class Bernoulli(AbstractObservation):
    """``y ~ Bernoulli(sigmoid(eta))``, ``y`` in ``{0, 1}``."""

    name: str = eqx.field(static=True, default="lik")

    def build(self, y, theta, data):
        return gx.BernoulliLikelihood(y)


class Binomial(AbstractObservation):
    """``y ~ Binomial(n, sigmoid(eta))`` with ``n = data["n_trials"]``."""

    name: str = eqx.field(static=True, default="lik")

    def build(self, y, theta, data):
        return gx.BinomialLikelihood(y, jnp.asarray(data["n_trials"]))


class NegativeBinomial(AbstractObservation):
    """``y ~ NegBin(mean exp(eta), size)``; ``size`` under ``Gamma(1, 0.1)``."""

    name: str = eqx.field(static=True, default="lik")
    size_prior: dist.Distribution = eqx.field(
        default_factory=lambda: dist.Gamma(1.0, 0.1)
    )

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        return {"size": (self.size_prior, default_transform(self.size_prior))}

    def build(self, y, theta, data):
        return gx.NegativeBinomialLikelihood(y, theta["size"])
