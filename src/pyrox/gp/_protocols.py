"""Layer 1 — abstract protocol classes for GP components.

Four orthogonal pyrox-local protocols that compose into a GP model.
Wave 2 ships the abstract definitions only for :class:`Guide`,
:class:`Integrator`, and :class:`Likelihood`; concrete implementations
land in later waves. :class:`Kernel` already has concrete implementations
in this wave (:class:`pyrox.gp.RBF`, etc.).

Solver strategies intentionally live in :mod:`gaussx`, not here. Use
``gaussx.AbstractSolverStrategy`` (combined solve + logdet),
``AbstractSolveStrategy``, or ``AbstractLogdetStrategy`` — with concretes
like ``gaussx.DenseSolver``, ``gaussx.CGSolver``, ``gaussx.BBMMSolver``,
and ``gaussx.ComposedSolver``. The pyrox model entry points
(``GPPrior``, ``gp_factor``, ``gp_sample``) accept any solver strategy.

* :class:`Kernel` — covariance structure, ``(X1, X2) -> Gram``.
* :class:`Guide` — variational posterior structure.
* :class:`Integrator` — expectations under a Gaussian.
* :class:`Likelihood` — observation model.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class Kernel(eqx.Module):
    """Abstract base for GP covariance functions.

    Subclasses implement :meth:`__call__` returning the Gram matrix on a pair
    of input batches. :meth:`gram` and :meth:`diag` are convenience defaults
    that derive from :meth:`__call__`; structured subclasses (Kronecker,
    state-space, etc.) should override them for efficiency.
    """

    @abstractmethod
    def __call__(
        self,
        X1: Float[Array, "N1 D"],
        X2: Float[Array, "N2 D"],
    ) -> Float[Array, "N1 N2"]:
        raise NotImplementedError

    def gram(self, X: Float[Array, "N D"]) -> Float[Array, "N N"]:
        """Symmetric Gram matrix ``K(X, X)``."""
        return self(X, X)

    def diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        """Diagonal of ``K(X, X)``.

        Default implementation extracts the diagonal of the full Gram. For
        stationary kernels with constant diagonal, override with a vectorized
        broadcast for the ``O(N)`` shortcut.
        """
        return jnp.diag(self(X, X))


class Guide(eqx.Module):
    """Abstract base for variational posterior families.

    Concrete guides (``DeltaGuide``, ``MeanFieldGuide``, ``LowRankGuide``,
    ``FullRankGuide``, etc.) land in the dedicated guide waves (#28, #29).
    The whitening principle keeps optimization geometry well-conditioned —
    sample from a unit-scale latent and unwhiten with the prior Cholesky.
    """

    @abstractmethod
    def sample(self, key: Any) -> Float[Array, " ..."]:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, f: Float[Array, " ..."]) -> Float[Array, ""]:
        raise NotImplementedError


class Integrator(eqx.Module):
    """Abstract base for Gaussian-expectation integrators.

    Computes :math:`\\mathbb{E}_{q(f)}[g(f)]` where ``q(f) = N(mean, var)``.
    Concrete integrators (Gauss-Hermite, sigma-points, cubature, Taylor,
    Monte Carlo) land in later waves and may delegate to ``gaussx``'s
    quadrature primitives.
    """

    @abstractmethod
    def integrate(
        self,
        fn: Callable[[Float[Array, " ..."]], Float[Array, " ..."]],
        mean: Float[Array, " ..."],
        var: Float[Array, " ..."],
    ) -> Float[Array, " ..."]:
        raise NotImplementedError


class Likelihood(eqx.Module):
    """Abstract base for observation models.

    Implements the conditional ``p(y | f)`` and a default
    :meth:`expected_log_prob` that integrates over a Gaussian latent via an
    :class:`Integrator`. Concrete likelihoods (Gaussian, Bernoulli, Poisson,
    StudentT, ...) land in later waves.
    """

    @abstractmethod
    def log_prob(
        self,
        f: Float[Array, " ..."],
        y: Float[Array, " ..."],
    ) -> Float[Array, ""]:
        raise NotImplementedError
