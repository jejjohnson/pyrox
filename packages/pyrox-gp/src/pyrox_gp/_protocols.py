"""Layer 1 — abstract protocol classes for GP components.

Three orthogonal pyrox-local protocols that compose into a GP model.
`Kernel` has concrete implementations in `pyrox_gp._kernels`
(`pyrox_gp.RBF`, etc.). `SDEKernel` is the state-space
face of stationary 1-D kernels and now lives in `gaussx._ssm`
(``gaussx.MaternSDE``, ``gaussx.SumSDE``, ``gaussx.PeriodicSDE``, ...);
this module re-exports it so the ``pyrox_gp.SDEKernel`` alias still
resolves. Concrete subclasses feed the Kalman-based
`pyrox_gp.MarkovGPPrior`.

Solver strategies intentionally live in `gaussx`, not here. Use
``gaussx.AbstractSolverStrategy`` (combined solve + logdet),
``AbstractSolveStrategy``, or ``AbstractLogdetStrategy`` — with concretes
like ``gaussx.DenseSolver``, ``gaussx.CGSolver``, ``gaussx.BBMMSolver``,
and ``gaussx.ComposedSolver``. The pyrox model entry points
(``GPPrior``, ``gp_factor``, ``gp_sample``) accept any solver strategy.

Gaussian-expectation integrators live in `gaussx` too — use
``gaussx.AbstractIntegrator`` (and its concretes
``GaussHermiteIntegrator``, ``MonteCarloIntegrator``,
``UnscentedIntegrator``, ``TaylorIntegrator``) wherever pyrox needs to
take expectations against a ``GaussianState``.

* `Kernel` — covariance structure, ``(X1, X2) -> Gram``.
* `SDEKernel` — re-exported from `gaussx._ssm`.
* `Guide` — variational posterior structure.
* `Likelihood` — observation model.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from gaussx import SDEKernel as SDEKernel
from jaxtyping import Array, Float


class Kernel(eqx.Module):
    """Abstract base for GP covariance functions.

    Subclasses implement `__call__` returning the Gram matrix on a pair
    of input batches. `gram` and `diag` are convenience defaults
    that derive from `__call__`; structured subclasses (Kronecker,
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


# ``SDEKernel`` lives in `gaussx._ssm` since gaussx 0.0.11; we
# re-export it at the top of this module so ``pyrox_gp.SDEKernel`` keeps
# resolving for downstream callers.


class Guide(eqx.Module):
    """Abstract base for variational posterior families.

    Concrete guides (``DeltaGuide``, ``MeanFieldGuide``, ``LowRankGuide``,
    ``FullRankGuide``, etc.) land in the dedicated guide waves (#28, #29).
    The whitening principle keeps optimization geometry well-conditioned —
    sample from a unit-scale latent and unwhiten with the prior Cholesky.

    Two distinct entry points:

    * `sample` / `log_prob` — pure variational draws and
      densities. ``sample(self, key)`` returns a draw from ``q(f)``;
      ``log_prob(self, f)`` evaluates ``log q(f)``. Neither touches the
      NumPyro trace.
    * ``register(name, prior)`` (optional) — the NumPyro-integration hook
      invoked by `pyrox_gp.gp_sample` when a guide is supplied. Use
      it to register a sample / param site (or compose one out of guide
      state) under ``name`` and return the latent function value. Concrete
      guides that participate in `gp_sample` should implement this;
      the protocol leaves it unspecified so guides usable purely outside
      NumPyro stay valid.
    """

    @abstractmethod
    def sample(self, key: Any) -> Float[Array, " ..."]:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, f: Float[Array, " ..."]) -> Float[Array, ""]:
        raise NotImplementedError


class Likelihood(eqx.Module):
    """Abstract base for observation models.

    Implements the conditional ``p(y | f)``. The advanced inference
    strategies in `pyrox_gp._inference_nongauss` integrate
    ``log p(y | f)`` against a Gaussian cavity via any
    `gaussx.AbstractIntegrator`. Concrete scalar-latent likelihoods
    (`GaussianLikelihood`, `BernoulliLikelihood`,
    `PoissonLikelihood`, `StudentTLikelihood`) and
    multi-latent ones (`SoftmaxLikelihood`,
    `HeteroscedasticGaussianLikelihood`) live in
    `pyrox_gp._likelihoods`.

    Multi-latent likelihoods declare ``latent_dim: int`` as a static
    field (e.g. ``latent_dim = num_classes`` for softmax). Scalar
    likelihoods may omit the field; consumers should read
    ``getattr(lik, "latent_dim", 1)``.
    """

    @abstractmethod
    def log_prob(
        self,
        f: Float[Array, " ..."],
        y: Float[Array, " ..."],
        X: Float[Array, " ..."] | None = None,
    ) -> Float[Array, ""]:
        """Conditional ``p(y | f)``, optionally conditioned on inputs.

        ``X`` is consumed only by likelihoods whose parameters are
        functions of the input (see `pyrox_gp.WarpedGaussianLikelihood`
        with a conditional warp). Scalar observation models ignore it.
        """
        raise NotImplementedError
