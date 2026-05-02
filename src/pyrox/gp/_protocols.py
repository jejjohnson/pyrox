"""Layer 1 — abstract protocol classes for GP components.

Four orthogonal pyrox-local protocols that compose into a GP model.
:class:`Kernel` has concrete implementations in :mod:`pyrox.gp._kernels`
(:class:`pyrox.gp.RBF`, etc.). :class:`SDEKernel` is the state-space
face of stationary 1-D kernels — concrete implementations
(:class:`pyrox.gp.MaternSDE`, :class:`pyrox.gp.SumSDE`,
:class:`pyrox.gp.PeriodicSDE`, ...) feed the Kalman-based
:class:`pyrox.gp.MarkovGPPrior`.

Solver strategies intentionally live in :mod:`gaussx`, not here. Use
``gaussx.AbstractSolverStrategy`` (combined solve + logdet),
``AbstractSolveStrategy``, or ``AbstractLogdetStrategy`` — with concretes
like ``gaussx.DenseSolver``, ``gaussx.CGSolver``, ``gaussx.BBMMSolver``,
and ``gaussx.ComposedSolver``. The pyrox model entry points
(``GPPrior``, ``gp_factor``, ``gp_sample``) accept any solver strategy.

Gaussian-expectation integrators live in :mod:`gaussx` too — use
``gaussx.AbstractIntegrator`` (and its concretes
``GaussHermiteIntegrator``, ``MonteCarloIntegrator``,
``UnscentedIntegrator``, ``TaylorIntegrator``) wherever pyrox needs to
take expectations against a ``GaussianState``.

* :class:`Kernel` — covariance structure, ``(X1, X2) -> Gram``.
* :class:`SDEKernel` — state-space representation of a stationary 1-D
  kernel for linear-time temporal GP inference.
* :class:`Guide` — variational posterior structure.
* :class:`Likelihood` — observation model.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
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


class SDEKernel(eqx.Module):
    r"""Abstract base for kernels with state-space (SDE) representations.

    Stationary kernels with rational spectral densities admit exact
    finite-dimensional state-space representations of the form

    .. math::
        d\mathbf{x}(t) = F\,\mathbf{x}(t)\, dt + L\, dw(t),
        \qquad f(t) = H\,\mathbf{x}(t)

    where :math:`w(t)` is white noise with spectral density :math:`Q_c`
    and :math:`P_\infty` is the stationary state covariance solving the
    Lyapunov equation :math:`F P_\infty + P_\infty F^\top + L Q_c L^\top = 0`.

    Discretisation at time step :math:`\Delta t` gives

    .. math::
        A_k = \exp(F\,\Delta t),
        \qquad Q_k = P_\infty - A_k\,P_\infty\,A_k^\top,

    so that :math:`x_{k+1} = A_k x_k + q_k` with :math:`q_k \sim \mathcal{N}(0, Q_k)`.

    Concrete subclasses implement :meth:`sde_params` returning the
    closed-form ``(F, L, H, Q_c, P_inf)`` tuple. :meth:`discretise`
    defaults to a generic ``expm``-based implementation; subclasses with
    closed-form transitions (e.g. Matern-1/2) may override it.

    The continuous-time autocovariance recovered from the SDE is
    :math:`k(\tau) = H\,\exp(F|\tau|)\,P_\infty\,H^\top` for stationary
    kernels.
    """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """State dimension :math:`d` of the SDE representation."""
        raise NotImplementedError

    @abstractmethod
    def sde_params(
        self,
    ) -> tuple[
        Float[Array, "d d"],
        Float[Array, "d s"],
        Float[Array, "1 d"],
        Float[Array, "s s"],
        Float[Array, "d d"],
    ]:
        """Return ``(F, L, H, Q_c, P_inf)`` defining the continuous SDE."""
        raise NotImplementedError

    def discretise(
        self,
        dt: Float[Array, " N"],
    ) -> tuple[Float[Array, "N d d"], Float[Array, "N d d"]]:
        r"""Discretise the SDE at time steps ``dt``.

        Default implementation evaluates ``A_k = expm(F dt_k)`` via
        ``jax.scipy.linalg.expm`` and ``Q_k = P_\infty - A_k P_\infty A_k^\top``.
        Subclasses with closed-form transitions should override.

        Args:
            dt: ``(N,)`` array of (non-negative) time steps.

        Returns:
            Tuple ``(A, Q)`` of ``(N, d, d)`` arrays.
        """
        F, _L, _H, _Q_c, P_inf = self.sde_params()

        def _step(
            dt_n: Float[Array, ""],
        ) -> tuple[Float[Array, "d d"], Float[Array, "d d"]]:
            A = jsl.expm(F * dt_n)
            Q = P_inf - A @ P_inf @ A.T
            # Symmetrise to absorb roundoff: ``Q`` is theoretically symmetric,
            # but float arithmetic can leave asymmetric perturbations that
            # break downstream Cholesky factorisation in Kalman steps.
            Q = 0.5 * (Q + Q.T)
            return A, Q

        return jax.vmap(_step)(jnp.asarray(dt))


class Guide(eqx.Module):
    """Abstract base for variational posterior families.

    Concrete guides (``DeltaGuide``, ``MeanFieldGuide``, ``LowRankGuide``,
    ``FullRankGuide``, etc.) land in the dedicated guide waves (#28, #29).
    The whitening principle keeps optimization geometry well-conditioned —
    sample from a unit-scale latent and unwhiten with the prior Cholesky.

    Two distinct entry points:

    * :meth:`sample` / :meth:`log_prob` — pure variational draws and
      densities. ``sample(self, key)`` returns a draw from ``q(f)``;
      ``log_prob(self, f)`` evaluates ``log q(f)``. Neither touches the
      NumPyro trace.
    * ``register(name, prior)`` (optional) — the NumPyro-integration hook
      invoked by :func:`pyrox.gp.gp_sample` when a guide is supplied. Use
      it to register a sample / param site (or compose one out of guide
      state) under ``name`` and return the latent function value. Concrete
      guides that participate in :func:`gp_sample` should implement this;
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
    strategies in :mod:`pyrox.gp._inference_nongauss` integrate
    ``log p(y | f)`` against a Gaussian cavity via any
    :class:`gaussx.AbstractIntegrator`. Concrete scalar-latent likelihoods
    (:class:`GaussianLikelihood`, :class:`BernoulliLikelihood`,
    :class:`PoissonLikelihood`, :class:`StudentTLikelihood`) and
    multi-latent ones (:class:`SoftmaxLikelihood`,
    :class:`HeteroscedasticGaussianLikelihood`) live in
    :mod:`pyrox.gp._likelihoods`.

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
    ) -> Float[Array, ""]:
        raise NotImplementedError
