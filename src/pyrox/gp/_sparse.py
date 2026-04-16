"""Sparse GP prior — inducing-input variational foundation.

The :class:`SparseGPPrior` wraps a kernel together with a fixed set of
*inducing inputs* ``Z`` of shape ``(M, D)``. It exposes the inducing
covariance ``K_zz``, cross covariances ``K_xz``, and the prior diagonal
``K_xx`` as building blocks. A sparse variational guide
(:class:`pyrox.gp.FullRankGuide`, :class:`MeanFieldGuide`, or
:class:`WhitenedGuide`) consumes these matrices to produce a predictive
distribution at any test set.

This wave intentionally stops at building blocks: the sparse ELBO entry
point — analogous to :func:`gp_factor` for the collapsed Gaussian path
— lands in Wave 3's variational inference issue. Until then the user
assembles the ELBO in their NumPyro model from
:meth:`Guide.kl_divergence` and the predictive moments returned by
:meth:`Guide.predict`.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from gaussx import AbstractSolverStrategy, DenseSolver
from jaxtyping import Array, Float

from pyrox.gp._protocols import Kernel


def _psd_operator(K: Float[Array, "M M"]) -> lx.AbstractLinearOperator:
    """Wrap a Gram matrix as a PSD ``lineax`` operator."""
    return lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)  # ty: ignore[invalid-return-type]


class SparseGPPrior(eqx.Module):
    r"""GP prior parameterized over inducing inputs ``Z``.

    Represents the prior over inducing values ``u = f(Z)``,

    .. math::

        p(u) = \mathcal{N}(\mu(Z),\, K_{ZZ} + \mathrm{jitter}\,I).

    Pair with a sparse variational guide that owns ``q(u) = N(m, S)`` to
    obtain the standard SVGP predictive

    .. math::

        \mu_*(x) = K_{xZ} K_{ZZ}^{-1} m, \qquad
        \sigma_*^2(x) = k(x, x) - K_{xZ} K_{ZZ}^{-1} K_{Zx}
                       + K_{xZ} K_{ZZ}^{-1} S K_{ZZ}^{-1} K_{Zx}.

    Attributes:
        kernel: Any :class:`pyrox.gp.Kernel` — evaluated on ``Z``.
        Z: Inducing inputs of shape ``(M, D)``.
        mean_fn: Callable ``X -> (N,)`` or ``None`` for the zero mean.
            The mean is evaluated at *test* inputs (or training inputs)
            when assembling the predictive — there is no separate
            "inducing mean" because the mean function is the same.
        solver: Any ``gaussx.AbstractSolverStrategy``. Defaults to
            ``gaussx.DenseSolver()``. Used by guides that need to solve
            against ``K_zz`` (e.g.\ for KL or unwhitening).
        jitter: Diagonal regularization added to ``K_zz`` for numerical
            stability. Not a noise model — sparse SVGP does not put
            observation noise on the inducing-value prior.
    """

    kernel: Kernel
    Z: Float[Array, "M D"]
    mean_fn: Callable[[Float[Array, "N D"]], Float[Array, " N"]] | None = None
    solver: AbstractSolverStrategy | None = None
    jitter: float = 1e-6

    @property
    def num_inducing(self) -> int:
        """Number of inducing inputs ``M``."""
        return self.Z.shape[0]

    def mean(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        """Evaluate the mean function at ``X``; zero by default."""
        if self.mean_fn is None:
            return jnp.zeros(X.shape[0], dtype=X.dtype)
        return self.mean_fn(X)

    def inducing_operator(self) -> lx.AbstractLinearOperator:
        r"""Return ``K_{ZZ} + \text{jitter}\,I`` as a PSD lineax operator."""
        K = self.kernel(self.Z, self.Z)
        K = K + self.jitter * jnp.eye(K.shape[0], dtype=K.dtype)
        return _psd_operator(K)

    def cross_covariance(self, X: Float[Array, "N D"]) -> Float[Array, "N M"]:
        r""":math:`K_{XZ}` — kernel between ``X`` and the inducing inputs."""
        return self.kernel(X, self.Z)

    def kernel_diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        r"""Prior diagonal ``\mathrm{diag}\,K(X, X)`` — variance at each ``x``."""
        return self.kernel.diag(X)

    def _resolved_solver(self) -> AbstractSolverStrategy:
        return DenseSolver() if self.solver is None else self.solver  # ty: ignore[invalid-return-type]
