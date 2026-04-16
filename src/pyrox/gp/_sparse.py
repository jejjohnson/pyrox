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
from gaussx import (
    AbstractSolverStrategy,
    DenseSolver,
    MultivariateNormal,
    gaussian_log_prob,
)
from jaxtyping import Array, Float

from pyrox.gp._context import _kernel_context
from pyrox.gp._protocols import Kernel


def _psd_operator(K: Float[Array, "M M"]) -> lx.AbstractLinearOperator:
    """Wrap a Gram matrix as a PSD ``lineax`` operator."""
    return lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)  # ty: ignore[invalid-return-type]


class SparseGPPrior(eqx.Module):
    r"""GP prior parameterized over inducing inputs ``Z``.

    Represents the *zero-mean* prior over inducing values ``u = f(Z)``
    used by sparse variational guides:

    .. math::

        p(u) = \mathcal{N}(0,\, K_{ZZ} + \mathrm{jitter}\,I).

    The standard SVGP convention is to subtract any global mean function
    before forming the prior over ``u`` and to add it back at predict
    time, so the inducing-prior mean is fixed to zero (this is what the
    guides' KL terms assume — see :meth:`FullRankGuide.kl_divergence`,
    :meth:`MeanFieldGuide.kl_divergence`, :meth:`WhitenedGuide.kl_divergence`).
    The :attr:`mean_fn` attribute on this class is exposed as a
    convenience for callers that want to add ``mu(X_*)`` back onto the
    predictive mean returned by :meth:`Guide.predict`; it is **not**
    incorporated in :meth:`inducing_operator` or in the guides' KL.

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
            Convenience accessor; not folded into the inducing prior.
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
        r"""Return ``K_{ZZ} + \text{jitter}\,I`` as a PSD lineax operator.

        Single kernel call; safe standalone for kernels with priors. For
        building several SVGP blocks together, prefer
        :meth:`predictive_blocks`, which scopes one shared kernel
        context across ``K_zz``, ``K_xz``, and ``K_xx_diag`` so
        Pattern B / C kernels register their NumPyro hyperparameter
        sites once instead of resampling per call.
        """
        with _kernel_context(self.kernel):
            K = self.kernel(self.Z, self.Z)
        K = K + self.jitter * jnp.eye(K.shape[0], dtype=K.dtype)
        return _psd_operator(K)

    def cross_covariance(self, X: Float[Array, "N D"]) -> Float[Array, "N M"]:
        r""":math:`K_{XZ}` — kernel between ``X`` and the inducing inputs.

        See :meth:`predictive_blocks` for the shared-context batch
        helper to use when assembling several SVGP blocks together.
        """
        with _kernel_context(self.kernel):
            return self.kernel(X, self.Z)

    def kernel_diag(self, X: Float[Array, "N D"]) -> Float[Array, " N"]:
        r"""Prior diagonal ``\mathrm{diag}\,K(X, X)`` — variance at each ``x``.

        See :meth:`predictive_blocks` for the shared-context batch
        helper to use when assembling several SVGP blocks together.
        """
        with _kernel_context(self.kernel):
            return self.kernel.diag(X)

    def predictive_blocks(
        self, X: Float[Array, "N D"]
    ) -> tuple[
        lx.AbstractLinearOperator,
        Float[Array, "N M"],
        Float[Array, " N"],
    ]:
        r"""Return ``(K_zz_op, K_xz, K_xx_diag)`` under one shared kernel context.

        For Pattern B / C kernels with prior'd hyperparameters, the three
        kernel evaluations needed for an SVGP predictive must share a
        single :class:`pyrox.PyroxModule` context so the underlying
        ``pyrox_sample`` sites register once and yield consistent
        hyperparameter draws across ``K_{ZZ}``, ``K_{XZ}``, and the
        diagonal ``\mathrm{diag}\,K(X, X)``. Without this scoping, three
        separate calls would draw three independent hyperparameter
        samples (under seed) or raise NumPyro duplicate-site errors
        (under tracing) — either way invalidating the SVGP math.

        For pure :class:`equinox.Module` kernels (no ``_get_context``),
        this is equivalent to calling :meth:`inducing_operator`,
        :meth:`cross_covariance`, and :meth:`kernel_diag` independently.
        """
        with _kernel_context(self.kernel):
            K_zz_raw = self.kernel(self.Z, self.Z)
            K_xz = self.kernel(X, self.Z)
            K_xx_diag = self.kernel.diag(X)
        K_zz = K_zz_raw + self.jitter * jnp.eye(K_zz_raw.shape[0], dtype=K_zz_raw.dtype)
        return _psd_operator(K_zz), K_xz, K_xx_diag

    def _resolved_solver(self) -> AbstractSolverStrategy:
        return DenseSolver() if self.solver is None else self.solver  # ty: ignore[invalid-return-type]

    def log_prob(self, u: Float[Array, " M"]) -> Float[Array, ""]:
        r"""Log-density under :math:`p(u) = \mathcal{N}(0, K_{ZZ} + \text{jitter}\,I)`.

        Delegates to :func:`gaussx.gaussian_log_prob` with the
        configured :attr:`solver` so the user-supplied solver controls
        the ``solve`` / ``logdet`` work on ``K_zz_op``. Useful for
        scoring inducing values against the SVGP prior in non-NumPyro
        contexts (e.g.\\ tests, diagnostics).
        """
        m = jnp.zeros(self.num_inducing, dtype=u.dtype)
        return gaussian_log_prob(
            m, self.inducing_operator(), u, solver=self._resolved_solver()
        )

    def sample(self, key: Array) -> Float[Array, " M"]:
        r"""Draw ``u \sim p(u)`` from the inducing prior.

        Wraps the inducing operator in a
        :class:`gaussx.MultivariateNormal` with the configured
        :attr:`solver`. ``MultivariateNormal.sample`` factors the
        covariance via :func:`gaussx.cholesky` and reparameterizes;
        the returned draw has shape ``(M,)``.

        Note: the SVGP variational workflow samples ``u`` from the
        *guide* :math:`q(u)`, not the prior. This method exists so the
        prior surface is symmetric with the guide surface and so users
        can score / draw inducing values against the prior directly
        (e.g.\\ for tests or for prior-sample initialization).
        """
        n = self.num_inducing
        op = self.inducing_operator()
        loc = jnp.zeros(n, dtype=op.out_structure().dtype)
        mvn = MultivariateNormal(loc, op, solver=self._resolved_solver())
        return mvn.sample(key)
