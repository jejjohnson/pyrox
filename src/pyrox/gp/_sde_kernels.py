"""State-space (SDE) representations of stationary GP kernels.

Stationary kernels with rational spectral densities can be represented
as linear time-invariant stochastic differential equations of the form

.. math::
    d\\mathbf{x}(t) = F\\,\\mathbf{x}(t)\\,dt + L\\,dw(t),
    \\qquad f(t) = H\\,\\mathbf{x}(t),

with :math:`P_\\infty` the stationary state covariance. Once in SDE form,
GP inference on a 1-D grid reduces to Kalman filtering in :math:`O(N\\,d^3)`
instead of :math:`O(N^3)` Cholesky.

This module ships the foundational :class:`MaternSDE` family covering
:math:`\\nu \\in \\{1/2,\\,3/2,\\,5/2\\}`. Composition rules (sum, product),
``CosineSDE``, ``PeriodicSDE``, and the Kalman-based ``MarkovGPPrior``
land in follow-up work (see issues #37, #38).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from pyrox.gp._protocols import SDEKernel


class MaternSDE(SDEKernel):
    r"""Matern kernel in state-space (companion) form for ``order in {0, 1, 2}``.

    The Matern-:math:`\nu` kernel with :math:`\nu = p + 1/2` for
    :math:`p \in \{0, 1, 2\}` has an exact :math:`d = p + 1` dimensional
    SDE representation. The closed-form parameters are:

    * **Matern-1/2** (``order=0``, :math:`d=1`): :math:`\lambda = 1/\ell`,

      .. math::
          F = [-\lambda],\quad L = [1],\quad H = [1],\quad
          Q_c = 2\sigma^2\lambda,\quad P_\infty = \sigma^2.

    * **Matern-3/2** (``order=1``, :math:`d=2`): :math:`\lambda = \sqrt{3}/\ell`,

      .. math::
          F = \begin{pmatrix} 0 & 1 \\ -\lambda^2 & -2\lambda \end{pmatrix},
          \quad L = \begin{pmatrix} 0 \\ 1 \end{pmatrix},\quad
          H = \begin{pmatrix} 1 & 0 \end{pmatrix},

      .. math::
          Q_c = 4\sigma^2\lambda^3,\quad
          P_\infty = \sigma^2\,\mathrm{diag}(1,\;\lambda^2).

    * **Matern-5/2** (``order=2``, :math:`d=3`): :math:`\lambda = \sqrt{5}/\ell`,

      .. math::
          F = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\
          -\lambda^3 & -3\lambda^2 & -3\lambda \end{pmatrix},
          \quad L = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix},\quad
          H = \begin{pmatrix} 1 & 0 & 0 \end{pmatrix},

      .. math::
          Q_c = \tfrac{16}{3}\sigma^2\lambda^5,\quad
          P_\infty = \sigma^2 \begin{pmatrix}
              1 & 0 & -\lambda^2/3 \\
              0 & \lambda^2/3 & 0 \\
              -\lambda^2/3 & 0 & \lambda^4
          \end{pmatrix}.

    ``order`` is a static (Python ``int``) field — it picks a code path,
    not a trainable parameter. ``variance`` and ``lengthscale`` are
    JAX-traced scalars suitable for autograd.

    Examples:
        >>> import jax.numpy as jnp
        >>> sde = MaternSDE(variance=1.0, lengthscale=0.5, order=1)
        >>> F, L, H, Q_c, P_inf = sde.sde_params()
        >>> A, Q = sde.discretise(jnp.array([0.1, 0.2, 0.3]))
        >>> A.shape, Q.shape
        ((3, 2, 2), (3, 2, 2))

    References:
        Sarkka & Solin (2019), *Applied Stochastic Differential Equations*,
        Ch. 12; Hartikainen & Sarkka (2010), *Kalman Filtering and
        Smoothing Solutions to Temporal Gaussian Process Regression
        Models*, IEEE MLSP.
    """

    variance: Float[Array, ""]
    lengthscale: Float[Array, ""]
    order: int = eqx.field(static=True)

    def __init__(
        self,
        variance: float | Float[Array, ""] = 1.0,
        lengthscale: float | Float[Array, ""] = 1.0,
        order: int = 1,
    ) -> None:
        if order not in (0, 1, 2):
            raise ValueError(
                "MaternSDE supports order in {0, 1, 2} (nu = order + 1/2), "
                f"got {order!r}"
            )
        self.variance = jnp.asarray(variance)
        self.lengthscale = jnp.asarray(lengthscale)
        self.order = order

    @property
    def state_dim(self) -> int:
        """State dimension ``d = order + 1``."""
        return self.order + 1

    @property
    def nu(self) -> float:
        """Smoothness ``nu = order + 1/2``."""
        return self.order + 0.5

    def sde_params(
        self,
    ) -> tuple[
        Float[Array, "d d"],
        Float[Array, "d 1"],
        Float[Array, "1 d"],
        Float[Array, "1 1"],
        Float[Array, "d d"],
    ]:
        """Return ``(F, L, H, Q_c, P_inf)`` for the chosen Matern order."""
        sigma2 = self.variance
        ell = self.lengthscale
        zero = jnp.zeros_like(ell)
        one = jnp.ones_like(ell)

        if self.order == 0:
            lam = one / ell
            F = jnp.stack([jnp.stack([-lam])])
            L = jnp.stack([jnp.stack([one])])
            H = jnp.stack([jnp.stack([one])])
            Q_c = jnp.stack([jnp.stack([2.0 * sigma2 * lam])])
            P_inf = jnp.stack([jnp.stack([sigma2])])
            return F, L, H, Q_c, P_inf

        if self.order == 1:
            lam = jnp.sqrt(jnp.asarray(3.0)) / ell
            F = jnp.stack(
                [
                    jnp.stack([zero, one]),
                    jnp.stack([-(lam**2), -2.0 * lam]),
                ]
            )
            L = jnp.stack([jnp.stack([zero]), jnp.stack([one])])
            H = jnp.stack([jnp.stack([one, zero])])
            Q_c = jnp.stack([jnp.stack([4.0 * sigma2 * lam**3])])
            P_inf = jnp.stack(
                [
                    jnp.stack([sigma2, zero]),
                    jnp.stack([zero, sigma2 * lam**2]),
                ]
            )
            return F, L, H, Q_c, P_inf

        # order == 2 (Matern-5/2)
        lam = jnp.sqrt(jnp.asarray(5.0)) / ell
        F = jnp.stack(
            [
                jnp.stack([zero, one, zero]),
                jnp.stack([zero, zero, one]),
                jnp.stack([-(lam**3), -3.0 * lam**2, -3.0 * lam]),
            ]
        )
        L = jnp.stack([jnp.stack([zero]), jnp.stack([zero]), jnp.stack([one])])
        H = jnp.stack([jnp.stack([one, zero, zero])])
        Q_c = jnp.stack([jnp.stack([(16.0 / 3.0) * sigma2 * lam**5])])
        kappa = sigma2 * lam**2 / 3.0  # off-diagonal magnitude
        P_inf = jnp.stack(
            [
                jnp.stack([sigma2, zero, -kappa]),
                jnp.stack([zero, kappa, zero]),
                jnp.stack([-kappa, zero, sigma2 * lam**4]),
            ]
        )
        return F, L, H, Q_c, P_inf
