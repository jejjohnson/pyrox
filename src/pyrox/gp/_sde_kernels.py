"""State-space (SDE) representations of stationary GP kernels.

Stationary kernels with rational spectral densities can be represented
as linear time-invariant stochastic differential equations of the form

.. math::
    d\\mathbf{x}(t) = F\\,\\mathbf{x}(t)\\,dt + L\\,dw(t),
    \\qquad f(t) = H\\,\\mathbf{x}(t),

with :math:`P_\\infty` the stationary state covariance. Once in SDE form,
GP inference on a 1-D grid reduces to Kalman filtering in :math:`O(N\\,d^3)`
instead of :math:`O(N^3)` Cholesky.

This module ships the Matern family (:class:`MaternSDE`), the elementary
primitives :class:`ConstantSDE` and :class:`CosineSDE`, the Fourier
truncated :class:`PeriodicSDE`, and the composition rules
:class:`SumSDE` (block-diagonal) and :class:`ProductSDE` (Kronecker).
:class:`QuasiPeriodicSDE` is a convenience wrapper for
``ProductSDE(MaternSDE, PeriodicSDE)``.

The Kalman-based :class:`MarkovGPPrior` and the temporal inference
strategies land in issue #38.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jax.scipy.special as jss
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
        # Eager positivity checks for concrete (non-traced) Python scalar inputs.
        # JAX tracer inputs (e.g. inside ``jax.jit``) are skipped — under tracing
        # we cannot inspect the value; downstream training-time priors handle
        # constraint enforcement.
        if isinstance(variance, (int, float)) and variance <= 0:
            raise ValueError(f"variance must be positive, got {variance!r}")
        if isinstance(lengthscale, (int, float)) and lengthscale <= 0:
            raise ValueError(f"lengthscale must be positive, got {lengthscale!r}")
        # Coerce to a floating dtype so integer inputs (``variance=1``) don't
        # propagate as integer-typed parameters.
        self.variance = jnp.asarray(variance, dtype=jnp.result_type(variance, 0.0))
        self.lengthscale = jnp.asarray(
            lengthscale, dtype=jnp.result_type(lengthscale, 0.0)
        )
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


class ConstantSDE(SDEKernel):
    r"""Constant kernel :math:`k(\tau) = \sigma^2` in state-space form.

    A degenerate 1-D state space with zero dynamics and zero diffusion:

    .. math::
        F = [0],\quad L = [0],\quad H = [1],\quad Q_c = [0],\quad
        P_\infty = [\sigma^2].

    The transition is the identity ``A_k = I`` and the process noise is
    zero ``Q_k = 0``. Useful as a non-trivial component of a
    :class:`SumSDE` (e.g. ``Matern + Constant`` for a fixed offset).
    """

    variance: Float[Array, ""]

    def __init__(self, variance: float | Float[Array, ""] = 1.0) -> None:
        self.variance = jnp.asarray(variance)

    @property
    def state_dim(self) -> int:
        return 1

    def sde_params(
        self,
    ) -> tuple[
        Float[Array, "1 1"],
        Float[Array, "1 1"],
        Float[Array, "1 1"],
        Float[Array, "1 1"],
        Float[Array, "1 1"],
    ]:
        sigma2 = self.variance
        zero = jnp.zeros_like(sigma2)
        one = jnp.ones_like(sigma2)
        F = jnp.stack([jnp.stack([zero])])
        L = jnp.stack([jnp.stack([zero])])
        H = jnp.stack([jnp.stack([one])])
        Q_c = jnp.stack([jnp.stack([zero])])
        P_inf = jnp.stack([jnp.stack([sigma2])])
        return F, L, H, Q_c, P_inf


class CosineSDE(SDEKernel):
    r"""Cosine kernel :math:`k(\tau) = \sigma^2 \cos(\omega_0 \tau)` in SDE form.

    A 2-D deterministic oscillator with rotation matrix transitions:

    .. math::
        F = \begin{pmatrix} 0 & -\omega_0 \\ \omega_0 & 0 \end{pmatrix},
        \quad L = \begin{pmatrix} 0 \\ 0 \end{pmatrix},\quad
        H = \begin{pmatrix} 1 & 0 \end{pmatrix},

    .. math::
        Q_c = 0,\quad P_\infty = \sigma^2 I_2.

    There is no driving noise, so the discrete-time transition is a pure
    rotation :math:`A_k = R(\omega_0\,\Delta t_k)` and :math:`Q_k = 0`.
    The :meth:`discretise` method overrides the default ``expm`` path
    with the closed-form rotation for efficiency.
    """

    variance: Float[Array, ""]
    frequency: Float[Array, ""]

    def __init__(
        self,
        variance: float | Float[Array, ""] = 1.0,
        frequency: float | Float[Array, ""] = 1.0,
    ) -> None:
        self.variance = jnp.asarray(variance)
        self.frequency = jnp.asarray(frequency)

    @property
    def state_dim(self) -> int:
        return 2

    def sde_params(
        self,
    ) -> tuple[
        Float[Array, "2 2"],
        Float[Array, "2 1"],
        Float[Array, "1 2"],
        Float[Array, "1 1"],
        Float[Array, "2 2"],
    ]:
        sigma2 = self.variance
        omega = self.frequency
        zero = jnp.zeros_like(omega)
        one = jnp.ones_like(omega)
        F = jnp.stack(
            [
                jnp.stack([zero, -omega]),
                jnp.stack([omega, zero]),
            ]
        )
        L = jnp.stack([jnp.stack([zero]), jnp.stack([zero])])
        H = jnp.stack([jnp.stack([one, zero])])
        Q_c = jnp.stack([jnp.stack([zero])])
        P_inf = jnp.stack(
            [
                jnp.stack([sigma2, zero]),
                jnp.stack([zero, sigma2]),
            ]
        )
        return F, L, H, Q_c, P_inf

    def discretise(
        self,
        dt: Float[Array, " N"],
    ) -> tuple[Float[Array, "N 2 2"], Float[Array, "N 2 2"]]:
        """Closed-form rotation: ``A_k = R(omega * dt_k)``, ``Q_k = 0``."""
        omega = self.frequency
        theta = jnp.asarray(dt) * omega
        c, s = jnp.cos(theta), jnp.sin(theta)
        # A has shape (N, 2, 2); stack rows then columns.
        A = jnp.stack(
            [jnp.stack([c, -s], axis=-1), jnp.stack([s, c], axis=-1)],
            axis=-2,
        )
        Q = jnp.zeros_like(A)
        return A, Q


class SumSDE(SDEKernel):
    r"""Sum of SDE kernels via block-diagonal state-space composition.

    For :math:`k(\tau) = \sum_i k_i(\tau)`, the SDE is the block-diagonal
    concatenation of each component:

    .. math::
        F = \mathrm{blkdiag}(F_1, \dots, F_K),\quad
        L = \mathrm{blkdiag}(L_1, \dots, L_K),\quad
        Q_c = \mathrm{blkdiag}(Q_{c,1}, \dots, Q_{c,K}),

    .. math::
        H = [H_1, \dots, H_K],\quad
        P_\infty = \mathrm{blkdiag}(P_{\infty,1}, \dots, P_{\infty,K}).

    Total state dimension is :math:`\sum_i d_i`. Components with disjoint
    state spaces evolve independently.
    """

    components: tuple[SDEKernel, ...]

    def __init__(self, components: tuple[SDEKernel, ...]) -> None:
        if len(components) < 1:
            raise ValueError("SumSDE requires at least one component.")
        self.components = tuple(components)

    @property
    def state_dim(self) -> int:
        return sum(c.state_dim for c in self.components)

    def sde_params(
        self,
    ) -> tuple[
        Float[Array, "d d"],
        Float[Array, "d s"],
        Float[Array, "1 d"],
        Float[Array, "s s"],
        Float[Array, "d d"],
    ]:
        params = [c.sde_params() for c in self.components]
        Fs, Ls, Hs, Qcs, Ps = zip(*params, strict=True)
        F = jsl.block_diag(*Fs)
        L = jsl.block_diag(*Ls)
        H = jnp.concatenate(Hs, axis=1)
        Q_c = jsl.block_diag(*Qcs)
        P_inf = jsl.block_diag(*Ps)
        return F, L, H, Q_c, P_inf


class ProductSDE(SDEKernel):
    r"""Product of two SDE kernels via Kronecker composition.

    For :math:`k(\tau) = k_1(\tau)\,k_2(\tau)`, the joint SDE has
    Kronecker-sum drift and Kronecker-product readout / stationary
    covariance:

    .. math::
        F = F_1 \otimes I_{d_2} + I_{d_1} \otimes F_2,\quad
        H = H_1 \otimes H_2,\quad
        P_\infty = P_{\infty,1} \otimes P_{\infty,2}.

    The diffusion is *not* a simple Kronecker product. Substituting into
    the Lyapunov equation yields

    .. math::
        L Q_c L^\top = (L_1 Q_{c,1} L_1^\top) \otimes P_{\infty,2}
        + P_{\infty,1} \otimes (L_2 Q_{c,2} L_2^\top).

    For simplicity we set :math:`L = I_{d_1 d_2}` and store the right-hand
    side as ``Q_c`` directly. Total state dimension is :math:`d_1 \cdot d_2`.
    """

    left: SDEKernel
    right: SDEKernel

    def __init__(self, left: SDEKernel, right: SDEKernel) -> None:
        self.left = left
        self.right = right

    @property
    def state_dim(self) -> int:
        return self.left.state_dim * self.right.state_dim

    def sde_params(
        self,
    ) -> tuple[
        Float[Array, "d d"],
        Float[Array, "d d"],
        Float[Array, "1 d"],
        Float[Array, "d d"],
        Float[Array, "d d"],
    ]:
        F1, L1, H1, Qc1, P1 = self.left.sde_params()
        F2, L2, H2, Qc2, P2 = self.right.sde_params()
        d1, d2 = F1.shape[0], F2.shape[0]
        I1 = jnp.eye(d1, dtype=F1.dtype)
        I2 = jnp.eye(d2, dtype=F2.dtype)
        F = jnp.kron(F1, I2) + jnp.kron(I1, F2)
        H = jnp.kron(H1, H2)
        P_inf = jnp.kron(P1, P2)
        D1 = L1 @ Qc1 @ L1.T  # (d1, d1)
        D2 = L2 @ Qc2 @ L2.T  # (d2, d2)
        Q_c = jnp.kron(D1, P2) + jnp.kron(P1, D2)
        L = jnp.eye(d1 * d2, dtype=F.dtype)
        return F, L, H, Q_c, P_inf


def _scaled_bessel_i_seq(
    x: Float[Array, ""],
    j_max: int,
    n_terms: int = 60,
) -> Float[Array, " j_max_plus_1"]:
    r"""Scaled modified Bessel ``exp(-x) I_j(x)`` for ``j = 0..j_max``.

    Computes each entry from the Taylor series

    .. math::
        I_j(x) = \sum_{k=0}^\infty \frac{(x/2)^{j+2k}}{k!\,(k+j)!}

    in log space and accumulates via :func:`jax.scipy.special.logsumexp`,
    multiplying through by ``exp(-x)``. This avoids the catastrophic
    overflow of upward / Miller-style recursion when ``x = 1/\ell^2``
    spans the long-lengthscale regime ``x \ll 1``, while remaining
    accurate across the practical periodic-kernel range.

    Args:
        x: Positive scalar argument (typically ``1 / lengthscale ** 2``).
        j_max: Maximum Bessel order required.
        n_terms: Number of Taylor terms; default 60 is enough for at
            least 1e-6 precision over ``x in [1e-3, 30]`` and
            ``j_max <= 12``.

    Returns:
        Array of shape ``(j_max + 1,)`` with entries ``exp(-x) I_j(x)``.
    """
    js = jnp.arange(j_max + 1, dtype=x.dtype)  # (J+1,)
    ks = jnp.arange(n_terms, dtype=x.dtype)  # (K,)
    log_half_x = jnp.log(x / 2.0)
    # log_term[j, k] = (j + 2k) log(x/2) - x - log(k!) - log((k+j)!)
    j_grid = js[:, None]
    k_grid = ks[None, :]
    log_term = (
        (j_grid + 2.0 * k_grid) * log_half_x
        - x
        - jss.gammaln(k_grid + 1.0)
        - jss.gammaln(k_grid + j_grid + 1.0)
    )
    return jnp.exp(jss.logsumexp(log_term, axis=1))


class PeriodicSDE(SDEKernel):
    r"""Periodic kernel in state-space form via Fourier-series truncation.

    The MacKay periodic kernel
    :math:`k(\tau) = \sigma^2 \exp\!\bigl(-2 \sin^2(\pi\tau/T)/\ell^2\bigr)`
    expands as

    .. math::
        k(\tau) = \sigma^2 e^{-1/\ell^2}
        \Bigl[I_0(1/\ell^2) + 2 \sum_{j=1}^\infty I_j(1/\ell^2)
        \cos(j\,\omega_0 \tau)\Bigr],

    with :math:`\omega_0 = 2\pi/T`. Truncating to ``J = n_harmonics``
    cosines gives a deterministic state-space model whose state collects
    a degenerate 1-D constant block (the :math:`j=0` DC term) and ``J``
    rotation blocks, one per harmonic. Total state dimension is
    :math:`1 + 2J`, ``L = 0``, ``Q_c = 0`` (no driving noise), and
    :math:`P_\infty` is block-diagonal with entries

    .. math::
        q_0 = \sigma^2 e^{-1/\ell^2} I_0(1/\ell^2),\qquad
        q_j = 2 \sigma^2 e^{-1/\ell^2} I_j(1/\ell^2)\quad (j \geq 1).

    Bessel coefficients are computed via Miller's downward recursion
    (see :func:`_scaled_bessel_i_seq`). For ``n_harmonics`` around 7 the
    truncation matches the dense MacKay periodic kernel to better than
    1e-6 across the typical hyperparameter regime.

    References:
        Solin & Sarkka (2014), *Explicit Link Between Periodic Covariance
        Functions and State Space Models*, AISTATS.
    """

    variance: Float[Array, ""]
    lengthscale: Float[Array, ""]
    period: Float[Array, ""]
    n_harmonics: int = eqx.field(static=True)

    def __init__(
        self,
        variance: float | Float[Array, ""] = 1.0,
        lengthscale: float | Float[Array, ""] = 1.0,
        period: float | Float[Array, ""] = 1.0,
        n_harmonics: int = 7,
    ) -> None:
        if n_harmonics < 1:
            raise ValueError(
                f"PeriodicSDE requires n_harmonics >= 1, got {n_harmonics!r}"
            )
        self.variance = jnp.asarray(variance)
        self.lengthscale = jnp.asarray(lengthscale)
        self.period = jnp.asarray(period)
        self.n_harmonics = n_harmonics

    @property
    def state_dim(self) -> int:
        return 1 + 2 * self.n_harmonics

    def sde_params(
        self,
    ) -> tuple[
        Float[Array, "d d"],
        Float[Array, "d 1"],
        Float[Array, "1 d"],
        Float[Array, "1 1"],
        Float[Array, "d d"],
    ]:
        J = self.n_harmonics
        d = 1 + 2 * J
        sigma2 = self.variance
        ell = self.lengthscale
        T = self.period
        omega0 = 2.0 * jnp.pi / T

        # Rotation blocks F_j = [[0, -j*omega0], [j*omega0, 0]] for j = 1..J.
        # Plus a 1x1 zero block at the top for the j=0 DC mode.
        blocks = [jnp.zeros((1, 1), dtype=ell.dtype)]
        for j in range(1, J + 1):
            wj = jnp.asarray(j, dtype=ell.dtype) * omega0
            zero = jnp.zeros_like(wj)
            blocks.append(jnp.stack([jnp.stack([zero, -wj]), jnp.stack([wj, zero])]))
        F = jsl.block_diag(*blocks)

        # H reads out the cosine ("first") coordinate of each block.
        # Layout: [DC, cos_1, sin_1, cos_2, sin_2, ..., cos_J, sin_J].
        H_entries = jnp.zeros(d, dtype=ell.dtype)
        idx = jnp.array([0] + [1 + 2 * (j - 1) for j in range(1, J + 1)])
        H_entries = H_entries.at[idx].set(1.0)
        H = H_entries[None, :]

        # No driving noise: L = 0, Q_c = 0.
        L = jnp.zeros((d, 1), dtype=ell.dtype)
        Q_c = jnp.zeros((1, 1), dtype=ell.dtype)

        # Bessel-weighted P_inf:
        #   q_0 = sigma^2 * i0e(1/ell^2),  q_j = 2*sigma^2 * i_j_e(1/ell^2).
        x = 1.0 / (ell * ell)
        i_seq = _scaled_bessel_i_seq(x, j_max=J)  # (J+1,)
        q_vals = jnp.concatenate(
            [
                sigma2 * i_seq[0:1],
                2.0 * sigma2 * i_seq[1:],
            ]
        )

        # Diagonal entries of P_inf: q_0 (1 entry), then q_j repeated twice
        # for the (cos, sin) coordinates of each rotation block.
        diag = jnp.concatenate(
            [
                q_vals[0:1],
                jnp.repeat(q_vals[1:], 2),
            ]
        )
        P_inf = jnp.diag(diag)
        return F, L, H, Q_c, P_inf


class QuasiPeriodicSDE(ProductSDE):
    r"""Quasi-periodic kernel: :math:`k(\tau) = k_{\rm Mat}(\tau)\,k_{\rm Per}(\tau)`.

    A thin documented subclass of :class:`ProductSDE` that captures the
    standard Matern :math:`\times` Periodic decomposition used for
    modulated periodic signals (stellar light curves, modulated seasonal
    patterns). The Matern envelope sets the timescale on which the
    amplitude drifts; the periodic factor sets the cycle.

    Example:
        >>> import jax.numpy as jnp
        >>> qp = QuasiPeriodicSDE(
        ...     MaternSDE(variance=1.0, lengthscale=2.0, order=1),
        ...     PeriodicSDE(variance=1.0, lengthscale=1.0, period=1.0, n_harmonics=5),
        ... )
        >>> qp.state_dim
        22

    References:
        Sarkka & Solin (2019), *Applied Stochastic Differential Equations*,
        Sec. 12.3; Wilkinson et al. (2021), *BayesNewton*.
    """

    def __init__(self, matern: SDEKernel, periodic: SDEKernel) -> None:
        super().__init__(left=matern, right=periodic)
