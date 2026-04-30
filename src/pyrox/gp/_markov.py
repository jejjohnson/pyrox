"""Kalman-based Markov GP prior on top of the SDE-kernel surface.

Stationary 1-D GP kernels with rational spectral densities admit exact
linear-Gaussian state-space representations (see :mod:`pyrox.gp._sde_kernels`
and the :class:`pyrox.gp.SDEKernel` protocol). This module turns those
representations into a working temporal-GP model: forward Kalman filter for
the marginal log-likelihood, backward RTS smoother for the posterior, and a
NumPyro-aware shell so the path drops into models the same way
:class:`pyrox.gp.GPPrior` does.

For a sorted observation grid of length :math:`N` and SDE state dimension
:math:`d`, both the marginal likelihood and the smoothed posterior cost
:math:`O(N\\,d^3)` — linear in :math:`N`, where the dense path is
:math:`O(N^3)`.

Test-time predictions for arbitrary ``t_star`` are produced by re-running the
filter+smoother over the merged grid ``sort(times \\cup t_star)`` with the
unobserved test points masked out of the update step. This handles
training-grid lookups, forecasting (``t_star`` after the data), backcasting
(``t_star`` before the data), and within-window interpolation under one code
path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
from jaxtyping import Array, Float
from numpyro import distributions as dist

from pyrox.gp._protocols import Likelihood, SDEKernel


if TYPE_CHECKING:
    from pyrox.gp._inference_nongauss_markov import NonGaussConditionedMarkovGP


class _NonGaussMarkovStrategy(Protocol):
    """Structural protocol for non-Gaussian Markov inference strategies."""

    def fit(
        self,
        prior: MarkovGPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
    ) -> NonGaussConditionedMarkovGP: ...


def _kalman_filter(
    F: Float[Array, "d d"],
    H: Float[Array, "1 d"],
    P_inf: Float[Array, "d d"],
    A_seq: Float[Array, "N d d"],
    Q_seq: Float[Array, "N d d"],
    residual: Float[Array, " N"],
    mask: Float[Array, " N"],
    R_seq: Float[Array, " N"],
) -> tuple[
    Float[Array, "N d"],
    Float[Array, "N d d"],
    Float[Array, "N d"],
    Float[Array, "N d d"],
    Float[Array, ""],
]:
    """Forward Kalman filter with optional per-step observation mask.

    Operates in zero-mean residual space: ``residual = y - mean_fn(times)``.
    ``mask[k] = 1`` performs the standard observation update at step ``k``;
    ``mask[k] = 0`` skips the update (no information from ``residual[k]``
    enters the state). The masked formulation makes test-time prediction a
    single forward+backward pass over the merged grid.

    ``R_seq`` is the per-step observation variance. For Gaussian-likelihood
    inference this is a constant ``noise_var`` broadcast to ``(N,)``; for
    site-based non-Gaussian inference the per-step value is
    ``1 / Lambda_n`` from the current site precisions.

    Returns predicted and filtered ``(mean, cov)`` per step plus the total
    log marginal likelihood (only observed steps contribute).
    """
    d = F.shape[0]
    eye = jnp.eye(d, dtype=F.dtype)
    log_2pi = jnp.log(2.0 * jnp.pi).astype(F.dtype)

    def step(
        carry: tuple[Float[Array, " d"], Float[Array, "d d"]],
        inputs: tuple[
            Float[Array, "d d"],
            Float[Array, "d d"],
            Float[Array, ""],
            Float[Array, ""],
            Float[Array, ""],
        ],
    ) -> tuple[
        tuple[Float[Array, " d"], Float[Array, "d d"]],
        tuple[
            Float[Array, " d"],
            Float[Array, "d d"],
            Float[Array, " d"],
            Float[Array, "d d"],
            Float[Array, ""],
        ],
    ]:
        m, P = carry
        A, Q, r_n, mask_n, R_n = inputs

        m_pred = A @ m
        P_pred = A @ P @ A.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)

        Hp = (H @ P_pred)[0]  # (d,)
        S = (Hp @ H.T[:, 0]) + R_n  # scalar variance of innovation
        innov = r_n - (H @ m_pred)[0]
        # NaN-safe division: when ``mask_n = 0`` we discard the result, but
        # ``S`` can still be 0 (noiseless conditioning, deterministic
        # segments). Replace ``S`` with 1 on the masked branch so neither
        # ``... / S`` nor ``log(S)`` produces inf/NaN that ``0 *`` would
        # propagate via ``0 * inf = NaN``.
        is_obs = mask_n == 1
        S_safe = jnp.where(is_obs, S, 1.0)
        K_full = (P_pred @ H.T)[:, 0] / S_safe  # (d,)
        K = jnp.where(is_obs, K_full, jnp.zeros_like(K_full))

        m_new = m_pred + K * innov
        I_minus_KH = eye - K[:, None] * H
        # ``K`` is exactly zero on masked steps, so ``I - K H`` is the
        # identity and the Joseph term collapses to ``P_pred`` — no extra
        # ``mask_n`` factor needed on ``R``.
        P_new = I_minus_KH @ P_pred @ I_minus_KH.T + R_n * (K[:, None] @ K[None, :])
        P_new = 0.5 * (P_new + P_new.T)

        ll_term = -0.5 * (log_2pi + jnp.log(S_safe) + innov * innov / S_safe)
        ll = jnp.where(is_obs, ll_term, 0.0)
        return (m_new, P_new), (m_pred, P_pred, m_new, P_new, ll)

    m0 = jnp.zeros(d, dtype=F.dtype)
    P0 = P_inf
    _, (m_pred_seq, P_pred_seq, m_filt_seq, P_filt_seq, ll_seq) = jax.lax.scan(
        step, (m0, P0), (A_seq, Q_seq, residual, mask, R_seq)
    )
    return m_pred_seq, P_pred_seq, m_filt_seq, P_filt_seq, jnp.sum(ll_seq)


def _rts_smoother(
    m_pred_seq: Float[Array, "N d"],
    P_pred_seq: Float[Array, "N d d"],
    m_filt_seq: Float[Array, "N d"],
    P_filt_seq: Float[Array, "N d d"],
    A_seq: Float[Array, "N d d"],
) -> tuple[Float[Array, "N d"], Float[Array, "N d d"]]:
    """Backward Rauch-Tung-Striebel smoother.

    ``A_seq[k]`` is the transition matrix used to predict step ``k`` from
    step ``k - 1``; the smoother walks backward from the last filtered state.
    """

    def step(
        carry: tuple[Float[Array, " d"], Float[Array, "d d"]],
        inputs: tuple[
            Float[Array, " d"],
            Float[Array, "d d"],
            Float[Array, " d"],
            Float[Array, "d d"],
            Float[Array, "d d"],
        ],
    ) -> tuple[
        tuple[Float[Array, " d"], Float[Array, "d d"]],
        tuple[Float[Array, " d"], Float[Array, "d d"]],
    ]:
        m_next_smooth, P_next_smooth = carry
        m_filt, P_filt, m_pred_next, P_pred_next, A_next = inputs
        # J = P_filt @ A_next^T @ P_pred_next^{-1}
        J = jnp.linalg.solve(P_pred_next.T, A_next @ P_filt.T).T
        m_smooth = m_filt + J @ (m_next_smooth - m_pred_next)
        P_smooth = P_filt + J @ (P_next_smooth - P_pred_next) @ J.T
        P_smooth = 0.5 * (P_smooth + P_smooth.T)
        return (m_smooth, P_smooth), (m_smooth, P_smooth)

    m_T = m_filt_seq[-1]
    P_T = P_filt_seq[-1]
    inputs = (
        m_filt_seq[:-1],
        P_filt_seq[:-1],
        m_pred_seq[1:],
        P_pred_seq[1:],
        A_seq[1:],
    )
    _, (m_smooth_seq, P_smooth_seq) = jax.lax.scan(
        step, (m_T, P_T), inputs, reverse=True
    )
    m_smooth_full = jnp.concatenate([m_smooth_seq, m_T[None]], axis=0)
    P_smooth_full = jnp.concatenate([P_smooth_seq, P_T[None]], axis=0)
    return m_smooth_full, P_smooth_full


def _build_dt_full(times: Float[Array, " N"]) -> Float[Array, " N"]:
    """Pad ``diff(times)`` with a leading zero so step 0 is the prior."""
    dt = jnp.diff(times)
    return jnp.concatenate([jnp.zeros((1,), dtype=dt.dtype), dt], axis=0)


class MarkovGPPrior(eqx.Module):
    r"""Linear-time temporal GP prior over a sorted 1-D grid.

    Wraps any :class:`pyrox.gp.SDEKernel` (e.g. :class:`pyrox.gp.MaternSDE`,
    :class:`pyrox.gp.SumSDE`, :class:`pyrox.gp.PeriodicSDE`) to give Kalman
    filtering for the marginal log-likelihood and RTS smoothing for the
    posterior on the training grid. Supports an optional mean function and a
    small observation-noise floor for numerical stability.

    Attributes:
        sde_kernel: Any :class:`SDEKernel`. Provides ``(F, L, H, Q_c, P_inf)``
            via ``sde_params()`` and the discrete transition tuple via
            ``discretise(dt)``.
        times: Sorted, strictly increasing observation times of shape
            ``(N,)``. Concrete (non-traced) ``times`` arrays are validated
            for monotonicity at construction time; under :func:`jax.jit` /
            SVI / MCMC the input is a tracer and the check is silently
            skipped — callers must guarantee monotonicity in that case.
        mean_fn: Optional callable mapping ``times -> (N,)`` mean values.
            Defaults to the zero mean. The mean is subtracted from
            observations before filtering and added back at predict time.
        obs_noise_floor: Small extra diagonal added to the observation
            variance ``R = noise_var + obs_noise_floor`` for stability when
            ``noise_var`` is near zero. Defaults to ``0.0``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from pyrox.gp import MaternSDE, MarkovGPPrior
        >>> times = jnp.linspace(0.0, 5.0, 50)
        >>> sde = MaternSDE(variance=1.0, lengthscale=0.5, order=1)
        >>> prior = MarkovGPPrior(sde, times)
        >>> y = jnp.sin(times) + 0.05 * jnp.cos(3.0 * times)
        >>> log_marg = prior.log_marginal(y, jnp.asarray(0.01))

    Notes:
        The solver-strategy plumbing used by :class:`pyrox.gp.GPPrior` does
        not apply here — Kalman filtering is its own linear-algebra path
        and does not factor through ``gaussx.AbstractSolverStrategy``.
    """

    sde_kernel: SDEKernel
    times: Float[Array, " N"]
    mean_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]] | None = None
    obs_noise_floor: float = eqx.field(static=True, default=0.0)

    def __init__(
        self,
        sde_kernel: SDEKernel,
        times: Float[Array, " N"],
        mean_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]] | None = None,
        obs_noise_floor: float = 0.0,
    ) -> None:
        if obs_noise_floor < 0:
            raise ValueError(
                f"obs_noise_floor must be non-negative, got {obs_noise_floor!r}"
            )
        times_arr = jnp.asarray(times, dtype=jnp.result_type(times, 0.0))
        if times_arr.ndim != 1:
            raise ValueError(f"times must be 1-D, got shape {tuple(times_arr.shape)!r}")
        # Eager monotonicity check for concrete (non-traced) inputs only.
        # Under ``jax.jit`` / SVI / similar transforms ``times`` may arrive as
        # a tracer; the ``bool`` conversion would raise, so we silence that
        # path and let downstream Kalman steps trust the contract.
        if times_arr.shape[0] >= 2:
            try:
                if not bool(jnp.all(jnp.diff(times_arr) > 0)):
                    raise ValueError("times must be strictly increasing")
            except jax.errors.TracerBoolConversionError:
                pass
        self.sde_kernel = sde_kernel
        self.times = times_arr
        self.mean_fn = mean_fn
        self.obs_noise_floor = float(obs_noise_floor)

    @property
    def state_dim(self) -> int:
        """SDE state dimension :math:`d` for this kernel."""
        return self.sde_kernel.state_dim

    def mean(self, times: Float[Array, " M"]) -> Float[Array, " M"]:
        """Evaluate the mean function at ``times``; zero by default."""
        if self.mean_fn is None:
            return jnp.zeros_like(times)
        return self.mean_fn(times)

    def _residual(self, y: Float[Array, " N"]) -> Float[Array, " N"]:
        return y - self.mean(self.times)

    def _R(self, noise_var: Float[Array, ""]) -> Float[Array, ""]:
        return jnp.asarray(noise_var) + jnp.asarray(self.obs_noise_floor)

    def filter(
        self,
        y: Float[Array, " N"],
        noise_var: Float[Array, ""],
    ) -> tuple[
        Float[Array, "N d"],
        Float[Array, "N d d"],
        Float[Array, "N d"],
        Float[Array, "N d d"],
        Float[Array, ""],
    ]:
        """Run the forward Kalman filter on the training grid.

        Returns:
            Tuple ``(m_pred, P_pred, m_filt, P_filt, log_marginal)`` where
            each ``*_pred`` / ``*_filt`` is shaped ``(N, d)`` or
            ``(N, d, d)`` and ``log_marginal`` is the scalar log-likelihood
            ``log p(y | theta)``.
        """
        F, _L, H, _Qc, P_inf = self.sde_kernel.sde_params()
        dt_full = _build_dt_full(self.times)
        A_seq, Q_seq = self.sde_kernel.discretise(dt_full)
        residual = self._residual(y)
        mask = jnp.ones_like(self.times)
        R_seq = jnp.broadcast_to(self._R(noise_var), self.times.shape)
        return _kalman_filter(F, H, P_inf, A_seq, Q_seq, residual, mask, R_seq)

    def log_marginal(
        self,
        y: Float[Array, " N"],
        noise_var: Float[Array, ""],
    ) -> Float[Array, ""]:
        r"""Marginal log-likelihood ``log p(y | theta)`` via Kalman filtering."""
        *_, log_marg = self.filter(y, noise_var)
        return log_marg

    def smooth(
        self,
        y: Float[Array, " N"],
        noise_var: Float[Array, ""],
    ) -> tuple[Float[Array, "N d"], Float[Array, "N d d"], Float[Array, ""]]:
        """Run filter + RTS smoother on the training grid.

        Returns ``(m_smooth, P_smooth, log_marginal)`` over the training
        times.
        """
        F, _L, H, _Qc, P_inf = self.sde_kernel.sde_params()
        dt_full = _build_dt_full(self.times)
        A_seq, Q_seq = self.sde_kernel.discretise(dt_full)
        residual = self._residual(y)
        mask = jnp.ones_like(self.times)
        R_seq = jnp.broadcast_to(self._R(noise_var), self.times.shape)
        m_pred, P_pred, m_filt, P_filt, log_marg = _kalman_filter(
            F, H, P_inf, A_seq, Q_seq, residual, mask, R_seq
        )
        m_smooth, P_smooth = _rts_smoother(m_pred, P_pred, m_filt, P_filt, A_seq)
        return m_smooth, P_smooth, log_marg

    def condition(
        self,
        y: Float[Array, " N"],
        noise_var: Float[Array, ""],
    ) -> ConditionedMarkovGP:
        """Condition on Gaussian-likelihood observations via filter + smoother."""
        m_smooth, P_smooth, log_marg = self.smooth(y, noise_var)
        return ConditionedMarkovGP(  # ty: ignore[invalid-return-type]
            prior=self,
            y=y,
            noise_var=jnp.asarray(noise_var),
            smoothed_means=m_smooth,
            smoothed_covs=P_smooth,
            log_marginal=log_marg,
        )

    def condition_nongauss(
        self,
        likelihood: Likelihood,
        y: Float[Array, " N"],
        *,
        strategy: _NonGaussMarkovStrategy,
    ) -> NonGaussConditionedMarkovGP:
        """Condition on a non-Gaussian likelihood via a site-based strategy.

        Convenience that forwards to ``strategy.fit(self, likelihood, y)``.
        Pick any of the Markov-aware site-based strategies in
        :mod:`pyrox.gp._inference_nongauss_markov`:
        :class:`pyrox.gp.LaplaceMarkovInference`,
        :class:`pyrox.gp.GaussNewtonMarkovInference`,
        :class:`pyrox.gp.PosteriorLinearizationMarkov`, or
        :class:`pyrox.gp.ExpectationPropagationMarkov`. Returns a
        :class:`pyrox.gp.NonGaussConditionedMarkovGP` with the same
        ``predict`` API as the Gaussian-likelihood
        :class:`ConditionedMarkovGP`.
        """
        return strategy.fit(self, likelihood, y)

    def log_prob(self, f: Float[Array, " N"]) -> Float[Array, ""]:
        r"""Log density of an exact-state path :math:`f(t_n) = H x_n` under the prior.

        Evaluates ``log N(f | mu(times), K_NN)`` where ``K_NN`` is the dense
        Gram of the kernel encoded by ``sde_kernel`` on ``self.times``.
        Computes the dense covariance via ``H exp(F |t_i - t_j|) P_inf H^T``
        — one ``expm`` per pairwise lag, costing :math:`O(N^2 d^3)` for the
        Gram plus :math:`O(N^3)` for the Cholesky solve — intended for
        sanity checks and small-grid use rather than scalable inference.
        For training, prefer :meth:`log_marginal`.
        """
        F, _L, H, _Qc, P_inf = self.sde_kernel.sde_params()
        diffs = jnp.abs(self.times[:, None] - self.times[None, :])
        # Vectorise H exp(F |dt|) P_inf H^T over the (N, N) lag grid.
        flat_dt = diffs.reshape(-1)

        def _k(tau: Float[Array, ""]) -> Float[Array, ""]:
            return (H @ jax.scipy.linalg.expm(F * tau) @ P_inf @ H.T)[0, 0]

        K_flat = jax.vmap(_k)(flat_dt)
        K = K_flat.reshape(diffs.shape)
        K = 0.5 * (K + K.T)
        n = self.times.shape[0]
        K = K + 1e-8 * jnp.eye(n, dtype=K.dtype)
        residual = f - self.mean(self.times)
        L = jnp.linalg.cholesky(K)
        alpha = jax.scipy.linalg.solve_triangular(L, residual, lower=True)
        log_2pi = jnp.log(2.0 * jnp.pi).astype(K.dtype)
        return (
            -0.5 * (alpha @ alpha) - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * n * log_2pi
        )


class ConditionedMarkovGP(eqx.Module):
    """Markov GP conditioned on Gaussian-likelihood observations.

    Holds the smoothed posterior on the training grid plus the marginal
    log-likelihood. Use :meth:`predict` for marginal posterior mean / variance
    at arbitrary test times.

    Attributes:
        prior: The originating :class:`MarkovGPPrior`.
        y: Observations of shape ``(N,)``.
        noise_var: Observation variance used for conditioning.
        smoothed_means: ``(N, d)`` smoothed state means at training times.
        smoothed_covs: ``(N, d, d)`` smoothed state covariances at training
            times.
        log_marginal: Scalar :math:`\\log p(y \\mid \\theta)`.
    """

    prior: MarkovGPPrior
    y: Float[Array, " N"]
    noise_var: Float[Array, ""]
    smoothed_means: Float[Array, "N d"]
    smoothed_covs: Float[Array, "N d d"]
    log_marginal: Float[Array, ""]

    def predict(
        self,
        t_star: Float[Array, " M"],
    ) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
        r"""Predictive marginals ``(mean, var)`` at arbitrary test times.

        Implementation: re-run the filter+smoother over the merged grid
        ``sort(times \\cup t_star)`` with the test points masked out of the
        update step, then read off the smoothed marginals at the test
        positions via ``H @ m`` and ``H @ P @ H^T``. Cost is
        :math:`O((N + M)\\,d^3)`. Handles training-grid lookups, forecasting,
        backcasting, and within-window interpolation under one code path.
        """
        F, _L, H, _Qc, P_inf = self.prior.sde_kernel.sde_params()
        times = self.prior.times
        t_star = jnp.asarray(t_star)

        N = times.shape[0]
        M = t_star.shape[0]
        merged = jnp.concatenate([times, t_star], axis=0)
        # Stable sort so the relative ordering of identical times is preserved
        # (training point sorts before a duplicate test point, so the test
        # point still sees the observation update earlier in the grid).
        order = jnp.argsort(merged, stable=True)
        merged_sorted = merged[order]

        is_obs = jnp.concatenate(
            [jnp.ones(N, dtype=times.dtype), jnp.zeros(M, dtype=times.dtype)]
        )[order]
        residual_full = jnp.concatenate(
            [self.y - self.prior.mean(times), jnp.zeros(M, dtype=self.y.dtype)]
        )[order]

        dt_full = _build_dt_full(merged_sorted)
        A_seq, Q_seq = self.prior.sde_kernel.discretise(dt_full)
        R_seq = jnp.full_like(merged_sorted, self.prior._R(self.noise_var))
        m_pred, P_pred, m_filt, P_filt, _ = _kalman_filter(
            F, H, P_inf, A_seq, Q_seq, residual_full, is_obs, R_seq
        )
        m_smooth, P_smooth = _rts_smoother(m_pred, P_pred, m_filt, P_filt, A_seq)

        # Inverse permutation: position in the sorted grid for each original
        # entry, then slice off the trailing M test entries.
        inv_order = jnp.argsort(order, stable=True)
        test_positions = inv_order[N:]
        m_test_state = m_smooth[test_positions]  # (M, d)
        P_test_state = P_smooth[test_positions]  # (M, d, d)
        means = (m_test_state @ H.T)[:, 0] + self.prior.mean(t_star)
        # var = H P H^T per test point — vmap over axis 0
        vars_ = jax.vmap(lambda P: (H @ P @ H.T)[0, 0])(P_test_state)
        return means, vars_


def markov_gp_factor(
    name: str,
    prior: MarkovGPPrior,
    y: Float[Array, " N"],
    noise_var: Float[Array, ""],
) -> None:
    """Register the collapsed Markov-GP marginal log-likelihood with NumPyro.

    Computes ``log p(y | times, theta)`` via Kalman filtering and adds it as
    ``numpyro.factor(name, ...)``. Use this inside a NumPyro model for
    Gaussian-likelihood temporal GP regression — the latent function is
    marginalized analytically.
    """
    numpyro.factor(name, prior.log_marginal(y, noise_var))


def markov_gp_sample(
    name: str,
    prior: MarkovGPPrior,
) -> Float[Array, " N"]:
    """Sample a latent function ``f`` at the prior's training times.

    Registers a single ``numpyro.sample(name, MVN(mu, K))`` site where ``K``
    is the dense Gram derived from the SDE autocovariance
    ``H exp(F|tau|) P_inf H^T``. This is the simple, dense path — use it
    when ``N`` is small. Scalable Markov-aware sample sites land in a
    later wave alongside non-Gaussian likelihood support.
    """
    F, _L, H, _Qc, P_inf = prior.sde_kernel.sde_params()
    times = prior.times
    diffs = jnp.abs(times[:, None] - times[None, :])
    flat_dt = diffs.reshape(-1)

    def _k(tau: Float[Array, ""]) -> Float[Array, ""]:
        return (H @ jax.scipy.linalg.expm(F * tau) @ P_inf @ H.T)[0, 0]

    K = jax.vmap(_k)(flat_dt).reshape(diffs.shape)
    K = 0.5 * (K + K.T)
    n = times.shape[0]
    K = K + 1e-8 * jnp.eye(n, dtype=K.dtype)
    mu = prior.mean(times)
    return numpyro.sample(  # ty: ignore[invalid-return-type]
        name, dist.MultivariateNormal(mu, covariance_matrix=K)
    )
