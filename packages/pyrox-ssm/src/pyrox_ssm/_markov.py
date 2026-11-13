"""Kalman-based Markov GP prior on top of the SDE-kernel surface.

Stationary 1-D GP kernels with rational spectral densities admit exact
linear-Gaussian state-space representations (see `gaussx._ssm`
and the `pyrox_ssm.SDEKernel` protocol). This module turns those
representations into a working temporal-GP model: forward Kalman filter for
the marginal log-likelihood, backward RTS smoother for the posterior, and a
NumPyro-aware shell so the path drops into models the same way
`pyrox_gp.GPPrior` does.

For a sorted observation grid of length $N$ and SDE state dimension
$d$, both the marginal likelihood and the smoothed posterior cost
$O(N\\,d^3)$ — linear in $N$, where the dense path is
$O(N^3)$.

Test-time predictions for arbitrary ``t_star`` are produced by re-running the
filter+smoother over the merged grid ``sort(times \\cup t_star)`` with the
unobserved test points masked out of the update step. This handles
training-grid lookups, forecasting (``t_star`` after the data), backcasting
(``t_star`` before the data), and within-window interpolation under one code
path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Protocol

import einx
import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
import lineax as lx
import numpyro
from gaussx import FilterState, SDEKernel
from jaxtyping import Array, Bool, Float
from numpyro import distributions as dist
from pyrox.likelihoods import Likelihood

from pyrox_ssm._lgssm import LinearGaussianSSMPrior


if TYPE_CHECKING:
    from pyrox_ssm._inference_nongauss_markov import NonGaussConditionedMarkovGP


class _NonGaussMarkovStrategy(Protocol):
    """Structural protocol for non-Gaussian Markov inference strategies."""

    def fit(
        self,
        prior: MarkovGPPrior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
        *,
        mask: Bool[Array, " N"] | None = None,
    ) -> NonGaussConditionedMarkovGP: ...


def _kalman_filter(
    F: Float[Array, "d d"],
    H: Float[Array, "1 d"],
    init_cov: Float[Array, "d d"],
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
    enters the state). ``R_seq`` is the per-step observation variance.

    Delegates to `pyrox_ssm.LinearGaussianSSMPrior.filter`, the one
    `gaussx.kalman_filter` call site, with the scalar observation lifted to
    ``(N, 1)`` and a zero initial mean. ``F`` is unused, since the predict
    comes from the discretised ``A_seq``; it is kept so the call sites and
    the existing test signature don't shift.
    """
    del F  # the predict comes from A_seq directly
    # Lift the per-step noise to the (N, 1, 1) stack of a scalar observation
    # model; the (N,) residual is lifted to (N, 1) by the prior.
    R_3d = einx.id("n -> n 1 1", R_seq)
    init_mean = jnp.zeros(init_cov.shape[0], dtype=init_cov.dtype)
    lgssm = LinearGaussianSSMPrior(
        transition=A_seq,
        obs_model=H,
        process_noise=Q_seq,
        obs_noise=R_3d,
        init_mean=init_mean,
        init_cov=init_cov,
    )
    return _filter_tuple(lgssm.filter(residual, mask=mask.astype(bool)))


def _filter_tuple(
    state: FilterState,
) -> tuple[
    Float[Array, "N d"],
    Float[Array, "N d d"],
    Float[Array, "N d"],
    Float[Array, "N d d"],
    Float[Array, ""],
]:
    """Unpack a `gaussx.FilterState` into the tuple the Markov helpers use."""
    return (
        state.predicted_means,
        state.predicted_covs,
        state.filtered_means,
        state.filtered_covs,
        state.log_likelihood,
    )


def _rts_smoother(
    m_pred_seq: Float[Array, "N d"],
    P_pred_seq: Float[Array, "N d d"],
    m_filt_seq: Float[Array, "N d"],
    P_filt_seq: Float[Array, "N d d"],
    A_seq: Float[Array, "N d d"],
) -> tuple[Float[Array, "N d"], Float[Array, "N d d"]]:
    """Backward Rauch-Tung-Striebel smoother.

    Delegates to `gaussx.rts_smoother`. Reconstructs a
    `gaussx.FilterState` from the unpacked per-step arrays so call
    sites that wire the filter and smoother together (passing tuples)
    don't have to change shape.
    """
    filter_state: FilterState = FilterState(
        filtered_means=m_filt_seq,
        filtered_covs=P_filt_seq,
        predicted_means=m_pred_seq,
        predicted_covs=P_pred_seq,
        log_likelihood=jnp.zeros((), dtype=m_filt_seq.dtype),
    )
    return gaussx.rts_smoother(filter_state, A_seq)


def _require_stationary(
    sde_kernel: SDEKernel,
    P_inf: Float[Array, "d d"] | None,
) -> Float[Array, "d d"]:
    r"""Narrow ``sde_params().P_inf`` to an array, or raise.

    `gaussx.SDEParams` types ``P_inf`` as optional for two different
    reasons, and the message says which applies. Some compositions have
    no *closed form* for the stationary covariance (a
    `gaussx.ProductSDE` over a factor without one) and fall back to
    matrix-fraction discretisation; a non-stationary kernel such as
    `gaussx.IntegratedWienerSDE` has no stationary covariance at all.

    Used by the surfaces that genuinely need $P_\infty$ — the
    stationary marginal variance and the dense Gram. The Kalman
    recursions go through `MarkovGPPrior.initial_covariance` instead,
    which a non-stationary kernel can satisfy.
    """
    if P_inf is None:
        if not _is_stationary(sde_kernel):
            raise ValueError(
                f"{type(sde_kernel).__name__} is not stationary, so it has "
                "no stationary covariance P_inf. Filtering and smoothing "
                "work — they seed from MarkovGPPrior.initial_covariance() "
                "— but this path is defined in terms of P_inf and has no "
                "meaning without it."
            )
        raise ValueError(
            f"{type(sde_kernel).__name__} reports no stationary covariance "
            "(sde_params().P_inf is None), so it cannot seed a Markov GP "
            "prior. Use a stationary SDE kernel, or one whose factors all "
            "define P_inf."
        )
    return P_inf


def _is_stationary(sde_kernel: SDEKernel) -> bool:
    """Whether ``sde_kernel`` declares itself stationary.

    ``stationary`` arrived on the gaussx kernel protocol alongside
    `gaussx.IntegratedWienerSDE`; a hand-rolled kernel predating it is
    read as stationary, which is what every kernel was before.
    """
    return bool(getattr(sde_kernel, "stationary", True))


def _kernel_initial_covariance(
    sde_kernel: SDEKernel,
) -> Float[Array, "d d"]:
    r"""Ask the kernel where the filter should start.

    `gaussx.SDEKernel.initial_covariance` returns $P_\infty$ for a
    stationary kernel — today's behaviour, unchanged — and an explicit
    (diffuse by default) covariance for a non-stationary one. Falls back
    to reading ``P_inf`` for a kernel that predates the method.
    """
    initial_covariance = getattr(sde_kernel, "initial_covariance", None)
    if initial_covariance is None:
        return _require_stationary(sde_kernel, sde_kernel.sde_params().P_inf)
    return initial_covariance()


def _observation_mask(
    mask: Bool[Array, " N"] | None, times: Float[Array, " N"]
) -> Bool[Array, " N"]:
    """The per-step observation mask, all ``True`` when ``mask`` is ``None``.

    Args:
        mask: ``True`` where a step is observed, or ``None`` for every step.
        times: The grid the mask is laid over, which sets its shape.

    Returns:
        A boolean mask shaped like ``times``.

    Raises:
        ValueError: If ``mask`` is not shaped like ``times``.
    """
    if mask is None:
        return jnp.ones(times.shape, dtype=bool)
    mask = jnp.asarray(mask).astype(bool)
    if mask.shape != times.shape:
        raise ValueError(
            f"mask has shape {tuple(mask.shape)!r}, but the observations are "
            f"laid over times of shape {tuple(times.shape)!r}; the mask marks "
            "which steps are observed, so the two must match."
        )
    return mask


def _sanitise(
    y: Float[Array, " N"], mask: Bool[Array, " N"] | None
) -> Float[Array, " N"]:
    """Zero the masked entries of ``y``.

    A ``NaN`` placeholder there then never reaches a value or a gradient.
    """
    if mask is None:
        return y
    return jnp.where(_observation_mask(mask, y), y, jnp.zeros((), dtype=y.dtype))


def _build_dt_full(times: Float[Array, " N"]) -> Float[Array, " N"]:
    """Pad ``diff(times)`` with a leading zero so step 0 is the prior."""
    dt = jnp.diff(times)
    return jnp.concatenate([jnp.zeros((1,), dtype=dt.dtype), dt], axis=0)


class MarkovGPPrior(eqx.Module):
    r"""Linear-time temporal GP prior over a sorted 1-D grid.

    Wraps any `pyrox_ssm.SDEKernel` (e.g. `pyrox_ssm.MaternSDE`,
    `pyrox_ssm.SumSDE`, `pyrox_ssm.PeriodicSDE`) to give Kalman
    filtering for the marginal log-likelihood and RTS smoothing for the
    posterior on the training grid. Supports an optional mean function and a
    small observation-noise floor for numerical stability.

    Attributes:
        sde_kernel: Any `SDEKernel`. Provides ``(F, L, H, Q_c, P_inf)``
            via ``sde_params()`` and the discrete transition sequence via
            ``discretise_sequence(dt)``.
        times: Sorted, strictly increasing observation times of shape
            ``(N,)``. Concrete (non-traced) ``times`` arrays are validated
            for monotonicity at construction time; under `jax.jit` /
            SVI / MCMC the input is a tracer and the check is silently
            skipped — callers must guarantee monotonicity in that case.
        mean_fn: Optional callable mapping ``times -> (N,)`` mean values.
            Defaults to the zero mean. The mean is subtracted from
            observations before filtering and added back at predict time.
        obs_noise_floor: Small extra diagonal added to the observation
            variance ``R = noise_var + obs_noise_floor`` for stability when
            ``noise_var`` is near zero. Defaults to ``0.0``.
        init_cov: Optional ``(d, d)`` covariance of the state at the first
            time point. ``None`` (the default) asks the kernel via
            ``initial_covariance()``, which is $P_\infty$ for a stationary
            kernel — so nothing changes for the kernels that have one — and
            an explicit diffuse covariance for a non-stationary one such as
            `gaussx.IntegratedWienerSDE`. Supply it to state what is known
            about the level (and its derivatives) before the first
            observation. Worth supplying for the site-based non-Gaussian
            strategies in particular: `pyrox_ssm.PosteriorLinearizationMarkov`
            and `pyrox_ssm.ExpectationPropagationMarkov` seed their cavities
            from the prior marginal variance, and a very diffuse prior can
            overflow their quadrature under an exponential-link likelihood
            — as a stationary kernel with an equally large variance does.

    Every method that reads observations (`filter`, `smooth`,
    `log_marginal`, `condition`, `condition_nongauss`) takes an optional
    ``mask`` of shape ``(N,)``. A masked step runs the Kalman predict only,
    so it adds nothing to the log marginal, and its observation is replaced
    by zero before any arithmetic, so a ``NaN`` placeholder there is safe.
    `pyrox_ssm.pad_sites` builds such masks for ragged per-site series.

    Examples:
        >>> import jax.numpy as jnp
        >>> from pyrox_ssm import MaternSDE, MarkovGPPrior
        >>> times = jnp.linspace(0.0, 5.0, 50)
        >>> sde = MaternSDE(variance=1.0, lengthscale=0.5, order=1)
        >>> prior = MarkovGPPrior(sde, times)
        >>> y = jnp.sin(times) + 0.05 * jnp.cos(3.0 * times)
        >>> log_marg = prior.log_marginal(y, jnp.asarray(0.01))
        >>> mask = jnp.arange(50) % 5 != 0  # every fifth step unobserved
        >>> log_marg_masked = prior.log_marginal(y, jnp.asarray(0.01), mask=mask)

    Notes:
        The solver-strategy plumbing used by `pyrox_gp.GPPrior` does
        not apply here — Kalman filtering is its own linear-algebra path
        and does not factor through ``gaussx.AbstractSolverStrategy``.
    """

    sde_kernel: SDEKernel
    times: Float[Array, " N"]
    mean_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]] | None = None
    obs_noise_floor: float = eqx.field(static=True, default=0.0)
    init_cov: Float[Array, "d d"] | None = None

    def __init__(
        self,
        sde_kernel: SDEKernel,
        times: Float[Array, " N"],
        mean_fn: Callable[[Float[Array, " N"]], Float[Array, " N"]] | None = None,
        obs_noise_floor: float = 0.0,
        init_cov: Float[Array, "d d"] | None = None,
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
        if init_cov is not None:
            # Promoted like ``times``: an integer covariance would seed the
            # scan with an int carry and fail inside it on a dtype mismatch.
            init_cov = jnp.asarray(init_cov, dtype=jnp.result_type(init_cov, 0.0))
            d = sde_kernel.state_dim
            if init_cov.shape != (d, d):
                raise ValueError(
                    f"init_cov must be ({d}, {d}) for a state dimension of "
                    f"{d}, got shape {tuple(init_cov.shape)!r}"
                )
        self.sde_kernel = sde_kernel
        self.times = times_arr
        self.mean_fn = mean_fn
        self.obs_noise_floor = float(obs_noise_floor)
        self.init_cov = init_cov

    @property
    def state_dim(self) -> int:
        """SDE state dimension $d$ for this kernel."""
        return self.sde_kernel.state_dim

    def initial_covariance(self) -> Float[Array, "d d"]:
        r"""Covariance the Kalman recursions start from.

        The ``init_cov`` given at construction if there is one, otherwise
        whatever the kernel reports — $P_\infty$ for a stationary kernel,
        an explicit diffuse covariance for a non-stationary one. This is
        the single seed every filtering surface here uses, so a kernel
        with no $P_\infty$ is only a problem for the paths that are
        *defined* in terms of it (`log_prob`, the dense Gram).
        """
        if self.init_cov is not None:
            return self.init_cov
        return _kernel_initial_covariance(self.sde_kernel)

    def mean(self, times: Float[Array, " M"]) -> Float[Array, " M"]:
        """Evaluate the mean function at ``times``; zero by default."""
        if self.mean_fn is None:
            return jnp.zeros_like(times)
        return self.mean_fn(times)

    def _R(self, noise_var: Float[Array, ""]) -> Float[Array, ""]:
        return jnp.asarray(noise_var) + jnp.asarray(self.obs_noise_floor)

    def _observed(self, mask: Bool[Array, " N"] | None) -> Bool[Array, " N"] | None:
        """``mask`` checked against ``times``, or ``None`` when not given."""
        return None if mask is None else _observation_mask(mask, self.times)

    def as_lgssm(self, noise_var: Float[Array, ""]) -> LinearGaussianSSMPrior:
        r"""This prior as a `pyrox_ssm.LinearGaussianSSMPrior`.

        With $\Delta t_1 = 0$ and $\Delta t_k = t_k - t_{k-1}$, the
        kernel's discretisation gives the transitions and process noise,

        $$
        (A_k, Q_k) = \mathrm{discretise}(\Delta t_k), \qquad
        H_k = H, \qquad R_k = \sigma_n^2 + \epsilon, \qquad
        b_k = \mu(t_k), \qquad (m_0, P_0) = (0, P_{\mathrm{init}}),
        $$

        where $\epsilon$ is ``obs_noise_floor``, $\mu$ the mean function
        and $P_{\mathrm{init}}$ `initial_covariance` ($P_\infty$ for a
        stationary kernel). The zero first step makes $A_1 = I$ and
        $Q_1 = 0$, so the first observation sees the initial state itself.
        `filter`, `smooth` and `log_marginal` run through this model, so
        the two give the same results bit for bit.

        Args:
            noise_var: Observation noise variance $\sigma_n^2$.

        Returns:
            The `pyrox_ssm.LinearGaussianSSMPrior` over ``times``, with
            $T = N$ steps, state dimension $d$ and one output.

        Examples:
            >>> import jax.numpy as jnp
            >>> from pyrox_ssm import MaternSDE, MarkovGPPrior
            >>> prior = MarkovGPPrior(MaternSDE(1.0, 0.5, order=1), jnp.arange(4.0))
            >>> lgssm = prior.as_lgssm(jnp.asarray(0.1))
            >>> lgssm.transition.shape, lgssm.obs_model.shape
            ((4, 2, 2), (1, 2))
            >>> y = jnp.array([0.1, 0.4, 0.3, 0.2])
            >>> log_marg = lgssm.log_marginal(y)  # == prior.log_marginal(y, 0.1)
        """
        _F, _L, H, _Qc, _P_inf = self.sde_kernel.sde_params()
        init_cov = self.initial_covariance()
        A_seq, Q_seq = self.sde_kernel.discretise_sequence(_build_dt_full(self.times))
        R_seq = jnp.broadcast_to(self._R(noise_var), self.times.shape)
        return LinearGaussianSSMPrior(
            transition=A_seq,
            obs_model=H,
            process_noise=Q_seq,
            obs_noise=einx.id("n -> n 1 1", R_seq),
            init_mean=jnp.zeros(init_cov.shape[0], dtype=init_cov.dtype),
            init_cov=init_cov,
            obs_offset=einx.id("n -> n 1", self.mean(self.times)),
        )

    def filter(
        self,
        y: Float[Array, " N"],
        noise_var: Float[Array, ""],
        *,
        mask: Bool[Array, " N"] | None = None,
    ) -> tuple[
        Float[Array, "N d"],
        Float[Array, "N d d"],
        Float[Array, "N d"],
        Float[Array, "N d d"],
        Float[Array, ""],
    ]:
        r"""Run the forward Kalman filter on the training grid.

        With $o_k \in \{0, 1\}$ the mask, an observed step ($o_k = 1$)
        runs the usual update and a masked step ($o_k = 0$) runs the
        predict only,

        $$
        \mu_k^+ = \mu_k^-, \qquad P_k^+ = P_k^-, \qquad
        \log p(y \mid \theta)
          = \sum_k o_k \log \mathcal{N}(y_k;\, H \mu_k^-,\, H P_k^- H^\top + R).
        $$

        Args:
            y: Observations of shape ``(N,)``. Masked entries are replaced
                by zero before the residual, so they may be ``NaN``.
            noise_var: Observation noise variance $\sigma_n^2$.
            mask: Optional ``(N,)`` boolean mask, ``True`` where a step is
                observed. ``None`` observes every step.

        Returns:
            Tuple ``(m_pred, P_pred, m_filt, P_filt, log_marginal)`` where
            each ``*_pred`` / ``*_filt`` is shaped ``(N, d)`` or
            ``(N, d, d)`` and ``log_marginal`` is the scalar log-likelihood
            ``log p(y | theta)`` of the observed steps.

        Raises:
            ValueError: If ``mask`` is not shaped like ``times``.

        Examples:
            >>> import jax.numpy as jnp
            >>> from pyrox_ssm import MaternSDE, MarkovGPPrior
            >>> prior = MarkovGPPrior(MaternSDE(1.0, 0.5, order=1), jnp.arange(4.0))
            >>> y = jnp.array([0.1, jnp.nan, 0.3, 0.2])
            >>> mask = jnp.array([True, False, True, True])
            >>> *_, log_marg = prior.filter(y, jnp.asarray(0.1), mask=mask)
        """
        lgssm = self.as_lgssm(noise_var)
        return _filter_tuple(lgssm.filter(y, mask=self._observed(mask)))

    def log_marginal(
        self,
        y: Float[Array, " N"],
        noise_var: Float[Array, ""],
        *,
        mask: Bool[Array, " N"] | None = None,
    ) -> Float[Array, ""]:
        r"""Marginal log-likelihood ``log p(y | theta)`` via Kalman filtering.

        A masked step contributes nothing,

        $$
        \log p(y \mid \theta)
          = \sum_k o_k \log \mathcal{N}(y_k;\, H \mu_k^-,\, H P_k^- H^\top + R),
        $$

        so padding a series after its last observation and masking the
        padded steps leaves the result unchanged.

        Args:
            y: Observations of shape ``(N,)``; masked entries may be ``NaN``.
            noise_var: Observation noise variance $\sigma_n^2$.
            mask: Optional ``(N,)`` boolean mask, ``True`` where observed.

        Returns:
            The scalar log marginal of the observed steps.

        Examples:
            >>> import jax.numpy as jnp
            >>> from pyrox_ssm import MaternSDE, MarkovGPPrior
            >>> prior = MarkovGPPrior(MaternSDE(1.0, 0.5, order=1), jnp.arange(4.0))
            >>> y = jnp.array([0.1, 0.4, 0.3, jnp.nan])
            >>> mask = jnp.array([True, True, True, False])
            >>> log_marg = prior.log_marginal(y, jnp.asarray(0.1), mask=mask)
        """
        lgssm = self.as_lgssm(noise_var)
        return lgssm.log_marginal(y, mask=self._observed(mask))

    def smooth(
        self,
        y: Float[Array, " N"],
        noise_var: Float[Array, ""],
        *,
        mask: Bool[Array, " N"] | None = None,
    ) -> tuple[Float[Array, "N d"], Float[Array, "N d d"], Float[Array, ""]]:
        r"""Run filter + RTS smoother on the training grid.

        Masked steps run the predict only (see `filter`), so their smoothed
        moments are interpolations from the observed neighbours.

        Args:
            y: Observations of shape ``(N,)``; masked entries may be ``NaN``.
            noise_var: Observation noise variance $\sigma_n^2$.
            mask: Optional ``(N,)`` boolean mask, ``True`` where observed.

        Returns:
            ``(m_smooth, P_smooth, log_marginal)`` over the training times,
            shaped ``(N, d)``, ``(N, d, d)`` and ``()``.

        Examples:
            >>> import jax.numpy as jnp
            >>> from pyrox_ssm import MaternSDE, MarkovGPPrior
            >>> prior = MarkovGPPrior(MaternSDE(1.0, 0.5, order=1), jnp.arange(4.0))
            >>> mask = jnp.array([True, False, True, True])
            >>> m, P, _ = prior.smooth(jnp.ones(4), jnp.asarray(0.1), mask=mask)
            >>> m.shape, P.shape
            ((4, 2), (4, 2, 2))
        """
        lgssm = self.as_lgssm(noise_var)
        return lgssm.smooth(y, mask=self._observed(mask))

    def condition(
        self,
        y: Float[Array, " N"],
        noise_var: Float[Array, ""],
        *,
        mask: Bool[Array, " N"] | None = None,
    ) -> ConditionedMarkovGP:
        """Condition on Gaussian-likelihood observations via filter + smoother.

        Args:
            y: Observations of shape ``(N,)``; masked entries may be ``NaN``.
            noise_var: Observation noise variance.
            mask: Optional ``(N,)`` boolean mask, ``True`` where observed. The
                conditioned GP keeps it, so `ConditionedMarkovGP.predict`
                treats the masked steps as unobserved too.

        Returns:
            The `ConditionedMarkovGP`, holding ``y`` with its masked entries
            set to zero.

        Examples:
            >>> import jax.numpy as jnp
            >>> from pyrox_ssm import MaternSDE, MarkovGPPrior
            >>> prior = MarkovGPPrior(MaternSDE(1.0, 0.5, order=1), jnp.arange(4.0))
            >>> mask = jnp.array([True, False, True, True])
            >>> cond = prior.condition(jnp.ones(4), jnp.asarray(0.1), mask=mask)
            >>> mean, var = cond.predict(jnp.array([1.0, 5.0]))
        """
        m_smooth, P_smooth, log_marg = self.smooth(y, noise_var, mask=mask)
        return ConditionedMarkovGP(
            prior=self,
            y=_sanitise(y, mask),
            noise_var=jnp.asarray(noise_var),
            smoothed_means=m_smooth,
            smoothed_covs=P_smooth,
            log_marginal=log_marg,
            mask=None if mask is None else _observation_mask(mask, self.times),
        )

    def condition_nongauss(
        self,
        likelihood: Likelihood,
        y: Float[Array, " N"],
        *,
        strategy: _NonGaussMarkovStrategy,
        mask: Bool[Array, " N"] | None = None,
    ) -> NonGaussConditionedMarkovGP:
        """Condition on a non-Gaussian likelihood via a site-based strategy.

        Convenience that forwards to ``strategy.fit(self, likelihood, y)``.
        Pick any of the Markov-aware site-based strategies in
        `pyrox_ssm._inference_nongauss_markov`:
        `pyrox_ssm.LaplaceMarkovInference`,
        `pyrox_ssm.GaussNewtonMarkovInference`,
        `pyrox_ssm.PosteriorLinearizationMarkov`, or
        `pyrox_ssm.ExpectationPropagationMarkov`. Returns a
        `pyrox_ssm.NonGaussConditionedMarkovGP` with the same
        ``predict`` API as the Gaussian-likelihood
        `ConditionedMarkovGP`.

        Args:
            likelihood: Any scalar-latent `pyrox.likelihoods.Likelihood`.
            y: Observations of shape ``(N,)``; masked entries may be ``NaN``.
            strategy: The site-based strategy that fits the sites.
            mask: Optional ``(N,)`` boolean mask, ``True`` where observed. A
                masked site has zero gradient and curvature, so its site
                precision is zero and the filter skips its update. Passed
                to ``strategy.fit`` only when given, so a strategy without
                a ``mask`` keyword still works unmasked.

        Returns:
            The `pyrox_ssm.NonGaussConditionedMarkovGP` from the strategy.

        Examples:
            >>> import jax.numpy as jnp
            >>> from pyrox_gp import BernoulliLikelihood
            >>> from pyrox_ssm import LaplaceMarkovInference, MaternSDE, MarkovGPPrior
            >>> prior = MarkovGPPrior(MaternSDE(1.0, 0.5, order=1), jnp.arange(4.0))
            >>> y = jnp.array([1.0, 0.0, jnp.nan, 1.0])
            >>> cond = prior.condition_nongauss(
            ...     BernoulliLikelihood(), y,
            ...     strategy=LaplaceMarkovInference(),
            ...     mask=jnp.array([True, True, False, True]),
            ... )
        """
        if mask is None:
            return strategy.fit(self, likelihood, y)
        return strategy.fit(self, likelihood, y, mask=mask)

    def log_prob(self, f: Float[Array, " N"]) -> Float[Array, ""]:
        r"""Log density of an exact-state path $f(t_n) = H x_n$ under the prior.

        Evaluates ``log N(f | mu(times), K_NN)`` where ``K_NN`` is the dense
        Gram of the kernel encoded by ``sde_kernel`` on ``self.times``.
        Computes the dense covariance via ``H exp(F |t_i - t_j|) P_inf H^T``
        — one ``expm`` per pairwise lag, costing $O(N^2 d^3)$ for the
        Gram plus $O(N^3)$ for the Cholesky solve — intended for
        sanity checks and small-grid use rather than scalable inference.
        For training, prefer `log_marginal`.
        """
        K = _dense_sde_gram(self.sde_kernel, self.times)
        cov_op = lx.MatrixLinearOperator(K, lx.positive_semidefinite_tag)
        return gaussx.gaussian_log_prob(self.mean(self.times), cov_op, f)


def _dense_sde_gram(
    sde_kernel: SDEKernel, times: Float[Array, " N"]
) -> Float[Array, "N N"]:
    r"""Dense Gram ``K_ij = H exp(F |t_i - t_j|) P_inf H^T`` with diagonal jitter.

    One ``expm`` per pairwise lag — $O(N^2 d^3)$. Intended for the
    small-``N`` dense paths (`MarkovGPPrior.log_prob`,
    `markov_gp_sample`); scalable inference goes through the Kalman
    filter instead.
    """
    if not _is_stationary(sde_kernel):
        raise ValueError(
            f"{type(sde_kernel).__name__} is not stationary, so it has no "
            "dense Gram of this form: K_ij = H exp(F |t_i - t_j|) P_inf H^T "
            "is a function of the lag alone, while a non-stationary "
            "covariance depends on both times (an integrated Wiener "
            "process has Cov(f(s), f(t)) = q (min(s,t)^3/3 + |t-s| "
            "min(s,t)^2/2)). Use log_marginal / condition, which filter "
            "rather than form a Gram."
        )
    F, _L, H, _Qc, P_inf = sde_kernel.sde_params()
    P_inf = _require_stationary(sde_kernel, P_inf)
    # Pairwise absolute time lags |tᵢ - tⱼ| as an (N, N) grid.
    diffs = jnp.abs(einx.subtract("i, j -> i j", times, times))
    # Vectorise H exp(F |dt|) P_inf H^T over the (N, N) lag grid.
    flat_dt = einx.id("i j -> (i j)", diffs)

    def _k(tau: Float[Array, ""]) -> Float[Array, ""]:
        return (H @ jax.scipy.linalg.expm(F * tau) @ P_inf @ H.T)[0, 0]

    K_flat = jax.vmap(_k)(flat_dt)
    # einx.id is typed as possibly returning a tuple (multi-output
    # patterns); narrow back to a single array for the typechecker.
    K_grid = jnp.asarray(einx.id("(i j) -> i j", K_flat, i=times.shape[0]))
    K = gaussx.symmetrize(K_grid)
    return K.at[jnp.diag_indices_from(K)].add(1e-8)


class ConditionedMarkovGP(eqx.Module):
    """Markov GP conditioned on Gaussian-likelihood observations.

    Holds the smoothed posterior on the training grid plus the marginal
    log-likelihood. Use `predict` for marginal posterior mean / variance
    at arbitrary test times.

    Attributes:
        prior: The originating `MarkovGPPrior`.
        y: Observations of shape ``(N,)``.
        noise_var: Observation variance used for conditioning.
        smoothed_means: ``(N, d)`` smoothed state means at training times.
        smoothed_covs: ``(N, d, d)`` smoothed state covariances at training
            times.
        log_marginal: Scalar $\\log p(y \\mid \\theta)$.
        mask: Optional ``(N,)`` boolean observation mask from
            `MarkovGPPrior.condition`; `predict` leaves the masked steps
            out of the update. ``None`` means every step is observed.
    """

    prior: MarkovGPPrior
    y: Float[Array, " N"]
    noise_var: Float[Array, ""]
    smoothed_means: Float[Array, "N d"]
    smoothed_covs: Float[Array, "N d d"]
    log_marginal: Float[Array, ""]
    mask: Bool[Array, " N"] | None = None

    def predict(
        self,
        t_star: Float[Array, " M"],
        *,
        include_noise: bool = False,
    ) -> tuple[Float[Array, " M"], Float[Array, " M"]]:
        r"""Predictive marginals ``(mean, var)`` at arbitrary test times.

        With ``include_noise=False`` (the default) ``var`` is the variance of
        the latent signal, $\sigma_*^2$. With ``include_noise=True`` it is the
        variance of a new observation at each test time,
        $\sigma_*^2 + R$, where $R$ = ``noise_var + obs_noise_floor`` is the
        observation variance the posterior was conditioned on. Use it for the
        band a withheld observation should fall in. The mean is the same
        either way.

        Implementation: re-run the filter+smoother over the merged grid
        ``sort(times \\cup t_star)`` with the test points masked out of the
        update step, then read off the smoothed marginals at the test
        positions via ``H @ m`` and ``H @ P @ H^T``. Cost is
        $O((N + M)\\,d^3)$. Handles training-grid lookups, forecasting,
        backcasting, and within-window interpolation under one code path.
        """
        F, _L, H, _Qc, _P_inf = self.prior.sde_kernel.sde_params()
        init_cov = self.prior.initial_covariance()
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

        observed = (
            jnp.ones(N, dtype=times.dtype)
            if self.mask is None
            else self.mask.astype(times.dtype)
        )
        is_obs = jnp.concatenate([observed, jnp.zeros(M, dtype=times.dtype)])[order]
        residual_full = jnp.concatenate(
            [self.y - self.prior.mean(times), jnp.zeros(M, dtype=self.y.dtype)]
        )[order]

        dt_full = _build_dt_full(merged_sorted)
        A_seq, Q_seq = self.prior.sde_kernel.discretise_sequence(dt_full)
        R_seq = jnp.broadcast_to(self.prior._R(self.noise_var), merged_sorted.shape)
        m_pred, P_pred, m_filt, P_filt, _ = _kalman_filter(
            F, H, init_cov, A_seq, Q_seq, residual_full, is_obs, R_seq
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
        if include_noise:
            vars_ = vars_ + self.prior._R(self.noise_var)
        return means, vars_


def markov_gp_factor(
    name: str,
    prior: MarkovGPPrior,
    y: Float[Array, " N"],
    noise_var: Float[Array, ""],
    *,
    mask: Bool[Array, " N"] | None = None,
) -> None:
    r"""Register the collapsed Markov-GP marginal log-likelihood with NumPyro.

    Computes ``log p(y | times, theta)`` via Kalman filtering and adds it as
    ``numpyro.factor(name, ...)``. Use this inside a NumPyro model for
    Gaussian-likelihood temporal GP regression — the latent function is
    marginalized analytically. With a mask, only the observed steps score,

    $$
    \log p(y \mid \theta)
      = \sum_k o_k \log \mathcal{N}(y_k;\, H \mu_k^-,\, H P_k^- H^\top + R).
    $$

    Args:
        name: The factor site name.
        prior: The `MarkovGPPrior` over the observation times.
        y: Observations of shape ``(N,)``; masked entries may be ``NaN``.
        noise_var: Observation noise variance.
        mask: Optional ``(N,)`` boolean mask, ``True`` where observed.

    Examples:
        One factor per site over padded ragged series:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> import pyrox_ssm as pss
        >>> t, y, mask = pss.pad_sites(
        ...     [jnp.arange(5.0), jnp.arange(3.0)], [jnp.ones(5), jnp.zeros(3)]
        ... )
        >>> def model():
        ...     with numpyro.plate("site", 2):
        ...         ell = numpyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))
        ...     def site(ell_s, t_s, y_s, m_s):
        ...         prior = pss.MarkovGPPrior(pss.MaternSDE(1.0, ell_s, order=1), t_s)
        ...         return prior.log_marginal(y_s, jnp.asarray(0.1), mask=m_s)
        ...     numpyro.factor("y", jax.vmap(site)(ell, t, y, mask).sum())
    """
    numpyro.factor(name, prior.log_marginal(y, noise_var, mask=mask))


def markov_gp_sample(
    name: str,
    prior: MarkovGPPrior,
) -> Float[Array, " N"]:
    """Sample a latent function ``f`` at the prior's training times.

    Registers a single ``numpyro.sample(name, MVN(mu, K))`` site where ``K``
    is the dense Gram derived from the SDE autocovariance
    ``H exp(F|tau|) P_inf H^T``. This is the simple, dense path: building
    ``K`` costs $O(N^2 d^3)$ and every log-density evaluation an $O(N^3)$
    Cholesky, so use it when ``N`` is a few hundred at most and the kernel
    is stationary. For longer series, a non-stationary kernel, or NUTS over
    a non-Gaussian likelihood, use `markov_gp_innovations_sample`, which
    builds the same path from whitened innovations in $O(N d^3)$.
    """
    K = _dense_sde_gram(prior.sde_kernel, prior.times)
    mu = prior.mean(prior.times)
    return numpyro.sample(  # ty: ignore[invalid-return-type]
        name, dist.MultivariateNormal(mu, covariance_matrix=K)
    )


_SqrtMethod = Literal["cholesky", "eigh"]
_SQRT_METHODS: tuple[str, ...] = ("cholesky", "eigh")


def _covariance_root(
    cov: Float[Array, "d d"], *, sqrt: _SqrtMethod, jitter: float
) -> Float[Array, "d d"]:
    r"""A square root $L$ of one state covariance, $L L^\top = \Sigma$.

    ``"cholesky"`` factors $\Sigma + \beta I$ with `gaussx.cholesky`, so
    $L L^\top = \Sigma + \beta I$; ``"eigh"`` takes the symmetric root
    $Q \operatorname{diag}(\sqrt{\max(\gamma, 0)}) Q^\top$ with
    `gaussx.sqrt`, exact for a singular $\Sigma$ and with finite gradients
    at repeated eigenvalues.
    """
    if sqrt == "cholesky":
        # Σ + βI added to the matrix, not with gaussx.add_jitter: under pmap
        # (NumPyro's parallel chains) the jitter diagonal add_jitter builds
        # carries a sharding the covariance operator's structure lacks, and
        # lineax rejects the sum as incompatible.
        eye = jnp.eye(cov.shape[-1], dtype=cov.dtype)
        jittered = lx.MatrixLinearOperator(
            cov + jitter * eye, lx.positive_semidefinite_tag
        )
        return gaussx.cholesky(jittered).as_matrix()
    op = lx.MatrixLinearOperator(cov, lx.positive_semidefinite_tag)
    return gaussx.sqrt(op).as_matrix()


def markov_gp_from_innovations(
    prior: MarkovGPPrior,
    eps: Float[Array, "*batch N d"],
    *,
    sqrt: _SqrtMethod = "cholesky",
    jitter: float = 1e-10,
) -> Float[Array, "*batch N"]:
    r"""Latent path of a Markov GP from whitened innovations, in linear time.

    With $\Delta t_k = t_k - t_{k-1}$ and $(A_k, Q_k)$ the kernel's
    discretisation over $\Delta t_k$ (the time convention of
    `MarkovGPPrior.as_lgssm`), the state is run forward from standard
    normal innovations $\varepsilon_k \sim \mathcal{N}(0, I_d)$ by a
    `jax.lax.scan`,

    $$
    x_1 = L_0 \varepsilon_1, \qquad
    x_k = A_k x_{k-1} + L_{Q,k} \varepsilon_k \quad (k = 2, \dots, N), \qquad
    f(t_k) = H x_k + \mu(t_k),
    $$

    where $L_0 L_0^\top = P_0$ (`MarkovGPPrior.initial_covariance`, so
    $P_\infty$ for a stationary kernel) and $L_{Q,k} L_{Q,k}^\top = Q_k$.
    The map $\varepsilon \mapsto f$ is linear, $f = \mu + M \varepsilon$
    with $M M^\top = K$ the prior covariance of the path, so for a
    stationary kernel $f$ has the law `markov_gp_sample` draws from. Time
    is $O(N d^3)$ and memory $O(N d^2)$: no $N \times N$ array is formed,
    and a non-stationary kernel such as `gaussx.IntegratedWienerSDE` works
    as well as a stationary one.

    Use it as a non-centred parameterisation of the latent path: sample
    ``eps`` and compute ``f`` deterministically, which is what NUTS prefers
    and what `markov_gp_innovations_sample` registers. `markov_gp_sample`'s
    dense ``MultivariateNormal`` site stays the simpler choice for short
    series (a few hundred points) of a stationary kernel.

    The square roots are taken by ``sqrt``. $Q_k$ is only positive
    *semi*-definite when $\Delta t_k$ is small (it is computed as a
    difference, and its round-off can be slightly indefinite) and, for
    `gaussx.IntegratedWienerSDE`, close to singular whenever $\Delta t_k$
    is small.

    * ``"cholesky"`` (the default): `gaussx.cholesky` of
      $\Sigma + \beta I$ with $\beta$ = ``jitter``, so each covariance is
      inflated by $\beta I$. Cheapest; fine in float64 for the default
      $\beta = 10^{-10}$ unless the round-off in $Q_k$ exceeds $\beta$
      (large variances, very short lengthscales), when the factor turns
      ``NaN``.
    * ``"eigh"``: the symmetric root
      $Q \operatorname{diag}(\sqrt{\max(\gamma, 0)}) Q^\top$ from
      `gaussx.sqrt` (a dense ``eigh`` with a Sylvester-equation
      derivative). It clips negative round-off, needs no jitter and is
      exact for a singular $Q_k$, so it is the one that handles a $Q_k$
      that is only positive semi-definite, and the one to use in float32.
      ``jitter`` is ignored.

    Args:
        prior: The `MarkovGPPrior` over the $N$ sorted times.
        eps: Standard normal innovations of shape ``(*batch, N, d)``, one
            row per time, with $d$ = ``prior.state_dim``. Leading batch
            dimensions (a ``numpyro.plate``, several paths) give one
            independent path each; the covariance roots are taken once and
            shared.
        sqrt: How the covariance square roots are taken, ``"cholesky"`` or
            ``"eigh"``.
        jitter: The diagonal $\beta$ added before the Cholesky factor.

    Returns:
        The path $f(t_1), \dots, f(t_N)$, shape ``(*batch, N)``, mean
        included.

    Raises:
        ValueError: If the trailing shape of ``eps`` is not ``(N, d)`` or
            ``sqrt`` is unknown.

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import pyrox_ssm as pss
        >>> times = jnp.linspace(0.0, 10.0, 500)
        >>> prior = pss.MarkovGPPrior(pss.MaternSDE(1.0, 0.5, order=1), times)
        >>> eps = jr.normal(jr.key(0), (500, prior.state_dim))
        >>> f = pss.markov_gp_from_innovations(prior, eps)
        >>> f.shape
        (500,)
        >>> eps_batch = jr.normal(jr.key(1), (8, 500, prior.state_dim))
        >>> paths = pss.markov_gp_from_innovations(prior, eps_batch)
        >>> paths.shape
        (8, 500)
    """
    if sqrt not in _SQRT_METHODS:
        raise ValueError(f"sqrt must be one of {_SQRT_METHODS!r}, got {sqrt!r}")
    n, d = prior.times.shape[0], prior.state_dim
    eps = jnp.asarray(eps)
    if eps.ndim < 2 or eps.shape[-2:] != (n, d):
        raise ValueError(
            f"eps must have shape (*batch, {n}, {d}), one row of {d} "
            f"innovations per time for a state dimension of {d}, got "
            f"{tuple(eps.shape)!r}"
        )
    batch_shape = eps.shape[:-2]
    _F, _L, H, _Qc, _P_inf = prior.sde_kernel.sde_params()
    # The leading zero step gives A_1 = I, so starting from x_0 = 0 and
    # replacing Q_1 = 0 by P_0 makes the first step draw x_1 = L_0 eps_1.
    A_seq, Q_seq = prior.sde_kernel.discretise_sequence(_build_dt_full(prior.times))
    cov_seq = Q_seq.at[0].set(prior.initial_covariance())
    L_seq = jax.vmap(lambda cov: _covariance_root(cov, sqrt=sqrt, jitter=jitter))(
        cov_seq
    )

    def step(
        x_prev: Float[Array, " d"],
        inputs: tuple[Float[Array, "d d"], Float[Array, "d d"], Float[Array, " d"]],
    ) -> tuple[Float[Array, " d"], Float[Array, " d"]]:
        A, L, e = inputs
        # xₖ = Aₖ xₖ₋₁ + L_{Q,k} εₖ
        x = einx.dot("i j, j -> i", A, x_prev) + einx.dot("i j, j -> i", L, e)
        return x, x

    x_init = jnp.zeros((d,), dtype=jnp.result_type(L_seq, eps))

    def path(e: Float[Array, "N d"]) -> Float[Array, " N"]:
        _, x_seq = jax.lax.scan(step, x_init, (A_seq, L_seq, e))
        # f(tₖ) = H xₖ
        return einx.dot("n d, o d -> n o", x_seq, H)[:, 0]

    # One scan per leading batch index, sharing A_seq and L_seq.
    f = jax.vmap(path)(eps.reshape((-1, n, d))).reshape((*batch_shape, n))
    # f(tₖ) += μ(tₖ)
    return f + prior.mean(prior.times)


def markov_gp_innovations_sample(
    name: str,
    prior: MarkovGPPrior,
    *,
    sqrt: _SqrtMethod = "cholesky",
    jitter: float = 1e-10,
) -> Float[Array, " N"]:
    r"""Sample a Markov-GP latent path through whitened innovations.

    Registers the innovations as one sample site and the path, built from
    them by `markov_gp_from_innovations`, as a deterministic one,

    $$
    \varepsilon \sim \mathcal{N}(0, I_{N d}), \qquad
    f = \mu + M \varepsilon, \qquad M M^\top = K,
    $$

    so the path costs $O(N d^3)$ per model evaluation instead of the
    $O(N^3)$ Cholesky of `markov_gp_sample`, and NUTS sees a whitened,
    non-centred parameterisation. Use it for long series, non-stationary
    kernels, and non-Gaussian likelihoods sampled with NUTS or fitted with
    SVI; keep `markov_gp_sample` for short series of a stationary kernel.
    The ``sqrt`` and ``jitter`` options are those of
    `markov_gp_from_innovations`; ``"eigh"`` handles a process noise $Q_k$
    that is only positive semi-definite (closely spaced times,
    `gaussx.IntegratedWienerSDE`).

    It registers two sites, unscoped like `markov_gp_sample`'s:

    * ``f"{name}_eps"``, the innovations: ``Normal(0, 1)`` of shape
      ``(N, d)`` with both dimensions in the event
      (``.expand([N, d]).to_event(2)``);
    * ``name``, a ``numpyro.deterministic`` holding the path of shape
      ``(N,)``, so `numpyro.infer.Predictive` returns it.

    Args:
        name: The deterministic site's name; the innovations site is
            ``f"{name}_eps"``.
        prior: The `MarkovGPPrior` over the $N$ sorted times.
        sqrt: How the covariance square roots are taken, ``"cholesky"`` or
            ``"eigh"``.
        jitter: The diagonal added before the Cholesky factor.

    Returns:
        The path $f(t_1), \dots, f(t_N)$, shape ``(N,)``, or
        ``(*batch, N)`` under enclosing ``numpyro.plate`` contexts, one
        independent path per plate index.

    Examples:
        A Bernoulli series with a latent Matérn-3/2 path:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> from numpyro.infer import MCMC, NUTS
        >>> import pyrox_ssm as pss
        >>> times = jnp.linspace(0.0, 10.0, 200)
        >>> y = (jnp.sin(times) > 0).astype(jnp.int32)
        >>> def model(times, y):
        ...     ell = numpyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))
        ...     sde = pss.MaternSDE(variance=1.0, lengthscale=ell, order=1)
        ...     f = pss.markov_gp_innovations_sample("f", pss.MarkovGPPrior(sde, times))
        ...     numpyro.sample("y", dist.Bernoulli(logits=f), obs=y)
        >>> mcmc = MCMC(
        ...     NUTS(model), num_warmup=100, num_samples=100, progress_bar=False
        ... )
        >>> mcmc.run(jr.key(0), times, y)
        >>> mcmc.get_samples()["f"].shape
        (100, 200)
    """
    n, d = prior.times.shape[0], prior.state_dim
    zero = jnp.zeros((), dtype=prior.times.dtype)
    eps = numpyro.sample(
        f"{name}_eps", dist.Normal(zero, 1.0).expand([n, d]).to_event(2)
    )
    # Enclosing plates prepend batch dimensions to eps; the path keeps them.
    f = markov_gp_from_innovations(prior, jnp.asarray(eps), sqrt=sqrt, jitter=jitter)
    return numpyro.deterministic(name, f)  # ty: ignore[invalid-return-type]
