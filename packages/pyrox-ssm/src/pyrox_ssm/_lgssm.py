r"""Linear-Gaussian state-space prior given directly by its matrices.

`LinearGaussianSSMPrior` is the general model behind every Gaussian Markov
prior in this package: a transition, observation and noise sequence over a
time grid, filtered by `gaussx.kalman_filter` and smoothed by
`gaussx.rts_smoother`. `pyrox_ssm.MarkovGPPrior` is its kernel-built special
case (`pyrox_ssm.MarkovGPPrior.as_lgssm`), and all of its Kalman filtering
runs through `LinearGaussianSSMPrior.filter`, the one call site of
`gaussx.kalman_filter` in the core package.

Use it directly when the dynamics are known as matrices rather than as a GP
kernel: a local-level model, a multi-output state, a discretised physical
model whose matrices carry priors of their own.
"""

from __future__ import annotations

import einx
import equinox as eqx
import gaussx
import jax.numpy as jnp
import numpyro
from gaussx import FilterState
from jaxtyping import Array, Bool, Float


def _as_float(x: Array) -> Array:
    """Promote an integer or boolean array to float, leaving floats alone.

    A float array (weakly typed or not) keeps its dtype, so float32 callers
    stay float32; an integer matrix would otherwise seed the filter's scan
    with an integer carry and fail inside it.
    """
    x = jnp.asarray(x)
    if jnp.issubdtype(x.dtype, jnp.inexact):
        return x
    return x.astype(jnp.result_type(x, 0.0))


def _check_shape(name: str, x: Array, allowed: tuple[tuple[int, ...], ...]) -> None:
    """Raise a `ValueError` naming ``name`` unless ``x.shape`` is in ``allowed``."""
    if tuple(x.shape) not in allowed:
        expected = " or ".join(repr(s) for s in allowed)
        raise ValueError(f"{name} must have shape {expected}, got {tuple(x.shape)!r}")


class LinearGaussianSSMPrior(eqx.Module):
    r"""Prior over a linear-Gaussian state-space model given by its matrices.

    With $T$ observation steps, a state of length $D$ and observations of
    length $M$, the model is

    $$
    x_0 \sim \mathcal{N}(m_0, P_0), \qquad
    x_k = A_k x_{k-1} + \eta_k, \quad \eta_k \sim \mathcal{N}(0, Q_k), \qquad
    y_k = H_k x_k + b_k + \varepsilon_k, \quad
    \varepsilon_k \sim \mathcal{N}(0, R_k),
    $$

    for $k = 1, \dots, T$, and row $k - 1$ of every time-stacked field
    holds step $k$. This is the convention of `gaussx.kalman_filter`, which
    predicts before every update: the first observation ``y[0]`` is $y_1$,
    scored against $A_1 x_0$, not against $x_0$. To observe the initial
    state itself, set ``transition[0] = I`` and ``process_noise[0] = 0``,
    which is what `pyrox_ssm.MarkovGPPrior.as_lgssm` does (its first step
    has $\Delta t = 0$).

    The marginal likelihood is the prediction-error decomposition

    $$
    \log p(y \mid \theta) = \sum_{k=1}^{T}
      \log \mathcal{N}\!\left(y_k;\, H_k m_k^- + b_k,\,
      H_k P_k^- H_k^\top + R_k\right),
    $$

    with $m_k^-, P_k^-$ the predicted moments of $x_k$ given
    $y_{1:k-1}$, at $O(T D^3)$ cost.

    Every method that reads observations takes an optional ``mask``, either
    per step, shaped ``(T,)`` (a masked step runs the predict only), or per
    entry, shaped ``(T, M)`` (the masked entries of $y_k$ are marginalised
    out exactly, so the log marginal is that of the observed entries).
    Masked entries of ``y`` are replaced by zero before any arithmetic, so a
    ``NaN`` placeholder there reaches neither the value nor a gradient.

    The prior registers no NumPyro sites itself; `lgssm_factor` adds its log
    marginal as a factor. Shapes are checked at construction (array shapes
    are static, so the check also runs on traced matrices), and integer
    arrays are promoted to float.

    Attributes:
        transition: Transitions $A_k$, shaped ``(T, D, D)``.
        obs_model: Observation matrices $H_k$, shaped ``(T, M, D)``, or one
            ``(M, D)`` matrix for every step.
        process_noise: Process-noise covariances $Q_k$, shaped
            ``(T, D, D)``.
        obs_noise: Observation-noise covariances $R_k$, shaped
            ``(T, M, M)``, or one ``(M, M)`` matrix for every step.
        init_mean: Mean $m_0$ of the initial state, shaped ``(D,)``.
        init_cov: Covariance $P_0$ of the initial state, shaped ``(D, D)``.
        obs_offset: Optional known observation offset $b_k$, shaped
            ``(T, M)``; ``None`` is $b_k = 0$. It is subtracted from ``y``
            before filtering, so a GP with a mean function becomes an exact
            special case (`pyrox_ssm.MarkovGPPrior.as_lgssm` sets it to the
            mean at the observation times).
        parallel: Filter with `gaussx.parallel_kalman_filter`, an
            associative scan, instead of the sequential
            `gaussx.kalman_filter`. Default ``False``. Both give the same
            filtered moments, log marginal and gradients to round-off. The
            trade-off, measured on CPU for a scalar observation
            and a Matérn state of dimension 2 to 3 at 1,095 to 10,000 steps:

            - The value and gradient together (``jax.value_and_grad``, what
              NUTS and SVI call) are 1.2 to 1.8 times cheaper in parallel,
              because a reverse-mode pass through a sequential scan of small
              steps is slow.
            - The value alone is 1.5 to 2.2 times slower in parallel, so
              leave it off where only the value is needed (a fixed-parameter
              fit or predict).
            - The parallel filter runs in covariance form with dense solves
              in its combine step and does not take a ``solver``; neither
              filter is a square-root filter.

            On an accelerator the parallel filter's depth is $O(\log T)$
            against $O(T)$; the numbers above are CPU only.

    Examples:
        A local-level model, $x_k = x_{k-1} + \eta_k$ observed with noise:

        >>> import jax.numpy as jnp
        >>> import pyrox_ssm as pss
        >>> T, q, r = 5, 0.1, 0.5
        >>> prior = pss.LinearGaussianSSMPrior(
        ...     transition=jnp.ones((T, 1, 1)),
        ...     obs_model=jnp.eye(1),
        ...     process_noise=jnp.full((T, 1, 1), q),
        ...     obs_noise=r * jnp.eye(1),
        ...     init_mean=jnp.zeros(1),
        ...     init_cov=10.0 * jnp.eye(1),
        ... )
        >>> y = jnp.array([[0.1], [0.4], [jnp.nan], [0.3], [0.2]])
        >>> mask = jnp.array([True, True, False, True, True])
        >>> log_marg = prior.log_marginal(y, mask=mask)
        >>> log_marg.shape
        ()
    """

    transition: Float[Array, "T D D"]
    obs_model: Float[Array, "T M D"] | Float[Array, "M D"]
    process_noise: Float[Array, "T D D"]
    obs_noise: Float[Array, "T M M"] | Float[Array, "M M"]
    init_mean: Float[Array, " D"]
    init_cov: Float[Array, "D D"]
    obs_offset: Float[Array, "T M"] | None = None
    parallel: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        transition: Float[Array, "T D D"],
        obs_model: Float[Array, "T M D"] | Float[Array, "M D"],
        process_noise: Float[Array, "T D D"],
        obs_noise: Float[Array, "T M M"] | Float[Array, "M M"],
        init_mean: Float[Array, " D"],
        init_cov: Float[Array, "D D"],
        obs_offset: Float[Array, "T M"] | None = None,
        parallel: bool = False,
    ) -> None:
        transition = _as_float(transition)
        obs_model = _as_float(obs_model)
        process_noise = _as_float(process_noise)
        obs_noise = _as_float(obs_noise)
        init_mean = _as_float(init_mean)
        init_cov = _as_float(init_cov)
        if transition.ndim != 3 or transition.shape[1] != transition.shape[2]:
            raise ValueError(
                "transition must be a (T, D, D) stack of square matrices, got "
                f"shape {tuple(transition.shape)!r}"
            )
        T, D = transition.shape[0], transition.shape[1]
        if obs_model.ndim not in (2, 3) or obs_model.shape[-1] != D:
            raise ValueError(
                f"obs_model must have shape (T, M, D) or (M, D) with D = {D}, "
                f"got {tuple(obs_model.shape)!r}"
            )
        M = obs_model.shape[-2]
        _check_shape("obs_model", obs_model, ((T, M, D), (M, D)))
        _check_shape("process_noise", process_noise, ((T, D, D),))
        _check_shape("obs_noise", obs_noise, ((T, M, M), (M, M)))
        _check_shape("init_mean", init_mean, ((D,),))
        _check_shape("init_cov", init_cov, ((D, D),))
        if obs_offset is not None:
            obs_offset = _as_float(obs_offset)
            _check_shape("obs_offset", obs_offset, ((T, M),))
        self.transition = transition
        self.obs_model = obs_model
        self.process_noise = process_noise
        self.obs_noise = obs_noise
        self.init_mean = init_mean
        self.init_cov = init_cov
        self.obs_offset = obs_offset
        self.parallel = parallel

    @property
    def n_steps(self) -> int:
        """Number of observation steps $T$."""
        return self.transition.shape[0]

    @property
    def state_dim(self) -> int:
        """State dimension $D$."""
        return self.transition.shape[-1]

    @property
    def obs_dim(self) -> int:
        """Observation dimension $M$."""
        return self.obs_model.shape[-2]

    def _observations(
        self, y: Float[Array, "T M"] | Float[Array, " T"]
    ) -> Float[Array, "T M"]:
        """``y`` as a float ``(T, M)`` array; ``(T,)`` is accepted when M = 1."""
        y = _as_float(y)
        T, M = self.n_steps, self.obs_dim
        if y.ndim == 1 and M == 1:
            y = einx.id("t -> t 1", y)
        if tuple(y.shape) != (T, M):
            accepted = f"({T}, {M})" + (f" or ({T},)" if M == 1 else "")
            raise ValueError(
                f"y must have shape {accepted} for {T} steps and {M} outputs, "
                f"got {tuple(y.shape)!r}"
            )
        return y

    def _mask(
        self, mask: Bool[Array, " T"] | Bool[Array, "T M"] | None
    ) -> Bool[Array, " T"] | Bool[Array, "T M"] | None:
        """Validate ``mask`` as per step ``(T,)`` or per entry ``(T, M)``."""
        if mask is None:
            return None
        mask = jnp.asarray(mask).astype(bool)
        T, M = self.n_steps, self.obs_dim
        if tuple(mask.shape) not in ((T,), (T, M)):
            raise ValueError(
                f"mask has shape {tuple(mask.shape)!r}, but must be ({T},) to "
                f"mask whole steps or ({T}, {M}) to mask single entries"
            )
        return mask

    def _sanitise(
        self,
        y: Float[Array, "T M"],
        mask: Bool[Array, " T"] | Bool[Array, "T M"] | None,
    ) -> Float[Array, "T M"]:
        """Zero the masked entries of ``y``, so a ``NaN`` there is inert."""
        if mask is None:
            return y
        keep = mask if mask.ndim == 2 else einx.id("t -> t 1", mask)
        return jnp.where(keep, y, jnp.zeros((), dtype=y.dtype))

    def _residual(self, y: Float[Array, "T M"]) -> Float[Array, "T M"]:
        """``y - b``, the observations of the zero-offset model."""
        if self.obs_offset is None:
            return y
        return y - self.obs_offset

    def filter(
        self,
        y: Float[Array, "T M"] | Float[Array, " T"],
        *,
        mask: Bool[Array, " T"] | Bool[Array, "T M"] | None = None,
    ) -> FilterState:
        r"""Run the forward Kalman filter.

        Each step predicts, then updates on the observed entries of $y_k$,

        $$
        m_k^- = A_k m_{k-1}, \qquad P_k^- = A_k P_{k-1} A_k^\top + Q_k,
        $$

        and adds $\log \mathcal{N}(y_k;\, H_k m_k^- + b_k,\,
        H_k P_k^- H_k^\top + R_k)$ over those entries to the log marginal.

        Args:
            y: Observations, shaped ``(T, M)``; ``(T,)`` is accepted when
                $M = 1$. Masked entries may be ``NaN``.
            mask: Optional boolean mask, ``(T,)`` per step or ``(T, M)`` per
                entry, ``True`` where observed. ``None`` observes everything.

        Returns:
            The `gaussx.FilterState`: filtered and predicted means
            ``(T, D)`` and covariances ``(T, D, D)``, row $k - 1$ holding
            step $k$, and the scalar ``log_likelihood``.

        Raises:
            ValueError: If ``y`` or ``mask`` has the wrong shape.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> prior = pss.LinearGaussianSSMPrior(
            ...     jnp.ones((3, 1, 1)), jnp.eye(1), jnp.full((3, 1, 1), 0.1),
            ...     0.5 * jnp.eye(1), jnp.zeros(1), jnp.eye(1),
            ... )
            >>> state = prior.filter(jnp.array([0.1, 0.3, 0.2]))
            >>> state.filtered_means.shape, state.log_likelihood.shape
            ((3, 1), ())
        """
        mask = self._mask(mask)
        residual = self._residual(self._sanitise(self._observations(y), mask))
        kalman_filter = (
            gaussx.parallel_kalman_filter if self.parallel else gaussx.kalman_filter
        )
        return kalman_filter(
            transition=self.transition,
            obs_model=self.obs_model,
            process_noise=self.process_noise,
            obs_noise=self.obs_noise,
            observations=residual,
            init_mean=self.init_mean,
            init_cov=self.init_cov,
            mask=mask,
        )

    def log_marginal(
        self,
        y: Float[Array, "T M"] | Float[Array, " T"],
        *,
        mask: Bool[Array, " T"] | Bool[Array, "T M"] | None = None,
    ) -> Float[Array, ""]:
        r"""Marginal log-likelihood $\log p(y_{\mathrm{obs}} \mid \theta)$.

        $$
        \log p(y \mid \theta) = \sum_{k=1}^{T}
          \log \mathcal{N}\!\left(y_k;\, H_k m_k^- + b_k,\,
          H_k P_k^- H_k^\top + R_k\right),
        $$

        restricted to the observed entries when a mask is given.

        Args:
            y: Observations, shaped ``(T, M)`` (or ``(T,)`` when $M = 1$);
                masked entries may be ``NaN``.
            mask: Optional ``(T,)`` or ``(T, M)`` mask, ``True`` where
                observed.

        Returns:
            The scalar log marginal of the observed entries.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> prior = pss.LinearGaussianSSMPrior(
            ...     jnp.broadcast_to(jnp.eye(2), (4, 2, 2)), jnp.eye(2),
            ...     jnp.broadcast_to(0.1 * jnp.eye(2), (4, 2, 2)),
            ...     0.5 * jnp.eye(2), jnp.zeros(2), jnp.eye(2),
            ... )
            >>> y = jnp.array([[0.1, 0.2], [0.3, jnp.nan], [0.2, 0.1], [0.0, 0.4]])
            >>> mask = ~jnp.isnan(y)  # one output missing at step 2
            >>> log_marg = prior.log_marginal(y, mask=mask)
        """
        return self.filter(y, mask=mask).log_likelihood

    def smooth(
        self,
        y: Float[Array, "T M"] | Float[Array, " T"],
        *,
        mask: Bool[Array, " T"] | Bool[Array, "T M"] | None = None,
    ) -> tuple[Float[Array, "T D"], Float[Array, "T D D"], Float[Array, ""]]:
        r"""Run the Kalman filter and the RTS smoother.

        Returns the moments of $p(x_k \mid y_{1:T})$ for $k = 1, \dots, T$
        (row $k - 1$), via `gaussx.rts_smoother`.

        Args:
            y: Observations, shaped ``(T, M)`` (or ``(T,)`` when $M = 1$);
                masked entries may be ``NaN``.
            mask: Optional ``(T,)`` or ``(T, M)`` mask, ``True`` where
                observed.

        Returns:
            ``(smoothed_means, smoothed_covs, log_marginal)``, shaped
            ``(T, D)``, ``(T, D, D)`` and ``()``.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> prior = pss.LinearGaussianSSMPrior(
            ...     jnp.ones((3, 1, 1)), jnp.eye(1), jnp.full((3, 1, 1), 0.1),
            ...     0.5 * jnp.eye(1), jnp.zeros(1), jnp.eye(1),
            ... )
            >>> m, P, _ = prior.smooth(jnp.array([0.1, 0.3, 0.2]))
            >>> m.shape, P.shape
            ((3, 1), (3, 1, 1))
        """
        state = self.filter(y, mask=mask)
        m_smooth, P_smooth = gaussx.rts_smoother(state, self.transition)
        return m_smooth, P_smooth, state.log_likelihood

    def condition(
        self,
        y: Float[Array, "T M"] | Float[Array, " T"],
        *,
        mask: Bool[Array, " T"] | Bool[Array, "T M"] | None = None,
    ) -> ConditionedLinearGaussianSSM:
        """Condition on ``y`` with the filter and the RTS smoother.

        Args:
            y: Observations, shaped ``(T, M)`` (or ``(T,)`` when $M = 1$);
                masked entries may be ``NaN``.
            mask: Optional ``(T,)`` or ``(T, M)`` mask, ``True`` where
                observed.

        Returns:
            The `ConditionedLinearGaussianSSM`, holding ``y`` as ``(T, M)``
            with its masked entries set to zero.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> prior = pss.LinearGaussianSSMPrior(
            ...     jnp.ones((3, 1, 1)), jnp.eye(1), jnp.full((3, 1, 1), 0.1),
            ...     0.5 * jnp.eye(1), jnp.zeros(1), jnp.eye(1),
            ... )
            >>> cond = prior.condition(jnp.array([0.1, 0.3, 0.2]))
            >>> cond.smoothed_means.shape
            (3, 1)
        """
        mask = self._mask(mask)
        m_smooth, P_smooth, log_marg = self.smooth(y, mask=mask)
        return ConditionedLinearGaussianSSM(
            prior=self,
            y=self._sanitise(self._observations(y), mask),
            smoothed_means=m_smooth,
            smoothed_covs=P_smooth,
            log_marginal=log_marg,
            mask=mask,
        )


class ConditionedLinearGaussianSSM(eqx.Module):
    r"""A `LinearGaussianSSMPrior` conditioned on observations.

    Holds the smoothed moments of $p(x_k \mid y_{1:T})$, row $k - 1$ for
    step $k$, and the log marginal.

    Attributes:
        prior: The originating `LinearGaussianSSMPrior`.
        y: Observations, ``(T, M)``, with masked entries set to zero.
        smoothed_means: Smoothed state means, ``(T, D)``.
        smoothed_covs: Smoothed state covariances, ``(T, D, D)``.
        log_marginal: Scalar $\log p(y_{\mathrm{obs}} \mid \theta)$.
        mask: The ``(T,)`` or ``(T, M)`` observation mask, or ``None`` when
            everything was observed.

    Examples:
        >>> import jax.numpy as jnp
        >>> import pyrox_ssm as pss
        >>> prior = pss.LinearGaussianSSMPrior(
        ...     jnp.ones((3, 1, 1)), jnp.eye(1), jnp.full((3, 1, 1), 0.1),
        ...     0.5 * jnp.eye(1), jnp.zeros(1), jnp.eye(1),
        ... )
        >>> cond = prior.condition(jnp.array([0.1, 0.3, 0.2]))
        >>> cond.smoothed_covs.shape, cond.log_marginal.shape
        ((3, 1, 1), ())
    """

    prior: LinearGaussianSSMPrior
    y: Float[Array, "T M"]
    smoothed_means: Float[Array, "T D"]
    smoothed_covs: Float[Array, "T D D"]
    log_marginal: Float[Array, ""]
    mask: Bool[Array, " T"] | Bool[Array, "T M"] | None = None


def lgssm_factor(
    name: str,
    prior: LinearGaussianSSMPrior,
    y: Float[Array, "T M"] | Float[Array, " T"],
    *,
    mask: Bool[Array, " T"] | Bool[Array, "T M"] | None = None,
) -> None:
    r"""Register a linear-Gaussian SSM's log marginal as a NumPyro factor.

    Adds ``numpyro.factor(name, prior.log_marginal(y, mask=mask))``, the
    state path marginalised out by the Kalman filter,

    $$
    \log p(y \mid \theta) = \sum_{k=1}^{T}
      \log \mathcal{N}\!\left(y_k;\, H_k m_k^- + b_k,\,
      H_k P_k^- H_k^\top + R_k\right).
    $$

    Registers the one site ``name``.

    Args:
        name: The factor site name.
        prior: The `LinearGaussianSSMPrior`.
        y: Observations, shaped ``(T, M)`` (or ``(T,)`` when $M = 1$);
            masked entries may be ``NaN``.
        mask: Optional ``(T,)`` or ``(T, M)`` mask, ``True`` where observed.

    Examples:
        A local-level model with priors on its two noise variances:

        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> import pyrox_ssm as pss
        >>> def model(y):
        ...     T = y.shape[0]
        ...     q = numpyro.sample("q", dist.LogNormal(-2.0, 1.0))
        ...     r = numpyro.sample("r", dist.LogNormal(-1.0, 1.0))
        ...     prior = pss.LinearGaussianSSMPrior(
        ...         jnp.ones((T, 1, 1)), jnp.eye(1), q * jnp.ones((T, 1, 1)),
        ...         r * jnp.eye(1), jnp.zeros(1), 10.0 * jnp.eye(1),
        ...     )
        ...     pss.lgssm_factor("y", prior, y)
        >>> trace = numpyro.handlers.trace(numpyro.handlers.seed(model, 0)).get_trace(
        ...     jnp.array([0.1, 0.4, 0.3])
        ... )
        >>> sorted(trace)
        ['q', 'r', 'y']
    """
    numpyro.factor(name, prior.log_marginal(y, mask=mask))
