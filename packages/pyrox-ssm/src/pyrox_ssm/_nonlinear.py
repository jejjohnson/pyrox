r"""A nonlinear state-space model: its prior, its factor and its posterior.

`NonlinearGaussianSSMPrior` holds a state-space model with nonlinear
dynamics or observations and runs the moment-matched Gaussian filter
`gaussx.nonlinear_kalman_filter` and smoother `gaussx.nonlinear_rts_smoother`
over a series; `condition` returns `ConditionedNonlinearGaussianSSM`, the
smoothed states and each observation's predictive moments.
`nonlinear_ssm_factor` adds the filter's log likelihood to a NumPyro model,
through the same prior. The choice of `gaussx` integrator is the choice of
filter (extended, unscented, cubature, Gauss-Hermite or Monte Carlo Kalman
filter); the prior fixes the conventions around it in one place: the default
integrator, the observation mask and the handling of masked entries.
"""

from __future__ import annotations

from collections.abc import Callable

import einx
import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
import lineax as lx
import numpyro
from gaussx import AbstractIntegrator, AbstractSolverStrategy, FilterState
from jaxtyping import Array, Bool, Float

from pyrox_ssm._lgssm import _as_float


def _check_observations(
    y: Float[Array, "T M"],
    mask: Bool[Array, " T"] | Bool[Array, "T M"] | None,
) -> tuple[Float[Array, "T M"], Bool[Array, " T"] | Bool[Array, "T M"] | None]:
    """Validate ``y`` as ``(T, M)`` and ``mask`` as ``(T,)`` or ``(T, M)``.

    Then zero the masked entries of ``y``, as `pyrox_ssm.LinearGaussianSSMPrior`
    does, so a ``NaN`` placeholder there is inert before any arithmetic,
    without relying on how gaussx gates a masked step.
    """
    y = _as_float(y)
    if y.ndim != 2:
        raise ValueError(
            f"y must have shape (T, M), one row per step, got {tuple(y.shape)!r}"
        )
    if mask is None:
        return y, None
    mask = jnp.asarray(mask).astype(bool)
    T, M = y.shape
    if tuple(mask.shape) not in ((T,), (T, M)):
        raise ValueError(
            f"mask has shape {tuple(mask.shape)!r}, but must be ({T},) to "
            f"mask whole steps or ({T}, {M}) to mask single entries"
        )
    keep = mask if mask.ndim == 2 else einx.id("t -> t 1", mask)
    return jnp.where(keep, y, jnp.zeros((), dtype=y.dtype)), mask


def _default_integrator() -> AbstractIntegrator:
    """``gaussx.UnscentedIntegrator(alpha=1.0)``, gaussx's filter default."""
    return gaussx.UnscentedIntegrator(alpha=1.0)


class NonlinearGaussianSSMPrior(eqx.Module):
    r"""Prior over a nonlinear state-space model, filtered by moment matching.

    With $T$ observation steps, a state of length $D$ and observations of
    length $M$, the model is

    $$
    x_0 \sim \mathcal{N}(m_0, P_0), \qquad
    x_k = f(x_{k-1}) + \eta_k, \quad \eta_k \sim \mathcal{N}(0, Q_k), \qquad
    y_k = h(x_k) + \varepsilon_k, \quad \varepsilon_k \sim \mathcal{N}(0, R_k),
    $$

    for $k = 1, \dots, T$, with row $k - 1$ of ``y`` holding step $k$: the
    convention of `gaussx.nonlinear_kalman_filter` and of
    `pyrox_ssm.LinearGaussianSSMPrior`, which predict before every update.

    `filter` runs `gaussx.nonlinear_kalman_filter`, which pushes the Gaussian
    belief through $f$ and $h$ by the integrator's moment transform, and
    `log_marginal` is its prediction-error sum

    $$
    \log p(y_{1:T} \mid \theta) \approx \sum_{k=1}^{T}
      \log \mathcal{N}(y_k;\, \hat{y}_k,\, S_k),
    $$

    with $\hat{y}_k = \mathbb{E}[h(x_k)]$ and
    $S_k = \operatorname{Cov}[h(x_k)] + R_k$ under the predicted Gaussian of
    $x_k$. In gaussx's words this is "a moment-matched surrogate, not the
    exact marginal likelihood"; it is exact when $f$ and $h$ are affine.
    `smooth` runs `gaussx.nonlinear_rts_smoother` backwards with the same
    integrator, whose gain uses the cross-covariance of $x_k$ and
    $f(x_k)$; for affine $f$ it is the Kalman RTS smoother. A Monte Carlo
    integrator makes the filter and the smoother stochastic; the tests use
    the deterministic rules.

    The integrator picks the filter: `gaussx.TaylorIntegrator` (EKF),
    `gaussx.UnscentedIntegrator` (UKF, the default),
    `gaussx.CubatureIntegrator` (CKF),
    `gaussx.FifthOrderCubatureIntegrator`, `gaussx.GaussHermiteIntegrator`
    or `gaussx.MonteCarloIntegrator`.

    Every method that reads observations takes an optional ``mask``, per
    step ``(T,)`` (a masked step runs the predict only) or per entry
    ``(T, M)`` (masked entries are marginalised out of the update). Masked
    entries of ``y`` are replaced by zero before any arithmetic, so a
    ``NaN`` there reaches neither the value nor a gradient. The prior
    registers no NumPyro sites; `nonlinear_ssm_factor` adds its log
    likelihood as a factor.

    Attributes:
        dynamics: Deterministic dynamics $f$, from a ``(D,)`` state to a
            ``(D,)`` state; a pytree (an ``eqx.Module`` or a closure), so its
            parameters get gradients.
        obs_fn: Observation map $h$, from a ``(D,)`` state to an ``(M,)``
            observation.
        process_noise: Process-noise covariance $Q_k$: ``(D, D)`` for every
            step, a ``(T, D, D)`` stack, or a lineax operator.
        obs_noise: Observation-noise covariance $R_k$: ``(M, M)``,
            ``(T, M, M)`` or a lineax operator.
        init_mean: Mean $m_0$ of the initial state, ``(D,)``.
        init_cov: Covariance $P_0$ of the initial state, ``(D, D)``.
        integrator: The gaussx moment-matching rule. ``None`` is
            ``gaussx.UnscentedIntegrator(alpha=1.0)``, the default of
            `gaussx.nonlinear_kalman_filter`: derivative-free and exact for
            affine maps. `gaussx.UnscentedIntegrator`'s own ``alpha=1e-3``
            loses about six digits to cancellation.
        joseph: Use the Joseph-form covariance update. Default ``True``.
        solver: Solver strategy for $S_k$, passed through to gaussx.
            ``None`` factorises $S_k$ once by Cholesky.

    Examples:
        A log-emission AR(1) state, $x_k = 0.8 x_{k-1} + \eta_k$ observed
        through $y_k = \exp(x_k) + \varepsilon_k$, with the third step
        missing:

        >>> import jax.numpy as jnp
        >>> import pyrox_ssm as pss
        >>> prior = pss.NonlinearGaussianSSMPrior(
        ...     lambda x: 0.8 * x, jnp.exp, 0.1 * jnp.eye(1), 0.05 * jnp.eye(1),
        ...     jnp.zeros(1), jnp.eye(1),
        ... )
        >>> y = jnp.array([[1.1], [0.9], [jnp.nan], [1.0]])
        >>> mask = jnp.array([True, True, False, True])
        >>> log_marg = prior.log_marginal(y, mask=mask)
        >>> posterior = prior.condition(y, mask=mask)
        >>> posterior.smoothed_means.shape
        (4, 1)
    """

    dynamics: Callable[[Float[Array, " D"]], Float[Array, " D"]]
    obs_fn: Callable[[Float[Array, " D"]], Float[Array, " M"]]
    process_noise: (
        Float[Array, "D D"] | Float[Array, "T D D"] | lx.AbstractLinearOperator
    )
    obs_noise: Float[Array, "M M"] | Float[Array, "T M M"] | lx.AbstractLinearOperator
    init_mean: Float[Array, " D"]
    init_cov: Float[Array, "D D"]
    integrator: AbstractIntegrator | None = None
    joseph: bool = eqx.field(static=True, default=True)
    solver: AbstractSolverStrategy | None = None

    def _integrator(self) -> AbstractIntegrator:
        return _default_integrator() if self.integrator is None else self.integrator

    def filter(
        self,
        y: Float[Array, "T M"],
        *,
        mask: Bool[Array, " T"] | Bool[Array, "T M"] | None = None,
    ) -> FilterState:
        """Run the moment-matched forward filter.

        Args:
            y: Observations, ``(T, M)``; masked entries may be ``NaN``.
            mask: Optional ``(T,)`` or ``(T, M)`` mask, ``True`` where
                observed.

        Returns:
            The `gaussx.FilterState`: filtered and predicted means ``(T, D)``
            and covariances ``(T, D, D)``, row $k - 1$ holding step $k$, and
            the scalar surrogate ``log_likelihood``.

        Raises:
            ValueError: If ``y`` is not ``(T, M)`` or ``mask`` is not ``(T,)``
                or ``(T, M)``.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> prior = pss.NonlinearGaussianSSMPrior(
            ...     jnp.sin, lambda x: x, 0.1 * jnp.eye(1), 0.2 * jnp.eye(1),
            ...     jnp.zeros(1), jnp.eye(1),
            ... )
            >>> state = prior.filter(jnp.array([[0.1], [0.3], [0.2]]))
            >>> state.filtered_means.shape
            (3, 1)
        """
        y, mask = _check_observations(y, mask)
        return gaussx.nonlinear_kalman_filter(
            self.dynamics,
            self.obs_fn,
            self.process_noise,
            self.obs_noise,
            y,
            self.init_mean,
            self.init_cov,
            integrator=self._integrator(),
            mask=mask,
            joseph=self.joseph,
            solver=self.solver,
        )

    def log_marginal(
        self,
        y: Float[Array, "T M"],
        *,
        mask: Bool[Array, " T"] | Bool[Array, "T M"] | None = None,
    ) -> Float[Array, ""]:
        r"""The moment-matched surrogate of $\log p(y_{\mathrm{obs}} \mid \theta)$.

        $$
        \log p(y_{1:T} \mid \theta) \approx \sum_{k=1}^{T}
          \log \mathcal{N}(y_k;\, \hat{y}_k,\, S_k),
        $$

        over the observed entries. Exact when $f$ and $h$ are affine; a
        surrogate otherwise, so maximising it or sampling under it targets
        the surrogate. It is the value `nonlinear_ssm_factor` registers.

        Args:
            y: Observations, ``(T, M)``; masked entries may be ``NaN``.
            mask: Optional ``(T,)`` or ``(T, M)`` mask, ``True`` where
                observed.

        Returns:
            The scalar surrogate log marginal.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> prior = pss.NonlinearGaussianSSMPrior(
            ...     lambda x: 0.8 * x, jnp.exp, 0.1 * jnp.eye(1), 0.05 * jnp.eye(1),
            ...     jnp.zeros(1), jnp.eye(1),
            ... )
            >>> log_marg = prior.log_marginal(jnp.array([[1.1], [0.9], [1.0]]))
        """
        return self.filter(y, mask=mask).log_likelihood

    def smooth(
        self,
        y: Float[Array, "T M"],
        *,
        mask: Bool[Array, " T"] | Bool[Array, "T M"] | None = None,
    ) -> tuple[Float[Array, "T D"], Float[Array, "T D D"], Float[Array, ""]]:
        r"""Run the filter and the moment-matched RTS smoother.

        Returns the moments of $p(x_k \mid y_{1:T})$ for $k = 1, \dots, T$
        (row $k - 1$), from `gaussx.nonlinear_rts_smoother` with the
        filter's integrator. The smoother gain is
        $G_k = \operatorname{Cov}[x_k, f(x_k)] (P_{k+1}^-)^{-1}$; for affine
        $f(x) = A x$ it is the Kalman gain $P_k A^\top (P_{k+1}^-)^{-1}$.

        Args:
            y: Observations, ``(T, M)``; masked entries may be ``NaN``.
            mask: Optional ``(T,)`` or ``(T, M)`` mask, ``True`` where
                observed.

        Returns:
            ``(smoothed_means, smoothed_covs, log_marginal)``, shaped
            ``(T, D)``, ``(T, D, D)`` and ``()``, as
            `pyrox_ssm.LinearGaussianSSMPrior.smooth` returns.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> prior = pss.NonlinearGaussianSSMPrior(
            ...     jnp.sin, lambda x: x, 0.1 * jnp.eye(1), 0.2 * jnp.eye(1),
            ...     jnp.zeros(1), jnp.eye(1),
            ... )
            >>> m, P, _ = prior.smooth(jnp.array([[0.1], [0.3], [0.2]]))
            >>> m.shape, P.shape
            ((3, 1), (3, 1, 1))
        """
        state = self.filter(y, mask=mask)
        m_smooth, P_smooth = gaussx.nonlinear_rts_smoother(
            state, self.dynamics, integrator=self._integrator(), solver=self.solver
        )
        return m_smooth, P_smooth, state.log_likelihood

    def condition(
        self,
        y: Float[Array, "T M"],
        *,
        mask: Bool[Array, " T"] | Bool[Array, "T M"] | None = None,
    ) -> ConditionedNonlinearGaussianSSM:
        """Condition on ``y`` with the filter and the moment-matched smoother.

        Args:
            y: Observations, ``(T, M)``; masked entries may be ``NaN``.
            mask: Optional ``(T,)`` or ``(T, M)`` mask, ``True`` where
                observed.

        Returns:
            The `ConditionedNonlinearGaussianSSM`, holding ``y`` with its
            masked entries set to zero.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> prior = pss.NonlinearGaussianSSMPrior(
            ...     jnp.sin, lambda x: x, 0.1 * jnp.eye(1), 0.2 * jnp.eye(1),
            ...     jnp.zeros(1), jnp.eye(1),
            ... )
            >>> posterior = prior.condition(jnp.array([[0.1], [0.3], [0.2]]))
            >>> mean, var = posterior.predict()
        """
        y_clean, mask = _check_observations(y, mask)
        m_smooth, P_smooth, log_marg = self.smooth(y_clean, mask=mask)
        return ConditionedNonlinearGaussianSSM(
            prior=self,
            y=y_clean,
            smoothed_means=m_smooth,
            smoothed_covs=P_smooth,
            log_marginal=log_marg,
            mask=mask,
        )


class ConditionedNonlinearGaussianSSM(eqx.Module):
    r"""A `NonlinearGaussianSSMPrior` conditioned on observations.

    Returned by `NonlinearGaussianSSMPrior.condition`. Holds the moments of
    $p(x_k \mid y_{1:T})$ from the moment-matched smoother, row $k - 1$ for
    step $k$, and the surrogate log marginal; `predict` pushes them through
    $h$ for each observation channel.

    Attributes:
        prior: The originating `NonlinearGaussianSSMPrior`.
        y: Observations, ``(T, M)``, with masked entries set to zero.
        smoothed_means: Smoothed state means, ``(T, D)``.
        smoothed_covs: Smoothed state covariances, ``(T, D, D)``.
        log_marginal: The scalar moment-matched surrogate of
            $\log p(y_{\mathrm{obs}} \mid \theta)$.
        mask: The ``(T,)`` or ``(T, M)`` mask, or ``None`` when everything was
            observed.
    """

    prior: NonlinearGaussianSSMPrior
    y: Float[Array, "T M"]
    smoothed_means: Float[Array, "T D"]
    smoothed_covs: Float[Array, "T D D"]
    log_marginal: Float[Array, ""]
    mask: Bool[Array, " T"] | Bool[Array, "T M"] | None = None

    def predict(
        self, *, include_noise: bool = False
    ) -> tuple[Float[Array, "T M"], Float[Array, "T M"]]:
        r"""Mean and variance of each observation channel at every step.

        The smoothed Gaussian of $x_k$ is pushed through $h$ by the prior's
        integrator, `gaussx.moment_transform`,

        $$
        \mathbb{E}[h(x_k) \mid y_{\mathrm{obs}}], \qquad
        \mathrm{Var}[h(x_k)_j \mid y_{\mathrm{obs}}], \qquad
        x_k \sim \mathcal{N}(m_{k \mid T}, P_{k \mid T}),
        $$

        and ``include_noise=True`` adds $(R_k)_{jj}$, the variance of a new
        observation. For affine $h(x) = H x$ this is
        `pyrox_ssm.ConditionedLinearGaussianSSM.predict`. Like the filter,
        it is a moment-matched approximation when $h$ is nonlinear. To
        forecast or fill a gap, extend ``y`` with masked steps and condition
        again.

        Args:
            include_noise: Add the observation-noise variance $(R_k)_{jj}$.
                Default ``False``.

        Returns:
            ``(mean, var)``, each ``(T, M)``.

        Examples:
            >>> import jax.numpy as jnp
            >>> import pyrox_ssm as pss
            >>> prior = pss.NonlinearGaussianSSMPrior(
            ...     lambda x: 0.8 * x, jnp.exp, 0.1 * jnp.eye(1), 0.05 * jnp.eye(1),
            ...     jnp.zeros(1), jnp.eye(1),
            ... )
            >>> posterior = prior.condition(jnp.array([[1.1], [0.9], [1.0]]))
            >>> mean, band = posterior.predict(include_noise=True)
            >>> mean.shape
            (3, 1)
        """
        integrator = self.prior._integrator()

        def step(m, P):
            mean, cov, _ = gaussx.moment_transform(
                self.prior.obs_fn, m, P, integrator=integrator
            )
            return mean, jnp.diagonal(cov)

        mean, var = jax.vmap(step)(self.smoothed_means, self.smoothed_covs)
        if include_noise:
            R = self.prior.obs_noise
            if isinstance(R, lx.AbstractLinearOperator):
                R = R.as_matrix()
            # diag(R_k): (M,) for one R or (T, M) for one per step; broadcasts.
            var = var + jnp.diagonal(jnp.asarray(R), axis1=-2, axis2=-1)
        return mean, var


def nonlinear_ssm_factor(
    name: str,
    dynamics: Callable[[Float[Array, " D"]], Float[Array, " D"]],
    obs_fn: Callable[[Float[Array, " D"]], Float[Array, " M"]],
    process_noise: Float[Array, "D D"]
    | Float[Array, "T D D"]
    | lx.AbstractLinearOperator,
    obs_noise: Float[Array, "M M"] | Float[Array, "T M M"] | lx.AbstractLinearOperator,
    y: Float[Array, "T M"],
    init_mean: Float[Array, " D"],
    init_cov: Float[Array, "D D"],
    *,
    integrator: AbstractIntegrator | None = None,
    mask: Bool[Array, " T"] | Bool[Array, "T M"] | None = None,
    joseph: bool = True,
    solver: AbstractSolverStrategy | None = None,
) -> FilterState:
    r"""Register a nonlinear state-space model's log likelihood as a factor.

    With $T$ observation steps, a state of length $D$ and observations of
    length $M$, the model is

    $$
    x_0 \sim \mathcal{N}(m_0, P_0), \qquad
    x_k = f(x_{k-1}) + \eta_k, \quad \eta_k \sim \mathcal{N}(0, Q_k), \qquad
    y_k = h(x_k) + \varepsilon_k, \quad \varepsilon_k \sim \mathcal{N}(0, R_k),
    $$

    for $k = 1, \dots, T$, with $f$ the ``dynamics`` and $h$ the
    ``obs_fn``, and row $k - 1$ of ``y`` holds step $k$. This is the
    convention of `gaussx.nonlinear_kalman_filter` (and of
    `pyrox_ssm.LinearGaussianSSMPrior`), which predicts before every
    update: the first observation ``y[0]`` is $y_1$, scored against
    $f(x_0) + \eta_1$, not against $x_0$.

    `gaussx.nonlinear_kalman_filter` pushes the Gaussian belief through $f$
    and $h$ by the integrator's moment transform. With
    $m_k^-, P_k^-$ the predicted moments of $x_k$ given $y_{1:k-1}$, the
    update matches

    $$
    \hat{y}_k = \mathbb{E}[h(x_k)], \qquad
    S_k = \operatorname{Cov}[h(x_k)] + R_k, \qquad
    x_k \sim \mathcal{N}(m_k^-, P_k^-),
    $$

    and the factor is the prediction-error sum

    $$
    \log p(y_{1:T} \mid \theta) \approx \sum_{k=1}^{T}
      \log \mathcal{N}(y_k;\, \hat{y}_k,\, S_k).
    $$

    In gaussx's words, this ``log_likelihood`` "is a moment-matched
    surrogate, not the exact marginal likelihood: Σ_y,t is the matched
    observation covariance rather than the true one. It is exact when f and
    h are affine." Maximising it, or sampling under it, targets that
    surrogate.

    The integrator picks the filter:

    | integrator | filter |
    |---|---|
    | `gaussx.TaylorIntegrator` | extended Kalman filter (EKF) |
    | `gaussx.UnscentedIntegrator` | unscented Kalman filter (UKF), the default |
    | `gaussx.CubatureIntegrator` | cubature Kalman filter (CKF) |
    | `gaussx.FifthOrderCubatureIntegrator` | degree-5 cubature Kalman filter |
    | `gaussx.GaussHermiteIntegrator` | Gauss-Hermite Kalman filter (GHKF) |
    | `gaussx.MonteCarloIntegrator` | Monte Carlo Kalman filter |

    Masked entries of ``y`` are replaced by zero before the filter runs, so
    a ``NaN`` placeholder there reaches neither the value nor a gradient. A
    per-step mask, shaped ``(T,)``, runs the predict only at a masked step;
    a per-entry mask, shaped ``(T, M)``, marginalises the masked entries
    out of the update, so either one scores exactly the observed entries.

    Registers the one NumPyro site ``name``, a factor, whose value is
    ``NonlinearGaussianSSMPrior(dynamics, obs_fn, ...).log_marginal(y,
    mask=mask)``; build that prior to smooth and predict after a fit.
    Library code adds no `jax.jit`; jit the model or the loss at the top.

    Args:
        name: The factor site name.
        dynamics: Deterministic dynamics $f$, from a ``(D,)`` state to a
            ``(D,)`` state; the process noise is added separately.
        obs_fn: Observation map $h$, from a ``(D,)`` state to an ``(M,)``
            observation.
        process_noise: Process-noise covariance $Q_k$: one ``(D, D)``
            matrix for every step, a ``(T, D, D)`` stack, or a lineax
            operator.
        obs_noise: Observation-noise covariance $R_k$, ``(M, M)``,
            ``(T, M, M)`` or a lineax operator.
        y: Observations $y_{1:T}$, shaped ``(T, M)``; masked entries may be
            ``NaN``.
        init_mean: Mean $m_0$ of the initial state $x_0$, shaped ``(D,)``.
        init_cov: Covariance $P_0$ of $x_0$, shaped ``(D, D)``.
        integrator: The gaussx moment-matching rule. ``None`` (the default)
            is ``gaussx.UnscentedIntegrator(alpha=1.0)``, the default of
            `gaussx.nonlinear_kalman_filter`: derivative-free and exact for
            affine maps. `gaussx.UnscentedIntegrator`'s own default
            ``alpha=1e-3`` recovers the moments by cancellation and loses
            about six digits, which float32 cannot afford.
        mask: Optional boolean mask, ``(T,)`` per step or ``(T, M)`` per
            entry, ``True`` where observed. ``None`` observes everything.
        joseph: Use the Joseph-form covariance update, passed through to
            `gaussx.nonlinear_kalman_filter`. Defaults to ``True``.
        solver: Solver strategy for $S_k$, passed through to
            `gaussx.nonlinear_kalman_filter`. ``None`` factorises $S_k$ once
            by Cholesky.

    Returns:
        The `gaussx.FilterState`: filtered and predicted means ``(T, D)``
        and covariances ``(T, D, D)``, row $k - 1$ holding step $k$, and the
        scalar ``log_likelihood`` registered as the factor.

    Raises:
        ValueError: If ``y`` is not ``(T, M)`` or ``mask`` is not ``(T,)``
            or ``(T, M)``; gaussx raises for misshapen noise covariances.

    Examples:
        A log-emission AR(1) state, $x_k = a x_{k-1} + \eta_k$ observed
        through $y_k = \exp(x_k) + \varepsilon_k$, with a prior on $a$:

        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> import pyrox_ssm as pss
        >>> def model(y):
        ...     a = numpyro.sample("a", dist.Uniform(-1.0, 1.0))
        ...     pss.nonlinear_ssm_factor(
        ...         "y",
        ...         lambda x: a * x,
        ...         jnp.exp,
        ...         0.1 * jnp.eye(1),
        ...         0.05 * jnp.eye(1),
        ...         y,
        ...         jnp.zeros(1),
        ...         jnp.eye(1),
        ...     )
        >>> y = jnp.array([[1.1], [0.9], [1.3], [1.0]])
        >>> trace = numpyro.handlers.trace(numpyro.handlers.seed(model, 0)).get_trace(y)
        >>> sorted(trace)
        ['a', 'y']
    """
    prior = NonlinearGaussianSSMPrior(
        dynamics,
        obs_fn,
        process_noise,
        obs_noise,
        init_mean,
        init_cov,
        integrator=integrator,
        joseph=joseph,
        solver=solver,
    )
    state = prior.filter(y, mask=mask)
    numpyro.factor(name, state.log_likelihood)
    return state
