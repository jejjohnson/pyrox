r"""A nonlinear state-space model as a NumPyro factor.

`nonlinear_ssm_factor` scores a series under a state-space model with
nonlinear dynamics or observations, by the moment-matched Gaussian filter
`gaussx.nonlinear_kalman_filter`, and adds the resulting log likelihood to a
NumPyro model. The choice of `gaussx` integrator is the choice of filter
(extended, unscented, cubature, Gauss-Hermite or Monte Carlo Kalman filter);
the wrapper fixes the conventions around it in one place: the default
integrator, the observation mask and the handling of masked entries.
"""

from __future__ import annotations

from collections.abc import Callable

import einx
import gaussx
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

    Registers the one NumPyro site ``name``, a factor. Library code adds no
    `jax.jit`; jit the model or the loss at the top.

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
    y, mask = _check_observations(y, mask)
    if integrator is None:
        integrator = gaussx.UnscentedIntegrator(alpha=1.0)
    state = gaussx.nonlinear_kalman_filter(
        dynamics,
        obs_fn,
        process_noise,
        obs_noise,
        y,
        init_mean,
        init_cov,
        integrator=integrator,
        mask=mask,
        joseph=joseph,
        solver=solver,
    )
    numpyro.factor(name, state.log_likelihood)
    return state
