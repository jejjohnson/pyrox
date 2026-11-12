r"""The ensemble Kalman log likelihood of a filterax assimilation run.

`ensemble_factor` adds the log likelihood of `filterax.differentiable_assimilate`
to a NumPyro model, for state-space models whose state is too large for a
moment filter. The wrapper fixes what makes that log density usable for
inference: the initial ensemble is data, so the random numbers are fixed, and
only deterministic square-root filters are accepted, so the log density is a
smooth function of the parameters.

Needs the ``ensemble`` extra (``pip install 'pyrox-ssm[ensemble]'``), which
installs filterax. ``pyrox_ssm`` itself never imports this module.
"""

from __future__ import annotations

from typing import Any


try:
    import filterax as flx
except ImportError as exc:
    raise ImportError(
        "pyrox_ssm.ensemble requires the ensemble extra: "
        "pip install 'pyrox-ssm[ensemble]'"
    ) from exc

import jax.numpy as jnp
import lineax as lx
import numpyro
from jaxtyping import Array, Float


# The filter and inflator flavours filterax.differentiable_assimilate refuses
# because they draw fresh random numbers at every analysis step.
_STOCHASTIC = (
    flx.filters.StochasticEnKF,
    flx.filters.ETKF_Livings,
    flx.AdditiveInflator,
)


def ensemble_factor(
    name: str,
    filter_: flx.AbstractSequentialFilter,
    dynamics: flx.AbstractDynamics,
    obs_op: flx.AbstractObsOperator,
    init_ensemble: Float[Array, "E D"],
    y: Float[Array, "T M"],
    obs_times: Float[Array, " T"],
    obs_noise: Float[Array, "M M"] | lx.AbstractLinearOperator,
    *,
    inflator: flx.AbstractInflator | None = None,
    t0: float | Float[Array, ""] = 0.0,
    adjoint: Any | None = None,
    **analysis_extra: Any,
) -> flx.AssimilationResult:
    r"""Register the ensemble Kalman log likelihood as a NumPyro factor.

    With an ensemble of $E$ members, a state of length $D$, observations of
    length $M$ at $T$ times, deterministic dynamics $f_\theta$ and an
    observation operator $h_\theta$, the filter forecasts every member from
    the previous analysis time to $t_k$ and scores $y_k$ under the Gaussian
    the forecast ensemble implies,

    $$
    \bar{r}_k = y_k - \frac{1}{E} \sum_{e=1}^{E} h_\theta(x_{k,e}^-), \qquad
    \hat{\Sigma}_{y,k} = U_k U_k^\top + \Sigma_\varepsilon,
    $$

    with $U_k$ the predicted-observation anomalies scaled by
    $1 / \sqrt{E - 1}$ and $\Sigma_\varepsilon$ the ``obs_noise``. The
    factor is the prediction-error sum

    $$
    \log p(y_{1:T} \mid \theta) \approx \sum_{k=1}^{T}
      \log \mathcal{N}(\bar{r}_k;\, 0,\, \hat{\Sigma}_{y,k}).
    $$

    Row $k - 1$ of ``y`` holds step $k$, observed at ``obs_times[k - 1]``;
    the first forecast runs from ``t0`` to ``obs_times[0]``.

    The log density is deterministic in $\theta$. The initial ensemble
    $X_0$ is data, drawn once outside the model, and the filter is a
    square-root filter, so every evaluation sees the same random numbers
    (common random numbers) and the log density is a smooth function of
    $\theta$. Draw $X_0$ once, before inference, and pass the same array to
    every call. Filters and inflators that draw at each step
    (``filterax.filters.StochasticEnKF``, ``filterax.filters.ETKF_Livings``,
    ``filterax.AdditiveInflator``) raise.

    The log likelihood is biased, not exact: the ensemble mean and
    covariance are Monte Carlo estimates, and the bias shrinks as $E$
    grows. For linear-Gaussian dynamics and observations it converges to
    the Kalman filter's exact log marginal (`pyrox_ssm.lgssm_factor`).
    Report $E$ with any result.

    Fitting: SVI or MAP is the default; NUTS suits a small set of
    parameters, since every gradient is one assimilation run and its
    backward pass; when $f_\theta$ or $h_\theta$ cannot be differentiated,
    use filterax's ensemble Kalman inversion (``filterax.EKI``) instead of
    this factor.

    Registers the one NumPyro site ``name``, a factor. Library code adds no
    `jax.jit`; jit the model or the loss at the top.

    Args:
        name: The factor site name.
        filter_: A deterministic square-root filter from
            ``filterax.filters``: ``ETKF``, ``EnSRF``, ``ESTKF``,
            ``EnSRF_Serial`` or ``LETKF``.
        dynamics: Deterministic dynamics $f_\theta$, called per member as
            ``dynamics(x, t_prev, t_next)`` on a ``(D,)`` state: a
            ``filterax.AbstractDynamics``, or a plain callable, which filterax
            also accepts.
        obs_op: Observation operator $h_\theta$, from a ``(D,)`` state to an
            ``(M,)`` observation: a ``filterax.AbstractObsOperator`` or a plain
            callable.
        init_ensemble: The initial ensemble $X_0$, shaped ``(E, D)``, drawn
            once outside the model.
        y: Observations $y_{1:T}$, shaped ``(T, M)``.
        obs_times: Observation times, shaped ``(T,)``.
        obs_noise: Observation-noise covariance $\Sigma_\varepsilon$, an
            ``(M, M)`` matrix (wrapped as a positive semi-definite lineax
            operator) or a lineax operator.
        inflator: Optional deterministic inflator
            (``filterax.MultiplicativeInflator``, ``RTPS`` or ``RTPP``).
        t0: The time of $X_0$, where the first forecast starts.
        adjoint: The gradient strategy from ``filterax.differentiable``.
            ``None`` (the default) is ``RecursiveCheckpointAdjoint()``: exact
            gradients that recompute the forecasts in the backward pass,
            trading compute for memory; ``DirectAdjoint()`` keeps the full
            tape.
        **analysis_extra: Passed to ``filter_.analysis``; ``LETKF`` takes
            ``state_coords`` and ``obs_coords`` here.

    Returns:
        The ``filterax.AssimilationResult``: the final ``particles``
        ``(E, D)``, the ``forecast_history`` and ``analysis_history``
        ``(T, E, D)``, and the per-step ``log_likelihoods`` ``(T,)`` whose
        sum is the factor.

    Raises:
        ValueError: If ``filter_`` or ``inflator`` draws random numbers at
            each step; filterax raises when ``y`` and ``obs_times`` have
            different lengths.

    Examples:
        A damped rotation of a 2-D state, observed in its first
        coordinate, with a prior on the damping:

        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>> import filterax as flx
        >>> from pyrox_ssm.ensemble import ensemble_factor
        >>> rotation = jnp.array([[0.8, -0.6], [0.6, 0.8]])
        >>> X0 = jr.normal(jr.key(0), (64, 2))  # E = 64, drawn once
        >>> def model(y, obs_times):
        ...     a = numpyro.sample("a", dist.Uniform(0.5, 1.0))
        ...     ensemble_factor(
        ...         "y",
        ...         flx.filters.ETKF(),
        ...         lambda x, t0, t1: a * rotation @ x,
        ...         lambda x: x[:1],
        ...         X0,
        ...         y,
        ...         obs_times,
        ...         0.1 * jnp.eye(1),
        ...     )
        >>> y = jnp.array([[0.5], [0.2], [-0.1]])
        >>> trace = numpyro.handlers.trace(numpyro.handlers.seed(model, 0)).get_trace(
        ...     y, jnp.array([1.0, 2.0, 3.0])
        ... )
        >>> sorted(trace)
        ['a', 'y']
    """
    if not isinstance(obs_noise, lx.AbstractLinearOperator):
        obs_noise = lx.MatrixLinearOperator(
            jnp.asarray(obs_noise), lx.positive_semidefinite_tag
        )
    if adjoint is None:
        adjoint = flx.differentiable.RecursiveCheckpointAdjoint()
    try:
        result = flx.differentiable_assimilate(
            filter_,
            dynamics,
            obs_op,
            init_ensemble,
            y,
            obs_times,
            obs_noise,
            inflator=inflator,
            t0=t0,
            adjoint=adjoint,
            **analysis_extra,
        )
    except ValueError as exc:
        stochastic = next(
            (c for c in (filter_, inflator) if isinstance(c, _STOCHASTIC)), None
        )
        if stochastic is None:
            raise
        raise ValueError(
            f"ensemble_factor needs a deterministic square-root filter and "
            f"inflator, but {type(stochastic).__name__} draws random numbers at "
            "each analysis step. The log density is deterministic and smooth "
            "in the parameters only when the initial ensemble is data and the "
            "filter is a square-root filter, so that every evaluation sees the "
            "same draws (common random numbers). Use ETKF, EnSRF, ESTKF, "
            "EnSRF_Serial or LETKF from filterax.filters, and "
            "MultiplicativeInflator, RTPS or RTPP."
        ) from exc
    # differentiable_assimilate always fills log_likelihoods (one per step)
    log_likelihoods = result.log_likelihoods
    assert log_likelihoods is not None
    numpyro.factor(name, log_likelihoods.sum())
    return result


__all__ = ["ensemble_factor"]
