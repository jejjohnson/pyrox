"""Pure-JAX primitives for non-stationary GEV return levels.

Block-maxima extremes are non-stationary when the GEV parameters vary
across blocks (years) — for example a warming trend that shifts the
location parameter. These helpers operate directly on GEV parameter
*fields* of shape ``(n_blocks, ...)`` rather than scalar parameters.

The core quantities are:

- :func:`assemble_nonstationary_gev_fields` — build ``(loc, scale, shape)``
  fields from a spatial intercept, one or more covariate trends, and a
  log-scale field, following :math:`\\mu_t(s) = \\mu_0(s) + x_t\\,\\mu_1(s)`.
- :func:`expected_exceedances` — expected number of block exceedances of a
  threshold, :math:`\\sum_t \\Pr(Y_t > z)`.
- :func:`nonstationary_return_period` / :func:`nonstationary_return_level`
  — the *effective* return period / level obtained from the average
  exceedance rate over a block sequence (Coles 2001, §6; the
  "expected number of events" definition).

Everything is stateless and ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` safe.
The stationary special case (``time_axis=None``) reduces to the closed-form
quantiles already provided by :func:`xtremax.primitives.gev.gev_return_level`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from xtremax.primitives.gev import gev_return_level, gev_survival


# Hard cap on geometric upper-bracket growth so the solve always terminates
# under ``jax.jit``. The residual tends to ``-target`` as the level grows, so
# a valid bracket is reached in only a handful of doublings in practice.
_MAX_BRACKET_EXPANSIONS = 64


def assemble_nonstationary_gev_fields(
    mu0: Float[Array, ...],
    mu1: Float[Array, ...],
    log_sigma: Float[Array, ...],
    shape: Float[Array, ...],
    time_covariate: Float[Array, ...],
) -> tuple[Float[Array, ...], Float[Array, ...], Float[Array, ...]]:
    r"""Build non-stationary spatio-temporal GEV parameter fields.

    Combines per-site intercept, trend, and log-scale fields with a
    temporal covariate into block-by-site GEV parameters:

    .. math::

        \mu_t(s) &= \mu_0(s) + \sum_k x_{t,k}\,\mu_{1,k}(s) \\
        \sigma_t(s) &= \exp(\eta_\sigma(s)) \\
        \xi_t(s) &= \xi

    The scale is constant in time and the shape is broadcast from a scalar
    or per-site value; only the location varies with the covariate. This is
    the parameterization underneath hierarchical non-stationary GEV models,
    decoupled from any particular prior / GP / regression machinery.

    Args:
        mu0: Spatial intercept :math:`\mu_0(s)` of shape ``(n_sites,)``.
        mu1: Trend coefficient field of shape ``(n_sites,)`` for a single
            temporal covariate, or ``(n_sites, n_covariates)`` for several.
        log_sigma: Log-scale field :math:`\log \sigma(s)` of shape
            ``(n_sites,)``.
        shape: GEV shape :math:`\xi`, a scalar or per-site array of shape
            ``(n_sites,)``.
        time_covariate: Temporal covariate of shape ``(n_blocks,)`` for a
            single covariate, or ``(n_blocks, n_covariates)`` for several.

    Returns:
        Tuple ``(loc, scale, shape)`` of fields, each of shape
        ``(n_blocks, n_sites)``.
    """
    mu0 = jnp.asarray(mu0)
    mu1 = jnp.asarray(mu1)
    log_sigma = jnp.asarray(log_sigma)
    time = jnp.asarray(time_covariate)

    if time.ndim == 1:
        time = time[:, None]
    if mu1.ndim == 1:
        mu1 = mu1[:, None]

    # x_{t,k} mu_{1,k}(s) summed over covariates k -> (n_blocks, n_sites).
    time_comp = jnp.einsum("tk,nk->tn", time, mu1)
    loc = mu0[None, :] + time_comp

    scale = jnp.broadcast_to(jnp.exp(log_sigma)[None, :], loc.shape)
    shape_field = jnp.broadcast_to(jnp.asarray(shape), loc.shape)
    return loc, scale, shape_field


def expected_exceedances(
    threshold: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
    time_axis: int | None = None,
) -> Float[Array, ...]:
    r"""Expected number of block exceedances of a threshold.

    For GEV block maxima the per-block exceedance probability is the
    survival function :math:`\Pr(Y_t > z) = 1 - F(z)`. Summing over blocks
    gives the expected number of exceedances:

    .. math:: \Lambda(z) = \sum_t \Pr(Y_t > z).

    Args:
        threshold: Threshold / candidate return level ``z``, broadcastable
            to the non-time dimensions of the parameter fields.
        loc: GEV location field.
        scale: GEV scale field.
        shape: GEV shape field.
        time_axis: Axis indexing blocks. If ``None``, the single-block
            exceedance probability is returned instead of a summed count.

    Returns:
        Either :math:`\Pr(Y > z)` (``time_axis is None``) or
        :math:`\sum_t \Pr(Y_t > z)` (summed over ``time_axis``).
    """
    threshold = jnp.asarray(threshold)
    if time_axis is None:
        return gev_survival(threshold, loc, scale, shape)
    # Honor the documented contract that ``threshold`` broadcasts against the
    # non-time dimensions for *any* ``time_axis`` (not just a leading one) by
    # inserting the block axis. A scalar already broadcasts everywhere.
    if threshold.ndim > 0:
        threshold = jnp.expand_dims(threshold, time_axis)
    survival = gev_survival(threshold, loc, scale, shape)
    return jnp.sum(survival, axis=time_axis)


def nonstationary_return_period(
    threshold: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
    time_axis: int | None = None,
) -> Float[Array, ...]:
    r"""Return period associated with a threshold under (non-)stationary GEV.

    Stationary (``time_axis is None``):

    .. math:: T(z) = 1 / \Pr(Y > z).

    Non-stationary (``time_axis`` given): the *effective* return period from
    the average exceedance rate over the supplied blocks,

    .. math:: T_\text{eff}(z) = N_\text{blocks} / \sum_t \Pr(Y_t > z).

    Args:
        threshold: Threshold ``z``, broadcastable to the non-time dimensions.
        loc: GEV location field.
        scale: GEV scale field.
        shape: GEV shape field.
        time_axis: Axis indexing blocks, or ``None`` for the stationary case.

    Returns:
        Return period with the shape of the non-time dimensions.
    """
    count = expected_exceedances(threshold, loc, scale, shape, time_axis=time_axis)
    if time_axis is None:
        return 1.0 / count
    n_blocks = jnp.asarray(loc).shape[time_axis]
    return n_blocks / count


def _bisect_monotone_decreasing(
    fn,
    lower: Float[Array, ...],
    upper: Float[Array, ...],
    num_steps: int = 64,
) -> Float[Array, ...]:
    """Solve ``fn(x) = 0`` for a monotone decreasing function via bisection.

    ``fn`` must satisfy ``fn(lower) >= 0`` and ``fn(upper) <= 0`` elementwise.
    The loop carries only arrays and uses ``jax.lax.fori_loop``, so it stays
    ``jax.jit`` safe and avoids host-device synchronization.
    """
    lower, upper = jnp.broadcast_arrays(jnp.asarray(lower), jnp.asarray(upper))

    def body(_, bracket):
        low, high = bracket
        mid = 0.5 * (low + high)
        # Decreasing fn: a positive residual means the root is to the right.
        move_lower = fn(mid) > 0
        return jnp.where(move_lower, mid, low), jnp.where(move_lower, high, mid)

    low, high = jax.lax.fori_loop(0, num_steps, body, (lower, upper))
    return 0.5 * (low + high)


def _expand_upper_bracket(
    fn,
    lower: Float[Array, ...],
    upper: Float[Array, ...],
    max_expansions: int = _MAX_BRACKET_EXPANSIONS,
) -> Float[Array, ...]:
    """Grow ``upper`` until ``fn(upper) <= 0`` everywhere, for decreasing ``fn``.

    Doubles the gap ``upper - lower`` until the residual is non-positive at the
    upper end (or the expansion cap is hit), guaranteeing a valid bracket for
    :func:`_bisect_monotone_decreasing`. Only the elements that still need it
    are grown each step, so one heavy-tailed site does not inflate the bracket
    of already-bracketed sites (which would waste the fixed bisection budget on
    a needlessly wide interval). Uses ``jax.lax.while_loop`` to stay
    ``jax.jit`` safe.
    """
    lower, upper = jnp.broadcast_arrays(jnp.asarray(lower), jnp.asarray(upper))

    def cond(state):
        up, i = state
        return jnp.any(fn(up) > 0) & (i < max_expansions)

    def body(state):
        up, i = state
        needs_growth = fn(up) > 0
        grown = lower + 2.0 * (up - lower)
        return jnp.where(needs_growth, grown, up), i + 1

    upper, _ = jax.lax.while_loop(cond, body, (upper, 0))
    return upper


def nonstationary_return_level(
    return_period: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
    time_axis: int | None = None,
    num_bisection_steps: int = 64,
    lower: Float[Array, ...] | None = None,
    upper: Float[Array, ...] | None = None,
) -> Float[Array, ...]:
    r"""Return level for stationary or non-stationary GEV parameter fields.

    Stationary (``time_axis is None``) uses the closed-form quantile
    :math:`z_T = F^{-1}(1 - 1/T)`.

    Non-stationary (``time_axis`` given): there is no single closed-form
    quantile because the parameters vary by block, so ``z_T`` is found by
    bisection such that the expected number of exceedances over the supplied
    blocks equals :math:`N_\text{blocks} / T`:

    .. math:: \sum_t \Pr(Y_t > z_T) = N_\text{blocks} / T.

    Args:
        return_period: Return period :math:`T > 1`, broadcastable to the
            non-time dimensions.
        loc: GEV location field.
        scale: GEV scale field.
        shape: GEV shape field.
        time_axis: Axis indexing blocks, or ``None`` for the stationary case.
        num_bisection_steps: Bisection iterations for the non-stationary solve.
        lower: Optional lower bracket for the non-stationary solve. Defaults
            to a conservative value derived from ``loc`` and ``scale``.
        upper: Optional starting upper bracket. It is grown automatically until
            it brackets the root, so an underestimate (including the default)
            is corrected rather than silently returned.

    Returns:
        Return level :math:`z_T` with the shape of the non-time dimensions.
    """
    loc = jnp.asarray(loc)
    scale = jnp.asarray(scale)
    shape = jnp.asarray(shape)

    if time_axis is None:
        # Route through the stable return-level path (parameterized by
        # -log1p(-1/T)) so large periods do not lose precision to the
        # 1 - 1/T rounding that a direct gev_icdf(1 - 1/T) would incur.
        return gev_return_level(return_period, loc, scale, shape)

    n_blocks = loc.shape[time_axis]
    target = n_blocks / return_period

    def residual(level):
        return (
            expected_exceedances(level, loc, scale, shape, time_axis=time_axis) - target
        )

    if lower is None:
        lower = jnp.min(loc - 8.0 * scale, axis=time_axis)
    if upper is None:
        upper = jnp.max(loc + 20.0 * scale, axis=time_axis)

    lower, upper = jnp.broadcast_arrays(jnp.asarray(lower), jnp.asarray(upper))

    def solve(fn, initial_guess):
        del initial_guess
        # The residual is monotone decreasing in ``level`` and tends to
        # ``-target`` as ``level -> +inf`` (every block's survival vanishes,
        # including past the finite Weibull endpoint where ``gev_survival``
        # saturates to 0). A fixed ``loc + 20*scale`` underestimates heavy
        # Fréchet tails, so grow the upper bracket until it brackets the root.
        up = _expand_upper_bracket(fn, lower, upper)
        return _bisect_monotone_decreasing(fn, lower, up, num_steps=num_bisection_steps)

    # ``custom_root`` supplies the derivative via the implicit function theorem
    # (dz/dθ = -(∂R/∂θ)/(∂R/∂z)) instead of differentiating the bracketing
    # ``while_loop`` (which reverse-mode AD cannot handle) or the discrete
    # bisection comparisons (whose gradient is identically zero). The linear
    # tangent problem is diagonal, so it is solved elementwise.
    def tangent_solve(g, y):
        return y / g(jnp.ones_like(y))

    return jax.lax.custom_root(residual, lower, solve, tangent_solve)
