"""Pure-JAX primitives for the Generalized Pareto Distribution.

GPD models threshold exceedances in the peaks-over-threshold (POT)
framework. The location is pinned to zero (``x`` is the excess over
the threshold); the scale ``σ > 0`` and shape ``ξ`` determine the tail.

The :math:`\\xi = 0` limit is the exponential distribution; a
``jnp.where`` branch handles that case so both formulas are traceable.

All functions are stateless, ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` safe.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


_EXPONENTIAL_THRESHOLD = 1e-7


def _safe_shape(shape: Float[Array, ...]) -> Float[Array, ...]:
    shape = jnp.asarray(shape)
    is_exp = jnp.abs(shape) < _EXPONENTIAL_THRESHOLD
    return jnp.where(is_exp, 1.0, shape)


def gpd_log_prob(
    x: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Log PDF of the Generalized Pareto Distribution (loc = 0)."""
    shape = jnp.asarray(shape)
    is_exp = jnp.abs(shape) < _EXPONENTIAL_THRESHOLD
    xi = _safe_shape(shape)

    # Support: x >= 0; when ξ < 0 (and we are not in the exponential-limit
    # regime) also x <= -σ/ξ. For ξ ≈ 0 the safe-shape substitution sets
    # `xi = 1`, so `-scale/xi` would incorrectly produce `-scale` here —
    # exclude the exponential branch from the finite-upper-bound case.
    upper_bound = jnp.where((shape < 0) & ~is_exp, -scale / shape, jnp.inf)
    in_support = (x >= 0.0) & (x <= upper_bound)

    exponential = -jnp.log(scale) - x / scale

    t = 1.0 + xi * x / scale
    t_valid = t > 0.0
    t_safe = jnp.where(t_valid, t, 1.0)
    pareto = -jnp.log(scale) - (1.0 / xi + 1.0) * jnp.log(t_safe)
    pareto = jnp.where(t_valid, pareto, -jnp.inf)

    result = jnp.where(is_exp, exponential, pareto)
    return jnp.where(in_support, result, -jnp.inf)


def gpd_cdf(
    x: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """CDF of the Generalized Pareto Distribution."""
    shape = jnp.asarray(shape)
    is_exp = jnp.abs(shape) < _EXPONENTIAL_THRESHOLD
    xi = _safe_shape(shape)

    exponential = 1.0 - jnp.exp(-x / scale)

    t = 1.0 + xi * x / scale
    t_valid = t > 0.0
    t_safe = jnp.where(t_valid, t, 1.0)
    pareto = 1.0 - jnp.power(t_safe, -1.0 / xi)
    # Beyond the support, ξ > 0 tails map out-of-support CDF to 0 (lower
    # bound); ξ < 0 tails map it to 1 (finite upper bound, including the
    # endpoint x = -σ/ξ where t = 0 exactly).
    boundary = jnp.where(shape < 0, 1.0, 0.0)
    pareto = jnp.where(t_valid, pareto, boundary)

    cdf = jnp.where(is_exp, exponential, pareto)
    # Clamp the universal lower bound (x < 0 is always out of support).
    return jnp.where(x >= 0.0, cdf, 0.0)


def gpd_icdf(
    q: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Quantile function (inverse CDF) of the GPD."""
    shape = jnp.asarray(shape)
    is_exp = jnp.abs(shape) < _EXPONENTIAL_THRESHOLD
    xi = _safe_shape(shape)

    exponential = -scale * jnp.log(1.0 - q)
    pareto = (scale / xi) * (jnp.power(1.0 - q, -xi) - 1.0)

    return jnp.where(is_exp, exponential, pareto)


def gpd_mean(
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    r"""GPD mean: :math:`\sigma / (1 - \xi)` for :math:`\xi < 1`, else ``+inf``."""
    shape = jnp.asarray(shape)
    # Substitute a safe divisor where shape >= 1 so the masked-out branch
    # traces without a concrete Python ZeroDivisionError.
    denom = jnp.where(shape < 1.0, 1.0 - shape, 1.0)
    mean_val = scale / denom
    return jnp.where(shape < 1.0, mean_val, jnp.inf)


def gpd_return_level(
    period: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    r"""T-period return level: :math:`Q(1 - 1/T)`."""
    return gpd_icdf(1.0 - 1.0 / period, scale, shape)
