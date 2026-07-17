"""Pure-JAX primitives for the Generalized Pareto Distribution.

GPD models threshold exceedances in the peaks-over-threshold (POT)
framework. The location is pinned to zero (``x`` is the excess over
the threshold); the scale ``σ > 0`` and shape ``ξ`` determine the tail.

The :math:`\\xi = 0` limit is the exponential distribution. As with the GEV
primitives, the limit is handled *smoothly* via
:func:`xtremax.primitives._common.log1p_over_x` / ``expm1_over_x`` rather than a
hard ``jnp.where`` branch, so values stay accurate and gradients finite through
:math:`\\xi = 0` — and there is no threshold constant to drift against the
Distribution layer.

All functions are stateless, ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` safe.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from xtremax.primitives._common import (
    clamped_standardize,
    expm1_over_x,
    log1p_over_x,
)


def _reduced_exponent(
    r: Float[Array, ...], v_safe: Float[Array, ...], valid: Array
) -> Array:
    r"""Return :math:`w = \log(1 + \xi x/\sigma)/\xi`, so ``t^{-1/ξ} = exp(-w)``.

    ``r = x/σ`` is the (clamped) standardized value and ``v_safe = ξ r`` must
    already be masked to ``> -1``. As ``ξ → 0`` this tends to ``r`` (the
    exponential rate term), smoothly.
    """
    # Out-of-support points carry a large ``r``; zero it so the downstream
    # ``exp(-w)`` cannot overflow into a NaN gradient in the discarded branch.
    r_safe = jnp.where(valid, r, 0.0)
    return r_safe * log1p_over_x(v_safe)


def _support(r: Float[Array, ...], shape: Float[Array, ...]) -> tuple[Array, Array]:
    r"""Return ``(valid, v_safe)`` for ``v = ξ·r`` and the support ``v > -1``.

    ``r = x/σ`` is the clamped standardized value, so ``v = ξr`` is always finite
    (even when a tiny ``σ`` would otherwise overflow the raw ratio).
    """
    v = shape * r
    valid = v > -1.0
    return valid, jnp.where(valid, v, 0.0)


def _standardize_exceedance(x: Float[Array, ...], scale: Float[Array, ...]) -> Array:
    """Clamped ``x/σ`` (loc is 0 for the GPD)."""
    return clamped_standardize(x, 0.0, scale)


def gpd_log_prob(
    x: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Log PDF of the Generalized Pareto Distribution (loc = 0)."""
    shape = jnp.asarray(shape)
    x = jnp.asarray(x)
    finite = jnp.isfinite(x)
    x_calc = jnp.where(finite, x, 0.0)
    r = _standardize_exceedance(x_calc, scale)
    valid, v_safe = _support(r, shape)

    w = _reduced_exponent(r, v_safe, valid)
    log_t = jnp.log1p(v_safe)
    # -log σ - (1/ξ + 1) log(1 + ξx/σ) = -log σ - (w + log_t); → -log σ - x/σ.
    log_pdf = -jnp.log(scale) - (w + log_t)

    # Density vanishes at x = ±inf; NaN input propagates.
    in_support = (x >= 0.0) & valid & finite
    result = jnp.where(in_support, log_pdf, -jnp.inf)
    return jnp.where(jnp.isnan(x), jnp.nan, result)


def gpd_cdf(
    x: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """CDF of the Generalized Pareto Distribution."""
    shape = jnp.asarray(shape)
    x = jnp.asarray(x)
    # x < 0 (incl. -inf) is below support; x = +inf is the unbounded tail. Both
    # get a finite in-support surrogate so no overflowing / parameter-dependent
    # inf reaches the kernel, then are overridden with their limits below.
    below = x < 0.0
    posinf = jnp.isposinf(x)
    x_calc = jnp.where(below | posinf, 0.0, x)
    r = _standardize_exceedance(x_calc, scale)
    valid, v_safe = _support(r, shape)

    w = _reduced_exponent(r, v_safe, valid)
    # 1 - t^{-1/ξ} = 1 - exp(-w) = -expm1(-w); → 1 - exp(-x/σ) in the limit.
    cdf_inside = -jnp.expm1(-w)
    # Beyond the support, ξ > 0 tails map out-of-support CDF to 0 (lower
    # bound); ξ < 0 tails map it to 1 (finite upper bound, including the
    # endpoint x = -σ/ξ where 1 + ξx/σ = 0 exactly).
    boundary = jnp.where(shape < 0, 1.0, 0.0)
    cdf = jnp.where(valid, cdf_inside, boundary)
    cdf = jnp.where(posinf, 1.0, cdf)  # F(+inf) = 1
    # Clamp the universal lower bound (x < 0, incl. -inf, is out of support).
    cdf = jnp.where(below, 0.0, cdf)
    return jnp.where(jnp.isnan(x), jnp.nan, cdf)  # NaN input propagates


def gpd_survival(
    x: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Survival function :math:`S(x) = (1 + \xi x/\sigma)^{-1/\xi}` of the GPD.

    Computed as ``exp(-w)`` directly (rather than ``1 - cdf``) so the deep tail
    stays accurate. Equals ``exp(-x/σ)`` in the exponential limit.
    """
    shape = jnp.asarray(shape)
    x = jnp.asarray(x)
    below = x < 0.0
    posinf = jnp.isposinf(x)
    x_calc = jnp.where(below | posinf, 0.0, x)
    r = _standardize_exceedance(x_calc, scale)
    valid, v_safe = _support(r, shape)

    w = _reduced_exponent(r, v_safe, valid)
    s_inside = jnp.exp(-w)
    # Above the finite Weibull-type upper bound (ξ < 0) S = 0; ξ > 0 has no
    # upper bound so `valid` never fails there for x ≥ 0.
    s = jnp.where(valid, s_inside, 0.0)
    s = jnp.where(posinf, 0.0, s)  # S(+inf) = 0
    # Below x = 0 (incl. -inf) the exceedance is certain, S = 1.
    s = jnp.where(below, 1.0, s)
    return jnp.where(jnp.isnan(x), jnp.nan, s)  # NaN input propagates


def gpd_log_survival(
    x: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Log survival :math:`\log S(x) = -\tfrac{1}{\xi}\log(1 + \xi x/\sigma)`.

    Equals ``-w`` inside the support, so the deep tail is exact instead of
    passing through ``log(1 - cdf)``.
    """
    shape = jnp.asarray(shape)
    x = jnp.asarray(x)
    below = x < 0.0
    posinf = jnp.isposinf(x)
    x_calc = jnp.where(below | posinf, 0.0, x)
    r = _standardize_exceedance(x_calc, scale)
    valid, v_safe = _support(r, shape)

    w = _reduced_exponent(r, v_safe, valid)
    ls = jnp.where(valid, -w, -jnp.inf)
    ls = jnp.where(posinf, -jnp.inf, ls)  # log S(+inf) = -inf
    ls = jnp.where(below, 0.0, ls)
    return jnp.where(jnp.isnan(x), jnp.nan, ls)  # NaN input propagates


def _gpd_icdf_from_log_exceedance(
    log_p: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    r"""GPD quantile from :math:`\log p`, the log exceedance probability ``1 - q``.

    .. math:: Q = \frac{\sigma}{\xi}\big(p^{-\xi} - 1\big)
        = -\sigma \log p \cdot \frac{\exp(-\xi \log p) - 1}{-\xi \log p}

    which tends to :math:`-\sigma \log p` (the exponential quantile) as
    :math:`\xi \to 0`. Taking ``log p`` directly avoids the ``1 - q``
    cancellation that wrecks large return periods.
    """
    shape = jnp.asarray(shape)
    log_p = jnp.asarray(log_p)
    at_upper = jnp.isneginf(log_p)  # q = 1

    # Sanitize the endpoint input so the interior product is finite there — both
    # its value and the reverse-mode gradient of the discarded branch — before
    # the analytic upper endpoint is selected below.
    safe = jnp.where(at_upper, -1.0, log_p)
    a = -shape * safe
    q_val = -scale * safe * expm1_over_x(a)

    # Upper endpoint: bounded at -σ/ξ for ξ < 0, otherwise +inf. shape_safe
    # avoids a 0-division in the branch the outer where discards.
    shape_safe = jnp.where(shape < 0.0, shape, -1.0)
    upper = jnp.where(shape < 0.0, -scale / shape_safe, jnp.inf)
    return jnp.where(at_upper, upper, q_val)


def gpd_icdf(
    q: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Quantile function (inverse CDF) of the GPD."""
    # log(1 - q) via log1p keeps the upper tail (q → 1) accurate.
    return _gpd_icdf_from_log_exceedance(jnp.log1p(-q), scale, shape)


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
    r"""T-period return level: :math:`Q(1 - 1/T)`.

    Parameterized by the exceedance probability ``1/T`` directly (``log p =
    -log T``) so it stays accurate for large ``T``.
    """
    return _gpd_icdf_from_log_exceedance(-jnp.log(period), scale, shape)
