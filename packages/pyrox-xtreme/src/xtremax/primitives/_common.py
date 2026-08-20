"""Shared numerical constants and stable series helpers for EVT primitives.

The GEV / GPD kernels all reduce to two removable-singularity quantities as
the shape parameter approaches its Gumbel / exponential limit:

- ``log1p(u) / u`` — appears as :math:`\\log(1 + \\xi z)/\\xi` (via ``u = ξz``),
- ``expm1(a) / a`` — appears in the quantile and mean expansions.

Both tend to ``1`` as their argument tends to ``0``, but the naive quotient is
``0 / 0`` there — numerically ``nan`` in float32 and, worse, ``nan`` in the
reverse-mode gradient. Evaluating them through these helpers keeps the value
accurate and the gradient finite across the whole shape range, which lets the
kernels drop their hard ``jnp.where`` branch on the Gumbel limit entirely: the
formulas are smooth through :math:`\\xi = 0`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


EULER_GAMMA = 0.5772156649015329

# Single repo-wide threshold below which the remaining class-layer branches
# on the Gumbel limit switch over (``mode``, and the quantile grid inside
# ``conditional_excess_mean``). The kernels themselves are smooth and
# threshold-free. Variance / skew / kurtosis used to sit behind this
# threshold too and were garbage between it and ξ ≈ 0.1 (#88); they now go
# through the cancellation-free reduced forms in
# :mod:`xtremax.primitives._moments`, which need no Gumbel branch at all.
# Every class must import this rather than caching its own copy: a
# class-local value (the old GPD 1e-8 vs GEV 1e-7 split) let sibling methods
# of the same object disagree by >100% inside the gap.
GUMBEL_THRESHOLD = 1e-7

# Below this magnitude the exact quotient loses precision (and its gradient is
# 0/0), so we switch to a Taylor series. 1e-3 keeps the 4th-order series within
# float64 eps at the crossover while staying comfortably accurate in float32.
_SERIES_THRESHOLD = 1e-3


def clamped_standardize(
    x: Float[Array, ...], loc: Float[Array, ...], scale: Float[Array, ...]
) -> Array:
    r"""Return ``(x - loc) / scale`` clamped only against literal overflow.

    Two guards keep ``z`` finite without ever distorting a physically
    meaningful standardized value:

    - A non-finite ``x`` is replaced by ``loc`` (so ``z → 0``) *before* the
      division. Callers override the value for ``±inf`` via their own
      extended-real limit; sanitizing the input here means a ``±inf`` never
      forms an ``inf / scale²`` term in the reverse-mode gradient.
    - The forward value is clipped only against genuine overflow (``±inf`` from
      a tiny / subnormal ``scale`` becomes ``±finfo.max``); every representable
      ratio is returned exactly, so a large finite standardized value is not
      distorted. The reverse-mode gradient is additionally clipped to
      ``±sqrt(max)`` so ``z²`` (which appears in the shape gradient) stays
      representable; ``stop_gradient`` stitches the exact value onto the
      gradient-safe branch. The deep-tail ``exp(-w)`` overflow is handled
      separately by :func:`safe_exp_neg`.
    """
    x = jnp.asarray(x)
    x_safe = jnp.where(jnp.isfinite(x), x, loc)
    z = (x_safe - loc) / scale
    max_val = jnp.finfo(z.dtype).max
    value = jnp.clip(z, -max_val, max_val)  # only ±inf is squeezed, to ±max
    z_grad = jnp.clip(z, -jnp.sqrt(max_val), jnp.sqrt(max_val))  # z² stays finite
    return z_grad + jax.lax.stop_gradient(value - z_grad)


def safe_exp_neg(w: Float[Array, ...]) -> Array:
    r"""Return ``exp(-w)`` with the argument floored so it can never overflow.

    In the Gumbel deep lower tail ``w → -∞`` and the raw ``exp(-w)`` overflows to
    ``inf``; worse, the reverse-mode shape gradient of the log density scales as
    ``exp(-w) · z²``, so an unbounded ``exp(-w)`` yields ``0 · inf = nan`` in the
    discarded ``jnp.where`` branch. Flooring ``w`` at ``-(log(max) - 2·log log
    max)`` caps ``exp(-w)`` a couple of e-folds below the dtype ceiling — chosen
    so ``exp(-w) · z²`` also stays finite (``z ≈ -w`` there) — and zeroes the
    gradient of the saturated branch. The value cap is immaterial: the density is
    already numerically zero that far into the tail.
    """
    w = jnp.asarray(w)
    log_max = jnp.log(jnp.finfo(w.dtype).max)
    exp_arg_max = log_max - 2.0 * jnp.log(log_max)
    return jnp.exp(-jnp.maximum(w, -exp_arg_max))


def safe_exp_neg_density(w: Float[Array, ...]) -> Array:
    r"""``exp(-w)`` for a log *density* term: exact value, overflow-free gradient.

    :func:`safe_exp_neg` caps the *value* a couple of e-folds below the dtype
    ceiling so that ``exp(-w)·z²`` stays finite — the right trade for an
    already-saturated CDF / survival output, but wrong for a log density, whose
    ``-exp(-w)`` term *is* the likelihood: capping it early would understate the
    penalty (and drop the dominant gradient) for every observation in the deep
    tail. Here the value is preserved right up to the representable limit and
    saturates at ``finfo.max`` (not ``inf``) beyond it, while the reverse-mode
    derivative is separately floored via :func:`safe_exp_neg` so the
    ``exp(-w)·z²`` shape gradient can never overflow to ``0·inf = nan``.
    ``stop_gradient`` stitches the exact forward value onto the safe gradient.
    """
    w = jnp.asarray(w)
    max_val = jnp.finfo(w.dtype).max
    # ``exp(-w)`` overflows to ``inf`` for ``w < -log(max)`` — and even at
    # exactly ``-log(max)`` the rounded exponent already overflows — so cap the
    # value at ``finfo.max`` directly. Only the forward value matters here (it
    # feeds ``stop_gradient``), so the min's kink carries no gradient.
    value = jnp.minimum(jnp.exp(-w), max_val)
    grad_safe = safe_exp_neg(w)
    return grad_safe + jax.lax.stop_gradient(value - grad_safe)


def log1p_over_x(u: Float[Array, ...]) -> Float[Array, ...]:
    r"""Stable :math:`\log(1 + u) / u`, with the removable singularity at 0.

    Value and reverse-mode gradient are both finite for every ``u > -1``,
    including ``u = +inf`` (the limit is 0). A ``+inf`` argument arises when a
    large finite shape overflows the ``ξz`` product; the caller must still
    sanitize out-of-support values (``u <= -1``) before calling — ``log1p`` is
    undefined there.
    """
    u = jnp.asarray(u)
    small = jnp.abs(u) < _SERIES_THRESHOLD
    posinf = jnp.isposinf(u)
    # Double-``where`` so no branch feeds a singular value (0 or +inf) to
    # autodiff: the small case takes the series, the +inf case takes the limit.
    u_exact = jnp.where(small | posinf, 1.0, u)
    exact = jnp.log1p(u_exact) / u_exact
    u_series = jnp.where(small, u, 0.0)
    # log1p(u)/u = 1 - u/2 + u²/3 - u³/4 + u⁴/5 - …
    series = 1.0 + u_series * (
        -0.5 + u_series * (1.0 / 3.0 + u_series * (-0.25 + u_series * 0.2))
    )
    result = jnp.where(small, series, exact)
    return jnp.where(posinf, 0.0, result)  # log1p(u)/u → 0 as u → +inf


def expm1_over_x(a: Float[Array, ...]) -> Float[Array, ...]:
    r"""Stable :math:`(\exp(a) - 1) / a`, with the removable singularity at 0.

    Value and reverse-mode gradient are both finite everywhere except at
    ``a = +inf``, whose extended-real limit is ``+inf`` (returned explicitly).
    A ``+inf`` argument arises when a large finite shape overflows the exponent
    of a quantile / return-level expansion; ``a = -inf`` already yields the
    correct limit 0 without special handling.
    """
    a = jnp.asarray(a)
    small = jnp.abs(a) < _SERIES_THRESHOLD
    posinf = jnp.isposinf(a)
    # Double-``where`` so no branch feeds a singular value (0 or +inf) to
    # autodiff before the +inf limit is selected below.
    a_exact = jnp.where(small | posinf, 1.0, a)
    exact = jnp.expm1(a_exact) / a_exact
    a_series = jnp.where(small, a, 0.0)
    # expm1(a)/a = 1 + a/2 + a²/6 + a³/24 + a⁴/120 + …
    series = 1.0 + a_series * (
        0.5
        + a_series * (1.0 / 6.0 + a_series * (1.0 / 24.0 + a_series * (1.0 / 120.0)))
    )
    result = jnp.where(small, series, exact)
    return jnp.where(posinf, jnp.inf, result)  # expm1(a)/a → +inf as a → +inf
