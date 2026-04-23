"""Pure-JAX primitives for the reverse Weibull (Type III) extreme value distribution.

Weibull Type III is the GEV family restricted to :math:`\\xi < 0`: bounded
upper tail :math:`x \\le \\mu - \\sigma/\\xi`. These are thin wrappers over
:mod:`xtremax.primitives.gev`; the shape constraint is a caller obligation
(the Distribution class enforces it at construction).
"""

from __future__ import annotations

from jaxtyping import Array, Float

from xtremax.primitives.gev import (
    gev_cdf,
    gev_icdf,
    gev_log_prob,
    gev_mean,
    gev_return_level,
)


def weibull_log_prob(
    x: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Reverse-Weibull log PDF (assumes ``shape < 0``)."""
    return gev_log_prob(x, loc, scale, shape)


def weibull_cdf(
    x: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Reverse-Weibull CDF."""
    return gev_cdf(x, loc, scale, shape)


def weibull_icdf(
    q: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Reverse-Weibull quantile function."""
    return gev_icdf(q, loc, scale, shape)


def weibull_mean(
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Reverse-Weibull mean (always finite for ``shape < 0``)."""
    return gev_mean(loc, scale, shape)


def weibull_return_level(
    period: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """T-period return level."""
    return gev_return_level(period, loc, scale, shape)
