"""Pure-JAX primitives for the Fréchet (Type II) extreme value distribution.

Fréchet is the GEV family restricted to :math:`\\xi > 0`: heavy polynomial
tails, lower-bounded support :math:`x \\ge \\mu - \\sigma/\\xi`. These are
thin wrappers over :mod:`xtremax.primitives.gev`; the shape constraint is
a caller obligation (the Distribution class enforces it at construction).
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


def frechet_log_prob(
    x: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Fréchet log PDF (assumes ``shape > 0``)."""
    return gev_log_prob(x, loc, scale, shape)


def frechet_cdf(
    x: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Fréchet CDF."""
    return gev_cdf(x, loc, scale, shape)


def frechet_icdf(
    q: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Fréchet quantile function."""
    return gev_icdf(q, loc, scale, shape)


def frechet_mean(
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Fréchet mean (``+inf`` when ``shape >= 1``)."""
    return gev_mean(loc, scale, shape)


def frechet_return_level(
    period: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """T-period return level."""
    return gev_return_level(period, loc, scale, shape)
