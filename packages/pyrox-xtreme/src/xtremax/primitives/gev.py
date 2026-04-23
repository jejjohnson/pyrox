"""Pure-JAX primitives for the Generalized Extreme Value distribution.

All functions are stateless, ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` safe,
and branch on the Gumbel limit (:math:`\\xi \\approx 0`) via ``jnp.where``
so both the general and limiting formulas are always traceable.

The parameterization follows Coles (2001):

.. math:: F(x; \\mu, \\sigma, \\xi) = \\exp\\!\\left\\{-\\left[1 + \\xi
    \\frac{x - \\mu}{\\sigma}\\right]^{-1/\\xi}\\right\\}

with support :math:`\\{x : 1 + \\xi(x - \\mu)/\\sigma > 0\\}` and the
convention :math:`\\xi = 0 \\Rightarrow F(x) = \\exp(-\\exp(-z))` where
:math:`z = (x - \\mu)/\\sigma`.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import gammaln
from jaxtyping import Array, Float


_EULER_GAMMA = 0.5772156649015329
_GUMBEL_THRESHOLD = 1e-7


def _safe_shape(shape: Float[Array, ...]) -> Float[Array, ...]:
    """Replace a near-zero shape with 1.0 so divisions trace cleanly.

    The caller is expected to mask the result with ``is_gumbel`` via
    ``jnp.where`` so the substituted value never leaks into the output.
    """
    shape = jnp.asarray(shape)
    is_gumbel = jnp.abs(shape) < _GUMBEL_THRESHOLD
    return jnp.where(is_gumbel, 1.0, shape)


def gev_log_prob(
    x: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Log PDF of the Generalized Extreme Value distribution."""
    shape = jnp.asarray(shape)
    is_gumbel = jnp.abs(shape) < _GUMBEL_THRESHOLD
    xi = _safe_shape(shape)

    z = (x - loc) / scale
    t = 1.0 + xi * z
    valid = t > 0.0
    t_safe = jnp.where(valid, t, 1.0)
    log_t = jnp.log(t_safe)

    gumbel = -jnp.log(scale) - z - jnp.exp(-z)
    gevd = -jnp.log(scale) - (1.0 / xi + 1.0) * log_t - jnp.power(t_safe, -1.0 / xi)
    gevd = jnp.where(valid, gevd, -jnp.inf)

    return jnp.where(is_gumbel, gumbel, gevd)


def gev_cdf(
    x: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """CDF of the Generalized Extreme Value distribution."""
    shape = jnp.asarray(shape)
    is_gumbel = jnp.abs(shape) < _GUMBEL_THRESHOLD
    xi = _safe_shape(shape)

    z = (x - loc) / scale
    t = 1.0 + xi * z
    valid = t > 0.0
    t_safe = jnp.where(valid, t, 1.0)

    gumbel = jnp.exp(-jnp.exp(-z))
    gevd_inside = jnp.exp(-jnp.power(t_safe, -1.0 / xi))
    # Fréchet tails (ξ > 0) map out-of-support to 0; Weibull tails to 1.
    boundary = jnp.where(xi > 0, 0.0, 1.0)
    gevd = jnp.where(valid, gevd_inside, boundary)

    return jnp.where(is_gumbel, gumbel, gevd)


def gev_icdf(
    q: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Quantile function (inverse CDF) of the GEV distribution."""
    shape = jnp.asarray(shape)
    is_gumbel = jnp.abs(shape) < _GUMBEL_THRESHOLD
    xi = _safe_shape(shape)

    log_q = jnp.log(q)
    gumbel = loc - scale * jnp.log(-log_q)
    gevd = loc + (scale / xi) * (jnp.power(-log_q, -xi) - 1.0)

    return jnp.where(is_gumbel, gumbel, gevd)


def gev_mean(
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Mean of the GEV distribution (``+inf`` when :math:`\xi \ge 1`)."""
    shape = jnp.asarray(shape)
    is_gumbel = jnp.abs(shape) < _GUMBEL_THRESHOLD
    xi = _safe_shape(shape)

    gumbel = loc + scale * _EULER_GAMMA
    gamma_term = jnp.exp(gammaln(1.0 - xi))
    gevd = loc + (scale / xi) * (gamma_term - 1.0)

    mean = jnp.where(is_gumbel, gumbel, gevd)
    return jnp.where(shape < 1.0, mean, jnp.inf)


def gev_return_level(
    period: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    r"""T-period return level: :math:`z_T = F^{-1}(1 - 1/T)`."""
    return gev_icdf(1.0 - 1.0 / period, loc, scale, shape)
