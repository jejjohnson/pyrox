"""Pure-JAX primitives for the Gumbel Type I distribution.

The Gumbel is the :math:`\\xi = 0` limit of the GEV. We keep a separate
specialization because the closed forms avoid the ``jnp.where``-guarded
division-by-zero dance that ``gev_*`` needs for the Gumbel branch.

All functions are stateless, ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` safe.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


_EULER_GAMMA = 0.5772156649015329


def gumbel_log_prob(
    x: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Gumbel log PDF: :math:`-\log\sigma - z - e^{-z}`, :math:`z=(x-\mu)/\sigma`."""
    z = (x - loc) / scale
    return -jnp.log(scale) - z - jnp.exp(-z)


def gumbel_cdf(
    x: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Gumbel CDF: :math:`F(x) = \exp(-\exp(-z))`."""
    z = (x - loc) / scale
    return jnp.exp(-jnp.exp(-z))


def gumbel_icdf(
    q: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Gumbel quantile: :math:`Q(p) = \mu - \sigma\log(-\log p)`."""
    return loc - scale * jnp.log(-jnp.log(q))


def gumbel_mean(
    loc: Float[Array, ...],
    scale: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Gumbel mean: :math:`\mu + \sigma\gamma_E`."""
    return loc + scale * _EULER_GAMMA


def gumbel_return_level(
    period: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
) -> Float[Array, ...]:
    r"""T-period return level: :math:`Q(1 - 1/T)`."""
    return gumbel_icdf(1.0 - 1.0 / period, loc, scale)
