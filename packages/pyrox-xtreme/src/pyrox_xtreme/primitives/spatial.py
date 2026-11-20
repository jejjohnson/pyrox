"""Pure-JAX spatial building blocks for hierarchical extreme value models.

These are the framework-agnostic computational kernels underneath
spatio-temporal GEV models: pairwise distances between sites, the
regression design matrix, and the two-range exponential correlation used
as a Gaussian-copula dependence structure across sites.

They carry no NumPyro / Equinox / GP-library state — every function takes
arrays and returns arrays, so they remain ``jax.jit`` / ``jax.grad`` /
``jax.vmap`` safe. Full latent-GP priors and conditioning are deliberately
left to a dedicated GP library rather than reimplemented here.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def pairwise_distances(coords: Float[Array, ...]) -> Float[Array, ...]:
    r"""Euclidean distance matrix between sites.

    .. math:: D_{ij} = \lVert s_i - s_j \rVert_2

    Args:
        coords: Site coordinates of shape ``(n_sites, n_dims)``. For spatial
            models these are typically normalized longitude / latitude, so
            the resulting distances are dimensionless.

    Returns:
        Dense distance matrix of shape ``(n_sites, n_sites)`` with a zero
        diagonal.
    """
    coords = jnp.asarray(coords)
    deltas = coords[:, None, :] - coords[None, :, :]
    # ‖·‖ has derivative Δ/‖Δ‖, which is 0/0 on the zero diagonal and yields NaN
    # gradients. Take sqrt of a masked-positive squared distance, then restore
    # the exact zero diagonal — value and gradient are both finite.
    sq = jnp.sum(deltas**2, axis=-1)
    positive = sq > 0.0
    dist = jnp.where(positive, jnp.sqrt(jnp.where(positive, sq, 1.0)), 0.0)
    # A NaN coordinate makes its squared distances NaN, which ``sq > 0`` reads as
    # False — masking it to a zero (coincident) distance and hiding the bad
    # input. Propagate the NaN instead of collapsing it onto the diagonal.
    return jnp.where(jnp.isnan(sq), jnp.nan, dist)


def design_matrix(covariates: Float[Array, ...]) -> Float[Array, ...]:
    r"""Prepend an intercept column to a static-covariate matrix.

    .. math:: X(s) = [1, z_1(s), \ldots, z_D(s)]

    Args:
        covariates: Static site covariates of shape ``(n_sites, n_covariates)``.
            A 1-D array of shape ``(n_sites,)`` is treated as a single column.

    Returns:
        Design matrix of shape ``(n_sites, 1 + n_covariates)`` whose first
        column is all ones.
    """
    covariates = jnp.asarray(covariates)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    n_sites = covariates.shape[0]
    intercept = jnp.ones((n_sites, 1), dtype=covariates.dtype)
    return jnp.concatenate([intercept, covariates], axis=1)


def two_range_correlation(
    dist_matrix: Float[Array, ...],
    weight: Float[Array, ...],
    range_short: Float[Array, ...],
    range_long: Float[Array, ...],
    jitter: float = 1e-6,
) -> Float[Array, ...]:
    r"""Two-range exponential correlation matrix with unit diagonal.

    Convex mixture of two exponential dependence ranges, used as the
    correlation structure of a Gaussian copula over sites:

    .. math:: \rho(h) = c_0 \exp(-h / c_1) + (1 - c_0) \exp(-h / c_2)

    where ``h`` is the inter-site distance, ``c_0 = weight`` mixes the
    short and long ranges, and ``c_1 = range_short`` / ``c_2 = range_long``
    are the two correlation length scales. The diagonal is renormalized to
    exactly one (after adding ``jitter`` for positive-definiteness) so the
    result is a valid correlation matrix.

    Args:
        dist_matrix: Pairwise distances of shape ``(n_sites, n_sites)``,
            e.g. from :func:`pairwise_distances`.
        weight: Mixture weight :math:`c_0 \in (0, 1)` on the short range.
        range_short: Short correlation length scale :math:`c_1 > 0`.
        range_long: Long correlation length scale :math:`c_2 > 0`.
        jitter: Small positive diagonal inflation added before
            renormalization for numerical stability.

    Returns:
        Correlation matrix of shape ``(n_sites, n_sites)`` with unit diagonal.
    """
    dist_matrix = jnp.asarray(dist_matrix)
    n_sites = dist_matrix.shape[0]

    base = weight * jnp.exp(-dist_matrix / range_short) + (1.0 - weight) * jnp.exp(
        -dist_matrix / range_long
    )
    jittered = base + jitter * jnp.eye(n_sites)

    # Renormalize to unit diagonal so the matrix is a valid correlation matrix.
    diag = jnp.sqrt(jnp.clip(jnp.diag(jittered), min=1e-12))
    return jittered / (diag[:, None] * diag[None, :])
