"""Analytical second-order summaries for spatial point processes.

These are the closed-form K, L, and pair-correlation functions for a
homogeneous Poisson process on :math:`\\mathbb{R}^d`. Empirical
estimators (edge-corrected K̂, residual diagnostics) are deferred to a
future PR — they need O(n²) loops that don't compose well with the
``jit`` / ``vmap`` style of the rest of the package.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def unit_ball_volume(n_dims: int) -> float:
    """Volume of the unit ``ℝᵈ`` ball: :math:`V_d = \\pi^{d/2}/\\Gamma(d/2+1)`."""
    d = float(n_dims)
    return float((jnp.pi ** (d / 2.0)) / jax.scipy.special.gamma(d / 2.0 + 1.0))


def csr_ripleys_k(
    r: Float[Array, ...],
    n_dims: int,
) -> Float[Array, ...]:
    """Theoretical Ripley's K-function under CSR, :math:`K(r) = V_d r^d`.

    Special-cased for the common dimensions to keep the formulas
    legible: :math:`2r` (1D), :math:`\\pi r^2` (2D),
    :math:`\\frac{4}{3}\\pi r^3` (3D). Higher dimensions fall through
    to the generic ``V_d r^d`` form.
    """
    r = jnp.asarray(r)
    d = n_dims
    if d == 1:
        return 2.0 * r
    if d == 2:
        return jnp.pi * r**2
    if d == 3:
        return (4.0 / 3.0) * jnp.pi * r**3
    return unit_ball_volume(d) * r**d


def csr_l_function(
    r: Float[Array, ...],
    n_dims: int,
) -> Float[Array, ...]:
    """Besag's L-function, the variance-stabilised K.

    :math:`L(r) = (K(r) / V_d)^{1/d}`. Under CSR, ``L(r) = r``
    identically, so any departure of the empirical estimator from the
    diagonal is a sign of clustering or regularity.
    """
    K = csr_ripleys_k(r, n_dims)
    v_d = unit_ball_volume(n_dims)
    return (K / v_d) ** (1.0 / float(n_dims))


def csr_pair_correlation(
    r: Float[Array, ...],
    n_dims: int,
) -> Float[Array, ...]:
    """Pair correlation function under CSR: ``g(r) ≡ 1``."""
    del n_dims
    r = jnp.asarray(r)
    return jnp.ones_like(r)
