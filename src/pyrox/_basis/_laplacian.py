"""Graph Laplacian eigenpairs for graph/manifold inducing features.

Eigendecomposition is a one-shot setup-time op (uses :mod:`numpy` and
returns ``jax.numpy`` arrays). For sparse inputs at scale, swap in
:func:`scipy.sparse.linalg.eigsh` — but stick with dense ``numpy`` here
to keep the dependency surface small.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def graph_laplacian_eigpairs(
    adjacency: Float[Array, "V V"],
    num_basis: int,
    *,
    normalized: bool = True,
) -> tuple[Float[Array, " M"], Float[Array, "V M"]]:
    r"""Smallest ``num_basis`` Laplacian eigenpairs of an undirected graph.

    Args:
        adjacency: Symmetric, non-negative adjacency matrix of shape ``(V, V)``.
            Diagonal is ignored (no self-loops folded into ``D``).
        num_basis: Number of low-frequency eigenpairs to return ``M``.
        normalized: If ``True``, use the symmetric normalized Laplacian
            :math:`I - D^{-1/2} A D^{-1/2}`; otherwise use the combinatorial
            Laplacian :math:`D - A`.

    Returns:
        ``(eigvals, eigvecs)`` with eigvals sorted ascending, ``eigvals``
        of shape ``(M,)`` and ``eigvecs`` of shape ``(V, M)``. Eigenvectors
        are :math:`\ell^2`-orthonormal.

    Raises:
        ValueError: If ``adjacency`` is not square, has negative entries,
            or ``num_basis`` exceeds ``V``.
    """
    A = np.asarray(adjacency)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"adjacency must be square 2D; got shape {A.shape}.")
    V = A.shape[0]
    if num_basis < 1 or num_basis > V:
        raise ValueError(f"num_basis must be in [1, {V}]; got {num_basis}.")
    if np.any(A < 0):
        raise ValueError("adjacency must have non-negative entries.")
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)

    deg = A.sum(axis=1)
    if normalized:
        d_inv_sqrt = np.where(deg > 0, deg ** (-0.5), 0.0)
        L = np.eye(V) - (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]
    else:
        L = np.diag(deg) - A
    L = 0.5 * (L + L.T)

    eigvals, eigvecs = np.linalg.eigh(L)
    eigvals = eigvals[:num_basis]
    eigvecs = eigvecs[:, :num_basis]
    # Clip tiny negatives that can appear from roundoff.
    eigvals = np.maximum(eigvals, 0.0)
    return jnp.asarray(eigvals), jnp.asarray(eigvecs)
