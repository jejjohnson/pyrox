r"""Laplacian eigenfunctions on the bounded box :math:`[-L, L]^D`.

Dirichlet eigenpairs of :math:`-d^2/dx^2` on :math:`[-L, L]`:

.. math::

    \phi_j(x) = \frac{1}{\sqrt{L}} \sin\!\left(\frac{j\pi(x + L)}{2L}\right),
    \qquad
    \lambda_j = \left(\frac{j\pi}{2L}\right)^2,
    \qquad j = 1, 2, \ldots

The 1D basis is :math:`L^2([-L, L])`-orthonormal. On a :math:`D`-dimensional
box the basis is the tensor product, indexed by a multi-index
:math:`(j_1, \ldots, j_D)`; we flatten in row-major order. These eigenfunctions
are the engine of both VFF (#49, GP-side) and HSGP (#41, NN-side).
"""

from __future__ import annotations

import einops
import jax.numpy as jnp
from jaxtyping import Array, Float


def fourier_basis_1d(
    x: Float[Array, " N"],
    num_basis: int,
    L: float,
) -> Float[Array, "N M"]:
    r"""Evaluate the first ``num_basis`` 1D Dirichlet eigenfunctions on ``[-L, L]``.

    .. math::

        \phi_j(x) = \frac{1}{\sqrt{L}} \sin\!\left(\frac{j\pi(x + L)}{2L}\right)

    Args:
        x: Inputs in ``[-L, L]``. Values outside the interval are
            extrapolated silently — callers wrap their own domain checks.
        num_basis: Number of basis functions ``M``.
        L: Half-width of the bounded domain (must be positive).

    Returns:
        ``(N, M)`` array whose ``(n, j-1)`` entry is :math:`\phi_j(x_n)`.

    Raises:
        ValueError: If ``num_basis < 1`` or ``L <= 0``.
    """
    if num_basis < 1:
        raise ValueError(f"num_basis must be >= 1, got {num_basis}.")
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L}.")
    j = jnp.arange(1, num_basis + 1, dtype=x.dtype)
    arg = einops.einsum(x + L, j, "n, m -> n m") * (jnp.pi / (2.0 * L))
    return jnp.sin(arg) / jnp.sqrt(L)


def fourier_eigenvalues_1d(
    num_basis: int,
    L: float,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> Float[Array, " M"]:
    r"""Return :math:`\lambda_j = (j\pi / (2L))^2` for ``j = 1, ..., num_basis``."""
    if num_basis < 1:
        raise ValueError(f"num_basis must be >= 1, got {num_basis}.")
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L}.")
    j = jnp.arange(1, num_basis + 1, dtype=dtype)
    return (j * jnp.pi / (2.0 * L)) ** 2


def _to_tuple(value: int | float | tuple, D: int, name: str) -> tuple:
    """Broadcast a scalar to a length-``D`` tuple, or validate an existing tuple."""
    if isinstance(value, tuple | list):
        out = tuple(value)
        if len(out) != D:
            raise ValueError(
                f"{name} must have length {D} (one per input dim); got {len(out)}."
            )
        return out
    return (value,) * D


def fourier_basis(
    x: Float[Array, "N D"],
    num_basis_per_dim: int | tuple[int, ...],
    L: float | tuple[float, ...],
) -> tuple[Float[Array, "N M"], Float[Array, " M"]]:
    r"""Tensor-product Dirichlet eigenpairs on :math:`[-L, L]^D`.

    The :math:`D`-dimensional eigenfunctions are products of 1D basis
    functions, with eigenvalues that *sum* across axes:

    .. math::

        \Phi_{(j_1, \ldots, j_D)}(x) = \prod_{d=1}^D \phi_{j_d}(x_d),
        \qquad
        \lambda_{(j_1, \ldots, j_D)} = \sum_{d=1}^D \lambda_{j_d}.

    The flattened index is row-major over the multi-index, i.e. the last
    dimension varies fastest. Total feature count is
    ``M = prod(num_basis_per_dim)``.

    Args:
        x: Inputs of shape ``(N, D)`` in :math:`[-L, L]^D`.
        num_basis_per_dim: Per-axis number of 1D basis functions; an
            ``int`` is broadcast to all ``D`` axes.
        L: Per-axis half-width; an ``int``/``float`` is broadcast to all axes.

    Returns:
        ``(Phi, lam)`` with ``Phi`` of shape ``(N, M)`` and ``lam`` of
        shape ``(M,)``.

    Raises:
        ValueError: If ``num_basis_per_dim`` or ``L`` has wrong length.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (N, D); got shape {x.shape}.")
    D = x.shape[-1]
    M_per = _to_tuple(num_basis_per_dim, D, "num_basis_per_dim")
    L_per = _to_tuple(L, D, "L")

    # Per-axis bases and eigenvalues.
    bases = [
        fourier_basis_1d(x[:, d], M_per[d], float(L_per[d])) for d in range(D)
    ]  # each (N, M_d)
    lams = [
        fourier_eigenvalues_1d(M_per[d], float(L_per[d]), dtype=x.dtype)
        for d in range(D)
    ]  # each (M_d,)

    # Tensor-product the bases (row-major flatten of the multi-index).
    Phi = bases[0]
    for b in bases[1:]:
        Phi = einops.einsum(Phi, b, "n a, n b -> n a b")
        Phi = einops.rearrange(Phi, "n a b -> n (a b)")

    # Sum-of-eigenvalues across axes via pure JAX broadcasting — row-major
    # flatten matches the basis layout above.
    lam = _tensor_product_sum(lams, dtype=x.dtype)
    return Phi, lam


def _tensor_product_sum(
    lams: list[Float[Array, " M_d"]],
    *,
    dtype: jnp.dtype,
) -> Float[Array, " M"]:
    """Row-major flatten of the sum-of-eigenvalues tensor over D per-axis vectors.

    ``lams[d]`` has shape ``(M_d,)``; the result has shape ``(prod_d M_d,)``
    with entry ``(j_0, ..., j_{D-1})`` equal to ``sum_d lams[d][j_d]``. Uses
    pure JAX broadcasting — no Python-side enumeration of multi-indices.
    """
    if not lams:
        return jnp.zeros(0, dtype=dtype)
    D = len(lams)
    grid = jnp.zeros(tuple(lam.shape[0] for lam in lams), dtype=dtype)
    for d, lam_d in enumerate(lams):
        # Reshape lam_d to broadcast along axis d of the D-dim grid.
        shape = [1] * D
        shape[d] = lam_d.shape[0]
        grid = grid + jnp.reshape(lam_d, shape)
    return jnp.reshape(grid, (-1,))


def fourier_eigenvalues(
    num_basis_per_dim: int | tuple[int, ...],
    L: float | tuple[float, ...],
    D: int,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> Float[Array, " M"]:
    """Return the flattened sum-of-squares eigenvalues for a ``D``-dimensional box.

    Useful when only the eigenvalues are needed (e.g. building ``K_uu`` for
    inducing features without evaluating ``k_ux``).
    """
    M_per = _to_tuple(num_basis_per_dim, D, "num_basis_per_dim")
    L_per = _to_tuple(L, D, "L")
    lams = [
        fourier_eigenvalues_1d(M_per[d], float(L_per[d]), dtype=dtype) for d in range(D)
    ]
    return _tensor_product_sum(lams, dtype=dtype)
