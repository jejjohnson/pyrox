r"""Data-driven initialisation of GPLVM-style latent variables (P1).

Probabilistic PCA, $y = Wx + \varepsilon$ with $x \sim \mathcal N(0, I_Q)$,
has its maximum-likelihood $W$ in the span of the top-$Q$ principal
components (Tipping & Bishop, 1999); dually, a GPLVM with the linear kernel
$K = XX^\top + \sigma^2 I$ has its maximum-likelihood latents there
(Lawrence, 2005). PCA is therefore the natural start for a nonlinear GPLVM,
and kernel PCA or Laplacian eigenmaps start it on a curled data manifold.
"""

from __future__ import annotations

from typing import Literal

import einx
import jax.numpy as jnp
import kernellib as kl
from jaxtyping import Array, Float


def latent_init(
    Y: Float[Array, "N D"],
    n_latent: int,
    *,
    method: Literal["pca", "kernel_pca", "laplacian_eigenmaps"] = "pca",
    kernel: kl.AbstractKernel | None = None,
    n_neighbors: int = 10,
    standardize: bool = True,
) -> Float[Array, "N Q"]:
    """Initial latents ``(N, n_latent)`` for a GPLVM or latent-factor model.

    Args:
        Y: Observations, one row per point.
        n_latent: Latent dimension $Q$.
        method: ``"pca"`` (scores of the centred ``Y``, no kernellib call),
            ``"kernel_pca"`` (`kernellib.KernelPCA` with ``kernel``) or
            ``"laplacian_eigenmaps"`` (`kernellib.LaplacianEigenmaps` on a
            ``n_neighbors``-nearest-neighbour graph).
        kernel: The kernel for ``"kernel_pca"``.
        n_neighbors: Graph degree for ``"laplacian_eigenmaps"``.
        standardize: Scale each column to unit variance, matching the GPLVM
            prior $X \\sim \\mathcal N(0, I_Q)$.

    Returns:
        The latents. Deterministic: each column's sign is fixed so that its
        largest-magnitude entry is positive.

    Raises:
        ValueError: For an unknown method, or ``"kernel_pca"`` without a
            kernel.

    Examples:
        Start a latent-factor model's latents ``Z_T`` (stored transposed,
        ``(Q, N)``) on the data manifold:

        ```python
        from numpyro.infer.autoguide import AutoDelta
        from numpyro.infer.initialization import init_to_value

        Z0 = latent_init(Y, 6, method="laplacian_eigenmaps")
        guide = AutoDelta(lfr_model, init_loc_fn=init_to_value(values={"Z_T": Z0.T}))
        ```
    """
    Y = jnp.asarray(Y)
    if method == "pca":
        centred = einx.subtract("n d, d -> n d", Y, jnp.mean(Y, axis=0))
        U, s, _ = jnp.linalg.svd(centred, full_matrices=False)
        Z = U[:, :n_latent] * s[:n_latent]
    elif method == "kernel_pca":
        if kernel is None:
            raise ValueError('method="kernel_pca" needs a kernel')
        Z = kl.KernelPCA(kernel, n_components=n_latent).fit(Y).embedding
    elif method == "laplacian_eigenmaps":
        Z = (
            kl.LaplacianEigenmaps(n_components=n_latent, n_neighbors=n_neighbors)
            .fit(Y)
            .embedding
        )
    else:
        raise ValueError(
            "method must be 'pca', 'kernel_pca' or 'laplacian_eigenmaps', "
            f"got {method!r}"
        )
    Z = jnp.asarray(Z)
    peak = jnp.take_along_axis(Z, jnp.argmax(jnp.abs(Z), axis=0)[None, :], axis=0)
    Z = Z * jnp.where(peak < 0, -1.0, 1.0)
    if standardize:
        centred = einx.subtract("n q, q -> n q", Z, jnp.mean(Z, axis=0))
        Z = einx.divide("n q, q -> n q", centred, jnp.std(Z, axis=0))
    return Z
