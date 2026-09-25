r"""Stationary-kernel spectral densities evaluated at frequency magnitudes.

For a stationary kernel $k(r)$ on $\mathbb{R}^D$ with spectral
density $S(\omega)$ (Bochner), the inter-domain inducing-feature
reduction gives a diagonal $K_{uu}$ whose entries are
$S(\sqrt{\lambda_j})$ evaluated at the basis eigenvalues. This
module computes $S(\sqrt{\lambda})$ by delegating to kernellib:
a pyrox kernel is frozen to its kernellib counterpart
(`_ParameterizedKernel.frozen`) and kernellib's
`AbstractStationaryKernel.unit_spectral_density` supplies the closed form.

Supported: kernels whose kernellib counterpart has a closed-form density,
`pyrox_gp.RBF` and `pyrox_gp.Matern` (and the kernellib ``RBF`` /
``Matern`` directly):

- RBF — $S(\omega) = \sigma^2 \ell^D (2\pi)^{D/2}\,\exp(-\ell^2 \omega^2 / 2)$.
- Matern — $S(\omega) = \sigma^2
  \frac{2^D \pi^{D/2} \Gamma(\nu+D/2) (2\nu)^\nu}{\Gamma(\nu)\,\ell^{2\nu}}
  \,(2\nu/\ell^2 + \omega^2)^{-(\nu + D/2)}$.

The density here is radial: it takes squared frequency *magnitudes*, so it is
defined for isotropic lengthscales only. Non-stationary kernels
(``Linear``, ``Polynomial``), bounded-spectrum kernels (``Periodic``,
``Cosine``) and ``RationalQuadratic`` (no closed form in kernellib) raise
`NotImplementedError`.
"""

from __future__ import annotations

import jax.numpy as jnp
import kernellib as kl
from jaxtyping import Array, Float

from pyrox_gp._context import _kernel_context
from pyrox_gp._protocols import Kernel


def spectral_density(
    kernel: Kernel,
    eigvals: Float[Array, " M"],
    *,
    D: int = 1,
) -> Float[Array, " M"]:
    """Kernel spectral density at ``sqrt(eigvals)``.

    Args:
        kernel: A stationary kernel with a closed-form density: a pyrox
            `pyrox_gp.RBF` / `pyrox_gp.Matern`, or a kernellib kernel.
        eigvals: Squared frequency magnitudes $\\lambda_j = \\omega_j^2$,
            shape ``(M,)``.
        D: Input dimension of the underlying domain (the kernel itself does
            not always carry this — pass it explicitly).

    Returns:
        ``S(sqrt(eigvals))`` of shape ``(M,)``.

    Raises:
        NotImplementedError: For kernels without a closed-form spectral
            density, or for ARD (per-dimension) lengthscales, which a radial
            density cannot represent.
    """
    frozen = _freeze(kernel)
    if not isinstance(frozen, kl.AbstractStationaryKernel):
        raise NotImplementedError(
            f"Spectral density for {type(kernel).__name__} is not registered. "
            "Currently only RBF and Matern are supported; open an issue to add more."
        )
    lengthscale = frozen.lengthscale
    if jnp.ndim(lengthscale) != 0:
        if jnp.size(lengthscale) == 1 and D == 1:
            # A one-element per-axis lengthscale (input_dim=1) is a scalar in
            # disguise, but only on a one-dimensional domain: for D > 1 the
            # kernel itself would reject the inputs, so the density must not
            # describe a domain it cannot evaluate.
            lengthscale = jnp.reshape(lengthscale, ())
        else:
            raise NotImplementedError(
                "Spectral densities are registered for isotropic kernels "
                f"only; got a lengthscale of shape {jnp.shape(lengthscale)} "
                "(ARD). The closed forms take a scalar lengthscale and would "
                "silently pair input dimensions with unrelated frequencies. Use "
                "a kernel built without input_dim for this path."
            )
    # S(w) = variance * l^D * s(l^2 |w|^2), with s kernellib's
    # unit-lengthscale density. Working in squared magnitudes avoids a sqrt,
    # whose gradient is infinite at a zero eigenvalue.
    unit = frozen.unit_spectral_density(lengthscale**2 * eigvals, D)
    return frozen.variance * lengthscale**D * unit


def _freeze(kernel: Kernel) -> kl.AbstractKernel:
    """kernellib kernel for ``kernel``, resolving pyrox params once."""
    frozen_fn = getattr(kernel, "frozen", None)
    if frozen_fn is None:
        return kernel
    try:
        # One context for every parameter, so a prior registers one site.
        with _kernel_context(kernel):
            return frozen_fn()
    except NotImplementedError:
        return kernel
