r"""Stationary-kernel spectral densities evaluated at frequency magnitudes.

For a stationary kernel $k(r)$ on $\mathbb{R}^D$ with spectral
density $S(\omega)$ (Bochner), the inter-domain inducing-feature
reduction gives a diagonal $K_{uu}$ whose entries are
$S(\sqrt{\lambda_j})$ evaluated at the basis eigenvalues. This
module computes $S(\sqrt{\lambda})$ for each kernel in
`pyrox_gp._kernels` that has a registered closed-form spectral density.

Supported (1D, isotropic):

- `pyrox_gp.RBF` —
  $S(\omega) = \sigma^2 \ell \sqrt{2\pi}\,\exp(-\ell^2 \omega^2 / 2)$.
- `pyrox_gp.Matern` (``nu in {0.5, 1.5, 2.5, ...}``) —
  $S(\omega) = c_\nu\,(2\nu/\ell^2 + \omega^2)^{-(\nu+1/2)}$ with
  $c_\nu = \sigma^2\,\tfrac{2\sqrt{\pi}\,\Gamma(\nu+1/2)}{\Gamma(\nu)}\,
  (2\nu/\ell^2)^\nu$.

For higher input dimensions the density is the radial form raised to the
``D``-th power for the lengthscale prefactor (RBF) or the standard ``D``-d
Matern formula. The two stationary kernels above carry the lengthscale
exponent of ``D``; non-stationary kernels (``Linear``, ``Polynomial``)
and bounded-spectrum kernels (``Periodic``, ``Cosine``) raise
`NotImplementedError`.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jaxtyping import Array, Float

from pyrox_gp._kernels import RBF, Matern
from pyrox_gp._protocols import Kernel


def _rbf_spectral_density(
    eigvals: Float[Array, " M"],
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""],
    D: int,
) -> Float[Array, " M"]:
    r"""$S(\omega) = \sigma^2 \ell^D (2\pi)^{D/2} \exp(-\ell^2 \omega^2 / 2)$."""
    omega_sq = eigvals  # eigvals are squared frequencies
    prefactor = variance * (lengthscale**D) * (2.0 * math.pi) ** (D / 2.0)
    return prefactor * jnp.exp(-0.5 * (lengthscale**2) * omega_sq)


def _matern_spectral_density(
    eigvals: Float[Array, " M"],
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""],
    nu: float,
    D: int,
) -> Float[Array, " M"]:
    r"""Matern spectral density.

    $$
    S(\omega) = \sigma^2
    \frac{2^D \pi^{D/2} \Gamma(\nu+D/2) (2\nu)^\nu}{\Gamma(\nu)\,\ell^{2\nu}}
    \,(2\nu/\ell^2 + \omega^2)^{-(\nu + D/2)}.
    $$
    """
    omega_sq = eigvals
    alpha = 2.0 * nu / (lengthscale**2)
    log_c = (
        D * math.log(2.0)
        + (D / 2.0) * math.log(math.pi)
        + math.lgamma(nu + D / 2.0)
        - math.lgamma(nu)
    )
    # alpha^nu = (2nu)^nu / lengthscale^(2nu) — carries the lengthscale exponent.
    prefactor = variance * jnp.exp(log_c) * alpha**nu
    return prefactor * (alpha + omega_sq) ** (-(nu + D / 2.0))


def spectral_density(
    kernel: Kernel,
    eigvals: Float[Array, " M"],
    *,
    D: int = 1,
) -> Float[Array, " M"]:
    """Dispatch to the kernel-specific spectral density at ``sqrt(eigvals)``.

    Args:
        kernel: A stationary kernel. Currently `pyrox_gp.RBF` and
            `pyrox_gp.Matern` are registered.
        eigvals: Squared frequency magnitudes $\\lambda_j = \\omega_j^2$,
            shape ``(M,)``.
        D: Input dimension of the underlying domain (the kernel itself does
            not always carry this — pass it explicitly).

    Returns:
        ``S(sqrt(eigvals))`` of shape ``(M,)``.

    Raises:
        NotImplementedError: For kernels without a registered closed-form
            spectral density, or for ARD (per-dimension) lengthscales,
            which these isotropic closed forms cannot represent.
    """
    lengthscale = kernel.get_param("lengthscale")  # ty: ignore[unresolved-attribute]
    if jnp.ndim(lengthscale) != 0:
        raise NotImplementedError(
            "Spectral densities are registered for isotropic kernels only; "
            f"got a lengthscale of shape {jnp.shape(lengthscale)} (ARD). The "
            "closed forms below take a scalar lengthscale and would silently "
            "pair input dimensions with unrelated frequencies. Use a kernel "
            "built without input_dim for the inducing-feature path."
        )
    if isinstance(kernel, RBF):
        return _rbf_spectral_density(
            eigvals,
            kernel.get_param("variance"),
            kernel.get_param("lengthscale"),
            D,
        )
    if isinstance(kernel, Matern):
        return _matern_spectral_density(
            eigvals,
            kernel.get_param("variance"),
            kernel.get_param("lengthscale"),
            kernel.nu,
            D,
        )
    raise NotImplementedError(
        f"Spectral density for {type(kernel).__name__} is not registered. "
        "Currently only RBF and Matern are supported; open an issue to add more."
    )
