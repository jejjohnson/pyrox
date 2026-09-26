r"""Random Fourier feature prior draws for stationary kernels.

The draw and the evaluation live in kernellib
(`kernellib.draw_rff_cosine_basis`, `kernellib.evaluate_rff_cosine_paths`);
this module adapts pyrox kernels to them. A single prior path is factored into
``(variance, lengthscale, omega, phase, weights)`` and evaluated as

$$
\tilde{f}(x) = \sum_{j=1}^F w_j
    \sqrt{2 \sigma^2 / F}\,
    \cos\!\bigl(\omega_j^\top x / \ell + b_j\bigr),
\qquad w_j \sim \mathcal{N}(0, 1),
\quad b_j \sim \mathrm{Unif}(0, 2\pi),
$$

with $\omega_j$ drawn at unit lengthscale from the kernel's spectral density.
The empirical path covariance converges to the kernel as ``F`` grows.

These helpers back `pyrox_gp._pathwise` (pathwise posterior samplers via
Matheron's rule). They register no NumPyro sample sites of their own: a pyrox
kernel is frozen to its kernellib counterpart (`_ParameterizedKernel.frozen`),
which resolves its hyperparameters once, and kernellib does the rest. The
sample-site RFF layers in ``pyrox_nn`` (`RBFFourierFeatures`,
`MaternFourierFeatures`) are separate: they register their frequencies so an
SVI guide can learn a posterior.

Supported kernels are those whose kernellib counterpart has a spectral
sampler: `pyrox_gp.RBF` ($\omega \sim \mathcal{N}(0, I)$), `pyrox_gp.Matern`
(joint multivariate Student-t with $2\nu$ degrees of freedom) and
`pyrox_gp.RationalQuadratic` (a Gamma scale mixture of Gaussians). Others
raise `NotImplementedError`.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import kernellib as kl
from jaxtyping import Array, Float

from pyrox_gp._context import _kernel_context
from pyrox_gp._protocols import Kernel


evaluate_rff_cosine_paths = kl.evaluate_rff_cosine_paths


def draw_rff_cosine_basis(
    kernel: Kernel,
    key: Array,
    *,
    n_paths: int,
    n_features: int,
    in_features: int,
    dtype: jnp.dtype,
    variance: Float[Array, ""] | None = None,
    lengthscale: Float[Array, ""] | None = None,
) -> tuple[
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, "S D F"],
    Float[Array, "S F"],
    Float[Array, "S F"],
]:
    r"""Draw ``(variance, lengthscale, omega, phase, weights)`` for a kernel.

    Args:
        kernel: A pyrox or kernellib stationary kernel with a spectral
            sampler.
        key: PRNG key — split internally into frequency / phase /
            weight subkeys.
        n_paths: Number of independent prior function draws ``S``.
        n_features: Number of random features per draw ``F``.
        in_features: Input dimension ``D``.
        dtype: Floating dtype for all outputs.
        variance, lengthscale: Optional pre-resolved overrides. When
            provided, the kernel's own ``variance`` / ``lengthscale`` are not
            read — essential when the same hyperparameter draw needs to be
            reused across a chain of operations (e.g. matching the cached
            operator on a `ConditionedGP`). When ``None``, they are read
            from the kernel under a fresh `_kernel_context`, which resamples
            hyperparameter priors for Pattern B/C kernels.

    Returns:
        ``(variance, lengthscale, omega, phase, weights)`` where
        ``variance`` and ``lengthscale`` are either the supplied
        overrides or the kernel's values; ``omega`` has shape
        ``(S, D, F)``, and ``phase`` / ``weights`` have shape ``(S, F)``.

    Raises:
        ValueError: If ``n_paths < 1`` or ``n_features < 1``, or if
            exactly one of ``variance`` / ``lengthscale`` is supplied.
        NotImplementedError: For unsupported kernels.
    """
    if (variance is None) != (lengthscale is None):
        raise ValueError(
            "variance and lengthscale must both be supplied together or both omitted."
        )
    overrides = (
        {}
        if variance is None
        else {
            "variance": jnp.asarray(variance, dtype=dtype),
            "lengthscale": jnp.asarray(lengthscale, dtype=dtype),
        }
    )
    return kl.draw_rff_cosine_basis(
        _freeze(kernel, overrides),
        key,
        n_paths=n_paths,
        n_features=n_features,
        in_features=in_features,
        dtype=dtype,
    )


def _freeze(kernel: Kernel, overrides: dict[str, Array]) -> kl.AbstractKernel:
    """kernellib kernel for ``kernel``, with ``overrides`` taking precedence."""
    frozen_fn = getattr(kernel, "frozen", None)
    if frozen_fn is None:
        # Already a kernellib kernel: swap the overrides in directly.
        if not overrides:
            return kernel
        if not isinstance(kernel, kl.AbstractStationaryKernel):
            return kernel  # kernellib raises the unsupported-kernel error
        return eqx.tree_at(
            lambda k: (k.variance, k.lengthscale),
            kernel,
            (overrides["variance"], overrides["lengthscale"]),
        )
    try:
        if overrides:
            # Overridden params are not read, so no context is needed for
            # them; frozen() scopes any others in its own context.
            return frozen_fn(**overrides)
        with _kernel_context(kernel):
            return frozen_fn()
    except (NotImplementedError, ValueError):
        # No kernellib counterpart, or no lengthscale to override: let
        # kernellib's draw report the kernel as unsupported.
        return kernel
