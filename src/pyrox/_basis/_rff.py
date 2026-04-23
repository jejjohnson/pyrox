r"""Random Fourier feature prior draws for stationary kernels.

Pure-JAX helpers that factor a single posterior-sample path into
``(variance, lengthscale, omega, phase, weights)`` so a zero-mean prior
function can be evaluated at arbitrary inputs via

.. math::

    \tilde{f}(x) = \sum_{j=1}^F w_j
        \sqrt{2 \sigma^2 / F}\,
        \cos\!\bigl(\omega_j^\top x / \ell + b_j\bigr),
    \qquad w_j \sim \mathcal{N}(0, 1),
    \quad b_j \sim \mathrm{Unif}(0, 2\pi),

with :math:`\omega_j` drawn from the kernel's spectral density. The
zero-mean prior mean ``E[\tilde f(x)\tilde f(x')]`` converges to the
stationary kernel :math:`k(x, x')` as ``F \to \infty``.

These helpers are the shared RFF primitive behind :mod:`pyrox.gp._pathwise`
(pathwise posterior samplers via Matheron's rule). They are stateless,
deterministic in a PRNG key, batched along a leading path axis, and
friendly to ``jax.jit`` / ``jax.grad`` — no NumPyro sample sites and no
:class:`pyrox._core.PyroxModule` state. The existing sample-site RFF
layers in :mod:`pyrox.nn._layers` (:class:`RBFFourierFeatures`,
:class:`MaternFourierFeatures`) register their frequencies as
``pyrox_sample`` sites so an SVI guide can learn a posterior; pathwise
samplers want a frozen single-key draw, which is what this module
provides.

Supported kernels (following the existing
:class:`pyrox.nn.MaternFourierFeatures` / :class:`RBFFourierFeatures`
convention so the two RFF stacks agree when given the same kernel and
key):

* :class:`pyrox.gp.RBF` — :math:`\omega \sim \mathcal{N}(0, I)`,
  effective frequency :math:`\omega / \ell`.
* :class:`pyrox.gp.Matern` — :math:`\omega \sim \mathrm{StudentT}(2\nu)`
  drawn coordinate-wise, effective frequency :math:`\omega / \ell`.
  For ``D > 1`` the coordinate-wise draw is an approximation to the
  true multivariate Matern spectrum (a multivariate Student-t rather
  than a product of 1D t's) — this matches the existing pyrox RFF
  layers and is widely used in practice.

Other stationary kernels will raise :class:`NotImplementedError`;
:func:`pyrox._basis.spectral_density` lists the same supported pair.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float

from pyrox.gp._context import _kernel_context
from pyrox.gp._kernels import RBF, Matern
from pyrox.gp._protocols import Kernel


def _draw_spectral_frequencies_and_hyperparams(
    kernel: Kernel,
    key: Array,
    *,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, " ..."]]:
    """Sample spectral frequencies + read kernel hyperparameters.

    Must be called inside a :func:`pyrox.gp._context._kernel_context`
    so Pattern B / C kernels register their ``pyrox_sample`` sites once.
    """
    if isinstance(kernel, RBF):
        variance = jnp.asarray(kernel.get_param("variance"), dtype=dtype)
        lengthscale = jnp.asarray(kernel.get_param("lengthscale"), dtype=dtype)
        omega = jax.random.normal(key, shape=shape, dtype=dtype)
        return variance, lengthscale, omega
    if isinstance(kernel, Matern):
        variance = jnp.asarray(kernel.get_param("variance"), dtype=dtype)
        lengthscale = jnp.asarray(kernel.get_param("lengthscale"), dtype=dtype)
        omega = jnp.asarray(
            dist.StudentT(df=2.0 * kernel.nu).sample(key, sample_shape=shape),
            dtype=dtype,
        )
        return variance, lengthscale, omega
    raise NotImplementedError(
        "RFF frequency sampling currently supports RBF and Matern kernels; "
        f"got {type(kernel).__name__}."
    )


def draw_rff_cosine_basis(
    kernel: Kernel,
    key: Array,
    *,
    n_paths: int,
    n_features: int,
    in_features: int,
    dtype: jnp.dtype,
) -> tuple[
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, "S D F"],
    Float[Array, "S F"],
    Float[Array, "S F"],
]:
    r"""Draw ``(variance, lengthscale, omega, phase, weights)`` for a kernel.

    Args:
        kernel: A stationary kernel supported by
            :func:`pyrox._basis._rff._draw_spectral_frequencies`.
        key: PRNG key — split internally into frequency / phase /
            weight subkeys.
        n_paths: Number of independent prior function draws ``S``.
        n_features: Number of random features per draw ``F``.
        in_features: Input dimension ``D``.
        dtype: Floating dtype for all outputs.

    Returns:
        ``(variance, lengthscale, omega, phase, weights)`` where
        ``variance`` and ``lengthscale`` are scalars read from the
        kernel under its :func:`pyrox.gp._context._kernel_context` (so
        Pattern B / C kernels with prior'd hyperparameters register
        their ``pyrox_sample`` sites here), ``omega`` has shape
        ``(S, D, F)``, and ``phase`` / ``weights`` have shape ``(S, F)``.

    Raises:
        ValueError: If ``n_paths < 1`` or ``n_features < 1``.
        NotImplementedError: For unsupported kernels.
    """
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}.")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}.")

    freq_key, phase_key, weight_key = jax.random.split(key, 3)

    with _kernel_context(kernel):
        variance, lengthscale, omega = _draw_spectral_frequencies_and_hyperparams(
            kernel,
            freq_key,
            shape=(n_paths, in_features, n_features),
            dtype=dtype,
        )

    phase = jax.random.uniform(
        phase_key,
        shape=(n_paths, n_features),
        minval=0.0,
        maxval=2.0 * jnp.pi,
        dtype=dtype,
    )
    weights = jax.random.normal(
        weight_key,
        shape=(n_paths, n_features),
        dtype=dtype,
    )
    return variance, lengthscale, omega, phase, weights


def evaluate_rff_cosine_paths(
    X: Float[Array, "N D"],
    *,
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""],
    omega: Float[Array, "S D F"],
    phase: Float[Array, "S F"],
    weights: Float[Array, "S F"],
) -> Float[Array, "S N"]:
    r"""Evaluate the zero-mean RFF prior path(s) at inputs ``X``.

    Implements

    .. math::

        \tilde f_s(x_n) = \sum_{j=1}^{F} w_{s,j}\,
            \sqrt{2\sigma^2 / F}\,
            \cos\!\bigl(\omega_{s,\cdot,j}^\top x_n / \ell
                       + b_{s,j}\bigr),

    vectorized over path index ``s`` and input index ``n``. See the
    module docstring for the reconstruction argument.
    """
    angles = jnp.einsum("nd,sdf->snf", X, omega) / lengthscale + phase[:, None, :]
    features = jnp.sqrt(2.0 * variance / omega.shape[-1]) * jnp.cos(angles)
    return jnp.sum(features * weights[:, None, :], axis=-1)
