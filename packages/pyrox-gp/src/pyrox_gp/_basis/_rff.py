r"""Random Fourier feature prior draws for stationary kernels.

Pure-JAX helpers that factor a single posterior-sample path into
``(variance, lengthscale, omega, phase, weights)`` so a zero-mean prior
function can be evaluated at arbitrary inputs via

$$
\tilde{f}(x) = \sum_{j=1}^F w_j
    \sqrt{2 \sigma^2 / F}\,
    \cos\!\bigl(\omega_j^\top x / \ell + b_j\bigr),
\qquad w_j \sim \mathcal{N}(0, 1),
\quad b_j \sim \mathrm{Unif}(0, 2\pi),
$$

with $\omega_j$ drawn from the kernel's spectral density. The
zero-mean prior mean ``E[\tilde f(x)\tilde f(x')]`` converges to the
stationary kernel $k(x, x')$ as ``F \to \infty``.

These helpers are the shared RFF primitive behind `pyrox_gp._pathwise`
(pathwise posterior samplers via Matheron's rule). They are stateless,
deterministic in a PRNG key, batched along a leading path axis, and
friendly to ``jax.jit`` / ``jax.grad`` — no NumPyro sample sites and no
`pyrox._core.PyroxModule` state. The existing sample-site RFF
layers in `pyrox_nn._layers` (`RBFFourierFeatures`,
`MaternFourierFeatures`) register their frequencies as
``pyrox_sample`` sites so an SVI guide can learn a posterior; pathwise
samplers want a frozen single-key draw, which is what this module
provides.

Supported kernels (following the existing
`pyrox_nn.MaternFourierFeatures` / `RBFFourierFeatures`
convention so the two RFF stacks agree when given the same kernel and
key):

* `pyrox_gp.RBF` — $\omega \sim \mathcal{N}(0, I)$,
  effective frequency $\omega / \ell$.
* `pyrox_gp.Matern` — $\omega \sim \mathrm{StudentT}(2\nu)$
  drawn coordinate-wise, effective frequency $\omega / \ell$.
  For ``D > 1`` the coordinate-wise draw is an approximation to the
  true multivariate Matern spectrum (a multivariate Student-t rather
  than a product of 1D t's) — this matches the existing pyrox RFF
  layers and is widely used in practice.

Other stationary kernels will raise `NotImplementedError`;
`pyrox_gp._basis.spectral_density` lists the same supported pair.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float

from pyrox_gp._context import _kernel_context
from pyrox_gp._kernels import RBF, Matern
from pyrox_gp._protocols import Kernel


def _draw_spectral_frequencies(
    kernel: Kernel,
    key: Array,
    *,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> Float[Array, " ..."]:
    """Sample spectral frequencies for the kernel's spectral density.

    Pure function of the kernel *type* (and ``Matern.nu``); no
    hyperparameter prior sites are registered, so this is safe to call
    outside any `_kernel_context`.
    """
    if isinstance(kernel, RBF):
        return jax.random.normal(key, shape=shape, dtype=dtype)
    if isinstance(kernel, Matern):
        return jnp.asarray(
            dist.StudentT(df=2.0 * kernel.nu).sample(key, sample_shape=shape),
            dtype=dtype,
        )
    raise NotImplementedError(
        "RFF frequency sampling currently supports RBF and Matern kernels; "
        f"got {type(kernel).__name__}."
    )


def _read_kernel_hyperparams(
    kernel: Kernel,
    *,
    dtype: jnp.dtype,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Read ``(variance, lengthscale)`` from the kernel.

    Must be called inside a `pyrox_gp._context._kernel_context`
    so Pattern B / C kernels register their ``pyrox_sample`` sites once.
    """
    if isinstance(kernel, (RBF, Matern)):
        return (
            jnp.asarray(kernel.get_param("variance"), dtype=dtype),
            jnp.asarray(kernel.get_param("lengthscale"), dtype=dtype),
        )
    raise NotImplementedError(
        "Hyperparameter read currently supports RBF and Matern kernels; "
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
        kernel: A stationary kernel supported by
            `pyrox_gp._basis._rff._draw_spectral_frequencies`.
        key: PRNG key — split internally into frequency / phase /
            weight subkeys.
        n_paths: Number of independent prior function draws ``S``.
        n_features: Number of random features per draw ``F``.
        in_features: Input dimension ``D``.
        dtype: Floating dtype for all outputs.
        variance, lengthscale: Optional pre-resolved scalar overrides.
            When provided, the helper skips ``kernel.get_param`` —
            essential when the same hyperparameter draw needs to be
            reused across a chain of operations (e.g. matching the
            cached operator on a `ConditionedGP`). When
            ``None``, the values are read from the kernel under a
            fresh `_kernel_context`, which resamples hyperparam
            priors for Pattern B/C kernels.

    Returns:
        ``(variance, lengthscale, omega, phase, weights)`` where
        ``variance`` and ``lengthscale`` are either the supplied
        overrides or scalars read from the kernel; ``omega`` has shape
        ``(S, D, F)``, and ``phase`` / ``weights`` have shape ``(S, F)``.

    Raises:
        ValueError: If ``n_paths < 1`` or ``n_features < 1``, or if
            exactly one of ``variance`` / ``lengthscale`` is supplied.
        NotImplementedError: For unsupported kernels.
    """
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}.")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}.")
    if (variance is None) != (lengthscale is None):
        raise ValueError(
            "variance and lengthscale must both be supplied together or both omitted."
        )

    freq_key, phase_key, weight_key = jax.random.split(key, 3)

    # Spectral frequency draw is pure (depends only on kernel *type*),
    # so it never needs a kernel context. Reading variance/lengthscale
    # does — gate behind the optional override path.
    omega = _draw_spectral_frequencies(
        kernel,
        freq_key,
        shape=(n_paths, in_features, n_features),
        dtype=dtype,
    )
    if variance is not None and lengthscale is not None:
        variance = jnp.asarray(variance, dtype=dtype)
        lengthscale = jnp.asarray(lengthscale, dtype=dtype)
    else:
        with _kernel_context(kernel):
            variance, lengthscale = _read_kernel_hyperparams(kernel, dtype=dtype)

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
    lengthscale: Float[Array, ""] | Float[Array, " D"],
    omega: Float[Array, "S D F"],
    phase: Float[Array, "S F"],
    weights: Float[Array, "S F"],
) -> Float[Array, "S N"]:
    r"""Evaluate the zero-mean RFF prior path(s) at inputs ``X``.

    Implements

    $$
    \tilde f_s(x_n) = \sum_{j=1}^{F} w_{s,j}\,
        \sqrt{2\sigma^2 / F}\,
        \cos\!\bigl(\omega_{s,\cdot,j}^\top x_n / \ell
                   + b_{s,j}\bigr),
    $$

    vectorized over path index ``s`` and input index ``n``. See the
    module docstring for the reconstruction argument.
    """
    # Project inputs onto each path's frequencies (contract feature dim d),
    # then add the per-path phase broadcast over the input axis n → (S, N, F).
    # Scale along the *input* axis before contracting it. For a scalar
    # lengthscale this is identical to dividing the projection, but it is
    # also correct for a ``(D,)`` ARD lengthscale, where dividing the
    # ``(S, N, F)`` projection would broadcast F against D and fail.
    if jnp.ndim(lengthscale) != 0:
        d = jnp.size(lengthscale)
        if X.shape[-1] != d or omega.shape[-2] != d:
            raise ValueError(
                f"ARD lengthscale of size {d} requires inputs and frequencies "
                f"with that many features; got X with {X.shape[-1]} and omega "
                f"with {omega.shape[-2]}. A singleton feature axis on X would "
                "broadcast silently and repeat one coordinate across every "
                "dimension."
            )
    projected = einx.dot("n d, s d f -> s n f", X / lengthscale, omega)
    angles = einx.add("s n f, s f -> s n f", projected, phase)
    features = jnp.sqrt(2.0 * variance / omega.shape[-1]) * jnp.cos(angles)
    # Weighted sum over the F random features → (S, N).
    return einx.dot("s n f, s f -> s n", features, weights)
