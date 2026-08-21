"""Pure JAX kernel evaluation primitives — math definitions only.

Each function takes two input matrices ``X1`` of shape ``(N1, D)``,
``X2`` of shape ``(N2, D)``, hyperparameters as JAX arrays, and returns
the ``(N1, N2)`` Gram matrix. All inputs are 2-D; callers that have 1-D
arrays must add a trailing singleton dimension first.

These are the canonical closed-form *math* for each kernel — small,
readable, and tutorial-facing. The companion *scalable construction*
surface (numerically stable matrix assembly, mixed-precision
accumulation, implicit/structured operators, batched matvec) lives in
`gaussx`; see `gaussx.stable_rbf_kernel` and
`gaussx.ImplicitKernelOperator` for the production path.

The composition helpers `kernel_add` and `kernel_mul` act on
already-evaluated Gram matrices, not on callables. Higher-level
`pyrox_gp.Kernel` classes (Wave 2 Layer 1, see issue #20) compose
callables and may opt in to gaussx's scalable variants when needed.

Index axes are named via `einx` (``einx.dot`` / ``einx.id`` /
``einx.add``) rather than raw broadcasting so shape intent stays legible
at the call site.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _pairwise_sq_dist(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    lengthscale: Float[Array, ""] | Float[Array, " D"] | float = 1.0,
    *,
    center: bool = True,
) -> Float[Array, "N1 N2"]:
    """Squared Euclidean distance matrix, optionally lengthscale-scaled.

    Scaling the inputs *before* the
    $\\|x\\|^2 + \\|x'\\|^2 - 2\\,x^\\top x'$ expansion supports a
    per-dimension lengthscale while keeping every intermediate at
    ``(N1,)``, ``(N2,)``, or ``(N1, N2)`` — no ``(N1, N2, D)`` broadcast
    tensor is ever built. Clipped at zero to absorb the small negative
    values that arise from float cancellation on near-identical points.

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) lengthscale.
            Broadcasts against the trailing input axis in both cases.
        center: Subtract a shared offset before scaling. Exact in real
            arithmetic (squared distance is translation invariant) but it
            keeps the ``‖x‖²`` terms on the scale of the data's *spread*
            rather than its absolute magnitude, which matters once inputs
            sit far from the origin and the lengthscale is small. Callers
            that do not scale (`periodic_kernel`, `cosine_kernel`) pass
            ``False`` so their output stays bit-identical to the
            pre-ARD implementation.

    Returns:
        ``(N1, N2)`` scaled squared distances.
    """
    if center:
        # Shared offset ⇒ exact, since squared distance is translation
        # invariant. Without it, inputs far from the origin divided by a
        # small lengthscale produce ‖x‖² terms that swamp their own
        # difference — losing precision off the diagonal and, in the limit,
        # overflowing to inf so the diagonal becomes inf - inf = NaN.
        offset = jax.lax.stop_gradient(jnp.mean(X1, axis=0))
        X1 = X1 - offset
        X2 = X2 - offset
    X1 = X1 / lengthscale
    X2 = X2 / lengthscale
    n1 = einx.dot("n1 d, n1 d -> n1", X1, X1)
    n2 = einx.dot("n2 d, n2 d -> n2", X2, X2)
    cross = einx.dot("n1 d, n2 d -> n1 n2", X1, X2)
    # ‖xᵢ‖² + ‖x′ⱼ‖² broadcast to (N1, N2) via a named outer sum.
    norm_sum = einx.add("n1, n2 -> n1 n2", n1, n2)
    return jnp.clip(norm_sum - 2.0 * cross, min=0.0)


def rbf_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""] | Float[Array, " D"],
) -> Float[Array, "N1 N2"]:
    r"""Radial basis function (squared exponential) kernel.

    $$
    k(x, x') = \sigma^2 \exp\!\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)
    $$

    With a ``(D,)`` lengthscale each input dimension is scaled by its own
    $\ell_d$ (automatic relevance determination, ARD).

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar signal variance ``sigma^2``.
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) lengthscale ``ell``.

    Returns:
        ``(N1, N2)`` kernel Gram matrix.
    """
    sq = _pairwise_sq_dist(X1, X2, lengthscale)
    return variance * jnp.exp(-0.5 * sq)


def matern_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""] | Float[Array, " D"],
    nu: float,
) -> Float[Array, "N1 N2"]:
    r"""Matern kernel with closed-form ``nu in {1/2, 3/2, 5/2}``.

    $$
    k(x, x') = \sigma^2\, f_\nu(r / \ell),
    \qquad r = \|x - x'\|
    $$

    Only the three common half-integer orders are supported because those
    admit closed-form expressions without Bessel evaluations. ``nu`` is a
    static Python float (not a JAX array) so the branch specializes at
    trace time. With a ``(D,)`` lengthscale each input dimension is scaled
    by its own $\ell_d$ (automatic relevance determination, ARD).

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar signal variance.
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) lengthscale.
        nu: Smoothness parameter; must be ``0.5``, ``1.5``, or ``2.5``.

    Returns:
        ``(N1, N2)`` Gram matrix.
    """
    sq = _pairwise_sq_dist(X1, X2, lengthscale)
    # Jitter inside sqrt avoids NaN gradients at r = 0 (sqrt' is undefined).
    r = jnp.sqrt(jnp.clip(sq, min=1e-30))
    if nu == 0.5:
        shape = jnp.exp(-r)
    elif nu == 1.5:
        a = jnp.sqrt(3.0) * r
        shape = (1.0 + a) * jnp.exp(-a)
    elif nu == 2.5:
        a = jnp.sqrt(5.0) * r
        shape = (1.0 + a + (a * a) / 3.0) * jnp.exp(-a)
    else:
        raise ValueError(f"matern_kernel supports nu in {{0.5, 1.5, 2.5}}, got {nu!r}")
    return variance * shape


def periodic_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""],
    period: Float[Array, ""],
) -> Float[Array, "N1 N2"]:
    r"""Periodic (MacKay) kernel.

    $$
    k(x, x') = \sigma^2 \exp\!\left(
        -\frac{2 \sin^2(\pi \|x - x'\| / p)}{\ell^2}
    \right)
    $$

    For multi-dimensional inputs the argument uses the Euclidean distance,
    matching the common GPML convention.

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar signal variance.
        lengthscale: Scalar lengthscale.
        period: Scalar period ``p``.

    Returns:
        ``(N1, N2)`` Gram matrix.
    """
    sq = _pairwise_sq_dist(X1, X2, center=False)
    # Jitter inside sqrt avoids NaN gradients at r = 0 (sqrt' is undefined).
    r = jnp.sqrt(jnp.clip(sq, min=1e-30))
    sinsq = jnp.sin(jnp.pi * r / period) ** 2
    return variance * jnp.exp(-2.0 * sinsq / (lengthscale * lengthscale))


def linear_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    bias: Float[Array, ""],
) -> Float[Array, "N1 N2"]:
    r"""Linear kernel.

    $$
    k(x, x') = \sigma^2\, x^\top x' + b
    $$

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar variance multiplier on the dot product.
        bias: Scalar additive bias.

    Returns:
        ``(N1, N2)`` Gram matrix.
    """
    return variance * einx.dot("n1 d, n2 d -> n1 n2", X1, X2) + bias


def rational_quadratic_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    lengthscale: Float[Array, ""] | Float[Array, " D"],
    alpha: Float[Array, ""],
) -> Float[Array, "N1 N2"]:
    r"""Rational quadratic kernel.

    $$
    k(x, x') = \sigma^2 \left(
        1 + \frac{\|x - x'\|^2}{2\alpha \ell^2}
    \right)^{-\alpha}
    $$

    Scale mixture of RBF kernels: the limit ``alpha -> infty`` recovers the
    RBF, small ``alpha`` yields heavier-tailed correlations. With a ``(D,)``
    lengthscale each input dimension is scaled by its own $\ell_d$
    (automatic relevance determination, ARD).

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar signal variance.
        lengthscale: Scalar (isotropic) or ``(D,)`` (ARD) lengthscale.
        alpha: Scalar shape parameter; must be positive.

    Returns:
        ``(N1, N2)`` Gram matrix.
    """
    sq = _pairwise_sq_dist(X1, X2, lengthscale)
    return variance * (1.0 + sq / (2.0 * alpha)) ** (-alpha)


def polynomial_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    bias: Float[Array, ""],
    degree: int,
) -> Float[Array, "N1 N2"]:
    r"""Polynomial kernel.

    $$
    k(x, x') = \sigma^2 \bigl(x^\top x' + b\bigr)^d
    $$

    `linear_kernel` is the special case ``degree == 1`` without the
    outer power. ``degree`` is a static Python int so the kernel specializes
    at trace time.

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar multiplier.
        bias: Scalar additive bias inside the power.
        degree: Positive integer polynomial degree.

    Returns:
        ``(N1, N2)`` Gram matrix.
    """
    if degree < 1:
        raise ValueError(f"polynomial_kernel requires degree >= 1, got {degree!r}")
    dot = einx.dot("n1 d, n2 d -> n1 n2", X1, X2)
    return variance * (dot + bias) ** degree


def cosine_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
    period: Float[Array, ""],
) -> Float[Array, "N1 N2"]:
    r"""Cosine kernel.

    $$
    k(x, x') = \sigma^2 \cos\!\left(
        \frac{2 \pi \|x - x'\|}{p}
    \right)
    $$

    Useful as a simple periodic building block alongside
    `periodic_kernel`; unlike the Mackay form this one uses plain
    cosine of distance and can go negative.

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar signal variance.
        period: Scalar period.

    Returns:
        ``(N1, N2)`` Gram matrix.
    """
    sq = _pairwise_sq_dist(X1, X2, center=False)
    # Jitter inside sqrt avoids NaN gradients at r = 0 (sqrt' is undefined).
    r = jnp.sqrt(jnp.clip(sq, min=1e-30))
    return variance * jnp.cos(2.0 * jnp.pi * r / period)


def white_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
) -> Float[Array, "N1 N2"]:
    r"""White-noise kernel.

    $$
    k(x, x') = \sigma^2 \,\delta(x, x')
    $$

    Nonzero only where ``X1[i]`` exactly matches ``X2[j]`` across all feature
    dimensions. When evaluated at ``X1 == X2`` this yields ``sigma^2 * I``.

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar noise variance.

    Returns:
        ``(N1, N2)`` Gram matrix.
    """
    # Pairwise feature difference (N1, N2, D) via named broadcasting.
    diff = einx.subtract("n1 d, n2 d -> n1 n2 d", X1, X2)
    match = jnp.all(diff == 0.0, axis=-1)
    return variance * match.astype(X1.dtype)


def constant_kernel(
    X1: Float[Array, "N1 D"],
    X2: Float[Array, "N2 D"],
    variance: Float[Array, ""],
) -> Float[Array, "N1 N2"]:
    r"""Constant kernel.

    $$
    k(x, x') = \sigma^2
    $$

    A rank-one kernel useful as a scalar offset additive component.

    Args:
        X1: ``(N1, D)`` inputs.
        X2: ``(N2, D)`` inputs.
        variance: Scalar value.

    Returns:
        ``(N1, N2)`` Gram matrix filled with ``variance``.
    """
    return variance * jnp.ones((X1.shape[0], X2.shape[0]), dtype=X1.dtype)


def kernel_add(
    K1: Float[Array, "N1 N2"],
    K2: Float[Array, "N1 N2"],
) -> Float[Array, "N1 N2"]:
    """Pointwise sum of two already-evaluated Gram matrices."""
    return K1 + K2


def kernel_mul(
    K1: Float[Array, "N1 N2"],
    K2: Float[Array, "N1 N2"],
) -> Float[Array, "N1 N2"]:
    """Pointwise (Hadamard) product of two already-evaluated Gram matrices."""
    return K1 * K2
