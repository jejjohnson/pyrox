"""Collapsed latent-factor regression — the decoder is marginalized, not fixed.

Unlike `pyrox_gp.LMCKernel` / `pyrox_gp.ICMKernel` /
`pyrox_gp.OILMMKernel`, which hold the mixing matrix as a concrete
array, this module treats it as a random variable with a fixed
$\\mathcal{N}(0, 1)$ prior and integrates it out analytically. The
resulting likelihood factorizes only a $Q \\times Q$ matrix and costs
$O(NQP)$ in the output dimension, so $P$ in the tens of thousands is
routine.

Adapted from the GPLFR reference implementation (Stevenson et al., 2026,
arXiv:2606.06576), MIT licensed.
"""

from __future__ import annotations

import einx
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
from jaxtyping import Array, Float


def _check_rows_match(Y: Float[Array, "N P"], Z: Float[Array, "N Q"]) -> None:
    """Raise if ``Y`` and ``Z`` disagree on the number of training points."""
    if Y.ndim != 2 or Z.ndim != 2:
        raise ValueError(
            f"Y and Z must be 2-D arrays of shapes (N, P) and (N, Q); "
            f"got shapes {jnp.shape(Y)} and {jnp.shape(Z)}."
        )
    if Y.shape[0] != Z.shape[0]:
        raise ValueError(
            f"Y and Z must share the leading (training-point) axis N; "
            f"got Y with {Y.shape[0]} rows and Z with {Z.shape[0]} rows."
        )


def _psi_factor(
    Z: Float[Array, "N Q"],
    noise_var: Float[Array, ""],
) -> tuple[Float[Array, "Q Q"], bool]:
    """Cholesky factor of the capacitance matrix.

    The capacitance matrix is $\\Psi = I_Q + \\sigma^{-2} Z^\\top Z$. One
    factorization serves the log-density, the decoder mean, and the decoder
    covariance, so it is computed here and shared.
    """
    Q = Z.shape[1]
    gram = einx.dot("n q, n r -> q r", Z, Z)
    return cho_factor(jnp.eye(Q, dtype=Z.dtype) + gram / noise_var, lower=True)


def collapsed_lfr_log_prob(
    Y: Float[Array, "N P"],
    Z: Float[Array, "N Q"],
    noise_var: Float[Array, ""],
    *,
    jitter: float = 1e-12,
) -> Float[Array, ""]:
    """Log-likelihood with the linear decoder marginalized analytically.

    Evaluates

    $$
    p(Y \\mid Z, \\sigma^2) = \\prod_{j=1}^{P}
        \\mathcal{N}(y_j \\mid 0,\\; ZZ^\\top + \\sigma^2 I_N)
    $$

    via the Woodbury identity in the $Q \\times Q$ capacitance matrix
    $\\Psi = I_Q + \\sigma^{-2} Z^\\top Z$, so no $(N, N)$ matrix is ever
    formed. Cost is $O(NQ^2 + NQP + Q^3)$; the output dimension $P$ enters
    only through $\\lVert Y \\rVert_F^2$ and $Z^\\top Y$.

    Args:
        Y: Observations of shape ``(N, P)``. Assumed centered.
        Z: Latent factor values at the training inputs, shape ``(N, Q)``.
        noise_var: Scalar isotropic observation noise variance. Per-output
            noise is not supported — it breaks the shared-covariance identity
            the derivation rests on.
        jitter: Added to ``noise_var`` before inversion.

    Returns:
        Scalar log-likelihood.

    Examples:
        >>> import jax.numpy as jnp
        >>> Y = jnp.zeros((5, 100))
        >>> Z = jnp.ones((5, 2))
        >>> float(collapsed_lfr_log_prob(Y, Z, 0.1)) < 0.0
        True
    """
    _check_rows_match(Y, Z)
    N, P = Y.shape
    s2 = noise_var + jitter
    c, low = _psi_factor(Z, s2)
    logdet_psi = 2.0 * jnp.sum(jnp.log(jnp.diagonal(c)))
    ZTY = einx.dot("n q, n p -> q p", Z, Y)
    alpha = cho_solve((c, low), ZTY)
    quad = jnp.sum(Y * Y) / s2 - jnp.sum(ZTY * alpha) / s2**2
    return -0.5 * (
        N * P * jnp.log(2.0 * jnp.pi) + N * P * jnp.log(s2) + P * logdet_psi + quad
    )


def decoder_posterior(
    Y: Float[Array, "N P"],
    Z: Float[Array, "N Q"],
    noise_var: Float[Array, ""],
    *,
    jitter: float = 1e-12,
) -> tuple[Float[Array, "Q P"], Float[Array, "Q Q"]]:
    """Closed-form matrix-normal posterior over the decoder.

    $$
    p(W \\mid Z, Y, \\sigma) = \\mathcal{MN}_{Q \\times P}
        \\big(\\sigma^{-2}\\Psi^{-1}Z^\\top Y,\\; \\Psi^{-1},\\; I_P\\big)
    $$

    The row covariance is shared across all $P$ columns and the column
    covariance is the identity, so the full $QP \\times QP$ posterior
    covariance is exactly $\\Psi^{-1} \\otimes I_P$ and is never built.

    Args:
        Y: Observations of shape ``(N, P)``.
        Z: Latent factor values, shape ``(N, Q)``.
        noise_var: Scalar isotropic observation noise variance.
        jitter: Added to ``noise_var`` before inversion.

    Returns:
        Tuple of ``(mean, row_cov)`` with shapes ``(Q, P)`` and ``(Q, Q)``.
    """
    _check_rows_match(Y, Z)
    s2 = noise_var + jitter
    c, low = _psi_factor(Z, s2)
    mean = cho_solve((c, low), einx.dot("n q, n p -> q p", Z, Y)) / s2
    row_cov = cho_solve((c, low), jnp.eye(Z.shape[1], dtype=Z.dtype))
    return mean, row_cov


def lfr_predictive_moments(
    z_mean: Float[Array, "T Q"],
    z_var: Float[Array, "T Q"],
    mu_W: Float[Array, "Q P"],
    Sigma_W: Float[Array, "Q Q"],
    noise_var: Float[Array, ""] | None = None,
) -> tuple[Float[Array, "T P"], Float[Array, "T P"]]:
    """Exact moments of the product of two independent Gaussians.

    For $z \\sim \\mathcal{N}(m, \\mathrm{diag}(v))$ independent of
    $W \\sim \\mathcal{MN}(\\mu_W, \\Sigma_W, I_P)$, the predictive
    $f = z^\\top W$ has

    $$
    \\mathrm{Var}[f_j] = m^\\top \\Sigma_W m
        + \\sum_q v_q \\mu_{W,qj}^2
        + \\sum_q v_q \\Sigma_{W,qq}
    $$

    The first and third terms do not depend on the output index, so they are
    computed once as ``(T, 1)`` columns and broadcast over ``P``. Predictive
    variance for very wide outputs therefore costs barely more than the mean.

    Args:
        z_mean: Latent predictive means, shape ``(T, Q)``.
        z_var: Latent predictive marginal variances, shape ``(T, Q)``.
        mu_W: Decoder posterior mean, shape ``(Q, P)``.
        Sigma_W: Decoder posterior row covariance, shape ``(Q, Q)``.
        noise_var: If given, added to the variance for the observation-noise
            predictive rather than the signal predictive.

    Returns:
        Tuple of ``(mean, variance)``, both shape ``(T, P)``.
    """
    mean = einx.dot("t q, q p -> t p", z_mean, mu_W)
    decoder = jnp.sum((z_mean @ Sigma_W) * z_mean, axis=-1, keepdims=True)
    latent = einx.dot("t q, q p -> t p", z_var, mu_W**2)
    cross = z_var @ jnp.diagonal(Sigma_W)[:, None]
    var = decoder + latent + cross
    return mean, var if noise_var is None else var + noise_var
