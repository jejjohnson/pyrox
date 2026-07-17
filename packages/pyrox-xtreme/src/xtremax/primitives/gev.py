"""Pure-JAX primitives for the Generalized Extreme Value distribution.

All functions are stateless, ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` safe.
The Gumbel limit (:math:`\\xi \\to 0`) is handled *smoothly* rather than by a
hard ``jnp.where`` branch: the kernels are written through
:func:`xtremax.primitives._common.log1p_over_x` and ``expm1_over_x`` so that
:math:`\\log(1 + \\xi z)/\\xi \\to z` and the shape-parameter gradient stays
finite and correct as :math:`\\xi \\to 0` (in particular
:math:`\\partial/\\partial\\xi \\neq 0` at :math:`\\xi = 0`).

The parameterization follows Coles (2001):

.. math:: F(x; \\mu, \\sigma, \\xi) = \\exp\\!\\left\\{-\\left[1 + \\xi
    \\frac{x - \\mu}{\\sigma}\\right]^{-1/\\xi}\\right\\}

with support :math:`\\{x : 1 + \\xi(x - \\mu)/\\sigma > 0\\}` and the
convention :math:`\\xi = 0 \\Rightarrow F(x) = \\exp(-\\exp(-z))` where
:math:`z = (x - \\mu)/\\sigma`.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import gammaln
from jaxtyping import Array, Float

from xtremax.primitives._common import (
    EULER_GAMMA,
    clamped_standardize,
    expm1_over_x,
    log1p_over_x,
    safe_exp_neg,
    safe_exp_neg_density,
)


# Coefficients of gammaln(1 - ξ)/ξ = γ + Σ_{k≥2} ζ(k)/k · ξ^{k-1}, used for the
# mean's small-ξ series. Accurate to ~3e-8 for |ξ| ≤ the threshold below.
_ZETA2_2 = 0.8224670334241132  # ζ(2)/2 = π²/12
_ZETA3_3 = 0.4006856343865314  # ζ(3)/3
_ZETA4_4 = 0.2705808084277845  # ζ(4)/4 = π⁴/360
_MEAN_SERIES_THRESHOLD = 2e-2


def _reduced_exponent(
    z: Float[Array, ...], u_safe: Float[Array, ...], valid: Array
) -> Array:
    r"""Return :math:`w = \log(1 + \xi z)/\xi = z \cdot \mathrm{log1p\_over\_x}(u)`.

    ``t^{-1/ξ} = exp(-w)`` and ``(1/ξ + 1) log(1 + ξz) = w + log1p(u)``, so the
    whole GEV kernel is expressed through ``w`` with no direct division by
    ``ξ`` — which is what keeps the Gumbel limit smooth. ``u_safe`` must already
    be masked to ``> -1``.
    """
    # Out-of-support points carry a large ``|z|``; zero it so the downstream
    # ``exp(-w)`` cannot overflow into a NaN gradient in the discarded branch.
    z_safe = jnp.where(valid, z, 0.0)
    return z_safe * log1p_over_x(u_safe)


def _support(shape: Float[Array, ...], z: Float[Array, ...]) -> tuple[Array, Array]:
    r"""Return ``(valid, u_safe)`` for ``u = ξz`` and the support ``u > -1``.

    Callers pass a finite ``z`` (``±inf`` inputs are handled by the extended-real
    override in each public function), so ``u = ξz`` is always finite here.
    """
    u = shape * z
    valid = u > -1.0
    return valid, jnp.where(valid, u, 0.0)


def gev_log_prob(
    x: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Log PDF of the Generalized Extreme Value distribution."""
    shape = jnp.asarray(shape)
    x = jnp.asarray(x)
    z = clamped_standardize(x, loc, scale)
    valid, u_safe = _support(shape, z)

    w = _reduced_exponent(z, u_safe, valid)
    log_t = jnp.log1p(u_safe)
    log_pdf = -jnp.log(scale) - (w + log_t) - safe_exp_neg_density(w)

    # Density vanishes at x = ±inf; NaN input propagates.
    result = jnp.where(jnp.isfinite(x) & valid, log_pdf, -jnp.inf)
    return jnp.where(jnp.isnan(x), jnp.nan, result)


def gev_cdf(
    x: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """CDF of the Generalized Extreme Value distribution."""
    shape = jnp.asarray(shape)
    x = jnp.asarray(x)
    z = clamped_standardize(x, loc, scale)
    valid, u_safe = _support(shape, z)

    w = _reduced_exponent(z, u_safe, valid)
    cdf_inside = jnp.exp(-safe_exp_neg(w))
    # Fréchet tails (ξ > 0) map out-of-support to 0; Weibull tails to 1.
    boundary = jnp.where(shape > 0, 0.0, 1.0)
    cdf = jnp.where(valid, cdf_inside, boundary)

    # Extended-real limits F(-inf) = 0, F(+inf) = 1; NaN input propagates.
    cdf = jnp.where(jnp.isinf(x), jnp.where(x > 0, 1.0, 0.0), cdf)
    return jnp.where(jnp.isnan(x), jnp.nan, cdf)


def gev_survival(
    x: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Survival function :math:`S(x) = 1 - F(x)` of the GEV distribution.

    Uses ``-expm1(log F)`` rather than ``1 - F`` so the deep upper tail
    (where :math:`F \approx 1` and :math:`S` is tiny) stays accurate
    instead of cancelling against 1.0.
    """
    shape = jnp.asarray(shape)
    x = jnp.asarray(x)
    z = clamped_standardize(x, loc, scale)
    valid, u_safe = _support(shape, z)

    w = _reduced_exponent(z, u_safe, valid)
    s_inside = -jnp.expm1(-safe_exp_neg(w))
    # Below the Fréchet (ξ > 0) lower endpoint S = 1; above the Weibull
    # (ξ < 0) upper endpoint S = 0.
    boundary = jnp.where(shape > 0, 1.0, 0.0)
    s = jnp.where(valid, s_inside, boundary)

    # Extended-real limits S(-inf) = 1, S(+inf) = 0; NaN input propagates.
    s = jnp.where(jnp.isinf(x), jnp.where(x > 0, 0.0, 1.0), s)
    return jnp.where(jnp.isnan(x), jnp.nan, s)


def gev_log_survival(
    x: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Log survival function :math:`\log S(x) = \log(1 - F(x))` of the GEV."""
    shape = jnp.asarray(shape)
    x = jnp.asarray(x)
    z = clamped_standardize(x, loc, scale)
    valid, u_safe = _support(shape, z)

    w = _reduced_exponent(z, u_safe, valid)
    ls_inside = jnp.log(-jnp.expm1(-safe_exp_neg(w)))
    boundary = jnp.where(shape > 0, 0.0, -jnp.inf)
    ls = jnp.where(valid, ls_inside, boundary)

    # Extended-real limits log S(-inf) = 0, log S(+inf) = -inf; NaN propagates.
    ls = jnp.where(jnp.isinf(x), jnp.where(x > 0, -jnp.inf, 0.0), ls)
    return jnp.where(jnp.isnan(x), jnp.nan, ls)


def _gev_icdf_from_neg_log_q(
    neg_log_q: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    r"""GEV quantile from :math:`-\log q` (avoids forming ``1 - tiny``).

    .. math:: F^{-1}(q) = \mu + \frac{\sigma}{\xi}\big[(-\log q)^{-\xi} - 1\big]
        = \mu - \sigma L \cdot \frac{\exp(-\xi L) - 1}{-\xi L},\quad L = \log(-\log q)

    which tends to :math:`\mu - \sigma L` (the Gumbel quantile) as
    :math:`\xi \to 0`, smoothly and with a correct shape gradient.
    """
    shape = jnp.asarray(shape)
    neg_log_q = jnp.asarray(neg_log_q)
    at_upper = neg_log_q == 0.0  # q = 1
    at_lower = jnp.isposinf(neg_log_q)  # q = 0 (only +inf; -inf is invalid q)

    # Sanitize the endpoint inputs so the interior product is finite there —
    # both its value and the reverse-mode gradient of the discarded branch —
    # before the analytic support endpoints are selected below.
    safe = jnp.where(at_upper | at_lower, 1.0, neg_log_q)
    log_term = jnp.log(safe)
    a = -shape * log_term
    z_std = -log_term * expm1_over_x(a)

    shape_safe = jnp.where(shape == 0.0, 1.0, shape)
    upper = jnp.where(shape < 0.0, -1.0 / shape_safe, jnp.inf)  # q → 1
    lower = jnp.where(shape > 0.0, -1.0 / shape_safe, -jnp.inf)  # q → 0
    z_std = jnp.where(at_upper, upper, z_std)
    z_std = jnp.where(at_lower, lower, z_std)

    return loc + scale * z_std


def gev_icdf(
    q: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    """Quantile function (inverse CDF) of the GEV distribution."""
    return _gev_icdf_from_neg_log_q(-jnp.log(q), loc, scale, shape)


def gev_mean(
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Mean of the GEV distribution (``+inf`` when :math:`\xi \ge 1`).

    .. math:: \mathbb{E}[X] = \mu + \sigma\,\frac{\Gamma(1 - \xi) - 1}{\xi}
        = \mu + \sigma\,\frac{\mathrm{expm1}(\log\Gamma(1 - \xi))}{\xi}

    The ``expm1`` form avoids the catastrophic ``Γ(1 - ξ) - 1`` cancellation
    for small ``ξ`` (which otherwise makes the mean wildly wrong in float32);
    a Taylor branch covers the immediate neighbourhood of ``ξ = 0`` where the
    quotient is ``0/0``.
    """
    shape = jnp.asarray(shape)
    finite = shape < 1.0
    # Sanitize the divergent branch (ξ ≥ 1, and gammaln(1 - ξ) for ξ ≥ 1) so it
    # contributes neither a nan value nor a nan gradient before being masked.
    xi = jnp.where(finite, shape, 0.0)

    # (Γ(1 - ξ) - 1)/ξ = [expm1(g)/g] · [g/ξ] with g = gammaln(1 - ξ). The first
    # factor is the smooth expm1_over_x; the second, g/ξ = γ + Σ_{k≥2} ζ(k)/k
    # ξ^{k-1}, has a removable singularity at ξ = 0. A direct g/ξ loses float32
    # precision for small ξ (tiny gammaln ÷ tiny ξ), so use the ζ-series there.
    g = gammaln(1.0 - xi)
    small = jnp.abs(xi) < _MEAN_SERIES_THRESHOLD
    xi_exact = jnp.where(small, 1.0, xi)
    ratio_exact = g / xi_exact
    xi_series = jnp.where(small, xi, 0.0)
    ratio_series = EULER_GAMMA + xi_series * (
        _ZETA2_2 + xi_series * (_ZETA3_3 + xi_series * _ZETA4_4)
    )
    ratio = jnp.where(small, ratio_series, ratio_exact)
    coeff = expm1_over_x(g) * ratio

    mean = loc + scale * coeff
    return jnp.where(finite, mean, jnp.inf)


def gev_return_level(
    period: Float[Array, ...],
    loc: Float[Array, ...],
    scale: Float[Array, ...],
    shape: Float[Array, ...],
) -> Float[Array, ...]:
    r"""T-period return level: :math:`z_T = F^{-1}(1 - 1/T)`.

    Parameterized by :math:`-\log(1 - 1/T) = -\mathrm{log1p}(-1/T)` so the
    quantile stays accurate for large ``T`` (where ``1 - 1/T`` rounds to 1 and
    ``log(1 - 1/T)`` would lose all its precision).
    """
    neg_log_q = -jnp.log1p(-1.0 / period)
    return _gev_icdf_from_neg_log_q(neg_log_q, loc, scale, shape)
