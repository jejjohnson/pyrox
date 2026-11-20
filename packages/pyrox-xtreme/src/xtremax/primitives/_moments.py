r"""Cancellation-free higher moments for the GEV family.

The exact GEV moment formulas are built from differences of gamma
functions evaluated at :math:`1 - k\xi`,

.. math::
    \mu_2 = \Gamma(1-2\xi) - \Gamma^2(1-\xi), \quad
    \mu_3 = \Gamma(1-3\xi) - 3\Gamma(1-\xi)\Gamma(1-2\xi) + 2\Gamma^3(1-\xi),

and every one of them cancels catastrophically as :math:`\xi \to 0`: the
terms are all :math:`1 + O(\xi)` while :math:`\mu_j = O(\xi^j)`. Dividing by
:math:`\xi^j` then amplifies the rounding noise, so the naive form loses
*all* significant digits well before the Gumbel branch takes over — a
float32 variance of exactly ``0`` (instead of :math:`\sigma^2\pi^2/6`) at
:math:`\xi = 10^{-5}`, and a float64 excess kurtosis of ``3e4`` instead of
``2.4`` (issue #88).

The stable route writes :math:`X = \mu + \sigma (U - 1) / \xi` with
:math:`U = e^{\xi G}` and :math:`G` standard Gumbel, so that
:math:`\mathbb{E}[U^k] = \Gamma(1 - k\xi)`. Factoring :math:`\Gamma^j(1-\xi)`
out of the :math:`j`-th central moment leaves the *reduced* moments

.. math::
    r_2 = \frac{\mathrm{expm1}(d_2)}{\xi^2}, \quad
    r_3 = \frac{\mathrm{expm1}(d_3) - 3\,\mathrm{expm1}(d_2)}{\xi^3}, \quad
    r_4 = \frac{\mathrm{expm1}(d_4) - 4\,\mathrm{expm1}(d_3)
                + 6\,\mathrm{expm1}(d_2)}{\xi^4},

where :math:`d_k = \log\Gamma(1-k\xi) - k\log\Gamma(1-\xi)`. Written this way
the variance needs no branch at all beyond the removable :math:`0/0`, and the
standardized moments drop the :math:`\Gamma(1-\xi)` factor entirely:

.. math::
    \mathrm{Var}[X] = \sigma^2\,\Gamma^2(1-\xi)\,r_2, \quad
    \mathrm{skew}[X] = \frac{r_3}{r_2^{3/2}}, \quad
    \mathrm{kurt}[X] = \frac{r_4}{r_2^2} - 3 .

The sign of :math:`\xi` needs no special handling: :math:`r_3` is odd in the
same way :math:`r_2^{3/2}` is even, so the ratio carries the sign of the
skew automatically.

Two orders of cancellation survive this rewrite — the :math:`\xi^2` and
:math:`\xi^3` terms of the :math:`r_4` numerator vanish identically, and the
:math:`\xi^2` term of the :math:`r_3` one does — so close to zero each
:math:`r_j` is evaluated from its Taylor series instead. The series are
:math:`r_2 = \zeta(2) + \dots`, :math:`r_3 = 2\zeta(3) + \dots`,
:math:`r_4 = 3\zeta(2)^2 + 6\zeta(4) + \dots`, reproducing the Gumbel limits
:math:`\sigma^2\pi^2/6`, ``1.13955`` and ``12/5`` exactly at :math:`\xi = 0`.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import gammaln
from jaxtyping import Array, ArrayLike, Float


# Taylor coefficients of r_j about ξ = 0, generated from the formal power
# series of expm1(d_k) with d_k = Σ_{n≥2} ζ(n)(k^n − k) ξ^n / n. The n-th
# coefficient grows like k^n, so the radius of convergence is 1/k — exactly
# the pole of Γ(1−kξ) that bounds the moment's existence anyway. Term counts
# are sized so the truncation error is below float32 eps at the crossover
# (see _series_threshold).
_R2_COEFFS = (
    1.6449340668482264,  # ζ(2) = π²/6
    2.4041138063191885,
    5.141035360127907,
    10.176175231454812,
    20.375465474724987,
    40.743987816707964,
    81.48792393561038,
    162.9749962158658,
    325.94955505023,
    651.8988001260199,
    1303.797396075081,
    2607.5946557735665,
    5215.189220671365,
    10430.378380759998,
)
_R3_COEFFS = (
    2.4041138063191885,  # 2ζ(3)
    17.85833335623378,
    66.69931095965242,
    232.3896571292965,
    758.0400849266214,
    2397.31434736477,
    7437.301074340566,
    22802.36457149247,
    69387.18212164848,
    210120.620475257,
    634278.3146137745,
    1910665.3202771668,
    5747652.916696094,
    17274266.96746622,
    51885408.79581522,
    155781429.36251712,
    467594674.82019025,
    1403284769.0996883,
    4210855753.338122,
    12634570087.233753,
    37907715818.851974,
    113731158424.92528,
)
_R4_COEFFS = (
    14.611363655100366,  # 3ζ(2)² + 6ζ(4)
    120.95548885758441,
    764.208674359849,
    3980.853143955183,
    18973.252883647772,
    85499.57900926215,
    371795.8864453327,
    1578486.464255545,
    6591689.094513316,
    27207633.310941048,
    111368446.13826104,
    453118060.1511877,
    1835466103.85382,
    7410968020.717287,
    29851426844.34226,
    120028859424.4942,
    481985869172.21704,
    1933556721488.5476,
    7751070520674.08,
    31054820786848.203,
    124370914858858.8,
    497938585766392.94,
    1993119184449373.5,
    7976571386727469.0,
    3.1918569743252016e16,
    1.277111320611315e17,
    5.1095508850707936e17,
    2.0441520368123267e18,
)

# (k, weight) pairs of the expm1(d_k) combination that forms ξ^j · r_j.
_R2_TERMS = ((2, 1.0),)
_R3_TERMS = ((2, -3.0), (3, 1.0))
_R4_TERMS = ((2, 6.0), (3, -4.0), (4, 1.0))


def _series_threshold(dtype) -> float:
    r"""|ξ| below which the Taylor branch beats the gamma-difference form.

    Two error curves cross here. The gamma-difference form loses the
    cancelled orders, so its relative error *grows* as ξ shrinks —
    empirically ``~40 eps/ξ`` for :math:`r_2` and ``~0.8 eps/\xi^3`` for
    :math:`r_4`. The truncated series does the opposite, its error falling
    like :math:`(k\xi)^N` with the coefficient tables above.

    Balancing the two puts the crossover at ``0.15`` in float32 — pushed
    right to the edge of what :math:`r_4`'s radius-``1/4`` series can still
    resolve — and at ``0.04`` in float64, where a shorter series already
    exhausts the dtype. Worst-case relative error at the crossover is
    ~1e-3 (excess kurtosis; ~5e-4 skew, ~5e-5 variance) in float32 and
    ~1e-10 in float64, against a *total* loss of significance in the naive
    form anywhere below ξ ≈ 1e-2.
    """
    return 0.15 if float(jnp.finfo(dtype).eps) > 1e-10 else 0.04


def _as_float(x: Float[ArrayLike, ...]) -> Array:
    """Array view of ``x`` in a floating dtype.

    Integral parameters are legal (``concentration=0``, ``scale=50000``)
    but must not stay integral: ``scale ** 2`` wraps at int32 and hands
    back a *negative* variance, and :func:`_series_threshold` reads the
    floating epsilon of whatever dtype it is given.
    """
    return jnp.asarray(x, dtype=jnp.result_type(x, float))


def _horner(coeffs: tuple[float, ...], x: Array) -> Array:
    acc = jnp.asarray(coeffs[-1], dtype=x.dtype)
    for c in reversed(coeffs[:-1]):
        acc = c + x * acc
    return acc


def _reduced_moment(
    xi: Array,
    terms: tuple[tuple[int, float], ...],
    order: int,
    coeffs: tuple[float, ...],
) -> Array:
    r"""``Σ_k w_k · expm1(d_k) / ξ**order``, series-stabilised near ξ = 0.

    ``xi`` must already be sanitized against the ``Γ(1 − kξ)`` poles: the
    caller masks the non-existent branch, but both ``jnp.where`` arms are
    still evaluated, so a poled input would poison the gradient.
    """
    # The coefficient tables run to ~1e18, so anything narrower than
    # float32 cannot hold them: the leading coefficient would round to inf
    # and Horner would return 0·inf = nan at the Gumbel point.
    xi = xi.astype(jnp.promote_types(xi.dtype, jnp.float32))
    small = jnp.abs(xi) < _series_threshold(xi.dtype)
    # Double-``where`` so neither branch feeds a singular value to autodiff:
    # the exact arm never sees ξ = 0, the series arm never sees a large ξ.
    # The inactive-arm sentinel is -1 rather than the more usual 1: at ξ = 1
    # every ``Γ(1 - kξ)`` here sits on a pole, and while the inner ``where``
    # does select the resulting nan cotangent away, there is no reason to
    # lean on that when a pole-free constant costs nothing.
    xi_exact = jnp.where(small, -1.0, xi)
    numerator = sum(
        w * jnp.expm1(gammaln(1.0 - k * xi_exact) - k * gammaln(1.0 - xi_exact))
        for k, w in terms
    )
    exact = numerator / xi_exact**order
    series = _horner(coeffs, jnp.where(small, xi, 0.0))
    return jnp.where(small, series, exact)


def gev_variance(
    scale: Float[ArrayLike, ...],
    shape: Float[ArrayLike, ...],
) -> Float[Array, ...]:
    r"""Variance of the GEV distribution (``+inf`` when :math:`\xi \ge 1/2`).

    .. math:: \mathrm{Var}[X] = \frac{\sigma^2}{\xi^2}
        \bigl(\Gamma(1-2\xi) - \Gamma^2(1-\xi)\bigr)

    evaluated through the reduced form :math:`\sigma^2\Gamma^2(1-\xi) r_2`,
    which is smooth and cancellation-free through the Gumbel limit
    :math:`\sigma^2\pi^2/6`.
    """
    xi = _as_float(shape)
    exists = xi < 0.5
    xi_safe = jnp.where(exists, xi, 0.0)
    r2 = _reduced_moment(xi_safe, _R2_TERMS, 2, _R2_COEFFS)
    var = _as_float(scale) ** 2 * jnp.exp(2.0 * gammaln(1.0 - xi_safe)) * r2
    return jnp.where(exists, var, jnp.inf)


def gev_skewness(shape: Float[ArrayLike, ...]) -> Float[Array, ...]:
    r"""Skewness of the GEV distribution (``nan`` when :math:`\xi \ge 1/3`).

    Returns :math:`r_3 / r_2^{3/2}`, which reduces to the Gumbel value
    :math:`12\sqrt{6}\,\zeta(3)/\pi^3 \approx 1.13955` at :math:`\xi = 0` and
    carries the sign of :math:`\xi` without an explicit branch.
    """
    xi = _as_float(shape)
    exists = xi < 1.0 / 3.0
    xi_safe = jnp.where(exists, xi, 0.0)
    r2 = _reduced_moment(xi_safe, _R2_TERMS, 2, _R2_COEFFS)
    r3 = _reduced_moment(xi_safe, _R3_TERMS, 3, _R3_COEFFS)
    # NaN, not inf: beyond ξ = 1/3 the standardized third moment is
    # undefined, unlike the variance which genuinely diverges to +inf.
    return jnp.where(exists, r3 / r2**1.5, jnp.nan)


def gev_excess_kurtosis(shape: Float[ArrayLike, ...]) -> Float[Array, ...]:
    r"""Excess kurtosis of the GEV (``nan`` when :math:`\xi \ge 1/4`).

    Returns :math:`r_4 / r_2^2 - 3`, which reduces to the Gumbel value
    :math:`12/5` at :math:`\xi = 0`.
    """
    xi = _as_float(shape)
    exists = xi < 0.25
    xi_safe = jnp.where(exists, xi, 0.0)
    r2 = _reduced_moment(xi_safe, _R2_TERMS, 2, _R2_COEFFS)
    r4 = _reduced_moment(xi_safe, _R4_TERMS, 4, _R4_COEFFS)
    # NaN for the same reason as the skew.
    return jnp.where(exists, r4 / r2**2 - 3.0, jnp.nan)
