r"""Real spherical harmonics on the unit 2-sphere.

For ``l_max``, the basis is :math:`(l_{\max} + 1)^2` real SHs indexed by
:math:`(l, m)` with :math:`l = 0, 1, \ldots, l_{\max}` and
:math:`m = -l, \ldots, l`. Inputs are unit Cartesian directions; the
azimuthal singularity at the poles is avoided by computing
:math:`Q_l^m(z) = P_l^m(z) / (1 - z^2)^{m/2}` (a polynomial in ``z``)
and pairing it with :math:`(x + iy)^m` directly — no
``arctan2`` or division by :math:`\sin\theta`.

Tested via the addition theorem
:math:`\sum_m Y_{lm}(x) Y_{lm}(x') = \frac{2l+1}{4\pi} P_l(x \cdot x')`.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jaxtyping import Array, Float


def harmonic_degrees(l_max: int) -> tuple[int, ...]:
    """Return the per-harmonic degree ``l`` for the flattened ``(l_max + 1)^2`` block.

    Layout matches :func:`real_spherical_harmonics`: outer loop over ``l``,
    inner loop over ``m = -l, ..., l``.
    """
    if l_max < 0:
        raise ValueError(f"l_max must be >= 0, got {l_max}.")
    return tuple(l for l in range(l_max + 1) for _ in range(2 * l + 1))


def _q_lm_table(
    z: Float[Array, " N"], l_max: int
) -> dict[tuple[int, int], Float[Array, " N"]]:
    r"""Build the table :math:`Q_l^m(z) = P_l^m(z) / (1 - z^2)^{m/2}`.

    Computes one entry per pair ``0 <= m <= l <= l_max`` via the standard
    three-term forward recursion in ``l`` for fixed ``m``, seeded from
    :math:`Q_m^m = (-1)^m (2m - 1)!!`.
    """
    Q: dict[tuple[int, int], Float[Array, " N"]] = {}
    one = jnp.ones_like(z)
    # Seed: Q_m^m = (-1)^m (2m-1)!!, then Q_{m+1}^m = (2m+1) z Q_m^m.
    for m in range(0, l_max + 1):
        if m == 0:
            Q[(0, 0)] = one
        else:
            # double factorial (2m-1)!!
            dfac = 1.0
            for k in range(1, 2 * m, 2):
                dfac *= float(k)
            sign = (-1.0) ** m
            Q[(m, m)] = jnp.full_like(z, sign * dfac)
        if m + 1 <= l_max:
            Q[(m + 1, m)] = (2.0 * m + 1.0) * z * Q[(m, m)]
        for l in range(m + 2, l_max + 1):
            num = (2.0 * l - 1.0) * z * Q[(l - 1, m)] - (l + m - 1.0) * Q[(l - 2, m)]
            Q[(l, m)] = num / (l - m)
    return Q


def _xy_powers(
    x: Float[Array, " N"], y: Float[Array, " N"], m_max: int
) -> tuple[list[Float[Array, " N"]], list[Float[Array, " N"]]]:
    """Real and imaginary parts of :math:`(x + iy)^m` for ``m = 0, 1, ..., m_max``."""
    re = [jnp.ones_like(x)]
    im = [jnp.zeros_like(x)]
    for _m in range(1, m_max + 1):
        # (x + iy) * (re_prev + i im_prev) = (x re - y im) + i (x im + y re)
        re_prev = re[-1]
        im_prev = im[-1]
        re.append(x * re_prev - y * im_prev)
        im.append(x * im_prev + y * re_prev)
    return re, im


def real_spherical_harmonics(
    unit_xyz: Float[Array, "N 3"],
    l_max: int,
) -> Float[Array, "N M"]:
    r"""Evaluate real spherical harmonics up to degree ``l_max``.

    Convention (matches the standard "tesseral" real basis):

    .. math::

        Y_l^0    &= N_l^0\, P_l(\cos\theta)                          \\
        Y_l^m    &= \sqrt{2}\, N_l^m\, P_l^m(\cos\theta) \cos(m\phi)  & m > 0 \\
        Y_l^{-m} &= \sqrt{2}\, N_l^m\, P_l^m(\cos\theta) \sin(m\phi)  & m > 0

    with :math:`N_l^m = \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}`. The
    pole singularity is avoided by evaluating
    :math:`P_l^m(z) \cos(m\phi) = Q_l^m(z) \cdot \mathrm{Re}[(x + iy)^m]`
    and similarly for the sine branch.

    Args:
        unit_xyz: Inputs of shape ``(N, 3)``; assumed to lie on the unit
            sphere (no normalization performed).
        l_max: Maximum degree, inclusive. Total feature count ``M = (l_max + 1)^2``.

    Returns:
        ``(N, (l_max + 1)^2)`` array with the SH index flattened as
        outer loop over ``l``, inner loop over ``m = -l, ..., l``.
    """
    if l_max < 0:
        raise ValueError(f"l_max must be >= 0, got {l_max}.")
    if unit_xyz.ndim != 2 or unit_xyz.shape[-1] != 3:
        raise ValueError(f"unit_xyz must be (N, 3); got shape {unit_xyz.shape}.")

    x = unit_xyz[:, 0]
    y = unit_xyz[:, 1]
    z = unit_xyz[:, 2]

    Q = _q_lm_table(z, l_max)
    re_pow, im_pow = _xy_powers(x, y, l_max)

    columns: list[Float[Array, " N"]] = []
    inv_sqrt_4pi = 1.0 / math.sqrt(4.0 * math.pi)
    for l in range(l_max + 1):
        # m = 0
        N_l0 = math.sqrt(2.0 * l + 1.0) * inv_sqrt_4pi
        # The first 2l+1 columns for this l: m = -l, ..., -1, 0, 1, ..., l
        for m in range(-l, l + 1):
            am = abs(m)
            if am == 0:
                col = N_l0 * Q[(l, 0)]
            else:
                # N_l^m with the sqrt(2) tesseral factor folded in.
                ratio = math.factorial(l - am) / math.factorial(l + am)
                N_lm = math.sqrt(2.0 * (2.0 * l + 1.0) * ratio) * inv_sqrt_4pi
                trig = re_pow[am] if m > 0 else im_pow[am]
                col = N_lm * Q[(l, am)] * trig
            columns.append(col)

    return jnp.stack(columns, axis=-1)
