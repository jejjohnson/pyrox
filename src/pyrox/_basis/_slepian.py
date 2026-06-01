r"""Spherical Slepian functions for axisymmetric caps on :math:`S^2`.

Slepian cap functions are band-limited linear combinations of real spherical
harmonics that maximize energy concentration inside a spherical cap.  This
module builds the cap concentration operator in the same flattened real-SH
layout as :func:`pyrox._basis.real_spherical_harmonics`, solves the
block-diagonal eigenproblem by azimuthal order, and evaluates retained modes by
matrix multiplication against the existing SH basis.
"""

from __future__ import annotations

import math

import einx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from pyrox._basis._spherical import real_spherical_harmonics


def _check_l_max(l_max: int) -> None:
    if l_max < 0:
        raise ValueError(f"l_max must be >= 0, got {l_max}.")


def _check_cap_radius(cap_radius: float | Float[Array, ""]) -> None:
    if isinstance(cap_radius, (int, float)):
        radius = float(cap_radius)
        if not 0.0 < radius <= math.pi:
            raise ValueError(f"cap_radius must lie in (0, pi], got {radius}.")


def _sh_index(ell: int, m: int) -> int:
    return ell * ell + (m + ell)


def _normalization(ell: int, m_abs: int) -> float:
    log_ratio = math.lgamma(ell - m_abs + 1) - math.lgamma(ell + m_abs + 1)
    return math.sqrt((2.0 * ell + 1.0) * math.exp(log_ratio) / (4.0 * math.pi))


def _associated_legendre_values(
    z: Float[Array, " Q"], l_max: int, m_abs: int
) -> list[Float[Array, " Q"]]:
    r"""Evaluate :math:`P_l^m(z)` for ``l=m,...,l_max`` by recurrence."""
    one_minus_z2 = jnp.maximum(1.0 - z**2, 0.0)
    p_mm = jnp.ones_like(z)
    if m_abs > 0:
        double_factorial = 1.0
        for k in range(1, 2 * m_abs, 2):
            double_factorial *= float(k)
        p_mm = ((-1.0) ** m_abs) * double_factorial * one_minus_z2 ** (0.5 * m_abs)

    values = [p_mm]
    if m_abs == l_max:
        return values

    p_m1m = (2.0 * m_abs + 1.0) * z * p_mm
    values.append(p_m1m)
    p_lm2 = p_mm
    p_lm1 = p_m1m
    for ell in range(m_abs + 2, l_max + 1):
        p_l = ((2.0 * ell - 1.0) * z * p_lm1 - (ell + m_abs - 1.0) * p_lm2) / (
            ell - m_abs
        )
        values.append(p_l)
        p_lm2, p_lm1 = p_lm1, p_l
    return values


def _cap_quadrature(
    cap_radius: float | Float[Array, ""], num_quadrature: int
) -> tuple[Float[Array, " Q"], Float[Array, " Q"]]:
    # Gauss-Legendre nodes/weights come from NumPy on the host: this is a
    # one-shot setup constant for the cap eigenproblem and is identical for
    # every call with the same ``num_quadrature``. Everything downstream
    # (cap_radius scaling, Legendre recurrence, eigensolve) stays in JAX.
    import numpy as np

    nodes, weights = np.polynomial.legendre.leggauss(num_quadrature)
    nodes = jnp.asarray(nodes)
    weights = jnp.asarray(weights)
    z_min = jnp.cos(jnp.asarray(cap_radius))
    z = 0.5 * (1.0 - z_min) * nodes + 0.5 * (1.0 + z_min)
    w = 0.5 * (1.0 - z_min) * weights
    return z, w


def _slepian_block(
    l_max: int,
    cap_radius: float | Float[Array, ""],
    m: int,
    num_quadrature: int,
) -> Float[Array, "B B"]:
    m_abs = abs(m)
    z, w = _cap_quadrature(cap_radius, num_quadrature)
    p_values = _associated_legendre_values(z, l_max, m_abs)
    rows = []
    for ell, p_lm in zip(range(m_abs, l_max + 1), p_values, strict=True):
        factor = _normalization(ell, m_abs)
        if m_abs > 0:
            factor *= math.sqrt(2.0)
        rows.append(factor * p_lm)
    basis = jnp.stack(rows, axis=0)  # (B, Q) Legendre values per node
    weighted = einx.multiply("b q, q -> b q", basis, w)
    phi_factor = 2.0 * jnp.pi if m_abs == 0 else jnp.pi
    # Quadrature inner products ∫ Pᵦ Pᵧ dz: contract the node axis q → (B, B).
    return phi_factor * einx.dot("b q, c q -> b c", weighted, basis)


def shannon_number(l_max: int, area: float | Float[Array, ""]) -> Float[Array, ""]:
    """Return the spherical-cap Shannon number for a bandlimit and area.

    Args:
        l_max: Maximum spherical-harmonic degree.
        area: Region area on the unit sphere.

    Returns:
        Expected number of well-concentrated Slepian modes.
    """
    _check_l_max(l_max)
    return ((l_max + 1) ** 2) * jnp.asarray(area) / (4.0 * jnp.pi)


def slepian_concentration_matrix(
    l_max: int,
    cap_radius: float | Float[Array, ""],
    *,
    num_quadrature: int | None = None,
) -> Float[Array, "M M"]:
    r"""Build the real-SH concentration matrix for a north-pole cap.

    Args:
        l_max: Maximum spherical-harmonic degree.
        cap_radius: Cap half-angle in radians.
        num_quadrature: Optional Gauss-Legendre quadrature order over
            ``cos(theta)``. Defaults to a conservative polynomial-exact order
            for products up to degree ``2 * l_max``.

    Returns:
        ``((l_max + 1)^2, (l_max + 1)^2)`` concentration matrix.
    """
    _check_l_max(l_max)
    _check_cap_radius(cap_radius)
    n_quad = num_quadrature or max(2 * l_max + 3, 8)
    if n_quad < 1:
        raise ValueError(f"num_quadrature must be >= 1, got {n_quad}.")

    n_harmonics = (l_max + 1) ** 2
    matrix = jnp.zeros((n_harmonics, n_harmonics))
    for m in range(-l_max, l_max + 1):
        block = _slepian_block(l_max, cap_radius, m, n_quad)
        ells = range(abs(m), l_max + 1)
        indices = jnp.asarray([_sh_index(ell, m) for ell in ells])
        matrix = matrix.at[jnp.ix_(indices, indices)].set(block)
    return matrix


def slepian_cap_eigh_per_m(
    l_max: int,
    cap_radius: float | Float[Array, ""],
    *,
    num_quadrature: int | None = None,
) -> tuple[Float[Array, " M"], Float[Array, "M M"]]:
    """Solve the cap Slepian eigenproblem one azimuthal block at a time."""
    _check_l_max(l_max)
    _check_cap_radius(cap_radius)
    n_quad = num_quadrature or max(2 * l_max + 3, 8)
    n_harmonics = (l_max + 1) ** 2
    eigenvalue_blocks = []
    coeff_blocks = []
    for m in range(-l_max, l_max + 1):
        block = _slepian_block(l_max, cap_radius, m, n_quad)
        vals, vecs = jnp.linalg.eigh(block)
        order = jnp.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        indices = jnp.asarray([_sh_index(ell, m) for ell in range(abs(m), l_max + 1)])
        coeff_block = jnp.zeros((n_harmonics, vecs.shape[1]))
        coeff_block = coeff_block.at[indices, :].set(vecs)
        eigenvalue_blocks.append(vals)
        coeff_blocks.append(coeff_block)

    vals = jnp.concatenate(eigenvalue_blocks, axis=0)
    coeffs = jnp.concatenate(coeff_blocks, axis=1)
    order = jnp.argsort(vals)[::-1]
    return jnp.clip(vals[order], 0.0, 1.0), coeffs[:, order]


def _lonlat_to_unit(lonlat: Float[Array, " 2"]) -> Float[Array, " 3"]:
    lon, lat = lonlat[0], lonlat[1]
    cos_lat = jnp.cos(lat)
    return jnp.asarray([cos_lat * jnp.cos(lon), cos_lat * jnp.sin(lon), jnp.sin(lat)])


def _centered_coordinates(
    unit_xyz: Float[Array, "N 3"], lonlat_centre: Float[Array, " 2"]
) -> Float[Array, "N 3"]:
    lon, lat = lonlat_centre[0], lonlat_centre[1]
    centre = _lonlat_to_unit(lonlat_centre)
    east = jnp.asarray([-jnp.sin(lon), jnp.cos(lon), 0.0])
    north = jnp.asarray(
        [-jnp.sin(lat) * jnp.cos(lon), -jnp.sin(lat) * jnp.sin(lon), jnp.cos(lat)]
    )
    # Project each point onto the local (east, north, centre) frame.
    return jnp.stack(
        [
            einx.dot("n d, d -> n", unit_xyz, east),
            einx.dot("n d, d -> n", unit_xyz, north),
            einx.dot("n d, d -> n", unit_xyz, centre),
        ],
        axis=-1,
    )


class SlepianCapBasis(eqx.Module):
    """Precomputed Slepian cap eigendecomposition in the real-SH basis."""

    l_max: int = eqx.field(static=True)
    coeffs: Float[Array, "M K"]
    eigenvalues: Float[Array, " K"]
    cap_radius: Float[Array, ""]
    lonlat_centre: Float[Array, " 2"]

    @property
    def num_modes(self) -> int:
        """Number of retained Slepian modes."""
        return int(self.eigenvalues.shape[0])

    def centred_coordinates(self, unit_xyz: Float[Array, "N 3"]) -> Float[Array, "N 3"]:
        """Rotate inputs into the frame where the cap centre is the north pole.

        Downstream consumers that need to evaluate the same real spherical
        harmonics in the cap frame (e.g.
        :class:`pyrox.gp.SlepianInducingFeatures`) should call this and feed
        the result into :func:`pyrox._basis.real_spherical_harmonics` to keep
        their evaluation consistent with :meth:`evaluate`.
        """
        if unit_xyz.ndim != 2 or unit_xyz.shape[-1] != 3:
            raise ValueError(f"unit_xyz must be (N, 3); got shape {unit_xyz.shape}.")
        return _centered_coordinates(unit_xyz, self.lonlat_centre)

    def evaluate(self, unit_xyz: Float[Array, "N 3"]) -> Float[Array, "N K"]:
        """Evaluate retained Slepian functions at unit Cartesian locations."""
        centered = self.centred_coordinates(unit_xyz)
        harmonics = real_spherical_harmonics(centered, self.l_max)
        # Mix harmonics into Slepian modes: contract the harmonic axis m.
        return einx.dot("n m, m k -> n k", harmonics, self.coeffs)

    def rotate_to(self, lonlat_centre: Float[Array, " 2"]) -> SlepianCapBasis:
        """Return an equivalent cap basis evaluated around ``lonlat_centre``."""
        centre = jnp.asarray(lonlat_centre)
        if centre.shape != (2,):
            raise ValueError(f"lonlat_centre must have shape (2,), got {centre.shape}.")
        return eqx.tree_at(lambda basis: basis.lonlat_centre, self, centre)


def slepian_cap_basis(
    l_max: int,
    cap_radius: float | Float[Array, ""],
    *,
    n_modes: int | None = None,
    eig_threshold: float | None = None,
    lonlat_centre: Float[Array, " 2"] | None = None,
    num_quadrature: int | None = None,
) -> SlepianCapBasis:
    """Construct a Slepian basis for an axisymmetric spherical cap.

    Args:
        l_max: Maximum spherical-harmonic degree.
        cap_radius: Cap half-angle in radians.
        n_modes: Optional maximum number of retained modes. Takes precedence
            over ``eig_threshold``: if both are given, the basis is first
            trimmed by threshold and then truncated to at most ``n_modes``
            (after sorting by concentration). A :class:`ValueError` is
            raised when ``eig_threshold`` would leave fewer than ``n_modes``
            modes — silent shape contraction would break JIT / static
            shape expectations in downstream code.
        eig_threshold: Optional concentration-ratio threshold.
        lonlat_centre: Optional cap centre in radians. Defaults to the north pole.
        num_quadrature: Optional Gauss-Legendre quadrature order.

    Returns:
        A :class:`SlepianCapBasis` with coefficient columns sorted by decreasing
        concentration ratio.
    """
    if n_modes is not None and n_modes < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}.")
    vals, coeffs = slepian_cap_eigh_per_m(
        l_max, cap_radius, num_quadrature=num_quadrature
    )
    if eig_threshold is not None:
        keep = jnp.nonzero(vals > eig_threshold, size=vals.shape[0], fill_value=-1)[0]
        keep = keep[keep >= 0]
        if n_modes is not None and int(keep.shape[0]) < n_modes:
            raise ValueError(
                f"eig_threshold={eig_threshold} retains only {int(keep.shape[0])} "
                f"modes but n_modes={n_modes} was requested; lower the threshold "
                f"or reduce n_modes to break the conflict."
            )
        vals = vals[keep]
        coeffs = coeffs[:, keep]
    if n_modes is not None:
        vals = vals[:n_modes]
        coeffs = coeffs[:, :n_modes]

    centre = (
        jnp.asarray([0.0, 0.5 * jnp.pi])
        if lonlat_centre is None
        else jnp.asarray(lonlat_centre)
    )
    if centre.shape != (2,):
        raise ValueError(f"lonlat_centre must have shape (2,), got {centre.shape}.")
    return SlepianCapBasis(
        l_max=l_max,
        coeffs=coeffs,
        eigenvalues=vals,
        cap_radius=jnp.asarray(cap_radius),
        lonlat_centre=centre,
    )


__all__ = [
    "SlepianCapBasis",
    "shannon_number",
    "slepian_cap_basis",
    "slepian_cap_eigh_per_m",
    "slepian_concentration_matrix",
]
