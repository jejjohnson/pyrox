"""Pure-JAX geographic and spherical feature helpers.

These helpers are deterministic, pandas-free building blocks for
longitude/latitude preprocessing and spherical-harmonic feature maps.
The corresponding stateful wrappers in :mod:`pyrox.nn._layers` expose
the same transforms as :class:`pyrox._core.pyrox_module.PyroxModule`
instances.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float

from pyrox._basis import real_spherical_harmonics


def _validate_lonlat_shape(lonlat: Float[Array, ...], *, name: str = "lonlat") -> None:
    if lonlat.ndim != 2 or lonlat.shape[-1] != 2:
        raise ValueError(f"{name} must be (N, 2); got shape {lonlat.shape}.")


def _validate_range(bounds: tuple[float, float], *, name: str) -> None:
    lower, upper = bounds
    if upper <= lower:
        raise ValueError(f"{name} must satisfy min < max; got {bounds}.")


def _validate_input_unit(input_unit: Literal["degrees", "radians"] | str) -> None:
    if input_unit not in {"degrees", "radians"}:
        raise ValueError(
            f"input_unit must be 'degrees' or 'radians'; got {input_unit!r}."
        )


def deg2rad(x: Float[Array, ...]) -> Float[Array, ...]:
    """Convert degrees to radians element-wise."""
    return x * (jnp.pi / 180.0)


def lonlat_scale(
    lonlat: Float[Array, "N 2"],
    *,
    lon_range: tuple[float, float] = (-180.0, 180.0),
    lat_range: tuple[float, float] = (-90.0, 90.0),
) -> Float[Array, "N 2"]:
    """Affine-rescale lon/lat columns into ``[-1, 1]``.

    Args:
        lonlat: Longitude/latitude matrix of shape ``(N, 2)``.
        lon_range: Closed longitude domain in degrees.
        lat_range: Closed latitude domain in degrees.

    Returns:
        Rescaled lon/lat array of shape ``(N, 2)``.
    """
    _validate_lonlat_shape(lonlat)
    _validate_range(lon_range, name="lon_range")
    _validate_range(lat_range, name="lat_range")

    lower = jnp.asarray([lon_range[0], lat_range[0]], dtype=lonlat.dtype)
    upper = jnp.asarray([lon_range[1], lat_range[1]], dtype=lonlat.dtype)
    return 2.0 * (lonlat - lower) / (upper - lower) - 1.0


def lonlat_to_cartesian3d(
    lonlat: Float[Array, "N 2"],
    *,
    input_unit: Literal["degrees", "radians"] = "radians",
) -> Float[Array, "N 3"]:
    r"""Lift lon/lat coordinates onto the unit sphere.

    Uses the standard parameterization

    .. math::

        x = \cos(\phi)\cos(\lambda), \quad
        y = \cos(\phi)\sin(\lambda), \quad
        z = \sin(\phi),

    where ``lon = λ`` and ``lat = ϕ``.

    Args:
        lonlat: Longitude/latitude matrix of shape ``(N, 2)``.
        input_unit: Whether ``lonlat`` is in degrees or radians.

    Returns:
        Unit Cartesian coordinates of shape ``(N, 3)``.
    """
    _validate_lonlat_shape(lonlat)
    _validate_input_unit(input_unit)

    angles = deg2rad(lonlat) if input_unit == "degrees" else lonlat
    lon = angles[:, 0]
    lat = angles[:, 1]
    cos_lat = jnp.cos(lat)
    return jnp.stack(
        [
            cos_lat * jnp.cos(lon),
            cos_lat * jnp.sin(lon),
            jnp.sin(lat),
        ],
        axis=-1,
    )


def cyclic_encode(
    angles: Float[Array, " N"] | Float[Array, "N D"],
) -> Float[Array, "N F"]:
    """Encode periodic inputs as concatenated cos/sin features.

    Args:
        angles: Angle vector ``(N,)`` or matrix ``(N, D)`` in radians.

    Returns:
        ``(N, 2)`` for vector input or ``(N, 2 * D)`` for matrix input,
        laid out as ``[cos_0, ..., cos_{D-1}, sin_0, ..., sin_{D-1}]``.
    """
    if angles.ndim == 1:
        promoted = angles[:, None]
    elif angles.ndim == 2:
        promoted = angles
    else:
        raise ValueError(f"angles must be (N,) or (N, D); got shape {angles.shape}.")
    return jnp.concatenate([jnp.cos(promoted), jnp.sin(promoted)], axis=-1)


def spherical_harmonic_encode(
    lonlat: Float[Array, "N 2"],
    l_max: int,
    *,
    input_unit: Literal["degrees", "radians"] = "radians",
) -> Float[Array, "N M"]:
    """Lift lon/lat to :math:`S^2` and evaluate real spherical harmonics.

    Args:
        lonlat: Longitude/latitude matrix of shape ``(N, 2)``.
        l_max: Maximum harmonic degree.
        input_unit: Whether ``lonlat`` is in degrees or radians.

    Returns:
        Real spherical-harmonic features of shape ``(N, (l_max + 1)^2)``.
    """
    unit_xyz = lonlat_to_cartesian3d(lonlat, input_unit=input_unit)
    return real_spherical_harmonics(unit_xyz, l_max=l_max)


__all__ = [
    "cyclic_encode",
    "deg2rad",
    "lonlat_scale",
    "lonlat_to_cartesian3d",
    "spherical_harmonic_encode",
]
