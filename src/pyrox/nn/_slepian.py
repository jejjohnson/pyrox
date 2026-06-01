"""Bayesian Slepian positional encoder.

The deterministic :class:`SlepianEncoder` and
:class:`HybridSphericalSlepianEncoder` now live in :mod:`geonnax` and are
re-exported via :mod:`pyrox.nn._geonnax`. This module keeps the Bayesian
variant whose forward registers NumPyro ``param`` sites for the cap
geometry (radius and centre), which is pyrox-specific and depends on the
``Parameterized`` base class.
"""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float

from pyrox._basis import slepian_cap_basis
from pyrox._core import Parameterized, pyrox_method
from geonnax.geo import lonlat_to_cartesian3d


def _unit_xyz(
    x: Float[Array, "N 3"] | Float[Array, "N 2"],
    input_mode: Literal["cartesian", "lonlat"],
) -> Float[Array, "N 3"]:
    if input_mode == "cartesian":
        if x.ndim != 2 or x.shape[-1] != 3:
            raise ValueError(
                f"x must be (N, 3) when input_mode='cartesian'; got shape {x.shape}."
            )
        return x
    if x.ndim != 2 or x.shape[-1] != 2:
        raise ValueError(f"x must be (N, 2) when input_mode='lonlat'; got {x.shape}.")
    return lonlat_to_cartesian3d(x, input_unit="radians")


class BayesianSlepianEncoder(Parameterized):
    """Slepian encoder with NumPyro sites for cap radius and centre.

    Truncation is controlled by ``n_modes`` only. ``eig_threshold`` is not
    exposed because filtering by concentration ratio under tracing would
    require a data-dependent output size (``jnp.nonzero`` needs a static
    ``size``), which breaks JIT and downstream shape contracts.
    """

    l_max: int = eqx.field(static=True)
    init_cap_radius_deg: float = eqx.field(static=True)
    init_cap_centre_lonlat_deg: tuple[float, float] = eqx.field(static=True)
    n_modes: int | None = eqx.field(static=True, default=None)
    input_mode: Literal["cartesian", "lonlat"] = eqx.field(
        static=True, default="lonlat"
    )
    weight_by_eigenvalue: bool = eqx.field(static=True, default=True)

    def setup(self) -> None:
        if self.l_max < 0:
            raise ValueError(f"l_max must be >= 0; got {self.l_max}.")
        if self.input_mode not in {"cartesian", "lonlat"}:
            raise ValueError(
                f"input_mode must be 'cartesian' or 'lonlat'; got {self.input_mode!r}."
            )
        self.register_param(
            "cap_radius",
            jnp.asarray(math.radians(self.init_cap_radius_deg)),
            constraint=dist.constraints.interval(0.0, math.pi),
        )
        self.register_param(
            "cap_centre",
            jnp.radians(jnp.asarray(self.init_cap_centre_lonlat_deg)),
            constraint=dist.constraints.real_vector,
        )

    @pyrox_method
    def __call__(
        self,
        x: Float[Array, "N 3"] | Float[Array, "N 2"],
    ) -> Float[Array, "N K"]:
        radius = self.get_param("cap_radius")
        centre = self.get_param("cap_centre")
        basis = slepian_cap_basis(
            self.l_max,
            radius,
            n_modes=self.n_modes,
            eig_threshold=None,
            lonlat_centre=centre,
        )
        features = basis.evaluate(_unit_xyz(x, self.input_mode))
        if self.weight_by_eigenvalue:
            features = features * jnp.sqrt(basis.eigenvalues)[None, :]
        return features


__all__ = [
    "BayesianSlepianEncoder",
]
