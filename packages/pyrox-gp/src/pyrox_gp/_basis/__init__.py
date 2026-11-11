"""Random-Fourier-feature draws for pyrox-gp, plus deprecated re-exports.

- `draw_rff_cosine_basis` / `evaluate_rff_cosine_paths` — pure
  random-Fourier-feature prior draws for RBF/Matern kernels, shared by
  `pyrox_gp._pathwise`.

Deprecated: `spectral_density` (use kernellib's ``unit_spectral_density`` on
``pyrox._core.freeze(kernel)``) and the basis names this module used to
re-export (import them from their `functax.basis` family module).
"""

import importlib
import warnings
from typing import Any

from pyrox_gp._basis._rff import draw_rff_cosine_basis, evaluate_rff_cosine_paths
from pyrox_gp._basis._spectral_density import spectral_density


# The basis names this module used to re-export, each with its functax family
# module (`functax.basis.<family>`), where it now lives unchanged.
_FUNCTAX_BASIS_FAMILIES = {
    "SlepianCapBasis": "slepian",
    "divfree_basis": "vector",
    "eof_basis": "eigen",
    "fourier_basis": "fourier",
    "fourier_basis_1d": "fourier",
    "fourier_eigenvalues": "fourier",
    "fourier_eigenvalues_1d": "fourier",
    "gabor_frame": "local",
    "gabor_frame_grid": "local",
    "gaussian_window_features": "features",
    "graph_laplacian_eigpairs": "eigen",
    "harmonic_degrees": "sphere",
    "rbf_basis": "local",
    "real_spherical_harmonics": "sphere",
    "shannon_number": "slepian",
    "slepian_cap_basis": "slepian",
    "slepian_cap_eigh_per_m": "slepian",
    "slepian_concentration_matrix": "slepian",
    "spherical_rbf_basis": "local",
    "wavelet_basis_1d": "local",
    "wavelet_basis_2d": "local",
    "wendland_c2": "local",
    "wendland_c4": "local",
}


def __getattr__(name: str) -> Any:
    """Serve the old basis re-exports from functax, with a `DeprecationWarning`."""
    family = _FUNCTAX_BASIS_FAMILIES.get(name)
    if family is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"pyrox_gp._basis.{name} is deprecated; import it from functax.basis.{family}.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(f"functax.basis.{family}"), name)


__all__ = [
    "draw_rff_cosine_basis",
    "evaluate_rff_cosine_paths",
    "spectral_density",
]
