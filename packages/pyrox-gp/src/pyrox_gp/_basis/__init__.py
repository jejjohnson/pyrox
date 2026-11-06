"""Random-Fourier-feature draws for pyrox-gp, plus deprecated re-exports.

- `draw_rff_cosine_basis` / `evaluate_rff_cosine_paths` — pure
  random-Fourier-feature prior draws for RBF/Matern kernels, shared by
  `pyrox_gp._pathwise`.

Deprecated: `spectral_density` (use kernellib's ``unit_spectral_density`` on
``pyrox._core.freeze(kernel)``) and the `geonnax.basis` names this module used
to re-export (import them from `geonnax.basis`).
"""

import importlib
import warnings
from typing import Any

from pyrox_gp._basis._rff import draw_rff_cosine_basis, evaluate_rff_cosine_paths
from pyrox_gp._basis._spectral_density import spectral_density


_GEONNAX_BASIS_NAMES = frozenset(
    {
        "SlepianCapBasis",
        "divfree_basis",
        "eof_basis",
        "fourier_basis",
        "fourier_basis_1d",
        "fourier_eigenvalues",
        "fourier_eigenvalues_1d",
        "gabor_frame",
        "gabor_frame_grid",
        "gaussian_window_features",
        "graph_laplacian_eigpairs",
        "harmonic_degrees",
        "rbf_basis",
        "real_spherical_harmonics",
        "shannon_number",
        "slepian_cap_basis",
        "slepian_cap_eigh_per_m",
        "slepian_concentration_matrix",
        "spherical_rbf_basis",
        "wavelet_basis_1d",
        "wavelet_basis_2d",
        "wendland_c2",
        "wendland_c4",
    }
)


def __getattr__(name: str) -> Any:
    """Serve the old `geonnax.basis` re-exports, with a `DeprecationWarning`."""
    if name not in _GEONNAX_BASIS_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"pyrox_gp._basis.{name} is deprecated; import it from geonnax.basis.",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module("geonnax.basis"), name)


__all__ = [
    "draw_rff_cosine_basis",
    "evaluate_rff_cosine_paths",
    "spectral_density",
]
