"""Pure-JAX eigenfunction bases shared by NN spectral layers and GP inducing features.

Both `pyrox.nn.HSGPFeatures` (NN-side, weight-space scalable GP) and
`pyrox.gp.FourierInducingFeatures` (GP-side, inter-domain inducing features)
evaluate the same Laplacian eigenfunctions on a bounded box. Owning that
math here removes the duplication and keeps the basis pandas-free.

Public surface:

- :func:`fourier_basis_1d` / :func:`fourier_eigenvalues_1d` — 1D Dirichlet
  eigenpairs of :math:`-d^2/dx^2` on :math:`[-L, L]`.
- :func:`fourier_basis` — tensor-product extension to :math:`[-L, L]^D`.
- :func:`real_spherical_harmonics` — real SHs on the unit 2-sphere.
- :func:`graph_laplacian_eigpairs` — smallest eigenpairs of a graph Laplacian.
- :func:`spectral_density` — kernel spectral density evaluated at frequency magnitudes.
- :func:`draw_rff_cosine_basis` / :func:`evaluate_rff_cosine_paths` — pure
  random-Fourier-feature prior draws for RBF/Matern kernels, shared by
  :mod:`pyrox.gp._pathwise`.
"""

from pyrox._basis._fourier import (
    fourier_basis,
    fourier_basis_1d,
    fourier_eigenvalues,
    fourier_eigenvalues_1d,
)
from pyrox._basis._laplacian import graph_laplacian_eigpairs
from pyrox._basis._rff import draw_rff_cosine_basis, evaluate_rff_cosine_paths
from pyrox._basis._spectral_density import spectral_density
from pyrox._basis._spherical import harmonic_degrees, real_spherical_harmonics


__all__ = [
    "draw_rff_cosine_basis",
    "evaluate_rff_cosine_paths",
    "fourier_basis",
    "fourier_basis_1d",
    "fourier_eigenvalues",
    "fourier_eigenvalues_1d",
    "graph_laplacian_eigpairs",
    "harmonic_degrees",
    "real_spherical_harmonics",
    "spectral_density",
]
