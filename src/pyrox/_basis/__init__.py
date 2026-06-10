"""Pure-JAX basis functions shared by NN spectral layers and GP inducing features.

Both `pyrox.nn.HSGPFeatures` (NN-side, weight-space scalable GP) and
`pyrox.gp.FourierInducingFeatures` (GP-side, inter-domain inducing features)
evaluate the same Laplacian eigenfunctions on a bounded box. The spatial
basis zoo (eigenfunction, localized, overcomplete, and data-driven bases)
lives in `geonnax.basis` and is re-exported here so pyrox code keeps a
single import surface; only the kernel-aware pieces stay local:

- `spectral_density` — kernel spectral density evaluated at frequency
  magnitudes (reads pyrox GP kernel hyperparameters).
- `draw_rff_cosine_basis` / `evaluate_rff_cosine_paths` — pure
  random-Fourier-feature prior draws for RBF/Matern kernels, shared by
  `pyrox.gp._pathwise`.

Re-exported from `geonnax.basis`:

- `fourier_basis_1d` / `fourier_eigenvalues_1d` — 1D Dirichlet
  eigenpairs of $-d^2/dx^2$ on $[-L, L]$.
- `fourier_basis` / `fourier_eigenvalues` — tensor-product
  extension to $[-L, L]^D$.
- `real_spherical_harmonics` / `harmonic_degrees` — real SHs on
  the unit 2-sphere.
- `SlepianCapBasis` and the ``slepian_*`` constructors — band-limited
  spherical-cap concentration bases.
- `graph_laplacian_eigpairs` — smallest eigenpairs of a graph Laplacian.
- `rbf_basis` / `spherical_rbf_basis` (+ `wendland_c2` /
  `wendland_c4`) — placeable Gaussian / compact-support bumps in
  Euclidean or geodesic distance.
- `gabor_frame` / `gabor_frame_grid` — overcomplete multiscale
  Gabor dictionaries.
- `wavelet_basis_1d` / `wavelet_basis_2d` — orthonormal DWT
  synthesis matrices.
- `divfree_basis` — divergence-free 2D vector basis from stream
  functions.
- `eof_basis` — data-driven empirical orthogonal functions (PCA).
- `gaussian_window_features` — Gaussian-window localized time gates.
"""

from geonnax.basis import (
    SlepianCapBasis,
    divfree_basis,
    eof_basis,
    fourier_basis,
    fourier_basis_1d,
    fourier_eigenvalues,
    fourier_eigenvalues_1d,
    gabor_frame,
    gabor_frame_grid,
    gaussian_window_features,
    graph_laplacian_eigpairs,
    harmonic_degrees,
    rbf_basis,
    real_spherical_harmonics,
    shannon_number,
    slepian_cap_basis,
    slepian_cap_eigh_per_m,
    slepian_concentration_matrix,
    spherical_rbf_basis,
    wavelet_basis_1d,
    wavelet_basis_2d,
    wendland_c2,
    wendland_c4,
)

from pyrox._basis._rff import draw_rff_cosine_basis, evaluate_rff_cosine_paths
from pyrox._basis._spectral_density import spectral_density


__all__ = [
    "SlepianCapBasis",
    "divfree_basis",
    "draw_rff_cosine_basis",
    "eof_basis",
    "evaluate_rff_cosine_paths",
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
    "spectral_density",
    "spherical_rbf_basis",
    "wavelet_basis_1d",
    "wavelet_basis_2d",
    "wendland_c2",
    "wendland_c4",
]
