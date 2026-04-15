---
status: draft
version: 0.1.0
---

# pyrox.gp x Inter-Domain Inducing Features

**Subject:** Spectral inducing feature methods (VISH, VFF, Laplacian eigenfunctions)
that use basis functions as inducing variables within the standard SVGP framework.

**Date:** 2026-04-02

---

## 1  Scope

pyrox.gp supports spectral inducing feature methods that preserve the
full GP prior and approximate the posterior via variational inference.
Weight-space methods (HSGP, Bayesian HSGP) that truncate the prior are
explicitly out of scope — those are prior approximations, not GP inference.

**In scope:** VISH, VFF, Laplacian eigenfunctions (inducing features in SVGP)
**Out of scope:** HSGP / weight-space methods ($f = \Phi \beta$, truncated prior)

**Key insight:** All these methods fit into the standard SVGP framework
(Titsias 2009, Hensman 2013) — only $K_{uu}$ and $k_u(x)$ change.
The ELBO, whitening, predictions, and guide families are reused unchanged.

---

## 2  Common Mathematical Framework

Every inducing feature method defines inducing variables as inner products
in the RKHS:

$$u_m = \langle f, \phi_m \rangle_{\mathcal{H}_k}$$

where $\{\phi_m\}$ are basis functions (eigenfunctions of an operator
associated with the kernel). This gives:

$$K_{uu,mn} = \langle \phi_m, \phi_n \rangle_{\mathcal{H}_k}, \qquad k_u(x)_m = \langle k(\cdot, x), \phi_m \rangle_{\mathcal{H}_k}$$

For orthogonal eigenfunction bases, $K_{uu}$ is **diagonal** —
reducing all $O(M^3)$ operations to $O(M)$.

---

## 3  Gap Catalog

### Gap 1: VISH — Variational Inducing Spherical Harmonics

**Domain:** Unit sphere $S^{d-1}$ (geospatial, climate, cosmology).

**Math:** Inducing variables are RKHS inner products with spherical harmonics:

$$u_{\ell m} = \langle f, Y_\ell^m \rangle_{\mathcal{H}_k}$$

For a zonal (isotropic) kernel $k(\cos\gamma) = \sum_\ell \hat{k}_\ell \sum_m Y_\ell^m(x) Y_\ell^m(x')$:

$$K_{uu} = \text{diag}(\hat{k}_\ell), \qquad k_u(x)_{\ell m} = \hat{k}_\ell \, Y_\ell^m(x)$$

where $\hat{k}_\ell = S(\lambda_\ell)$ is the kernel's spectral density at
Laplace-Beltrami eigenvalue $\lambda_\ell = -\ell(\ell+1)/R^2$.

**Complexity:** $O(NM)$ for $k_u$; $O(M)$ for $K_{uu}$ (diagonal). $M = (L+1)^2$ features for max degree $L$.

```python
class SphericalHarmonicFeatures(eqx.Module):
    """VISH inducing features on the unit sphere."""
    max_degree: int                          # L -> M = (L+1)^2 features

    def K_uu(self, kernel: Kernel) -> Float[Array, "M M"]: ...   # diagonal
    def k_u(self, X: Float[Array, "N 2"], kernel: Kernel) -> Float[Array, "N M"]: ...
    def is_diagonal(self) -> bool: ...       # True — enables O(M) operations
```

**Basis functions from:** `pyrox.gp._src.basis.real_spherical_harmonics`
**Spectral coefficients from:** `pyrox.gp._src.basis.zonal_kernel_spectral_coefficients`
**Production SHT via:** `spectraldiffx.SphericalHarmonicTransform`

**Ref:** Dutordoir et al. (2020) *Sparse Gaussian Processes with Spherical Harmonic Features.* ICML.

**Impl ref:** [jej_vc_snippets/basis_functions/spherical_harmonics/gps/hsgp.py](https://github.com/jejjohnson)

---

### Gap 2: VFF — Variational Fourier Features

**Domain:** Bounded interval $[-L, L]^D$ (temporal, 1-D spatial).

**Math:** Inducing variables are RKHS inner products with Fourier eigenfunctions:

$$u_k = \langle f, \phi_k \rangle_{\mathcal{H}_k}, \quad \phi_k(x) = \sqrt{\frac{1}{L}}\sin\!\left(\frac{k\pi(x+L)}{2L}\right)$$

For a stationary kernel with spectral density $S(\omega)$:

$$K_{uu} = \text{diag}\bigl(S(\omega_k)\bigr), \qquad k_u(x)_k = S(\omega_k)\,\phi_k(x)$$

where $\omega_k = k\pi / (2L)$ are the Fourier frequencies.

**Complexity:** $O(ND)$ for $k_u$; $O(D)$ for $K_{uu}$ (diagonal). $D$ = number of Fourier features.

```python
class FourierFeatures(eqx.Module):
    """VFF inducing features on a bounded interval."""
    n_basis: int                             # D
    domain_length: float                     # L (half-length)
    bc: str                                  # "dirichlet" | "neumann" | "periodic"

    def K_uu(self, kernel: Kernel) -> Float[Array, "D D"]: ...   # diagonal
    def k_u(self, X: Float[Array, "N 1"], kernel: Kernel) -> Float[Array, "N D"]: ...
    def is_diagonal(self) -> bool: ...       # True
```

**Basis functions from:** `pyrox.gp._src.basis.fourier_basis`
**Eigenvalues from:** `pyrox.gp._src.basis.fourier_eigenvalues`
**Production transforms via:** `spectraldiffx.dst`, `spectraldiffx.dct`, `spectraldiffx.fft`

**Ref:** Hensman, Durrande & Solin (2017) *Variational Fourier Features for Gaussian Processes.* JMLR.

---

### Gap 3: Laplacian Eigenfunctions

**Domain:** Compact Riemannian manifold or graph (mesh, point cloud, molecule).

**Math:** Inducing variables use eigenfunctions of the Laplace-Beltrami operator $\Delta$:

$$\Delta \phi_k = \lambda_k \phi_k, \qquad u_k = \langle f, \phi_k \rangle_{\mathcal{H}_k}$$

For a Matern kernel, the spectral density is:

$$S(\lambda_k) = \frac{\sigma^2 \, 2^d \pi^{d/2} \Gamma(\nu + d/2)}{\Gamma(\nu)} \left(\frac{2\nu}{\ell^2} + |\lambda_k|\right)^{-(\nu + d/2)}$$

$K_{uu}$ is diagonal when the eigenfunctions are orthonormal.

**Complexity:** $O(ND)$ for $k_u$. Precomputation: $O(V^3)$ or $O(V D)$ (Lanczos) for the eigendecomposition of the $V \times V$ graph Laplacian.

```python
class LaplacianFeatures(eqx.Module):
    """Laplacian eigenfunction inducing features on a manifold or graph."""
    eigvecs: Float[Array, "V D"]             # precomputed eigenvectors
    eigvals: Float[Array, " D"]              # precomputed eigenvalues

    def K_uu(self, kernel: Kernel) -> Float[Array, "D D"]: ...   # diagonal
    def k_u(self, X: Int[Array, " N"], kernel: Kernel) -> Float[Array, "N D"]: ...
    def is_diagonal(self) -> bool: ...       # True
```

**Basis evaluation from:** `pyrox.gp._src.basis.laplacian_eigenfunctions`

**Ref:** Borovitskiy et al. (2020) *Matern Gaussian Processes on Riemannian Manifolds.* NeurIPS.

---

### Gap 4: Decoupled Inter-Domain Features

**Domain:** General — mixes spatial inducing points with spectral features.

**Math:** Separate inducing variables for the mean and covariance:

$$\mu_{q}(x) = k(x, Z_\mu)\,\alpha, \qquad \Sigma_{q}(x, x') \approx k_u^{\text{spec}}(x)^\top S\, k_u^{\text{spec}}(x')$$

where $Z_\mu$ are standard inducing points and $k_u^{\text{spec}}$ uses spectral features.

**Complexity:** $O(N M_\mu + N M_\Sigma)$ — can use fewer spectral features for covariance.

```python
class DecoupledInducingFeatures(eqx.Module):
    """Mixed point + spectral inducing features."""
    mean_inducing: Float[Array, "M_mu D"]        # spatial inducing points
    cov_features: SphericalHarmonicFeatures | FourierFeatures  # spectral

    def K_uu_mean(self, kernel: Kernel) -> Float[Array, "M_mu M_mu"]: ...
    def K_uu_cov(self, kernel: Kernel) -> Float[Array, "M_cov M_cov"]: ...
    def k_u_mean(self, X, kernel) -> Float[Array, "N M_mu"]: ...
    def k_u_cov(self, X, kernel) -> Float[Array, "N M_cov"]: ...
```

**Ref:** Shi, Titsias & Minka (2020) *Sparse Orthogonal Variational Inference for Gaussian Processes.*

---

## 4  Shared Infrastructure

All inducing feature methods share the same SVGP machinery — they only
provide different $K_{uu}$ and $k_u(x)$:

| Component | Source | Notes |
|---|---|---|
| Basis function evaluation | `pyrox.gp._src.basis` | See [api/gp/primitives.md](../../api/gp/primitives.md) §Basis Functions |
| Spectral density evaluation | `pyrox.gp._src.basis.zonal_kernel_spectral_coefficients` | Mercer coefficients |
| Eigenvalues (spherical) | `pyrox.gp._src.basis.spherical_harmonic_eigenvalues` | $-\ell(\ell+1)/R^2$ |
| Eigenvalues (Fourier) | `pyrox.gp._src.basis.fourier_eigenvalues` | Delegates to `spectraldiffx` |
| Production SHT | `spectraldiffx.SphericalHarmonicTransform` | Grid-based forward/inverse |
| SVGP prediction | Standard `pyrox.gp.guides.InducingPointGuide` | Unchanged — just swap $K_{uu}$, $k_u$ |
| Diagonal $K_{uu}$ optimization | `gaussx.DiagonalLinearOperator` | $O(M)$ instead of $O(M^3)$ |

---

## 5  References

1. Dutordoir, V., Durrande, N., & Hensman, J. (2020). *Sparse Gaussian Processes with Spherical Harmonic Features.* ICML.
2. Hensman, J., Durrande, N., & Solin, A. (2017). *Variational Fourier Features for Gaussian Processes.* JMLR.
3. Borovitskiy, V., et al. (2020). *Matern Gaussian Processes on Riemannian Manifolds.* NeurIPS.
4. Shi, J., Titsias, M. K., & Minka, T. (2020). *Sparse Orthogonal Variational Inference for Gaussian Processes.* AISTATS.
5. Titsias, M. (2009). *Variational Learning of Inducing Variables in Sparse Gaussian Processes.* AISTATS.
6. Solin, A. & Särkkä, S. (2020). *Hilbert Space Methods for Reduced-Rank Gaussian Process Regression.* Statistics and Computing.
