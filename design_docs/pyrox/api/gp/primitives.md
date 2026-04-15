---
status: draft
version: 0.1.0
---

# Layer 0 — Primitives

Pure JAX functions. Stateless, differentiable, composable. No NumPyro, no protocols — just `(Array, ...) → Array`.

These implement the mathematical core that Layer 1 protocols compose into richer interfaces. For the full subsystem-level API, see [gp_moments.md](gp_moments.md), [gp_state_space.md](gp_state_space.md), and [gp_integration.md](gp_integration.md).

---

## Kernel Evaluation (`pyrox.gp._src.kernels`)

Pure functions that evaluate kernel (covariance) functions on pairs of inputs.

### `rbf_kernel(X1, X2, variance, lengthscale)`

**Mathematical definition:**

$$k(x, x') = \sigma^2 \exp\!\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$

**Complexity:** $O(N_1 N_2 D)$ where $D$ is input dimension.

### `matern_kernel(X1, X2, variance, lengthscale, nu)`

$$k(x, x') = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}\,r}{\ell}\right)^\nu K_\nu\!\left(\frac{\sqrt{2\nu}\,r}{\ell}\right), \quad r = \|x - x'\|$$

Special cases: $\nu = 1/2$ (exponential), $\nu = 3/2$, $\nu = 5/2$ (closed-form), $\nu \to \infty$ (RBF).

### `periodic_kernel(X1, X2, variance, lengthscale, period)`

$$k(x, x') = \sigma^2 \exp\!\left(-\frac{2\sin^2(\pi|x-x'|/p)}{\ell^2}\right)$$

### `linear_kernel(X1, X2, variance, bias)`

$$k(x, x') = \sigma^2 x^T x' + b$$

### `kernel_add(k1_matrix, k2_matrix)`, `kernel_mul(k1_matrix, k2_matrix)`

Kernel composition: $(k_1 + k_2)(x,x') = k_1(x,x') + k_2(x,x')$.

---

## Covariance Math (`pyrox.gp._src.covariance`)

### `cholesky_solve(L, b)`

Solve $K x = b$ given Cholesky factor $L$: forward-solve $Lz = b$, back-solve $L^T x = z$.

### `log_marginal_likelihood(y, K, noise_var)`

**The two atoms of GP inference** (Rasmussen & Williams 2006):

$$\log p(y) = -\frac{1}{2}\underbrace{y^T (K + \sigma^2 I)^{-1} y}_{\text{solve}} - \frac{1}{2}\underbrace{\log|K + \sigma^2 I|}_{\text{logdet}} - \frac{N}{2}\log(2\pi)$$

### `predictive_mean_and_var(K_star, K, K_star_star, alpha, L)`

**GP posterior predictive:**

$$\mu_* = K_{*f} \alpha, \qquad \sigma^2_* = K_{**} - \|L^{-1} K_{f*}\|^2$$

where $\alpha = (K + \sigma^2 I)^{-1} y$ and $L = \operatorname{chol}(K + \sigma^2 I)$.

---

## Whitening & Projection (`pyrox.gp._src.whitening`)

Functions extracted from the VGP/SVGP reference implementations. These are the core building blocks for variational GP inference.

### `compute_cholesky(kernel, X, jitter)`

Compute stabilized Gram matrix and Cholesky: $\tilde{K} = K + \varepsilon I = LL^T$.

**Complexity:** $O(N^2 D + N^3)$.

### `whiten_forward(L, v)`

Map whitened latent to function values (Eq. 5): $f = Lv$ where $v \sim \mathcal{N}(0, I)$.

### `project_A(K_XZ, L_Z)`

Compute projection matrix (Eq. 7): $A_X = K_{XZ} K_{ZZ}^{-1}$ via Cholesky solves.

$$A_X^T = L_Z^{-T}(L_Z^{-1} K_{XZ}^T)$$

**Complexity:** $O(BM^2)$ for $B$ batch points, $M$ inducing.

### `unwhiten_mean(m_tilde, L_Z)`

Unwhiten variational mean: $m_u = L_Z \tilde{m}$.

### `conditional_base_variance(kernel, X, K_XZ, A_X)`

Conditional variance ignoring variational uncertainty (Eq. 9):

$$\text{Var}_\text{cond}(x_i) = k(x_i, x_i) - (A_X K_{ZX})_{ii}$$

---

## Variational Covariance Sugar (`pyrox.gp._src.covariance_sugar`)

Compute $\Sigma_f = L \tilde{S} L^T$ for different variational families. All exploit structure to avoid forming full $N \times N$ matrices where possible.

### `diag_cov_mean_field(L, std)`

Mean-field $\tilde{S} = \text{diag}(\sigma^2)$: $\text{Var}_q[f_i] = \sum_k (L_{ik} \sigma_k)^2$. **Complexity:** $O(N^2)$.

### `diag_cov_low_rank(L, Lr, diag_std)`

Low-rank + diagonal $\tilde{S} = L_r L_r^T + \text{diag}(\sigma^2)$:

$$\text{Var}_q[f_i] = \|(LL_r)_{i,:}\|^2 + \|(L_{i,:} \odot \sigma)\|^2$$

### `diag_cov_full_rank(L, L_tilde)`

Full-rank $\tilde{S} = \tilde{L}\tilde{L}^T$: $\text{Var}_q[f_i] = \|(L\tilde{L})_{i,:}\|^2$.

### `full_cov_low_rank(L, Lr, diag_std)` / `full_cov_full_rank(L, L_tilde)`

Full $N \times N$ covariance (when needed for predictions): $\Sigma_f = (LL_r)(LL_r)^T + (L\text{diag}(\sigma))(L\text{diag}(\sigma))^T$.

### SVGP variants: `full_S_u_mean_field`, `full_S_u_low_rank`, `full_S_u_full_rank`

Same operations but in inducing space: $S_u = L_Z \tilde{S} L_Z^T$.

### SVGP diagonal variance contributions: `diag_variance_contrib_*`

Per-family computation of $\text{diag}(A_X S_u A_X^T)$ — the variational correction to the conditional variance.

---

## Kalman Primitives (`pyrox.gp._src.kalman`)

### `kalman_predict(m, P, A, Q)`

$$\hat{m}_{t|t-1} = A\,m_{t-1|t-1}, \qquad \hat{P}_{t|t-1} = A\,P_{t-1|t-1}\,A^T + Q$$

### `kalman_update(m_pred, P_pred, H, R, y)`

$$v = y - H\hat{m}, \quad S = H\hat{P}H^T + R, \quad K = \hat{P}H^T S^{-1}$$
$$m_{t|t} = \hat{m} + Kv, \quad P_{t|t} = \hat{P} - KSK^T$$

### `rts_smooth(m_filt, P_filt, m_pred, P_pred, A)`

$$G = P_{t|t}\,A^T\,P_{t+1|t}^{-1}, \quad m_{t|T} = m_{t|t} + G(m_{t+1|T} - \hat{m}_{t+1|t})$$

**Complexity:** $O(N S^3)$ for $N$ timesteps, state dimension $S$.

---

## Quadrature Rules (`pyrox.gp._src.quadrature`)

### `gauss_hermite_points_and_weights(K)`

Points and weights for $\int f(x) \mathcal{N}(x; 0, 1)\,dx \approx \sum_{k=1}^K w_k f(x_k)$.

Exact for polynomials of degree $\leq 2K - 1$.

### `sigma_points(mean, cov, alpha, beta, kappa)`

Unscented transform: $2P + 1$ deterministic points for $P$-dimensional Gaussian.

$$\mathcal{X}_0 = \mu, \quad \mathcal{X}_i = \mu \pm \sqrt{(P + \lambda)\Sigma}_i, \quad \lambda = \alpha^2(P + \kappa) - P$$

Third-order accuracy. Scales linearly in $P$.

### `cubature_points(mean, cov)`

Spherical-radial cubature: $2P$ points, third-order, no tuning parameters.

### `spherical_harmonics_basis(X, max_degree)`

Evaluate real spherical harmonics $Y_\ell^m(x)$ up to degree $L$ at points $X$ on $S^{d-1}$.
Returns the $N \times (L+1)^2$ basis matrix $\Phi$. Used by `SphericalHarmonicFeatures` (VISH).

### `spectral_density(kernel, omega)`

Evaluate the spectral density $S_D(\omega)$ of a stationary kernel at frequencies $\omega$.
Returns the Mercer coefficients used by inducing feature methods (VISH, VFF).
See [features/gp_inducing_features.md](../../features/gp/inducing_features.md).

---

## Basis Functions (`pyrox.gp._src.basis`)

Pure functions for evaluating spectral basis functions on various domains. These are the building blocks for inducing feature methods (VISH, VFF, Laplacian) and Hilbert-space GP approximations. Delegates to spectraldiffx for production-grade transforms.

### `real_spherical_harmonics(theta, phi, max_degree)`

Evaluate real spherical harmonics $Y_\ell^m(\theta, \varphi)$ for $\ell = 0, \ldots, L$ and $m = -\ell, \ldots, \ell$.

**Math:**

$$Y_\ell^m(\theta, \varphi) = \begin{cases} \sqrt{2}\,N_\ell^m\,P_\ell^m(\cos\theta)\,\cos(m\varphi) & m > 0 \\ N_\ell^0\,P_\ell^0(\cos\theta) & m = 0 \\ \sqrt{2}\,N_\ell^{|m|}\,P_\ell^{|m|}(\cos\theta)\,\sin(|m|\varphi) & m < 0 \end{cases}$$

where $P_\ell^m$ are Associated Legendre Polynomials and $N_\ell^m = \sqrt{\frac{2\ell+1}{4\pi}\frac{(\ell-m)!}{(\ell+m)!}}$ is the normalization.

**Returns:** $\Phi \in \mathbb{R}^{N \times M}$ where $M = (L+1)^2$.

**Complexity:** $O(N L^2)$ using the ALP three-term recurrence.

```python
def real_spherical_harmonics(
    theta: Float[Array, " N"],       # colatitude [0, pi]
    phi: Float[Array, " N"],         # longitude [0, 2*pi]
    max_degree: int,                 # L
) -> Float[Array, "N M"]: ...       # M = (L+1)^2
```

### `associated_legendre_matrix(cos_theta, max_degree)`

Precompute the Associated Legendre Polynomial matrix via three-term recurrence.

**Math:** $P_\ell^m(x)$ satisfies:

$$(2\ell+1)\,x\,P_\ell^m = (\ell-m+1)\,P_{\ell+1}^m + (\ell+m)\,P_{\ell-1}^m$$

**Returns:** $P \in \mathbb{R}^{N \times (L+1)^2}$.

**Complexity:** $O(N L^2)$.

```python
def associated_legendre_matrix(
    cos_theta: Float[Array, " N"],
    max_degree: int,
) -> Float[Array, "N M"]: ...
```

### `spherical_harmonic_eigenvalues(max_degree)`

Laplace-Beltrami eigenvalues on $S^2$:

$$\lambda_\ell = -\frac{\ell(\ell+1)}{R^2}, \quad \text{multiplicity } 2\ell+1$$

Used for spectral density evaluation of zonal kernels on the sphere.

```python
def spherical_harmonic_eigenvalues(
    max_degree: int,
    radius: float = 1.0,
) -> Float[Array, " M"]: ...    # M = (L+1)^2, repeated per order m
```

### `zonal_kernel_spectral_coefficients(kernel, max_degree)`

Mercer coefficients of a zonal (isotropic) kernel on $S^{d-1}$:

$$k(\cos\gamma) = \sum_{\ell=0}^{\infty} \hat{k}_\ell \sum_{m=-\ell}^{\ell} Y_\ell^m(x)\,Y_\ell^m(x')$$

where $\hat{k}_\ell = S_d(\lambda_\ell)$ is the spectral density evaluated at the Laplace-Beltrami eigenvalue $\lambda_\ell$.

```python
def zonal_kernel_spectral_coefficients(
    kernel_fn: Callable,             # k(cos_angle) -> scalar
    max_degree: int,
) -> Float[Array, " L"]: ...        # coefficients per degree
```

### `fourier_basis(X, n_basis, domain_length)`

Fourier eigenfunctions on $[-L, L]$ for VFF inducing features.

**Math:** Dirichlet:

$$\phi_k(x) = \sqrt{\frac{1}{L}}\sin\!\left(\frac{k\pi(x + L)}{2L}\right), \quad k = 1, \ldots, D$$

**Returns:** $\Phi \in \mathbb{R}^{N \times D}$.

**Complexity:** $O(ND)$.

```python
def fourier_basis(
    X: Float[Array, "N 1"],          # 1-D inputs
    n_basis: int,                    # D
    domain_length: float,            # L (half-length)
    bc: str = "dirichlet",           # "dirichlet" | "neumann" | "periodic"
) -> Float[Array, "N D"]: ...
```

### `fourier_eigenvalues(n_basis, domain_length, bc)`

Laplacian eigenvalues on $[-L, L]$ for a given boundary condition.

$$\lambda_k = -\left(\frac{k\pi}{2L}\right)^2 \quad \text{(Dirichlet)}$$

```python
def fourier_eigenvalues(
    n_basis: int,
    domain_length: float,
    bc: str = "dirichlet",
) -> Float[Array, " D"]: ...
```

### `laplacian_eigenfunctions(eigvecs, X_indices)`

Evaluate precomputed Laplacian eigenvectors at data indices (for graph/manifold domains).

```python
def laplacian_eigenfunctions(
    eigvecs: Float[Array, "V D"],     # D eigenvectors of the graph Laplacian
    X_indices: Int[Array, " N"],      # indices into the vertex set
) -> Float[Array, "N D"]: ...
```

### spectraldiffx integration

For production-grade spherical harmonic transforms (SHT), pyrox.gp delegates to spectraldiffx:

```python
from spectraldiffx import SphericalHarmonicTransform, SphericalGrid2D

grid = SphericalGrid2D(n_lat=64, n_lon=128)
sht = SphericalHarmonicTransform(grid)
coeffs = sht.to_spectral(field)       # forward SHT
field = sht.from_spectral(coeffs)     # inverse SHT
```

The `pyrox.gp._src.basis` functions are lower-level: they evaluate basis functions at *arbitrary* points (not on a regular grid), which is what the inducing feature methods need.

---

## Pathwise Sampling (`pyrox.gp._src.pathwise`)

### `rff_prior_draw(key, kernel, n_features)`

Draw a prior function sample via Random Fourier Features:

$$\hat{f}(x) = \sqrt{\frac{2\sigma^2}{D}} \sum_{d=1}^{D} \cos(\omega_d^T x + b_d), \quad \omega_d \sim S(\omega), \; b_d \sim \text{Uniform}[0, 2\pi]$$

where $S(\omega)$ is the spectral density of the kernel (Bochner's theorem).
Returns a callable `fn(X) -> Array` that evaluates the random function at arbitrary inputs.

Valid for stationary kernels only. The returned function is a JAX-compatible PyTree (can be vmapped, jitted, differentiated).

### `pathwise_update(prior_at_X, y, L, K_cross)`

Compute the Matheron update correction: $\Delta f(x_*) = K(x_*, X)\,K_{XX}^{-1}\,(y - f_\text{prior}(X))$.

Wraps `gaussx.matheron_update` with GP-specific reshaping. The $O(M^3)$ Cholesky $L$ is pre-computed during training.

**Complexity:** $O(M \cdot N_*)$ per sample.

---

## State-Space Conversion (`pyrox.gp._src.sde`)

### `matern_to_sde(variance, lengthscale, nu)`

Convert a Matern kernel to a Linear Time-Invariant SDE representation:

$$dx(t) = A\,x(t)\,dt + L\,d\beta(t)$$

Returns $(A, Q, H, P_\infty)$ where $A$ is the drift matrix, $Q$ the diffusion, $H$ the observation projection, and $P_\infty$ the stationary covariance.

| Kernel | State dim $S$ |
|---|---|
| Matern-1/2 | 1 |
| Matern-3/2 | 2 |
| Matern-5/2 | 3 |
| Matern-$(\nu+1/2)$ | $\nu + 1$ |
