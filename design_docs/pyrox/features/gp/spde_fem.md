---
status: draft
version: 0.1.0
---

# pyrox.gp x SPDE / Finite Element GP Approximation

**Subject:** Matérn GPs via the SPDE approach (Lindgren, Rue & Lindström 2011).
Discretize the Whittle-Matérn SPDE on a triangulated mesh using piecewise-linear
finite elements. Produces a sparse precision matrix (GMRF), enabling $O(n^{3/2})$
inference instead of $O(N^3)$.

**Date:** 2026-04-03

---

## 1  The SPDE

A Gaussian random field $u(s)$ with Matérn covariance is the stationary solution to:

$$(\kappa^2 - \Delta)^{\alpha/2}\bigl(\tau\,u(s)\bigr) = \mathcal{W}(s), \qquad s \in \mathbb{R}^d$$

where $\Delta$ is the Laplacian, $\mathcal{W}$ is Gaussian white noise, and $\alpha = \nu + d/2$.

**Parameter relationships:**

$$\text{range} = \frac{\sqrt{8\nu}}{\kappa}, \qquad \sigma^2 = \frac{\Gamma(\nu)}{\Gamma(\nu + d/2)\,(4\pi)^{d/2}\,\kappa^{2\nu}\,\tau^2}$$

| $\alpha$ | $\nu$ (2-D) | $\nu$ (1-D) | Regularity |
|----------|-------------|-------------|------------|
| 1 | 0 | 0.5 | Continuous, not differentiable |
| 2 | 1 | 1.5 | Once differentiable |
| 3 | 2 | 2.5 | Twice differentiable |

---

## 2  Finite Element Discretization

Triangulate the domain with $n$ nodes. Approximate the solution as:

$$u(s) \approx \sum_{k=1}^{n} \psi_k(s)\,w_k$$

where $\psi_k$ are piecewise-linear (P1) basis functions: $\psi_k = 1$ at node $k$, $0$ at all other nodes, linear within each triangle.

### FEM matrices

**Mass matrix** $C \in \mathbb{R}^{n \times n}$ (sparse):

$$C_{ij} = \int \psi_i(s)\,\psi_j(s)\,ds$$

Element mass matrix for triangle $T$ with area $|T|$:

$$C^T = \frac{|T|}{12}\begin{pmatrix} 2 & 1 & 1 \\ 1 & 2 & 1 \\ 1 & 1 & 2 \end{pmatrix}$$

**Lumped mass matrix** $\tilde{C}$ (diagonal approximation, used in practice):

$$\tilde{C}_{ii} = \sum_{T \ni i} \frac{|T|}{3}$$

**Stiffness matrix** $G \in \mathbb{R}^{n \times n}$ (sparse):

$$G_{ij} = \int \nabla\psi_i \cdot \nabla\psi_j\,ds$$

For triangle $T$ with vertices $(x_1,y_1), (x_2,y_2), (x_3,y_3)$, define:

$$D = \begin{pmatrix} x_3 - x_2 & x_1 - x_3 & x_2 - x_1 \\ y_3 - y_2 & y_1 - y_3 & y_2 - y_1 \end{pmatrix}$$

Element stiffness: $G^T = D^\top D / (4|T|)$.

---

## 3  Precision Matrix

Define $K_\kappa = \kappa^2 C + G$. The precision matrix of the GMRF weights $w \sim \mathcal{N}(0, Q^{-1})$:

**$\alpha = 1$:**

$$Q_1 = \tau^2\,K_\kappa$$

**$\alpha = 2$** (most common):

$$Q_2 = \tau^2\,K_\kappa\,\tilde{C}^{-1}\,K_\kappa = \tau^2\bigl(\kappa^4 C + 2\kappa^2 G + G\,\tilde{C}^{-1}\,G\bigr)$$

**General recursion** (integer $\alpha$):

$$Q_\alpha = \tau^2\,K_\kappa\,\tilde{C}^{-1}\,Q_{\alpha-2}\,\tilde{C}^{-1}\,K_\kappa$$

**Sparsity:** $\alpha = 1$ gives mesh-adjacency sparsity. $\alpha = 2$ extends to second-order neighbors. Each additional order widens the band by one neighborhood ring.

---

## 4  Observation Model

Observations $y_i$ at locations $s_i$ (not necessarily mesh nodes) are linked via a projection matrix $A \in \mathbb{R}^{N \times n}$:

$$y = A\,w + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma_e^2 I)$$

Row $i$ of $A$ contains the **barycentric coordinates** of $s_i$ in its enclosing triangle — exactly 3 nonzero entries per row.

---

## 5  Log-Likelihood

Prior: $w \sim \mathcal{N}(0, Q^{-1})$. Posterior precision and mean:

$$Q_\text{post} = Q + \frac{1}{\sigma_e^2} A^\top A, \qquad \mu_\text{post} = Q_\text{post}^{-1}\,\frac{A^\top y}{\sigma_e^2}$$

Marginal log-likelihood:

$$\log p(y \mid \theta) = \frac{1}{2}\Bigl[\log|Q| - \log|Q_\text{post}| - \frac{\|y\|^2}{\sigma_e^2} + \mu_\text{post}^\top Q_\text{post}\,\mu_\text{post} - N\log(2\pi\sigma_e^2)\Bigr]$$

Log-determinants computed via sparse Cholesky: $\log|Q| = 2\sum_i \log L_{ii}$.

---

## 6  Prediction

For new locations $s_*$, construct projection $A_*$ (same barycentric interpolation):

$$u(s_*) \mid y \;\sim\; \mathcal{N}\!\bigl(A_*\,\mu_\text{post},\; A_*\,Q_\text{post}^{-1}\,A_*^\top\bigr)$$

Predictive variance without forming $Q_\text{post}^{-1}$: solve $Q_\text{post}\,Z = A_*^\top$ via sparse Cholesky.

---

## 7  Complexity

| Operation | Dense GP | SPDE ($n$ mesh nodes, $N$ obs, 2-D) |
|---|---|---|
| Matrix assembly | $O(N^2)$ | $O(n)$ |
| Cholesky | $O(N^3)$ | $O(n^{3/2})$ sparse |
| Log-determinant | $O(N^3)$ | $O(n^{3/2})$ from sparse Cholesky |
| Solve | $O(N^2)$ per RHS | $O(n \log n)$ with sparse factors |
| Memory | $O(N^2)$ | $O(n \log n)$ for sparse factors |
| Prediction | $O(N)$ per point | $O(1)$ barycentric + sparse solve |

For 3-D meshes, sparse Cholesky costs $O(n^2)$ — still better than $O(N^3)$.

---

## 8  Extensions

### Non-stationary

Make parameters spatially varying:

$$\bigl(\kappa(s)^2 - \Delta\bigr)^{\alpha/2}\bigl(\tau(s)\,u(s)\bigr) = \mathcal{W}(s)$$

Precision becomes: $Q_2 = T\,(K^2 C K^2 + KG + G^\top K + G\tilde{C}^{-1}G)\,T$ where $T = \text{diag}(\tau_i)$, $K = \text{diag}(\kappa_i)$.

### Fractional $\alpha$ (non-integer $\nu$)

Rational approximation of the fractional operator (Bolin & Kirchner 2020):

$$L^{-\alpha} \approx L^{-\lfloor\alpha\rfloor} \cdot p(L^{-1}) / q(L^{-1})$$

### Manifold domains

Replace $\Delta$ with the Laplace-Beltrami operator on the manifold. The FEM mesh is a surface triangulation (e.g., icosahedral mesh for the sphere).

---

## 9  API

### Layer 0 — Primitives (`pyrox.gp._src.fem`)

Pure functions for mesh operations and FEM matrix assembly.

```python
def build_mesh(
    domain: Float[Array, "V 2"],          # boundary polygon vertices
    max_area: float,                       # target triangle area
    extension: float = 0.1,               # domain extension fraction
) -> tuple[Float[Array, "n 2"], Int[Array, "m 3"]]:
    """Delaunay triangulation with boundary extension.

    Returns (vertices, triangles).
    """
    ...

def assemble_mass_matrix(
    vertices: Float[Array, "n 2"],
    triangles: Int[Array, "m 3"],
    lumped: bool = True,
) -> SparseArray:
    """Mass matrix C (or lumped diagonal approximation tilde{C}).

    Complexity: O(m) where m = number of triangles.
    """
    ...

def assemble_stiffness_matrix(
    vertices: Float[Array, "n 2"],
    triangles: Int[Array, "m 3"],
) -> SparseArray:
    """Stiffness matrix G.

    Complexity: O(m).
    """
    ...

def build_precision(
    C: SparseArray,                        # mass matrix (lumped)
    G: SparseArray,                        # stiffness matrix
    kappa: float | Float[Array, " n"],     # scale (scalar or per-node)
    tau: float | Float[Array, " n"],       # variance (scalar or per-node)
    alpha: int = 2,                        # SPDE order (1, 2, or 3)
) -> SparseArray:
    """Sparse precision matrix Q.

    Here `C` denotes the lumped diagonal mass matrix approximation
    (written as tilde{C} in the derivation).

    alpha=1: Q = tau^2 * (kappa^2 C + G)
    alpha=2: Q = tau^2 * K tilde{C}^{-1} K    where K = kappa^2 C + G
    alpha=3: Q = tau^2 * K tilde{C}^{-1} Q_1 tilde{C}^{-1} K

    Complexity: O(nnz) where nnz = number of nonzeros in Q.
    """
    ...

def build_projection(
    obs_locs: Float[Array, "N 2"],         # observation locations
    vertices: Float[Array, "n 2"],
    triangles: Int[Array, "m 3"],
) -> SparseArray:
    """Projection matrix A (N x n) via barycentric interpolation.

    Each row has exactly 3 nonzero entries (barycentric coords).
    Complexity: O(N log n) with spatial index, O(Nm) brute force.
    """
    ...

def spde_log_marginal_likelihood(
    y: Float[Array, " N"],
    A: SparseArray,                        # projection (N x n)
    Q: SparseArray,                        # precision (n x n)
    noise_var: float,
) -> Scalar:
    """Marginal log-likelihood via sparse Cholesky.

    Complexity: O(n^{3/2}) in 2-D.
    """
    ...
```

### Layer 1 — Solver (`pyrox.gp.solvers`)

```python
class SPDESolver(eqx.Module):
    """Solver for SPDE-based Matérn GPs.

    Represents the GP via a sparse precision matrix on a FEM mesh.
    Implements the Solver protocol: solve, logdet, log_marginal.
    """
    vertices: Float[Array, "n 2"]
    triangles: Int[Array, "m 3"]
    C: SparseArray                         # lumped mass matrix
    G: SparseArray                         # stiffness matrix
    alpha: int                             # SPDE order

    @staticmethod
    def from_mesh(
        vertices: Float[Array, "n 2"],
        triangles: Int[Array, "m 3"],
        alpha: int = 2,
    ) -> "SPDESolver":
        """Assemble FEM matrices from mesh."""
        ...

    @staticmethod
    def from_domain(
        domain: Float[Array, "V 2"],
        max_area: float,
        alpha: int = 2,
        extension: float = 0.1,
    ) -> "SPDESolver":
        """Build mesh + assemble FEM matrices."""
        ...

    def precision(
        self,
        kappa: float | Float[Array, " n"],
        tau: float | Float[Array, " n"],
    ) -> SparseArray:
        """Build sparse Q from current hyperparameters."""
        ...

    def solve(self, Q: SparseArray, y: Array, noise_var: float) -> SolveResult:
        """Sparse Cholesky solve. O(n^{3/2}) in 2-D."""
        ...

    def logdet(self, Q: SparseArray, noise_var: float) -> Scalar:
        """Log-determinant via sparse Cholesky. O(n^{3/2}) in 2-D."""
        ...

    def log_marginal(
        self,
        Q: SparseArray,
        A: SparseArray,
        y: Array,
        noise_var: float,
    ) -> Scalar:
        """Marginal log-likelihood. O(n^{3/2}) in 2-D."""
        ...

    def predict(
        self,
        Q_post: SparseArray,
        mu_post: Float[Array, " n"],
        A_star: SparseArray,
    ) -> tuple[Float[Array, " N_star"], Float[Array, " N_star"]]:
        """Posterior mean and variance at new locations."""
        ...
```

### Layer 2 — NumPyro Integration

```python
def spde_gp_factor(
    name: str,
    solver: SPDESolver,
    obs_locs: Float[Array, "N 2"],
    y: Float[Array, " N"],
    kappa: float,
    tau: float,
    noise_var: float,
) -> None:
    """Register SPDE GP log-marginal-likelihood as a NumPyro factor.

    Collapsed form — the latent field w is integrated out analytically.
    Use for Matérn GPs with Gaussian likelihood on meshes.

    Usage:
        def model(obs_locs, y):
            range_ = numpyro.sample("range", dist.LogNormal(0, 1))
            sigma = numpyro.sample("sigma", dist.LogNormal(0, 1))
            noise = numpyro.sample("noise", dist.HalfNormal(1))

            kappa = jnp.sqrt(8 * 1.0) / range_   # nu=1 in 2D
            tau = 1.0 / (sigma * kappa * jnp.sqrt(4 * jnp.pi))

            spde_gp_factor("gp", solver, obs_locs, y, kappa, tau, noise)
    """
    A = build_projection(obs_locs, solver.vertices, solver.triangles)
    Q = solver.precision(kappa, tau)
    lml = solver.log_marginal(Q, A, y, noise_var)
    numpyro.factor(name, lml)
```

---

## 10  Implementation Notes

### Sparse Cholesky in JAX

The main challenge. Options:

| Approach | Pros | Cons |
|---|---|---|
| `jax.experimental.sparse` | Pure JAX, JIT-able | No sparse Cholesky yet |
| `scipy.sparse.linalg` (CPU callback) | Mature, stable | Not JIT-able, CPU-only |
| SuiteSparse/CHOLMOD wrapper | Fastest, fill-reducing ordering | External C dependency |
| Custom sparse Cholesky in JAX | Full JAX ecosystem | Significant engineering effort |

Recommended approach: wrap SuiteSparse/CHOLMOD for the sparse Cholesky, with a fallback to scipy. The FEM assembly and projection are pure JAX.

### Mesh boundary effects

The mesh must extend beyond the observation domain to avoid boundary artifacts. R-INLA uses an outer extension of $\sim\text{range}/3$ beyond the data convex hull.

### Differentiability

The FEM matrices $C$ and $G$ depend only on the mesh (fixed). The precision $Q$ depends on $\kappa, \tau$ — the only parameters that need gradients. Since $Q$ is a polynomial in $\kappa$ with fixed sparse structure, the gradient flows through the sparse matrix entries.

---

## 11  References

1. Lindgren, F., Rue, H. & Lindström, J. (2011). *An Explicit Link between Gaussian Fields and Gaussian Markov Random Fields: The Stochastic Partial Differential Equation Approach.* JRSS-B.

2. Bolin, D. & Kirchner, K. (2020). *The Rational SPDE Approach for Gaussian Random Fields with General Smoothness.* J. Comput. Graph. Statist.

3. Fuglstad, G.-A., Simpson, D., Lindgren, F. & Rue, H. (2019). *Constructing Priors that Penalize the Complexity of Gaussian Random Fields.* JASA.

4. Bakka, H., et al. (2018). *Spatial Modeling with R-INLA: A Review.* WIREs Comput. Stat.

5. Solin, A. & Särkkä, S. (2020). *Hilbert Space Methods for Reduced-Rank Gaussian Process Regression.* Stat. Comput.
