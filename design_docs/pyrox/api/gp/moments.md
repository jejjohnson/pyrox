---
status: draft
version: 0.1.0
---

# pyrox.gp — Design Document

> Core architecture, component protocols, and mathematical foundations.
> For worked model examples and usage patterns, see [`../examples/README.md`](../examples/README.md).

---

## 1. Philosophical Stance

pyrox.gp is **not** a GP library that happens to work with NumPyro.  It is a
set of GP building blocks designed to be embedded inside arbitrary Bayesian
hierarchical models.

The fundamental objects are:

| Component | What it is | What it provides |
|-----------|-----------|------------------|
| **Kernel** | Covariance function k(·,·) | A `CovarianceRepresentation` |
| **Solver** | Linear algebra backend | `solve(K, y)` and `log_det(K)` |
| **GPPrior** | `numpyro.Distribution` over f | `sample(key)` and `log_prob(f)` |
| **gp_sample** | Functional entry point | Latent f as a NumPyro site |
| **gp_factor** | Functional entry point | Collapsed marginal likelihood as a NumPyro factor |
| **Guide** | Structured q(f) | Variational family respecting GP geometry |

The user owns the model.  pyrox.gp owns the GP math.

---

## 2. The Two Atoms of GP Inference

All GP computations reduce to two operations on the noisy kernel matrix
K_y = K + σ²I:

$$
\text{(A1) solve:} \quad \alpha = K_y^{-1} y
$$

$$
\text{(A2) log-det:} \quad \log |K_y|
$$

These appear in the log marginal likelihood:

$$
\log p(y \mid X, \theta) = -\tfrac{1}{2} y^\top \alpha - \tfrac{1}{2} \log|K_y| - \tfrac{N}{2}\log(2\pi)
$$

and in the posterior predictive:

$$
\mu_* = K_{*f}\,\alpha, \qquad \Sigma_* = K_{**} - K_{*f}\,K_y^{-1}\,K_{f*}
$$

Different GP models differ **only** in how K is represented and how (A1)
and (A2) are computed.  This is why the Solver protocol is the central
abstraction.

---

## 3. Kernel Protocol

### Contract

```
KernelProtocol[R]:
    __call__(X, X2=None) -> R          # covariance representation
    diagonal(X) -> (N,)                 # k(x_i, x_i) in O(N)
```

A kernel maps inputs to a **covariance representation** R.  The type R
determines which solvers are compatible.

### Instances

| Kernel | Output R | Parameters | Notes |
|--------|----------|-----------|-------|
| `RBF` | `DenseCov` | σ², ℓ (scalar or ARD) | Most common; isotropic or anisotropic |
| `Matern` | `DenseCov` or `StateSpaceRep` | σ², ℓ, ν ∈ {1/2, 3/2, 5/2} | State-space form available for 1-D |
| `Periodic` | `DenseCov` | σ², ℓ, period | Can compose with Matern for quasi-periodic |
| `ArcCosine` | `DenseCov` | order ∈ {0,1,2}, weight/bias variances | Neural network kernel (Cho & Saul 2009) |
| `KroneckerKernel` | `KroneckerCov` | k₁, k₂, ... (factor kernels) | Product of independent kernels on subspaces |
| `InducingKernel` | `LowRankPlusDiag` | base kernel + Z ∈ ℝ^{M×D} | Nyström / inducing-point approximation |

### Composition

Kernel combinators preserve or degrade the representation type:

- **Sum**: `k₁ + k₂`.  If both produce `DenseCov`, result is `DenseCov(K₁ + K₂)`.
  If representations differ, materialize to dense.
- **Product**: `k₁ · k₂`.  Dense: elementwise `K₁ ⊙ K₂`.
  Kronecker: `KroneckerCov((K₁, K₂))` when inputs factorize.
- **Scale**: `c · k`.  Preserves representation, scales internal matrix.

---

## 4. Covariance Representations

The representation is the "currency" between Kernel and Solver.  Each
representation encodes structure that a compatible solver can exploit.

### 4.1 DenseCov

The default.  Stores the full N×N matrix.

```
DenseCov:
    matrix: (N, N)    symmetric positive (semi-)definite
```

No structure exploited.  Compatible with all moment-based solvers.

### 4.2 LowRankPlusDiag

Approximation K ≈ WWᵀ + diag(d),  W ∈ ℝ^{N×M},  M ≪ N.

```
LowRankPlusDiag:
    W: (N, M)     low-rank factor
    d: (N,)       diagonal
```

Arises from inducing-point methods.  Given inducing locations Z ∈ ℝ^{M×D}:

$$
W = K_{fZ}\,L_{ZZ}^{-1}, \quad d_i = k(x_i, x_i) - W_i W_i^\top
$$

where L_ZZ = chol(K_ZZ + εI).  This is the Nyström approximation;
setting d = 0 gives FITC when combined with diagonal correction.

### 4.3 KroneckerCov

Product structure K = K₁ ⊗ K₂ ⊗ ··· ⊗ K_p.

```
KroneckerCov:
    factors: tuple of (N_i, N_i) arrays
```

Arises when inputs live on a grid: X = X₁ × X₂ × ··· × X_p and the
kernel factorizes.  Total size is ∏N_i but each factor is only N_i × N_i.

### 4.4 StateSpaceRep

Linear time-invariant (LTI) state-space model encoding a temporal GP:

```
StateSpaceRep:
    A:  (S, S)     state transition
    Q:  (S, S)     process noise covariance
    H:  (1, S)     emission matrix
    x0: (S,)       initial state mean
    P0: (S, S)     initial state covariance
    dt: (N-1,)     time increments
```

Encodes the GP covariance implicitly through the dynamics.  The state
dimension S is small (e.g. S=2 for Matérn-3/2, S=3 for Matérn-5/2).

The correspondence between SDE and GP kernel (Hartikainen & Särkkä, 2010):

$$
dx(t) = A\,x(t)\,dt + B\,dW(t), \qquad f(t) = H\,x(t)
$$

$$
\text{Cov}[f(t), f(t')] = H\,e^{A|t-t'|}\,P_\infty\,H^\top
$$

---

## 5. Solver Protocol

### Contract

```
SolverProtocol[R]:
    solve(rep: R, y, noise_var) -> SolveResult(alpha, aux)
    log_det(rep: R, noise_var) -> scalar
    predictive_moments(rep_train, cross_cov, test_diag,
                       solve_result, noise_var, full_cov) -> (mean, var)
```

The `SolveResult.aux` field carries solver-specific cached data (Cholesky
factor, preconditioner, filter states, etc.) that `predictive_moments`
reuses.  This avoids redundant computation between training and prediction.

### Taxonomy

Solvers fall into three families based on how they approach (A1) and (A2).

---

### 5.1 Moment Solvers

Operate on explicit matrix representations via linear algebra decompositions.

#### 5.1.1 CholeskySolver

**Representation**: `DenseCov`

The textbook approach.  Form K_y = K + σ²I, compute the Cholesky
factorization, then solve via back-substitution.

$$
K_y = LL^\top, \qquad \alpha = L^{-\top}(L^{-1}y)
$$

$$
\log|K_y| = 2\sum_i \log L_{ii}
$$

| Property | Value |
|----------|-------|
| Complexity (solve) | O(N³/3) for Cholesky + O(N²) for back-sub |
| Complexity (log_det) | O(N) after Cholesky |
| Memory | O(N²) for L |
| Exact | Yes |
| Caches | L in `aux` |

**Literature**: Rasmussen & Williams (2006), Algorithm 2.1.

#### 5.1.2 CGSolver (Conjugate Gradients)

**Representation**: `DenseCov` (via matrix-vector products only)

Iteratively solves K_y α = y without forming or factorizing K_y.
Only requires the matrix-vector product K_y v, which can be applied
lazily.

$$
\alpha^{(k+1)} = \alpha^{(k)} + \gamma_k\,p_k, \quad
p_{k+1} = r_{k+1} + \beta_k\,p_k
$$

where r_k = y − K_y α^{(k)} is the residual and γ_k, β_k are the
standard CG step sizes.

For log_det, use stochastic trace estimation (Hutchinson, 1990):

$$
\log|K_y| \approx \frac{1}{J}\sum_{j=1}^J z_j^\top \log(K_y)\,z_j, \quad z_j \sim \mathcal{N}(0, I)
$$

where log(K_y) z is computed via the Lanczos algorithm.

| Property | Value |
|----------|-------|
| Complexity (solve) | O(k · N²) per CG iteration, k ≪ N with preconditioning |
| Complexity (log_det) | O(J · k · N²) with Lanczos |
| Memory | O(N) working memory (no matrix stored if lazy matvec) |
| Exact | No (but controllable residual tolerance) |
| Caches | α, preconditioner in `aux` |

**Literature**: Gibbs & MacKay (1997); Gardner et al. (2018).

#### 5.1.3 BBMMSolver (Blackbox Matrix-Matrix)

**Representation**: `DenseCov` (via matrix-vector products)

Extension of CG that simultaneously solves multiple right-hand sides
and computes stochastic log_det estimates.  The key idea (Gardner et al.,
2018) is to run *modified batched CG* (mBCG) on the augmented system:

$$
K_y\,[\alpha \mid T_1 \mid \cdots \mid T_J] = [y \mid z_1 \mid \cdots \mid z_J]
$$

where z_j are probe vectors for the stochastic trace estimator.
This amortizes the matrix-vector products: each CG iteration
advances both the solve and the log_det estimate.

For the log_det, the Lanczos tridiagonalization T_j produced during
CG on each probe vector gives:

$$
z_j^\top \log(K_y)\,z_j \approx e_1^\top \log(T_j)\,e_1 \cdot \|z_j\|^2
$$

$$
\log|K_y| \approx \frac{1}{J}\sum_{j=1}^J z_j^\top \log(K_y)\,z_j
$$

| Property | Value |
|----------|-------|
| Complexity | O(k · (J+1) · N²) total (solve + log_det amortized) |
| Memory | O((J+1) · N) working vectors |
| Exact | No (stochastic log_det, iterative solve) |
| Preconditioning | Pivoted Cholesky partial factorization rank-r: O(r²N) |
| Caches | α, Lanczos tridiagonals, preconditioner in `aux` |

**Literature**: Gardner et al. (2018) "GPyTorch: Blackbox Matrix-Matrix
Gaussian Process Inference with GPU Acceleration".

#### 5.1.4 WoodburySolver

**Representation**: `LowRankPlusDiag`

Exploits the Woodbury identity for K_y = WWᵀ + D where D = diag(d) + σ²I:

$$
K_y^{-1} = D^{-1} - D^{-1}W\,(I + W^\top D^{-1}W)^{-1}\,W^\top D^{-1}
$$

$$
\log|K_y| = \log|I + W^\top D^{-1}W| + \sum_i \log D_{ii}
$$

The inner matrix I + Wᵀ D⁻¹ W is only M×M, so Cholesky on it costs O(M³).

| Property | Value |
|----------|-------|
| Complexity (solve) | O(NM² + M³) |
| Complexity (log_det) | O(NM² + M³) |
| Memory | O(NM) for W, O(M²) for inner Cholesky |
| Exact | Yes (given the low-rank approximation) |
| Caches | Inner Cholesky, D⁻¹W in `aux` |

**Literature**: Titsias (2009) "Variational Learning of Inducing Variables
in Sparse Gaussian Processes"; Hensman et al. (2013, 2015).

#### 5.1.5 KroneckerSolver

**Representation**: `KroneckerCov`

Eigendecompose each factor independently:

$$
K_i = U_i \Lambda_i U_i^\top, \quad i = 1, \ldots, p
$$

Then the full eigendecomposition of K = K_1 ⊗ ··· ⊗ K_p is:

$$
K = (U_1 \otimes \cdots \otimes U_p)\,(\Lambda_1 \otimes \cdots \otimes \Lambda_p)\,(U_1 \otimes \cdots \otimes U_p)^\top
$$

The eigenvalues of K_y are λ_{i_1} · λ_{i_2} ··· λ_{i_p} + σ², giving:

$$
\alpha = (U_1 \otimes \cdots \otimes U_p)\,\text{diag}\!\left(\frac{1}{\lambda_{i_1}\cdots\lambda_{i_p} + \sigma^2}\right)(U_1 \otimes \cdots \otimes U_p)^\top y
$$

$$
\log|K_y| = \sum_{i_1,\ldots,i_p} \log(\lambda_{i_1}\cdots\lambda_{i_p} + \sigma^2)
$$

The Kronecker matrix-vector products use the "vec trick":
(A ⊗ B) vec(X) = vec(BXAᵀ), avoiding explicit formation of the
∏N_i × ∏N_i matrix.

| Property | Value |
|----------|-------|
| Complexity (decompose) | O(Σ_i N_i³) |
| Complexity (solve) | O(∏N_i · Σ_i N_i) via vec trick |
| Complexity (log_det) | O(∏N_i) sum over eigenvalue products |
| Memory | O(Σ_i N_i²) for eigenvectors |
| Exact | Yes |
| Caches | Per-factor eigendecompositions in `aux` |

**Literature**: Saatçi (2012) "Scalable Inference for Structured
Gaussian Process Models"; Wilson et al. (2014).

---

### 5.2 Temporal Solvers

Operate on state-space representations via sequential filtering and smoothing.

#### 5.2.1 KalmanSolver

**Representation**: `StateSpaceRep`

Converts the GP inference problem into Kalman filtering (forward pass)
and RTS smoothing (backward pass).

**Forward pass (filter)** — processes observations sequentially:

$$
\text{Predict:} \quad \hat{x}_{t|t-1} = A\,\hat{x}_{t-1|t-1}, \quad P_{t|t-1} = A\,P_{t-1|t-1}\,A^\top + Q
$$

$$
\text{Update:} \quad S_t = H\,P_{t|t-1}\,H^\top + \sigma^2, \quad K_t = P_{t|t-1}\,H^\top\,S_t^{-1}
$$

$$
\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t(y_t - H\,\hat{x}_{t|t-1}), \quad P_{t|t} = (I - K_t H)\,P_{t|t-1}
$$

The log marginal likelihood accumulates innovation terms:

$$
\log p(y \mid \theta) = -\tfrac{1}{2}\sum_{t=1}^N \left[\frac{(y_t - H\hat{x}_{t|t-1})^2}{S_t} + \log S_t + \log 2\pi\right]
$$

**Backward pass (RTS smoother)** — for the full posterior:

$$
G_t = P_{t|t}\,A^\top\,P_{t+1|t}^{-1}
$$

$$
\hat{x}_{t|N} = \hat{x}_{t|t} + G_t(\hat{x}_{t+1|N} - A\,\hat{x}_{t|t}), \quad P_{t|N} = P_{t|t} + G_t(P_{t+1|N} - P_{t+1|t})G_t^\top
$$

| Property | Value |
|----------|-------|
| Complexity (filter) | O(N · S³) where S is state dim (typically 2–6) |
| Complexity (smooth) | O(N · S³) backward |
| Memory | O(N · S²) for stored filter states |
| Exact | Yes (for the state-space representation) |
| Parallelizable | Yes, via associative scan (parallel Kalman filter) |
| Caches | Filter means, covariances, innovations in `aux` |

**Additional methods beyond SolverProtocol**:

```
TemporalSolver(SolverProtocol[StateSpaceRep]):
    filter(rep, y, noise_var) -> FilterResult
    smooth(filter_result) -> SmoothResult
```

The smoother output can double as a **guide** for the temporal component
in variational inference — the smoothed posterior is the optimal Gaussian
approximation to q(f) for the temporal subproblem.

**Literature**: Hartikainen & Särkkä (2010) "Kalman filtering and smoothing
solutions to temporal Gaussian process regression models";
Chang et al. (2020) "Parallel Kalman filtering".

---

---

## 6. GPPrior — The Core Distribution

`GPPrior` is a `numpyro.distributions.Distribution` over function values
f ∈ ℝᴺ at locations X.

```
GPPrior(kernel, X, solver, mean_fn=None, jitter=1e-6)
    .sample(key)     -> f ~ N(m(X), K(X,X))
    .log_prob(f)     -> log N(f; m, K)
    .condition(y, σ²) -> ConditionedGP
```

### Design: Eager vs Lazy Construction

GPPrior computes the kernel representation at construction time.
This is deliberate: inside a NumPyro model, the kernel hyperparameters
are traced by JAX, so the representation is a function of traced values.
The eager computation ensures JIT compilation captures the full
computational graph.

```python
# Inside a NumPyro model, σ² and ℓ are traced JAX values
kernel = RBF(variance=σ², lengthscale=ℓ)
prior = GPPrior(kernel, X, solver)        # K computed here (traced)
f = numpyro.sample("f", prior)            # uses K for log_prob
```

### ConditionedGP

After observing y through a Gaussian likelihood:

```
ConditionedGP(prior, y, noise_var)
    .predict(X_test, full_cov=False) -> (mean, var)
```

This is a convenience for the conjugate case.  For non-conjugate likelihoods,
there is no `ConditionedGP` — the user places a `GPPrior` on f and lets
NumPyro handle the posterior via MCMC or VI.

---

## 7. Entry Points: gp_sample vs gp_factor

Two functional entry points for embedding GPs in NumPyro models.

### 7.1 gp_sample

```python
f = gp_sample(name, kernel, X, solver, mean_fn=None)
```

Registers f as an explicit latent variable via `numpyro.sample(name, GPPrior(...))`.

**Internals**: Constructs a `GPPrior` from the kernel and solver, then
calls `numpyro.sample`.  The `log_prob` of the GPPrior is used by NUTS
for gradient computation and by SVI for the ELBO.

**Implications**:
- f is N-dimensional → expensive for MCMC (but fine with good guides in SVI).
- f is available for downstream computation (feed into any likelihood).
- Requires a guide for q(f) in SVI.

### 7.2 gp_factor

```python
gp_factor(name, kernel, X, y, noise_var, solver, mean_fn=None)
```

Registers the *collapsed* log marginal likelihood as a `numpyro.factor`.
The latent f is analytically integrated out.

**Internals**: Calls `solver.solve` and `solver.log_det` on the kernel
representation, computes the log marginal likelihood, and adds it via
`numpyro.factor(name, lml)`.

**Implications**:
- No latent f → low-dimensional posterior over hyperparams only.
- Only valid for Gaussian likelihoods (conjugacy required for analytic integration).
- No guide needed for f (only for hyperparameters if using SVI).
- f is not available inside the model; prediction is a separate post-hoc step.

### Decision Matrix

| Question | gp_sample | gp_factor |
|----------|-----------|-----------|
| Is the likelihood Gaussian? | Either | Required |
| Need f inside the model? | Yes | No |
| Total latent dims | N + dim(θ) | dim(θ) |
| MCMC cost | High (N-dim) | Low (dim(θ)-dim) |
| SVI guide needed for f? | Yes (structured) | No |
| Prediction | From guide posterior | From ConditionedGP post-hoc |

---

## 8. Guide Protocol

### Contract

```
GPGuideProtocol:
    __call__(site_name, *args, **kwargs) -> None    # register q(f) in NumPyro
    get_posterior(params) -> (mean_f, var_f)          # extract moments post-training
```

### The Whitening Principle

All GP guides should work in a **whitened** parameterization.  Given
L = chol(K + εI), define v = L⁻¹f so that the prior is p(v) = N(0, I).

The variational distribution q(v) is placed on v, and f = Lv maps back
to function space.  This has two benefits:

1. **Isotropic optimization landscape**: The prior is spherical regardless
   of the kernel's lengthscale or variance.  Without whitening, the
   posterior geometry mirrors K's eigenspectrum, which can span many
   orders of magnitude.

2. **Simplified KL divergence**: KL[q(v) ‖ p(v)] = KL[q(v) ‖ N(0,I)]
   has closed form for any Gaussian q(v).

### Instances

| Guide | q(v) | Parameters | Cost |
|-------|------|-----------|------|
| `WhitenedDeltaGuide` | δ(v − m̃) | m̃ ∈ ℝᴺ | O(N²) for f = Lm̃ |
| `WhitenedMeanFieldGuide` | N(m̃, diag(σ²)) | m̃, σ ∈ ℝᴺ | O(N²) per step |
| `WhitenedFullRankGuide` | N(m̃, L̃L̃ᵀ) | m̃ ∈ ℝᴺ, L̃ ∈ ℝ^{N×N} | O(N³) per step |
| `InducingPointGuide` | N(m_u, S_u) on M inducing values | m_u ∈ ℝᴹ, S_u ∈ ℝ^{M×M} | O(NM²) per step |
| `KalmanGuide` | RTS smoother output | Per-timestep means/covs | O(NS³) per step |
| `FlowGuide` | Normalizing flow on v | Flow parameters | O(N · flow_cost) |

### ComposedGuide

Combines GP-specific guides with generic autoguides for non-GP sites:

```
ComposedGuide(
    gp_guides: {site_name: GPGuideProtocol},
    model: callable,
    auto_guide_cls: AutoNormal | AutoMultivariateNormal | ...,
)
```

Internally uses `numpyro.handlers.block` to partition sites: GP sites are
handled by their respective structured guides, everything else by the
autoguide.

---

## 9. Solver–Representation Compatibility Matrix

| | DenseCov | LowRank+Diag | Kronecker | StateSpace |
|---|:---:|:---:|:---:|:---:|
| **CholeskySolver** | ✓ | | | |
| **CGSolver** | ✓ | ✓ (lazy) | ✓ (lazy) | |
| **BBMMSolver** | ✓ | ✓ (lazy) | ✓ (lazy) | |
| **WoodburySolver** | | ✓ | | |
| **KroneckerSolver** | | | ✓ | |
| **KalmanSolver** | | | | ✓ |

"Lazy" means the solver only needs the matrix-vector product K·v, not the
explicit matrix.  This is natural for CG and BBMM, which are Krylov methods.

---

## 10. Package Structure (Proposed)

```
pyrox/gp/
├── __init__.py
├── protocols.py          # Protocol definitions
├── representations.py    # DenseCov, LowRankPlusDiag, KroneckerCov, etc.
├── kernels/
│   ├── __init__.py
│   ├── base.py           # KernelProtocol, combinators (Sum, Product, Scale)
│   ├── stationary.py     # RBF, Matern, Periodic, ArcCosine
│   ├── kronecker.py      # KroneckerKernel
│   └── inducing.py       # InducingKernel (Nyström wrapper)
├── solvers/
│   ├── __init__.py
│   ├── cholesky.py       # CholeskySolver
│   ├── cg.py             # CGSolver
│   ├── bbmm.py           # BBMMSolver
│   ├── woodbury.py       # WoodburySolver
│   ├── kronecker.py      # KroneckerSolver
│   ├── kalman.py         # KalmanSolver (filter + smoother)
├── distributions/
│   ├── __init__.py
│   └── gp_prior.py       # GPPrior, ConditionedGP
├── guides/
│   ├── __init__.py
│   ├── whitened.py        # WhitenedDelta, WhitenedMeanField, WhitenedFullRank
│   ├── inducing.py        # InducingPointGuide
│   ├── kalman.py          # KalmanGuide
│   ├── flow.py            # FlowGuide
│   └── composed.py        # ComposedGuide
├── functional.py          # gp_sample, gp_factor, gp_log_marginal_likelihood
├── nn/                    # Equinox module wrappers
│   ├── __init__.py
│   ├── exact_gp.py        # ExactGP(eqx.Module)
│   ├── sparse_gp.py       # SparseGP(eqx.Module)
│   └── kernels.py         # RBFKernel(eqx.Module) with learnable params
└── examples.md            # Usage patterns catalog
```

---

## 11. References

- **Rasmussen & Williams (2006)**. Gaussian Processes for Machine Learning. MIT Press.
- **Titsias (2009)**. Variational Learning of Inducing Variables in Sparse Gaussian Processes. AISTATS.
- **Hensman, Fusi & Lawrence (2013)**. Gaussian Processes for Big Data. UAI.
- **Hensman, Matthews & Ghahramani (2015)**. Scalable Variational Gaussian Process Classification. AISTATS.
- **Gardner, Pleiss, Weinberger, Bindel & Wilson (2018)**. GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration. NeurIPS.
- **Gibbs & MacKay (1997)**. Efficient Implementation of Gaussian Processes for Interpolation.
- **Hartikainen & Särkkä (2010)**. Kalman Filtering and Smoothing Solutions to Temporal GP Regression Models. MLSP.
- **Chang, Wilkinson, Khan & Solin (2020)**. Fast Variational Learning in State-Space Gaussian Process Models. MLSP.
- **Rahimi & Recht (2007)**. Random Features for Large-Scale Kernel Machines. NeurIPS.
- **Lázaro-Gredilla, Quiñonero-Candela, Rasmussen & Figueiras-Vidal (2010)**. Sparse Spectrum Gaussian Process Regression. JMLR.
- **Wilson & Adams (2013)**. Gaussian Process Kernels for Pattern Discovery and Extrapolation. ICML.
- **Wilson & Nickisch (2015)**. Kernel Interpolation for Scalable Structured Gaussian Processes (KISS-GP). ICML.
- **Saatçi (2012)**. Scalable Inference for Structured Gaussian Process Models. PhD Thesis, Cambridge.
- **Dietrich & Newsam (1997)**. Fast and Exact Simulation of Stationary Gaussian Processes through Circulant Embedding. SIAM.
- **Hutchinson (1990)**. A Stochastic Estimator of the Trace of the Influence Matrix. Communications in Statistics.
