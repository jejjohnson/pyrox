---
status: reference
version: 0.1.0
surveyed: 2026-04-16
---

# Kernel libraries — ecosystem survey

Snapshot of the kernel surface across five GP / ML-kernel libraries, taken while scoping Wave 2 Epic 2.A (pyrox `#19` / `#20`). The goal was to identify closed-form stationary (and a few well-known non-stationary) kernels worth shipping as `pyrox.gp` math primitives, and to distinguish them from structured / scalable / spectral constructions that belong in `gaussx`.

Re-run this survey if you're about to extend `pyrox.gp._src.kernels` — it answers "why does pyrox have X but not Y?" without relitigating the decision.

## Libraries surveyed

| Library | URL | Focus |
|---|---|---|
| GPyTorch | https://github.com/cornellius-gp/gpytorch | Production GP library in PyTorch |
| GPJax | https://github.com/JaxGaussianProcesses/GPJax | Production GP library in JAX |
| tinygp | https://github.com/dfm/tinygp | Lightweight exact GPs in JAX |
| kernelmethods | https://github.com/raamana/kernelmethods | ML/SVM kernels (classical) |
| mlkernels | https://github.com/wesselb/mlkernels | Functional kernel composition library |

## Per-library inventory

**GPyTorch** (`gpytorch/kernels/`)
: RBF, Matern, RationalQuadratic, Constant, Linear, Polynomial, Cosine, Periodic, Cylindrical, Hamming, ArcCosine (ARC), Gibbs, PiecewisePolynomial, SpectralMixture, SpectralDelta, RFF, GaussianSymmetrizedKL, DistributionalInput

**GPJax** (`gpjax/kernels/{stationary,nonstationary}/`)
: Stationary — RBF, Matern12, Matern32, Matern52, Periodic, PoweredExponential, RationalQuadratic, White. Non-stationary — ArcCosine, Linear, Polynomial.

**tinygp** (`src/tinygp/kernels/`)
: Exp, ExpSquared, Matern32, Matern52, Cosine, ExpSineSquared, RationalQuadratic, DotProduct, Polynomial. Plus L1/L2 distance metric helpers.

**kernelmethods** (`numeric_kernels.py`)
: Gaussian, Laplacian, Polynomial, Linear, Sigmoid, Chi2, Hadamard. ML/SVM flavor — thin overlap with GP literature.

**mlkernels**
: EQ, CEQ, RQ, Matern12/Exp, Matern32, Matern52, Linear, Delta, Decaying, Log, Posterior, Subspace, TensorProduct.

## Deduplicated comparison

| Kernel | GPyTorch | GPJax | tinygp | kernelmethods | mlkernels | In pyrox? |
|---|---|---|---|---|---|---|
| RBF / SquaredExp / EQ | yes | yes | yes | yes | yes | **yes** |
| Matern 1/2 (Exp/Laplacian) | yes | yes | yes | yes | yes | **yes** |
| Matern 3/2 | yes | yes | yes | — | yes | **yes** |
| Matern 5/2 | yes | yes | yes | — | yes | **yes** |
| Periodic / ExpSineSquared | yes | yes | yes | — | (via EQ+warp) | **yes** |
| Linear | yes | yes | yes | yes | yes | **yes** |
| RationalQuadratic | yes | yes | yes | — | yes | **yes** |
| Polynomial | yes | yes | yes | yes | — | **yes** |
| Cosine | yes | — | yes | — | — | **yes** |
| Constant / White noise | yes | yes | — | — | yes (Delta) | **yes** |
| PoweredExponential (gamma-exp) | — | yes | — | — | — | deferred |
| ArcCosine | yes (ARC) | yes | — | — | — | deferred |
| PiecewisePolynomial / Wendland | yes | — | — | — | — | deferred |
| Gibbs (non-stationary lengthscale) | yes | — | — | — | — | later wave |
| Cylindrical / Hamming / DistributionalInput | yes | — | — | — | — | not planned |
| SpectralMixture / SpectralDelta / RFF | yes | — | — | — | — | gaussx |
| Sigmoid / Chi2 / Hadamard | — | — | — | yes | — | not planned |
| Decaying / Log / Subspace / TensorProduct | — | — | — | — | yes | composition |

## Decisions

### Shipped in Wave 2 (Issue #19)

The pyrox layer owns closed-form *math functions* (`pyrox.gp._src.kernels`). Nine kernels shipped:

| Kernel | Rationale |
|---|---|
| RBF | Universal baseline |
| Matern (ν ∈ {0.5, 1.5, 2.5}) | Common half-integer orders with closed forms; no Bessel evaluation needed |
| Periodic (MacKay) | Standard periodic baseline |
| Linear | Standard non-stationary baseline |
| RationalQuadratic | Appears in all four GP libs; scale mixture of RBFs |
| Polynomial | Generalization of Linear; degree as static param |
| Cosine | Plain `cos(2π r/p)` — pairs with Periodic |
| White | `σ² δ(x, x')`; additive noise component |
| Constant | `σ²`; rank-one offset component |

Each kernel is re-exposed as a `Parameterized` class in `pyrox.gp` (Issue #20) with positivity constraints on `variance`, `lengthscale`, `period`, `alpha` where appropriate.

### Deferred (candidates for a follow-up issue)

These are legitimate closed-form primitives, but weren't Wave 2 blockers and didn't want to expand `#19` scope beyond survey consensus. Good fit for a `Wave 2.5` issue when someone needs them:

- **PoweredExponential** (gamma-exponential): `σ² exp(-(r/ℓ)^γ)`. One-line generalization; GPJax ships it.
- **ArcCosine** (orders 0/1/2): non-stationary, "neural-flavored" without a neural net. Closed form exists (Cho & Saul 2009); GPyTorch and GPJax ship it.
- **PiecewisePolynomial / Wendland**: compactly supported, closed form. Useful for sparse GP workloads (diagonal-dominant sparse Gram). GPyTorch ships it.

### Out of scope for `pyrox.gp` kernel layer

- **Spectral / feature methods** (SpectralMixture, SpectralDelta, RFF, VFF, VISH): these are *feature/spectral approximations*, not pure `k(x, x')` primitives. They belong in gaussx (Layer 1 operators) or a later pyrox scalable-inference wave.
- **Non-stationary with input-dependent machinery** (Gibbs, Cylindrical, DistributionalInput): defer until a concrete need surfaces.
- **SVM / ML kernels** (Sigmoid, Chi2, Hadamard): not PD in the standard GP sense or not standard in the GP literature.
- **Deep / NTK / quasisep / state-space**: explicitly excluded from Wave 2; belong to later waves or sister libraries.
- **Multi-output / structured** (LMC, ICM, Coregion, TensorProduct): multi-output waves in pyrox or gaussx operators.

## Split with gaussx — where to find what

| You want… | Go to… |
|---|---|
| The closed-form math of a standard kernel | `pyrox.gp._src.kernels` |
| A `Parameterized` wrapper with priors/constraints | `pyrox.gp.{RBF, Matern, Periodic, Linear, ...}` |
| Mixed-precision / numerically stable RBF matrix | `gaussx.stable_rbf_kernel` |
| Implicit / structured kernel operators | `gaussx.ImplicitKernelOperator`, `gaussx.KernelOperator` |
| Cholesky, solve, logdet, log-marginal-likelihood | `gaussx.{cholesky, solve, logdet, log_marginal_likelihood}` |
| Predictive mean/variance, whitening, quadrature | `gaussx.{predict_mean, predict_variance, whiten_covariance, gauss_hermite_points, sigma_points, cubature_points}` |
| Random Fourier Features, VFF, VISH | `gaussx` (Layer 1 operators) or later pyrox scalable-inference wave |
