---
status: draft
version: 0.1.0
---

# Architecture and Dependencies

## Overview

pyrox merges the former pyrox-nn (Bayesian deep learning) and pyrox-gp (GP building blocks) into a single package with three subpackages: `_core`, `nn`, and `gp`. The core provides the shared foundation (PyroxModule, Parameterized), while `nn` and `gp` build on it independently.

---

## Layer Stacks

### Core + NN (from pyrox-nn)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 1 — Probabilistic Layers (edward2-style)                        │
│  eqx.Module subclasses. Drop-in Bayesian replacements.                 │
│  DenseVariational, DenseFlipout, MCDropout, RandomFourierFeatures,     │
│  DenseNCP, DenseBatchEnsemble                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 0 — Core Abstractions                                           │
│  PyroxModule, PyroxParam, PyroxSample, Parameterized, _Context            │
│  Bridges Equinox modules ↔ NumPyro tracing                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### GP (from pyrox-gp)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 3 — Integrators                                                  │
│  Compute E_q[h(f)] under Gaussian q(f).                                 │
│  Sigma points, cubature, Taylor, MC, exact RBF.                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 2 — Inference Strategies                                         │
│  Approximate non-conjugate likelihoods via natural parameters.          │
│  Laplace, EP, VI, Posterior Linearisation, Gauss-Newton.                │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 1 — Solvers (backed by GaussX)                                    │
│  Linear algebra: solve(K, y), log_det(K).                               │
│  Delegates to gaussx.ops / gaussx.solvers for structured operators.     │
│  Dense, CG, Nystrom, Kronecker, Kalman (temporal).                      │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 0 — Kernels                                                      │
│  Covariance functions k(.,.) -> CovarianceRepresentation.               │
│  Stationary, additive, product, state-space, multi-output.              │
└─────────────────────────────────────────────────────────────────────────┘
```

### Shared Foundation (not owned by pyrox)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GaussX — structured linear operators, solve, logdet, cholesky          │
│  NumPyro — numpyro.sample, numpyro.param, distributions, inference     │
│  Equinox — eqx.Module, eqx.tree_at, eqx.filter_jit                    │
│  JAX — jax.numpy, jax.random, jax.lax, vmap, jit                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Protocols (GP subpackage)

| Protocol | Purpose | Key Methods |
|---|---|---|
| `Kernel` | Covariance function | `__call__(X, Y)` -> `CovarianceRepresentation` |
| `Solver` | Linear algebra backend | `solve(K, y)`, `log_det(K)` |
| `InferenceStrategy` | Approximate inference | `compute_sites(...)` -> natural parameters |
| `Integrator` | Gaussian expectations | `integrate(fn, mean, cov)` -> `(E[h], Cov[h])` |
| `Guide` | Structured variational posterior | `sample(key)`, `log_prob(f)` |
| `InducingFeatures` | Inter-domain inducing variables | `K_uu(kernel)`, `k_u(X, kernel)` |
| `MultiOutputKernel` | Multi-output kernel (latent + mixing) | `Kgg(X1, X2)`, `mixing_matrix()` |

---

## Package Layout

```
pyrox/
├── __init__.py
├── _core/
│   ├── __init__.py
│   ├── pyrox_module.py         # PyroxModule(eqx.Module), _Context
│   ├── descriptors.py         # PyroxParam (NamedTuple), PyroxSample (dataclass)
│   ├── parameterized.py       # Parameterized — register_param, set_prior,
│   │                          #   autoguide, set_mode, get_param
│   └── utils.py               # pyrox_method decorator, helpers
├── nn/
│   ├── __init__.py
│   ├── dense.py               # DenseReparameterization, DenseFlipout,
│   │                          #   DenseVariational, DenseBatchEnsemble
│   ├── random_feature.py      # RandomFourierFeatures, RandomKitchenSinks
│   ├── dropout.py             # MCDropout
│   └── noise.py               # NCPContinuousPerturb, DenseNCP
├── gp/
│   ├── __init__.py
│   ├── _src/                  # Layer 0: pure JAX functions
│   │   ├── kernels.py
│   │   ├── covariance.py
│   │   ├── kalman.py
│   │   ├── quadrature.py
│   │   ├── sde.py
│   │   ├── pathwise.py
│   │   ├── whitening.py
│   │   └── covariance_sugar.py
│   ├── kernels/               # Kernel protocol + implementations
│   ├── covs/                  # CovarianceRepresentation types
│   ├── solvers/               # Solver protocol + implementations
│   ├── priors/                # GPPrior distribution
│   ├── guides/                # Structured variational guides
│   ├── inducing_features/     # InducingFeatures protocol (VISH, VFF, Laplacian)
│   ├── multi_output/          # MultiOutputKernel protocol, mixing, convenience functions
│   ├── sampling/              # PathwiseSampler, DecoupledPathwiseSampler (Matheron's rule)
│   ├── inference/             # InferenceStrategy implementations
│   ├── integrators/           # Integrator protocol + implementations
│   ├── temporal/              # State-space kernels, Kalman, parallel scan
│   └── ops.py                 # gp_sample, gp_factor entry points
```

---

## Dependency Diagram

```
              ┌──────────┐
              │   jax    │
              │ equinox  │
              └────┬─────┘
                   │
         ┌─────────┼──────────┐
         │         │          │
    ┌────▼────┐ ┌──▼───┐ ┌───▼────┐
    │ lineax  │ │gaussx│ │numpyro │
    └─────────┘ └──┬───┘ └───┬────┘
                   │         │
                ┌──▼─────────▼──┐
                │     pyrox     │
                │  ._core       │
                │  .nn          │
                │  .gp          │
                └───────────────┘
```

---

## Dependencies

### Required

| Package | Version | Role |
|---|---|---|
| `jax` | >=0.4 | Array computation, autodiff, JIT, vmap |
| `equinox` | -- | Neural network modules (immutable PyTrees) |
| `numpyro` | >=0.14 | Probabilistic programming, inference, distributions |
| `gaussx` | >=0.1 | Structured linear operators, operations (`solve`, `logdet`, `cholesky`, `diag`), solver strategies, Gaussian distributions |
| `lineax` | >=0.1 | Base operator abstraction (transitive via gaussx) |

### Optional

| Package | Role | Used by |
|---|---|---|
| `optax` | Optimizers | SVI training |

### Examples only (not library dependencies)

| Package | Role |
|---|---|
| `einops` | Readable tensor operations in examples |
| `matplotlib` | Plotting in examples |
| `scikit-learn` | Data generation, metrics in examples |
| `numpy` | Data handling in examples |

---

## Integration Points

### Core + NN Integration

Component-level mapping of pyrox core and NN abstractions to their ecosystem dependencies.

| pyrox Component | Equinox Dependency | NumPyro Dependency |
|---|---|---|
| `PyroxModule` | `eqx.Module` (base class) | `numpyro.sample`, `numpyro.param` |
| `PyroxParam` | -- | `numpyro.param`, `dist.constraints` |
| `PyroxSample` | -- | `numpyro.sample`, `dist.Distribution` |
| `Parameterized` | `eqx.Module` (base class) | `numpyro.sample`, `numpyro.param`, `dist.constraints` |
| Pattern B (`tree_at`) | `eqx.tree_at` | `numpyro.sample`, `numpyro.deterministic` |
| `DenseVariational` / `DenseFlipout` | -- | `numpyro.sample`, `dist.Normal` |
| `MCDropout` | `eqx.Module` | -- (pure JAX randomness) |
| `RandomFourierFeatures` | `eqx.Module` | -- (deterministic layer) |
| `DenseNCP` | -- | `numpyro.sample`, `numpyro.deterministic` |
| GP kernels (`RBFKernel`, etc.) | `eqx.Module` (via `Parameterized`) | Priors via `set_prior()` |

### GaussX Integration Points

pyrox's GP `Solver` protocol delegates to GaussX's `SolverStrategy` protocol and structured operators:

| pyrox GP Solver | GaussX Backend | Notes |
|---|---|---|
| `DenseSolver` | `gaussx.solvers.DenseSolver` + dense operator | Cholesky-based, small-to-medium |
| `CGSolver` | `gaussx.solvers.CGSolver` + matrix-free operator | CG + SLQ logdet via matfree |
| `KroneckerSolver` | `gaussx.ops.solve` / `gaussx.ops.logdet` on `KroneckerOperator` | Per-factor decomposition |
| `NystromSolver` | `gaussx.operators.LowRankUpdate` + Woodbury | Low-rank + diagonal structure |
| `KalmanSolver` | Direct (state-space, not operator-based) | No GaussX dependency |
| `PathwiseSampler` | `gaussx.matheron_update` + `gaussx.sample_joint_conditional` | Posterior function samples via Matheron's rule |

### CovarianceRepresentation -> GaussX Operator Mapping

pyrox's `CovarianceRepresentation` types map to GaussX operators:

| CovarianceRepresentation | GaussX Operator |
|---|---|
| `Dense` | `lineax.MatrixLinearOperator` |
| `Diagonal` | `lineax.DiagonalLinearOperator` |
| `LowRank` | `gaussx.operators.LowRankUpdate` |
| `Woodbury` | `gaussx.operators.LowRankUpdate` (general base) |
| `Kronecker` | `gaussx.operators.KroneckerOperator` |

### Cross-Subpackage Integration

| Consumer | Provider | Integration Point |
|---|---|---|
| `pyrox.nn` | `pyrox._core` | NN layers subclass `PyroxModule` or use `Parameterized` |
| `pyrox.gp.kernels` | `pyrox._core` | GP kernels subclass `Parameterized` for prior/guide management |
| `pyrox.gp.solvers` | `gaussx` | Solvers delegate to GaussX structured operators |
| `pyrox.gp.sampling` | `gaussx` | PathwiseSampler uses `gaussx.matheron_update` |

---

## CI / Quality Gates

| Check | Command | Scope |
|-------|---------|-------|
| Tests | `uv run pytest tests -x` | Full suite |
| Lint | `uv run ruff check .` | Entire repo |
| Format | `uv run ruff format --check .` | Entire repo |
| Typecheck | `uv run ty check pyrox` | Package only |

All four must pass before merge. GitHub Actions on push/PR.
Conventional commits required (`feat:`, `fix:`, `docs:`, `test:`, etc.).

**Build system:** hatchling (PEP 621)
**Python:** >= 3.12, < 3.14
**License:** MIT
