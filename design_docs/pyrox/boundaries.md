---
status: draft
version: 0.1.0
---

# Boundaries and Ecosystem

## Overview

pyrox owns the Equinox-NumPyro bridge, Edward2-style probabilistic layers, GP kernels/priors/guides, and solver protocols. It delegates inference to NumPyro, neural network primitives to Equinox, structured linear algebra to gaussx, and domain-specific modeling to downstream packages.

The package is organized into three submodules:

- **`pyrox._core`** — shared foundations: `Parameterized`, `PyroxModule`, `PyroxParam`, `PyroxSample`, context cache, RFF spectral densities
- **`pyrox.nn`** — Edward2-style probabilistic NN layers (dense, dropout, NCP, spectral norm)
- **`pyrox.gp`** — GP kernels, mean functions, priors, guides, solver protocol, temporal GP, Gaussian integration

---

## Ownership Map

| Responsibility | Owner | Notes |
|---|---|---|
| Equinox-NumPyro bridge (`PyroxModule`, `PyroxParam`, `PyroxSample`) | **pyrox._core** | Core competency |
| `Parameterized` (prior/guide management) | **pyrox._core** | Shared by nn and gp |
| Per-call context cache | **pyrox._core** | Prevents duplicate sample sites |
| RFF spectral density samplers | **pyrox._core** | Shared by nn (RFF layers) and gp (pathwise sampling) |
| Edward2-style probabilistic layers | **pyrox.nn** | Dense, dropout, RFF, NCP, flipout |
| MC-Dropout layers | **pyrox.nn** | Dropout-based uncertainty |
| Spectral normalization | **pyrox.nn** | SNGP-style distance awareness |
| GP kernels and mean functions | **pyrox.gp** | Kernels subclass `_core.Parameterized` |
| GP priors, guides, inference strategies | **pyrox.gp** | Laplace, EP, VI, PL |
| GP solver protocol (dispatch to LA backend) | **pyrox.gp** | Thin wrapper over gaussx |
| Temporal GP via state-space | **pyrox.gp** | Kalman, Bayes-Newton |
| Gaussian expectation integration | **pyrox.gp** | Sigma points, cubature, MC, exact |
| Structured GP guides | **pyrox.gp** | Variational families respecting GP geometry |
| Inducing features (VISH, VFF) | **pyrox.gp** | Spectral inducing variables for SVGP |
| Pathwise sampling (Matheron's rule) | **pyrox.gp** | Function-valued posterior draws |
| Structured linear operators (Kronecker, BlockDiag, LowRank) | **gaussx** | pyrox uses, doesn't own |
| Linear operations (solve, logdet, cholesky, diag, trace) | **gaussx** | Structure-exploiting dispatch |
| Solver strategies (Dense, CG, BBMM, Auto) | **gaussx** | pyrox.gp Solver delegates to these |
| MVN distributions | **gaussx** | NumPyro-compatible, solver-parameterized |
| Solver algorithms (Cholesky, CG, GMRES, LU) | **lineax** | Transitive via gaussx |
| Iterative/stochastic LA (SLQ, Hutchinson, partial eig) | **matfree** | Transitive via gaussx |
| Inference (MCMC, SVI, Predictive, AutoGuides) | **numpyro** | Never reimplemented |
| Distributions | **numpyro** | `numpyro.distributions` |
| Neural network primitives | **equinox** | `eqx.Module`, `eqx.tree_at` |
| Bayesian hierarchical model structure | **user** | pyrox provides components |
| Data loading, preprocessing, visualization | **external** | User's responsibility |

---

## Decision Table

| Scenario | Recommendation |
|---|---|
| Bayesian linear regression | `PyroxModule` with `pyrox_sample` for weights, NumPyro MCMC or SVI |
| Variational BNN | Stack `pyrox.nn.DenseVariational` / `DenseFlipout` layers, SVI with AutoNormal |
| GP with learnable kernel | `pyrox.gp` kernel (subclasses `Parameterized`) with `set_prior()`, SVI |
| MC-Dropout uncertainty | `pyrox.nn.MCDropout` layers, multiple forward passes at inference |
| Add priors to existing Equinox model | Wrap with `PyroxModule`, use `pyrox_sample()` for selected parameters |
| MAP estimation | `PyroxModule`, SVI with AutoDelta |
| Deep GP | Stack `pyrox.nn.RandomFourierFeatures` layers, SVI |
| Exact GP in hierarchical model | `pyrox.gp.gp_factor` / `gp_sample` inside NumPyro model |
| Sparse GP (SVGP) | `pyrox.gp` inducing features + structured guide, SVI |
| Temporal GP | `pyrox.gp` state-space Kalman solver |
| GP with structured covariance | `pyrox.gp` solver protocol dispatching to gaussx operators |
| GEV/GPD with GP-varying parameters | GP latents from `pyrox.gp` in xtremax extreme value models |

---

## Ecosystem Interactions

| External Package | Integration Point | Pattern |
|---|---|---|
| **equinox** | `eqx.Module` base, `eqx.tree_at` injection | All pyrox modules are `eqx.Module` subclasses. `eqx.tree_at` for immutable parameter injection. |
| **numpyro** | `numpyro.sample`, `numpyro.param`, distributions, inference | pyrox registers sample/param sites; NumPyro traces and runs inference. All AutoGuide families work out of the box. `GPPrior` is a native `numpyro.distributions.Distribution`. |
| **gaussx** | Solver protocol backend, structured operators, `MultivariateNormal`, Matheron sampling | `pyrox.gp.Solver.solve/log_det` delegates to `gaussx.ops.solve/logdet`; `CovarianceRepresentation` maps to gaussx operators; `PathwiseSampler` uses `gaussx.matheron_update`. |
| **lineax** | Base operator abstraction (`AbstractLinearOperator`) | Transitive via gaussx; pyrox doesn't import lineax directly. |
| **matfree** | SLQ logdet, Hutchinson trace, partial eig/SVD | Transitive via gaussx solvers (CG, BBMM). |
| **jax** | All computation, autodiff, vmap, scan | Foundation. |
| **xtremax** | GEV/GPD with GP-varying parameters | GP latents in extreme value models. |
| **einops** | Tensor reshaping in examples | Examples use einops for readable reshaping; not a library dependency. |

---

## Scope

### In Scope (`_core`)

- Equinox-NumPyro bridge (`PyroxModule`, `PyroxParam`, `PyroxSample`)
- GP-style `Parameterized` base class (register_param, set_prior, autoguide, mode switching)
- Per-call context cache preventing duplicate sample sites
- Dependent priors via callables
- Shared RFF spectral density samplers (multi-kernel, variational frequencies)

### In Scope (`nn`)

- Edward2-style probabilistic layers (dense, dropout, RFF, NCP)
- MC-Dropout layers for uncertainty estimation
- Spectral normalization for distance-aware uncertainty (SNGP)
- Extended random Fourier feature layers (orthogonal RFF, variational frequencies)

### In Scope (`gp`)

- GP primitives for NumPyro hierarchical models (`gp_sample`, `gp_factor`)
- Exact, sparse, and approximate GP inference
- Temporal GP via state-space / Kalman methods
- Gaussian expectation computation (sigma points, cubature, MC, exact)
- Structured variational guides for GP latents
- Inter-domain inducing features (VISH, VFF) -- spectral methods that preserve the full GP prior
- Pathwise posterior sampling via Matheron's rule

### Out of Scope

- Custom inference (MCMC, SVI, guides) -- NumPyro
- Neural network primitives -- Equinox
- Standalone GP regression (use GPJax for that)
- Weight-space / HSGP methods that reduce the GP to a linear model in feature space
- Data loading, preprocessing, visualization -- user code
- Multi-backend abstraction -- pure JAX/Equinox only
- Thin wrappers around NumPyro -- users call NumPyro inference directly

---

## Testing Strategy

| Category | What it tests | Submodule | Example |
|---|---|---|---|
| **Core contract** | PyroxModule registers sites correctly | `_core` | `pyrox_sample("w", dist)` creates a NumPyro sample site |
| **Context** | Per-call cache prevents duplicates | `_core` | Same param referenced twice yields one site in trace |
| **Parameterized** | register/prior/guide/mode switching | `_core` | `set_mode("guide")` uses learned posterior |
| **NN layers** | Each layer produces correct output shapes | `nn` | `DenseVariational(50)(x)` yields shape `(N, 50)` |
| **NN inference round-trip** | Layers work with MCMC and SVI | `nn` | NUTS runs, SVI converges on toy problem |
| **Kernel contract** | All kernels produce PSD Gram matrices | `gp` | `eigvals(K) >= 0` for all kernel types |
| **Solver correctness** | Solver results match dense Cholesky reference | `gp` | `CG.solve(K, y) ~ cholesky_solve(K, y)` |
| **GP posterior** | Predictions match analytical GP posterior | `gp` | `ConditionedGP.predict` vs dense formula |
| **Kalman** | Filter/smoother match dense GP on temporal data | `gp` | KalmanSolver result ~ CholeskySolver result |
| **Integrator accuracy** | Quadrature matches analytical integrals | `gp` | Gauss-Hermite on polynomial = exact |
| **NumPyro compat** | gp_sample / gp_factor work in MCMC and SVI | `gp` | NUTS runs, SVI converges on toy problem |
| **JAX transforms** | All components work under jit, vmap, grad | all | `jax.jit(model)(x)` works |

### Test Priorities

1. **NumPyro trace compatibility** -- sites register correctly, inference runs (all submodules)
2. **JAX transform compatibility** -- jit, vmap, grad all work (all submodules)
3. **Parameterized round-trip** -- register, prior, guide, mode switch (`_core`)
4. **NN layer correctness** -- output shapes, gradient flow, prior sampling (`nn`)
5. **Solver-representation correctness** -- all (Solver x Representation) pairs produce matching results (`gp`)
6. **Kalman = dense** -- temporal GP via Kalman matches exact GP on same data (`gp`)
7. **Kernel PSD** -- all kernels produce valid covariance matrices (`gp`)

---

## Non-goals

1. **No custom inference** -- the library never reimplements MCMC, SVI, or guide construction. NumPyro owns inference entirely.
2. **No NN primitives** -- Equinox provides layers; pyrox wraps them with probabilistic semantics.
3. **No standalone GP regression** -- pyrox.gp provides GP components for NumPyro models, not a standalone GP framework.
4. **No weight-space GP approximations** -- HSGP and similar methods that reduce GPs to linear models are out of scope.
5. **No data loading or preprocessing** -- operates on JAX arrays.
6. **No visualization** -- examples may plot, but the library ships no plotting code.
7. **No multi-backend abstraction** -- pure JAX/Equinox. No Flax, no PyTorch, no keras.ops.
8. **No thin wrappers around NumPyro** -- users call NumPyro inference directly. No syntactic sugar re-exports.

---

## Future Work

1. **Additional NN layer families** -- convolutional, recurrent, attention probabilistic layers (`nn`)
2. **Extended random features** -- multi-kernel spectral densities, variational frequencies (VSSGP), orthogonal RFF (`_core` + `nn`)
3. **Multi-output GPs** -- LMC, ICM, OILMM via latent-space inference + mixing (`gp`)
4. **Inter-domain inducing features** -- VISH (spherical harmonics), VFF (Fourier), Laplacian eigenfunctions as SVGP inducing variables (`gp`)
5. **Scalable Kronecker methods** -- for large spatiotemporal grids (`gp`)
6. **Parallel associative scan** -- GPU-accelerated Kalman for long time series (`gp`)

---

## Open Questions

1. **Registry cleanup** -- class-level registries keyed by `id(self)` risk memory leaks. Need a cleanup strategy (`_teardown()`, weak references, or context managers). Affects `_core.Parameterized`.
