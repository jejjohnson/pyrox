---
status: draft
version: 0.1.0
---

# Vision

## One-Liner

> **pyrox** is a JAX library for probabilistic modeling with Equinox and NumPyro — Bayesian neural network layers, GP building blocks for hierarchical models, and a shared bridge that makes Equinox modules probabilistic.

---

## Motivation

Modern probabilistic modeling in JAX splits across two worlds. On the neural network side, researchers want calibrated uncertainty from Equinox models but face boilerplate when lifting frozen PyTrees into NumPyro's effectful tracing. On the Gaussian process side, existing libraries are either standalone (GPyTorch, GPJax) or tightly coupled to a single inference engine, leaving researchers who need GP components inside hierarchical Bayesian models without good tooling.

These two worlds share the same foundation: Equinox for immutable modules, NumPyro for inference, and the awkward bridge between them. pyrox unifies them.

**pyrox provides three things:**

1. **`pyrox._core`** — The shared bridge. `PyroxModule`, `PyroxParam`, `PyroxSample`, and `Parameterized` base classes that inject NumPyro-managed parameters and samples into Equinox module forward passes, with support for dependent priors, constraint transforms, and mode switching (model/guide).

2. **`pyrox.nn`** — Bayesian neural network layers. Edward2-style drop-in probabilistic replacements for standard Equinox layers: variational dense, flipout, MC-Dropout, random Fourier features, noise contrastive priors, and GP layers.

3. **`pyrox.gp`** — GP building blocks for hierarchical models. Composable GP primitives that plug into NumPyro's probabilistic programming framework: kernels, solvers, integrators, inference strategies, and multi-output support. The user owns the model; pyrox.gp owns the GP math.

---

## User Stories

**BDL researcher** — "I want to put priors on my Equinox MLP weights and run NUTS. I shouldn't need to rewrite my module or build a custom inference loop — just declare priors and call NumPyro."

**GP practitioner** — "I have an RBF kernel with learnable lengthscale and variance. I want `set_prior('lengthscale', LogNormal(...))` and have SVI learn the posterior. The GP kernel should be an Equinox module, not a NumPyro-specific construct."

**Hierarchical modeler** — "I'm building a spatiotemporal model where a GP prior sits inside a larger Bayesian hierarchy with changepoints and covariate effects. I need GP components that act as NumPyro distribution primitives, not a standalone GP library that fights my model structure."

**Uncertainty engineer** — "I want to compare MAP, MC-Dropout, variational BNN, and deep GP on the same regression problem. Each should use the same Equinox modules with different probabilistic wrappers."

**Student / newcomer** — "I want `DenseVariational(50)` as a drop-in replacement for `eqx.nn.Linear(50)` that gives me uncertainty. The inference should just work with AutoNormal."

---

## Design Principles

### Shared (all of pyrox)

| # | Principle | Meaning |
|---|-----------|---------|
| 1 | **Equinox-first** | All modules are `eqx.Module` subclasses. Immutable PyTrees, functional transforms (`jax.vmap`, `jax.jit`), and `eqx.tree_at` for parameter injection. No mutable state hacks. |
| 2 | **NumPyro for inference** | The library never reimplements inference. MCMC, SVI, Predictive, AutoGuide — all from NumPyro. pyrox makes Equinox modules compatible with NumPyro's tracing, nothing more. |
| 3 | **Explicit over magic** | Pyro (PyTorch) uses `__setattr__`/`__getattr__` descriptor interception. Equinox modules are frozen dataclasses — that trick doesn't work. Instead, pyrox uses explicit method calls (`self.pyrox_param()`, `self.pyrox_sample()`) and class-level registries. The wiring is visible. |
| 4 | **JAX all the way down** | Pure JAX. No framework abstractions beyond Equinox and NumPyro. |

### Additional principles for pyrox.nn

| # | Principle | Meaning |
|---|-----------|---------|
| 5 | **Edward2-style layers** | Probabilistic layer variants (variational dense, MC-Dropout, etc.) are organized by family. They compose like normal Equinox layers but carry uncertainty through the forward pass. |
| 6 | **Parameterized for GPs** | The `Parameterized` base class provides `register_param()`, `set_prior()`, `autoguide()`, and model/guide mode switching. Essential for GP kernels and any module where parameters need priors and variational posteriors. |

### Additional principles for pyrox.gp

| # | Principle | Meaning |
|---|-----------|---------|
| 7 | **GP as distribution primitive** | GPs are `numpyro.Distribution` objects, not standalone models. |
| 8 | **Protocol-driven** | Kernels, solvers, integrators, inference strategies, and guides are all protocols. Swap any component independently. |
| 9 | **Composable** | Components snap together — swap kernel, solver, or integrator independently. |
| 10 | **Math-first** | Software objects mirror the math (covariance, marginal likelihood, posterior moments). |

---

## Identity

### What pyrox IS

**Core (`pyrox._core`)**

- `PyroxModule` base class bridging Equinox <-> NumPyro tracing
- `PyroxParam` and `PyroxSample` descriptors for parameter/sample declaration
- `Parameterized` base class with `register_param()`, `set_prior()`, `autoguide()`, mode switching
- Per-call `_Context` cache to prevent duplicate sample sites
- Dependent priors via callables (`lambda self_: dist.LogNormal(self_.loc, 0.1)`)
- Works with NumPyro MCMC (NUTS), SVI, Predictive, and AutoGuide family

**Neural networks (`pyrox.nn`)**

- Edward2-style layers: variational dense, flipout, MC-Dropout, RFF, NCP, GP layers
- Drop-in Bayesian replacements for standard Equinox layers
- Model progression patterns: linear -> BNN -> deep GP

**Gaussian processes (`pyrox.gp`)**

- GP building blocks for NumPyro hierarchical models
- Kernel, Solver, GPPrior, Guide, InferenceStrategy, Integrator protocols
- Functional entry points: `gp_sample` (latent $f$ as a NumPyro site), `gp_factor` (collapsed marginal likelihood as a NumPyro factor)
- Temporal GP inference via state-space methods (Kalman, Bayes-Newton)
- Gaussian expectation computation (sigma points, cubature, Taylor, MC)
- Exact, sparse, and approximate GP inference
- Inter-domain inducing features (VISH, VFF) — spectral methods that preserve the full GP prior
- Multi-output GPs via latent-space inference + mixing (LMC, ICM, OILMM)
- Pathwise posterior sampling via Matheron's rule (efficient function-valued posterior draws)

### What pyrox is NOT

| Not this | Use instead |
|----------|-------------|
| Custom inference (MCMC, SVI, guides) | NumPyro |
| Neural network primitives (layers, activations) | Equinox |
| Probability distributions | `numpyro.distributions` |
| Standalone GP regression library | GPJax, GPyTorch |
| Structured linear algebra (Kronecker, block-diag, logdet) | gaussx |
| Weight-space / feature-approximation GP methods (HSGP, RFF-as-solver) | Not in scope |
| Data loading or preprocessing | User code / geo_toolz |
| Visualization | matplotlib / user code |
| Multi-backend abstraction | JAX only (Equinox only) |

---

## Core Components

### pyrox._core

| Component | What it is | What it provides |
|-----------|------------|------------------|
| **PyroxModule** | Equinox module base class | NumPyro-compatible `__call__` with sample site injection |
| **PyroxParam** | Parameter descriptor | Constrained parameter registered as a NumPyro `param` site |
| **PyroxSample** | Sample descriptor | Stochastic parameter registered as a NumPyro `sample` site |
| **Parameterized** | Module base class for GP-style components | `register_param()`, `set_prior()`, `autoguide()`, mode switching |
| **_Context** | Per-call cache | Prevents duplicate sample sites across shared submodules |

### pyrox.nn

| Component | What it is | What it provides |
|-----------|------------|------------------|
| **DenseVariational** | Bayesian linear layer | Weight and bias with variational posteriors |
| **DenseFlipout** | Flipout linear layer | Low-variance gradient estimates via sign perturbation |
| **MCDropout** | MC-Dropout layer | Stochastic dropout at test time |
| **RandomFourierFeatures** | RFF layer | Approximate kernel feature map |
| **NoiseContrastivePrior** | NCP layer | Data-dependent prior for BNN regularization |

### pyrox.gp

| Component | What it is | What it provides |
|-----------|------------|------------------|
| **Kernel** | Covariance function $k(\cdot, \cdot)$ | A `CovarianceRepresentation` |
| **Solver** | Linear algebra backend | `solve(K, y)` and `log_det(K)` |
| **GPPrior** | `numpyro.Distribution` over $f$ | `sample(key)` and `log_prob(f)` |
| **gp_sample** | Functional entry point | Latent $f$ as a NumPyro site |
| **gp_factor** | Functional entry point | Collapsed marginal likelihood as a NumPyro factor |
| **Guide** | Structured $q(f)$ | Variational family respecting GP geometry |
| **InferenceStrategy** | Approximate inference | Laplace, EP, VI, PL via natural parameters |
| **Integrator** | Gaussian expectations | Sigma points, cubature, Taylor, MC, exact RBF |
| **InducingFeatures** | Inter-domain inducing variables | `K_uu(kernel)`, `k_u(X, kernel)` for VISH/VFF |
| **MultiOutputKernel** | Multi-output kernel (latent + mixing) | `Kgg(X1, X2)`, `mixing_matrix()` |
| **PathwiseSampler** | Efficient posterior function samples | `sample_paths(key, n_paths)` -> callable |

---

## Migration Context

**Replaces:** The separate `pyrox-nn` and `pyrox-gp` design docs and packages. Both shared the same Equinox + NumPyro foundation, making a unified package the natural structure. The `_core` module extracts the shared bridge; `nn` and `gp` become submodules with clear ownership boundaries.

**External inspiration:**

| Tool | What pyrox takes from it |
|------|--------------------------|
| Pyro (PyTorch) | `PyroxModule`, `PyroxParam`, `PyroxSample` naming and semantics |
| TensorFlow Probability (Edward2) | Edward2-style probabilistic layers organized by family |
| GPflow | `Parameterized` base class with `set_prior()`, mode switching |
| Aboleth | Regression masterclass model progression (linear -> deep GP) |

---

## Connection to Ecosystem

```
                    ┌──────────────┐
                    │   equinox    │  Neural network modules (immutable PyTrees)
                    │   numpyro    │  Probabilistic programming, inference
                    └──────┬───────┘
                           │ bridges
                    ┌──────▼───────┐
                    │ pyrox._core  │  PyroxModule, Parameterized, PyroxParam, PyroxSample
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
       ┌──────▼──────┐          ┌──────▼──────┐
       │  pyrox.nn   │          │  pyrox.gp   │
       │  Bayesian   │          │  GP building│
       │  layers     │          │  blocks     │
       └─────────────┘          └──────┬──────┘
                                       │ uses
                                ┌──────▼──────┐
                                │   gaussx    │
                                │  structured │
                                │  linear alg │
                                └─────────────┘

Depends on: jax, equinox, numpyro
Optional:   gaussx (GP solvers), einops (examples)
```
