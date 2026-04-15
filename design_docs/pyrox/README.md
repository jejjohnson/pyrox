---
status: draft
version: 0.1.0
---

# pyrox Design Doc

**Probabilistic modeling with Equinox and NumPyro: Bayesian neural networks, Gaussian processes, and composable GP building blocks.**

*Merges the former pyrox-nn (Bayesian deep learning) and pyrox-gp (GP building blocks) into a single package.*

## Structure

```
pyrox/
├── README.md              # This file
├── vision.md              # Motivation, user stories, design principles, identity
├── architecture.md        # Package layout, layer stacks, dependency diagram
├── boundaries.md          # Ownership, ecosystem, scope, testing, roadmap
├── decisions.md           # Design decisions with rationale
├── api/
│   ├── README.md          # Surface inventory, conventions, imports
│   ├── core.md            # Core — PyroxModule, PyroxParam, PyroxSample, Parameterized
│   ├── nn/
│   │   └── layers.md      # NN — probabilistic layers (dense, dropout, RFF, NCP)
│   └── gp/
│       ├── primitives.md  # GP Layer 0 — pure JAX: kernel eval, covariance math, quadrature
│       ├── components.md  # GP Layer 1 — protocols: Kernel, Solver, InferenceStrategy, Integrator, Guide
│       ├── models.md      # GP Layer 2 — entry points: gp_sample, gp_factor, GPPrior, ConditionedGP
│       ├── moments.md     # GP detail — core subsystem (kernels, solvers, guides, representations)
│       ├── state_space.md # GP detail — temporal subsystem (Markovian GPs, Kalman, Bayes-Newton)
│       └── integration.md # GP detail — integrator subsystem (Gaussian expectations, quadrature)
├── examples/
│   ├── README.md          # Index and reading order
│   ├── core.md            # Core — PyroxModule and tree_at patterns
│   ├── integration.md     # Ecosystem — gaussx, xtremax, somax composition
│   ├── nn/
│   │   ├── layers.md      # NN — Bayesian layers, RFF, NCP
│   │   ├── models.md      # NN — composition patterns, inference methods, model progression
│   │   ├── regression_masterclass_eqx.py   # Tutorial — 9-model regression (Equinox + einops)
│   │   └── regression_masterclass_eqx.md   # Full narrative tutorial
│   └── gp/
│       ├── primitives.md          # GP Layer 0 — kernel evaluation, covariance construction
│       ├── components.md          # GP Layer 1 — solvers, strategies, guides, integrators
│       ├── models.md              # GP Layer 2 — gp_sample, gp_factor, full GP workflows
│       ├── moments.md             # GP detail — 16+ core GP examples
│       ├── state_space.md         # GP detail — 10 temporal examples
│       ├── integration_detail.md  # GP detail — 10 integrator examples
│       ├── vgp_numpyro.py         # Tutorial — Full VGP with 5 guide families
│       └── svgp_numpyro.py        # Tutorial — Sparse VGP with 6 guide families
├── features/
│   ├── nn/
│   │   ├── layers_conv_rnn.md     # Gap: Conv, RNN, attention probabilistic layers
│   │   ├── spectral_norm.md       # Gap: SNGP-style spectral normalization
│   │   └── random_features.md     # Gap: SSGP, VSSGP, orthogonal RFF
│   └── gp/
│       ├── sde_kernels.md         # Gap: SDE kernel representations
│       ├── inference_strategies.md # Gap: concrete InferenceStrategy implementations
│       ├── variational_families.md # Gap: Guide implementations
│       ├── likelihoods.md         # Gap: likelihood implementations
│       ├── models.md              # Gap: model patterns (Deep GP, GPLVM, etc.)
│       ├── inducing_features.md   # Gap: VISH, VFF, Laplacian eigenfunction features
│       ├── pathwise_sampling.md   # Feature: pathwise posterior sampling
│       ├── multi_output.md        # Gap: multi-output GP (LMC, ICM, OILMM)
│       ├── gpflow.md              # Gap: novel GPflow components
│       ├── logfalkon.md           # Gap: LogFalkon / GSC-Falkon
│       └── metrics.md             # Gap: GP evaluation metrics
└── research/
    ├── README.md          # Index
    ├── core.py            # Reference: PyroxModule, Parameterized, layers
    └── gp.py              # Reference: Kernel, Solver, GPPrior, guides
```

## Reading Order

1. **[vision.md](vision.md)** — understand the why
2. **[architecture.md](architecture.md)** — understand the package layout and layer stacks
3. **[boundaries.md](boundaries.md)** — understand the scope
4. **[api/README.md](api/README.md)** — scan the surface
5. **[api/core.md](api/core.md)** — PyroxModule, Parameterized (shared foundation)
6. **[api/nn/layers.md](api/nn/layers.md)** — probabilistic neural network layers
7. **[api/gp/primitives.md](api/gp/primitives.md)** → **[components.md](api/gp/components.md)** → **[models.md](api/gp/models.md)** — GP layer-by-layer
8. **[api/gp/moments.md](api/gp/moments.md)** → **[state_space.md](api/gp/state_space.md)** → **[integration.md](api/gp/integration.md)** — GP subsystem deep-dives
9. **[examples/](examples/)** — see it in action
10. **[decisions.md](decisions.md)** — understand the tradeoffs
11. **[features/](features/)** — gap analysis and roadmap
12. **[research/](research/)** — reference Python implementations
