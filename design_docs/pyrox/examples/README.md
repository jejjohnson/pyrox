---
status: draft
version: 0.1.0
---

# pyrox -- Examples

Usage patterns organized by API layer, plus detailed subsystem examples and full end-to-end tutorials.

pyrox merges the former pyrox-nn and pyrox-gp into a single package with three submodules: `pyrox._core`, `pyrox.nn`, and `pyrox.gp`.

## Structure

```
examples/
├── README.md                          # This file
├── core.md                            # Core — PyroxModule and tree_at patterns
├── integration.md                     # Ecosystem — gaussx, xtremax composition
├── nn/
│   ├── layers.md                      # NN — Bayesian layers, RFF, NCP
│   ├── models.md                      # NN — composition patterns (3 patterns, SSGP, SNGP, 9-model progression)
│   ├── regression_masterclass_eqx.py  # Tutorial — 9-model regression
│   └── regression_masterclass_eqx.md  # Full narrative tutorial
└── gp/
    ├── primitives.md                  # GP L0 — kernel evaluation, covariance math
    ├── components.md                  # GP L1 — solvers, strategies, guides, integrators
    ├── models.md                      # GP L2 — gp_sample, gp_factor, full GP workflows
    ├── moments.md                     # GP detail — 16+ core GP examples
    ├── state_space.md                 # GP detail — 10 temporal examples
    ├── integration_detail.md          # GP detail — 10 integrator examples
    ├── vgp_numpyro.py                 # Tutorial — Full VGP with 5 guide families
    └── svgp_numpyro.py                # Tutorial — Sparse VGP with 6 guide families
```

## Reading Order

**By layer** (recommended for first read):

1. **[core.md](core.md)** -- L0: how to bridge Equinox modules with NumPyro (`pyrox._core`)
2. **[nn/layers.md](nn/layers.md)** -- L1: how to use probabilistic NN layers (`pyrox.nn`)
3. **[nn/models.md](nn/models.md)** -- L2: composition patterns, model progression, SSGP, SNGP
4. **[gp/primitives.md](gp/primitives.md)** -- GP L0: pure kernel functions, covariance math
5. **[gp/components.md](gp/components.md)** -- GP L1: solvers, strategies, guides, integrators
6. **[gp/models.md](gp/models.md)** -- GP L2: full GP workflows with `gp_sample` and `gp_factor`
7. **[integration.md](integration.md)** -- ecosystem composition: gaussx, xtremax, somax, deep kernel learning

**Reference implementations** (richly annotated .py tutorials):

- **[gp/vgp_numpyro.py](gp/vgp_numpyro.py)** -- Full VGP (N latent variables, O(N^3)): 5 guide families (delta, mean-field, low-rank, full-rank, flow), whitened parameterization, extensive equation tracking
- **[gp/svgp_numpyro.py](gp/svgp_numpyro.py)** -- Sparse VGP (M inducing, O(NM^2)): 6 guide families (+ orthogonal decoupled), projection/unwhitening sugar, all covariance structures

**By subsystem** (deep dives):

- **[gp/moments.md](gp/moments.md)** -- 16+ examples: regression, BHMs, sparse, deep kernel, classification, ...
- **[gp/state_space.md](gp/state_space.md)** -- 10 examples: Kalman, streaming, changepoints, temporal extremes, ...
- **[gp/integration_detail.md](gp/integration_detail.md)** -- 10 examples: EKS, CKS, PILCO, BO, custom integrators, ...

**Full tutorial**:

- **[nn/regression_masterclass_eqx.md](nn/regression_masterclass_eqx.md)** -- 9-model regression tutorial (Equinox + einops, 900+ lines)
