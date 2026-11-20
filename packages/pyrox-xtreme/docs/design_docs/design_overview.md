---
status: draft
version: 0.1.0
---

# xtremax Design Doc

**Extreme value modeling with JAX and NumPyro.**

## Structure

```
xtremax/
├── design_overview.md          # This file
├── vision.md              # Motivation, user stories, design principles, identity
├── architecture.md        # Layer stack, package layout, dependencies
├── boundaries.md          # Ownership, ecosystem, scope, testing, roadmap
├── api/
│   ├── api_overview.md    # Surface inventory, conventions, notation
│   ├── api_primitives.md  # Layer 0 — pure JAX functions (GEV, GPD, point process, max-stable math)
│   ├── api_components.md  # Layer 1 — NumPyro distributions, xarray utilities, plotting, simulations, datasets
│   └── api_models.md      # Layer 2 — model zoo (stationary/nonstationary/spatial GEV, POT, PP)
├── examples/
│   ├── examples_overview.md    # Index and reading order
│   ├── examples_primitives.md  # Layer 0 — pure JAX functions, return levels, gradients
│   ├── examples_components.md  # Layer 1 — NumPyro distributions, xarray extraction, plotting
│   ├── examples_models.md      # Layer 2 — model zoo workflows (MCMC, SVI)
│   └── examples_integration.md # Layer 3 — geo_toolz, ecosystem composition
└── decisions.md           # Design decisions with rationale
```

## Reading Order

1. **[vision.md](vision.md)** — understand the why
2. **[architecture.md](architecture.md)** — understand the layer stack
3. **[boundaries.md](boundaries.md)** — understand the scope
4. **[api/api_overview.md](api/api_overview.md)** — scan the surface
5. **[api/api_primitives.md](api/api_primitives.md)** → **[api_components.md](api/api_components.md)** → **[api_models.md](api/api_models.md)** — drill into detail
6. **[examples/](examples/examples_overview.md)** — see it in action
7. **[decisions.md](decisions.md)** — understand the tradeoffs
