---
status: draft
version: 0.1.0
---

# xtremax Design Doc

**Extreme value modeling with JAX and NumPyro.**

## Structure

```
xtremax/
├── README.md              # This file
├── vision.md              # Motivation, user stories, design principles, identity
├── architecture.md        # Layer stack, package layout, dependencies
├── boundaries.md          # Ownership, ecosystem, scope, testing, roadmap
├── api/
│   ├── README.md          # Surface inventory, conventions, notation
│   ├── primitives.md      # Layer 0 — pure JAX functions (GEV, GPD, point process, max-stable math)
│   ├── components.md      # Layer 1 — NumPyro distributions, xarray utilities, plotting, simulations, datasets
│   └── models.md          # Layer 2 — model zoo (stationary/nonstationary/spatial GEV, POT, PP)
├── examples/
│   ├── README.md          # Index and reading order
│   ├── primitives.md      # Layer 0 — pure JAX functions, return levels, gradients
│   ├── components.md      # Layer 1 — NumPyro distributions, xarray extraction, plotting
│   ├── models.md          # Layer 2 — model zoo workflows (MCMC, SVI)
│   └── integration.md     # Layer 3 — geo_toolz, ecosystem composition
└── decisions.md           # Design decisions with rationale
```

## Reading Order

1. **[vision.md](vision.md)** — understand the why
2. **[architecture.md](architecture.md)** — understand the layer stack
3. **[boundaries.md](boundaries.md)** — understand the scope
4. **[api/README.md](api/README.md)** — scan the surface
5. **[api/primitives.md](api/primitives.md)** → **[components.md](api/components.md)** → **[models.md](api/models.md)** — drill into detail
6. **[examples/](examples/)** — see it in action
7. **[decisions.md](decisions.md)** — understand the tradeoffs
