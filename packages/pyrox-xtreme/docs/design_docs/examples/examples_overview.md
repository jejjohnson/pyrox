---
status: draft
version: 0.1.0
---

# xtremax — Examples

Usage patterns organized by API layer.

## Structure

```
examples/
├── examples_overview.md     # This file
├── examples_primitives.md   # Layer 0 — distribution usage, return levels
├── examples_components.md   # Layer 1 — xarray extraction, threshold selection
├── examples_models.md       # Layer 2 — model zoo workflows (MCMC, SVI)
└── examples_integration.md  # Layer 3 — geo_toolz, ecosystem composition
```

## Reading Order

1. **[examples_primitives.md](examples_primitives.md)** — L0: GEV/GPD distributions, return levels
2. **[examples_components.md](examples_components.md)** — L1: block maxima, threshold exceedances, declustering
3. **[examples_models.md](examples_models.md)** — L2: stationary/nonstationary/spatial GEV, POT
4. **[examples_integration.md](examples_integration.md)** — L3: geo_toolz preprocessing, end-to-end pipelines
