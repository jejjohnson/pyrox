---
status: draft
version: 0.1.0
---

# xtremax — Examples

Usage patterns organized by API layer.

## Structure

```
examples/
├── README.md              # This file
├── primitives.md          # Layer 0 — distribution usage, return levels
├── components.md          # Layer 1 — xarray extraction, threshold selection
├── models.md              # Layer 2 — model zoo workflows (MCMC, SVI)
└── integration.md         # Layer 3 — geo_toolz, ecosystem composition
```

## Reading Order

1. **[primitives.md](primitives.md)** — L0: GEV/GPD distributions, return levels
2. **[components.md](components.md)** — L1: block maxima, threshold exceedances, declustering
3. **[models.md](models.md)** — L2: stationary/nonstationary/spatial GEV, POT
4. **[integration.md](integration.md)** — L3: geo_toolz preprocessing, end-to-end pipelines
