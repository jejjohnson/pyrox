---
status: draft
version: 0.1.0
---

# pyrox — Research

Reference implementations of all pyrox protocols and patterns.

## Files

| File | Contents |
|---|---|
| `core.py` | Minimal reference implementation: `PyroxModule`, `PyroxParam`, `PyroxSample`, `Parameterized`, `DenseVariational`, `MCDropout`, `RandomFourierFeatures`, `RBFKernel` (Parameterized) |
| `gp.py` | Reference implementation: Kernel, Solver, GPPrior, gp_sample, gp_factor, guides, variational GP workflows |

## Purpose

- Executable specification of the design doc
- Test bed for design exploration
- NOT production code — no error handling, no edge cases

## How This Relates to the Design Docs

| File | Informs | Key content |
|---|---|---|
| `core.py` | `api/core.md`, `api/nn.md` | Working implementations of PyroxModule bridge, Parameterized base class, and probabilistic layers |
| `gp.py` | `api/gp_components.md`, `api/gp_models.md` | Working implementations of Kernel, Solver, InferenceStrategy, Integrator, Guide, GPPrior, gp_sample, gp_factor |
