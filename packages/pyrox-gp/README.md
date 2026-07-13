# pyrox-gp

Gaussian process building blocks on top of
[`pyrox`](../pyrox): kernels and kernel protocols, variational guides,
likelihoods, exact / sparse / Markov GP models, multi-output
structures, pathwise sampling, and the shared spectral basis helpers
(`pyrox_gp._basis`).

## Install

```bash
uv add pyrox-gp
```

## Layout

| Module | Purpose |
|--------|---------|
| `pyrox_gp` | Public API: `GPPrior`, `ConditionedGP`, `SparseGPPrior`, kernels, guides, likelihoods, … |
| `pyrox_gp._src.kernels` | Pure kernel functions (closed-form math primitives) |
| `pyrox_gp._basis` | Kernel spectral densities + RFF prior draws shared with `pyrox-nn` |
