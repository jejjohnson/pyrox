# <img src="../../docs/assets/icon-gp.svg" width="36" alt="" align="top"> pyrox-gp

Gaussian process building blocks on top of
[`pyrox`](../pyrox): kernels and kernel protocols, variational guides,
likelihoods, exact / sparse / Markov GP models, multi-output
structures, and pathwise sampling.

## Install

Not on PyPI yet; install from GitHub with uv (see the [root README](../../README.md#installation)):

```bash
uv add "pyrox-gp @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-gp"
```

## Layout

| Module | Purpose |
|--------|---------|
| `pyrox_gp` | Public API: `GPPrior`, `ConditionedGP`, `SparseGPPrior`, kernels, guides, likelihoods, … |
| `kernellib.functional` | Pure kernel functions (closed-form math primitives), hosted upstream in kernellib |
| `pyrox_gp._basis` | RFF prior draws for pyrox kernels |
