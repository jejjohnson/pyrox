# <img src="../../docs/assets/icon.svg" width="36" alt="" align="top"> pyrox

Core primitives for probabilistic modeling with Equinox and NumPyro:
the Equinox-to-NumPyro bridge (`PyroxModule`, `PyroxParam`,
`PyroxSample`, `Parameterized`) and ensemble-of-MAP / ensemble-of-VI
inference primitives (`pyrox.inference`).

The GP building blocks live in
[`pyrox-gp`](../pyrox-gp) and the Bayesian NN layers in
[`pyrox-nn`](../pyrox-nn); both build on this package.

## Install

Not on PyPI yet; install from GitHub with uv (see the [root README](../../README.md#installation)):

```bash
uv add "pyrox @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox"
# ensemble-of-MAP inference needs optax:
uv add "pyrox[optax] @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox"
```

## Layout

| Module | Purpose |
|--------|---------|
| `pyrox._core` | `PyroxModule`, `pyrox_method`, `PyroxParam`, `PyroxSample`, `Parameterized` |
| `pyrox.inference` | `ensemble_map`, `ensemble_vi`, `EnsembleMAP`, `EnsembleVI`, functional primitives |
