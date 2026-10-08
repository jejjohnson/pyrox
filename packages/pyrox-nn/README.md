# <img src="../../docs/assets/icon-nn.svg" width="36" alt="" align="top"> pyrox-nn

Bayesian, spectral, and coordinate-encoding neural network layers on
top of [`pyrox`](../pyrox) and [`pyrox-gp`](../pyrox-gp): SIREN, MFN,
SNGP, deep VSSGP, heteroscedastic heads, BatchEnsemble/Rank-1 layers,
the Bayesian Neural Field (BNF), and the sklearn-style estimator API
(`pyrox_nn.api`) with its pandas preprocessing helpers
(`pyrox_nn.preprocessing`).

## Install

Not on PyPI yet; install from GitHub with uv (see the [root README](../../README.md#installation)):

```bash
uv add "pyrox-nn @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-nn"
# the BNF stack (pandas preprocessing + SGD-MAP/SVI inference) needs:
uv add "pyrox-nn[bnf] @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-nn"
```

## Layout

| Module | Purpose |
|--------|---------|
| `pyrox_nn` | Public API: Bayesian layer wrappers + geonnax re-exports |
| `pyrox_nn.api` | `BayesianNeuralFieldMAP` / estimator entry points |
| `pyrox_nn.preprocessing` | pandas → array preprocessing for the BNF stack |
