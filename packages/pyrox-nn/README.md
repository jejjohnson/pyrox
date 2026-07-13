# pyrox-nn

Bayesian, spectral, and coordinate-encoding neural network layers on
top of [`pyrox`](../pyrox) and [`pyrox-gp`](../pyrox-gp): SIREN, MFN,
SNGP, deep VSSGP, heteroscedastic heads, BatchEnsemble/Rank-1 layers,
the Bayesian Neural Field (BNF), and the sklearn-style estimator API
(`pyrox_nn.api`) with its pandas preprocessing helpers
(`pyrox_nn.preprocessing`).

## Install

```bash
uv add pyrox-nn
# the BNF stack (pandas preprocessing + SGD-MAP/SVI inference) needs:
uv add "pyrox-nn[bnf]"
```

## Layout

| Module | Purpose |
|--------|---------|
| `pyrox_nn` | Public API: Bayesian layer wrappers + geonnax re-exports |
| `pyrox_nn.api` | `BayesianNeuralFieldMAP` / estimator entry points |
| `pyrox_nn.preprocessing` | pandas → array preprocessing for the BNF stack |
