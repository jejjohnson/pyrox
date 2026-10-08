# <img src="../../docs/assets/icon-lgm.svg" width="36" alt="" align="top"> pyrox-lgm

Latent Gaussian models (LGMs) for JAX: GMRF latent components, PC priors and an `inla()` driver.
It builds on [`pyrox`](../pyrox), [gaussx](https://github.com/jejjohnson/gaussx) (sparse precision operators, GMRF distributions, Laplace modes) and [kernellib](https://github.com/jejjohnson/kernellib) (graphs and GMRF structure matrices).

LGMs are written in **precision form** and live here rather than in [`pyrox-gp`](../pyrox-gp), whose GPs are covariance-form.
This package imports nothing from pyrox-gp.

## What's inside

| Area | Public names |
|---|---|
| Temporal components | `RW1`, `RW2`, `AR1`, `IID` |
| Areal components | `Besag`, `BYM2`, `CAR`, `Leroux` |
| Spatial and generic components | `SPDE`, `Generic`, and the combinators `Kronecker`, `Replicate` |
| Penalised-complexity priors | `PCPrecision`, `PCMatern`, `PCAR1Rho`, `PCBYM2Phi`, `structure_spectrum` |
| Observations | `Gaussian`, `Bernoulli`, `Binomial`, `Poisson`, `NegativeBinomial` |
| Model and inference | `LGM`, `FixedEffects`, `inla()`, `INLAResult`, `Summary` |
| Diagnostics and formulas | `diagnostics`, `Diagnostics`, the `f(...)` formula sugar |

The [`lgm_mcmc_inla`](https://jejjohnson.github.io/pyrox/notebooks/lgm_mcmc_inla/) notebook compares `inla()` against NUTS on the same model.

## Install

pyrox-lgm is not on PyPI yet; install it from GitHub with uv:

```bash
uv add "pyrox-lgm @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-lgm"
# xarray-backed INLAResult marginals:
uv add "pyrox-lgm[xarray] @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-lgm"
```

Until the next kernellib release, the project also needs the gaussx override described in the [root README](../../README.md#installation).
