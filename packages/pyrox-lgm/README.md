# pyrox-lgm

Latent Gaussian models (LGMs) for JAX: GMRF latent components, PC priors and
an `inla()` driver, built on [`pyrox`](../pyrox),
[gaussx](https://github.com/jejjohnson/gaussx) (sparse precision operators,
GMRF distributions, Laplace modes) and
[kernellib](https://github.com/jejjohnson/kernellib) (graphs and GMRF
structure matrices).

LGMs are written in **precision form** and live here rather than in
[`pyrox-gp`](../pyrox-gp), whose GPs are covariance-form; this package imports
nothing from pyrox-gp.

## Status

Scaffold only (P6 of the [`inla()` epic](https://github.com/jejjohnson/pyrox/issues/258)).
The modules are filled in by later phases:

| Module | Phase | Contents |
|---|---|---|
| `_components/` | P7 | temporal (RW1, RW2, AR1), areal (ICAR, BYM2), SPDE, generic and combinators |
| `_priors/_pc.py` | P7 | penalised-complexity priors |
| `_model.py`, `_inla.py`, `_result.py`, `_numpyro.py` | P8 | `LGM`, `inla()`, `INLAResult`, NumPyro faces |
| `_diagnostics.py`, `_formula.py` | P9 | diagnostics and the `f(...)` formula sugar |

## Install

```bash
uv add pyrox-lgm
# xarray-backed INLAResult marginals:
uv add "pyrox-lgm[xarray]"
```
