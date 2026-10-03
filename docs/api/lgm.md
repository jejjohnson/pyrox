# LGM API

`pyrox_lgm` holds latent Gaussian models (LGMs) in **precision form**:
GMRF latent components, penalised-complexity priors and the `inla()` driver,
built on gaussx's sparse precision operators and Laplace modes and
kernellib's graph and GMRF structure matrices. GPs in covariance form stay in
[pyrox-gp](gp.md); `pyrox_lgm` imports nothing from it.

This page is a stub: the package is a scaffold (P6 of the
[`inla()` epic](https://github.com/jejjohnson/pyrox/issues/258)), and the
API lands with its phases.

| Phase | Adds |
|---|---|
| P7 | Components (RW1, RW2, AR1, ICAR, BYM2, SPDE, generic, combinators) and PC priors |
| P8 | `LGM`, `inla()` and `INLAResult` |
| P9 | `f(...)` formula sugar, diagnostics, simplified Laplace, MCMC-INLA hybrid |
