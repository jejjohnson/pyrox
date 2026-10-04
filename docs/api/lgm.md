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

## Penalised-complexity priors

Hyperpriors calibrated by a tail statement (Simpson et al., 2017). Each is a
NumPyro distribution, usable under NUTS as well as in `inla()`.

::: pyrox_lgm.PCPrecision

::: pyrox_lgm.PCAR1Rho

::: pyrox_lgm.PCBYM2Phi

::: pyrox_lgm.PCMatern

### Structure spectra

`PCBYM2Phi` needs the spectrum of the scaled structure's generalised inverse,
computed once per graph: exactly for a grid (`gaussx.KroneckerSum`) or
`n ≤ 5000`, else by deflated stochastic Lanczos quadrature.

::: pyrox_lgm.structure_spectrum

::: pyrox_lgm.StructureSpectrum

## Components

Each component is a GMRF over its own nodes with hyperpriors and a projector
to observations. `prior(theta)` gives the gaussx distribution that `inla()`
uses; `sample(index)` is the NumPyro face, for NUTS.

::: pyrox_lgm.AbstractComponent

### Temporal and unstructured

::: pyrox_lgm.IID

::: pyrox_lgm.RW1

::: pyrox_lgm.RW2

::: pyrox_lgm.AR1

### Generic

::: pyrox_lgm.Generic

### Areal

On a kernellib graph (from edges, polygons' contiguity, or a grid).

::: pyrox_lgm.Besag

::: pyrox_lgm.BYM2

::: pyrox_lgm.CAR

::: pyrox_lgm.Leroux

### SPDE (Matérn)

::: pyrox_lgm.SPDE

### Combinators

::: pyrox_lgm.Kronecker

::: pyrox_lgm.Replicate

## Models and inference

An `LGM` assembles components, fixed effects and an observation model;
`inla()` integrates over the hyperparameters (Rue, Martino & Chopin, 2009).

::: pyrox_lgm.LGM

::: pyrox_lgm.FixedEffects

::: pyrox_lgm.inla

::: pyrox_lgm.INLAResult

::: pyrox_lgm.Summary

### Formula sugar

`f(column, model, ...)` builds a component named after the data column that
indexes it, as R-INLA's `f()` does.

::: pyrox_lgm.f

### Diagnostics

CPO, PIT, WAIC and DIC from one fit, with no refits.

::: pyrox_lgm.diagnostics

::: pyrox_lgm.Diagnostics

### Observation models

::: pyrox_lgm.Gaussian

::: pyrox_lgm.Poisson

::: pyrox_lgm.Bernoulli

::: pyrox_lgm.Binomial

::: pyrox_lgm.NegativeBinomial
