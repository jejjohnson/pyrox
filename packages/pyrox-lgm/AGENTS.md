# pyrox-lgm (`pyrox_lgm`) — agent rules

Latent Gaussian models in precision form: GMRF components, PC priors and an
`inla()` driver checked against R-INLA. The root [`AGENTS.md`](../../AGENTS.md)
applies too.

## Public layout

One facade, `pyrox_lgm`:

| Concept | Module |
|---|---|
| Components | `_components/` — `_temporal.py` (`IID`, `RW1`, `RW2`, `AR1`), `_areal.py` (`Besag`, `BYM2`, `CAR`, `Leroux`), `_spde.py` (`SPDE`), `_generic.py` (`Generic`), `_combinators.py` (`Kronecker`, `Replicate`), `_base.py` (`AbstractComponent`) |
| Priors | `_priors/_pc.py` — `PCPrecision`, `PCAR1Rho`, `PCBYM2Phi`, `PCMatern`, `StructureSpectrum` |
| Observations | `_likelihood.py` — `AbstractObservation`, `Gaussian`, `Poisson`, `Bernoulli`, `Binomial`, `NegativeBinomial` |
| Model and fitting | `_model.py` (`LGM`, `FixedEffects`), `_formula.py` (`f`), `_assembly.py`, `_inla.py` (`inla`), `_result.py` (`INLAResult`, `Summary`), `_diagnostics.py`, `_numpyro.py` |

## Contracts

- **Never import pyrox-gp** (`tests/test_import_guard.py`: an AST scan plus a
  subprocess import check). Precisions come from gaussx GMRF builders,
  graphs from kernellib, Lanczos from matfree, L-BFGS from optax.
- **Components are plain `eqx.Module`s**, not `PyroxModule`s: subclass
  `AbstractComponent` and provide `name` (static), `n_nodes`,
  `theta_spec() -> {key: (prior, transform)}` and `prior(theta, *,
  constraint=...)` returning a `gx.GaussianMRF` / `gx.IntrinsicGMRF`;
  override `n_index`, `assembly_precision`, `constraint_basis` or
  `projector` only when needed. Keyword-only `name=` and `*_prior=` with
  a `PCPrecision` default; exemplars `IID` (proper) and `RW1` (intrinsic) in
  `_temporal.py`.
- **Wiring a new component:** export it from `_components/__init__.py` and
  the package `__init__`; register it in `_MODELS` in `_formula.py`; extend
  `_inla._vb_layout` for a combinator; make sure `_assembly.full_coo`
  supports its precision (else it densifies); a θ-dependent intrinsic
  structure sets `include_normalizer=True`.
- **Keep precisions sparse.** Assembly builds a block-diagonal sparse `Q` on a
  fixed host pattern so the symbolic Cholesky is cached; a component that
  materialises its precision defeats it.
- **PC priors** are NumPyro `Distribution` subclasses (`arg_constraints`,
  `support`, `sample`, `@validate_sample log_prob`, `cdf` / `icdf` where
  analytic), calibrated against `scipy.integrate.quad` in
  `tests/test_pc_priors.py`; document the R-INLA counterpart in
  `tests/fixtures/rinla/make_fixtures.R`.
- **Hyperparameter keys** are `"<component>.<key>"`; `y`, `offset` and
  `n_trials` are reserved data names. NumPyro sites (`_numpyro.py`) are
  `"<component>_<key>"`.

## R-INLA fixtures

`tests/fixtures/rinla/make_fixtures.R` (R + INLA + fmesher + jsonlite;
`Rscript tests/fixtures/rinla/make_fixtures.R`) writes one JSON per case.
`tests/test_rinla_fixtures.py` rebuilds each case and compares (slow) with
per-case `TOLERANCES` whose measured values are recorded in comments. A new
case needs an R block, a `_case` branch and entries in `CASES` and
`TOLERANCES`; never loosen a tolerance without recording the new
measurement.

## Docs and tests

- Add `::: pyrox_lgm.Name` to `docs/api/lgm.md`.
- `uv run pytest --no-cov packages/pyrox-lgm/tests -m "not slow"`.
- Docstring `Examples:` blocks **run** (`tests/test_doctests.py`, ELLIPSIS;
  `_inla` is slow): keep them executable.
- `tests/test_public_api.py` checks every `__all__` name exists and none is
  duplicated.
