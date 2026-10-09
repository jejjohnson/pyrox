---
name: add-lgm-component
description: Add a latent-Gaussian-model building block to pyrox-lgm — a GMRF component (temporal, areal, SPDE, combinator), a penalised-complexity prior, or an observation model — wired into f(), assembly and inla(), with doctests and, where R-INLA has a counterpart, a reference fixture. Use when asked to add, port or implement an LGM / INLA component, prior or likelihood in packages/pyrox-lgm.
---

# Add an LGM component, PC prior or observation model (pyrox-lgm)

Read `packages/pyrox-lgm/AGENTS.md` first. pyrox-lgm never imports pyrox-gp.

## 1. Make sure it does not exist yet

Search `docs/capabilities.md` for `pyrox_lgm` and for the gaussx GMRF
builders (`iid_precision`, `rw1_structure`, `rw2_structure`,
`ar1_precision`, `besag_structure`, `bym2_precision`, `spde_precision`,
`fem_matrices`, …). A new precision structure belongs in gaussx; the
component here wraps it with hyperparameters and constraints.

## 2a. A component (`src/pyrox_lgm/_components/_<family>.py`)

- Subclass `AbstractComponent` (an `eqx.Module`, not a `PyroxModule`):
  `name` and `n` as `eqx.field(static=True)`, a custom `__init__` with
  keyword-only `name=` and `*_prior=None` defaulting to `PCPrecision(1,
  0.01)` (see `_tau_prior`).
- Provide `n_nodes`, `theta_spec() -> {key: (prior, transform)}` (transform
  from `default_transform`) and `prior(theta, *, constraint="hard" | "soft"
  | "none", soft_constraint_scale=1e-3)` returning `gx.GaussianMRF` /
  `gx.IntrinsicGMRF`. Override `n_index`, `assembly_precision` (keep it a
  `SparseOperator`: `scale_operator`), `constraint_basis` or `projector` only
  when needed.
- Exemplars: `IID` (proper) and `_IntrinsicWalk` / `RW1` (intrinsic) in
  `_temporal.py`.
- Wire it up: export from `_components/__init__.py` and
  `src/pyrox_lgm/__init__.py`; add a row to `_MODELS` in `_formula.py`
  (`(ctor, "n" | "graph" | "none", {hyper: kwarg})`); extend
  `_inla._vb_layout` for a combinator; check `_assembly.full_coo` supports
  its precision (otherwise it is densified); a θ-dependent intrinsic
  structure sets `include_normalizer=True`; BYM2 cannot be a Kronecker /
  Replicate factor.

## 2b. A PC prior (`src/pyrox_lgm/_priors/_pc.py`)

A NumPyro `Distribution` subclass: class-level `arg_constraints` and
`support` (`# noqa: RUF012`), `promote_shapes` + `batch_shape` in
`__init__`, a `rate` property, `sample`, `@validate_sample log_prob`, and
`cdf` / `icdf` where analytic. Exemplar: `PCPrecision`. Export from
`_priors/__init__.py` and the package `__init__`; record the R-INLA
counterpart in the header of `tests/fixtures/rinla/make_fixtures.R`.

## 2c. An observation model (`src/pyrox_lgm/_likelihood.py`)

Subclass `AbstractObservation`: `name`, `theta_spec`, `build(y, theta,
data) -> gx.AbstractLikelihood` (a gaussx likelihood), and
`site_distribution` if diagnostics should support it. Exemplar: `Gaussian`.

## 3. Docs

Docstring with the precision / density in MathJax, the R-INLA equivalent
(model name, hyperparameter mapping), and an `Examples:` block that **runs**
(`tests/test_doctests.py`, ELLIPSIS). A `::: pyrox_lgm.Name` entry in
`docs/api/lgm.md`; `make capabilities`.

## 4. Tests

- Components: the precision matches a dense construction
  (`test_components_temporal.py` pattern), constraints hold, and an `inla()`
  fit on simulated data recovers the truth (slow if > ~1.5 s).
- PC priors: calibration with `scipy.integrate.quad` (total mass 1, the
  tail probability equals α, atol 1e-6) and a support test
  (`test_pc_priors.py`).
- R-INLA, when it has the model: add a case to `make_fixtures.R`, run
  `Rscript tests/fixtures/rinla/make_fixtures.R` (R + INLA + fmesher +
  jsonlite), commit the JSON, and add a `_case` branch plus `CASES` and
  `TOLERANCES` entries in `test_rinla_fixtures.py`, recording the measured
  discrepancy next to each tolerance. If R is unavailable, say so in the PR
  instead of inventing reference values.
- `tests/test_import_guard.py` and `tests/test_public_api.py` must still
  pass.

## 5. Verify

`uv run pytest --no-cov packages/pyrox-lgm/tests -m "not slow"`, the slow
tests you added (`-m slow -k <name>`), then the `pre-pr-check` skill.
