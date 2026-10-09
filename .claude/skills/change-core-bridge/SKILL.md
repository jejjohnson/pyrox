---
name: change-core-bridge
description: Change the pyrox core — PyroxModule, pyrox_method, the per-call cache, site naming, the duplicate-param guard, Parameterized (params, priors, autoguides, modes) or pyrox.inference (ensemble MAP / VI, param groups) — without breaking the NumPyro handler semantics every other package relies on. Use when asked to fix, extend or refactor anything in packages/pyrox/src/pyrox.
---

# Change the core bridge (pyrox)

Read `packages/pyrox/AGENTS.md`. Every model in the workspace goes through
this code, so a change here is tested against every NumPyro handler and every
downstream package.

## 1. Before changing anything

- Reproduce the problem as a test: a minimal module + model under the
  handler where it fails, in `tests/test_core_pyrox_module.py`,
  `test_core_parameterized.py` or `test_core_numpyro_integration.py`.
- Find downstream users of what you touch:
  `grep -rn "_get_context\|_pyrox_scope_name\|_pyrox_fullname\|<name>" packages/*/src`.
  `pyrox_gp._context`, `pyrox_gp._multi_output` and `pyrox_nn` call private
  core names; renaming one breaks them.

## 2. Invariants to keep

- `pyrox_sample` / `pyrox_param` behave exactly like `numpyro.sample` /
  `numpyro.param` under `trace`, `seed`, `substitute`, `condition`, `block`,
  `scope`, `mask`, `scale`, `reparam`, `do`, `lift`, `infer_config`, `plate`,
  `factor`, NUTS, SVI (AutoDelta / AutoNormal / AutoMVN), `Predictive`,
  `jit`, `vmap` and `grad`.
- Site names are `"<pyrox_name or ClassName>.<name>"` and stay stable across
  `eqx.tree_at`, `jit` and checkpoints (no `id`-based names).
- The per-call cache makes one site per name per outermost
  `@pyrox_method` call; a cached `None` is a hit (`_MISSING` sentinel).
- The duplicate-param guard never raises on a legitimate program (weight
  sharing within one instance, separate traces); extend `_visible_traces`
  for a new handler composition rather than loosen the check.
- `Parameterized`: `setup()` from `__post_init__`; `delta` guides are `Delta`
  *sample* sites on the prior's support and event dim; `normal` guides are
  sized in unconstrained space; `<name>_loc` / `<name>_scale` reserved; a
  rebuilt module raises the diagnostic `KeyError`.
- `pyrox.inference`: optax only through `_require_optax()`; `EnsembleMAP` /
  `ensemble_map` keep the `log_joint(params, x, y) -> (loglik, logprior)` +
  `init_fn(key)` contract; `ensemble_vi` calls the model as `(x, y)`.

## 3. Tests

- A regression test for the bug (cite the issue in its docstring, as the
  existing ones cite #184 / #187).
- A handler-composition case in `test_core_numpyro_integration.py` for any
  change to naming, caching or the guard.
- Run every package's suite, not just core's: `uv run pytest --no-cov -m "not slow"`
  from the root (it includes the core's NUTS / SVI / handler checks), then
  `-m slow` for pyrox-gp's inference tests and pyrox-nn's BNF estimator
  tests, which exercise the bridge in longer fits.

## 4. Docs

Update the docstrings and `docs/api/core.md` / `docs/api/inference.md`; if
the change alters a documented pattern (A / B / C in the README), update the
README example and the notebooks that use it. `design_docs/pyrox/` predates
the workspace split; note a superseded decision there rather than leaving it
contradictory.
