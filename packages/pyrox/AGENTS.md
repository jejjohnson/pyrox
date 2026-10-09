# pyrox (`pyrox`) — agent rules

The Equinox ↔ NumPyro bridge and ensemble inference. Every other package
depends on it, so a change here is a change to all of them. The root
[`AGENTS.md`](../../AGENTS.md) applies too; this file adds what is specific to
the core.

## Public layout

| Namespace | Holds |
|---|---|
| `pyrox` | `__version__` and the two namespaces below; no class re-exports |
| `pyrox._core` | `PyroxModule`, `pyrox_method`, `Parameterized`, `PyroxParam`, `PyroxSample` (the bridge's public home, despite the underscore) |
| `pyrox.inference` | `ensemble_map`, `ensemble_vi`, `EnsembleMAP`, `EnsembleVI`, `ensemble_init`, `ensemble_loss`, `ensemble_step`, `ensemble_predict`, `EnsembleState`, `EnsembleResult`, `param_group_optimizer` |

Implementation: `_core/pyrox_module.py` (`PyroxModule`, the per-call
`_Context`, the duplicate-param guard, `pyrox_method`),
`_core/parameterized.py`, `_core/descriptors.py` (value objects nothing
consumes), `inference/_ensemble.py`, `inference/_param_groups.py`.

## Contracts

- **Site semantics are the product.** `pyrox_sample` / `pyrox_param` must stay
  transparent delegates to `numpyro.sample` / `numpyro.param` under every
  handler. Any change to naming, caching or the guard needs a case in
  `tests/test_core_numpyro_integration.py` (one test per handler, plus MCMC,
  SVI, `Predictive`, `jit`, `vmap`, `grad`).
- **Scope names are stable.** The scope is `pyrox_name` or the class name —
  never an `id`-based name, which changed whenever Equinox rebuilt a module
  (regression test in `tests/test_core_pyrox_module.py`).
- **The duplicate-param guard never false-positives.** `_visible_traces`
  predicts the recorded name through `trace` / `scope` / `block`; extend it
  rather than loosen it, and add the handler composition to the tests.
- **Downstream uses private names.** `_get_context`, `_pyrox_scope_name` and
  `_pyrox_fullname` are called by pyrox-gp and pyrox-nn; renaming one breaks
  them (search `packages/*/src` first).
- **Class-level registries** (`PyroxModule._contexts`,
  `Parameterized._registry`) are keyed by `id(self)` and cleaned by
  `weakref.finalize`; `_teardown()` cleans up explicitly. Don't move state
  into them that should be in the pytree.
- **`Parameterized`**: `setup()` runs from `__post_init__`; guides are
  `delta` (a `Delta` *sample* site on the prior's support and event dim, so
  `replay` sees it) or `normal` (sized in unconstrained space); `<name>_loc` /
  `<name>_scale` are reserved (a clashing user param raises when the guide
  runs).
- **optax is optional.** `pyrox.inference` imports it through
  `_require_optax()`, which names the `pyrox[optax]` extra; never import optax
  at module scope.
- **Ensembles.** `EnsembleMAP` / `ensemble_map` take a
  `log_joint(params, x, y) -> (loglik, logprior)` and an `init_fn(key)`;
  `EnsembleVI` / `ensemble_vi` take a NumPyro model and guide called as
  `(x, y)`. `param_group_optimizer` labels by key path, never by leaf value.

## Dependencies

`jax`, `equinox`, `numpyro`, `einx` (+ `jaxtyping`, which arrives through
equinox). Extras: `optax`, `colab`. No internal dependencies.

## Tests

- `uv run pytest --no-cov packages/pyrox/tests -v` from the repo root.
- Test modules define their module classes at module level with an explicit
  `pyrox_name`; seed with `handlers.seed(rng_seed=0)` and assert
  `tr["Scope.site"]["type"]` under `handlers.trace()`.
- Ensemble tests check against closed-form ridge / OLS (`linear_problem`
  fixture in `tests/inference/test_ensemble.py`).
