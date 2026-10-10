# Copilot Instructions

Read [`AGENTS.md`](../AGENTS.md) at the repository root first: it is the
single source of truth for every coding agent working here (the layer map,
the boundaries with NumPyro / pipekit / the GP libraries, "reuse before you
write", the contracts, the tests that enforce them, commands, test lanes,
the pre-commit checklist, the two-tool docs, git and PR rules).

The essentials, in case you only read this file:

- One package, `src/xtremax/`: pure JAX `primitives/` → NumPyro
  `distributions/`; point processes in three layers
  (`point_processes/primitives/` → equinox `operators/` → NumPyro
  `distributions/`); xarray `extraction/`; NumPy `simulations/`. Search
  [`docs/api/capabilities.md`](../docs/api/capabilities.md) before writing a
  helper.
- xtremax never imports pipekit (it satisfies pipekit's Protocols
  structurally), NumPyro owns inference, and scikit-learn is optional (the
  `[threshold]` extra, used only by the quantile-regression threshold).
- Keep the contracts in `AGENTS.md`:
  - **primitives**: arrays in, arrays out; smooth through ξ = 0 via
    `log1p_over_x` / `expm1_over_x` (no hard Gumbel branch); stable tail
    forms; accurate in float32;
  - **EVT distributions**: NumPyro `Distribution`s that route every
    quantity to the primitives; `concentration` (with the deprecated
    `shape=` alias); a `support` from the `concrete_sign` dispatch, never
    `interval(lo, ±inf)`; `check_prng_key` in `sample`;
  - **point processes**: padded buffers whose mask is a contiguous prefix,
    explicit keys and static buffer sizes, `NamedTuple` sample results,
    `eqx.Module` operators with array-leaf parameters, thin NumPyro
    wrappers with `support = constraints.dependent`;
  - **extraction**: xarray in, xarray out, NumPy-style docstrings.
- Tests run with x64 off (float32); float64 checks run in a subprocess.
  Mark tests that draw point-process realisations or check statistics
  `slow`.
- Before committing, from the repo root: `make test`,
  `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`,
  `make typecheck`; `make capabilities` and the `docs/api/` `members:`
  entry after a public API change.
- Step-by-step recipes (add an EVT distribution, a point process, an
  extraction utility, a notebook; pre-PR check; review) are plain Markdown
  in `.claude/skills/<name>/SKILL.md`, and the two review checklists in
  `.claude/agents/`; follow them as written.
- Path-scoped standards live in `.github/instructions/`; code review follows
  [`CODE_REVIEW.md`](../CODE_REVIEW.md).
