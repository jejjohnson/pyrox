# Copilot Instructions

Read [`AGENTS.md`](../AGENTS.md) at the repository root first: it is the
single source of truth for every coding agent working here (the package map,
what pyrox is built on, "reuse before you write", the contracts, the tests
that enforce them, commands, the pre-commit checklist, git and PR rules).
Each package adds its own rules in `packages/<package>/AGENTS.md`; read that
file before changing the package.

The essentials, in case you only read this file:

- A uv workspace of four packages under `packages/` — `pyrox` (the
  Equinox ↔ NumPyro bridge, `pyrox._core`, and `pyrox.inference`),
  `pyrox-gp`, `pyrox-nn`, `pyrox-lgm` — each with `src/<import name>/` and
  `tests/`. `pyrox-lgm` never imports `pyrox-gp`.
- Search [`docs/capabilities.md`](../docs/capabilities.md) before writing a
  helper; linear algebra goes through gaussx, kernel math through kernellib,
  network cores through geonnax.
- Keep the contracts in `AGENTS.md`:
  - **sites**: inside a `PyroxModule`, register sites with `pyrox_sample` /
    `pyrox_param` in a `@pyrox_method`; distinct `pyrox_name`s for sibling
    instances; everything keeps working under every NumPyro handler;
  - **`Parameterized`**: declare params, priors and guides in `setup()`; a
    rebuilt pytree loses its registry;
  - **numerics**: `eqx.Module` (never dataclasses), explicit keys, gaussx on
    PSD-tagged operators, `_kernel_context` for repeated kernel calls.
- Before committing, from the repo root: `uv run pytest -m "not slow"`,
  `uv run --group lint ruff check .`, `uv run --group lint ruff format --check .`,
  `make typecheck`; `make capabilities` after a public API change.
- Path-scoped standards live in `.github/instructions/`; code review follows
  [`CODE_REVIEW.md`](../CODE_REVIEW.md).
