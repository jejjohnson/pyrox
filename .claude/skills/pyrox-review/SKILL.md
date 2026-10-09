---
name: pyrox-review
description: Review a change or pull request in the pyrox workspace against the repo's own rules — CODE_REVIEW.md, the site / Parameterized / numerics contracts in AGENTS.md, the package AGENTS.md files, package boundaries, and reuse of gaussx / kernellib / geonnax. Use when asked to review a diff, branch or PR in this repo.
---

# Review a pyrox change

1. **Get the diff** as `CODE_REVIEW.md` describes, or from the PR. Note
   which packages it touches and read their `packages/<package>/AGENTS.md`.
2. **Reuse** — run the `reuse-reviewer` subagent on the diff: hand-rolled
   linear algebra, kernel math or network cores are the main way this stack
   drifts from gaussx / kernellib / geonnax.
3. **Model correctness** — run the `model-reviewer` subagent on the diff (in
   parallel with step 2): sites outside the bridge, missing `@pyrox_method`,
   scope collisions, rebuilt `Parameterized` modules, kernels resampled
   within a call, traced control flow, dtypes, keys.
4. **Boundaries** — imports point down the stack; pyrox-lgm never imports
   pyrox-gp; `import pyrox_nn` stays pandas-free; optax stays lazy; private
   core names used downstream are not renamed.
5. **Checklist** — the rest of `CODE_REVIEW.md` (public API, docs entry,
   capability index, tests with dense references and provenance, slow tests
   run locally), skipping what ruff, ty or the tests enforce.
6. **Verify claims** — for anything you would flag as a bug, run it: a
   minimal model under `handlers.trace()` + `handlers.seed(rng_seed=0)`, or
   the computation against a dense reference.

Report in the format `CODE_REVIEW.md` gives (overview, suggestions with
priority, file, lines and a concrete change, summary).
