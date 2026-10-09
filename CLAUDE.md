# CLAUDE.md

The rules for every agent live in `AGENTS.md`; this file adds only what is
specific to Claude Code.

@AGENTS.md

## Claude Code specifics

- **Package rules load on demand.** Each `packages/<package>/CLAUDE.md`
  imports that package's `AGENTS.md`, so its layout, extension points and
  tests are in context whenever you work on files there.
- **Reuse first.** Search `docs/capabilities.md` (the four packages, plus
  gaussx, kernellib and geonnax) before writing a helper, a layer, a kernel
  or any linear algebra; the "Reuse before you write" table in `AGENTS.md`
  maps the usual hand-rolled code to what already exists.
- **Skills** in `.claude/skills/` load on their own when a task matches
  their description (or run them as `/<name>`):
  - building: `add-bayesian-layer`, `add-kernel`, `add-gp-component`,
    `add-lgm-component`, `change-core-bridge`, `bump-geoml-deps`,
    `add-notebook`;
  - shipping: `pre-pr-check`, `pyrox-review`, `squash-commit`;
  - GitHub housekeeping: `create-gh-issue`, `link-gh-issues`.
- **Subagents** (`.claude/agents/`), both read-only, both used by
  `pyrox-review`; run them on any diff that adds code, before committing:
  - `reuse-reviewer`: does the diff re-implement something in
    `docs/capabilities.md` (the packages, gaussx, kernellib, geonnax)?
  - `model-reviewer`: sites outside the bridge, scope collisions, rebuilt
    `Parameterized` modules, kernels resampled within a call, traced
    control flow, dtypes, keys.
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
