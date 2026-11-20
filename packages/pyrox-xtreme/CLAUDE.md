# CLAUDE.md

The rules for every agent live in `AGENTS.md`; this file adds only what is
specific to Claude Code.

@AGENTS.md

## Claude Code specifics

- **Reuse first.** Search `docs/api/capabilities.md` (every public xtremax
  name, each at its subpackage home) before writing a density, a quantile,
  a return level, a point-process likelihood or sampler, or an xarray
  extraction helper; the "Reuse before you write" table in `AGENTS.md` maps
  the usual hand-rolled code to what already exists.
- **Skills** in `.claude/skills/` load on their own when a task matches
  their description (or run them as `/<name>`):
  - building: `add-evt-distribution`, `add-point-process`,
    `add-extraction-utility`, `add-notebook`;
  - shipping: `pre-pr-check`, `xtremax-review`, `squash-commit`;
  - GitHub housekeeping: `create-gh-issue`, `link-gh-issues`.
- **Subagents** (`.claude/agents/`), both read-only, both used by
  `xtremax-review`; run them on any diff that adds code, before
  committing:
  - `reuse-reviewer`: does the diff re-implement something in
    `docs/api/capabilities.md`, a shared private helper, or NumPyro?
  - `distribution-reviewer`: the ξ → 0 limit, tail cancellation, supports
    and endpoints, the NumPyro contract, NaN gradients from discarded
    branches or padded slots, compensators and thinning bounds, keys,
    float32, Monte Carlo tolerances.
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
