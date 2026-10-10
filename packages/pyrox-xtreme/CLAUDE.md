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
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
