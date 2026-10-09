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
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
