---
name: bump-geoml-deps
description: Bump pyrox's git-pinned GeoML upstreams — gaussx, kernellib, geonnax (and gauss-flows) — across the package pyprojects, the workspace override, uv.lock and the capability index, then fix what the new versions break. Use when asked to upgrade / bump gaussx, kernellib or geonnax, after an upstream release, or when a new upstream feature is needed.
---

# Bump the GeoML upstreams

gaussx, kernellib and geonnax are installed from git tags, so a bump touches
several files that must agree.

## 1. Where the pins live

| Upstream | Files |
|---|---|
| gaussx | `packages/pyrox-gp/pyproject.toml` and `packages/pyrox-lgm/pyproject.toml` (`>=` floor in `dependencies`, `tag` in `[tool.uv.sources]`); root `pyproject.toml` `[tool.uv] override-dependencies` (kernellib pins its own gaussx tag, so the override keeps one gaussx) |
| kernellib | `packages/pyrox-gp/pyproject.toml`, `packages/pyrox-lgm/pyproject.toml` (floor + `[tool.uv.sources]` tag) |
| geonnax | `packages/pyrox-gp/pyproject.toml`, `packages/pyrox-nn/pyproject.toml` (direct `geonnax @ git+…@vX` reference) |
| gauss-flows | `packages/pyrox-gp/pyproject.toml` `[flows]` extra (pinned by commit) |

Also check the README's installation note (the `override-dependencies`
snippet) and the notebooks' Colab install cells if they pin a tag.

## 2. Bump

1. Read the upstream changelog between the old and new tags; note removals,
   renames and deprecations (gaussx warns with `GaussxDeprecationWarning`,
   which the root `conftest.py` turns into errors).
2. Update every file in the table consistently (floor = new version, tag =
   new tag, override = new tag).
3. `uv lock` and `uv sync --all-packages --all-groups --all-extras`.
4. `make capabilities`: the upstream sections and their recorded versions
   change; a new upstream name may now clash with a pyrox one (rename, or
   add it to `ALLOWED_SHARED_NAMES` with the reason).

## 3. Fix and verify

- `uv run pytest -m "not slow"` from the root, then `-m slow` for the
  packages that use what changed (pyrox-gp inference, pyrox-lgm `inla`
  against the R-INLA fixtures).
- Replace deprecated upstream calls rather than silencing the warning.
- If the upstream now provides something pyrox hand-rolls (a `cho_solve`
  that a new gaussx primitive covers), mention it in the PR; converting it
  is a separate change unless it is a one-liner.
- `make typecheck`, lint and format, then the `pre-pr-check` skill.

The PR title is `build(deps): bump gaussx to vX.Y.Z` (or the upstreams
bumped together).
