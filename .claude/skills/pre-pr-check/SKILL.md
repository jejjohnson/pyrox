---
name: pre-pr-check
description: Run pyrox's full pre-PR verification — lint and format on the whole repo, ty on all four packages, the fast test tier with the coverage gate, the slow tests of what changed (CI never runs them), the capability index, the lockfile and the docs build. Use before committing, pushing or opening a pull request, and after any change to public API, dependencies, docstrings or docs.
---

# Pre-PR check

Run from the repo root and fix what fails before committing. Report results
honestly: what ran, what passed, what was skipped and why.

## Always

```bash
uv run --group lint ruff check .          # entire repo, incl. every package's tests/ and scripts/
uv run --group lint ruff format --check .
make typecheck                            # ty on all four packages
uv run pytest -m "not slow"               # what CI runs, with the 80 % coverage gate
```

While iterating, run the touched package's tests with `--no-cov` (a subset
never reaches the coverage gate):
`uv run pytest --no-cov packages/<package>/tests -m "not slow"`.

`make typecheck` in an environment with every extra installed reports three
unused `ty: ignore` comments on optional imports (`gauss_flows`, `xarray`);
those three are expected locally (CI's environment lacks the extras). Any
other diagnostic is yours.

## The slow tests of what you touched

No workflow runs `@pytest.mark.slow`, so run them yourself and say so in the
PR:

```bash
uv run pytest --no-cov -m slow packages/<package>/tests/<area>
```

Core changes (`packages/pyrox`) run the slow inference tests of pyrox-gp and
pyrox-nn too; pyrox-lgm changes run `test_inla.py` and
`test_rinla_fixtures.py`.

## When the public API changed

```bash
make capabilities
uv run pytest --no-cov packages/pyrox/tests/test_capabilities.py
```

…and add the `::: package.Name` entry to the `docs/api` page (no test
checks it). For pyrox-lgm, `test_public_api.py` and `test_doctests.py`.

## When dependencies changed

`uv lock`, commit `uv.lock`; for a GeoML upstream, the `bump-geoml-deps`
skill.

## When docs or docstrings changed

`make docs` must build (non-strict; it renders 29 executed notebooks, so it
takes several minutes). Docstrings: no Sphinx markup
(`tests/**/test_docstrings.py`).

## Before pushing

- `git status` shows no stray files (local notebook `.py` files are
  gitignored; `.plans/` too).
- Conventional Commits title with a lowercase subject; `!` and a
  `BREAKING CHANGE:` footer for a breaking change.
- Push only to your feature branch, only when asked.
