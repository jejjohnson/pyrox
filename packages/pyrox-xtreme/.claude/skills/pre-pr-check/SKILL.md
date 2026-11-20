---
name: pre-pr-check
description: Run xtremax's full pre-PR verification — ruff lint and format on the whole repo, ty on src/xtremax, the fast lane CI runs and the slow / integration lanes of what changed, the capability index and API-page coverage, the lockfile and the two-tool docs build. Use before committing, pushing or opening a pull request, and after any change to public API, dependencies, docstrings or docs.
---

# Pre-PR check

Run from the repo root and fix what fails before committing. Report results
honestly: what ran, what passed, what was skipped and why.

## Setup

`make install` skips the `[threshold]` extra that CI installs; without it
`tests/test_quantile_regression.py` and `tests/test_capabilities.py` skip.
Install it once:

```bash
uv sync --all-groups --no-group tutorials --extra threshold
```

## Always

```bash
uv run --group lint ruff check .          # entire repo: tests/, scripts/, notebook cells
uv run --group lint ruff format --check .
make typecheck                            # ty check src/xtremax (what CI runs)
make test-fast                            # the fast lane, exactly as CI runs it on a PR
```

## The heavy lane of what you touched

PRs run only the fast lane; `slow` and `integration` run after merge
(Extended Tests), so run them locally for the areas you changed. With no
marker filter in `addopts`, a bare `pytest` on a path runs every lane:

```bash
uv run pytest -n auto tests/<file or directory>     # all lanes of those tests
uv run pytest -n auto -m "slow or integration" tests/point_processes
```

An EVT change runs `tests/test_oracles.py`, `tests/test_gradients.py`,
`tests/test_primitives.py`, `tests/test_primitives_nonstationary.py` and
`tests/test_distributions.py`; a point-process change `tests/point_processes`;
an extraction change `tests/test_extraction.py` and
`tests/test_quantile_regression.py` (add `--with dask` after `uv run` for the
chunked-input case). `make test` runs the whole suite and is what a commit
must pass; `make test-cov` adds the coverage report (`fail_under = 40`).
Under heavy machine load, xdist workers can be killed (`node down: Not
properly terminated`); re-run the crashed tests without `-n auto` before
calling it a failure.

## When the public API changed

```bash
make capabilities
uv run pytest tests/test_capabilities.py tests/test_docs_api_coverage.py
```

…plus the name in the `members:` list of the right section on its
`docs/api/*.md` page (and `src/xtremax/__init__.py` for a distribution or a
`xtremax.primitives` function).

## When dependencies changed

`uv lock` and commit `uv.lock`. A new runtime dependency needs discussion;
an optional one goes behind an extra with a guarded import (as scikit-learn
does for `[threshold]`), and `import xtremax` must keep working without it.
xtremax never imports pipekit.

## When docs or docstrings changed

- `make docs-api` (MkDocs, strict) for docstrings and `docs/api/`.
- `make docs-check` (strict, both halves) and `make docs` (the assembled
  site with link checking) for prose, notebooks and cross-references —
  both need the mystmd CLI (`npm install -g mystmd`); if the theme download
  is blocked, follow `docs/README.md` and restore `docs/myst.yml` after.
- Docstring `Examples:` blocks are not executed by any test: run the ones
  you changed by hand.

## Before pushing

- `git status` shows no stray files (`.plans/`, notebook `.py` drafts,
  `site/`, `public/`) and `uv.lock` changed only if dependencies did.
- Conventional Commits title with a lowercase subject; `!` and a
  `BREAKING CHANGE:` footer for a breaking change.
- Push only to your feature branch, only when asked.
