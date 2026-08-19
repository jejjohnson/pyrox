# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

xtremax: JAX/NumPyro-native library for extreme value modeling — NumPyro EVT distributions (GEV, GPD, Gumbel, Fréchet, Weibull), temporal/spatial/spatiotemporal point processes, and xarray-native extremes extraction. Built on JAX, NumPyro, equinox, and xarray.

## Common Commands

```bash
make install              # Install all deps (uv sync --all-groups) + pre-commit hooks
make test                 # Run tests: uv run pytest -v
make format               # Auto-fix: ruff format . && ruff check --fix .
make lint                 # Lint code: ruff check .
make typecheck            # Type check: ty check src/xtremax
make precommit            # Run pre-commit on all files
make docs-serve           # Local docs server
```

### Running a single test

```bash
uv run pytest tests/test_example.py::TestClass::test_method -v
```

### Pre-commit checklist (all four must pass)

```bash
uv run pytest -v                              # Tests
uv run --group lint ruff check .              # Lint — ENTIRE repo, not just src/xtremax/
uv run --group lint ruff format --check .     # Format — ENTIRE repo
uv run --group typecheck ty check src/xtremax  # Typecheck — package only
```

**Critical**: Always lint/format with `.` (repo root), not `src/xtremax/`. CI runs `ruff check .` which includes `tests/` and `scripts/`.

## Architecture

### Package structure

All implementation lives in `src/xtremax/`. The public API is re-exported through `src/xtremax/__init__.py`.

### Key directories

| Path | Purpose |
|------|---------|
| `src/xtremax/distributions/` | NumPyro-native EVT distributions (GEV, GPD, Gumbel, Fréchet, Weibull) |
| `src/xtremax/primitives/` | Pure JAX functions per family (`gev_*`, `gpd_*`, …) + non-stationary and spatial helpers |
| `src/xtremax/extraction/` | xarray-native block maxima, thresholds, declustering |
| `src/xtremax/point_processes/primitives/` | Pure functions: intensities, compensators, log-probs, samplers, diagnostics |
| `src/xtremax/point_processes/operators/` | `equinox.Module` process objects (intensity + sampling + GOF) |
| `src/xtremax/point_processes/distributions/` | NumPyro `Distribution` wrappers for point processes |
| `src/xtremax/simulations/` | Synthetic extremes generators |
| `tests/` | Test suite |
| `docs/` | Documentation (MkDocs); API pages in `docs/api/`, design docs in `docs/design_docs/` |
| `notebooks/` | Jupyter notebooks |
| `scripts/` | Example scripts |

## API Reference Pages

The pages in `docs/api/` are **section-structured**, not whole-module dumps:
each `## Section` gets a short prose paragraph explaining what the group is
for, followed by a `::: module` block with an explicit `members:` list. This
keeps the rendered sidebar/TOC organised instead of an alphabetical wall.

When you add a public symbol, add it to the `members:` list of the section it
belongs to. `tests/test_docs_api_coverage.py` enforces that every name in a
module's `__all__` appears on exactly one page, and that no page lists a name
the package no longer exports.

Docstring style is `auto` in `mkdocs.yml` — griffe detects per docstring,
because most of the package is Google style while `src/xtremax/extraction/`
is NumPy style. Don't pin it to a single style: the other one then goes
unparsed and its section underlines leak into the page as headings.

## Documentation Examples

Example notebooks live in `docs/notebooks/` as jupytext percent-format `.py` files. The workflow:

1. Write the `.py` source (jupytext percent format)
2. Convert and execute: `jupytext --to notebook foo.py` then `jupyter nbconvert --execute --inplace foo.ipynb`
3. Delete the `.py` — the executed `.ipynb` is the committed source of truth
4. `mkdocs-jupyter` renders the pre-executed `.ipynb` with `execute: false`

Figures render inline via `plt.show()` — do **not** use `savefig` or commit separate PNG files. The `.ipynb` cell outputs are the single source of rendered figures.

See `.github/instructions/docs-examples.instructions.md` for full standards.

## Coding Conventions

- Google-style docstrings
- `dataclasses` or `attrs` for data containers
- Type hints on all public functions and methods
- Pure functions where possible; side effects isolated and explicit
- Surgical changes only — don't refactor adjacent code or add docstrings to unchanged code

## Plans

Plans and design documents go in `.plans/` (gitignored, never committed). Track work via GitHub issues instead.

## PR Review Comments

When addressing PR review comments, always resolve each review thread after fixing it via the GitHub GraphQL API (`resolveReviewThread` mutation). Do not leave addressed comments unresolved. To obtain the required `threadId`, first list the pull request's review threads via the GitHub GraphQL API (see the "Pull Request Review Comments" section in `AGENTS.md` for a minimal query and end-to-end workflow).

## Code Review

Follow the guidance in `/CODE_REVIEW.md` for all code review tasks.
