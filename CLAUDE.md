# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyrox: probabilistic modeling with Equinox and NumPyro. Bayesian neural networks, Gaussian processes, and composable GP building blocks. Built with Python 3.12+, uv, pytest, and MkDocs.

Top-level areas:

- `pyrox._core` — Equinox-to-NumPyro bridge (PyroxModule, PyroxParam, PyroxSample, Parameterized).
- `pyrox.gp` — Gaussian process building blocks and protocols.
- `pyrox.nn` — Bayesian and uncertainty-aware neural network layers.

## Common Commands

```bash
make install              # Install all deps (uv sync --all-groups) + pre-commit hooks
make test                 # Run tests: uv run pytest -v
make format               # Auto-fix: ruff format . && ruff check --fix .
make lint                 # Lint code: ruff check .
make typecheck            # Type check: ty check src/pyrox
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
uv run --group lint ruff check .              # Lint — ENTIRE repo, not just src/pyrox/
uv run --group lint ruff format --check .     # Format — ENTIRE repo
uv run --group typecheck ty check src/pyrox   # Typecheck — package only
```

**Critical**: Always lint/format with `.` (repo root), not `src/pyrox/`. CI runs `ruff check .` which includes `tests/` and `scripts/`.

## Architecture

### Package structure

All implementation lives in `src/pyrox/`. The public API is re-exported through `src/pyrox/__init__.py`.

### Key directories

| Path | Purpose |
|------|---------|
| `src/pyrox/` | Main package source code |
| `src/pyrox/_core/` | Equinox-to-NumPyro bridge primitives |
| `src/pyrox/gp/` | Gaussian process building blocks |
| `src/pyrox/nn/` | Bayesian/uncertainty-aware NN layers |
| `tests/` | Test suite |
| `docs/` | Documentation (MkDocs) |
| `design_docs/pyrox/` | Design documents (vision, architecture, boundaries, decisions, API, examples) |
| `notebooks/` | Jupyter notebooks |

## Documentation Examples

Example notebooks live in `docs/notebooks/` as jupytext percent-format `.py` files. The workflow:

1. Run notebooks locally to generate figures and tables
2. Save figures to `docs/images/{notebook_name}/` via `savefig`
3. Embed saved PNGs in markdown cells for static rendering (`execute: false`)
4. Commit both `.py` source and generated PNGs

See `.github/instructions/docs-examples.instructions.md` for full standards.

## Coding Conventions

- Google-style docstrings
- `dataclasses` or `attrs` for data containers
- Type hints on all public functions and methods
- Pure functions where possible; side effects isolated and explicit
- Surgical changes only — don't refactor adjacent code or add docstrings to unchanged code

## Plans

Plans and design documents go in `.plans/` (gitignored, never committed) or `design_docs/pyrox/` (committed design references). Track work via GitHub issues instead.

## PR Review Comments

When addressing PR review comments, always resolve each review thread after fixing it via the GitHub GraphQL API (`resolveReviewThread` mutation). Do not leave addressed comments unresolved. To obtain the required `threadId`, first list the pull request's review threads via the GitHub GraphQL API (see the "Pull Request Review Comments" section in `AGENTS.md` for a minimal query and end-to-end workflow).

## Code Review

Follow the guidance in `/CODE_REVIEW.md` for all code review tasks.
