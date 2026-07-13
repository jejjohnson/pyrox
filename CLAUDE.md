# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyrox is a uv workspace of three Python packages for probabilistic
modeling with Equinox and NumPyro. Built with Python 3.12+, uv,
pytest, and MkDocs.

The packages, in dependency order:

| Package    | Import name | Purpose |
|------------|-------------|---------|
| `pyrox`    | `pyrox`     | Equinox-to-NumPyro bridge (`_core`: PyroxModule, PyroxParam, PyroxSample, Parameterized) + ensemble inference (`inference`). No internal deps. |
| `pyrox-gp` | `pyrox_gp`  | Gaussian process building blocks: kernels, guides, likelihoods, Markov/sparse GPs, pathwise sampling, and the shared spectral basis helpers (`_basis`). Depends on `pyrox`. |
| `pyrox-nn` | `pyrox_nn`  | Bayesian/uncertainty-aware NN layers, plus the BNF estimator API (`pyrox_nn.api`) and pandas preprocessing (`pyrox_nn.preprocessing`). Depends on `pyrox` and `pyrox-gp`. |

## Common Commands

```bash
make install              # Install all deps (uv sync --all-groups) + pre-commit hooks
make test                 # Run tests: uv run pytest -v (all packages)
make format               # Auto-fix: ruff format . && ruff check --fix .
make lint                 # Lint code: ruff check .
make typecheck            # Type check: ty check across all package src dirs
make precommit            # Run pre-commit on all files
make docs-serve           # Local docs server
```

### Running a single test

```bash
uv run pytest packages/pyrox-gp/tests/gp/test_kernels.py::test_rbf -v
```

### Pre-commit checklist (all four must pass)

```bash
uv run pytest -v                              # Tests
uv run --group lint ruff check .              # Lint — ENTIRE repo
uv run --group lint ruff format --check .     # Format — ENTIRE repo
uv run --group typecheck ty check \
    packages/pyrox/src/pyrox \
    packages/pyrox-gp/src/pyrox_gp \
    packages/pyrox-nn/src/pyrox_nn            # Typecheck — packages only
```

**Critical**: Always lint/format with `.` (repo root). CI runs `ruff check .` which includes every package's `tests/`.

## Architecture

### Workspace layout

```
packages/
├── pyrox/                    # Core (no internal deps)
│   ├── src/pyrox/
│   │   ├── _core/            # PyroxModule, pyrox_method, PyroxParam, PyroxSample, Parameterized
│   │   └── inference/        # ensemble_map, ensemble_vi, EnsembleMAP, EnsembleVI
│   └── tests/
├── pyrox-gp/                 # GP building blocks (depends on pyrox)
│   ├── src/pyrox_gp/
│   │   ├── __init__.py       # Public API: GPPrior, ConditionedGP, SparseGPPrior, kernels, guides, …
│   │   ├── _src/kernels.py   # Pure kernel functions (closed-form math primitives)
│   │   ├── _basis/           # Kernel spectral densities + RFF draws (shared with pyrox-nn)
│   │   └── _*.py             # kernels, guides, likelihoods, models, markov, sparse, pathwise, …
│   └── tests/                # tests/gp, tests/basis
└── pyrox-nn/                 # Bayesian NN layers (depends on pyrox + pyrox-gp)
    ├── src/pyrox_nn/
    │   ├── __init__.py       # Public API: Bayesian layer wrappers + geonnax re-exports
    │   ├── api/              # BNF estimator entry points (needs [bnf] extra)
    │   ├── preprocessing/    # pandas → array preprocessing (needs [bnf] extra)
    │   └── _*.py             # dense, features, siren, mfn, vssgp, sngp, ensemble, bnf, …
    └── tests/                # tests/nn, tests/api, tests/preprocessing
```

Each package's public API is re-exported through its
`src/<pkg>/__init__.py`. The workspace root ships no code — the
top-level `pyproject.toml` only configures `[tool.uv.workspace]`.

### Dependency rules

- `pyrox` has no internal deps (jax, equinox, numpyro, einx).
- `pyrox-gp` depends on `pyrox` only (+ gaussx, geonnax, lineax).
- `pyrox-nn` depends on `pyrox` and `pyrox-gp` (+ geonnax); pandas and
  optax are gated behind the `[bnf]` optional extra.

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
