# pyrox

[![Tests](https://github.com/jejjohnson/pyrox/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/pyrox/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/pyrox/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/pyrox/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/pyrox/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/pyrox/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/pyrox/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/pyrox/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/pyrox/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/pyrox)
[![PyPI version](https://img.shields.io/pypi/v/pyrox.svg)](https://pypi.org/project/pyrox/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyrox.svg)](https://pypi.org/project/pyrox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

Author: J. Emmanuel Johnson
Repo: [https://github.com/jejjohnson/pyrox](https://github.com/jejjohnson/pyrox)
Website: [jejjohnson.netlify.com](https://jejjohnson.netlify.com)

**Probabilistic modeling with Equinox and NumPyro: Bayesian neural networks, Gaussian processes, and composable GP building blocks.**

pyrox unifies Bayesian deep learning and Gaussian process modeling on top of a shared Equinox-to-NumPyro bridge. Declare priors on Equinox modules, run NumPyro inference (MCMC, SVI, AutoGuide), and compose GP primitives (kernels, solvers, integrators, inference strategies) inside hierarchical probabilistic programs.

---

## 📦 Package Layout

```
pyrox/
├── _core/     # Equinox-to-NumPyro bridge (PyroxModule, PyroxParam, PyroxSample, Parameterized)
├── gp/        # Gaussian process building blocks and protocols
└── nn/        # Bayesian and uncertainty-aware neural network layers
```

See [`design_docs/pyrox/`](design_docs/pyrox/) for the full vision, architecture, boundaries, decisions, API surface, and worked examples.

---

## 🚀 Installation

```bash
pip install pyrox
```

Or with `uv`:

```bash
uv add pyrox
```

### Runtime dependencies

- Required: `jax`, `equinox`, `numpyro`, `gaussx`, `lineax`
- Optional: `optax` (install via `pip install 'pyrox[optax]'`)

### From source

```bash
git clone https://github.com/jejjohnson/pyrox.git
cd pyrox
make install
```

---

## 🧪 Quickstart

```python
import pyrox
from pyrox import _core, gp, nn

print(pyrox.__version__)
```

Wave 0 ships the package layout and runtime boundaries; concrete `_core`, `gp`, and `nn` APIs land in subsequent waves. Track progress on GitHub Issues.

---

## 🛠️ Development

```bash
make install              # Install all deps (uv sync --all-groups) + pre-commit hooks
make test                 # Run tests
make format               # Auto-fix formatting and lint
make lint                 # Lint entire repo
make typecheck            # Type check src/pyrox
make precommit            # Run pre-commit on all files
make docs-serve           # Local docs server
```

### Pre-commit checklist

```bash
uv run pytest -v                              # Tests
uv run --group lint ruff check .              # Lint — ENTIRE repo
uv run --group lint ruff format --check .     # Format — ENTIRE repo
uv run --group typecheck ty check src/pyrox   # Typecheck — package only
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contributor workflow and [`AGENTS.md`](AGENTS.md) for AI agent guidance.

---

## 📚 Documentation

- [Vision](design_docs/pyrox/vision.md) — motivation, user stories, design principles
- [Architecture](design_docs/pyrox/architecture.md) — package layout and layer stacks
- [Boundaries](design_docs/pyrox/boundaries.md) — scope and ecosystem
- [Decisions](design_docs/pyrox/decisions.md) — design decisions with rationale
- [API](design_docs/pyrox/api/) — surface inventory and conventions
- [Examples](design_docs/pyrox/examples/) — worked examples across core/nn/gp

Rendered docs deploy from `docs/` via MkDocs + Material.

---

## 🪪 License

MIT — see [`LICENSE`](LICENSE).
