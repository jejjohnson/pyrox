---
applyTo: "packages/**/*.py,scripts/**/*.py"
---

# Python Coding Standards

## Modern Python (3.12+)

- `from __future__ import annotations` at the top of every module
- Type hints on **all** public functions, methods, and module-level variables
- Modern union syntax: `X | None` not `Optional[X]`, `X | Y` not `Union[X, Y]`
- Built-in generics: `list[int]`, `dict[str, Any]` not `List[int]`, `Dict[str, Any]`
- `pathlib.Path` over `os.path`
- f-strings for string formatting
- `equinox.Module` for anything that flows through JAX (layers, kernels, guides, states, results): a dataclass is not a pytree; plain `dataclasses` only for host-side records that are never traced (the `Parameterized` registry entries)
- `Enum` for fixed sets of constants
- Context managers (`with` statements) for resource handling
- Specific exception types (never bare `except:`)
- Proper exception chaining (`raise ... from ...`)
- Early returns / guard clauses to reduce nesting

## Package Preferences

No new runtime dependency without discussion; build on what the packages
already depend on (see "What pyrox is built on" in `AGENTS.md`).

| Purpose | Preferred Package |
|---------|-------------------|
| Probabilistic sites, inference | `numpyro`, through the pyrox bridge |
| Modules / pytrees | `equinox` |
| Linear algebra, Gaussians, GMRFs | `gaussx` |
| Kernels | `kernellib` |
| Network cores, bases | `geonnax` |
| Axis-naming array ops | `einx` |
| Optimisers | `optax` (optional; import lazily) |
| Path handling | `pathlib` (stdlib) |
| Testing | `pytest` |

## Documentation

- Module-level docstrings explaining purpose
- Function/method docstrings for all public APIs (Google style)
- Inline comments explaining *why*, not *what*
- Equations in docstrings in MathJax (`$…$`, `$$…$$`; no Sphinx `:math:`), Unicode in comments (e.g. `# σ² = Σ(xᵢ − μ)² / N`)
- Docstrings of modules with sites list the site names they register
- Public classes and functions should include 2–3 example use cases in docstrings
