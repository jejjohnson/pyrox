---
applyTo: "src/**/*.py,tests/**/*.py,scripts/**/*.py"
---

# Python Coding Standards

## Modern Python (3.12+)

- `from __future__ import annotations` at the top of every module
- Type hints on **all** public functions, methods, and module-level variables
- Modern union syntax: `X | None` not `Optional[X]`, `X | Y` not `Union[X, Y]`
- Built-in generics: `list[int]`, `dict[str, Any]` not `List[int]`, `Dict[str, Any]`
- `pathlib.Path` over `os.path`
- f-strings for string formatting
- `equinox.Module` for point-process operators, domains and event histories (immutable pytrees; a dataclass is not one); NamedTuples for small results such as sample results; `numpyro.distributions.Distribution` subclasses for distributions
- `Enum` for fixed sets of constants
- Context managers (`with` statements) for resource handling
- Specific exception types (never bare `except:`)
- Proper exception chaining (`raise ... from ...`)
- Early returns / guard clauses to reduce nesting

## Package Preferences

No new runtime dependency without discussion; build on what xtremax already
depends on (see "Boundaries" in `AGENTS.md`). Optional ones go behind an
extra and a guarded import, as scikit-learn does for `[threshold]`.

| Purpose | Preferred Package |
|---------|-------------------|
| Distributions, constraints, inference | `numpyro` |
| Array maths, autodiff, PRNG | `jax` |
| Operators / pytrees | `equinox` |
| Shape annotations | `jaxtyping` |
| Labelled data (extraction) | `xarray` |
| Path handling | `pathlib` (stdlib) |
| Testing (reference values from `scipy.stats`) | `pytest` |

## Documentation

- Module-level docstrings explaining purpose
- Function/method docstrings for all public APIs (Google style; `src/xtremax/extraction/` is NumPy style)
- Inline comments explaining *why*, not *what*
- Scientific algorithms should include Unicode equations in docstrings (e.g. `# σ² = Σ(xᵢ − μ)² / N`)
- Public classes and functions include an `Examples:` block (no test runs them, so check them by hand)
