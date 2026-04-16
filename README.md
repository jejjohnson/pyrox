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

pyrox bridges [Equinox](https://docs.kidger.site/equinox/) modules and [NumPyro](https://num.pyro.ai/) traces so a single module class can host deterministic parameters, random sample sites, priors, guides, and a model/guide mode switch — without duplicating inference logic. NumPyro owns inference; pyrox just makes modules visible to it. The goal is to write one `__call__` and have it run unchanged under `handlers.seed`, `handlers.trace`, MCMC/NUTS, SVI with `AutoGuide`s, `Predictive`, `jit`, `vmap`, and `grad`.

### Why pyrox?

Equinox gives JAX clean, immutable modules. NumPyro gives JAX a high-quality probabilistic programming surface. But the two don't compose out of the box: Equinox modules are frozen PyTrees, and NumPyro expects per-call access to `numpyro.param` / `numpyro.sample` calls that carry unique site names. `pyrox._core` provides the missing bridge — a light per-instance context that caches site lookups within a call, instance-qualified site names to prevent collisions between sibling modules of the same class, and declarative registries for modules that want priors and guides attached to their parameters.

### What's in the box

- **`PyroxModule`** — an `eqx.Module` subclass with `pyrox_param` / `pyrox_sample` methods that register sites under a stable, module-qualified name. Sites are cached per-call so a parameter referenced twice inside one `__call__` registers exactly once.
- **`Parameterized`** — a `PyroxModule` for GP-style workflows. Declare parameters in `setup()`, attach priors and autoguides, and flip between `"model"` and `"guide"` mode with a single method call. Constrained autoguides respect positivity, simplex, and other supports via `TransformedDistribution`.
- **`pyrox_method`** — a decorator that activates the per-call context; apply it to `__call__` (and any other method that registers sites).
- **`PyroxParam` / `PyroxSample`** — lightweight declarative descriptors for parameter and sample metadata.

---

## 📦 Package Layout

```
pyrox/
├── _core/     # Equinox-to-NumPyro bridge (PyroxModule, PyroxParam, PyroxSample, Parameterized)
├── gp/        # Gaussian process building blocks and protocols (Wave 2+)
└── nn/        # Bayesian and uncertainty-aware neural network layers (Wave 3+)
```

Wave 1 (Core) ships `pyrox._core`. GP and NN subpackages are scaffolded as placeholders until their dedicated waves land — see the [GitHub issue tracker](https://github.com/jejjohnson/pyrox/issues) for the wave roadmap. The core surface is designed so that later waves can add kernels, solvers, variational guides, and uncertainty-aware layers *on top of* the bridge, not next to it.

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

- Required: `jax`, `equinox`, `numpyro`
- Optional: `optax` (install via `pip install 'pyrox[optax]'`)

### From source

```bash
git clone https://github.com/jejjohnson/pyrox.git
cd pyrox
make install
```

---

## 🧪 Three modeling patterns

pyrox is opinionated about *how* to compose Equinox and NumPyro, but not *when* to reach for which primitive. Three patterns cover the common cases, ordered from lightest to heaviest machinery.

### Pattern A — pure Equinox module injected into a NumPyro model

When you already have an Equinox module and just want to treat one of its fields as a random variable, you don't need any pyrox machinery at all. Sample the value inside a plain NumPyro model function and splice it back in with `eqx.tree_at`. This is the right pattern for one-off Bayesian extensions of an otherwise deterministic network — no base class change, no shared registry.

```python
import equinox as eqx
import numpyro


def model(x, y=None):
    net = MLP(key=key)                      # any eqx.Module
    W = numpyro.sample("W", prior)
    net = eqx.tree_at(lambda m: m.W, net, W)
    f = numpyro.deterministic("f", net(x))
    numpyro.sample("obs", dist.Normal(f, 0.1), obs=y)
```

### Pattern B — `PyroxModule` owns its probabilistic semantics

When the module itself is inherently probabilistic — a Bayesian layer, a hierarchical component, anything that "is" a set of sample and param sites — subclass `PyroxModule`. Register sites declaratively inside `__call__` and let the module's qualified name scope them. Sibling instances of the same class automatically get distinct site names, so stacking two `BayesianLinear` layers in one model doesn't collide.

```python
import jax.numpy as jnp
import numpyro.distributions as dist
from pyrox._core import PyroxModule, pyrox_method


class BayesianLinear(PyroxModule):
    pyrox_name = "BayesianLinear"
    in_features: int
    out_features: int

    @pyrox_method
    def __call__(self, x):
        W = self.pyrox_sample(
            "weight",
            dist.Normal(0, 1)
                .expand([self.in_features, self.out_features])
                .to_event(2),
        )
        b = self.pyrox_param("bias", jnp.zeros(self.out_features))
        return x @ W + b
```

### Pattern C — `Parameterized` for constrained params, priors, and guides

When a module has hyperparameters with positivity or simplex constraints, prior/posterior semantics, and a natural train/evaluate split — GP kernels are the canonical case — subclass `Parameterized`. Register parameters with constraints in `setup()`, attach priors, and pick an autoguide per-parameter. Flip `set_mode("model")` vs `set_mode("guide")` to switch between prior sampling (for MCMC) and variational draws (for SVI) without touching `__call__`.

```python
import jax.numpy as jnp
import numpyro.distributions as dist
from pyrox._core import Parameterized, pyrox_method


class RBFKernel(Parameterized):
    pyrox_name = "RBFKernel"

    def setup(self):
        self.register_param(
            "variance", jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        self.register_param(
            "lengthscale", jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        self.set_prior("variance", dist.LogNormal(0.0, 1.0))
        self.autoguide("variance", "normal")

    @pyrox_method
    def __call__(self, X1, X2):
        v = self.get_param("variance")
        ls = self.get_param("lengthscale")
        sq = jnp.sum((X1[:, None] - X2[None, :]) ** 2 / ls ** 2, axis=-1)
        return v * jnp.exp(-0.5 * sq)
```

Switch `kernel.set_mode("guide")` to draw variational params instead of sampling the prior. `"normal"` autoguides respect the registered constraint via `TransformedDistribution`, so a positive-support parameter never yields a negative sample at step zero or during optimization.

### Composing pyrox modules with NumPyro handlers

Because `pyrox_param` and `pyrox_sample` are thin wrappers over `numpyro.param` / `numpyro.sample`, every NumPyro handler composes transparently: `handlers.trace` captures sites, `handlers.substitute` and `handlers.condition` replace or observe them, `handlers.scope` and `handlers.block` control visibility, and `handlers.reparam` rewrites sites in place. The same modules drop into `MCMC(NUTS(model))`, `SVI(model, AutoNormal(model), …)`, and `Predictive(model, ...)` with no extra glue. See `tests/test_core_numpyro_integration.py` for a worked inventory across the handler and inference surface.

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
