---
status: draft
version: 0.1.0
---

# Design Decisions

Merged from the former `pyrox-nn` and `pyrox-gp` design documents into one unified `pyrox` package.

## Resolved Questions

### Core

| Question | Resolution |
|---|---|
| Package structure | Single package: `pyrox` — merged from `pyrox-nn` and `pyrox-gp` (see Decision 1) |
| Primary framework | Equinox + NumPyro (pure JAX) — no Flax, no multi-backend |
| Inference ownership | NumPyro — never reimplemented |
| AutoGuide | Use NumPyro's AutoGuide family directly |
| `to_pyro_module()` / `random_pyro_module()` | Not wrapped — users call NumPyro directly |
| `Parameterized` | Core abstraction, not contrib — essential for GPs and any prior-bearing module |
| Linear algebra backend | GaussX — structured operators + operations + solver strategies. pyrox does not hand-roll LA; it delegates to `gaussx.ops` and `gaussx.solvers`. See [gaussx design doc](../gaussx/README.md). |

### NN

| Question | Resolution |
|---|---|
| Layer organization | Edward2-style: by family (dense, dropout, noise, random_feature) |
| Bilevel optimization | Examples only, not in library |
| Regression masterclass | Examples only — showcases library components in end-to-end workflows |
| Data loading / visualization | Out of scope |

### GP

| Question | Resolution |
|---|---|
| Library identity | GP building blocks for NumPyro, not a standalone GP library |
| GP as what? | `numpyro.Distribution` — works with sample/factor/scan |
| Two inference atoms | gp_sample (latent) and gp_factor (collapsed) |
| Solver abstraction | Protocol with `solve(K, y)` + `log_det(K)`, backed by GaussX |
| Temporal inference | State-space via Kalman + Bayes-Newton unification |
| Non-conjugate inference | InferenceStrategy protocol (Laplace, EP, VI, PL, Gauss-Newton) |
| Gaussian expectations | Integrator protocol (sigma points, cubature, Taylor, MC, exact) |
| Guide design | Structured variational families respecting GP geometry |
| Covariance representation | Multiple types (Dense, Diagonal, LowRank, Woodbury, Kronecker) for solver dispatch |

## Key Design Decisions

### 1. Single Unified Package

`pyrox-nn` and `pyrox-gp` are merged into a single `pyrox` package.

**Rationale:**

1. **No code exists yet** — merging now costs nothing and avoids future migration pain.
2. **Shared core is unavoidable** — `Parameterized` extends `PyroxModule`; GP layers (RFF, spectral mixtures) are simultaneously NN layers. Two packages would require a shared `pyrox-core` dependency or duplicated code.
3. **Matches ecosystem granularity** — sister packages (`gaussx`, `finitevolX`, `spectraldiffx`) are each one package. A single `pyrox` keeps the dependency graph flat.
4. **Resolves scope ambiguities** — RFF sharing and GP-as-NN-layer questions disappear when everything lives in one namespace.

### 2. Immutability Workaround

Equinox modules are frozen dataclasses — Pyro's `__setattr__`/`__getattr__` descriptor trick doesn't work. pyrox uses:

- **Explicit method calls** — `self.pyrox_param("bias", init, constraint)` and `self.pyrox_sample("weight", prior)` instead of transparent attribute interception.
- **Class-level registries** keyed by `id(self)` — `PyroxModule._contexts` for per-call caches, `Parameterized._registry` for priors/guides/mode.
- **`eqx.tree_at`** for immutable parameter injection — replacing PyTree leaves functionally.

### 3. Per-Call Context

Each `__call__` invocation gets a fresh `_Context` that caches `pyrox_param` / `pyrox_sample` results. This ensures each sample site appears exactly once per NumPyro trace, even if the same parameter is referenced multiple times in a forward pass.

### 4. Dependent Priors

Priors can be callables that receive the module instance, enabling dependent distributions:

```python
scale = self.pyrox_sample(
    "scale",
    lambda self_: dist.LogNormal(self_.loc, 0.1),
)
```

### 5. Two Patterns for Probabilistic Equinox Modules

The library supports two complementary patterns:

**Pattern A: `PyroxModule` with `pyrox_sample` / `pyrox_param`** — for modules that own their probabilistic semantics:

```python
class BayesianLinear(PyroxModule):
    in_features: int
    out_features: int

    def __call__(self, x):
        W = self.pyrox_sample("weight", dist.Normal(0, 1).expand([self.in_features, self.out_features]).to_event(2))
        b = self.pyrox_param("bias", jnp.zeros(self.out_features))
        return x @ W + b
```

**Pattern B: Plain Equinox + `eqx.tree_at`** — for keeping architecture and probabilistic model separate:

```python
class MLP(eqx.Module):
    W1: jax.Array
    b1: jax.Array
    # ... pure deterministic module

def model(x, y=None):
    net = MLP(hidden_dim=50, key=jr.PRNGKey(0))
    W1 = numpyro.sample("W1", dist.Normal(jnp.zeros((1, 50)), 1.0))
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(50), 1.0))
    net = eqx.tree_at(lambda m: (m.W1, m.b1), net, (W1, b1))
    f = numpyro.deterministic("f", net(x))
    numpyro.sample("obs", dist.Normal(f, sigma), obs=y)
```

Pattern A is more concise for inherently Bayesian layers. Pattern B is better when you want to keep a standard Equinox module and add priors externally. The regression masterclass demonstrates both.

### 6. Mode Switching in Parameterized

`Parameterized` supports `set_mode("model")` / `set_mode("guide")` to switch between sampling from priors and sampling from learned variational distributions. This is essential for SVI workflows where the same module serves as both model and guide.

### 7. GP as `numpyro.Distribution`

GPs are exposed as `numpyro.Distribution` subclasses, making them first-class citizens in NumPyro's probabilistic programming model. This means they compose naturally with `numpyro.sample`, `numpyro.factor`, and `numpyro.contrib.control_flow.scan` without special-casing.

The two inference atoms — `gp_sample` (latent function draws) and `gp_factor` (collapsed marginal likelihood) — map directly to NumPyro's `sample` and `factor` primitives.

### 8. Solver Abstraction via Protocol

GP solvers follow a protocol requiring `solve(K, y)` and `log_det(K)`. All linear algebra is delegated to GaussX, which provides structured operators and dispatch-aware solver strategies. This keeps pyrox free of hand-rolled linear algebra and allows swapping between direct, iterative, and preconditioned solvers without changing GP code.

### 9. Non-Conjugate Inference Strategies

Non-conjugate likelihoods are handled through an `InferenceStrategy` protocol supporting Laplace, EP, VI, PL (posterior linearization), and Gauss-Newton methods. Each strategy implements the same interface, so the GP model code is agnostic to the approximation method. Gaussian expectations within these strategies are computed via an `Integrator` protocol (sigma points, cubature, Taylor expansion, Monte Carlo, or exact).
