# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this
repository. pyrox is a probabilistic-modelling workspace on Equinox and
NumPyro: most defects worth finding are about **sites** (a raw
`numpyro.sample` in a module, a missing `@pyrox_method`, two instances
colliding on one scope, a `Parameterized` module rebuilt and emptied) or
about **numerics** (hand-rolled linear algebra, a kernel resampled within
one model call, Python control flow on traced values), not style. Read
"Reuse before you write" and "The contracts" in [`AGENTS.md`](AGENTS.md) and
the `AGENTS.md` of each package the diff touches; this file is the checklist
and the report format.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify main >/dev/null 2>&1 && echo main || echo master)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

Skip anything ruff, ty or the tests already enforce (formatting, import
order, RST markup in pyrox / pyrox-gp / pyrox-nn docstrings — pyrox-lgm's
source is not scanned); review what they cannot see.

### 1. Reuse and package boundaries

- Every function, class or module the diff **adds** has been checked
  against [`docs/capabilities.md`](docs/capabilities.md) (the four packages
  plus gaussx, kernellib, geonnax). A re-implemented solve, Cholesky,
  Gaussian KL, Kalman step, kernel function, basis function or network core
  is a **High** finding, with the existing name to use.
- Kernel math belongs in kernellib, deterministic network cores and bases in
  geonnax, structured linear algebra in gaussx; the probabilistic wrapper
  belongs here.
- Imports point down the stack: `pyrox` ← `pyrox-gp` ← `pyrox-nn`;
  `pyrox-lgm` imports `pyrox` only, never `pyrox-gp`. A helper two packages
  need lives in the lowest one.
- `import pyrox_nn` stays pandas-free (`api` / `preprocessing` are not
  imported by the root); optax is lazy in `pyrox` (`_require_optax`) and
  pyrox-gp (`QuasiNewtonInference`), and a required dependency of pyrox-lgm.

### 2. Sites (`PyroxModule`)

- Sites inside a module go through `self.pyrox_sample` /
  `self.pyrox_param`, in a method decorated with `@pyrox_method`; a raw
  `factor` / `deterministic` is named with `self._pyrox_fullname(...)`.
- Every class that may be instantiated twice in one model has a
  `pyrox_name` field (`eqx.field(static=True, default=None)`), and the code
  that builds siblings gives each a distinct name.
- Priors are built in the call at full shape (`.expand([...]).to_event(k)`);
  no sampling or key at construction for pure-prior layers.
- The module still works under `handlers.trace`, `seed`, `substitute`,
  `condition`, `block`, `scope`, `plate`, SVI, MCMC and `Predictive`; under
  `jit` it is closed over, not passed as an argument.
- Private bridge names used across packages (`_get_context`,
  `_pyrox_scope_name`, `_pyrox_fullname`) are not renamed.

### 3. `Parameterized`

- Params, priors and guides are declared in `setup()`; `__post_init__` is
  not overridden (validation goes in `__check_init__`).
- No code path rebuilds a `Parameterized` module (`eqx.tree_at`,
  `apply_updates`, `filter_jit` argument, checkpoint load) and then calls
  `get_param` on the copy.
- `<name>_loc` / `<name>_scale` are not used as user param names; structural
  settings (`nu`, `degree`) are fields, not params.

### 4. Numerics

- **Linear algebra:** Grams and precisions are PSD-tagged lineax operators
  passed to gaussx (`solve`, `cholesky`, `logdet`, `gaussian_kl`, …), honouring
  the model's `solver=`; no new `jnp.linalg.solve` / `cho_solve` /
  `solve_triangular` / `inv` on a covariance; no `.as_matrix()` that
  densifies a structured operator outside a documented dense fallback.
- **Kernel context:** a kernel with priors evaluated more than once per model
  call (Gram + `diag`, several blocks or latents) sits inside
  `_kernel_context` / `_kernel_contexts`.
- **Traceability:** no Python `if` / `while` / `bool()` / `float()` /
  `.item()` on traced values outside the documented eager `fit` loops.
- **Dtypes:** arrays built with the input's dtype (`jnp.eye(n,
  dtype=K.dtype)`); a bare Python scalar combined with an array is weakly
  typed and fine.
- **Randomness:** keys explicit and split before reuse; `numpyro.prng_key()`
  only where a `seed` handler is guaranteed.
- **Stability:** jitter on Gram diagonals, `safe_cholesky` for
  ill-conditioned input, log-space densities, no explicit inverses.
- **Pytrees:** layers, kernels, guides and likelihoods are `eqx.Module`s;
  states and results are `eqx.Module`s or NamedTuples; never a dataclass
  for anything traced; configuration is `eqx.field(static=True)`; no array in a
  static field.

### 5. Public API and documentation

- New public names: exported from the package `__init__.py` and `__all__`,
  given a `::: package.Name` entry on their `docs/api` page, and
  `docs/capabilities.md` regenerated.
- Docstrings: Google style, jaxtyping shapes, the model in MathJax, **the
  site names the module registers**, an `Examples:` block (executed in
  pyrox-lgm), a reference for a published method.
- Renames and removals keep a deprecated path with a `DeprecationWarning`
  naming the replacement.

### 6. Tests

- New behaviour is checked against a dense, closed-form or reference value
  (R-INLA fixtures for `inla()`), with the tolerance's provenance in a
  comment.
- A new module's site set is asserted under `handlers.trace()` +
  `handlers.seed(rng_seed=0)`; it is exercised under SVI / `Predictive`
  where that is its use.
- Expensive tests are `@pytest.mark.slow` — and since CI never runs them,
  the PR says they were run locally — with one unmarked smoke test kept fast.

### 7. Modern Python (≥ 3.12)

- Type hints on every public function; `X | None`, built-in generics;
  f-strings; specific exceptions with `raise ... from ...`; guard clauses
  over deep nesting.

### 8. Dependencies and security

- No new runtime dependency without discussion; optional ones go behind an
  extra and a lazy import. `uv.lock` is updated with any dependency change;
  git-pinned upstreams (gaussx, kernellib, geonnax) move together with the
  root `override-dependencies`.
- No secrets, no network access in fast tests.

---

## pyrox-Specific Checks

### Sites through the bridge

```python
# ❌ Unscoped, uncached, unguarded: two layers collide; a second read resamples
class Layer(eqx.Module):
    def __call__(self, x):
        W = numpyro.sample("W", dist.Normal(0, 1).expand([3, 2]).to_event(2))
        return einx.dot("... i, i o -> ... o", x, W)


# ✅ Scoped to the instance, cached per call, guarded against siblings
class Layer(PyroxModule):
    pyrox_name: str | None = eqx.field(static=True, default=None)

    @pyrox_method
    def __call__(self, x):
        W = self.pyrox_sample("W", dist.Normal(0, 1).expand([3, 2]).to_event(2))
        return einx.dot("... i, i o -> ... o", x, W)
```

### Distinct names for siblings

```python
# ❌ Both register "RBF.variance": the second raises in one trace
# (ValueError for a param, AssertionError once it has a prior)
k1, k2 = pgp.RBF(), pgp.RBF()

# ✅
k1, k2 = pgp.RBF(pyrox_name="RBF_q0"), pgp.RBF(pyrox_name="RBF_q1")
```

### A rebuilt `Parameterized` module

```python
# ❌ tree_at skips setup(): the copy's registry is empty → KeyError on call
kernel = eqx.tree_at(lambda k: k.init_lengthscale, kernel, 0.5)
K = kernel(X, X)

# ✅ Construct a new instance (setup() runs)
kernel = pgp.RBF(init_lengthscale=0.5)
```

### One kernel draw per model call

```python
# ❌ Under seed the diagonal comes from a second hyperparameter draw
K = kernel(X, X)
d = kernel.diag(X)

# ✅
with _kernel_context(kernel):
    K = kernel(X, X)
    d = kernel.diag(X)
```

### Linear algebra through gaussx

```python
# ❌ Dense, ignores the model's solver, no structure
L = jnp.linalg.cholesky(K + noise * jnp.eye(n))
alpha = jax.scipy.linalg.cho_solve((L, True), y)

# ✅
op = lx.MatrixLinearOperator(
    K + noise * jnp.eye(n, dtype=K.dtype), lx.positive_semidefinite_tag
)
alpha = gaussx.solve(op, y)
```

---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${emoji} ${Summary of suggestion with necessary context}

* **Priority**: ${priority_emoji} ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions…)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Emoji | Level | Use when |
|-------|-------|----------|
| 🔥 | **Critical** | Bugs, security issues, or code that will fail |
| ⚠️ | **High** | Significant issues affecting maintainability or correctness |
| 🟡 | **Medium** | Improvements for code quality or consistency |
| 🟢 | **Low** | Minor polish or optional enhancements |

## Suggestion Type Emojis

Prefix each suggestion title with a type indicator:

| Emoji | Type |
|-------|------|
| 🐛 | Bug or potential bug |
| 🔒 | Security concern |
| 🔧 | Change request (must fix) |
| ♻️ | Refactor suggestion |
| 📝 | Documentation improvement |
| 🎨 | Style / formatting issue |
| ⚡ | Performance consideration |
| 🧪 | Testing suggestion |
| ❓ | Question or clarification needed |
| ⛏️ | Nitpick (very minor) |
| 💭 | Design consideration |
| 👍 | Positive feedback (highlight good patterns) |
| 🌱 | Future consideration (not blocking) |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions (use 👍 liberally)
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step
