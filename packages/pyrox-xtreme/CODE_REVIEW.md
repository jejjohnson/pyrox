# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this
repository. xtremax is an extreme value library on JAX and NumPyro: most
defects worth finding are **numerical** (a hard branch at ξ = 0 that
returns NaN gradients, a tail computed as `1 - F`, a return level that
rounds to infinity in float32, a support whose bijector breaks NUTS) or
**contractual** (a distribution that skips the primitives, a point-process
mask with holes, a sampler without an explicit key, a hand-written
re-implementation of something in the capability index), not style. Read
"Boundaries", "Reuse before you write" and "The contracts" in
[`AGENTS.md`](AGENTS.md) first; this file is the checklist and the report
format.

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
order, a public name missing from `docs/api/`); review what they cannot
see.

### 1. Reuse and boundaries

- Every function, class or module the diff **adds** has been checked
  against [`docs/api/capabilities.md`](docs/api/capabilities.md) and the
  shared helpers (`primitives/_common.py`, `primitives/_moments.py`,
  `distributions/_support.py`, `_rng.py`, `point_processes/operators/_base.py`,
  the point-process `_domain` / `_history` / `_results` / `_integration*`
  modules). A re-implemented density, quantile, return level, compensator,
  sampler, thinning loop, quadrature or GOF statistic is a **High**
  finding, with the existing name to use.
- No `import pipekit` / `pipekit_cycle` anywhere in `src/xtremax/` (interop
  is structural only); no hand-written MCMC / SVI (NumPyro owns inference);
  scikit-learn imported only in `extraction/quantile_regression.py`; no
  GP machinery beyond `primitives/spatial.py`.
- Layers point one way: primitives → distributions; point-process
  primitives → operators → distributions; point-process primitives never
  import equinox.

### 2. Primitives and EVT numerics

- Arrays in, arrays out; the value first, then the parameters
  (`gev_*(x, loc, scale, shape)`, `gpd_*(x, scale, shape)`).
- No `jnp.where(ξ == 0, …)` (or `abs(ξ) < eps`) branch inside a kernel:
  `log1p_over_x` / `expm1_over_x` keep value and gradient smooth through
  ξ = 0. A class-layer branch that must remain uses `GUMBEL_THRESHOLD`
  from `_common`, not a local constant.
- Stable forms: survival as `-expm1(log F)`, quantiles from `-log q`,
  return levels from the exceedance probability (`-log1p(-1/T)`, `-log T`),
  never `1 - 1/T`, moments via `_moments.py`, standardising
  via `clamped_standardize`, tail exponentials via `safe_exp_neg` /
  `safe_exp_neg_density`.
- Out-of-support values give `-inf` log-density and the boundary CDF /
  survival; `±inf` inputs give the limits; NaN propagates; a discarded
  `jnp.where` branch is sanitised first so its gradient is not NaN.
- Accurate in **float32** (the suite's default): no constants or
  intermediates that only work in float64.

### 3. EVT distributions

- A `numpyro.distributions.Distribution` (or the NumPyro distribution it
  extends) whose quantities route to the primitives — no second copy of a
  formula in the class.
- `loc` / `scale` / `concentration`; the deprecated `shape=` alias warns
  and refuses both spellings; `arg_constraints`, `reparametrized_params`,
  `promote_shapes`, `batch_shape` from the broadcast; `validate_args`
  passed through.
- `support` from the `concrete_sign` dispatch (never `interval` with an
  infinite bound), so `biject_to(d.support)` works on a latent site.
- `sample` validates the key (`check_prng_key`) and clips the uniform away
  from 0 and 1; `log_prob` is `@validate_sample`; `expand` rebuilds through
  `__init__`; GEV / GPD keep the `ObservationNoise` seam (`shape=` keyword
  on `sample`, `covariance()`).

### 4. Point processes

- Padded buffers: the mask is a contiguous prefix, temporal padding at the
  window's right edge, spatial padding at `domain.lo`; a sampler that
  rejects candidates compacts accepted events.
- Samplers take an explicit key and a static `max_events` /
  `max_candidates` and return the right `NamedTuple` from `_results.py`;
  every random draw gets its own split key.
- Compensators and log-likelihoods: the sign of `Λ` (`Σ log λ(tᵢ) − Λ`),
  the integration window, history conditioning (no event excites itself
  or earlier times), and padded slots contributing nothing — including to
  the gradient.
- Thinning bounds come from `effective_lambda_max` (pinned bound, else
  `max_intensity()`, else an error), never a derived estimate that can
  under-cover the intensity.
- Operators are `eqx.Module`s with array-leaf parameters (gradient updates
  must reach sampling and log-probs without a rebuild); the HPP, IPP,
  renewal and Hawkes families get diagnostics from `GoodnessOfFitMixin`
  through `_compensator_fn`, marked families their plumbing from
  `SeparableMarkedPP`; signature dispatch through `call_sequence_log_prob`
  / `call_sampler`.
- NumPyro wrappers: `self._op`, `support = constraints.dependent`,
  a non-empty `sample_shape` raises (the temporal HPP excepted),
  `_validate_sample` a no-op; the name matches the
  operator it wraps.

### 5. Extraction and simulations

- xarray in, xarray out, with coordinates and attrs kept; the dimension is
  a parameter (`dim=` / `time_dim=`); chunked (dask) inputs keep working;
  all-missing slices and `min_periods` handled; NumPy-style docstrings.
- Simulations take an integer `seed` (salted per variable), not a global
  RNG.

### 6. Public API and documentation

- New names: in their subpackage `__all__` (and the top-level
  `src/xtremax/__init__.py` for distributions and `xtremax.primitives`), in
  the `members:` list of the right section of their `docs/api/*.md` page,
  and `docs/api/capabilities.md` regenerated (`make capabilities`).
- Docstrings: Google style (NumPy style in `extraction/`); the formula, the
  parameterisation and its relation to scipy's sign convention, a
  reference for a published method, and an `Examples:` block that is
  correct (no test runs it).
- Prose pages use MyST directives and `xref:api#xtremax.<module>.<name>`;
  notebook basenames stay unique; new notebooks join the `docs/myst.yml`
  toc and are committed executed.

### 7. Tests

- Values pinned to an independent oracle (`scipy.stats`, a closed form,
  mpmath), not to xtremax's own formula; parameters away from 0 / 1 so a
  parameterisation slip shows; the ξ grid straddles 0 including the small
  band (1e-6, 1e-4).
- Gradients checked against finite differences, including at ξ = 0.
- Random draws from fixed keys; statistical checks bounded by the
  statistic's own sampling distribution, with the bound's source in a
  comment.
- No `jax.config.update("jax_enable_x64", …)` in-process; float64 checks in
  a subprocess.
- Lane markers right: unmarked for fast unit tests; `slow` for samplers,
  Monte Carlo and `jit` + `grad` sweeps; `integration` for MCMC fits and
  subprocess runs.

### 8. Modern Python and dependencies

- Type hints on every public function; `X | None`; specific exceptions
  with `raise ... from ...`; guard clauses over deep nesting.
- `eqx.Module` (not a dataclass) for anything that flows through JAX;
  NamedTuples for small results.
- No new runtime dependency without discussion; optional ones behind an
  extra with a guarded import; `uv.lock` updated.

---

## xtremax-Specific Checks

### Smooth through the Gumbel limit

```python
# ❌ log(1 + ξz)/ξ is 0/0 at ξ = 0: the value is patched, the gradient is NaN
w = jnp.where(xi == 0, z, jnp.log1p(xi * z) / xi)

# ✅ Smooth through ξ = 0; at ξ = 0 the gradient is the exact -z²/2
from xtremax.primitives._common import log1p_over_x

w = z * log1p_over_x(xi * z)
```

### A support NUTS can use

```python
# ❌ interval with an infinite bound: biject_to(support) maps every point to inf
@property
def support(self):
    return constraints.interval(self.lower_bound(), jnp.inf)


# ✅ Dispatch on the statically known sign of ξ
@property
def support(self):
    sign = concrete_sign(self.concentration)
    if sign == 1:
        return constraints.greater_than_eq(self.lower_bound())
    if sign == -1:
        return constraints.less_than_eq(self.upper_bound())
    return constraints.real
```

### Return levels and tails in float32

```python
# ❌ 1 - 1/T rounds to 1.0 in float32 for T = 1e8: the "return level" is inf
z_T = gev_icdf(1.0 - 1.0 / T, loc, scale, xi)
s = 1.0 - gev_cdf(x, loc, scale, xi)  # 0.0 in the deep tail

# ✅ log1p(-1/T) and -expm1(log F) keep the precision
z_T = gev_return_level(T, loc, scale, xi)
s = gev_survival(x, loc, scale, xi)
```

### Float64 checks without flipping x64 in the test session

```python
# ❌ Mutates global JAX state mid-session (flaky under xdist, leaks JIT caches)
jax.config.update("jax_enable_x64", True)

# ✅ Run the float64 check in a subprocess, as TestFloat64Isolation does
script = textwrap.dedent(
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    from xtremax.primitives import gev_survival
    s = gev_survival(np.float64(40.0), 0.0, 1.0, 0.0)
    assert s.dtype == np.float64 and abs(s / np.exp(-40.0) - 1) < 1e-12
    """
)
subprocess.run([sys.executable, "-c", script], check=True)
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
