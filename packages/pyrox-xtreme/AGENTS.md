# AGENTS.md

Standing instructions for **every** coding agent working in this repository
(Claude Code, Copilot, Codex, Gemini, …). This is the single source of truth:
`CLAUDE.md` and `.github/copilot-instructions.md` point here.

## What this repo is

xtremax is extreme value modelling for JAX and NumPyro: NumPyro-native EVT
distributions (GEV, GPD, Gumbel, Fréchet, Weibull) whose maths lives in pure
JAX primitives, temporal / spatial / spatiotemporal point processes in three
layers (pure functions → equinox operators → NumPyro wrappers), and
xarray-native extremes extraction (block maxima, thresholds, declustering).
Bayesian EVT models are written by users as plain NumPyro models on top of
it, so **xtremax is a library of primitives**: new code composes what is
here, and anything genuinely new lands where the next person will find it.

The design docs in `docs/design_docs/` are a draft roadmap: they describe
parts that do not exist (a model zoo, max-stable processes, a GPJax wrapper,
dataset loaders). Where they disagree with the code, the code wins.

It is one package, `src/xtremax/`. Each subpackage is public and has its own
`__all__` and its own page in `docs/api/`; `import xtremax` re-exports the
five distribution classes and every name in `xtremax.primitives`.

| Layer | Path | Owns |
|---|---|---|
| 0 · Primitives | `primitives/` | Pure JAX per family: `gev_*`, `gpd_*`, `gumbel_*`, `frechet_*`, `weibull_*` (log-prob, CDF, quantile, mean, return level; survival and log-survival for GEV and GPD); non-stationary return levels (`nonstationary.py`); spatial building blocks (`spatial.py`); the shared stable kernels in `_common.py` and cancellation-free moments in `_moments.py` |
| 1 · Distributions | `distributions/` | `numpyro.distributions.Distribution` subclasses that route every quantity to the primitives; the support dispatch in `_support.py` |
| Point processes · 0 | `point_processes/primitives/` | Pure functions per family (HPP, IPP, renewal, Hawkes, marked, thinning; spatial and spatiotemporal variants): intensities, compensators, log-likelihoods, samplers, diagnostics |
| Point processes · 1 | `point_processes/operators/` | `equinox.Module` process objects; shared surface in `operators/_base.py` (`GoodnessOfFitMixin`, `LiveIntensityMixin`, `SeparableMarkedPP`) |
| Point processes · 2 | `point_processes/distributions/` | NumPyro `Distribution` wrappers around the operators, named like them |
| Point processes · shared | `point_processes/_domain.py`, `_history.py`, `_results.py`, `_adapters.py`, `_integration*.py` | `RectangularDomain`, `TemporalDomain`, `EventHistory`, the `*SampleResult` NamedTuples, retention / mark adapters, log-intensity quadrature |
| Data | `extraction/` | xarray in, xarray out: block maxima, thresholds, declustering, extremal index; quantile-regression thresholds behind the `[threshold]` extra |
| Data | `simulations/` | NumPy synthetic extremes, GMST trajectories, spatial domains (integer `seed`, not PRNG keys) |
| shared | `_rng.py` | `check_prng_key` |

Dependencies point one way: primitives → distributions; point-process
primitives → operators → distributions. Point-process primitives take a
NumPyro distribution as an inter-event or mark law where they need one, but
never import equinox; extraction and simulations import no JAX at all.

### Boundaries

- **xtremax never imports pipekit.** Workflow code composes xtremax into
  pipekit graphs, not the reverse; xtremax classes satisfy pipekit's
  runtime-checkable Protocols structurally (`docs/interop.md`):
  `GeneralizedExtremeValueDistribution` and `GeneralizedParetoDistribution`
  are `pipekit_cycle.ObservationNoise` (`covariance()`,
  `sample(key, shape=...)`), `XarrayQuantileRegressor` is a
  `pipekit.Predictor`. `tests/test_interop.py` checks this against local
  mirrors of the Protocols. No test checks the import ban itself: review it.
- **NumPyro owns inference.** MCMC, SVI, `Predictive`, handlers and masking
  (`d.mask(m)`) are NumPyro's; never re-implement them, and delegate to an
  existing NumPyro distribution where one fits (`GumbelType1GEVD` subclasses
  `numpyro.distributions.Gumbel`).
- **No GeoML library is a runtime dependency.** GP latent fields (spatially
  varying GEV parameters) come from a GP library — the spatial-extremes
  tutorials use pyrox-gp from the `tutorials` group; `primitives/spatial.py`
  stops at distances, design matrices and a two-range correlation on
  purpose. pyrox's design docs list xtremax as a consumer of its GP
  latents, and pipekit-evaluate's protocols as a library that satisfies
  them without importing pipekit; neither is imported here.
- **scikit-learn is optional.** Only `extraction/quantile_regression.py`
  imports it; `xtremax.extraction` exports `XarrayQuantileRegressor` and
  `quantile_regression_threshold` only when it is installed (the
  `[threshold]` extra), so `import xtremax` works without it.

## What xtremax is built on

Each foundation brings a rule. Breaking one runs fine on one example and
fails under `jit`, `grad`, NUTS, float32 or a batch.

| Library | xtremax uses it for | The rule it brings |
|---|---|---|
| **NumPyro** | `Distribution` base, constraints, `biject_to`, `promote_shapes`, `validate_sample`, inference | Distributions declare `arg_constraints`, a `support` whose bijector works for half-unbounded supports, a `batch_shape`, and take `validate_args`; samples come from an explicit key. |
| **jax** | All maths: `jit`, `grad`, `vmap`, PRNG keys | Pure functions, explicit keys, no Python control flow on traced values. The suite runs with x64 **off** (float32), so float32 accuracy is a requirement, not a bonus. |
| **equinox** | Point-process operators, domains, `EventHistory` | Operators are immutable `eqx.Module` pytrees; parameters are array leaves (so a gradient update flows into sampling and log-probs without rebuilding); sizes and flags are static. |
| **jaxtyping** | Shape annotations (`Float[Array, ...]`, `PRNGKeyArray`) | Annotate public array arguments and returns. |
| **xarray** (+ NumPy, dask) | The extraction interface | Take and return `xr.DataArray`; keep coordinates, dims and attrs; stay dask-friendly (`xr.apply_ufunc(..., dask="parallelized")`). |
| **scipy.stats** | The tests' independent oracle | Pin values to scipy, not to xtremax's own formulas; mind its sign conventions (`genextreme(c=-ξ)`, `genpareto(c=ξ)`). |

## Reuse before you write

1. **Search the capability index.** [`docs/api/capabilities.md`](docs/api/capabilities.md)
   lists every public name, each once at its subpackage home, grouped by the
   module that defines it, with a one-line summary. It is generated
   (`make capabilities`) and checked by `tests/test_capabilities.py`.
2. **Search the shared private helpers** in the table below.
3. **If it is missing, add it at its layer**, in the module that owns the
   concept: the maths of a new quantity as a primitive first, then the class
   method that calls it; a helper two modules need goes into the owning
   shared module (`primitives/_common.py`, `point_processes/operators/_base.py`,
   …), not inline in one caller.
4. **One object, one name.** The one deliberate exception: each NumPyro
   wrapper in `point_processes.distributions` shares its name with the
   operator it wraps (the capability script allows exactly that pattern).

| You are about to write… | Use instead |
|---|---|
| A GEV / GPD / Gumbel log-density, CDF, survival, quantile, mean or return level | `xtremax.primitives` (`gev_log_prob`, `gpd_survival`, `gumbel_icdf`, `gev_return_level`, …) or the distribution's method |
| `log1p(ξz) / ξ` with a `jnp.where(ξ == 0, …)` Gumbel branch | `primitives._common.log1p_over_x` / `expm1_over_x` (smooth, finite gradient through ξ = 0) |
| `(x - loc) / scale`, `exp(-w)` in a tail | `_common.clamped_standardize`, `safe_exp_neg` (CDF / survival), `safe_exp_neg_density` (log-density) |
| GEV variance / skewness / kurtosis from `Γ(1 - kξ)` differences | `primitives._moments` (`gev_variance`, `gev_skewness`, `gev_excess_kurtosis`) |
| `icdf(1 - 1/T)` for a return level | `gev_return_level` / `gpd_return_level` (parameterised by the exceedance probability, accurate for large T), `d.return_level(T)` on a GEV or GPD; `gumbel_return_level` still forms `1 - 1/T`, so for a large T use `gev_return_level(T, loc, scale, 0.0)` |
| A return level / period over a changing climate, expected exceedances | `nonstationary_return_level`, `nonstationary_return_period`, `expected_exceedances`, `assemble_nonstationary_gev_fields` |
| Site-distance matrices, regression design matrices, a copula correlation | `pairwise_distances` (finite gradient on the diagonal), `design_matrix`, `two_range_correlation` |
| `constraints.interval(lo, hi)` with an infinite bound as a support | the sign dispatch in `distributions._support.concrete_sign` → `greater_than_eq` / `less_than_eq` / `real` |
| `assert key.shape == (2,)` | `xtremax._rng.check_prng_key` |
| A GEV / GPD likelihood in `numpyro.factor` | the distribution classes; ragged records with `d.mask(m)` |
| `da.resample(time="YE").max()`, a percentile threshold, runs declustering | `temporal_block_maxima` (with `keep_time`, `min_periods`), `quantile_threshold`, `seasonal_threshold`, `rolling_threshold`, `decluster_runs`, `decluster_separation`, `estimate_extremal_index` |
| `n log λ − λT`, a compensator, a Poisson / Hawkes / renewal sampler | `xtremax.point_processes.primitives` (`hpp_*`, `ipp_*`, `exp_hawkes_*`, `general_hawkes_*`, `renewal_*`, `stpp_hawkes_*`, …) or the operators |
| `∫ exp(log λ)` by hand | `integrate_log_intensity`, `cumulative_log_intensity`, `integrate_log_intensity_spatial`, `integrate_log_intensity_spatiotemporal` |
| An Ogata thinning loop, retention probabilities | `thinning_sample`, `thinning_retention_log_prob`, `retention_compensator`; `ThinningProcess` |
| Time-rescaling residuals, KS, QQ | `time_rescaling_residuals`, `ks_statistic_exp1`, `qq_exp1_quantiles`; on an operator, `goodness_of_fit` via `GoodnessOfFitMixin` |
| Ripley's K, L, pair correlation under CSR | `csr_ripleys_k`, `csr_l_function`, `csr_pair_correlation` |
| A padded event buffer, a domain, a sample tuple | `EventHistory`, `RectangularDomain`, `TemporalDomain`, `SampleResult` and its marked / spatial siblings |
| A retention or mark callable from a constant or a function of time | `constant_retention`, `time_varying_retention`, `constant_mark_distribution`, `time_varying_marks` |
| Synthetic extremes for a test or a demo | `xtremax.simulations` (`simulate_temp_extremes`, `generate_gmst_trajectory`, …) |
| A GP prior on GEV parameters, MCMC / SVI loops | pyrox-gp (or another GP library) and NumPyro; not here |

## The contracts

### 1. Primitives: pure, smooth through ξ = 0, float32-accurate

- **Arrays in, arrays out** (`jax.jit` / `grad` / `vmap` safe), no NumPyro,
  no class state. Argument order is the value first, then the parameters:
  `gev_log_prob(x, loc, scale, shape)`, `gpd_cdf(x, scale, shape)` (the GPD
  is for excesses, so it has no `loc`), `gumbel_icdf(q, loc, scale)`,
  `gev_return_level(period, loc, scale, shape)`. The primitives call ξ
  `shape`; the classes call it `concentration`.
- **No hard branch on the Gumbel / exponential limit.** Write the kernel
  through `log1p_over_x` / `expm1_over_x` so value and gradient are smooth
  through ξ = 0 (`∂/∂ξ ≠ 0` there). The only remaining class-layer branches
  (`mode`, the quantile grid of `conditional_excess_mean`) import
  `GUMBEL_THRESHOLD` from `_common`; never cache a local copy.
- **Stable forms.** Survival never as `1 - F` (the GEV uses
  `-expm1(log F)`), GEV quantiles from `-log q`, return levels from the
  exceedance probability (`-log1p(-1/T)` for the GEV, `-log T` for the
  GPD), moments through the reduced forms in `_moments.py`;
  standardise with `clamped_standardize`.
- **Extended reals and support.** Out of support: log-density `-inf`, CDF /
  survival the boundary value; `x = ±inf` gives the limit; NaN propagates.
  Discarded `jnp.where` branches are sanitised first (double-`where`) so
  they never feed `0/0` or `inf` to autodiff.
- **Fréchet and Weibull** are GEV restricted to ξ > 0 / ξ < 0: their
  primitives delegate to `gev_*`, and the sign is a caller obligation (the
  class enforces it).

### 2. EVT distributions: the NumPyro contract

- **Subclass `numpyro.distributions.Distribution`** (or the NumPyro
  distribution it extends, as Gumbel does) and route `log_prob`, `cdf`,
  `icdf`, `mean`, `return_level`, survival and moments to the primitives;
  the class adds no maths of its own beyond EVT conveniences built from
  them (hazard, mean residual life, …).
- **Parameters** `loc`, `scale`, `concentration` (ξ) (GPD: `scale`,
  `concentration`; Gumbel: `loc`, `scale`); `shape=` is a deprecated alias
  that warns (`DeprecationWarning`) and raises if passed with
  `concentration`. `arg_constraints` covers each (`positive` scale;
  `positive` / `less_than(0.0)` concentration for Fréchet / Weibull);
  `reparametrized_params` lists them. Parameters go through
  `promote_shapes`, and `batch_shape` is their broadcast shape.
- **Support** is a property dispatched on `concrete_sign(concentration)`:
  `greater_than_eq(lower_bound())` for ξ > 0, `less_than_eq(upper_bound())`
  for ξ < 0, `real` when the sign is zero, traced or mixed. Never
  `interval(lo, ±inf)`: its bijector returns inf / NaN and breaks NUTS and
  SVI on a latent site.
- **`sample(key, sample_shape=())`** calls `check_prng_key(key)` first, then
  inverts the CDF at a uniform clipped to `[eps, 1 - eps]`; `log_prob`
  carries `@validate_sample` (Gumbel inherits both from NumPyro's `Gumbel`
  and adds only the key check).
- **`expand`** rebuilds through `__init__` so every EVT method survives.
- **The `ObservationNoise` seam** (GEV and GPD only): `sample` also takes
  the keyword `shape=` (passing it with a non-empty `sample_shape` raises),
  and `covariance()` is the variance broadcast to `batch_shape` (`+inf`
  where the moment does not exist).

### 3. Point processes: three layers over padded buffers

- **Padded buffers.** Variable-length events are a fixed-size buffer plus a
  boolean mask whose `True` entries form a **contiguous prefix**; temporal
  padding sits at the window's right edge, spatial padding at `domain.lo`.
  Every sampler emits this form and sequential consumers (time-rescaling
  residuals, Hawkes, renewal) assume it.
- **Samplers** take an explicit key and a static `max_events` /
  `max_candidates`, return a `NamedTuple` from `_results.py`
  (`SampleResult(times, mask, n_events)`, …) whose `n_events` semantics are
  per family, and are batched with `jax.vmap` / `eqx.filter_vmap` over keys
  (only the temporal HPP takes `sample_shape`).
- **Primitives** (`primitives/`) are pure functions: no equinox, and no
  NumPyro base-class machinery (a NumPyro distribution is accepted as a
  law). They are the reference implementation the other layers delegate to.
- **Operators** (`operators/`) are `eqx.Module`s with array-leaf parameters.
  The HPP, IPP, renewal and Hawkes operators get the diagnostics trio
  (`residuals`, `goodness_of_fit`, `compensator_curve`) from
  `GoodnessOfFitMixin` by implementing `_compensator_fn`; thinning samplers
  (the IPPs) read their bound through
  `effective_lambda_max` (a pinned `lambda_max`, else the intensity module's
  `max_intensity()`, else an error — never a derived guess); separable
  marked families subclass `SeparableMarkedPP`. Dispatch over operator
  signatures goes through `call_sequence_log_prob` / `call_sampler` in
  `_base.py`, not `try / except TypeError`.
- **Distributions** (`distributions/`) wrap an operator as `self._op`, set
  `support = constraints.dependent` (a sample is a `(times, mask)`-style
  pytree, not a vector), make `_validate_sample` a no-op, and raise on a
  non-empty `sample_shape` (except the temporal HPP wrapper, whose sampler
  batches). Temporal wrappers take `batch_shape` from their parameters'
  broadcast shape; spatial and spatiotemporal ones have `batch_shape == ()`
  (the homogeneous ones reject a non-scalar rate).
- **User callables** (retention, marks) receive history as an
  `EventHistory`, whatever the family.

### 4. Extraction and simulations

- Extraction functions take an `xr.DataArray`, name the dimension they act
  on (`dim=` / `time_dim=`, default `"time"`), return xarray with
  coordinates and attrs intact (or a scalar, for a constant threshold),
  and keep chunked (dask) inputs working.
  Their docstrings are **NumPy style**, the rest of the package Google
  style; `docstring_style: auto` in `mkdocs.yml` handles both, so never pin
  it.
- Simulations are NumPy generators seeded with an integer `seed` (salted
  per variable); they are the one place that does not take PRNG keys.

### The public API

- Export a public name from its subpackage `__init__.py` (import + `__all__`)
  — and from `src/xtremax/__init__.py` if it is a distribution class or a
  `xtremax.primitives` function — add it to the `members:` list of the right
  section on its `docs/api/*.md` page, and run `make capabilities`.
- `tests/test_docs_api_coverage.py` fails when a name in a documented
  module's `__all__` is on no page, on two, or listed but no longer
  exported.
- Docstrings: Google style (NumPy style in `extraction/`), the formula in
  Unicode or `$…$`, the parameterisation (scipy's sign convention where it
  differs), and an `Examples:` block. Examples are **not executed** by any
  test; keep them correct by hand.
- Renames keep the old spelling as a warning alias first (as `shape=` →
  `concentration=`).

## What enforces them

| Test | Enforces |
|---|---|
| `tests/test_oracles.py` | Every EVT quantity (primitives and classes) against `scipy.stats` across a ξ grid that straddles 0; sample moments and KS of draws; quantile round trips in float32; float64 deep tails in a subprocess (integration); small-ξ moments against mpmath values |
| `tests/test_gradients.py` | Gradients against central finite differences of `scipy.stats` (float64), inside the small-ξ band and across the Gumbel switch; batched-parameter methods |
| `tests/test_primitives.py`, `tests/test_primitives_nonstationary.py` | Round trips, the Gumbel / exponential limits, finite gradients, `jit` / `vmap`, class ↔ primitive parity; non-stationary return levels and the spatial helpers |
| `tests/test_distributions.py` | The NumPyro contract: support bounds, `biject_to(support)` round trips and NUTS on a latent site (`TestBijectToSupport`), PRNG-key validation, the `shape=` alias, masked likelihoods, boundary semantics |
| `tests/test_interop.py` | `ObservationNoise` / `Predictor` conformance and `covariance()` |
| `tests/point_processes/test_cross_family_invariants.py` | Degeneracies agree across families (renewal-exponential = HPP, Hawkes with α = 0 = HPP, thinning with p = 1 = base, …) |
| `tests/point_processes/test_property_suite.py` | Closed forms and statistical checks at non-degenerate parameters (slow) |
| `tests/point_processes/test_epic38_regressions.py`, `test_epic39_consolidation.py`, `test_review_regressions.py` | Fixed bugs stay fixed: compensator signs, contiguous masks, typed sample results, mixin-hosted diagnostics |
| `tests/point_processes/test_*.py` (per family and layer) | Primitives, operators and NumPyro wrappers for every family |
| `tests/test_extraction.py`, `tests/test_quantile_regression.py`, `tests/test_simulations.py` | Extraction (missing data, kept coordinates; the chunked-input cases skip unless dask is installed, and no dependency group installs it), the optional quantile-regression path (skips without scikit-learn), simulators |
| `tests/test_docs_api_coverage.py` | Every public name on exactly one `docs/api/` page, no stale entries |
| `tests/test_capabilities.py` | `docs/api/capabilities.md` is current; no name bound to two different objects outside the operator / wrapper pattern |
| `tests/test_build_docs.py` | The two-tool docs pipeline: inventory parsing, URL rewriting, link checking, `API_PORT` matching `docs/myst.yml` |

## Recipes

Step-by-step recipes for the common jobs live as plain Markdown in
`.claude/skills/<name>/SKILL.md` (Claude Code loads them automatically; any
agent can read and follow them):

| Job | Recipe |
|---|---|
| Add an EVT distribution, or a quantity / primitive for one (incl. non-stationary and spatial helpers) | `add-evt-distribution` |
| Add a point-process family or layer (primitives → operator → NumPyro wrapper) | `add-point-process` |
| Add a block-maxima, threshold or declustering utility on xarray | `add-extraction-utility` |
| Add or update an example notebook or tutorial | `add-notebook` |
| Verify before a PR | `pre-pr-check` |
| Review a change | `xtremax-review` (+ the read-only `.claude/agents/reuse-reviewer.md` and `distribution-reviewer.md`) |
| Write a squash commit message | `squash-commit` |
| Open or link GitHub issues | `create-gh-issue`, `link-gh-issues` (templates in `.github/ISSUE_TEMPLATE/`; `make gh-labels`, `gh-sub`, `gh-block`, `gh-show`) |

## Working in the repo

Always run Python tools through `uv run` (never the system Python); `git`,
`ls` and other non-Python commands need no `uv run`.

```bash
make install              # uv sync --all-groups --no-group tutorials + pre-commit hooks
make test                 # the whole suite, in parallel, no coverage
make test-fast            # fast lane only: what CI runs on a PR
make test-heavy           # slow + integration only
make test-cov             # the whole suite with a coverage report
make lint                 # ruff check .   (entire repo)
make format               # ruff format . && ruff check --fix .
make typecheck            # ty check src/xtremax
make capabilities         # regenerate docs/api/capabilities.md
make docs                 # build both docs halves into public/ and check every link
make docs-check           # strict validation of both halves, no themed render
make docs-api             # MkDocs API reference only (mkdocs build --strict)
make docs-serve           # build, then serve public/ at :8000
```

`make install` does not install the `[threshold]` extra; CI does
(`uv sync --group dev --extra threshold`), and so does `make capabilities`.
Add it locally with `uv sync --all-groups --no-group tutorials --extra
threshold` to run `tests/test_quantile_regression.py` and
`tests/test_capabilities.py` instead of skipping them. The `tutorials`
group (pyrox, xrreader, cartopy, …) is needed only to re-run
`docs/tutorials/spatial_extremes/`.

Run one test from the repo root:
`uv run pytest tests/test_oracles.py::TestGEVPrimitiveOracles -v`.

### Test lanes

The suite is split by marker (registered in `pyproject.toml`) so PRs get a
fast signal (#90):

| Marker | Meaning | Runs on |
|--------|---------|---------|
| *(none)* | fast: primitives, oracles, gradients, extraction | every PR ("Tests", `ci.yml`: Python 3.12, `-m "not slow and not integration"`) |
| `slow` | samplers, Monte Carlo, `jit` + `grad` sweeps | "Extended Tests" (`tests-extended.yml`): every push to `main`, `v*` tags, weekly (Monday 05:00 UTC), on demand |
| `integration` | end-to-end MCMC fits, subprocess runs | same as `slow` |

- Add `@pytest.mark.slow` to any new test that draws point-process
  realisations or does a statistical check; sampler-heavy modules set
  `pytestmark` at module level instead. A test that is slow *once* is
  usually the first XLA compile of a parametrised group being amortised —
  mark by behaviour, not by stopwatch.
- Extended Tests runs the whole suite on Python 3.12 and 3.13 with
  coverage (`fail_under = 40` in `pyproject.toml`), or only the heavy lane
  with `gh workflow run tests-extended.yml -f suite=heavy`. Coverage is not
  in `addopts`: use `make test-cov` rather than `pytest --cov` on one lane.
- **x64 stays off.** Never call `jax.config.update("jax_enable_x64", …)`
  in-process (flaky under xdist, leaks JIT caches); a float64 check runs in
  a subprocess, as `TestFloat64Isolation` in `tests/test_oracles.py` does.

### Tests that assert on random draws

- **Incidental randomness** (any draw would do): a fixed key
  (`jax.random.PRNGKey(0)`, the `key` fixture in `tests/test_distributions.py`).
- **Sampling behaviour under test** (sample moments, KS against scipy,
  point-process counts): bound the statistic by its own sampling
  distribution — a number of standard errors (`4 * sqrt(var / n)` in
  `TestSampleStatistics`), a KS p-value floor — and say in a comment where
  the bound comes from. By the lane rule above, a new statistical check is
  `slow`.

### Before every commit

All of these must pass, from the repo root:

1. `make test` — the **whole** suite, zero failures (`make test-fast` is
   what CI runs on a PR; a commit must pass both lanes).
2. `uv run --group lint ruff check .` — the **entire** repo, which includes
   `tests/`, `scripts/` and the notebook cells under `docs/notebooks/`.
   Running it on `src/xtremax/` alone misses errors CI catches.
3. `uv run --group lint ruff format --check .`
4. `make typecheck` (`ty check src/xtremax`, as CI runs it).
5. After changing a public API: `make capabilities`, and the `members:`
   entry on its `docs/api/` page.
6. After changing docs or docstrings: `make docs-api` at least; `make docs`
   when prose, notebooks or cross-references changed.
7. After changing a dependency: `uv lock`, and commit `uv.lock`.

## Coding principles

1. **Think before coding.** State assumptions; if a request has several
   readings, name them instead of picking one silently; ask when unsure;
   push back when a simpler approach exists.
2. **Simplicity first.** The minimum code that solves the problem: no
   speculative features, no single-use abstractions, no configurability
   nobody asked for, no error handling for impossible cases.
3. **Surgical changes.** Touch only what the task needs; match the existing
   style; don't refactor or add docstrings to code you didn't change;
   remove only what your change made unused, and mention (don't delete)
   unrelated dead code.
4. **Goal-driven.** Turn the task into a check (a failing test, a
   reproduced bug, a scipy oracle to match) and loop until it passes; for
   multi-step work state the plan with a check per step.

Also: `from __future__ import annotations` in every module, type hints on
every public function, `eqx.Module` (not dataclasses) for anything that
flows through JAX, NamedTuples for small results, pure functions with side
effects isolated and explicit.

## Git, commits and pull requests

- Never push to or merge into `main` unless explicitly told to ("push to
  main", "merge to main"). Work on a feature branch, commit locally, and
  push only when asked. "Merge the branch" means push the feature branch,
  not merge into `main`. Confirm before anything that affects a shared
  branch.
- Commit messages and PR titles follow
  [Conventional Commits](https://www.conventionalcommits.org/) with a
  lowercase subject (`feat(distributions): add …`); CI validates PR titles.
  Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`,
  `build`, `ci`, `chore`, `revert`. Breaking changes use `!` and a
  `BREAKING CHANGE:` footer.
- Releases are cut by release-please; don't bump versions by hand.
- **Never replace or remove an existing PR title or description.** An agent
  called for a small follow-up tends to write a fresh description scoped to
  its own work and drop the rest: read the existing one first and only
  append checklist items or update their status. Change the title only to
  fix a Conventional Commits violation.
- Code review follows [`CODE_REVIEW.md`](CODE_REVIEW.md); issues follow the
  label taxonomy and epic model in [`docs/contributing.md`](docs/contributing.md).

### Pull Request Review Comments

After fixing a review comment, resolve its thread. Don't resolve threads you
didn't address.

```bash
# 1. List the review threads and their IDs
gh api graphql -f query='
  query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100) {
          nodes { id isResolved comments(first: 1) { nodes { body path line } } }
        }
      }
    }
  }' -f owner=OWNER -f repo=REPO -F pr=PR_NUMBER

# 2. Resolve an addressed thread
gh api graphql -f query='mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) { thread { isResolved } } }' \
  -f threadId=THREAD_ID
```

When the `gh` CLI is unavailable, use the GitHub MCP tools for the same
operations.

## Documentation

The docs are built by **two tools** and deployed as one site
(`docs/README.md` has the rationale); `docs.yml` builds them on every PR and
push to `main`, and `pages.yml` deploys `public/` to GitHub Pages from
`main`.

| Half | Tool | Source | Deployed at |
|---|---|---|---|
| Prose — home, example notebooks, tutorials, design docs | mystmd | `docs/*.md`, `docs/notebooks/`, `docs/tutorials/`, `docs/design_docs/` (toc in `docs/myst.yml`) | `/` |
| API reference | MkDocs + mkdocstrings | `docs/api/` (`docs_dir` in `mkdocs.yml`) | `/reference/` |

`make docs` runs `scripts/build_docs.py`: it builds the API with
`mkdocs build --strict`, serves `site/` on port 8910 so mystmd can read its
`objects.inv`, runs `myst build --strict`, rewrites the localhost API URLs
to `/reference/`, and fails if any internal link in the assembled `public/`
site does not resolve. mystmd is a Node CLI (`npm install -g mystmd`); if
its theme download is blocked, `docs/README.md` says how to build against a
local clone of the theme.

- **Prose is MyST Markdown**, not MkDocs-Material syntax: `:::{note}`
  directives, not `!!!` admonitions or `===` tabs.
- **Link to the API** with the `xref:` protocol:
  ``[`temporal_block_maxima`](xref:api#xtremax.extraction.temporal_block_maxima)``.
  A missing target fails `myst build --strict`.
- **URLs are flat**: a page's URL is its basename minus any leading `NN_`
  prefix, with `_` turned into `-` (`docs/design_docs/vision.md` →
  `/vision/`), so keep basenames (and MyST labels) unique across the prose
  tree. API pages link back to prose with relative directory URLs such as
  `../vision/`.
- **API pages** (`docs/api/*.md`) are section-structured, not whole-module
  dumps: each `## Section` gets a short prose paragraph, then a
  `::: module` block with an explicit `members:` list, which
  `tests/test_docs_api_coverage.py` checks.
- **Notebooks** in `docs/notebooks/` are committed as executed `.ipynb`
  files, authored as jupytext percent `.py` and deleted after conversion;
  mystmd renders the stored outputs and never re-runs them, so each must be
  listed in the `docs/myst.yml` toc. Figures render inline with
  `plt.show()`: no `savefig`, no committed PNGs. Longer real-data curricula
  live in `docs/tutorials/` (e.g. `spatial_extremes/`), also pre-executed,
  with helper code in a `_helpers/` folder next to the notebooks, outside
  the package; ruff skips their `.ipynb` on purpose (`extend-exclude`).
  Full standards: `.github/instructions/docs-examples.instructions.md`.

## Plans

Plans and scratch design notes go in `.plans/` (gitignored, never
committed); track work in GitHub issues. The committed design references
are `docs/design_docs/`.
