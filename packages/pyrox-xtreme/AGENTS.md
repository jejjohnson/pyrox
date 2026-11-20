# pyrox-xtreme (`pyrox_xtreme`) — agent rules

Extreme value modelling in JAX and NumPyro. The root [`AGENTS.md`](../../AGENTS.md) applies too; where they differ on the test environment, this file wins for this package. The package came from the xtremax repository (jejjohnson/pyrox#454) with its history; the import is `pyrox_xtreme` and the distribution `pyrox-xtreme`.

## What this package is

pyrox-xtreme is extreme value modelling for JAX and NumPyro: NumPyro-native EVT
distributions (GEV, GPD, Gumbel, Fréchet, Weibull) whose maths lives in pure
JAX primitives, temporal / spatial / spatiotemporal point processes in three
layers (pure functions → equinox operators → NumPyro wrappers), and
xarray-native extremes extraction (block maxima, thresholds, declustering).
Bayesian EVT models are written by users as plain NumPyro models on top of
it, so **pyrox-xtreme is a library of primitives**: new code composes what is
here, and anything genuinely new lands where the next person will find it.

The design docs in `design_docs/pyrox-xtreme/` (at the repo root) are a draft roadmap: they describe
parts that do not exist (a model zoo, max-stable processes, a GPJax wrapper,
dataset loaders). Where they disagree with the code, the code wins.

It is one package, `src/pyrox_xtreme/`. Each subpackage is public and has its own
`__all__` and its own page in `docs/api/xtreme/` (at the repo root); `import pyrox_xtreme` re-exports the
five distribution classes and every name in `pyrox_xtreme.primitives`.

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

- **pyrox-xtreme never imports pipekit.** Workflow code composes pyrox-xtreme into
  pipekit graphs, not the reverse; pyrox-xtreme classes satisfy pipekit's
  runtime-checkable Protocols structurally (`docs/xtreme/interop.md`):
  `GeneralizedExtremeValueDistribution` and `GeneralizedParetoDistribution`
  are `pipekit_cycle.ObservationNoise` (`covariance()`,
  `sample(key, shape=...)`), `XarrayQuantileRegressor` is a
  `pipekit_ops.Predictor`. `tests/test_xtreme_interop.py` checks this against local
  mirrors of the Protocols. No test checks the import ban itself: review it.
- **NumPyro owns inference.** MCMC, SVI, `Predictive`, handlers and masking
  (`d.mask(m)`) are NumPyro's; never re-implement them, and delegate to an
  existing NumPyro distribution where one fits (`GumbelType1GEVD` subclasses
  `numpyro.distributions.Gumbel`).
- **No pyrox package and no GeoML library is a runtime dependency.** pyrox-xtreme sits in tier 0 of this workspace, beside the pyrox core, and never imports `pyrox` or a sibling package (`tests/test_xtreme_import_guard.py`). GP latent fields (spatially
  varying GEV parameters) come from a GP library — the spatial-extremes
  tutorials use pyrox-gp from the `tutorials` group; `primitives/spatial.py`
  stops at distances, design matrices and a two-range correlation on
  purpose. pyrox's design docs list pyrox-xtreme as a consumer of its GP
  latents, and pipekit-evaluate's protocols as a library that satisfies
  them without importing pipekit; neither is imported here.
- **scikit-learn is optional.** Only `extraction/quantile_regression.py`
  imports it; `pyrox_xtreme.extraction` exports `XarrayQuantileRegressor` and
  `quantile_regression_threshold` only when it is installed (the
  `[threshold]` extra), so `import pyrox_xtreme` works without it.

## What pyrox-xtreme is built on

Each foundation brings a rule. Breaking one runs fine on one example and
fails under `jit`, `grad`, NUTS, float32 or a batch.

| Library | pyrox-xtreme uses it for | The rule it brings |
|---|---|---|
| **NumPyro** | `Distribution` base, constraints, `biject_to`, `promote_shapes`, `validate_sample`, inference | Distributions declare `arg_constraints`, a `support` whose bijector works for half-unbounded supports, a `batch_shape`, and take `validate_args`; samples come from an explicit key. |
| **jax** | All maths: `jit`, `grad`, `vmap`, PRNG keys | Pure functions, explicit keys, no Python control flow on traced values. CI runs this suite with x64 **off** (float32), so float32 accuracy is a requirement, not a bonus. |
| **equinox** | Point-process operators, domains, `EventHistory` | Operators are immutable `eqx.Module` pytrees; parameters are array leaves (so a gradient update flows into sampling and log-probs without rebuilding); sizes and flags are static. |
| **jaxtyping** | Shape annotations (`Float[Array, ...]`, `PRNGKeyArray`) | Annotate public array arguments and returns. |
| **xarray** (+ NumPy, dask) | The extraction interface | Take and return `xr.DataArray`; keep coordinates, dims and attrs; stay dask-friendly (`xr.apply_ufunc(..., dask="parallelized")`). |
| **scipy.stats** | The tests' independent oracle | Pin values to scipy, not to pyrox-xtreme's own formulas; mind its sign conventions (`genextreme(c=-ξ)`, `genpareto(c=ξ)`). |

## Reuse before you write

1. **Search the capability index.** The workspace's [`docs/capabilities.md`](../../docs/capabilities.md)
   lists every public name, each once at its subpackage home, grouped by the
   module that defines it, with a one-line summary. It is generated
   (`make capabilities`) and checked by `packages/pyrox/tests/test_capabilities.py`.
2. **Search the shared private helpers** in the table below.
3. **If it is missing, add it at its layer**, in the module that owns the
   concept: the maths of a new quantity as a primitive first, then the class
   method that calls it; a helper two modules need goes into the owning
   shared module (`primitives/_common.py`, `point_processes/operators/_base.py`,
   …), not inline in one caller.
4. **One object, one name.** The one deliberate exception: each NumPyro
   wrapper in `point_processes.distributions` shares its name with the
   operator it wraps (`ALLOWED_SHARED_NAMES` in `scripts/capabilities.py` lists each one).

| You are about to write… | Use instead |
|---|---|
| A GEV / GPD / Gumbel log-density, CDF, survival, quantile, mean or return level | `pyrox_xtreme.primitives` (`gev_log_prob`, `gpd_survival`, `gumbel_icdf`, `gev_return_level`, …) or the distribution's method |
| `log1p(ξz) / ξ` with a `jnp.where(ξ == 0, …)` Gumbel branch | `primitives._common.log1p_over_x` / `expm1_over_x` (smooth, finite gradient through ξ = 0) |
| `(x - loc) / scale`, `exp(-w)` in a tail | `_common.clamped_standardize`, `safe_exp_neg` (CDF / survival), `safe_exp_neg_density` (log-density) |
| GEV variance / skewness / kurtosis from `Γ(1 - kξ)` differences | `primitives._moments` (`gev_variance`, `gev_skewness`, `gev_excess_kurtosis`) |
| `icdf(1 - 1/T)` for a return level | `gev_return_level` / `gpd_return_level` (parameterised by the exceedance probability, accurate for large T), `d.return_level(T)` on a GEV or GPD; `gumbel_return_level` still forms `1 - 1/T`, so for a large T use `gev_return_level(T, loc, scale, 0.0)` |
| A return level / period over a changing climate, expected exceedances | `nonstationary_return_level`, `nonstationary_return_period`, `expected_exceedances`, `assemble_nonstationary_gev_fields` |
| Site-distance matrices, regression design matrices, a copula correlation | `pairwise_distances` (finite gradient on the diagonal), `design_matrix`, `two_range_correlation` |
| `constraints.interval(lo, hi)` with an infinite bound as a support | the sign dispatch in `distributions._support.concrete_sign` → `greater_than_eq` / `less_than_eq` / `real` |
| `assert key.shape == (2,)` | `pyrox_xtreme._rng.check_prng_key` |
| A GEV / GPD likelihood in `numpyro.factor` | the distribution classes; ragged records with `d.mask(m)` |
| `da.resample(time="YE").max()`, a percentile threshold, runs declustering | `temporal_block_maxima` (with `keep_time`, `min_periods`), `quantile_threshold`, `seasonal_threshold`, `rolling_threshold`, `decluster_runs`, `decluster_separation`, `estimate_extremal_index` |
| `n log λ − λT`, a compensator, a Poisson / Hawkes / renewal sampler | `pyrox_xtreme.point_processes.primitives` (`hpp_*`, `ipp_*`, `exp_hawkes_*`, `general_hawkes_*`, `renewal_*`, `stpp_hawkes_*`, …) or the operators |
| `∫ exp(log λ)` by hand | `integrate_log_intensity`, `cumulative_log_intensity`, `integrate_log_intensity_spatial`, `integrate_log_intensity_spatiotemporal` |
| An Ogata thinning loop, retention probabilities | `thinning_sample`, `thinning_retention_log_prob`, `retention_compensator`; `ThinningProcess` |
| Time-rescaling residuals, KS, QQ | `time_rescaling_residuals`, `ks_statistic_exp1`, `qq_exp1_quantiles`; on an operator, `goodness_of_fit` via `GoodnessOfFitMixin` |
| Ripley's K, L, pair correlation under CSR | `csr_ripleys_k`, `csr_l_function`, `csr_pair_correlation` |
| A padded event buffer, a domain, a sample tuple | `EventHistory`, `RectangularDomain`, `TemporalDomain`, `SampleResult` and its marked / spatial siblings |
| A retention or mark callable from a constant or a function of time | `constant_retention`, `time_varying_retention`, `constant_mark_distribution`, `time_varying_marks` |
| Synthetic extremes for a test or a demo | `pyrox_xtreme.simulations` (`simulate_temp_extremes`, `generate_gmst_trajectory`, …) |
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
  style; the workspace's `mkdocs.yml` renders both.
- Simulations are NumPy generators seeded with an integer `seed` (salted
  per variable); they are the one place that does not take PRNG keys.

### The public API

- Export a public name from its subpackage `__init__.py` (import + `__all__`)
  — and from `src/pyrox_xtreme/__init__.py` if it is a distribution class or a
  `pyrox_xtreme.primitives` function — add it to the `members:` list of the right
  section on its `docs/api/xtreme/*.md` page, and run `make capabilities`.
- `tests/test_xtreme_docs_api_coverage.py` fails when a name in a documented
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
| `tests/test_xtreme_oracles.py` | Every EVT quantity (primitives and classes) against `scipy.stats` across a ξ grid that straddles 0; sample moments and KS of draws; quantile round trips in float32; float64 deep tails in a subprocess (integration); small-ξ moments against mpmath values |
| `tests/test_xtreme_gradients.py` | Gradients against central finite differences of `scipy.stats` (float64), inside the small-ξ band and across the Gumbel switch; batched-parameter methods |
| `tests/test_xtreme_primitives.py`, `tests/test_xtreme_primitives_nonstationary.py` | Round trips, the Gumbel / exponential limits, finite gradients, `jit` / `vmap`, class ↔ primitive parity; non-stationary return levels and the spatial helpers |
| `tests/test_xtreme_distributions.py` | The NumPyro contract: support bounds, `biject_to(support)` round trips and NUTS on a latent site (`TestBijectToSupport`), PRNG-key validation, the `shape=` alias, masked likelihoods, boundary semantics |
| `tests/test_xtreme_import_guard.py` | No source file imports, and no module loads, `pyrox` or a sibling package |
| `tests/test_xtreme_interop.py` | `ObservationNoise` / `Predictor` conformance and `covariance()` |
| `tests/point_processes/test_cross_family_invariants.py` | Degeneracies agree across families (renewal-exponential = HPP, Hawkes with α = 0 = HPP, thinning with p = 1 = base, …) |
| `tests/point_processes/test_property_suite.py` | Closed forms and statistical checks at non-degenerate parameters (slow) |
| `tests/point_processes/test_epic38_regressions.py`, `test_epic39_consolidation.py`, `test_review_regressions.py` | Fixed bugs stay fixed: compensator signs, contiguous masks, typed sample results, mixin-hosted diagnostics |
| `tests/point_processes/test_*.py` (per family and layer) | Primitives, operators and NumPyro wrappers for every family |
| `tests/test_xtreme_extraction.py`, `tests/test_xtreme_quantile_regression.py`, `tests/test_xtreme_simulations.py` | Extraction (missing data, kept coordinates; the chunked-input cases skip unless dask is installed, and no dependency group installs it), the optional quantile-regression path (skips without scikit-learn), simulators |
| `tests/test_xtreme_docs_api_coverage.py` | Every public name on exactly one `docs/api/xtreme/` page, no stale entries |
| `tests/test_xtreme_plugin_skill.py` | The downstream plugin's names exist; its worked example runs and its claims hold (slow) |

| `packages/pyrox/tests/test_capabilities.py` | The workspace index is current; no name bound to two objects across the packages, outside the operator / wrapper pattern |
| `packages/pyrox/tests/test_dependency_rules.py` | Every import of a stack library is declared, and none sits above pyrox in geostack's graph |

## Recipes

The workspace's `.claude/skills/` holds this package's recipes (Claude Code loads them automatically; any agent can read and follow them):

| Job | Recipe |
|---|---|
| Add an EVT distribution, or a quantity / primitive for one (incl. non-stationary and spatial helpers) | `add-evt-distribution` |
| Add a point-process family or layer (primitives → operator → NumPyro wrapper) | `add-point-process` |
| Add a block-maxima, threshold or declustering utility on xarray | `add-extraction-utility` |
| Review a change for extreme-value numerics | the read-only `.claude/agents/xtreme-distribution-reviewer.md`, beside the workspace's `pyrox-review` |

Downstream users get pyrox-xtreme's guidance through the Claude Code plugin in `plugins/pyrox-xtreme/` (published by the workspace's `.claude-plugin/marketplace.json`) and `docs/llms.txt`; see `docs/agents.md`. When the public API or the headline usage changes, update `plugins/pyrox-xtreme/skills/extreme-value-analysis-with-pyrox-xtreme/` too: `tests/test_xtreme_plugin_skill.py` runs its worked example (slow) and checks that every `pyrox_xtreme.X` / `xtx.X` path it and the plugin reviewer name still exists.

## Tests

Run them from the repo root, so the root `conftest.py` and pytest configuration apply:

```bash
uv run pytest --no-cov -m "not slow and not integration" packages/pyrox-xtreme/tests
```

The `[threshold]` extra (scikit-learn) is in the dev group, so `tests/test_xtreme_quantile_regression.py` runs instead of skipping. The `notebooks` and `tutorials` dependency groups in this package's `pyproject.toml` are needed only to re-run the notebooks under `docs/xtreme/` (`uv sync --package pyrox-xtreme --group notebooks`).

| Marker | Meaning | Runs in CI |
|---|---|---|
| *(none)* | fast: primitives, oracles, gradients, extraction | the "Tests" job for pyrox-xtreme |
| `integration` | end-to-end MCMC fits, subprocess runs | its own step in that job, after the fast tier |
| `slow` | samplers, Monte Carlo, `jit` + `grad` sweeps | no workflow; run the slow tests of what you touched locally |

- Add `@pytest.mark.slow` to any new test that draws point-process realisations or does a statistical check; sampler-heavy modules set `pytestmark` at module level instead. Mark by behaviour, not by stopwatch.
- **x64 is off in CI.** The root `conftest.py` turns x64 on for the workspace; CI runs this package with `PYROX_TEST_X64=0`, so the float32 contract stays tested, and the suite also passes with x64 on (a whole-workspace run). Reproduce a CI job with `PYROX_TEST_X64=0 uv run pytest -m "not slow and not integration" -o addopts="" --cov=packages/pyrox-xtreme/src --cov-fail-under=75 packages/pyrox-xtreme/tests`; its coverage gate is 75 %, below the workspace's 80 % (it was 40 % in xtremax). Never call `jax.config.update("jax_enable_x64", …)` in a test; a float64 check runs in a subprocess, as `TestFloat64Isolation` in `tests/test_xtreme_oracles.py` does.

### Tests that assert on random draws

- **Incidental randomness** (any draw would do): a fixed key
  (`jax.random.PRNGKey(0)`, the `key` fixture in `tests/test_xtreme_distributions.py`).
- **Sampling behaviour under test** (sample moments, KS against scipy,
  point-process counts): bound the statistic by its own sampling
  distribution — a number of standard errors (`4 * sqrt(var / n)` in
  `TestSampleStatistics`), a KS p-value floor — and say in a comment where
  the bound comes from. By the lane rule above, a new statistical check is
  `slow`.

