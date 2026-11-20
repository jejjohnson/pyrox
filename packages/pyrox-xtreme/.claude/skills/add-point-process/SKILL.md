---
name: add-point-process
description: Add a point-process family to xtremax — or a missing layer of one — across the three layers in src/xtremax/point_processes (pure primitives → equinox operator → NumPyro Distribution wrapper), keeping the padded contiguous-mask buffers, typed NamedTuple sample results, explicit keys, the shared operator mixins and the cross-family invariants. Use when asked to add, port or fix a temporal, spatial, spatiotemporal, marked, self-exciting (Hawkes / ETAS), renewal, Cox or thinned point process, its intensity, compensator, log-likelihood, sampler or goodness-of-fit diagnostics.
---

# Add a point process

Read "3. Point processes" under "The contracts" in `AGENTS.md` and the
package docstring of `src/xtremax/point_processes/__init__.py` (the
padded-buffer invariant and the sample-result convention) first.

## 1. Make sure it does not exist yet

- `docs/api/capabilities.md`, sections `xtremax.point_processes*`: HPP,
  IPP, renewal (any NumPyro inter-event law), exponential and general
  Hawkes, spatiotemporal (ETAS-style) Hawkes, marked (temporal, spatial,
  spatiotemporal), thinning; quadrature (`integrate_log_intensity*`),
  diagnostics (`time_rescaling_residuals`, `ks_statistic_exp1`,
  `qq_exp1_quantiles`, `csr_*`).
- Many "new" processes are compositions: a time-varying rate is an
  `InhomogeneousPoissonProcess` with a `log_intensity_fn` (a
  `PiecewiseConstantLogIntensity`, or any `eqx.Module` with an optional
  `max_intensity()`); marks are `MarkedTemporalPointProcess` /
  `MarkedSpatialPP` / `MarkedSpatioTemporalPP` over a ground process;
  random deletion is `ThinningProcess`; a new Hawkes kernel is an
  `eqx.Module` passed to `GeneralHawkesProcess` (as `ExponentialKernel`
  is).

## 2. Primitives (`src/xtremax/point_processes/primitives/<family>.py`)

- Pure functions, no equinox, no NumPyro base-class machinery (a NumPyro
  distribution may be passed in as a law): `<family>_intensity`,
  `_cumulative_intensity` (the compensator Λ), `_log_prob`
  (`Σ log λ(tᵢ) − Λ` over the real events), `_sample`, plus whatever the
  family has in closed form (`_predict_count`, `_survival`, …).
- Buffers: `times` / `locations` padded to a static size with a `mask`
  that is a contiguous prefix; temporal padding at the window's right edge
  `T`, spatial padding at `domain.lo`. Padded slots must contribute nothing
  to the value **and** to the gradient (mask before `log`, sanitise
  discarded branches).
- Samplers take `key` first and a static `max_events` (or
  `max_candidates` for thinning), split keys for every draw, compact
  accepted events, and return the `NamedTuple` from `_results.py`
  (`SampleResult`, `MarkedSampleResult`, `SpatialSampleResult`, …),
  documenting what `n_events` counts for this family.
- Integrals of `exp(log λ)` go through `integrate_log_intensity` /
  `cumulative_log_intensity` (`_integration.py`) or the spatial /
  spatiotemporal variants; thinning through `thinning_sample`
  (`primitives/thinning.py`); history for user callables through
  `EventHistory` (`_history.py`).
- Export from `primitives/__init__.py` (import + `__all__`).

## 3. Operator (`src/xtremax/point_processes/operators/<family>.py`)

Exemplar: `RenewalProcess` (`renewal.py`); temporal IPP for the mixins
(`temporal.py`).

- An `eqx.Module`: parameters are array leaves (`jnp.asarray` in
  `__init__`), sizes and grid settings `eqx.field(static=True)`; every
  method delegates to the primitives.
- A temporal family with its own compensator (as the HPP, IPP, renewal and
  Hawkes operators have) subclasses `GoodnessOfFitMixin` and implements
  `_compensator_fn(event_times, mask) -> Callable` to get `residuals`,
  `goodness_of_fit` and `compensator_curve`. Thinning samplers read their
  bound through `effective_lambda_max` (`LiveIntensityMixin`: a pinned
  `lambda_max`, else the intensity module's `max_intensity()`, else
  raise — never a derived estimate). Separable marked families subclass
  `SeparableMarkedPP` and implement `_sample_marks`.
- Code that must call operators of different families uses
  `call_sequence_log_prob` / `call_sampler` (`operators/_base.py`), and
  wrapper operators dispatch on the sample-result **type**.
- Export from `operators/__init__.py`.

## 4. NumPyro wrapper (`src/xtremax/point_processes/distributions/<family>.py`)

Exemplar: `distributions/temporal.py` (temporal),
`distributions/hpp_spatial.py` (spatial).

- `class <SameNameAsOperator>(dist.Distribution)`: build the operator in
  `__init__` as `self._op`; `arg_constraints` for the parameters;
  `support = constraints.dependent`; `reparametrized_params: list[str] = []`;
  a static `max_events` buffer size.
- Temporal: `batch_shape` from the parameters' broadcast shape. Spatial /
  spatiotemporal: `batch_shape == ()`; reject a non-scalar parameter with a
  pointer to vmapping at the operator layer (as `hpp_spatial.py` does).
- `sample(key, sample_shape=())`: `check_prng_key(key)`; raise
  `NotImplementedError` for a non-empty `sample_shape` unless the
  operator's sampler batches (only the temporal HPP's does); return the
  `(times, mask)`-style pytree. `log_prob(value)` delegates to
  `self._op.log_prob`; `_validate_sample` returns `None`.
- Export from `distributions/__init__.py`. The shared name with the
  operator is allowed by `scripts/capabilities.py` (`_allowed_by_layout`).

## 5. Docs and capability index

`members:` entries on `docs/api/point-process-primitives.md`,
`point-process-operators.md` and `point-process-distributions.md` (shared
types on `point-processes.md`); `make capabilities`. A from-scratch
notebook, if wanted, follows the `add-notebook` skill (the existing ones
are `docs/notebooks/<family>_from_scratch_and_numpyro.ipynb`).

## 6. Tests (`tests/point_processes/`)

- `test_primitives_<family>.py`: closed forms (counts, compensators,
  log-likelihoods) on hand-built sequences; padded slots inert, including
  their gradient; `jit` / `vmap`.
- Operator and wrapper tests next to the existing ones
  (`test_operators*.py`, `test_distributions*.py`): parameters updated as
  leaves flow into `log_prob` and `sample`; `sample → log_prob` round trip
  through the wrapper; a non-empty `sample_shape` raises. Wrapper
  *operators* (marked, thinning) also reject `sample_shape`
  (`TestSampleShapeThroughWrappers` in `test_property_suite.py`).
- Degeneracies in `test_cross_family_invariants.py` (the new family equals
  an existing one at a special parameter, as Hawkes with α = 0 equals the
  HPP).
- Sampler output keeps the mask invariant (see `TestSamplerMaskInvariant`
  in `test_property_suite.py`); sample statistics bounded by their sampling
  distribution with the source of the bound in a comment.
- Mark any test that draws realisations or runs large quadrature `slow`
  (module-level `pytestmark = pytest.mark.slow` for sampler-heavy files).

## 7. Verify

```bash
uv run pytest -n auto tests/point_processes   # every lane: no addopts filter
uv run pytest tests/test_docs_api_coverage.py tests/test_capabilities.py
```

then the `pre-pr-check` skill.
