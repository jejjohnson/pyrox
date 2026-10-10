---
name: add-extraction-utility
description: Add an xarray-native extremes extraction utility to xtremax — block maxima (temporal, spatial, sliding, r-largest, declustered), a threshold-selection rule, a declustering method or extremal-index estimator — in src/xtremax/extraction, keeping DataArray in / DataArray out, coordinates and attrs, missing-data rules and dask support, with NumPy-style docstrings, tests and docs. Use when asked to add, port or fix block-maxima extraction, peaks-over-threshold thresholds, declustering, run lengths or extremal-index estimation on xarray data.
---

# Add an extraction utility

Read "4. Extraction and simulations" under "The contracts" in `AGENTS.md`.

## 1. Make sure it does not exist yet

`docs/api/capabilities.md`, section `xtremax.extraction`:
`temporal_block_maxima`, `spatial_block_maxima`, `sliding_block_maxima`,
`r_largest_block_maxima`, `declustered_block_maxima`; `constant_threshold`,
`quantile_threshold`, `temporal_threshold`, `seasonal_threshold`,
`rolling_threshold`, `quantile_regression_threshold` (+
`XarrayQuantileRegressor`, `[threshold]` extra); `decluster_runs`,
`decluster_separation`, `estimate_extremal_index`. Many requests are an
option of one of these (`keep_time=` records where each maximum occurred,
`min_periods=` masks sparse blocks, `method=` picks runs vs separation).
Plain xarray (`resample`, `rolling`, `quantile`, `groupby`) is fine inside
them; don't wrap a one-liner in a new public function.

## 2. Write it (`src/xtremax/extraction/<module>.py`)

- Take an `xr.DataArray` first; name the dimension it acts on as a
  parameter (`dim="time"` / `time_dim="time"`, or `dims=` for several);
  return an `xr.DataArray` with the input's other coordinates and `attrs`
  (a scalar only for a constant threshold).
- Missing data: all-NaN slices give NaN (or `NaT` for a datetime
  coordinate; `_na_for` in `block_maxima.py`), not a warning or an error;
  `min_periods` masks blocks with too few observations.
- Internal dimension names come from `_unique_dim` so they cannot collide
  with the caller's; per-block positions go through `_argmax_coord`, which
  ignores missing samples and keeps chunked inputs working.
- Dask: use `xr.apply_ufunc(..., dask="parallelized")` or xarray's own
  lazy operations; never `.values` / `.compute()` on the full array inside
  the function.
- No JAX here: extraction is NumPy / xarray (results feed the NumPyro
  models as arrays).
- A scikit-learn dependency belongs only in `quantile_regression.py`,
  behind the guarded import in `extraction/__init__.py`.
- **NumPy-style docstring** (`Parameters` / `Returns` / `Examples`
  sections with underlines), as the rest of `extraction/`; the
  `docstring_style: auto` setting in `mkdocs.yml` renders it.

## 3. Export, docs, capability index

`extraction/__init__.py` (import + `__all__`); the `members:` list of the
right section (Block maxima / Threshold selection / Declustering & the
extremal index / Quantile-regression thresholds) on
`docs/api/extraction.md`; `make capabilities`.

## 4. Tests (`tests/test_extraction.py`)

- A small synthetic `DataArray` with known maxima / exceedances (see the
  fixture at the top of the file); results checked value by value, plus
  the coordinates and attrs that must survive.
- Missing data: all-NaN blocks, `min_periods`, a genuine `-inf`
  observation next to a missing one.
- A chunked input (`da.chunk({"time": 4})`) gives a lazy result equal to
  the eager one (see
  `TestKeepOriginalCoordinates::test_keep_time_matches_numpy_on_dask_backed_input`).
  Guard it with `pytest.importorskip("dask")`: dask is not a dependency,
  so run it locally with `uv run --with dask pytest …` — CI skips it.
- Every parameter has an effect (`TestIgnoredParameters` pins ones that
  used to be silently ignored).
- Anything needing scikit-learn goes in `tests/test_quantile_regression.py`
  (module-level `pytest.importorskip("sklearn")`).

## 5. Verify

```bash
uv run --with dask pytest -n auto tests/test_extraction.py tests/test_quantile_regression.py
uv run pytest tests/test_docs_api_coverage.py tests/test_capabilities.py
```

then the `pre-pr-check` skill.
