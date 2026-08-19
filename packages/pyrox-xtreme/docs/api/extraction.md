# Extraction

xarray-native extremes extraction: block maxima (temporal, spatial, sliding,
r-largest, declustered), threshold selection (constant, quantile, rolling,
seasonal, quantile-regression), declustering, and extremal-index estimation.
Inputs are `xr.DataArray`s; array-valued results preserve coordinates,
dimensions, and metadata, while fully reduced results — `constant_threshold`,
`quantile_threshold` reduced over all dimensions, `estimate_extremal_index`
on 1-D data — come back as plain floats.

The quantile-regression selectors (`XarrayQuantileRegressor`,
`quantile_regression_threshold`) require scikit-learn and are only exported
when the `xtremax[threshold]` extra is installed:

```bash
pip install "xtremax[threshold]"
```

::: xtremax.extraction
    options:
      show_root_heading: false
      show_root_toc_entry: false
