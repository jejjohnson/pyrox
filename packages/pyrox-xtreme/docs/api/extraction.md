# Extraction

xarray-native extremes extraction — the preprocessing that turns a raw
spatiotemporal series into the sample an EVT model actually expects. Inputs
are `xr.DataArray`s; array-valued results preserve coordinates, dimensions,
and metadata, while fully reduced results (`constant_threshold`,
`quantile_threshold` reduced over all dimensions, `estimate_extremal_index` on
1-D data) come back as plain floats.

The two routes through this module mirror the two EVT paradigms: **block
maxima** feed a [GEV](distributions.md#generalized-extreme-value), and
**threshold exceedances** feed a
[GPD](distributions.md#generalized-pareto).

## Block maxima

Take the maximum over fixed blocks (usually years) and model those with a GEV.
Simple and robust, but it discards all but one observation per block —
`r_largest_block_maxima` recovers some of that lost information by keeping the
top $r$ order statistics, and `sliding_block_maxima` trades independence for a
larger sample.

::: xtremax.extraction
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - temporal_block_maxima
        - spatial_block_maxima
        - sliding_block_maxima
        - r_largest_block_maxima
        - declustered_block_maxima

## Threshold selection

Peaks-over-threshold keeps every exceedance above $u$, so the threshold choice
*is* the bias–variance tradeoff: too low and the GPD asymptotics fail, too
high and you run out of data. A constant threshold is only defensible for
stationary data — under trends or seasonality, use the time-varying variants
so the exceedance rate stays roughly constant.

::: xtremax.extraction
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - constant_threshold
        - quantile_threshold
        - rolling_threshold
        - seasonal_threshold
        - temporal_threshold

## Declustering & the extremal index

Raw exceedances arrive in clusters — one storm produces several consecutive
threshold crossings — which violates the independence the GPD likelihood
assumes. Declustering reduces each cluster to one peak; the extremal index
$\theta \in (0, 1]$ quantifies how much clustering there was ($\theta = 1$
means none).

::: xtremax.extraction
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - decluster_runs
        - decluster_separation
        - estimate_extremal_index

## Quantile-regression thresholds

A threshold that follows a covariate-driven trend, fitted by quantile
regression rather than taken as a fixed quantile — the right tool when sea
level rise or warming shifts the whole distribution under you.

These require scikit-learn and are only exported when the
`xtremax[threshold]` extra is installed:

```bash
pip install "xtremax[threshold]"
```

::: xtremax.extraction.quantile_regression
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - XarrayQuantileRegressor
        - quantile_regression_threshold
