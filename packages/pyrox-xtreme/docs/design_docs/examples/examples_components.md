---
status: draft
version: 0.1.0
---

# Layer 1 — Component Examples

xarray extraction, threshold selection, and data utilities. *(P3: xarray as data interface)*

---

## Block Maxima Extraction

### Annual maxima from daily data

```python
from xtremax.xarray import block_maxima

# P3: xarray in, xarray out — coordinates preserved
annual_max = block_maxima.temporal_block_maxima(
    ds["temperature"],
    block="year",
)
# Returns xr.DataArray with time=years, preserving lat/lon/station coords
```

### r-largest order statistics

```python
# Keep the 3 largest values per year (for r-largest GEV)
r_largest = block_maxima.r_largest_block_maxima(
    ds["temperature"],
    block="year",
    r=3,
)
```

---

## Threshold Selection

### Three-tier threshold system

```python
from xtremax.xarray import threshold

# Tier 1: Constant quantile threshold
u_const = threshold.quantile_threshold(ds["wind_speed"], q=0.95)

# Tier 2: Trend-following threshold (tracks warming/trends)
u_trend = threshold.trend_quantile_threshold(
    ds["temperature"], q=0.95, time_dim="time", covariates=gmst,
)

# Tier 3: Parametric threshold (GP quantile regression)
u_param = threshold.parametric_quantile_threshold(
    ds["temperature"], q=0.95, covariates=[gmst, elevation],
)
```

---

## Declustering

### Ensure independence of threshold exceedances

```python
from xtremax.xarray import decluster

# Runs declustering: events within run_length of each other → keep max
exceedances = decluster.decluster_runs(excess_values, run_length=3)

# Estimate extremal index (measure of clustering)
theta = decluster.estimate_extremal_index(ds["wind_speed"], threshold=25.0)
```

---

## Dataset Loaders

### Load station data with standardized interface

```python
from xtremax.datasets import ghcnd, ndbc

# GHCN-Daily temperature
temp_ds = ghcnd.load_temperature(station_id="USW00014739", start="1970", end="2023")

# NDBC buoy wave data
wave_ds = ndbc.load_waves(station_id="44025", start="2000", end="2023")

# Climate covariates
from xtremax.datasets.covariates import global_warming, enso
gmst = global_warming.load_gmst()
oni = enso.load_oni()
```
