---
status: draft
version: 0.1.0
---

# Layer 3 — Integration Examples

End-to-end workflows combining xtremax with the broader ecosystem.

---

## With geo_toolz (Climate Data Preprocessing)

### Preprocess raw station data → extract extremes → fit model

```python
import geo_toolz as gt
from xtremax.xarray import block_maxima
from xtremax.models import nonstationary_gev
from xtremax.datasets.covariates import global_warming
from numpyro.infer import MCMC, NUTS

# geo_toolz: preprocess raw station data
preprocess = gt.Sequential([
    gt.validation.ValidateCoords(),
    gt.subset.SubsetTime(time_min="1950", time_max="2023"),
    gt.detrend.CalculateClimatology(freq="day", smoothing=60),
])
ds_clean = preprocess(xr.open_dataset("station_temperature.nc"))

# xtremax: extract annual maxima
annual_max = block_maxima.temporal_block_maxima(ds_clean["tmax"], block="year")

# xtremax: load GMST covariate
gmst = global_warming.load_gmst()

# xtremax: nonstationary GEV with GMST trend
kernel = NUTS(nonstationary_gev)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000)
mcmc.run(key, obs=annual_max.values, covariates=gmst.values)
```

---

## With geo_toolz (Spatial Evaluation)

### Evaluate spatial return level maps

```python
import geo_toolz as gt
from xtremax.distributions import GEVD

# Posterior GEV parameters at each station
samples = mcmc.get_samples()
rl_100 = GEVD(samples["loc"], samples["scale"], samples["concentration"]).return_level(100)

# Build xarray dataset of return levels
rl_ds = xr.Dataset({
    "return_level_100yr": (["station"], rl_100.mean(axis=0)),
    "return_level_100yr_std": (["station"], rl_100.std(axis=0)),
}, coords={"lat": station_lats, "lon": station_lons})

# geo_toolz: evaluate and compare
eval_pipeline = gt.Sequential([
    gt.regrid.Regrid(target_lon=lon_grid, target_lat=lat_grid, method="nearest"),
    gt.metrics.RMSE(variable="return_level_100yr", reference=reference_ds),
])
evaluation = eval_pipeline(rl_ds)
```

---

## Composition Patterns

| Pattern | Components | Use Case |
|---|---|---|
| Station preprocessing → EVA | `geo_toolz.Sequential` → `block_maxima` → `stationary_gev` | Clean station data → extreme analysis |
| Nonstationary attribution | `geo_toolz` + `GMST covariate` → `nonstationary_gev` | Climate change attribution |
| Spatial return level maps | `spatial_gev` → `geo_toolz.Regrid` → `geo_toolz.metrics` | Evaluate spatial model output |
| Multi-hazard comparison | `xtremax` per-hazard → `geo_toolz.Graph` multi-output | Compare temperature / wind / precip extremes |
