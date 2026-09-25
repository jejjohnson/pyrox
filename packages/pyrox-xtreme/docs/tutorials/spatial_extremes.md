---
title: Spatial Extremes
short_title: Spatial Extremes
---

# Spatial Extremes

A step-by-step curriculum on modelling **climate extremes in space**: how often
will a temperature this high recur, and how does that risk vary across a region?
It builds up from a single station to a full spatial model, one concept per
notebook, on **real station data** from the Copernicus Climate Data Store (CDS)
over Spain.

Four packages do the heavy lifting, one per layer:

| Layer | Package | Role |
|-------|---------|------|
| Data | [`xrreader`](https://github.com/jejjohnson/xrreader) | pull + cache CDS in-situ land stations over Iberia |
| Extremes | [`xtremax`](https://github.com/jejjohnson/xtremax) | block-maxima extraction, GEV distribution, return levels |
| Gaussian processes | [`pyrox`](https://github.com/jejjohnson/pyrox) | kernels, latent GP fields, variational inference |
| Dynamics | [`diffrax`](https://github.com/patrick-kidger/diffrax) | ODE/SDE integration for the time-varying trends (10–12) |

From xtremax the tutorials use
[`temporal_block_maxima`](xref:api#xtremax.extraction.temporal_block_maxima)
to reduce daily series to annual maxima,
[`GeneralizedExtremeValueDistribution`](xref:api#xtremax.distributions.GeneralizedExtremeValueDistribution)
as the NumPyro likelihood,
[`gev_log_prob`](xref:api#xtremax.primitives.gev_log_prob) and
[`gev_return_level`](xref:api#xtremax.primitives.gev_return_level) as pure-JAX
primitives, and
[`assemble_nonstationary_gev_fields`](xref:api#xtremax.primitives.assemble_nonstationary_gev_fields)
for the time-varying models.

:::{note} Pre-executed notebooks
These notebooks are committed **with their outputs** and are not re-run by the
docs build — the real-data path needs CDS credentials and a long,
resumable download. See [](#spatial-extremes-rerun) to reproduce them.
:::

## The build-up

Each notebook is short and adds exactly one idea.

**00 — Data.** [Pull daily near-surface air temperature](spatial_extremes/00_data/00_cds_insitu_iberia.ipynb)
for Spanish land stations from CDS with `xrreader`, cache it, and look at it.

**01–03 — Extreme-value foundations (one station).**
[01](spatial_extremes/01_evt_foundations/01_block_maxima.ipynb) turns a daily
series into annual maxima (`xtremax.extraction`);
[02](spatial_extremes/01_evt_foundations/02_gev_one_station.ipynb) fits a
generalized extreme value (GEV) distribution to one station and interprets
location/scale/shape $(\mu, \sigma, \xi)$;
[03](spatial_extremes/01_evt_foundations/03_extremal_types.ipynb) covers the
extremal-types theorem, and
[03 (return levels)](spatial_extremes/01_evt_foundations/03_return_levels.ipynb)
turns the fit into **return levels** with posterior uncertainty.

**04–06 — Pooling and Gaussian processes.**
[04](spatial_extremes/02_pooling/04_many_stations_independent.ipynb) fits every
station independently with NUTS, and
[04b](spatial_extremes/02_pooling/04_many_stations_laplace.ipynb) does the same
with a fast Laplace approximation, then maps the parameters — the noisy result
motivates pooling. [05](spatial_extremes/02_pooling/05_hierarchical_pooling.ipynb)
pools them with a **hierarchical** Bayesian model.
[06](spatial_extremes/02_pooling/06_gp_primer.ipynb) is a Gaussian-process
primer with `pyrox`: interpolate a field over `(lon, lat)`, then add physical
features (elevation, distance-to-coast, slope) and use ARD to see which
actually matter.

**07–09 — Spatial GEV models.** Tie the strands together — the GEV parameters
become latent GP fields, inferred with NumPyro.
[07](spatial_extremes/03_spatial_gev/07_spatial_gp_mu.ipynb) makes the
**location** $\mu(s)$ spatial;
[08](spatial_extremes/03_spatial_gev/08_spatial_gp_mu_sigma.ipynb) adds a
spatial **scale** $\sigma(s)$ driven by an elevation covariate;
[09](spatial_extremes/03_spatial_gev/09_spatial_gp_mu_sigma_xi.ipynb) frees the
**shape** $\xi(s)$ too, and asks honestly whether the tail carries any
recoverable geography.

**10–12 — Non-stationary in *time* (one long station).** Switch axes: take the
single longest record (Albacete, 1901–2025) and let the GEV location drift as
the climate warms, three escalating ways.
[10](spatial_extremes/04_nonstationary/10_nonstationary_parametric.ipynb) fits
a **parametric** linear trend $\mu(t)=\mu_0+\mu_1 z(t)$ (Coles' model) and turns
it into time-varying return levels.
[11](spatial_extremes/04_nonstationary/11_nonstationary_ode.ipynb) replaces the
line with a **mechanistic ODE** — a forced energy-balance relaxation integrated
with `diffrax` inside NUTS.
[12](spatial_extremes/04_nonstationary/12_nonstationary_gp.ipynb) goes
nonparametric with a **state-space Gaussian process** (a local-linear-trend /
integrated random walk, the stochastic sibling of the ODE), shows why a free
stationary GP over-fits a short record, and puts all three trends on one set of
axes.

(spatial-extremes-rerun)=
## Re-running the notebooks

The notebooks share a small helper package, `spatial_extremes`, that lives next
to them in `docs/tutorials/spatial_extremes/_helpers/` — it is tutorial glue
(data loading, station features, place names, cartopy maps), **not** part of
the installed `xtremax` package. Each notebook's first code cell puts
`_helpers/` on `sys.path`, so run them from anywhere inside the repository.

Its loader, `spatial_extremes.data`, serves **real CDS data when cached** and a
deterministic **synthetic** series otherwise — so the whole curriculum runs
offline with no credentials, just on synthetic numbers.

### 1. Install the tutorial dependencies

The extra stack lives in an optional `tutorials` dependency group, which a
plain `uv sync` does not install:

```bash
uv sync --group tutorials --group docs
```

It adds `xrreader[cds-insitu]` and a pinned pre-workspace-split `pyrox` (both
from GitHub, since neither is on PyPI yet), plus `cartopy`, `shapely`, `pyproj`,
`seaborn`, `loguru`, `optax` and `diffrax`.

### 2. (Optional) Fetch the real CDS data

Accept the licence for the
[in-situ surface land observations](https://cds.climate.copernicus.eu/datasets/insitu-observations-surface-land)
dataset, then provide credentials — exported in the environment, in a `.env`
file, or in `~/.cdsapirc` (see `docs/tutorials/spatial_extremes/.env.example`):

```bash
export CDSAPI_URL=https://cds.climate.copernicus.eu/api
export CDSAPI_KEY=<your-key>
```

Fetch once into the cache the notebooks read, and derive the covariates used
from notebook 06 on:

```bash
H=docs/tutorials/spatial_extremes/_helpers
uv run python $H/scripts/fetch_cds_insitu.py   # resumable; logs to .../logs/
uv run python $H/scripts/build_features.py     # elevation, distance-to-coast, slope
```

The cache lands in `docs/tutorials/spatial_extremes/data/` (gitignored); set
`CDS_INSITU_SCRATCH_ROOT` to put it elsewhere. `build_features.py` queries the
public OpenTopoData API for elevations, so it needs network access too.

### 3. Execute

```bash
uv run --group tutorials --group docs jupyter nbconvert --to notebook --execute \
  --inplace docs/tutorials/spatial_extremes/01_evt_foundations/01_block_maxima.ipynb
```

Each notebook reports whether it ran on the real record or the synthetic
fallback. Commit re-executed notebooks only when they ran on real data, so
the published outputs stay comparable.
