# pyrox-xtreme

> Extreme value theory for JAX — NumPyro-native EVT distributions, temporal / spatial / spatiotemporal point processes, and xarray-native extremes extraction.

Built on [JAX](https://github.com/jax-ml/jax), [NumPyro](https://github.com/pyro-ppl/numpyro), [equinox](https://github.com/patrick-kidger/equinox), and [xarray](https://github.com/pydata/xarray).

**New here?** Start with the [Vision](https://github.com/jejjohnson/pyrox/blob/main/design_docs/pyrox-xtreme/vision.md) to understand why pyrox-xtreme exists. The [Architecture](https://github.com/jejjohnson/pyrox/blob/main/design_docs/pyrox-xtreme/architecture.md) design doc is a draft roadmap — it sketches where the library is headed, not the current module layout; the [API reference](https://jejjohnson.github.io/pyrox/reference/xtreme/) reflects what ships today.

## Installation

pyrox-xtreme is not on PyPI yet. Install it from its subdirectory of the pyrox repository with [uv](https://docs.astral.sh/uv/); it does not pull in `pyrox`:

```bash
uv add "pyrox-xtreme @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-xtreme"
```

The `[threshold]` extra adds scikit-learn for the quantile-regression thresholds.

## Quickstart

```python
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import pyrox_xtreme

# NumPyro-native EVT distributions
gev = pyrox_xtreme.GeneralizedExtremeValueDistribution(
    loc=30.0, scale=2.0, concentration=0.1
)
maxima = gev.sample(jr.key(0), (100,))
z100 = gev.return_level(100)  # 100-year return level

# Bayesian inference with NUTS
def model(y):
    loc = numpyro.sample("loc", dist.Normal(30.0, 10.0))
    scale = numpyro.sample("scale", dist.LogNormal(0.0, 1.0))
    conc = numpyro.sample("concentration", dist.Normal(0.0, 0.3))
    numpyro.sample(
        "obs",
        pyrox_xtreme.GeneralizedExtremeValueDistribution(loc, scale, conc),
        obs=y,
    )

mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=1000)
mcmc.run(jr.key(1), maxima)
```

Extraction utilities stay in xarray — coordinates and metadata are preserved:

```python
import pyrox_xtreme.extraction as xtx

# da: any xr.DataArray with a "time" dimension, e.g. daily station temperatures
annual_max = xtx.temporal_block_maxima(da, freq="YE")
u = xtx.quantile_threshold(da, quantile=0.95)
peaks = xtx.decluster_runs(da, threshold=u)
```

## Examples

**Extreme value distributions** — pick the tail behaviour:

- [Distributions overview](notebooks/evt_distributions_overview.ipynb) — the whole family side by side
- [GEVD: annual block maxima](notebooks/evt_gevd_block_maxima.ipynb) — the canonical block-maxima workflow
- [Fréchet: heavy-tailed maxima](notebooks/evt_frechet_heavy_tails.ipynb) and [Weibull: bounded extremes](notebooks/evt_weibull_bounded.ipynb) — fixed-type fits

**Peaks over threshold** — use more of the data than block maxima:

- [GPD: peaks-over-threshold](notebooks/evt_gpd_pot.ipynb) — threshold selection, declustering, and GPD fits
- [GPD: quantile-regression threshold](notebooks/evt_quantile_regression_gpd.ipynb) — trend-following thresholds

**Non-stationary EVT** — when the climate moves under you:

- [Parametric trend](notebooks/evt_nonstat_parametric.ipynb) — covariate-dependent parameters
- [P-splines](notebooks/evt_nonstat_splines.ipynb) — smooth non-parametric trends
- [Neural ODE](notebooks/evt_nonstat_neural_ode.ipynb) — learned parameter dynamics

**Point processes** — each built from scratch with primitives, then fit through NumPyro:

- *Temporal*: [homogeneous Poisson](notebooks/hpp_from_scratch_and_numpyro.ipynb), [inhomogeneous Poisson](notebooks/ipp_from_scratch_and_numpyro.ipynb), [renewal](notebooks/renewal_from_scratch_and_numpyro.ipynb), [Hawkes](notebooks/hawkes_from_scratch_and_numpyro.ipynb), [marked](notebooks/marked_from_scratch_and_numpyro.ipynb), [thinning](notebooks/thinning_from_scratch_and_numpyro.ipynb)
- *Spatial*: [homogeneous](notebooks/hpp_spatial_from_scratch_and_numpyro.ipynb), [inhomogeneous](notebooks/ipp_spatial_from_scratch_and_numpyro.ipynb), [marked](notebooks/marked_spatial_from_scratch_and_numpyro.ipynb)
- *Spatiotemporal*: [homogeneous](notebooks/hpp_spatiotemporal_from_scratch_and_numpyro.ipynb), [inhomogeneous](notebooks/ipp_spatiotemporal_from_scratch_and_numpyro.ipynb), [marked](notebooks/marked_spatiotemporal_from_scratch_and_numpyro.ipynb), [Hawkes / ETAS](notebooks/hawkes_spatiotemporal_from_scratch_and_numpyro.ipynb)
- *Neural*: [latent ODE for irregular IPPs](notebooks/latent_ode_ipp.ipynb)

**Tutorials** — a longer, real-data curriculum:

- [Spatial extremes](tutorials/spatial_extremes.md) — from one station's annual maxima to spatial GEV fields and non-stationary trends, on CDS station data over Spain

## Links

- [API reference](https://jejjohnson.github.io/pyrox/reference/xtreme/)
- [Design docs](https://github.com/jejjohnson/pyrox/blob/main/design_docs/pyrox-xtreme/design_overview.md)
- [Interop](interop.md)
- [Changelog](https://github.com/jejjohnson/pyrox/blob/main/packages/pyrox-xtreme/CHANGELOG.md)
- [Source](https://github.com/jejjohnson/pyrox/tree/main/packages/pyrox-xtreme)
