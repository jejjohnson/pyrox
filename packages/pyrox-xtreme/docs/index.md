# xtremax

> Extreme value theory for JAX — NumPyro-native EVT distributions, temporal / spatial / spatiotemporal point processes, and xarray-native extremes extraction.

Built on [JAX](https://github.com/jax-ml/jax), [NumPyro](https://github.com/pyro-ppl/numpyro), [equinox](https://github.com/patrick-kidger/equinox), and [xarray](https://github.com/pydata/xarray).

**New here?** Start with the [Vision](design_docs/vision.md) to understand why xtremax exists. The [Architecture](design_docs/architecture.md) design doc is a draft roadmap — it sketches where the library is headed, not the current module layout; the [API Reference](api/index.md) reflects what ships today.

## Installation

```bash
pip install xtremax
```

Or with `uv`:

```bash
uv add xtremax
```

## Quickstart

```python
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import xtremax

# NumPyro-native EVT distributions
gev = xtremax.GeneralizedExtremeValueDistribution(
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
        xtremax.GeneralizedExtremeValueDistribution(loc, scale, conc),
        obs=y,
    )

mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=1000)
mcmc.run(jr.key(1), maxima)
```

Extraction utilities stay in xarray — coordinates and metadata are preserved:

```python
import xtremax.extraction as xtx

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

## Links

- [API Reference](api/index.md)
- [Design docs](design_docs/README.md)
- [Interop](interop.md)
- [Changelog](changelog.md)
- [GitHub](https://github.com/jejjohnson/xtremax)
