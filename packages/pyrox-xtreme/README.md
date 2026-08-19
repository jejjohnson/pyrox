# xtremax

[![Tests](https://github.com/jejjohnson/xtremax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/xtremax/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/xtremax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/xtremax/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/xtremax/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/xtremax/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/xtremax/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/xtremax/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/xtremax/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/xtremax)
[![PyPI version](https://img.shields.io/pypi/v/xtremax.svg)](https://pypi.org/project/xtremax/)
[![Python versions](https://img.shields.io/pypi/pyversions/xtremax.svg)](https://pypi.org/project/xtremax/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

**Extreme value theory for JAX — NumPyro-native EVT distributions, temporal / spatial / spatiotemporal point processes, and xarray-native extremes extraction.**

Built on top of [JAX](https://github.com/jax-ml/jax), [NumPyro](https://github.com/pyro-ppl/numpyro), [equinox](https://github.com/patrick-kidger/equinox), and [xarray](https://github.com/pydata/xarray).

Extreme value analysis is central to climate risk, engineering reliability, finance, and insurance — yet the Python ecosystem forces practitioners to cobble together `scipy.stats`, hand-rolled log-likelihoods, and ad hoc xarray scripts. xtremax puts the whole workflow in one differentiable place: extract extremes from spatiotemporal data, fit Bayesian EVT and point-process models with NumPyro, and compute return levels with full posterior uncertainty.

## Installation

```bash
pip install xtremax
```

Or with `uv`:

```bash
uv add xtremax
```

## Quick Start

```python
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import xtremax

# NumPyro-native EVT distributions — sample, log_prob, cdf, icdf,
# moments, and return levels, all differentiable
gev = xtremax.GeneralizedExtremeValueDistribution(
    loc=30.0, scale=2.0, concentration=0.1
)
maxima = gev.sample(jr.key(0), (100,))
z100 = gev.return_level(100)  # 100-year return level

# Drop straight into NumPyro inference — NUTS, SVI, Predictive all work
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

Extracting extremes from spatiotemporal data stays in xarray — coordinates, dimensions, and metadata are preserved:

```python
import xtremax.extraction as xtx

# da: any xr.DataArray with a "time" dimension, e.g. daily station temperatures
annual_max = xtx.temporal_block_maxima(da, freq="YE")       # block maxima
u = xtx.quantile_threshold(da, quantile=0.95)               # POT threshold
peaks = xtx.decluster_runs(da, threshold=u)                 # independent peaks
```

## What's Inside

### Distributions — NumPyro-native EVT

Every distribution subclasses `numpyro.distributions.Distribution`, so it plugs directly into MCMC / SVI / `Predictive` and is differentiable end to end. Beyond `sample` / `log_prob`, each carries the full EVT toolkit: `cdf`, `icdf`, `survival_function`, `mean`, `variance`, `entropy`, and `return_level`.

| Distribution | Role |
|--------------|------|
| `GeneralizedExtremeValueDistribution` | GEV for block maxima — smoothly spans all three types via the concentration ξ |
| `GeneralizedParetoDistribution` | GPD for peaks-over-threshold exceedances |
| `GumbelType1GEVD` | Type I (ξ = 0): exponential tails |
| `FrechetType2GEVD` | Type II (ξ > 0): heavy polynomial tails |
| `WeibullType3GEVD` | Type III (ξ < 0): bounded upper tail |

### Primitives — pure JAX functions

The distribution math as standalone pure functions, for when you want the density without the distribution object (custom likelihoods, vmapped grids, non-stationary fields):

- **Per family** — `gev_*`, `gpd_*`, `gumbel_*`, `frechet_*`, `weibull_*` variants of `log_prob` | `cdf` | `icdf` | `mean` | `return_level`, plus `survival` / `log_survival` for the GEV and GPD families
- **Non-stationary EVT** — `design_matrix`, `assemble_nonstationary_gev_fields`, `nonstationary_return_level`, `nonstationary_return_period`, `expected_exceedances`
- **Spatial helpers** — `pairwise_distances`, `two_range_correlation`

### Extraction — xarray-native

Turn raw spatiotemporal data into modeling-ready extremes. Inputs are `xr.DataArray`s, and array-valued results preserve coordinates and metadata; fully reduced results (`constant_threshold`, scalar `quantile_threshold`, 1-D `estimate_extremal_index`) come back as plain floats:

| Group | Functions |
|-------|-----------|
| **Block maxima** | `temporal_block_maxima`, `spatial_block_maxima`, `sliding_block_maxima`, `r_largest_block_maxima`, `declustered_block_maxima` |
| **Thresholds** | `constant_threshold`, `quantile_threshold`, `rolling_threshold`, `seasonal_threshold`, `temporal_threshold`, quantile-regression thresholds (needs the `xtremax[threshold]` extra) |
| **Declustering** | `decluster_runs`, `decluster_separation`, `estimate_extremal_index` |

### Point Processes — three-tier API

Temporal, spatial, and spatiotemporal point processes — homogeneous / inhomogeneous Poisson, renewal, Hawkes (exponential and general kernels), marked variants, and thinning — each exposed at three levels:

| Tier | What it is | Examples |
|------|-----------|----------|
| **Primitives** | Pure functions: intensities, compensators, log-probs, samplers, diagnostics | `hpp_log_prob`, `ipp_sample_thinning`, `exp_hawkes_intensity`, `csr_ripleys_k` |
| **Operators** | Immutable `equinox.Module` process objects bundling intensity + sampling; temporal operators add time-rescaling goodness-of-fit | `InhomogeneousPoissonProcess`, `SpatioTemporalHawkes`, `MarkedTemporalPointProcess` |
| **Distributions** | NumPyro `Distribution` wrappers — use a point process as a likelihood in MCMC / SVI | `HomogeneousPoissonProcess`, `RenewalProcess`, `ExponentialHawkes` |

Shared infrastructure: `TemporalDomain` / `RectangularDomain` for observation windows, `EventHistory` for conditioning, structured `SampleResult` types, and `GoodnessOfFit` diagnostics (time-rescaling residuals, KS statistics, Ripley's K).

### Simulations — synthetic extremes

Generators for testing and teaching: `simulate_temp_extremes`, `simulate_precip_extremes`, `simulate_wind_extremes`, GMST trajectories (`generate_gmst_trajectory`, `generate_physical_gmst`), spatial fields with fractal terrain, and a ready-made Iberian demo domain.

## Documentation

- **[Documentation site](https://jejjohnson.github.io/xtremax/)** — examples, design docs, and full API reference
- **[Examples](https://jejjohnson.github.io/xtremax/)** — 20+ executed notebooks: GEV on block maxima, GPD peaks-over-threshold, non-stationary GEV (parametric trends, P-splines, neural ODEs), and every point process from scratch *and* through NumPyro
- **[Design docs](https://jejjohnson.github.io/xtremax/design_docs/)** — vision, architecture, API design, and decision records
- **[Interop](https://jejjohnson.github.io/xtremax/interop/)** — how xtremax composes with the wider ecosystem (NumPyro, pipekit) without depending on it

## Development

```bash
git clone https://github.com/jejjohnson/xtremax.git
cd xtremax
make install      # install all dependency groups + pre-commit hooks
make test         # run tests
make lint         # ruff check .
make typecheck    # ty check src/xtremax
make docs-serve   # preview docs locally
```

## License

MIT
