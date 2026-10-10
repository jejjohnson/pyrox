---
name: extreme-value-analysis-with-xtremax
description: Write extreme value analyses in JAX and NumPyro on xtremax — GEV, GPD, Gumbel, Fréchet and Weibull distributions that work as NumPyro likelihoods and latent sites, stable densities, quantiles and return levels through ξ = 0, non-stationary return levels, temporal / spatial / spatiotemporal point processes (Poisson, renewal, Hawkes, marked, thinned), and xarray block maxima, thresholds, declustering and extremal-index estimates. Use whenever a task fits a GEV or GPD, computes a return level or return period, models threshold exceedances or event times as a point process, or extracts block maxima / peaks over threshold from climate or other xarray data, in a project that uses (or could use) xtremax.
---

# Extreme value analysis on xtremax

xtremax gives you the extreme value distributions as NumPyro
`Distribution`s (so NUTS, SVI and `Predictive` work unchanged), their
maths as pure JAX functions that stay accurate and differentiable through
the Gumbel limit ξ → 0 and in the far tail, point processes as equinox
operators with NumPyro wrappers, and the extraction steps (block maxima,
thresholds, declustering) on xarray. Before writing a GEV density, a
return-level formula, a Poisson likelihood or a block-maxima loop, look it
up:

1. **The capability index** lists every public name with a one-line
   summary: <https://jejjohnson.github.io/xtremax/reference/capabilities/>.
   Or list the installed version:

   ```python
   import importlib, inspect

   for name in (
       "xtremax.distributions",
       "xtremax.primitives",
       "xtremax.extraction",
       "xtremax.point_processes",
       "xtremax.point_processes.primitives",
       "xtremax.point_processes.operators",
       "xtremax.point_processes.distributions",
       "xtremax.simulations",
   ):
       module = importlib.import_module(name)
       for attr in getattr(module, "__all__", []):
           obj = getattr(module, attr)
           if inspect.ismodule(obj):
               continue
           doc = (inspect.getdoc(obj) or "").split("\n")[0]
           print(f"{name}.{attr}: {doc}")
   ```

2. Compose what exists. Install from GitHub:
   `uv add "xtremax @ git+https://github.com/jejjohnson/xtremax.git"`
   (extra `[threshold]` for the scikit-learn quantile-regression
   threshold).

## What lives where

| You need… | Use |
|---|---|
| A GEV / GPD / Gumbel / Fréchet / Weibull likelihood or prior | `xtremax.GeneralizedExtremeValueDistribution(loc, scale, concentration)`, `xtremax.GeneralizedParetoDistribution(scale, concentration)`, `xtremax.GumbelType1GEVD`, `xtremax.FrechetType2GEVD`, `xtremax.WeibullType3GEVD` |
| Return levels, survival, hazards, moments of a fitted distribution | the distribution's methods: `return_level(T)`, `survival_function`, `hazard_rate`, `exceedance_probability`, `mean`, `variance`, `conditional_excess_mean`, … (the GEV class carries the full set) |
| The same maths on arrays (vectorised over posterior draws or a parameter field) | `xtremax.primitives`: `gev_log_prob`, `gev_cdf`, `gev_survival`, `gev_icdf`, `gev_return_level`, `gpd_*`, `gumbel_*`, … (ξ is called `shape` there) |
| Return levels / periods under a changing climate | `xtremax.primitives.nonstationary_return_level`, `nonstationary_return_period`, `expected_exceedances`, `assemble_nonstationary_gev_fields` |
| Site distances, design matrices, a copula correlation for spatial models | `xtremax.primitives.pairwise_distances`, `design_matrix`, `two_range_correlation` |
| Annual / seasonal / spatial block maxima | `xtremax.extraction.temporal_block_maxima` (`keep_time=True` records when), `spatial_block_maxima`, `sliding_block_maxima`, `r_largest_block_maxima`, `declustered_block_maxima` |
| A threshold, declustered peaks, the extremal index | `xtremax.extraction.quantile_threshold`, `seasonal_threshold`, `rolling_threshold`, `quantile_regression_threshold`, `decluster_runs`, `decluster_separation`, `estimate_extremal_index` |
| Event times as a Poisson, renewal, Hawkes, marked or thinned process | operators in `xtremax.point_processes.operators` (`HomogeneousPoissonProcess`, `InhomogeneousPoissonProcess`, `RenewalProcess`, `ExponentialHawkes`, `MarkedTemporalPointProcess`, `ThinningProcess`, spatial and spatiotemporal variants); the same names in `xtremax.point_processes.distributions` as NumPyro likelihoods |
| Goodness of fit of an event model | an operator's `goodness_of_fit(times, mask)`; `xtremax.point_processes.primitives.time_rescaling_residuals`, `ks_statistic_exp1` |
| Synthetic extremes for tests and demos | `xtremax.simulations` (`simulate_temp_extremes`, `simulate_precip_extremes`, `generate_gmst_trajectory`, …) |

Inference (MCMC, SVI, `Predictive`, masking) is NumPyro's; GP priors on
spatially varying parameters come from a GP library (pyrox-gp in the
xtremax tutorials).

## The rules your code must keep

- **Use the distributions as NumPyro sites.** `numpyro.sample("obs",
  GEV(loc, scale, xi), obs=y)`; ragged records use `d.mask(m)`, not a
  hand-written `numpyro.factor`. ξ is `concentration` (the old `shape=`
  spelling warns).
- **Initialise inside the data.** Start NUTS at the prior medians
  (`init_strategy=init_to_median`) with a location prior near the data:
  NumPyro's default init can start the location far below the data and
  strand the chain at ξ < -1, where the GEV likelihood is unbounded
  (hundreds of divergences and a degenerate fit).
- **Mind the sign convention against scipy.** xtremax uses ξ > 0 for heavy
  (Fréchet) tails; `scipy.stats.genextreme` takes `c = -ξ`,
  `scipy.stats.genpareto` takes `c = ξ`.
- **Return levels through the stable forms.** `gev_return_level(T, …)` /
  `gpd_return_level(T, …)` or `d.return_level(T)` on a GEV or GPD, not
  `icdf(1 - 1/T)` (which rounds to 1 for large T in float32), and
  `*_survival`, not `1 - cdf`. The Gumbel's `return_level` and
  `gumbel_return_level` still form `1 - 1/T`; for a large T use
  `gev_return_level(T, loc, scale, 0.0)`.
- **Point processes use padded buffers.** Events are a fixed-size `times`
  array plus a boolean `mask` whose `True` entries come first; padding
  slots hold the window end. Samplers take an explicit key and a static
  buffer size (`max_events`, or `max_candidates` for thinning samplers).
- **Keys are explicit** (`jax.random.split`, never reused); float32 is the
  default and is accurate enough — don't switch on x64 just to avoid NaNs.

## Worked example

```python
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import xarray as xr
from numpyro.infer import MCMC, NUTS, init_to_median

import xtremax
import xtremax.extraction as xtx
from xtremax.point_processes.distributions import HomogeneousPoissonProcess


# 60 years of daily maximum temperature, iid 𝒩(25, 3²) °C, as an xarray series
time = pd.date_range("1960-01-01", "2019-12-31", freq="D")
rng = np.random.default_rng(0)
da = xr.DataArray(25.0 + 3.0 * rng.standard_normal(time.size), coords={"time": time})
n_years = 60.0

# 1. Block maxima → GEV with NUTS
annual = xtx.temporal_block_maxima(da, freq="YE")  # (60,) annual maxima


def gev_model(y):
    loc = numpyro.sample("loc", dist.Normal(30.0, 10.0))
    scale = numpyro.sample("scale", dist.LogNormal(0.0, 1.0))
    xi = numpyro.sample("xi", dist.Normal(0.0, 0.3))
    gev = xtremax.GeneralizedExtremeValueDistribution(loc, scale, xi)
    numpyro.sample("obs", gev, obs=y)


key_gev, key_pot = jax.random.split(jax.random.PRNGKey(0))
# init_to_median starts at the prior medians; NumPyro's default init_to_uniform
# starts loc near 0, far below the data, and can strand the chain at ξ < -1,
# where the GEV likelihood is unbounded
nuts = NUTS(gev_model, init_strategy=init_to_median)
gev_fit = MCMC(nuts, num_warmup=500, num_samples=1000, progress_bar=False)
gev_fit.run(key_gev, jnp.asarray(annual.values))
post = gev_fit.get_samples()
# the 100-year return level, one value per posterior draw (primitives broadcast)
z100_bm = xtremax.gev_return_level(100.0, post["loc"], post["scale"], post["xi"])

# 2. Peaks over threshold: a threshold, declustered peaks, the extremal index
u = float(xtx.quantile_threshold(da, 0.98))  # 98th percentile
peaks = xtx.decluster_runs(da, threshold=u).dropna("time")  # one value per cluster
theta = float(xtx.estimate_extremal_index(da, threshold=u))  # 1 = no clustering
excess = jnp.asarray(peaks.values - u)  # (n,) cluster excesses
years = (peaks["time"].values - time.values[0]) / np.timedelta64(1, "D") / 365.25
max_events = 512  # static buffer for the padded (times, mask) event sequence
times = jnp.full(max_events, n_years).at[: years.size].set(jnp.asarray(years))
mask = jnp.arange(max_events) < years.size  # real events form a contiguous prefix


# 3. Poisson–GPD: cluster times as a homogeneous Poisson process, excesses as GPD
def pot_model(times, mask, excess):
    rate = numpyro.sample("rate", dist.LogNormal(0.0, 2.0))  # clusters per year
    scale = numpyro.sample("scale", dist.LogNormal(0.0, 1.0))
    xi = numpyro.sample("xi", dist.Normal(0.0, 0.3))
    events = HomogeneousPoissonProcess(rate, n_years)
    numpyro.sample("events", events, obs=(times, mask))
    gpd = xtremax.GeneralizedParetoDistribution(scale, xi)
    numpyro.sample("excess", gpd, obs=excess)


nuts = NUTS(pot_model, init_strategy=init_to_median)
pot_fit = MCMC(nuts, num_warmup=500, num_samples=1000, progress_bar=False)
pot_fit.run(key_pot, times, mask, excess)
post_pot = pot_fit.get_samples()
# exceeded once in 100 years = once in 100 · rate clusters
z100_pot = u + xtremax.gpd_return_level(
    100.0 * post_pot["rate"], post_pot["scale"], post_pot["xi"]
)
```

The exact 100-year level of this series is 25 + 3 Φ⁻¹(0.99^{1/365}) ≈
37.1 °C. Both routes recover it within their posterior spread: about
37.0 ± 0.9 °C from the 60 annual maxima and 36.8 ± 0.5 °C from the ~430
declustered peaks (more data, tighter). The posterior ξ is close to 0 in
both fits (Normal maxima are in the Gumbel domain), the extremal index is
≈ 0.98 (iid days barely cluster), and the Poisson rate is ≈ 7 clusters a
year (2 % of 365 days, less the few merged runs).

## Don't write it — use xtremax

| Don't write… | Use |
|---|---|
| `-log σ - (1 + 1/ξ) log t - t^(-1/ξ)` with a `jnp.where(ξ == 0, …)` Gumbel branch | `xtremax.primitives.gev_log_prob` or `GeneralizedExtremeValueDistribution(...).log_prob` (smooth and differentiable through ξ = 0) |
| `μ + σ/ξ ((-log(1 - 1/T))^(-ξ) - 1)` | `xtremax.primitives.gev_return_level(T, μ, σ, ξ)` or `d.return_level(T)` on a GEV |
| `1 - F(x)` for an exceedance probability | `xtremax.primitives.gev_survival`, `gpd_survival`, `d.exceedance_probability` |
| A GEV / GPD `numpyro.factor` | `numpyro.sample(..., xtremax.GeneralizedExtremeValueDistribution(...), obs=y)`; masked records with `.mask(m)` |
| `da.resample(time="YE").max()` plus an argmax for the date | `xtremax.extraction.temporal_block_maxima(da, freq="YE", keep_time=True)` |
| A runs-declustering loop, an extremal-index estimator | `xtremax.extraction.decluster_runs`, `decluster_separation`, `estimate_extremal_index` |
| `n * log(λ) - λ * T`, a Hawkes compensator, an Ogata thinning loop | `xtremax.point_processes.primitives.hpp_log_prob`, `exp_hawkes_log_prob`, `thinning_sample`; the operators in `xtremax.point_processes.operators` |
| A non-stationary return level by root-finding by hand | `xtremax.primitives.nonstationary_return_level` |

## Self-check before you finish

- Every GEV / GPD likelihood is an xtremax distribution (or primitive);
  no hand-written density, return level or `1 - cdf` tail.
- NUTS started with `init_to_median` (or another in-support init) and
  reports few or no divergences; posterior ξ is not stuck below -1.
- Any comparison with scipy uses `genextreme(c=-ξ)`.
- `jax.grad` of your log-likelihood is finite at ξ = 0.
- Event sequences are padded with the mask as a prefix; every random draw
  got its own key.

If xtremax lacks what you need, keep your addition small and shaped like
xtremax (a pure primitive plus a NumPyro `Distribution` that calls it, or a
point-process primitive plus an `equinox.Module` operator) and consider
proposing it upstream at <https://github.com/jejjohnson/xtremax/issues>.
