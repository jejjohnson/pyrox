<div align="center">

<img alt="pyrox-xtreme" src="../../docs/assets/icon-xtreme.svg" width="120">

# pyrox-xtreme

**Extreme value modelling for JAX: GEV and GPD likelihoods that stay smooth through ξ = 0 in float32, point processes in time, space and space-time, and extremes extracted from xarray, on NumPyro, equinox and xarray.**

[![Tests](https://github.com/jejjohnson/pyrox/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/pyrox/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/pyrox/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/pyrox/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/pyrox/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/pyrox/actions/workflows/typecheck.yml)
[![Docs](https://github.com/jejjohnson/pyrox/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/pyrox/)
[![codecov](https://codecov.io/gh/jejjohnson/pyrox/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/pyrox)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

📖 [Documentation](https://jejjohnson.github.io/pyrox/extremes/) ·
📓 [Examples](https://jejjohnson.github.io/pyrox/evt-distributions-overview/) ·
🗺️ [Spatial extremes tutorials](https://jejjohnson.github.io/pyrox/spatial-extremes/) ·
📚 [API reference](https://jejjohnson.github.io/pyrox/reference/xtreme/) ·
🧭 [Design](https://github.com/jejjohnson/pyrox/blob/main/design_docs/pyrox-xtreme/design_overview.md)

</div>

---

A hundred-year return level is usually estimated from forty or fifty years of data, so the answer lives in the part of the distribution the record has barely seen. Ordinary fits are dominated by the bulk, and the usual tools (`scipy.stats` fits, hand-written log-likelihoods, ad hoc xarray scripts) give a point estimate without its uncertainty. Extreme value theory says what the tail must look like instead: the largest value of a long block of observations follows the generalized extreme value (GEV) distribution,

```math
F(z) = \exp\left\{ -\left[ 1 + \xi \frac{z - \mu}{\sigma} \right]^{-1/\xi} \right\}
```

defined where $`1 + \xi (z - \mu) / \sigma > 0`$, with location $`\mu \in \mathbb{R}`$, scale $`\sigma > 0`$ and shape $`\xi \in \mathbb{R}`$ (`concentration` in the classes), each a scalar or an array broadcast to the batch. A positive $`\xi`$ gives the heavy Fréchet tail, a negative one the bounded Weibull tail, and $`\xi \to 0`$ the Gumbel limit $`F(z) = \exp(-e^{-(z - \mu)/\sigma})`$. pyrox-xtreme makes this a NumPyro likelihood, so the parameters, and every return level computed from them, come with a posterior.

```python
# Shapes: Y years (daily, loc, scale and xi as in the quickstart)

# today (pyrox-xtreme)
# y_k = max of the daily values x_d over year k
annual_max = temporal_block_maxima(daily, freq="YE")
# today (pyrox-xtreme)
# y_k ~ GEV(μ, σ, ξ), a NumPyro distribution
gev = GeneralizedExtremeValueDistribution(loc, scale, xi)
# today (pyrox-xtreme)
# z_100 = F⁻¹(1 − 1/100), the 100-year return level
z100 = gev.return_level(100.0)
```

None of it is new machinery. The distributions subclass [`numpyro`](https://github.com/pyro-ppl/numpyro)'s `Distribution` (and `GumbelType1GEVD` extends NumPyro's own `Gumbel`), so NUTS, SVI, `Predictive`, plates and masks work as they do for any NumPyro likelihood. All the maths is [`jax`](https://github.com/jax-ml/jax), the point-process operators are [`equinox`](https://github.com/patrick-kidger/equinox) modules, and the extraction functions take and return [`xarray`](https://github.com/pydata/xarray) DataArrays through `xr.apply_ufunc`. [`scipy`](https://scipy.org/) supplies the complex log-gamma of the Gumbel characteristic function and the simulators' ODE solver, and [`scikit-learn`](https://scikit-learn.org/) the quantile-regression thresholds behind the `threshold` extra.

pyrox-xtreme was the xtremax repository until it moved into the pyrox workspace (#454) with its history; the import changed from `xtremax` to `pyrox_xtreme` and nothing else in the API did. It imports nothing from the other pyrox packages; the [workspace README](../../README.md) introduces them.

<p align="center"><img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_xtreme_overview/pyrox_xtreme_overview.png" alt="pyrox-xtreme in one picture: you supply a record, events, covariates and a NumPyro model; one tail with shape ξ gives the GEV for block maxima, the GPD for excesses over a threshold and an intensity for events in time or space; pyrox-xtreme extracts the extremes, holds the EVT likelihoods, the pure-JAX maths, the point processes and synthetic climates; out come extremes on their grid, a posterior over μ, σ and ξ, return levels with intervals and an intensity with its fit; numpyro, jax, equinox, and xarray with scipy underneath" width="100%"></p>

## 🌟 Highlights

- 📈 **EVT distributions that NumPyro can sample**: `GeneralizedExtremeValueDistribution`, `GeneralizedParetoDistribution`, `GumbelType1GEVD`, `FrechetType2GEVD` and `WeibullType3GEVD` are NumPyro distributions whose supports have working bijectors, so each one is a likelihood or a latent site under NUTS, SVI and `Predictive`.
- 🧮 **Through ξ = 0 without a branch**: the primitives write the GEV and GPD kernels through `log1p(ξz)/ξ` and `expm1(ξz)/ξ` helpers, so values and gradients are smooth where the GEV becomes the Gumbel, and the suite checks them against `scipy.stats` on ξ grids that straddle zero, in float32.
- 🎯 **Return levels that hold for long periods**: `return_level` and `gev_return_level` start from the exceedance probability $`-\log(1 - 1/T)`$ rather than $`1 - 1/T`$, and `nonstationary_return_level` solves for the one level that $`N`$ blocks with changing parameters exceed $`N / T`$ times on average, differentiable through the implicit function theorem.
- ⏱️ **Point processes in three layers**: Poisson, renewal, Hawkes, marked and thinned processes in time, Poisson and marked processes in space, and Poisson, marked and Hawkes processes in space-time, each as pure functions, an equinox operator and a NumPyro wrapper; the Poisson, renewal and Hawkes operators in time also return time-rescaling goodness-of-fit diagnostics.
- 🗺️ **Extremes straight out of xarray**: temporal, spatial, sliding, r-largest and declustered block maxima, five kinds of threshold, runs and separation declustering and the extremal index, each taking a DataArray and returning one with its coordinates, and accepting dask-chunked input.
- 🌦️ **Synthetic climates for tests and teaching**: `pyrox_xtreme.simulations` draws GEV temperature, precipitation and wind maxima from location fields you build with its warming trajectories (`generate_gmst_trajectory`) and its Iberian domain (`create_iberian_domain`), seeded with an integer.
- 🚧 **No pyrox inside**: pyrox-xtreme never imports `pyrox` or a sibling package, which [`test_xtreme_import_guard.py`](tests/test_xtreme_import_guard.py) checks; a Gaussian-process field over GEV parameters comes from pyrox-gp in your own model, as the spatial-extremes tutorials do.

## 📐 Three routes into the tail

The record can be reduced to its extremes in three ways, and the package has a module for each step of each route.

| Route | What you keep | Model | Modules |
|---|---|---|---|
| Block maxima | the largest value of each block (a year, a season) | $`y_k \sim \mathrm{GEV}(\mu, \sigma, \xi)`$ | `temporal_block_maxima`, then `GeneralizedExtremeValueDistribution` |
| Peaks over threshold | the excess $`x - u`$ of each cluster above a threshold $`u`$ | $`x - u \mid x > u \sim \mathrm{GPD}(\sigma_u, \xi)`$ | `quantile_threshold`, `decluster_runs`, then `GeneralizedParetoDistribution` |
| Point process | the times (and places and sizes) of the events | an intensity $`\lambda(t)`$ | `pyrox_xtreme.point_processes`: Poisson, renewal, Hawkes, marked, thinned |

The first two routes share the shape $`\xi`$. Above a high enough threshold the excesses follow the generalized Pareto distribution,

```math
\Pr(X - u > y \mid X > u) = \left( 1 + \xi \frac{y}{\sigma_u} \right)^{-1/\xi}
```

where $`y \ge 0`$ is the excess and $`\sigma_u > 0`$ the scale at threshold $`u`$. The quantity a user reports is the $`T`$-year return level, the GEV quantile exceeded once in $`T`$ years on average,

```math
z_T = \mu + \frac{\sigma}{\xi} \left[ \left( -\log\left(1 - \frac{1}{T}\right) \right)^{-\xi} - 1 \right]
```

and the third route fits event times $`t_1 < \ldots < t_n`$ on a window $`[0, t_{\max}]`$ by the point-process log-likelihood

```math
\log L = \sum_{i=1}^{n} \log \lambda(t_i) - \int_0^{t_{\max}} \lambda(t) \mathrm{d}t
```

whose integral, the compensator, every family computes in closed form or by the quadrature in `integrate_log_intensity`. The [EVT distributions tour](https://jejjohnson.github.io/pyrox/evt-distributions-overview/) derives the first two and the [Poisson notebook](https://jejjohnson.github.io/pyrox/hpp-from-scratch-and-numpyro/) the third.

## 📥 Install

We have not put pyrox-xtreme on PyPI, and the name `pyrox` there belongs to an unrelated project, so install it from GitHub with [uv](https://docs.astral.sh/uv/). It brings in JAX, NumPyro, equinox, jaxtyping, xarray, NumPy, SciPy, pandas and einops, none of them from git, and needs Python 3.12 or newer.

```bash
uv add "pyrox-xtreme @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-xtreme"
```

One extra adds the rest: `threshold` brings in scikit-learn for `XarrayQuantileRegressor` and `quantile_regression_threshold`, which `pyrox_xtreme.extraction` exports only when it is installed.

```bash
uv add "pyrox-xtreme[threshold] @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-xtreme"
```

## 🌡️ Quickstart: a 100-year return level from forty years of daily data

The data are forty years of daily values $`x_d \sim \mathrm{Gumbel}(20, 2)`$, independent from day to day. The maximum of 365 such values is again Gumbel, with location $`20 + 2 \log 365`$, so the true 100-year return level is known: 41.0. We take the annual maxima $`y_k`$ with `temporal_block_maxima`, put weak priors on $`(\mu, \sigma, \xi)`$, and sample the posterior with NUTS; the model records $`z_{100}`$ for every draw. Everything runs in float32, JAX's default and the precision the package's tests hold it to.

```python
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import xarray as xr
from jaxtyping import Array, Float
from numpyro.infer import MCMC, NUTS, init_to_median
from pyrox_xtreme import GeneralizedExtremeValueDistribution
from pyrox_xtreme.extraction import temporal_block_maxima


# Shapes: D = 14 610 days; Y = 40 years; S = 2000 posterior draws

# x_d ~ Gumbel(20, 2), one value a day for forty years
# → (D,)
time = pd.date_range("1981-01-01", "2020-12-31", freq="D")
values = np.random.default_rng(0).gumbel(20.0, 2.0, time.size)
daily = xr.DataArray(values, coords={"time": time}, dims="time", name="tmax")

# today (pyrox-xtreme)
# y_k = max of x_d over year k
# (D,) → (Y,)
annual_max = temporal_block_maxima(daily, freq="YE")
maxima = jnp.asarray(annual_max.values, dtype=jnp.float32)


# user
# μ ~ 𝒩(30, 10²), σ ~ LogNormal(0, 1), ξ ~ 𝒩(0, 0.2²), y_k ~ GEV(μ, σ, ξ)
def model(y: Float[Array, " Y"] | None = None) -> None:
    loc = numpyro.sample("loc", dist.Normal(30.0, 10.0))
    scale = numpyro.sample("scale", dist.LogNormal(0.0, 1.0))
    xi = numpyro.sample("xi", dist.Normal(0.0, 0.2))
    # today (pyrox-xtreme)
    # z_100 = F⁻¹(1 − 1/100), the 100-year return level of each draw
    gev = GeneralizedExtremeValueDistribution(loc, scale, xi)
    numpyro.deterministic("z100", gev.return_level(100.0))
    with numpyro.plate("years", 40):
        numpyro.sample("obs", gev, obs=y)


# user
# draws from p(μ, σ, ξ | y) by NUTS, started at the prior medians
nuts = NUTS(model, init_strategy=init_to_median)
mcmc = MCMC(nuts, num_warmup=1000, num_samples=2000, progress_bar=False)
mcmc.run(jr.key(0), maxima)
draws = mcmc.get_samples()
# z_100 of the law the data came from: Gumbel(20 + 2 log 365, 2)
truth = 20.0 + 2.0 * np.log(365.0) - 2.0 * np.log(-np.log(1.0 - 1.0 / 100.0))

low, high = jnp.quantile(draws["z100"], jnp.array([0.05, 0.95]))
xi, z100 = draws["xi"].mean(), draws["z100"].mean()
print(f"xi {xi:.2f}  z100 {z100:.1f} [{low:.1f}, {high:.1f}]  truth {truth:.1f}")
```

This prints

```
xi -0.06  z100 41.7 [38.2, 47.8]  truth 41.0
```

that is, the posterior mean of $`\xi`$, close to the true 0 of a Gumbel, and the posterior mean of $`z_{100}`$ with its 90 % interval, which covers the true 41.0. The interval is wide because forty maxima say little about $`\xi`$, and that width is the honest answer a point estimate hides. `init_to_median` matters: NumPyro's default start puts $`\mu`$ near zero, far below the data, and the chain can strand at $`\xi < -1`$, where the GEV likelihood is unbounded. The same data can go the peaks-over-threshold route with `quantile_threshold`, `decluster_runs` and `GeneralizedParetoDistribution`, which the [GPD notebook](https://jejjohnson.github.io/pyrox/evt-gpd-pot/) works through, and a trend in $`\mu`$ turns `return_level` into `nonstationary_return_level`, which the [parametric-trend notebook](https://jejjohnson.github.io/pyrox/evt-nonstat-parametric/) uses.

## ⏱️ Events in time: a Hawkes process, simulated, fitted and checked

The point processes come in three layers, and one script touches all three. An operator simulates a self-exciting Hawkes process, whose intensity jumps by $`\alpha`$ at every event and decays at rate $`\beta`$; its NumPyro wrapper is the likelihood NUTS fits; and the operator rebuilt at the posterior mean checks the fit by time rescaling. The events sit in a padded buffer of $`B`$ slots whose mask marks the real ones.

```python
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Bool, Float
from numpyro.infer import MCMC, NUTS, init_to_median
from pyrox_xtreme.point_processes import distributions as ppd, operators as ppo


# Shapes: B = 512 buffer slots; S = 1000 posterior draws

# today (pyrox-xtreme)
# λ(t) = μ + Σ_{t_i < t} α e^{−β (t − t_i)}, with μ = 0.5, α = 0.8, β = 2 on [0, 200]
process = ppo.ExponentialHawkes(mu=0.5, alpha=0.8, beta=2.0, observation_window=200.0)
# today (pyrox-xtreme)
# event times in a padded buffer (real events a prefix of the mask), and the
# number of thinning proposals the sampler used, which must stay below the cap
# → (B,), (B,), ()
times, mask, n_proposals = process.sample(jr.key(0), 512, max_candidates=32768)


# user
# μ, α, β ~ LogNormal priors, and the event times ~ Hawkes(μ, α, β) on [0, 200]
def model(times: Float[Array, " B"], mask: Bool[Array, " B"]) -> None:
    mu = numpyro.sample("mu", dist.LogNormal(0.0, 1.0))
    alpha = numpyro.sample("alpha", dist.LogNormal(0.0, 1.0))
    beta = numpyro.sample("beta", dist.LogNormal(0.0, 1.0))
    # today (pyrox-xtreme)
    # log L = Σ_i log λ(t_i) − ∫ λ(t) dt over [0, 200], by the Ozaki recursion
    hawkes = ppd.ExponentialHawkes(mu, alpha, beta, 200.0, max_events=512)
    numpyro.sample("events", hawkes, obs=(times, mask))


nuts = NUTS(model, init_strategy=init_to_median)
mcmc = MCMC(nuts, num_warmup=500, num_samples=1000, progress_bar=False)
mcmc.run(jr.key(1), times, mask)
draws = mcmc.get_samples()
mean = {name: float(draws[name].mean()) for name in ("mu", "alpha", "beta")}
# today (pyrox-xtreme)
# the process at the posterior mean
posterior_mean = ppo.ExponentialHawkes(**mean, observation_window=200.0)
# today (pyrox-xtreme)
# τ_i = Λ(t_i) − Λ(t_{i−1}), the rescaled residuals, and their KS distance to Exp(1)
fit = posterior_mean.goodness_of_fit(times, mask)

ratio = (draws["alpha"] / draws["beta"]).mean()
print(
    f"events {int(mask.sum())}  proposals {int(n_proposals)}  "
    f"mu {mean['mu']:.2f}  alpha/beta {ratio:.2f}  KS {fit.ks_statistic:.3f}"
)
```

This prints `events 165  proposals 13946  mu 0.59  alpha/beta 0.30  KS 0.064`. The process is stationary with mean rate $`\mu / (1 - \alpha/\beta) = 0.83`$, so 165 events on $`[0, 200]`$ is what it should give, and the 13 946 proposals stayed below the cap of 32 768, so the buffer was not cut short. The posterior means of the baseline $`\mu`$ and the branching ratio $`\alpha / \beta`$ sit near the true 0.5 and 0.4, and a KS distance of 0.064 between the rescaled residuals and Exp(1) is below the 5 % critical value of about 0.106 for 165 events. The thinning sampler bounds the intensity by $`\mu + \alpha k`$ after $`k`$ events, which is loose, so a long window needs a generous `max_candidates`; the default of four proposals per buffer slot truncates this process near $`t = 73`$. The [Hawkes notebook](https://jejjohnson.github.io/pyrox/hawkes-from-scratch-and-numpyro/) builds the same process from the primitives up.

## 🔬 What else is in the box

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_xtreme_roles/pyrox_xtreme_roles-dark.svg">
    <img alt="pyrox-xtreme on the GeoStack role ladder: the samplers and the non-stationary return-level solve as algorithms; the five EVT distributions, the fourteen point processes, their NumPyro wrappers and the quantile regressor as models; the Hawkes kernel, the piecewise intensity and the retention and mark adapters as strategies; the EVT maths, the point-process functions, the quadrature, the diagnostics, the extraction and the models' methods as primitives; the domains, event histories and sample results as structures; NumPyro's inference and the synthetic-climate generators beside the ladder" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_xtreme_roles/pyrox_xtreme_roles-light.svg" width="100%">
  </picture>
</p>

Each public name sits on one rung of the GeoStack role ladder, by the job it does; the figure shows a few names per group, and the table below lists them all.

`import pyrox_xtreme` exports the five distributions and every primitive; each subpackage has its own `__all__`.

| Module | Symbols | Notes |
|---|---|---|
| `pyrox_xtreme.distributions` | `GeneralizedExtremeValueDistribution`, `GeneralizedParetoDistribution`, `GumbelType1GEVD`, `FrechetType2GEVD`, `WeibullType3GEVD` | NumPyro distributions with survival, hazard, moments, `return_level` and mean excess |
| `pyrox_xtreme.primitives` | `gev_log_prob`, `gev_cdf`, `gev_icdf`, `gev_survival`, `gev_log_survival`, `gev_mean`, `gev_return_level`, and the same for `gpd_*`; `gumbel_*`, `frechet_*` and `weibull_*` without the survival pair | Pure JAX; Fréchet and Weibull delegate to the GEV |
| | `nonstationary_return_level`, `nonstationary_return_period`, `expected_exceedances`, `assemble_nonstationary_gev_fields`, `design_matrix` | Return levels and periods over changing parameters |
| | `pairwise_distances`, `two_range_correlation` | Building blocks for spatial models |
| `pyrox_xtreme.extraction` | `temporal_block_maxima`, `spatial_block_maxima`, `sliding_block_maxima`, `r_largest_block_maxima`, `declustered_block_maxima` | Block maxima on DataArrays |
| | `constant_threshold`, `quantile_threshold`, `rolling_threshold`, `seasonal_threshold`, `temporal_threshold`, `quantile_regression_threshold`, `XarrayQuantileRegressor` | Thresholds; the last two need the `threshold` extra |
| | `decluster_runs`, `decluster_separation`, `estimate_extremal_index` | Declustering and the extremal index |
| `pyrox_xtreme.point_processes` | `TemporalDomain`, `RectangularDomain`, `EventHistory`, `SampleResult` and its marked, spatial and spatiotemporal siblings, `GoodnessOfFit`, `integrate_log_intensity` and its spatial and spatiotemporal forms, retention and mark adapters | What the three layers share |
| `point_processes.primitives` | `hpp_*`, `ipp_*`, `renewal_*`, `exp_hawkes_*`, `general_hawkes_*`, `stpp_hawkes_*`, `thinning_sample`, `time_rescaling_residuals`, `ks_statistic_exp1`, `csr_ripleys_k`, … (85 names) | Intensities, compensators, log-likelihoods, samplers and diagnostics |
| `point_processes.operators` | `HomogeneousPoissonProcess`, `InhomogeneousPoissonProcess`, `RenewalProcess`, `ExponentialHawkes`, `GeneralHawkesProcess`, `MarkedTemporalPointProcess`, `ThinningProcess`, the spatial and spatiotemporal processes, `ExponentialKernel`, `PiecewiseConstantLogIntensity` | equinox modules; the temporal Poisson, renewal and Hawkes ones carry goodness of fit |
| `point_processes.distributions` | the fourteen processes above, named as their operators | NumPyro wrappers, for a process as a likelihood |
| `pyrox_xtreme.simulations` | `simulate_temp_extremes`, `simulate_precip_extremes`, `simulate_wind_extremes`, `generate_gmst_trajectory`, `generate_physical_gmst`, `create_iberian_domain`, `generate_spatial_field`, `generate_fractal_terrain`, … | NumPy generators seeded with an integer |

## 📓 Tutorials

| # | Tutorial | What it shows |
|---|---|---|
| 1 | [EVT distributions tour](https://jejjohnson.github.io/pyrox/evt-distributions-overview/) | the five distributions, their tails, and how the GEV and the GPD share $`\xi`$ |
| 2 | [GEV on block maxima](https://jejjohnson.github.io/pyrox/evt-gevd-block-maxima/), [Fréchet](https://jejjohnson.github.io/pyrox/evt-frechet-heavy-tails/), [Weibull](https://jejjohnson.github.io/pyrox/evt-weibull-bounded/) | annual maxima fitted by NUTS, with heavy and with bounded tails |
| 3 | [GPD on peaks over threshold](https://jejjohnson.github.io/pyrox/evt-gpd-pot/), [quantile-regression thresholds](https://jejjohnson.github.io/pyrox/evt-quantile-regression-gpd/) | thresholds, declustering and GPD fits, then a threshold that moves with the covariates |
| 4 | [Parametric trend](https://jejjohnson.github.io/pyrox/evt-nonstat-parametric/), [P-splines](https://jejjohnson.github.io/pyrox/evt-nonstat-splines/), [neural ODE](https://jejjohnson.github.io/pyrox/evt-nonstat-neural-ode/) | a non-stationary GEV whose location $`\mu(t)`$ changes in three ways |
| 5 | Temporal processes: [Poisson](https://jejjohnson.github.io/pyrox/hpp-from-scratch-and-numpyro/), [inhomogeneous Poisson](https://jejjohnson.github.io/pyrox/ipp-from-scratch-and-numpyro/), [renewal](https://jejjohnson.github.io/pyrox/renewal-from-scratch-and-numpyro/), [Hawkes](https://jejjohnson.github.io/pyrox/hawkes-from-scratch-and-numpyro/), [marked](https://jejjohnson.github.io/pyrox/marked-from-scratch-and-numpyro/), [thinning](https://jejjohnson.github.io/pyrox/thinning-from-scratch-and-numpyro/) | each process written from scratch, then fitted through its NumPyro wrapper |
| 6 | Spatial processes: [Poisson](https://jejjohnson.github.io/pyrox/hpp-spatial-from-scratch-and-numpyro/), [inhomogeneous Poisson](https://jejjohnson.github.io/pyrox/ipp-spatial-from-scratch-and-numpyro/), [marked](https://jejjohnson.github.io/pyrox/marked-spatial-from-scratch-and-numpyro/) | the same build-up in a rectangle |
| 7 | Spatiotemporal processes: [Poisson](https://jejjohnson.github.io/pyrox/hpp-spatiotemporal-from-scratch-and-numpyro/), [inhomogeneous Poisson](https://jejjohnson.github.io/pyrox/ipp-spatiotemporal-from-scratch-and-numpyro/), [marked](https://jejjohnson.github.io/pyrox/marked-spatiotemporal-from-scratch-and-numpyro/), [Hawkes](https://jejjohnson.github.io/pyrox/hawkes-spatiotemporal-from-scratch-and-numpyro/) | space and time together, up to an ETAS-style Hawkes process |
| 8 | [Latent ODE intensity](https://jejjohnson.github.io/pyrox/latent-ode-ipp/) | an inhomogeneous Poisson process observed irregularly, with a neural ODE for its intensity |
| 9 | [Spatial extremes](https://jejjohnson.github.io/pyrox/spatial-extremes/) (15 notebooks) | station temperatures over Iberia from the Copernicus Climate Data Store: block maxima, a GEV at one station, pooling across stations, GEV parameters as Gaussian-process fields with pyrox-gp, and non-stationary trends |

## 📖 Documentation

- **[Documentation](https://jejjohnson.github.io/pyrox/extremes/)**: the extremes section of the pyrox site, with the examples and the tutorials above.
- **[API reference](https://jejjohnson.github.io/pyrox/reference/xtreme/)**: every public name, generated from the docstrings, on one page per subpackage.
- **[Interop](https://jejjohnson.github.io/pyrox/interop/)**: how the GEV and GPD satisfy pipekit's observation-noise protocol and `XarrayQuantileRegressor` its predictor protocol, without importing pipekit.
- **[Design docs](https://github.com/jejjohnson/pyrox/blob/main/design_docs/pyrox-xtreme/design_overview.md)**: the vision, boundaries, architecture and decisions; they also describe parts that are not built (a model zoo, max-stable processes), and the code wins where they disagree.
- **[Capability index](https://jejjohnson.github.io/pyrox/capabilities/)**: every public name in the workspace, to check before writing something new.
- **[`AGENTS.md`](AGENTS.md)**: the contracts of each layer (stable forms, supports, padded buffers, xarray in and out) and the tests that hold them.

## 🧩 Ecosystem

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_xtreme_ecosystem/pyrox_xtreme_ecosystem-dark.svg">
    <img alt="No package imports pyrox-xtreme yet; plumax's population extra still imports xtremax.point_processes lazily from the archived xtremax repository, drawn dashed in GeoModels; pyrox-xtreme builds on JAX, NumPyro, Equinox, xarray, NumPy, SciPy, pandas, einops and jaxtyping, and on scikit-learn through its threshold extra" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_xtreme_ecosystem/pyrox_xtreme_ecosystem-light.svg" width="100%">
  </picture>
</p>

pyrox-xtreme is a package of the pyrox [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/), in the GeoML layer of the [GeoStack](https://github.com/jejjohnson/geostack). It owns the extreme-value maths, the point processes and the extraction, and composes the rest in:

| Concern | Owner |
|---|---|
| The `Distribution` base, constraints, NUTS, SVI, `Predictive`, plates and masks | [`numpyro`](https://github.com/pyro-ppl/numpyro) |
| Arrays, `jit`, `grad`, `vmap` and PRNG keys | [`jax`](https://github.com/jax-ml/jax) |
| Point-process operators, domains and event histories as pytrees | [`equinox`](https://github.com/patrick-kidger/equinox) |
| Labelled arrays and chunked computation for extraction | [`xarray`](https://github.com/pydata/xarray) |
| The Gumbel characteristic function and the simulators' ODEs | [`scipy`](https://scipy.org/) |
| Quantile-regression thresholds (the `threshold` extra) | [`scikit-learn`](https://scikit-learn.org/) |

Dependencies point one way: pyrox-xtreme imports no pyrox package and no other GeoStack library, and geostack's dependency graph places it in GeoML tier 1, below the pyrox core. Nothing imports `pyrox_xtreme` yet. plumax's `population` extra still pins the archived xtremax repository by commit, and moving it to pyrox-xtreme is a plumax change.

## 🌳 Repository layout

```
packages/pyrox-xtreme/
├── src/pyrox_xtreme/
│   ├── __init__.py          ← the five distributions and every primitive
│   ├── primitives/          ← pure JAX: GEV, GPD, Gumbel, Fréchet, Weibull, non-stationary, spatial
│   ├── distributions/       ← the NumPyro distributions
│   ├── point_processes/
│   │   ├── primitives/      ← intensities, compensators, log-likelihoods, samplers
│   │   ├── operators/       ← equinox process objects
│   │   └── distributions/   ← NumPyro wrappers around the operators
│   ├── extraction/          ← block maxima, thresholds, declustering on xarray
│   └── simulations/         ← synthetic extremes and domains
├── tests/                   ← scipy oracles, gradients, point-process invariants, extraction
└── AGENTS.md                ← the contracts and the tests that hold them
```

The docs live at the workspace root, in `docs/xtreme/` (examples and tutorials) and `docs/api/xtreme/`, and the design docs in `design_docs/pyrox-xtreme/`.

## ⚙️ Development

The package is developed inside the workspace, so clone the repository and run the tasks from its root.

```bash
git clone https://github.com/jejjohnson/pyrox.git
cd pyrox
make install                                                                         # uv sync --all-groups + pre-commit hooks
PYROX_TEST_X64=0 uv run pytest --no-cov -m "not slow and not integration" packages/pyrox-xtreme/tests   # the fast tests, in float32 as CI runs them
make lint                                                                            # ruff check on the whole repo
make typecheck                                                                       # ty check on every package
```

`make help` lists every task. The root [`AGENTS.md`](../../AGENTS.md) holds the rules for the whole workspace, and this package's [`AGENTS.md`](AGENTS.md) adds the numerical contracts and the test markers.

## 🎓 References

- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.
- Embrechts, P., Klüppelberg, C., & Mikosch, T. (1997). *Modelling Extremal Events*. Springer.
- Beirlant, J., Goegebeur, Y., Segers, J., & Teugels, J. (2004). *Statistics of Extremes*. Wiley.
- Lewis, P. A. W., & Shedler, G. S. (1979). [Simulation of nonhomogeneous Poisson processes by thinning](https://doi.org/10.1002/nav.3800260304). *Naval Research Logistics Quarterly* 26, 403–413.
- Ozaki, T. (1979). [Maximum likelihood estimation of Hawkes' self-exciting point processes](https://doi.org/10.1007/BF02480272). *Annals of the Institute of Statistical Mathematics* 31, 145–155.
- Ogata, Y. (1981). [On Lewis' simulation method for point processes](https://doi.org/10.1109/TIT.1981.1056305). *IEEE Transactions on Information Theory* 27, 23–31.

## 📄 License

MIT. See [LICENSE](LICENSE).
