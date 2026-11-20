<div align="center">

<img alt="pyrox-lgm" src="../../docs/assets/icon-lgm.svg" width="120">

# pyrox-lgm

**Latent Gaussian models for JAX and NumPyro: GMRF components in precision form, penalised-complexity priors and an `inla()` driver checked against R-INLA, on pyrox, pyrox-gauss, gaussx, kernellib, functax, argminax, optimistix and matfree.**

[![Tests](https://github.com/jejjohnson/pyrox/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/pyrox/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/pyrox/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/pyrox/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/pyrox/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/pyrox/actions/workflows/typecheck.yml)
[![Docs](https://github.com/jejjohnson/pyrox/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/pyrox/)
[![codecov](https://codecov.io/gh/jejjohnson/pyrox/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/pyrox)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

📖 [Documentation](https://jejjohnson.github.io/pyrox/) ·
📚 [LGM API](https://jejjohnson.github.io/pyrox/reference/lgm/) ·
🚀 [MCMC with INLA](https://jejjohnson.github.io/pyrox/lgm-mcmc-inla/) ·
🗂️ [Capability index](https://jejjohnson.github.io/pyrox/capabilities/)

</div>

---

Disease counts by week and district, rainfall on a grid, house prices by neighbourhood: each is a field over time or space seen through a noisy, often non-Gaussian likelihood, and each wants a smooth trend, a spatial effect and their uncertainties. Such a field is sparse in its *precision*: a week depends on its neighbours, a district on the districts it borders. pyrox-lgm writes these models as latent Gaussian models, a sum of components with sparse precision matrices under a likelihood,

```math
p(\mathbf{y}, \mathbf{x}, \boldsymbol{\theta}) = \prod_{i=1}^{N} p(y_i \mid \eta_i, \boldsymbol{\theta}) \ \mathcal{N}(\mathbf{x} \mid \mathbf{0}, \mathbf{Q}(\boldsymbol{\theta})^{-1}) \ p(\boldsymbol{\theta})
```

where $`\mathbf{y} \in \mathbb{R}^{N}`$ holds the $`N`$ observations, $`\boldsymbol{\eta} = \mathbf{o} + \mathbf{A} \mathbf{x} \in \mathbb{R}^{N}`$ is the linear predictor with a known offset $`\mathbf{o}`$, $`\mathbf{x} \in \mathbb{R}^{n}`$ stacks the fixed effects and the latent field of every component, $`\mathbf{A} \in \mathbb{R}^{N \times n}`$ maps it to the observations, $`\mathbf{Q}(\boldsymbol{\theta}) \in \mathbb{R}^{n \times n}`$ is its sparse, block-diagonal precision, and $`\boldsymbol{\theta}`$ holds the components' hyperparameters (a precision, a correlation, a range) and the likelihood's. `inla()` approximates the posterior of $`\boldsymbol{\theta}`$ and every latent marginal without sampling; the same components also sample inside an ordinary NumPyro model, so NUTS can check it.

```python
# Shapes: N observations; n latent nodes; T time points
# (counts, expected, weeks and the rest as in the quickstart)

# today (pyrox-lgm)
# x ~ RW2(τ): a second-order random walk over T weeks, τ under a PC prior
trend = lgm.RW2(104, name="week")
# today (pyrox-lgm)
# log λ_i = log E_i + β₀ + x_week(i), y_i ~ Poisson(λ_i)
model = lgm.LGM(
    components=(trend,),
    fixed=lgm.FixedEffects(("intercept",)),
    likelihood=lgm.Poisson(),
)
# today (pyrox-lgm)
# π(θ | y) and π(x_j | y) by nested Laplace approximations
result = lgm.inla(model, {"y": counts, "offset": np.log(expected), "week": weeks})
# today (pyrox-lgm)
# the same component as NumPyro sites, for NUTS: week_tau and week
field = trend.sample(weeks)
```

None of it is new machinery. Every precision is a [`gaussx`](https://github.com/jejjohnson/gaussx) GMRF on a sparse operator, factorised once symbolically and reused across $`\boldsymbol{\theta}`$; the graphs of the areal components and their structure matrices are [`kernellib`](https://github.com/jejjohnson/kernellib)'s; the SPDE's mesh geometry and interpolation are [`functax`](https://github.com/jejjohnson/functax)'s. The inner Laplace fit is [`argminax`](https://github.com/jejjohnson/argminax)'s damped Newton loop (`argminax.solve.laplace_mode`), the search for the mode of $`\boldsymbol{\theta}`$ is [`optimistix`](https://github.com/patrick-kidger/optimistix)'s L-BFGS, and the Lanczos spectra the PC priors need come from [`matfree`](https://github.com/pnkraemer/matfree). The observation models build on the [`pyrox`](../pyrox) core's `AbstractObservation`, the NumPyro face of each component draws its field from [`pyrox-gauss`](../pyrox-gauss)'s `GMRFDistribution`, and the PC priors are [`numpyro`](https://github.com/pyro-ppl/numpyro) distributions.

pyrox-lgm is one package of the pyrox workspace; the [workspace README](../../README.md) introduces the others.

## 🌟 Highlights

- 🧱 **Components in precision form**: `IID`, `RW1`, `RW2`, `AR1` and `MarkovGP` over time, `Besag`, `BYM2`, `CAR` and `Leroux` over areas, `SPDE` for a Matérn field on a mesh or a grid, `Generic` for any structure matrix, and `Kronecker` and `Replicate` to combine them; each keeps its precision sparse.
- 🎯 **Penalised-complexity priors**: `PCPrecision`, `PCAR1Rho`, `PCBYM2Phi` and `PCMatern`, R-INLA's `pc.prec`, `pc.cor0`, the BYM2 mixing prior and `inla.spde2.pcmatern`, as NumPyro distributions calibrated against numerical integration in the tests; each component defaults to its matching one.
- 🧮 **`inla()` as R-INLA does it**: the VB mean correction by default (R-INLA's default since 22.11), the Gaussian or simplified-Laplace strategies, and R-INLA's integration design over $`\boldsymbol{\theta}`$ (a grid for one or two hyperparameters, a CCD above), whose latent means match R-INLA 26.8.7 to within about $`10^{-3}`$, relative, on five reference cases (Gaussian RW2 and AR1, Poisson BYM2 on the Scotland data, Poisson SPDE, Bernoulli).
- 📝 **Formula sugar**: `f("week", "rw2", n=52)` and `f("region", "bym2", graph=g)` build a component named after the data column it indexes, as R-INLA's `f()` does.
- 🩺 **Diagnostics without refits**: `diagnostics(result)` gives DIC, WAIC, CPO, PIT and the log score from one `inla()` run.
- 🎲 **The same model by NUTS**: every component's `.sample()` registers its hyperparameters and its field as NumPyro sites, soft-constraining an intrinsic field, so a component sits in any NumPyro model and NUTS can check `inla()`.
- 🚧 **No GP package needed**: pyrox-lgm never imports pyrox-gp, which [`test_import_guard.py`](tests/test_import_guard.py) checks; a temporal GP enters as `MarkovGP`, a gaussx SDE kernel in precision form.

## 📐 One latent field, many components

A model is the sum of its components on the scale of the linear predictor, $`\boldsymbol{\eta} = \mathbf{o} + \mathbf{X} \boldsymbol{\beta} + \sum_c \mathbf{A}_c \mathbf{x}_c`$, each component $`\mathbf{x}_c \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_c(\boldsymbol{\theta})^{-1})`$ with its own sparse precision, so $`\mathbf{Q}`$ is block-diagonal and stays sparse.

| Family | Components | Precision $`\mathbf{Q}_c`$ | Reach for it when |
|---|---|---|---|
| Unstructured | `IID` | $`\tau \mathbf{I}`$ | overdispersion, a random intercept per group |
| Temporal | `RW1`, `RW2`, `AR1`, `MarkovGP` | differences of neighbouring times, or a Markov GP's state | a smooth trend or seasonality over time |
| Areal | `Besag`, `BYM2`, `CAR`, `Leroux` | $`\tau (\mathbf{D} - \mathbf{W})`$ and its proper or mixed forms on a graph | counts or rates by district |
| Spatial | `SPDE` | the Matérn SPDE on a mesh or a grid | a continuous spatial field from points |
| Generic | `Generic` | $`\tau \mathbf{R}`$ for your structure $`\mathbf{R}`$ | a structure no other component has |
| Combined | `Kronecker`, `Replicate` | $`\mathbf{Q}_\text{main} \otimes \mathbf{Q}_\text{group}`$, or copies sharing $`\boldsymbol{\theta}`$ | space-time interaction, one field per group |

`inla()` integrates the latent field out at each $`\boldsymbol{\theta}`$ with a Laplace approximation, giving the posterior of the hyperparameters

```math
\tilde{\pi}(\boldsymbol{\theta} \mid \mathbf{y}) \propto \frac{p(\mathbf{y}, \hat{\mathbf{x}}(\boldsymbol{\theta}), \boldsymbol{\theta})}{\tilde{\pi}_G(\hat{\mathbf{x}}(\boldsymbol{\theta}) \mid \mathbf{y}, \boldsymbol{\theta})}
```

where $`\hat{\mathbf{x}}(\boldsymbol{\theta})`$ is the mode of the latent field and $`\tilde{\pi}_G`$ the Gaussian approximation there, found by Newton's method on the sparse precision. The latent marginals mix the conditional ones over a design of points $`\boldsymbol{\theta}_k`$ around the mode,

```math
\tilde{\pi}(x_j \mid \mathbf{y}) = \sum_{k} \tilde{\pi}(x_j \mid \mathbf{y}, \boldsymbol{\theta}_k) \ \tilde{\pi}(\boldsymbol{\theta}_k \mid \mathbf{y}) \ \Delta_k
```

with weights $`\Delta_k`$ from the design, and the `strategy` picks how each conditional $`\tilde{\pi}(x_j \mid \mathbf{y}, \boldsymbol{\theta}_k)`$ is corrected from the Gaussian (a VB shift of the mean, or a skew-normal from the simplified Laplace expansion).

## 📥 Install

We have not put pyrox-lgm on PyPI, and the name `pyrox` there belongs to an unrelated project, so install it from GitHub with [uv](https://docs.astral.sh/uv/). It brings in the pyrox core and pyrox-gauss from the same repository, gaussx, kernellib, functax, argminax, optimistix, matfree, NumPyro and einx, and needs Python 3.12 or newer.

```bash
uv add "pyrox-lgm @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-lgm"
```

argminax v0.0.6 and kernellib v0.0.22 pin older gaussx releases, while pyrox-lgm needs gaussx v0.7.0, so add this override to your project's `pyproject.toml`; without it uv cannot resolve the install.

```toml
[tool.uv]
override-dependencies = [
    "gaussx @ git+https://github.com/jejjohnson/gaussx.git@v0.7.0",
]
```

One extra adds the rest: `xarray`, for `INLAResult.to_xarray()`. The quickstart needs none.

## 🧠 Quickstart: two years of weekly counts

The data are 104 weekly counts $`y_k`$ of a disease, with an expected count $`E_k = 20`$ each week (the population times a baseline rate) and a true log relative risk $`0.2 + 0.4 \sin(2\pi t_k / 52)`$, one season per year. The model is ours to write: a Poisson likelihood with the log expected count as an offset, an intercept $`\beta_0`$, and a second-order random walk $`\mathbf{x}`$ over the weeks for the seasonal shape,

```math
y_k \sim \mathrm{Poisson}(\lambda_k), \ \log \lambda_k = \log E_k + \beta_0 + x_k
```

with $`x_{k+1} - 2 x_k + x_{k-1} \sim \mathcal{N}(0, 1/\tau)`$ and $`\tau`$ under its default PC prior, $`P(\tau^{-1/2} > 1) = 0.01`$.

```python
import jax
import numpy as np
import pyrox_lgm as lgm


# The package's tests run in double precision, so the quickstart does too.
jax.config.update("jax_enable_x64", True)

# Shapes: N = T = 104 weeks

# t_k, two years of weeks, and E_k, the expected count each week
# → (T,), (T,)
weeks = np.arange(104)
expected = np.full(104, 20.0)
# the true log relative risk, 0.2 + 0.4 sin(2π t / 52)
# → (T,)
season = 0.4 * np.sin(2 * np.pi * weeks / 52)
# y_k ~ Poisson(E_k exp(0.2 + season_k))
# → (N,)
counts = np.random.default_rng(0).poisson(expected * np.exp(0.2 + season))

# today (pyrox-lgm)
# log λ_k = log E_k + β₀ + x_k, x ~ RW2(τ) over the T weeks
model = lgm.LGM(
    components=(lgm.RW2(104, name="week"),),
    fixed=lgm.FixedEffects(("intercept",)),
    likelihood=lgm.Poisson(),
)
# y, the offset log E, and the week each count indexes in the RW2
data = {"y": counts, "offset": np.log(expected), "week": weeks}
# today (pyrox-lgm)
# π(τ | y), π(β₀ | y) and π(x_k | y), with no sampling
result = lgm.inla(model, data)
```

**What `inla()` found.** `result.hyperpar`, `result.fixed` and `result.random` hold a `Summary` (mean, standard deviation and the 2.5 %, 50 % and 97.5 % quantiles) per hyperparameter, fixed effect and latent node. The truth is known here, so we score the seasonal shape against it.

```python
# Shapes: T weeks (the result and the truth as above)

# the posterior of x, one Summary per week
trend = result.random["week"]
error = np.sqrt(np.mean((np.asarray(trend.mean) - season) ** 2))
inside = (np.asarray(trend.q025) <= season) & (season <= np.asarray(trend.q975))
print(
    f"tau {float(result.hyperpar['week.tau'].mean):.2f}",
    f"intercept {float(result.fixed['intercept'].mean):.3f}",
    f"trend RMSE {error:.3f}, {100 * inside.mean():.0f} % of weeks in the 95 % band",
)
```

This prints

```
tau 3.02 intercept 0.234 trend RMSE 0.040, 100 % of weeks in the 95 % band
```

that is, the posterior mean of the random walk's precision $`\tau`$, of the intercept $`\beta_0`$ (against the true 0.2, with a posterior standard deviation of 0.02), and how well the posterior mean of $`\mathbf{x}`$ recovers the true seasonal shape. It is within 0.04 of it as a root mean square, and its 95 % band covers the truth in every one of the 104 weeks. `inla()` found this without a sample: one Laplace fit per design point over $`\tau`$, eleven of them for one hyperparameter. The [MCMC with INLA](https://jejjohnson.github.io/pyrox/lgm-mcmc-inla/) tutorial goes further.

## 🎲 The same model by NUTS

`inla()` is an approximation, and the components are also NumPyro models, so the same fit can be checked by sampling. `RW2(...).sample(index)` registers the precision as the site `week_tau` with its PC prior and the field as the site `week`, soft-constraining the random walk's null space (its level and its linear trend) because NUTS cannot sample a hard constraint. `inla()` imposes only a sum to zero on the RW2 and leaves the linear trend to the data, so the NumPyro model below adds a slope to make the two models the same; then we compare the linear predictor $`\beta_0 + x_k`$ (plus the slope) week by week.

```python
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


# Shapes: T weeks; 500 posterior draws

# today (pyrox-lgm)
# the same RW2, as NumPyro sites week_tau and week
rw2 = lgm.RW2(104, name="week")
# the weeks, centred and in years, for the slope
years = (weeks - weeks.mean()) / 52


# user
# β₀, β₁ ~ N(0, 1000), as FixedEffects' default precision 0.001
def nuts_model(counts: np.ndarray) -> None:
    trend = rw2.sample(weeks)
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 1000.0**0.5))
    slope = numpyro.sample("slope", dist.Normal(0.0, 1000.0**0.5))
    eta = numpyro.deterministic("eta", intercept + slope * years + trend)
    numpyro.sample("y", dist.Poisson(jnp.exp(np.log(expected) + eta)), obs=counts)


mcmc = MCMC(NUTS(nuts_model), num_warmup=500, num_samples=500, progress_bar=False)
mcmc.run(jr.key(0), counts)
draws = mcmc.get_samples()
# η_k by inla(): β₀ + x_k, the slope inside the RW2
inla_eta = float(result.fixed["intercept"].mean) + np.asarray(trend.mean)
gap = np.max(np.abs(np.asarray(draws["eta"].mean(axis=0)) - inla_eta))
print(f"tau {float(draws['week_tau'].mean()):.2f}, largest gap in eta {gap:.3f}")
```

This prints

```
tau 2.79, largest gap in eta 0.012
```

that is, NUTS's posterior mean of $`\tau`$, 2.79 against `inla()`'s 3.02 (both with a posterior standard deviation above 1, so the two agree), and the largest difference between the two posterior means of the linear predictor over the 104 weeks, 0.012. The approximation and the sampler give the same fit, and the component did not change between them: the same `RW2` is a GMRF block inside `inla()` and two sample sites inside NumPyro.

## 🔬 What else is in the box

`pyrox_lgm` is one namespace; its `__all__` holds every name below.

| Area | Symbols | Notes |
|---|---|---|
| Temporal components | `IID`, `RW1`, `RW2`, `AR1`, `MarkovGP` | Unstructured effects, random walks, AR(1), and a gaussx SDE kernel in precision form |
| Areal components | `Besag`, `BYM2`, `CAR`, `Leroux` | Fields on a kernellib graph: intrinsic, mixed, proper and Leroux CAR |
| Spatial and generic | `SPDE`, `Generic` | A Matérn field by the SPDE on a mesh or a grid; any structure matrix |
| Combinators | `Kronecker`, `Replicate`, `AbstractComponent` | Separable products (R-INLA's `group`), shared-hyperparameter copies, and the base every component subclasses |
| PC priors | `PCPrecision`, `PCAR1Rho`, `PCBYM2Phi`, `PCMatern`, `StructureSpectrum`, `structure_spectrum` | R-INLA's penalised-complexity priors as NumPyro distributions; the spectrum `PCBYM2Phi` needs, computed once |
| Observations | `Gaussian`, `Poisson`, `Bernoulli`, `Binomial`, `NegativeBinomial`, `AbstractObservation` | Likelihoods with their own hyperparameters, over the core's observation model |
| Model and inference | `LGM`, `FixedEffects`, `f`, `inla`, `INLAResult`, `Summary` | The model, the formula sugar, the driver and what it returns |
| Diagnostics | `diagnostics`, `Diagnostics` | DIC, WAIC, CPO, PIT and the log score, without refits |

## 📓 Tutorials

| # | Tutorial | What it shows |
|---|---|---|
| 1 | [MCMC with INLA](https://jejjohnson.github.io/pyrox/lgm-mcmc-inla/) | `inla()` against NUTS on the same model, and NUTS over a parameter outside the latent Gaussian model with `inla()` inside |

## 📖 Documentation

- **[Documentation](https://jejjohnson.github.io/pyrox/)**: the whole workspace, with the tutorials and examples of every package.
- **[LGM API](https://jejjohnson.github.io/pyrox/reference/lgm/)**: every public name, generated from the docstrings, whose examples run as doctests.
- **[Capability index](https://jejjohnson.github.io/pyrox/capabilities/)**: every public name in the workspace and in the libraries it builds on, to check before writing something new.
- **[`AGENTS.md`](AGENTS.md)**: the contracts every component, prior and observation model keeps (sparse precisions, the wiring of a new component, the R-INLA fixtures) and the tests that hold them.

## 🧩 Ecosystem

pyrox-lgm is a package of the pyrox [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/), in the GeoML tier of the [GeoStack](https://github.com/jejjohnson/geostack). It owns the latent Gaussian models and their NumPyro surface, and composes the rest in:

| Concern | Owner |
|---|---|
| The `AbstractObservation` base of the likelihoods | [`pyrox`](../pyrox) (`likelihoods`) |
| `GMRFDistribution`, the NumPyro distribution of a GMRF | [`pyrox-gauss`](../pyrox-gauss) |
| GMRF builders, sparse precision operators and their factorisations, the VB correction, the integration design | [`gaussx`](https://github.com/jejjohnson/gaussx) and [`lineax`](https://github.com/patrick-kidger/lineax) |
| Graphs and structure matrices for the areal components | [`kernellib`](https://github.com/jejjohnson/kernellib) |
| The SPDE's mesh geometry and interpolation | [`functax`](https://github.com/jejjohnson/functax) |
| The inner Laplace fit (`laplace_mode`) | [`argminax`](https://github.com/jejjohnson/argminax) |
| The L-BFGS search for the mode of $`\boldsymbol{\theta}`$ | [`optimistix`](https://github.com/patrick-kidger/optimistix) |
| Lanczos spectra for the PC priors | [`matfree`](https://github.com/pnkraemer/matfree) |
| Sample sites, NUTS and the PC priors' distribution base | [`numpyro`](https://github.com/pyro-ppl/numpyro) |
| Marginals as labelled arrays (the `xarray` extra) | [`xarray`](https://xarray.dev/) |

Dependencies point one way. pyrox-lgm imports the pyrox core and pyrox-gauss and never pyrox-gp ([`test_import_guard.py`](tests/test_import_guard.py) checks it). No other package of the workspace imports pyrox-lgm (pyrox-gauss, pyrox-ssm and pyrox-xtreme forbid it in their import guards), and nothing outside the workspace imports it either.

## 🌳 Repository layout

```
packages/pyrox-lgm/
├── src/pyrox_lgm/
│   ├── __init__.py          ← the public names
│   ├── _components/         ← temporal, areal, SPDE, generic and combinator components
│   ├── _priors/             ← the PC priors
│   ├── _likelihood.py       ← the observation models
│   ├── _model.py, _formula.py, _assembly.py ← LGM, f(...) and the sparse assembly
│   ├── _inla.py, _result.py ← inla() and INLAResult
│   ├── _diagnostics.py      ← DIC, WAIC, CPO, PIT
│   └── _numpyro.py          ← each component's NumPyro face
├── tests/                   ← R-INLA fixtures, PC-prior calibration, doctests, the import guard
└── AGENTS.md                ← the contracts and the tests that hold them
```

## ⚙️ Development

The package is developed inside the workspace, so clone the repository and run the tasks from its root.

```bash
git clone https://github.com/jejjohnson/pyrox.git
cd pyrox
make install                                                     # uv sync --all-groups + pre-commit hooks
uv run pytest --no-cov packages/pyrox-lgm/tests -m "not slow"    # this package's fast tests
uv run pytest --no-cov packages/pyrox-lgm/tests -m slow          # the R-INLA comparisons
make lint                                                        # ruff check on the whole repo
make typecheck                                                   # ty check on every package
```

`make help` lists every task. The root [`AGENTS.md`](../../AGENTS.md) holds the rules for the whole workspace, and this package's [`AGENTS.md`](AGENTS.md) adds the contracts of the components and `inla()`.

## 🎓 References

- Rue, H., Martino, S., & Chopin, N. (2009). [Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations](https://doi.org/10.1111/j.1467-9868.2008.00700.x). *JRSS B* 71(2), 319–392.
- Lindgren, F., Rue, H., & Lindström, J. (2011). [An explicit link between Gaussian fields and Gaussian Markov random fields: the stochastic partial differential equation approach](https://doi.org/10.1111/j.1467-9868.2011.00777.x). *JRSS B* 73(4), 423–498.
- Martins, T. G., Simpson, D., Lindgren, F., & Rue, H. (2013). [Bayesian computing with INLA: new features](https://doi.org/10.1016/j.csda.2013.04.014). *Computational Statistics & Data Analysis* 67, 68–83.
- Sørbye, S. H., & Rue, H. (2014). [Scaling intrinsic Gaussian Markov random field priors in spatial modelling](https://doi.org/10.1016/j.spasta.2013.06.004). *Spatial Statistics* 8, 39–51.
- Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H. (2016). [An intuitive Bayesian spatial model for disease mapping that accounts for scaling](https://doi.org/10.1177/0962280216660421). *Statistical Methods in Medical Research* 25(4), 1145–1165.
- Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sørbye, S. H. (2017). [Penalising model component complexity: a principled, practical approach to constructing priors](https://doi.org/10.1214/16-STS576). *Statistical Science* 32(1), 1–28.
- Sørbye, S. H., & Rue, H. (2017). [Penalised complexity priors for stationary autoregressive processes](https://doi.org/10.1111/jtsa.12242). *Journal of Time Series Analysis* 38(6), 923–935.
- Fuglstad, G.-A., Simpson, D., Lindgren, F., & Rue, H. (2019). [Constructing priors that penalize the complexity of Gaussian random fields](https://doi.org/10.1080/01621459.2017.1415907). *JASA* 114(525), 445–452.

## 📄 License

MIT. See [LICENSE](LICENSE).
