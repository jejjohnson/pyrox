<div align="center">

<img alt="pyrox-ssm" src="../../docs/assets/icon-ssm.svg" width="120">

# pyrox-ssm

**State-space models for JAX and NumPyro: temporal Gaussian processes as Kalman filters, linear and nonlinear state-space models, non-Gaussian and sparse Markov GPs, and many sites on one grid, on pyrox, pyrox-gauss, gaussx, lineax, optimistix and argminax.**

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
📚 [SSM API](https://jejjohnson.github.io/pyrox/reference/ssm/) ·
🚀 [One site, end to end](https://jejjohnson.github.io/pyrox/ssm-one-site/) ·
🗂️ [Capability index](https://jejjohnson.github.io/pyrox/capabilities/)

</div>

---

A station record, a tide gauge or a sensor log is a long series on a time grid with gaps in it, and a Gaussian process over it costs $`\mathcal{O}(N^3)`$ in the number of time steps. Many of the kernels used for time (Matérn, periodic, their sums and products) are the covariance of a linear stochastic differential equation, so the same prior is a Markov chain on a small state, and a Kalman filter computes its likelihood in $`\mathcal{O}(N d^3)`$. pyrox-ssm writes these models as NumPyro models over the state-space form

```math
p(\mathbf{y}, \mathbf{x}_{1:N} \mid \boldsymbol{\theta}) = p(\mathbf{x}_1) \prod_{k=2}^{N} \mathcal{N}(\mathbf{x}_k \mid \mathbf{A}_k \mathbf{x}_{k-1}, \mathbf{Q}_k) \ \prod_{k=1}^{N} p(y_k \mid \mathbf{H} \mathbf{x}_k)
```

where $`\mathbf{x}_k \in \mathbb{R}^{d}`$ is the state at time $`t_k`$, $`\mathbf{A}_k \in \mathbb{R}^{d \times d}`$ and $`\mathbf{Q}_k \in \mathbb{R}^{d \times d}`$ are the transition and process-noise covariance over the step $`\Delta t_k = t_k - t_{k-1}`$, $`\mathbf{H} \in \mathbb{R}^{1 \times d}`$ reads the latent function $`f(t_k) = \mathbf{H} \mathbf{x}_k`$ off the state, $`p(y_k \mid \mathbf{H} \mathbf{x}_k)`$ is the likelihood of one observation $`y_k`$, and $`\boldsymbol{\theta}`$ holds the kernel's hyperparameters and the noise. For a Matérn-3/2 kernel the state is the function and its derivative, $`d = 2`$. With Gaussian noise the filter integrates $`\mathbf{x}_{1:N}`$ out exactly, so NUTS and SVI see only $`\boldsymbol{\theta}`$.

```python
# Shapes: N time steps; d state size; S sites
# (times, targets, mask, variance, lengthscale and the rest as in the quickstart)

# today (pyrox-ssm)
# k(t, t'), a Matérn-3/2 kernel as a linear SDE with a state of size d = 2
sde = MaternSDE(variance, lengthscale, order=1)
# today (pyrox-ssm)
# log p(y_obs | θ) by one Kalman filter pass over the N steps, gaps masked
markov_gp_factor("y", MarkovGPPrior(sde, times), targets, noise_var, mask=mask)
# today (pyrox-ssm)
# p(f(t_*) | y) by the Kalman filter and RTS smoother, at any t_*
posterior = MarkovGPPrior(sde, times).condition(targets, noise_var, mask=mask)
# today (pyrox-ssm)
# q(f) ≈ p(f | y) for counts or labels, one Kalman pass per iteration
approx = prior.condition_nongauss(likelihood, counts, strategy=strategy, mask=mask)
# today (pyrox-ssm)
# S ragged series padded to one grid, with the mask that skips the padding
times_s, targets_s, mask_s = pad_sites(site_times, site_targets)
```

None of it is new machinery. The SDE kernels (`MaternSDE`, `PeriodicSDE`, `SumSDE`, …), the Kalman filter, the RTS smoother, the nonlinear filters and the per-site derivatives and expectations are [`gaussx`](https://github.com/jejjohnson/gaussx)'s, and pyrox-ssm re-exports the kernels so a model needs one import. The likelihoods and guides are the [`pyrox`](../pyrox) core's protocols, so a `pyrox.likelihoods.DistLikelihood` or any of [pyrox-gp](../pyrox-gp)'s likelihoods and guides work unchanged, though pyrox-ssm never imports pyrox-gp. The non-Gaussian strategies iterate to a fixed point with [`optimistix`](https://github.com/patrick-kidger/optimistix) and differentiate it with [`argminax`](https://github.com/jejjohnson/argminax)'s sampler-safe implicit adjoint, the sparse prior's multivariate normal is [`pyrox-gauss`](../pyrox-gauss)'s, the operators are [`lineax`](https://github.com/patrick-kidger/lineax)'s, and the models are plain [`numpyro`](https://github.com/pyro-ppl/numpyro) functions. Two extras reach further: `flows` adds the normalizing Kalman filter over [gauss_flows](https://github.com/jejjohnson/gauss_flows) warps, and `ensemble` adds the ensemble Kalman log likelihood of a [filterax](https://github.com/jejjohnson/filterax) run.

pyrox-ssm is one package of the pyrox workspace; the [workspace README](../../README.md) introduces the others.

<p align="center"><img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_ssm_overview/pyrox_ssm_overview.png" alt="pyrox-ssm in one picture: you supply a series with gaps, an SDE kernel with priors, a likelihood, or the state-space matrices; the model writes one state-space joint density, whose SDE step works on any grid, whose likelihood is one Kalman pass and whose masked steps run the predict only; pyrox-ssm offers Markov GPs, linear-Gaussian and nonlinear state-space models, non-Gaussian likelihoods, and sparse and many-site models; out come the hyperparameters, the posterior through the gaps, the smoothed states and a Gaussian approximation for counts; pyrox and pyrox-gauss, gaussx and lineax, optimistix, and flowjax and filterax underneath" width="100%"></p>

## 🌟 Highlights

- ⏱️ **Temporal GPs in linear time**: `MarkovGPPrior` turns any gaussx SDE kernel (Matérn, periodic, quasi-periodic, cosine, constant, integrated Wiener, and their sums and products) into a Kalman filter for the marginal likelihood and an RTS smoother for the posterior, both $`\mathcal{O}(N d^3)`$.
- 🕳️ **Gaps as masks**: every entry point takes a `mask`, a masked step runs the predict only, and a `NaN` placeholder there reaches neither the value nor a gradient, so a record with an outage needs no reindexing; `predict` interpolates, forecasts and backcasts on one code path.
- 🧮 **Any linear-Gaussian model**: `LinearGaussianSSMPrior` takes the transition, observation and noise matrices directly, and `nonlinear_ssm_factor` scores a nonlinear model by an extended, unscented, cubature, Gauss-Hermite or Monte Carlo Kalman filter.
- 🎲 **Non-Gaussian likelihoods**: Laplace, Gauss-Newton, posterior linearisation and expectation propagation behind `condition_nongauss`, each iteration one Kalman pass, and `markov_laplace_factor` puts the approximate evidence in a NumPyro model for NUTS over the hyperparameters.
- 🪶 **Sparse Markov GPs**: `SparseMarkovGPPrior` with inducing times and `sparse_markov_elbo`, taking any guide with the pyrox `Guide` protocol.
- 🗺️ **Many sites**: `pad_sites` pads ragged per-site series to one shape with the mask that skips the padding, `markov_gp_sites_factor` scores them as one factor, and `SharedPlusSiteMarkovGPPrior` fits a shared GP plus one GP per site as one state-space model.
- 🌊 **Beyond Gaussian states**: `pyrox_ssm.flows` warps the observations with a normalizing flow over a Gaussian state (the `flows` extra), and `pyrox_ssm.ensemble.ensemble_factor` adds the ensemble Kalman log likelihood of a filterax run (the `ensemble` extra).

## 📐 One recursion, many models

Every model in the package is a Kalman recursion over some state, and they differ in where the state and the likelihood come from.

| Model | Entry points | The state and the likelihood | Reach for it when |
|---|---|---|---|
| Markov GP | `MarkovGPPrior`, `markov_gp_factor`, `.condition()` | an SDE kernel's state; Gaussian noise | a GP over a long series |
| Linear-Gaussian SSM | `LinearGaussianSSMPrior`, `lgssm_factor`, `.condition()` | your matrices $`\mathbf{A}_k`$, $`\mathbf{H}_k`$, $`\mathbf{Q}_k`$, $`\mathbf{R}_k`$ | known linear dynamics, or several outputs |
| Nonlinear SSM | `nonlinear_ssm_factor` | dynamics $`f`$ and observation $`h`$, moment-matched | nonlinear dynamics, at the cost of a surrogate likelihood |
| Non-Gaussian | `.condition_nongauss()`, `markov_laplace_factor` | an SDE kernel's state; a Gaussian site per observation | counts, labels or heavy tails |
| Sparse | `SparseMarkovGPPrior`, `sparse_markov_elbo` | inducing times; any likelihood, through a guide | a variational fit with fewer inducing times than steps |
| Many sites | `pad_sites`, `markov_gp_sites_factor`, `SharedPlusSiteMarkovGPPrior` | one state per site, or a shared state plus one per site | a network of stations |

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_ssm_hub/pyrox_ssm_hub-dark.svg">
    <img alt="One MarkovGPPrior fans out to five uses: markov_gp_factor as a NumPyro factor for NUTS and SVI, log_marginal by one Kalman pass under jit and grad, condition and predict for smoothed marginals at any times, condition_nongauss with Laplace, Gauss-Newton, posterior linearisation and EP, and filter, smooth and as_lgssm for the Kalman moments and the general state-space form" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_ssm_hub/pyrox_ssm_hub-light.svg" width="100%">
  </picture>
</p>

The figure draws the Markov GP, the one model with a prior, a factor, `condition` and `predict` today; [#480](https://github.com/jejjohnson/pyrox/issues/480) tracks giving every other model the same parts.

For an SDE kernel $`\mathrm{d}\mathbf{x} = \mathbf{F} \mathbf{x} \ \mathrm{d}t + \mathbf{L} \ \mathrm{d}\boldsymbol{\beta}`$ with stationary covariance $`\mathbf{P}_\infty`$, the step from $`t_{k-1}`$ to $`t_k`$ is

```math
\mathbf{A}_k = \exp(\mathbf{F} \Delta t_k), \ \mathbf{Q}_k = \mathbf{P}_\infty - \mathbf{A}_k \mathbf{P}_\infty \mathbf{A}_k^\top
```

so the grid need not be regular. With Gaussian noise of variance $`\sigma^2`$ the filter returns the marginal likelihood as a sum of one-step prediction errors,

```math
\log p(\mathbf{y} \mid \boldsymbol{\theta}) = \sum_{k=1}^{N} \log \mathcal{N}\left(y_k \mid \mathbf{H} \mathbf{m}_k^-, \ \mathbf{H} \mathbf{P}_k^- \mathbf{H}^\top + \sigma^2\right)
```

with $`\mathbf{m}_k^-`$ and $`\mathbf{P}_k^-`$ the predicted mean and covariance of $`\mathbf{x}_k`$ given $`y_1, \ldots, y_{k-1}`$; a masked step adds nothing to the sum. The non-Gaussian strategies replace each $`p(y_k \mid f_k)`$ by a Gaussian site and run this filter and the smoother once per iteration, and differ only in how they choose the site (the exact curvature at the mode, its Gauss-Newton part, a statistical linearisation or moment matching).

## 📥 Install

We have not put pyrox-ssm on PyPI, and the name `pyrox` there belongs to an unrelated project, so install it from GitHub with [uv](https://docs.astral.sh/uv/). It brings in the pyrox core and pyrox-gauss from the same repository, gaussx, lineax, optimistix, argminax, NumPyro and einx, and needs Python 3.12 or newer.

```bash
uv add "pyrox-ssm @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-ssm"
```

argminax pins gaussx v0.6.13 (and filterax v0.0.8, behind the `ensemble` extra, pins gaussx v0.6.9), while pyrox-ssm needs gaussx v0.7.0, so add this override to your project's `pyproject.toml`; without it uv cannot resolve the install.

```toml
[tool.uv]
override-dependencies = [
    "gaussx @ git+https://github.com/jejjohnson/gaussx.git@v0.7.0",
]
```

Two extras add the rest: `flows` for `pyrox_ssm.flows` (flowjax and gauss-flows) and `ensemble` for `pyrox_ssm.ensemble` (filterax). The quickstart needs neither.

## 🧠 Quickstart: a year of daily data with a month missing

The data are one year of a daily anomaly $`f(t)`$, drawn from a Matérn-3/2 GP with variance $`\sigma_f^2 = 4`$ and lengthscale $`\ell = 5`$ days, observed with noise $`\varepsilon_k \sim \mathcal{N}(0, 0.5^2)`$. A 30-day outage (days 200 to 229) and 10 % of the other days are missing. The model is ours to write: log-normal priors on $`\boldsymbol{\theta} = (\sigma_f^2, \ell, \sigma^2)`$, a Matérn-3/2 SDE, and `markov_gp_factor` for the marginal likelihood. NUTS samples $`\boldsymbol{\theta}`$, and the Kalman smoother then fills the gaps.

```python
import jax
import jax.numpy as jnp
import jax.random as jr
from pyrox_ssm import MarkovGPPrior, MaternSDE, markov_gp_from_innovations


# The package's tests run in double precision, so the quickstart does too.
jax.config.update("jax_enable_x64", True)

# Shapes: N = 365 days; d = 2 (the Matérn-3/2 state)

# t_k, one per day
# → (N,)
times = jnp.arange(365.0)
# today (pyrox-ssm)
# the true anomaly's prior, Matérn-3/2 with σ_f² = 4 and ℓ = 5 days
truth = MarkovGPPrior(MaternSDE(4.0, 5.0, order=1), times)
# ε_k ~ 𝒩(0, I), standard normal innovations
# → (N, d)
innovations = jr.normal(jr.key(0), (365, truth.state_dim))
# today (pyrox-ssm)
# f(t_k), the SDE run forward from the innovations
# (N, d) → (N,)
anomaly = markov_gp_from_innovations(truth, innovations)
# y_k = f(t_k) + ε_k, with ε_k ~ 𝒩(0, 0.5²)
# → (N,)
observed = anomaly + 0.5 * jr.normal(jr.key(1), (365,))
# o_k = 1 on an observed day: not in the outage, and not one of the random gaps
# → (N,)
outage = (times >= 200) & (times < 230)
mask = ~(outage | (jr.uniform(jr.key(2), (365,)) < 0.1))
# what the record holds: NaN on the missing days
# → (N,)
targets = jnp.where(mask, observed, jnp.nan)
```

**A. The model.** Each hyperparameter is an ordinary NumPyro sample site, and `markov_gp_factor` adds $`\log p(\mathbf{y} \mid \boldsymbol{\theta})`$, one filter pass over the 365 days with the missing ones masked. The trace holds the three sites and the factor and no latent path, so NUTS samples three numbers.

```python
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Bool, Float
from numpyro.infer import MCMC, NUTS
from pyrox_ssm import markov_gp_factor


# Shapes: N days; 300 posterior draws


# user
# σ_f² ~ LogNormal(0, 1), ℓ ~ LogNormal(log 7, 1), σ² ~ LogNormal(log 0.1, 1)
def model(
    times: Float[Array, " N"], targets: Float[Array, " N"], mask: Bool[Array, " N"]
) -> None:
    variance = numpyro.sample("variance", dist.LogNormal(0.0, 1.0))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(jnp.log(7.0), 1.0))
    noise_var = numpyro.sample("noise_var", dist.LogNormal(jnp.log(0.1), 1.0))
    # today (pyrox-ssm)
    # f ~ GP(0, k_Matérn-3/2) on the N days, as a linear SDE
    prior = MarkovGPPrior(MaternSDE(variance, lengthscale, order=1), times)
    # today (pyrox-ssm)
    # log p(y_obs | θ) = ∑ₖ oₖ log 𝒩(yₖ; H m⁻ₖ, H P⁻ₖ Hᵀ + σ²)
    markov_gp_factor("y", prior, targets, noise_var, mask=mask)


# θ ~ p(θ | y), 300 draws after 300 warm-up steps
mcmc = MCMC(NUTS(model), num_warmup=300, num_samples=300, progress_bar=False)
mcmc.run(jr.key(3), times, targets, mask)
# θ̄, the posterior mean of each site
theta = {name: draws.mean() for name, draws in mcmc.get_samples().items()}
```

**B. Through the gaps.** At $`\bar{\boldsymbol{\theta}}`$, `condition` runs the filter and the RTS smoother, and `predict` gives the posterior mean and variance of $`f`$ at any times: here every day of the year, then day 395, a month after the record ends.

```python
# Shapes: N days

# today (pyrox-ssm)
# k_Matérn-3/2 at θ̄
sde = MaternSDE(theta["variance"], theta["lengthscale"], order=1)
# today (pyrox-ssm)
# p(x_1:N | y; θ̄) by the Kalman filter and RTS smoother, gaps masked
posterior = MarkovGPPrior(sde, times).condition(targets, theta["noise_var"], mask=mask)
# today (pyrox-ssm)
# 𝔼[f(t_k) | y] and Var[f(t_k) | y] on every day
# (N,) → (N,), (N,)
mean, var = posterior.predict(times)
# today (pyrox-ssm)
# the same a month after the record ends
# (1,) → (1,), (1,)
forecast_mean, forecast_var = posterior.predict(jnp.array([395.0]))
```

**Compare.** The truth is known here, so we score the posterior against it on the observed days and on the outage.

```python
# Shapes: N days (the posterior and the truth as above)

print({name: round(float(value), 2) for name, value in theta.items()})
# z_k = (f(t_k) − 𝔼[f(t_k) | y]) / √Var[f(t_k) | y]
z = (anomaly - mean) / jnp.sqrt(var)
for name, days in {"observed": mask, "outage": outage}.items():
    rmse = jnp.sqrt(jnp.mean((anomaly - mean)[days] ** 2))
    within = 100 * jnp.mean(jnp.abs(z[days]) < 2)
    print(f"{name:>8}: RMSE {rmse:.2f}, {within:.0f} % within 2 sd")
print(f"day 395: {forecast_mean[0]:.2f} ± {jnp.sqrt(forecast_var[0]):.2f}")
```

This prints

```
{'lengthscale': 5.2, 'noise_var': 0.31, 'variance': 3.31}
observed: RMSE 0.35, 96 % within 2 sd
  outage: RMSE 1.61, 100 % within 2 sd
day 395: -0.00 ± 1.82
```

The first line is $`\bar{\boldsymbol{\theta}}`$: $`\ell`$ at 5.20 days against the true 5, $`\sigma^2`$ at 0.31 against 0.25, and $`\sigma_f^2`$ at 3.31 against 4. On the observed days the posterior mean is within 0.35 of the true anomaly (as a root mean square), less than the noise's 0.5, and 96 % of them fall within two posterior standard deviations. Inside the outage the error grows to 1.61, and the band grows with it: every outage day lies inside. A month after the record ends the smoother is back at the prior, a mean of zero and a standard deviation of $`1.82 = \sqrt{3.31}`$. The [one site, end to end](https://jejjohnson.github.io/pyrox/ssm-one-site/) tutorial adds a seasonal mean and SVI, and measures the cost of one log density and its gradient per day staying flat as the record grows sixteen-fold.

## 🎲 Counts on the same days: swap the strategy, keep the prior

With counts the path no longer integrates out. The prior stays a `MarkovGPPrior`, the likelihood is any NumPyro distribution wrapped in the core's `DistLikelihood`, here $`y_k \sim \mathrm{Poisson}(e^{f_k})`$, and `condition_nongauss` takes the strategy that builds the Gaussian approximation $`q(\mathbf{f})`$. The log-rate is $`1 + f(t)/2`$ with the anomaly above, so the prior below has mean 1, $`\sigma_f^2 = 1`$ and $`\ell = 5`$ days, fixed at the truth; the same mask hides the same days.

```python
from pyrox.likelihoods import DistLikelihood
from pyrox_ssm import ExpectationPropagationMarkov, LaplaceMarkovInference


# Shapes: N days

# λ_k = exp(1 + f(t_k) / 2), the true log-rate on each day
# → (N,)
log_rate = 1.0 + 0.5 * anomaly
counts = jr.poisson(jr.key(5), jnp.exp(log_rate)).astype(times.dtype)
# today (pyrox)
# p(y_k | f_k) = Poisson(y_k; exp(f_k))
likelihood = DistLikelihood(lambda f: dist.Poisson(jnp.exp(f)))
# today (pyrox-ssm)
# f ~ GP(1, k_Matérn-3/2) with σ_f² = 1 and ℓ = 5 days
count_prior = MarkovGPPrior(
    MaternSDE(1.0, 5.0, order=1), times, mean_fn=lambda t: jnp.ones_like(t)
)

for strategy in (LaplaceMarkovInference(), ExpectationPropagationMarkov()):
    # today (pyrox-ssm)
    # q(f) = 𝒩(m, V) ∝ p(f) ∏ₖ tₖ(fₖ), a Gaussian site tₖ per observed day
    approx = count_prior.condition_nongauss(
        likelihood, counts, strategy=strategy, mask=mask
    )
    # today (pyrox-ssm)
    # 𝔼_q[f(t_k)] on every day
    # (N,) → (N,), (N,)
    rate_mean, _ = approx.predict(times)
    errors = {
        name: jnp.sqrt(jnp.mean((log_rate - rate_mean)[days] ** 2))
        for name, days in {"observed": mask, "outage": outage}.items()
    }
    print(
        f"{type(strategy).__name__:>28}: {approx.n_iter} iterations,",
        ", ".join(f"RMSE {name} {err:.2f}" for name, err in errors.items()),
    )
```

This prints

```
      LaplaceMarkovInference: 10 iterations, RMSE observed 0.30, RMSE outage 0.87
ExpectationPropagationMarkov: 24 iterations, RMSE observed 0.31, RMSE outage 0.86
```

Laplace converges in 10 iterations and expectation propagation in 24, and their errors agree to 0.01: on the observed days the posterior mean of the log-rate is within about 0.3 of the truth, and through the outage, where only the prior speaks, within about 0.87. Gauss-Newton and posterior linearisation take the same call, and `markov_laplace_factor` puts the Laplace evidence in a NumPyro model so that NUTS samples the hyperparameters too, as the [non-Gaussian likelihood](https://jejjohnson.github.io/pyrox/ssm-nongauss/) tutorial does.

## 🔬 What else is in the box

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_ssm_roles/pyrox_ssm_roles-dark.svg">
    <img alt="pyrox-ssm on the GeoStack role ladder: conditioning, the evidence factors, the Kalman recursions and the path samplers as algorithms; the priors and their conditioned results as models; the non-Gaussian strategies and the implicit adjoint as strategies; prediction, the linear-Gaussian form, the site padding and a non-centred site helper as primitives; no structure of its own; gaussx's SDE kernels beside the ladder" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_ssm_roles/pyrox_ssm_roles-light.svg" width="100%">
  </picture>
</p>

Each public name sits on one rung of the GeoStack role ladder, by the job it does; the table below lists them by area.

`pyrox_ssm` is one namespace, and two modules behind extras hold the rest; each has its own `__all__`.

| Area | Symbols | Notes |
|---|---|---|
| SDE kernels (from gaussx) | `MaternSDE`, `PeriodicSDE`, `QuasiPeriodicSDE`, `CosineSDE`, `ConstantSDE`, `IntegratedWienerSDE`, `SumSDE`, `ProductSDE`, `SDEKernel`, `SDEParams` | Kernels as linear SDEs, re-exported unchanged |
| Markov GP | `MarkovGPPrior`, `ConditionedMarkovGP`, `markov_gp_factor`, `markov_gp_sample`, `markov_gp_from_innovations`, `markov_gp_innovations_sample` | The filter's marginal likelihood, the smoothed posterior, and latent paths in linear time |
| Linear-Gaussian SSM | `LinearGaussianSSMPrior`, `ConditionedLinearGaussianSSM`, `lgssm_factor` | A state-space model given by its matrices, with per-step or per-entry masks |
| Nonlinear SSM | `nonlinear_ssm_factor` | The moment-matched log likelihood of gaussx's nonlinear Kalman filter |
| Non-Gaussian likelihoods | `LaplaceMarkovInference`, `GaussNewtonMarkovInference`, `PosteriorLinearizationMarkov`, `ExpectationPropagationMarkov`, `ConditionedNonGaussMarkovGP`, `markov_laplace_factor`, `markov_implicit_adjoint` | Strategies for `condition_nongauss`, the approximate evidence as a factor, and the adjoint that makes its gradient safe inside NUTS |
| Sparse Markov GP | `SparseMarkovGPPrior`, `ConditionedSparseMarkovGP`, `sparse_markov_elbo`, `sparse_markov_factor` | Inducing times and a variational guide |
| Many sites | `pad_sites`, `markov_gp_sites_factor`, `noncentred_log_normal`, `SharedPlusSiteMarkovGPPrior`, `ConditionedSharedPlusSiteMarkovGP`, `shared_plus_site_factor` | Ragged series on one grid, pooled hyperparameters, and a shared GP plus one per site |
| `pyrox_ssm.flows` | `NormalizingKalmanPrior`, `normalizing_kalman_factor` | The normalizing Kalman filter (the `flows` extra) |
| `pyrox_ssm.ensemble` | `ensemble_factor` | The ensemble Kalman log likelihood of a filterax run (the `ensemble` extra) |

The Markov GPs moved here from pyrox-gp with every name and signature unchanged; `from pyrox_gp import MarkovGPPrior` now raises an `ImportError` that names `pyrox_ssm`.

## 📓 Tutorials

| # | Tutorial | What it shows |
|---|---|---|
| 1 | [One site, end to end](https://jejjohnson.github.io/pyrox/ssm-one-site/) | a station record with gaps, NUTS and SVI over the hyperparameters, and prediction through an outage |
| 2 | [Many sites, pooled hyperparameters](https://jejjohnson.github.io/pyrox/ssm-hierarchical-sites/) | ragged records padded with `pad_sites` and scored as one factor |
| 3 | [A regional signal plus a local signal per site](https://jejjohnson.github.io/pyrox/ssm-shared-plus-site/) | `SharedPlusSiteMarkovGPPrior` |
| 4 | [A spatial field, per-facility paths and Poisson counts](https://jejjohnson.github.io/pyrox/ssm-spatial-hierarchical/) | a hierarchical count model with NUTS and SVI |
| 5 | [A non-Gaussian likelihood, collapsed and latent](https://jejjohnson.github.io/pyrox/ssm-nongauss/) | `markov_laplace_factor` against a latent path, with a cross-check |
| 6 | [An ensemble Kalman likelihood](https://jejjohnson.github.io/pyrox/ssm-ensemble/) | `ensemble_factor` and an ensemble of variational fits |
| 7 | [A nonlinear state-space model](https://jejjohnson.github.io/pyrox/ssm-nonlinear/) | `nonlinear_ssm_factor` |
| 8 | [Matérn kernels in state-space form](https://jejjohnson.github.io/pyrox/markov-gp-sde-kernels/) | the SDE kernels and their compositions |
| 9 | [Kalman filtering and RTS smoothing](https://jejjohnson.github.io/pyrox/markov-gp-kalman/) | the Markov GP step by step, against a dense GP |
| 10 | [Non-Gaussian Markov GP inference](https://jejjohnson.github.io/pyrox/markov-gp-nongauss/) | Laplace, Gauss-Newton, posterior linearisation and EP side by side |
| 11 | [Training a Markov GP](https://jejjohnson.github.io/pyrox/markov-gp-training/) | maximum likelihood, NUTS, variational inference and identifiability |
| 12 | [Sparse variational Markov GP](https://jejjohnson.github.io/pyrox/sparse-markov-gp/) | inducing times, non-Gaussian likelihoods and NumPyro |

## 📖 Documentation

- **[Documentation](https://jejjohnson.github.io/pyrox/)**: the whole workspace, with the tutorials and examples of every package.
- **[SSM API](https://jejjohnson.github.io/pyrox/reference/ssm/)**: every public name, generated from the docstrings, with the kernel compositions and the masking rules.
- **[Capability index](https://jejjohnson.github.io/pyrox/capabilities/)**: every public name in the workspace and in the libraries it builds on, to check before writing something new.
- **[`AGENTS.md`](AGENTS.md)**: the contracts every model keeps (no import of pyrox-gp, pyrox-lgm or pyrox-nn, gaussx for the linear algebra, the extras) and the tests that hold them.

## 🧩 Ecosystem

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_ssm_ecosystem/pyrox_ssm_ecosystem-dark.svg">
    <img alt="No package imports pyrox-ssm; pyrox-gauss and pyrox-xtreme forbid importing it in their import-guard tests; pyrox-ssm builds on pyrox, pyrox-gauss, gaussx, lineax, optimistix, JAX, Equinox, NumPyro and einx, and on flowjax, gauss-flows and filterax through extras" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_ssm_ecosystem/pyrox_ssm_ecosystem-light.svg" width="100%">
  </picture>
</p>

pyrox-ssm is a package of the pyrox [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/), in the GeoML tier of the [GeoStack](https://github.com/jejjohnson/geostack). It owns the state-space models and their NumPyro surface, and composes the rest in:

| Concern | Owner |
|---|---|
| The `Likelihood`, `AbstractObservation` and `Guide` protocols, and `DistLikelihood` | [`pyrox`](../pyrox) (`likelihoods`) |
| Multivariate normals over operators | [`pyrox-gauss`](../pyrox-gauss) |
| SDE kernels, Kalman filters (sequential, parallel and nonlinear), the RTS smoother, per-site derivatives and expectations | [`gaussx`](https://github.com/jejjohnson/gaussx) and [`lineax`](https://github.com/patrick-kidger/lineax) |
| The fixed-point loop of the non-Gaussian strategies | [`optimistix`](https://github.com/patrick-kidger/optimistix) |
| The implicit adjoint that keeps a NUTS chain alive when its linear solve fails (`safe_implicit_adjoint`) | [`argminax`](https://github.com/jejjohnson/argminax) |
| Sample and factor sites, NUTS, SVI and `Predictive` | [`numpyro`](https://github.com/pyro-ppl/numpyro) |
| Modules as pytrees | [`equinox`](https://github.com/patrick-kidger/equinox) |
| Observation warps for the normalizing Kalman filter (the `flows` extra) | [`flowjax`](https://github.com/danielward27/flowjax) and [`gauss_flows`](https://github.com/jejjohnson/gauss_flows) |
| The ensemble Kalman filter and its differentiable assimilation (the `ensemble` extra) | [`filterax`](https://github.com/jejjohnson/filterax) |

Dependencies point one way. pyrox-ssm imports the pyrox core and pyrox-gauss and never pyrox-gp, pyrox-lgm or pyrox-nn ([`test_ssm_import_guard.py`](tests/test_ssm_import_guard.py) checks it). No other package of the workspace imports pyrox-ssm (pyrox-gauss and pyrox-xtreme forbid it in their import guards, and pyrox-gp only names it in the `ImportError` for the moved Markov-GP names), and nothing outside the workspace imports it either.

## 🌳 Repository layout

```
packages/pyrox-ssm/
├── src/pyrox_ssm/
│   ├── __init__.py          ← the public names and the gaussx SDE kernels
│   ├── _markov.py           ← MarkovGPPrior, ConditionedMarkovGP, the factors and samplers
│   ├── _lgssm.py            ← LinearGaussianSSMPrior, lgssm_factor
│   ├── _nonlinear.py        ← nonlinear_ssm_factor
│   ├── _inference_nongauss_markov.py ← Laplace, Gauss-Newton, PL and EP on a Markov GP
│   ├── _sparse_markov.py    ← the sparse Markov GP
│   ├── _sites.py, _hierarchical.py ← many sites
│   ├── flows/               ← the normalizing Kalman filter (flows extra)
│   └── ensemble.py          ← the ensemble Kalman log likelihood (ensemble extra)
├── tests/                   ← dense references, the masking rules, the import guard
└── AGENTS.md                ← the contracts and the tests that hold them
```

## ⚙️ Development

The package is developed inside the workspace, so clone the repository and run the tasks from its root.

```bash
git clone https://github.com/jejjohnson/pyrox.git
cd pyrox
make install                                                     # uv sync --all-groups + pre-commit hooks
uv run pytest --no-cov packages/pyrox-ssm/tests -m "not slow"    # this package's fast tests
make lint                                                        # ruff check on the whole repo
make typecheck                                                   # ty check on every package
```

`make help` lists every task. The root [`AGENTS.md`](../../AGENTS.md) holds the rules for the whole workspace, and this package's [`AGENTS.md`](AGENTS.md) adds the contracts of the state-space models.

## 🎓 References

- Hartikainen, J., & Särkkä, S. (2010). [Kalman filtering and smoothing solutions to temporal Gaussian process regression models](https://doi.org/10.1109/MLSP.2010.5589113). *IEEE MLSP*.
- Särkkä, S., & Solin, A. (2019). [Applied stochastic differential equations](https://doi.org/10.1017/9781108186735). *Cambridge University Press*.
- Minka, T. P. (2001). [Expectation propagation for approximate Bayesian inference](https://arxiv.org/abs/1301.2294). *UAI*.
- García-Fernández, Á. F., Svensson, L., Morelande, M. R., & Särkkä, S. (2015). [Posterior linearization filter: principles and implementation using sigma points](https://doi.org/10.1109/TSP.2015.2454485). *IEEE Transactions on Signal Processing* 63(20), 5561–5573.
- Nickisch, H., Solin, A., & Grigorievskiy, A. (2018). [State space Gaussian processes with non-Gaussian likelihood](https://arxiv.org/abs/1802.04846). *ICML*.
- Wilkinson, W. J., Chang, P. E., Andersen, M. R., & Solin, A. (2020). [State space expectation propagation: efficient inference schemes for temporal Gaussian processes](https://arxiv.org/abs/2007.05994). *ICML*.
- Wilkinson, W. J., Särkkä, S., & Solin, A. (2023). [Bayes-Newton methods for approximate Bayesian inference with PSD guarantees](https://arxiv.org/abs/2111.01721). *JMLR* 24(83).
- Adam, V., Eleftheriadis, S., Artemev, A., Durrande, N., & Hensman, J. (2020). [Doubly sparse variational Gaussian processes](https://arxiv.org/abs/2001.05363). *AISTATS*.
- Wilkinson, W. J., Solin, A., & Adam, V. (2021). [Sparse algorithms for Markovian Gaussian processes](https://arxiv.org/abs/2103.10710). *AISTATS*.
- de Bézenac, E., et al. (2020). [Normalizing Kalman filters for multivariate time series analysis](https://proceedings.neurips.cc/paper/2020/hash/1f47cef5e38c952f94c5d61726027439-Abstract.html). *NeurIPS*.

## 📄 License

MIT. See [LICENSE](LICENSE).
