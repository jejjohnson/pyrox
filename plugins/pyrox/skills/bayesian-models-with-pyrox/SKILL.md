---
name: bayesian-models-with-pyrox
description: Build Bayesian / probabilistic models in JAX with Equinox and NumPyro on pyrox — Equinox modules that own NumPyro sample and param sites, kernels with priors and per-parameter guides, exact / sparse / Markov / multi-output / non-Gaussian GPs, Bayesian neural-network layers, ensemble MAP / VI, and latent Gaussian models fitted with INLA. Use whenever a task puts priors on the parameters of an Equinox module, writes a GP or Bayesian NN in NumPyro, or fits a hierarchical / spatial / temporal latent Gaussian model, in a project that uses (or could use) pyrox.
---

# Bayesian models on pyrox

pyrox makes an Equinox module a first-class NumPyro citizen: the module owns
its sites, names them stably, and runs unchanged under `handlers.trace`,
NUTS, SVI and `Predictive`. On top sit GP building blocks (pyrox-gp),
Bayesian layers (pyrox-nn) and latent Gaussian models with INLA (pyrox-lgm).
Before hand-wiring `numpyro.sample` into a module, writing a GP likelihood or
a Bayesian layer, look it up:

1. **The capability index** lists every public name with a one-line summary,
   plus the gaussx, kernellib and geonnax APIs underneath:
   <https://jejjohnson.github.io/pyrox/capabilities/>. Or list the installed
   version:

   ```python
   import importlib, inspect

   for name in ("pyrox._core", "pyrox.inference", "pyrox_gp", "pyrox_nn", "pyrox_lgm"):
       try:
           module = importlib.import_module(name)
       except ImportError:
           continue  # that package is not installed
       for attr in getattr(module, "__all__", []):
           doc = (inspect.getdoc(getattr(module, attr)) or "").split("\n")[0]
           print(f"{name}.{attr}: {doc}")
   ```

2. Compose what exists. pyrox is not on PyPI (the name belongs to another
   project): install each package from its subdirectory,
   `uv add "pyrox-gp @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-gp"`.

## Which package does what

| You need… | Package | Use |
|---|---|---|
| An Equinox module with random / learned parameters | `pyrox` | `pyrox._core.PyroxModule` + `pyrox_method` (pattern B); `Parameterized` for constrained params with priors and guides (pattern C) |
| Many MAP or VI fits in parallel (deep ensembles) | `pyrox` | `pyrox.inference.ensemble_map`, `ensemble_vi`, `EnsembleMAP`, `EnsembleVI` |
| GP kernels with hyperpriors | `pyrox-gp` | `pyrox_gp.RBF`, `Matern`, `Periodic`, … + `set_prior` / `autoguide` / `set_mode` |
| An exact GP marginal likelihood or posterior | `pyrox-gp` | `GPPrior` + `gp_factor` / `gp_sample`; `GPPrior.condition(y, noise).predict(X*)` |
| Sparse / variational GPs | `pyrox-gp` | `SparseGPPrior`, the guides (`WhitenedGuide`, `NaturalGuide`, …), `svgp_factor`, `svgp_elbo` |
| Long time series (linear in N) | `pyrox-gp` | `MarkovGPPrior` + `markov_gp_factor`, SDE kernels (`MaternSDE`, …) |
| Non-Gaussian likelihoods | `pyrox-gp` | the likelihoods + `LaplaceInference`, `ExpectationPropagation`, `GaussNewtonInference` |
| Multi-output GPs, pathwise samples | `pyrox-gp` | `MultiOutputGPPrior`, `LMCKernel`, `OILMMGPPrior`; `PathwiseSampler` |
| Bayesian NN layers | `pyrox-nn` | `DenseReparameterization`, `DenseFlipout`, `BayesianSIREN`, random-feature layers, `RandomFeatureGaussianProcess` (SNGP), `DeepVSSGP` |
| A spatio-temporal regressor from a DataFrame | `pyrox-nn[bnf]` | `pyrox_nn.api.BNFEstimator` |
| Latent Gaussian models (GLMMs, disease mapping, SPDE) with INLA | `pyrox-lgm` | `pyrox_lgm.LGM`, components (`IID`, `RW1`, `AR1`, `BYM2`, `SPDE`, …), PC priors, `inla()`, `f()` |

## The three modelling patterns

- **A. Plain sites + `eqx.tree_at`** — when one field of an existing module
  becomes random, sample it in the model and splice it in. No pyrox needed.
- **B. A `PyroxModule` owns its sites** — the module is itself probabilistic
  (a Bayesian layer, a hierarchical component).
- **C. `Parameterized`** — constrained hyperparameters with priors and
  per-parameter guides, declared once in `setup()` (the pyrox-gp kernels).

All three emit plain NumPyro sites, so every handler, MCMC, SVI and
`Predictive` work on them.

## The rules your code must keep

- **Sites through the bridge.** Inside a `PyroxModule`, call
  `self.pyrox_sample(name, prior)` / `self.pyrox_param(name, init)` in a
  method decorated with `@pyrox_method` — never `numpyro.sample` directly.
  The decorator caches sites per call, so reading a site twice is one site.
- **Unique scopes.** Sites are named `"<pyrox_name>.<name>"` (class name by
  default). Two instances of one class in one model need distinct
  `pyrox_name`s (`pgp.RBF(pyrox_name="RBF_q0")`); otherwise the trace
  rejects them (MCMC and SVI trace the model), and under a bare `seed` the
  collision goes unnoticed.
- **Don't rebuild a `Parameterized` module.** Its params, priors and guides
  live outside the pytree: `eqx.tree_at`, `eqx.apply_updates`, passing it
  through `jax.jit` as an argument or loading a checkpoint gives a copy with
  an empty registry (`KeyError`). Keep the original instance and close over
  it (`jax.jit(handlers.seed(model, 0))`).
- **Model / guide mode.** A `Parameterized` module with priors samples its
  prior after `set_mode("model")` and its own guide after
  `set_mode("guide")`; with an autoguide over the whole model
  (`AutoNormal(model)`), just leave it in model mode.
- **Linear algebra through gaussx.** Don't `cho_solve` a Gram yourself:
  `GPPrior` / `gp_factor` do it with a pluggable `solver=`.
- **JAX rules.** Explicit keys; static configuration as
  `eqx.field(static=True)`; `eqx.Module`, not dataclasses.

## Worked example

```python
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

import pyrox_gp as pgp
from pyrox._core import PyroxModule, pyrox_method


# 1. A module that owns its sites (pattern B).
class BayesianLinear(PyroxModule):
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    pyrox_name: str | None = eqx.field(static=True, default=None)

    @pyrox_method
    def __call__(self, x):  # (N, D) → (N, P)
        # W ~ 𝒩(0, I), b ~ 𝒩(0, I): sites "<pyrox_name>.weight", "<pyrox_name>.bias"
        W = self.pyrox_sample(
            "weight",
            dist.Normal(0.0, 1.0)
            .expand([self.in_features, self.out_features])
            .to_event(2),
        )
        b = self.pyrox_sample(
            "bias", dist.Normal(0.0, 1.0).expand([self.out_features]).to_event(1)
        )
        return x @ W + b


layer = BayesianLinear(in_features=1, out_features=1, pyrox_name="linear")


def model(x, y=None):
    f = layer(x)[:, 0]  # (N,)
    numpyro.sample("obs", dist.Normal(f, 0.1), obs=y)


x = jnp.linspace(-1.0, 1.0, 50)[:, None]  # (N, 1)
y = 2.0 * x[:, 0] + 0.1 * jr.normal(jr.key(0), (50,))  # (N,)
sites = list(
    handlers.trace(handlers.seed(model, 0)).get_trace(x, y)
)  # ['linear.weight', 'linear.bias', 'obs']
guide = AutoNormal(model)
svi_result = SVI(model, guide, Adam(1e-2), Trace_ELBO()).run(
    jr.key(1), 1000, x, y, progress_bar=False
)
draws = Predictive(model, guide=guide, params=svi_result.params, num_samples=100)(
    jr.key(2), x
)["obs"]  # (S, N)

# 2. A GP whose kernel carries its own priors (pattern C): an evidence model.
kernel = pgp.RBF()
kernel.set_prior("lengthscale", dist.LogNormal(0.0, 1.0))
kernel.set_prior("variance", dist.LogNormal(0.0, 1.0))
X = jnp.linspace(0.0, 1.0, 40)[:, None]  # (N, 1)
Y = jnp.sin(6.0 * X[:, 0]) + 0.1 * jr.normal(jr.key(3), (40,))  # (N,)


def gp_model(X, Y):
    kernel.set_mode("model")  # θ ~ p(θ)
    noise = numpyro.sample("noise", dist.LogNormal(-2.0, 1.0))  # σ²
    # log p(Y | θ) = log 𝒩(Y; 0, K_θ + σ² I), through gaussx
    pgp.gp_factor("Y", pgp.GPPrior(kernel=kernel, X=X), Y, noise)


gp_guide = AutoNormal(gp_model)
gp_result = SVI(gp_model, gp_guide, Adam(1e-2), Trace_ELBO()).run(
    jr.key(4), 500, X, Y, progress_bar=False
)
```

The posterior mean of `linear.weight` comes out at about 1.97 against the
true slope of 2, and the GP's guide holds `RBF.lengthscale`,
`RBF.variance` and `noise`.

## Don't write it — use pyrox

| Don't write… | Use |
|---|---|
| `numpyro.sample` / `numpyro.param` inside an `eqx.Module` | `PyroxModule.pyrox_sample` / `pyrox_param` in a `@pyrox_method` |
| A per-parameter constraint + prior + guide registry | `Parameterized` (`register_param`, `set_prior`, `autoguide`, `set_mode`, `get_param`) |
| A kernel function and its hyperpriors | `pyrox_gp.RBF` / `Matern` / … (+ `set_prior`), built on `kernellib` |
| `-0.5 yᵀ (K + σ²I)⁻¹ y − ½ log|K + σ²I| − …` as a `numpyro.factor` | `pyrox_gp.gp_factor(name, GPPrior(kernel, X), y, noise_var)` |
| GP predictive mean / variance by hand | `GPPrior(kernel, X).condition(y, noise_var).predict(X_star)` |
| An SVGP ELBO, whitening, inducing-point KL | `SparseGPPrior` + a guide + `svgp_factor` / `svgp_elbo` |
| A Kalman filter for a temporal GP | `MarkovGPPrior` + `markov_gp_factor` |
| Laplace / EP for a classification GP | `LaplaceInference`, `ExpectationPropagation` with a `pyrox_gp` likelihood |
| A Bayesian dense layer, Flipout, SIREN, random-feature GP | `pyrox_nn.DenseReparameterization`, `DenseFlipout`, `BayesianSIREN`, `RandomFeatureGaussianProcess` |
| A deep-ensemble training loop | `pyrox.inference.ensemble_map` / `ensemble_vi` |
| A GLMM / BYM2 / SPDE model and its INLA approximation | `pyrox_lgm.LGM` + components + `inla()` |
| `cho_solve` / `slogdet` on a covariance | gaussx (`gaussx.solve`, `logdet` on a PSD-tagged operator) |

## Self-check before you finish

- `list(handlers.trace(handlers.seed(model, 0)).get_trace(...))` shows the
  site names you expect, with no duplicates and no unscoped names from
  inside a module.
- Two instances of one class have distinct `pyrox_name`s.
- No `Parameterized` module is rebuilt by `eqx.tree_at` / `jit` arguments.
- The model runs under SVI and `Predictive` (and NUTS if you use it), and
  recovers known parameters on simulated data.

If pyrox lacks what you need, keep your addition small and shaped like pyrox
(a `PyroxModule` whose sites go through the bridge) and consider proposing
it upstream at <https://github.com/jejjohnson/pyrox/issues>.
