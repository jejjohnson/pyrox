<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../docs/assets/logo-dark.svg">
  <img alt="pyrox" src="../../docs/assets/logo-light.svg" width="300">
</picture>

**The Equinox-to-NumPyro bridge for JAX: modules that declare their own sample and param sites, priors and guides, and ensembles of MAP and VI fits, on Equinox, NumPyro and optax.**

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
📚 [Core API](https://jejjohnson.github.io/pyrox/api/core/) ·
🧮 [Inference API](https://jejjohnson.github.io/pyrox/api/inference/) ·
🗂️ [Capability index](https://jejjohnson.github.io/pyrox/capabilities/)

</div>

---

An Equinox model keeps its weights in a pytree, whereas NumPyro wants every random quantity as a named site inside a model function. Bridging the two by hand means choosing a site name for every weight, keeping those names the same when `jit` or `eqx.tree_at` rebuilds the module, and making sure two layers never claim the same name. pyrox does that bridging once, so an Equinox module declares its own parameters as NumPyro sites and calling the module writes its part of the joint density,

```math
p(\mathbf{y}, \boldsymbol{\theta} \mid \mathbf{X}) = p(\mathbf{y} \mid f(\mathbf{X}; \boldsymbol{\theta})) \ p(\boldsymbol{\theta})
```

where $`\mathbf{X} \in \mathbb{R}^{N \times D_x}`$ holds the $`N`$ inputs, $`\mathbf{y} \in \mathbb{R}^{N}`$ the observations, $`f(\cdot; \boldsymbol{\theta})`$ is the Equinox module and $`\boldsymbol{\theta} \in \mathbb{R}^{D_\theta}`$ are the parameters it declares as sites (a weight with a prior is a sample site, a point estimate is a param site). The module never runs inference itself. NUTS, SVI and `Predictive` read its sites as they would in any NumPyro model, and `pyrox.inference` adds what NumPyro lacks, an ensemble of independent fits of the same model.

```python
# Shapes: N samples; D_x inputs; E ensemble members
# (BayesianLinear, model, the data and the keys as in the quickstart below)

# today (pyrox)
# f(X; θ), with θ recorded as the sites linear.weight and linear.bias
layer = BayesianLinear()
# today (numpyro)
# θ⁽ˢ⁾ ~ p(θ | y), by NUTS on the model that calls the layer
MCMC(NUTS(model), num_warmup=500, num_samples=500).run(key, inputs, targets)
# today (pyrox)
# ν⁽ᵉ⁾ for e = 1, …, E, the same model fitted by SVI E times under one vmap
guide_params, losses = ensemble_vi(
    model,
    AutoNormal(model),
    ensemble_size=4,
    num_epochs=2000,
    data=(inputs, targets),
    seed=key,
)
```

None of it is a new inference engine. A `PyroxModule` is an [`equinox`](https://github.com/patrick-kidger/equinox) module, so `jit`, `vmap`, `grad` and `eqx.tree_at` treat it as any other pytree. Its sites are [`numpyro`](https://github.com/pyro-ppl/numpyro) sites, because `pyrox_sample` and `pyrox_param` hand each one to `numpyro.sample` and `numpyro.param` under every handler, so NUTS, SVI with any autoguide and `Predictive` need no adapter. The ensemble VI runner wraps `numpyro.infer.SVI`, and the ensemble MAP runner steps [`optax`](https://github.com/google-deepmind/optax) optimisers.

This is the core of the pyrox workspace. The GP, neural-network, latent-Gaussian and state-space packages build on it; the [workspace README](../../README.md) introduces all of them.

## 🌟 Highlights

- 🏷️ **Stable site names**: every site is recorded as `<pyrox_name>.<site>`, with the class name when no `pyrox_name` is set and never an object `id`, so the names an MCMC run recorded still match after `eqx.tree_at`, `jit` or a checkpoint rebuilds the module.
- 🔁 **One site per call**: `pyrox_method` caches each site for the duration of a call, so a module that reads its weight twice records it once, and `call_context` and `call_contexts` hold that cache open across several calls of one module or of several.
- 🛡️ **No silent weight sharing**: NumPyro checks only sample sites for duplicate names, so two module instances that share a scope would share their param sites under SVI; `pyrox_param` raises a `ValueError` naming the clash instead, through `scope` and `block` handlers too.
- 🎛️ **Priors, constraints and guides declared once**: a `Parameterized` module registers each parameter on its support in `setup()`, attaches a prior and a `"delta"` or `"normal"` guide, and switches between the two with `set_mode`, so the module serves as its own SVI guide.
- 🧮 **Ensembles of fits**: `ensemble_map` fits $`E`$ MAP or maximum-likelihood estimates with any optax optimiser, and `ensemble_vi` fits $`E`$ variational posteriors with any NumPyro guide (autoguides included), each member from its own random start under one `vmap`; `param_group_optimizer` gives each labelled group of parameters its own optimiser.
- 🧾 **Shared observation protocols**: `pyrox.likelihoods` holds `Likelihood`, `Guide`, `AbstractObservation` and `DistLikelihood`, which pyrox-gp, pyrox-lgm and pyrox-ssm subclass, with no linear algebra and no gaussx import.
- ✅ **Tested under every handler**: `trace`, `seed`, `substitute`, `condition`, `scope`, `block`, `reparam`, `lift`, `mask`, `scale` and `do`, plus MCMC, SVI, `Predictive`, `jit`, `vmap` and `grad`, in [`test_core_numpyro_integration.py`](tests/test_core_numpyro_integration.py).

## 📐 Three ways to declare a site

We are opinionated about how Equinox and NumPyro compose, not about which piece you reach for. Three patterns cover the common cases, from the lightest machinery to the heaviest, and all three emit plain NumPyro sites, so they fit the same model to the same loss.

| Pattern | You write | The site | Reach for it when |
|---|---|---|---|
| A. Plain sites | `numpyro.sample` in the model, then `eqx.tree_at` into the module | named by you | one field of an existing Equinox module becomes random |
| B. `PyroxModule` | `self.pyrox_sample` and `self.pyrox_param` inside a `@pyrox_method` call | `<pyrox_name>.<site>`, cached per call | the module is itself probabilistic, such as a Bayesian layer |
| C. `Parameterized` | `register_param`, `set_prior` and `autoguide` in `setup()`, then `get_param` | $`\boldsymbol{\theta} \sim p(\boldsymbol{\theta})`$ in model mode, $`\boldsymbol{\theta} \sim q(\boldsymbol{\theta}; \boldsymbol{\nu})`$ in guide mode | constrained hyperparameters with priors, as in a GP kernel |

In pattern C, $`q(\boldsymbol{\theta}; \boldsymbol{\nu})`$ is the module's own guide with variational parameters $`\boldsymbol{\nu}`$. The `"normal"` guide is a normal distribution in unconstrained space, mapped back onto the parameter's support, and the `"delta"` guide is a point mass on the prior's support, which makes SVI with it a MAP fit.

Once the sites exist, `pyrox.inference` fits the model $`E`$ times, member $`e`$ starting from its own random key. The MAP runner takes a log joint and minimises, for each member and at each step $`i`$,

```math
\mathcal{L}(\boldsymbol{\theta}^{(e)}) = -\frac{N}{N_\text{batch}} \sum_{(\mathbf{x}_n, y_n) \in \mathcal{D}^{(i)}} \log p(y_n \mid \mathbf{x}_n, \boldsymbol{\theta}^{(e)}) - w \log p(\boldsymbol{\theta}^{(e)})
```

where $`\mathcal{D}^{(i)}`$ is the batch of $`N_\text{batch}`$ samples drawn at step $`i`$ (all $`N`$ when `batch_size` is `None`) and $`w`$ is `prior_weight`, so $`w = 0`$ gives maximum likelihood and $`w = 1`$ gives MAP. The VI runner takes a NumPyro model and guide and minimises the tempered negative ELBO,

```math
\mathcal{L}_\text{ELBO}(\boldsymbol{\nu}^{(e)}) = -\mathbb{E}_{q(\boldsymbol{\theta}; \boldsymbol{\nu}^{(e)})}[\log p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\theta})] + \beta \ \mathrm{KL}(q(\boldsymbol{\theta}; \boldsymbol{\nu}^{(e)}) \Vert p(\boldsymbol{\theta}))
```

with $`\beta`$ the `kl_weight`, which is 1 for the standard ELBO. Each runner comes in three layers: the functional primitives `ensemble_init`, `ensemble_loss` and `ensemble_step` for a training loop of your own, the `EnsembleMAP` and `EnsembleVI` classes with the `init`, `update` and `run` methods of `numpyro.infer.SVI`, and the one-call `ensemble_map` and `ensemble_vi`.

## 📥 Install

We have not put pyrox on PyPI, and the name `pyrox` there belongs to an unrelated project, so install it from GitHub. It brings in JAX, Equinox, NumPyro and einx, and needs Python 3.12 or newer.

```bash
pip install "pyrox @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox"
```

Two extras add the rest: `optax` for the ensemble MAP path and `param_group_optimizer`, and `colab` for what the example notebooks need (matplotlib, watermark and pandas). optax is imported only when a function needs it, which raises an `ImportError` naming the extra when it is missing; the ensemble VI path does not need it.

```bash
pip install "pyrox[optax] @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox"
```

## 📈 Quickstart: one Bayesian layer, NUTS and an ensemble of SVI fits

A line through 50 noisy points, $`y_n = 2 x_n + 0.5 + \varepsilon_n`$ with $`\varepsilon_n \sim \mathcal{N}(0, \sigma_\varepsilon^2)`$ and $`\sigma_\varepsilon = 0.1`$. The module is $`f(\mathbf{X}; \boldsymbol{\theta}) = \mathbf{X} \mathbf{w} + b`$, whose parameters $`\boldsymbol{\theta} = (\mathbf{w}, b)`$ are the slope $`\mathbf{w} \in \mathbb{R}^{D_x}`$ and the intercept $`b`$, each with a standard normal prior. NUTS draws $`S`$ samples $`\boldsymbol{\theta}^{(s)}`$ from the posterior, and `ensemble_vi` fits $`E`$ mean-field normal guides with variational parameters $`\boldsymbol{\nu}^{(e)}`$ to the same model.

```python
import einx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer.autoguide import AutoNormal
from pyrox._core import PyroxModule, pyrox_method
from pyrox.inference import ensemble_vi


# The package's tests run in double precision, so the quickstart does too.
jax.config.update("jax_enable_x64", True)

# Shapes: N = 50 samples; D_x = 1 input; S = 500 posterior draws; E = 4 members


class BayesianLinear(PyroxModule):
    pyrox_name: str = "linear"
    in_features: int = 1

    @pyrox_method
    def __call__(self, inputs: Float[Array, "N D_x"]) -> Float[Array, " N"]:
        # today (pyrox)
        # w ~ 𝒩(0, I), recorded as the site linear.weight
        # → (D_x,)
        weight = self.pyrox_sample(
            "weight", dist.Normal(0.0, 1.0).expand([self.in_features]).to_event(1)
        )
        # today (pyrox)
        # b ~ 𝒩(0, 1), recorded as the site linear.bias
        # → ()
        bias = self.pyrox_sample("bias", dist.Normal(0.0, 1.0))
        # f(X; θ) = X w + b
        # (N, D_x) → (N,)
        return einx.dot("n d, d -> n", inputs, weight) + bias


layer = BayesianLinear()


# user
# y = f(X; θ) + ε, with ε ~ 𝒩(0, σ_ε² I) and σ_ε = 0.1
def model(inputs: Float[Array, "N D_x"], targets: Float[Array, " N"]) -> None:
    # user
    # f(X; θ)
    # (N, D_x) → (N,)
    mean = layer(inputs)
    # today (numpyro)
    # p(y | X, θ) = 𝒩(y; f(X; θ), σ_ε² I), recorded as the site obs
    numpyro.sample("obs", dist.Normal(mean, 0.1).to_event(1), obs=targets)


# X, evenly spaced on [−1, 1]
# → (N, D_x)
inputs = jnp.linspace(-1.0, 1.0, 50)[:, None]
# ε ~ 𝒩(0, σ_ε² I)
# → (N,)
noise = 0.1 * jr.normal(jr.key(0), (50,))
# y = 2 X + 0.5 + ε
# (N, D_x) → (N,)
targets = 2.0 * inputs[:, 0] + 0.5 + noise

# today (numpyro)
# the names NumPyro records for the sites
trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(inputs, targets)

# today (numpyro)
mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500, progress_bar=False)
# today (numpyro)
# θ⁽ˢ⁾ ~ p(θ | y) ∝ p(y | X, θ) p(θ), by NUTS
mcmc.run(jr.key(1), inputs, targets)
# today (numpyro)
# 𝔼[w | y], the posterior mean of the slope
# (S, D_x) → (D_x,)
nuts_slope = mcmc.get_samples()["linear.weight"].mean(axis=0)

# today (pyrox)
# ν⁽ᵉ⁾ = argmin_ν ℒ_ELBO(ν) for e = 1, …, E, the same model fitted E times
guide_params, losses = ensemble_vi(
    model,
    AutoNormal(model),
    ensemble_size=4,
    num_epochs=2000,
    data=(inputs, targets),
    seed=jr.key(2),
)
# the mean of w under each member's q(θ; ν⁽ᵉ⁾)
# → (E, D_x)
member_slopes = guide_params["linear.weight_auto_loc"]

print(list(trace), nuts_slope.round(3), member_slopes[:, 0].round(3))
```

This prints `['linear.weight', 'linear.bias', 'obs'] [1.958] [1.96  1.963 1.955 1.957]`: the three sites NumPyro sees, named after the module, the NUTS posterior mean of the slope, and the slope each of the four SVI members settled on, all within 0.01 of one another and close to the true slope of 2. The posterior of a line has one mode, so the members agree; on a multimodal objective (a neural network, a latent-factor GP) their spread is what the ensemble is for. Nothing here is specific to a line, so a pyrox-nn layer or a pyrox-gp kernel in place of `BayesianLinear` runs through the same NUTS and SVI calls.

The [tutorials](#-tutorials) take this further: the same model written with each of the three patterns, and the ensemble runners from the functional primitives up.

## 🔬 What else is in the box

The package has three public namespaces, each with its own `__all__`; `pyrox` itself exports only the version and the namespaces.

| Module | Symbols | Notes |
|---|---|---|
| `pyrox._core` | `PyroxModule`, `pyrox_method`, `Parameterized`, `PyroxParam`, `PyroxSample`, `call_context`, `call_contexts`, `freeze` | The bridge's public home despite the underscore; `freeze` runs `module.frozen()` under one `call_context`, so each hyperparameter is drawn once |
| `pyrox.inference` | `ensemble_map`, `ensemble_vi`, `ensemble_predict`, `EnsembleMAP`, `EnsembleVI`, `ensemble_init`, `ensemble_loss`, `ensemble_step`, `EnsembleState`, `EnsembleResult`, `param_group_optimizer` | The MAP path and `param_group_optimizer` need the `optax` extra; `ensemble_predict` maps a predictive over the leading ensemble axis |
| `pyrox.likelihoods` | `Likelihood`, `Guide`, `GaussianLikelihood`, `AbstractObservation`, `DistLikelihood`, `likelihood_accepts_inputs` | The observation model $`p(\mathbf{y} \mid \mathbf{f})`$ and the variational posterior $`q(\mathbf{f})`$ as protocols; `DistLikelihood` wraps any NumPyro distribution |

`PyroxParam` and `PyroxSample` are value objects that nothing in the package consumes yet. `pyrox.inference.LastLayerLaplace` and `last_layer_laplace` moved to pyrox-nn and still resolve here with a `DeprecationWarning`.

## 📓 Tutorials

| # | Tutorial | What it shows |
|---|---|---|
| 1 | [Regression masterclass, pattern A](https://jejjohnson.github.io/pyrox/notebooks/regression_masterclass_treeat/) | plain NumPyro sites spliced into Equinox modules with `eqx.tree_at` |
| 2 | [Regression masterclass, pattern B](https://jejjohnson.github.io/pyrox/notebooks/regression_masterclass_pyrox_sample/) | the same models as `PyroxModule`s that own their sites |
| 3 | [Regression masterclass, pattern C](https://jejjohnson.github.io/pyrox/notebooks/regression_masterclass_parameterized/) | the same models as `Parameterized` modules with priors, constraints and guides |
| 4 | [Ensemble primitives](https://jejjohnson.github.io/pyrox/notebooks/ensemble_primitives_tutorial/) | a training loop of your own from `ensemble_init`, `ensemble_loss` and `ensemble_step` |
| 5 | [Ensemble runners](https://jejjohnson.github.io/pyrox/notebooks/ensemble_runner_tutorial/) | `EnsembleMAP` and `EnsembleVI` with the `init`, `update` and `run` methods |

## 📖 Documentation

- **[Documentation](https://jejjohnson.github.io/pyrox/)**: the whole workspace, with the tutorials and examples of every package.
- **[Core API](https://jejjohnson.github.io/pyrox/api/core/)**: `PyroxModule`, `Parameterized` and the call contexts, generated from the docstrings.
- **[Inference API](https://jejjohnson.github.io/pyrox/api/inference/)**: the three layers of the ensemble runners and `param_group_optimizer`.
- **[Capability index](https://jejjohnson.github.io/pyrox/capabilities/)**: every public name in the workspace and in the libraries it builds on, to check before writing something new.
- **[`AGENTS.md`](AGENTS.md)**: the contracts this package keeps (site semantics, stable scope names, the duplicate-param guard) and the tests that hold them.

## 🧩 Ecosystem

pyrox is the core of a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) in the GeoML tier of the [GeoStack](https://github.com/jejjohnson/geostack). It owns the bridge between a module and its sites, and composes the rest in:

| Concern | Owner |
|---|---|
| Modules as pytrees, and the transforms over them (`PyroxModule` subclasses `eqx.Module`) | [`equinox`](https://github.com/patrick-kidger/equinox) |
| Sample and param sites, handlers, NUTS, SVI, autoguides and `Predictive` | [`numpyro`](https://github.com/pyro-ppl/numpyro) |
| Optimisers for the ensemble MAP path (the `optax` extra) | [`optax`](https://github.com/google-deepmind/optax) |
| Array programming, `jit`, `vmap` and `grad` | [`jax`](https://github.com/jax-ml/jax) |
| Named-axis contractions | [`einx`](https://github.com/fferflo/einx) |

Dependencies point one way. Four packages of the workspace import the core: [pyrox-gp](../pyrox-gp) (`_core`, `likelihoods` and `param_group_optimizer`), [pyrox-nn](../pyrox-nn) (`_core` and the ensemble runners), [pyrox-lgm](../pyrox-lgm) (`AbstractObservation`) and [pyrox-ssm](../pyrox-ssm) (`likelihoods`). xtremax installs it in its `tutorials` dependency group. The core imports none of them, and no gaussx, lineax, kernellib or geonnax either, which [`test_core_import_guard.py`](tests/test_core_import_guard.py) checks.

## 🌳 Repository layout

```
packages/pyrox/
├── src/pyrox/
│   ├── __init__.py        ← the version and the three namespaces
│   ├── _core/             ← PyroxModule, Parameterized, the call contexts
│   ├── inference/         ← the ensemble runners and param_group_optimizer
│   └── likelihoods.py     ← the Likelihood, Guide and observation protocols
├── tests/                 ← the site semantics under every handler, ensembles against ridge and OLS
└── AGENTS.md              ← the contracts and the tests that hold them
```

## ⚙️ Development

The package is developed inside the workspace, so clone the repository and run the tasks from its root.

```bash
git clone https://github.com/jejjohnson/pyrox.git
cd pyrox
make install                                       # uv sync --all-groups + pre-commit hooks
uv run pytest --no-cov packages/pyrox/tests -v     # this package's tests
make lint                                          # ruff check on the whole repo
make typecheck                                     # ty check on every package
```

`make help` lists every task. The root [`AGENTS.md`](../../AGENTS.md) holds the rules for the whole workspace, and this package's [`AGENTS.md`](AGENTS.md) adds the contracts of the core.

## 📄 License

MIT. See [LICENSE](LICENSE).
