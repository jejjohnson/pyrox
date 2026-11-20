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
# (net, the two MLP classes, log_joint, init_fn and the data as in the quickstart)

# today (pyrox)
# f(X; θ), an Equinox MLP whose weights θ are the sites mlp.hidden_weight, …
mlp_b = BayesianMLP(net)
# today (pyrox)
# the same MLP, with p(θ) and its own guide q(θ; ν) declared once in setup()
mlp_c = ParameterizedMLP(net)
# today (pyrox)
# θ⁽ᵉ⁾ = argmin_θ ℒ(θ) for e = 1, …, E, a deep ensemble under one vmap
members, losses = ensemble_map(
    log_joint,
    init_fn,
    ensemble_size=8,
    num_epochs=2000,
    data=(inputs, targets),
    seed=key,
)
```

None of it is a new inference engine. A `PyroxModule` is an [`equinox`](https://github.com/patrick-kidger/equinox) module, so `jit`, `vmap`, `grad` and `eqx.tree_at` treat it as any other pytree. Its sites are [`numpyro`](https://github.com/pyro-ppl/numpyro) sites, because `pyrox_sample` and `pyrox_param` hand each one to `numpyro.sample` and `numpyro.param` under every handler, so NUTS, SVI with any autoguide and `Predictive` need no adapter. The ensemble VI runner wraps `numpyro.infer.SVI`, and the ensemble MAP runner steps [`optax`](https://github.com/google-deepmind/optax) optimisers.

This is the core of the pyrox workspace. The GP, neural-network, latent-Gaussian and state-space packages build on it; the [workspace README](../../README.md) introduces all of them.

<p align="center"><img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_overview/pyrox_overview.png" alt="pyrox in one picture: you supply an Equinox module, priors and a NumPyro model or a log joint; they define one joint density p(y, θ | X) whose parameters are named sites; pyrox declares the sites in one of three ways (plain sites, PyroxModule, Parameterized), holds one draw across calls and fits the model E times; out come the named sites, posterior samples, a variational posterior and E ensemble members; equinox, numpyro and optax underneath" width="100%"></p>

## 🌟 Highlights

- 🏷️ **Stable site names**: every site is recorded as `<pyrox_name>.<site>`, with the class name when no `pyrox_name` is set and never an object `id`, so the names an MCMC run recorded still match after `eqx.tree_at`, `jit` or a checkpoint rebuilds the module.
- 🔁 **One site per call**: `pyrox_method` caches each site for the duration of a call, so a module that reads its weight twice records it once, and `call_context` and `call_contexts` hold that cache open across several calls of one module or of several.
- 🛡️ **No silent weight sharing**: NumPyro checks only sample sites for duplicate names, so two module instances that share a scope would share their param sites under SVI; `pyrox_param` raises a `ValueError` naming the clash instead, through `scope` and `block` handlers too.
- 🎛️ **Priors, constraints and guides declared once**: a `Parameterized` module registers each parameter on its support in `setup()`, attaches a prior and a `"delta"` or `"normal"` guide, and switches between the two with `set_mode`, so the module serves as its own SVI guide.
- 🧮 **Ensembles of fits**: `ensemble_map` fits $`E`$ MAP or maximum-likelihood estimates with any optax optimiser, and `ensemble_vi` fits $`E`$ variational posteriors with any NumPyro guide (autoguides included), each member from its own random start under one `vmap`; `param_group_optimizer` gives each labelled group of parameters its own optimiser.
- 🧾 **Shared observation protocols**: `pyrox.likelihoods` holds `Likelihood`, `Guide`, `AbstractObservation` and `DistLikelihood`, which pyrox-gp, pyrox-lgm and pyrox-ssm subclass, with no linear algebra and no gaussx import.
- ✅ **Tested under every handler**: `trace`, `seed`, `substitute`, `condition`, `scope`, `block`, `reparam`, `lift`, `mask`, `scale` and `do`, plus MCMC, SVI, `Predictive`, `jit`, `vmap` and `grad`, in [`test_core_numpyro_integration.py`](tests/test_core_numpyro_integration.py).

## 📐 Three ways to declare a site

We are opinionated about how Equinox and NumPyro compose, not about which piece you reach for. Three patterns cover the common cases, from the lightest machinery to the heaviest, and all three emit plain NumPyro sites, so one model written three ways is one joint density, which the quickstart checks.

| Pattern | You write | The site | Reach for it when |
|---|---|---|---|
| A. Plain sites | `numpyro.sample` in the model, then `eqx.tree_at` into the module | named by you | one field of an existing Equinox module becomes random |
| B. `PyroxModule` | `self.pyrox_sample` and `self.pyrox_param` inside a `@pyrox_method` call | `<pyrox_name>.<site>`, cached per call | the module is itself probabilistic, such as a Bayesian network |
| C. `Parameterized` | `register_param`, `set_prior` and `autoguide` in `setup()`, then `get_param` | $`\boldsymbol{\theta} \sim p(\boldsymbol{\theta})`$ in model mode, $`\boldsymbol{\theta} \sim q(\boldsymbol{\theta}; \boldsymbol{\nu})`$ in guide mode | constrained parameters with priors, and the module as its own guide |

In pattern C, $`q(\boldsymbol{\theta}; \boldsymbol{\nu})`$ is the module's own guide with variational parameters $`\boldsymbol{\nu}`$. The `"normal"` guide is a normal distribution in unconstrained space, mapped back onto the parameter's support, and the `"delta"` guide is a point mass on the prior's support, which makes SVI with it a MAP fit.

Once the model exists, `pyrox.inference` fits it $`E`$ times, member $`e`$ starting from its own random key. The MAP runner takes a log joint and minimises, for each member and at each step $`i`$,

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

## 🧠 Quickstart: one Bayesian MLP, three ways

One MLP with $`H = 16`$ tanh units, $`f(\mathbf{x}; \boldsymbol{\theta}) = \mathbf{W}_2 \tanh(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2`$, made Bayesian once per pattern. Its weights $`\boldsymbol{\theta} = (\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2)`$, with $`\mathbf{W}_1 \in \mathbb{R}^{H \times D_x}`$, $`\mathbf{b}_1 \in \mathbb{R}^{H}`$, $`\mathbf{W}_2 \in \mathbb{R}^{1 \times H}`$ and $`\mathbf{b}_2 \in \mathbb{R}^{1}`$, get a standard normal prior $`p(\boldsymbol{\theta})`$, and the data are 64 noisy points of $`y_n = \sin(3 x_n) + \varepsilon_n`$ with $`\varepsilon_n \sim \mathcal{N}(0, \sigma_\varepsilon^2)`$ and $`\sigma_\varepsilon = 0.1`$. The setup is shared by all three: an ordinary `eqx.nn.MLP`, the prior, `splice` (which puts a set of weights into the MLP with `eqx.tree_at`) and one NumPyro model that takes any of the Bayesian versions.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float


# The package's tests run in double precision, so the quickstart does too.
jax.config.update("jax_enable_x64", True)

# Shapes: N = 64 samples; D_x = 1 input; H = 16 hidden units

# today (equinox)
# f(x; θ) = W₂ tanh(W₁ x + b₁) + b₂, an ordinary Equinox MLP
net = eqx.nn.MLP(1, "scalar", 16, 1, activation=jnp.tanh, key=jr.key(0))
NAMES = ("hidden_weight", "hidden_bias", "output_weight", "output_bias")


# user
# θ = (W₁, b₁, W₂, b₂)
def weights_of(mlp: eqx.nn.MLP) -> tuple[Array, ...]:
    hidden, output = mlp.layers
    return hidden.weight, hidden.bias, output.weight, output.bias


# user
# p(θ) = 𝒩(0, I), at the shape of one weight array
def prior(value: Array) -> dist.Distribution:
    return dist.Normal(0.0, 1.0).expand(value.shape).to_event(value.ndim)


# user
# f(X; θ), the MLP with its weights replaced by θ, on each of the N inputs
def splice(mlp: eqx.nn.MLP, weights: list[Array], inputs: Array) -> Array:
    return jax.vmap(eqx.tree_at(weights_of, mlp, tuple(weights)))(inputs)


# user
# p(y | X, θ) = 𝒩(y; f(X; θ), σ_ε² I), with σ_ε = 0.1
def model(mlp, inputs: Float[Array, "N D_x"], targets: Float[Array, " N"]) -> None:
    numpyro.sample("obs", dist.Normal(mlp(inputs), 0.1).to_event(1), obs=targets)


# X evenly spaced on [−1, 1], and y = sin(3 X) + ε
# → (N, D_x), (N,)
inputs = jnp.linspace(-1.0, 1.0, 64)[:, None]
targets = jnp.sin(3.0 * inputs[:, 0]) + 0.1 * jr.normal(jr.key(1), (64,))
```

**A. Plain sites.** The model samples $`\boldsymbol{\theta}`$ with `numpyro.sample` and splices it into the MLP. This needs nothing from pyrox, but the function names every site itself.

```python
# Shapes: N samples; D_x inputs (net, prior and splice as above)


def plain_mlp(inputs: Float[Array, "N D_x"]) -> Float[Array, " N"]:
    # today (numpyro)
    # θ ~ p(θ), recorded as the sites hidden_weight, …, output_bias
    weights = [
        numpyro.sample(name, prior(value))
        for name, value in zip(NAMES, weights_of(net), strict=True)
    ]
    # (N, D_x) → (N,)
    return splice(net, weights, inputs)
```

**B. A `PyroxModule` owns its sites.** The module samples its own weights inside a `@pyrox_method` call, and pyrox records them as `mlp.<name>`, so a second `BayesianMLP(net, pyrox_name="mlp2")` in the same model gets names of its own.

```python
from pyrox._core import PyroxModule, pyrox_method


# Shapes: N samples; D_x inputs (prior and splice as above)


class BayesianMLP(PyroxModule):
    net: eqx.nn.MLP
    pyrox_name: str = eqx.field(static=True, default="mlp")

    @pyrox_method
    def __call__(self, inputs: Float[Array, "N D_x"]) -> Float[Array, " N"]:
        # today (pyrox)
        # θ ~ p(θ), recorded as the sites mlp.hidden_weight, …, mlp.output_bias
        weights = [
            self.pyrox_sample(name, prior(value))
            for name, value in zip(NAMES, weights_of(self.net), strict=True)
        ]
        # (N, D_x) → (N,)
        return splice(self.net, weights, inputs)
```

**C. `Parameterized` declares each weight's prior and guide once.** `setup()` registers each weight, attaches $`p(\boldsymbol{\theta})`$ and a normal guide $`q(\boldsymbol{\theta}; \boldsymbol{\nu})`$ with variational parameters $`\boldsymbol{\nu}`$, and `get_param` samples whichever of the two the mode asks for. After `set_mode("guide")` the same call registers $`\boldsymbol{\nu}`$ as the param sites `mlp.hidden_weight_loc`, `mlp.hidden_weight_scale` and so on, so the module is its own SVI guide.

```python
from pyrox._core import Parameterized


# Shapes: N samples; D_x inputs (prior and splice as above)


class ParameterizedMLP(Parameterized):
    net: eqx.nn.MLP
    pyrox_name: str = eqx.field(static=True, default="mlp")

    def setup(self) -> None:
        for name, value in zip(NAMES, weights_of(self.net), strict=True):
            # today (pyrox)
            # θ on its support, p(θ) = 𝒩(0, I) and a normal guide q(θ; ν)
            self.register_param(name, value)
            self.set_prior(name, prior(value))
            self.autoguide(name, "normal")

    @pyrox_method
    def __call__(self, inputs: Float[Array, "N D_x"]) -> Float[Array, " N"]:
        # today (pyrox)
        # θ ~ p(θ) in model mode, θ ~ q(θ; ν) in guide mode
        weights = [self.get_param(name) for name in NAMES]
        # (N, D_x) → (N,)
        return splice(self.net, weights, inputs)
```

**One joint.** All three emit plain NumPyro sites, so at the same weights they write the same log joint.

```python
from numpyro.infer.util import log_density


# Shapes: N samples; D_x inputs (net, model and the three patterns as above)

# θ, the MLP's initial weights, under A's site names and under B's and C's
plain = dict(zip(NAMES, weights_of(net), strict=True))
scoped = {f"mlp.{name}": value for name, value in plain.items()}
patterns = [
    (plain_mlp, plain),
    (BayesianMLP(net), scoped),
    (ParameterizedMLP(net), scoped),
]
# today (numpyro)
# log p(y, θ | X), the joint each pattern writes, at the same θ
log_joints = [
    log_density(model, (mlp, inputs, targets), {}, values)[0]
    for mlp, values in patterns
]

print(list(scoped), [round(float(value), 3) for value in log_joints])
```

This prints `['mlp.hidden_weight', 'mlp.hidden_bias', 'mlp.output_weight', 'mlp.output_bias'] [-3549.83, -3549.83, -3549.83]`: the site names B and C record, and the same log joint from all three models (low, because the initial weights do not fit the data yet). The regression masterclass fits a model with each pattern: [A](https://jejjohnson.github.io/pyrox/notebooks/regression_masterclass_treeat/), [B](https://jejjohnson.github.io/pyrox/notebooks/regression_masterclass_pyrox_sample/) and [C](https://jejjohnson.github.io/pyrox/notebooks/regression_masterclass_parameterized/).

## 👥 A deep ensemble of the same MLP

`pyrox.inference` fits a model $`E`$ times under one `vmap`, each member from its own random start. `ensemble_map` minimises the loss $`\mathcal{L}(\boldsymbol{\theta}^{(e)})`$ above for each member, which for a neural network with `prior_weight` at its default of 1 is a deep ensemble of MAP fits. It takes a log joint and an `init_fn` rather than sites, so the members are plain `eqx.nn.MLP`s. The block continues from the quickstart and needs the `optax` extra; $`\mathbf{X}_* \in \mathbb{R}^{N_* \times D_x}`$ holds three test inputs, two inside the data and $`x = 2`$ outside it.

```python
from pyrox.inference import ensemble_map, ensemble_predict


# Shapes: E = 8 members; N_* = 3 test inputs (prior, splice and the data as above)


# user
# log p(y | X, θ) and log p(θ), the two terms of the MAP loss
def log_joint(
    mlp: eqx.nn.MLP, inputs: Float[Array, "N D_x"], targets: Float[Array, " N"]
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    mean = splice(mlp, list(weights_of(mlp)), inputs)
    log_likelihood = dist.Normal(mean, 0.1).log_prob(targets).sum()
    log_prior = sum(prior(value).log_prob(value) for value in weights_of(mlp))
    return log_likelihood, log_prior


# user
# θ⁽ᵉ⁾ at the start, a fresh MLP from member e's key
def init_fn(key: Array) -> eqx.nn.MLP:
    return eqx.nn.MLP(1, "scalar", 16, 1, activation=jnp.tanh, key=key)


# today (pyrox)
# θ⁽ᵉ⁾ = argmin_θ ℒ(θ) for e = 1, …, E, each from its own start
members, losses = ensemble_map(
    log_joint,
    init_fn,
    ensemble_size=8,
    num_epochs=2000,
    data=(inputs, targets),
    seed=jr.key(2),
    learning_rate=1e-2,
)
# X_*, two inputs inside the data and one outside
# → (N_*, D_x)
test_inputs = jnp.array([[0.0], [0.5], [2.0]])
# today (pyrox)
# f(X_*; θ⁽ᵉ⁾) for every member e
# (N_*, D_x) → (E, N_*)
predictions = ensemble_predict(members, lambda mlp, x: jax.vmap(mlp)(x), test_inputs)

print(predictions.mean(axis=0).round(2), predictions.std(axis=0).round(2))
```

This prints `[ 0.04  0.99 -1.81] [0.   0.   0.24]`: the members' mean prediction at $`x = 0`$, 0.5 and 2, and their spread. Inside the data the eight members agree to two decimals and come within 0.04 of $`\sin(3x)`$. At $`x = 2`$ they disagree by 0.24, which flags the extrapolation, but their mean of −1.81 is far from $`\sin(6) = -0.28`$, so the spread there is a warning and not an error bar. `ensemble_vi` does the same for SVI fits of a NumPyro model and guide, and the [ensemble tutorials](#-tutorials) build both runners from the functional primitives up.

## 🔬 What else is in the box

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_roles/pyrox_roles-dark.svg">
    <img alt="pyrox on the GeoStack role ladder: the ensemble runners as algorithms; the bases your models subclass, the observation models and the fitted ensembles as models; EnsembleMAP, EnsembleVI and param_group_optimizer as strategies; the verbs that make a site and hold one draw across calls as primitives; two site descriptors as structures; NumPyro beside the ladder as the floor every site is handed to, and ensemble_init as setup" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_roles/pyrox_roles-light.svg" width="100%">
  </picture>
</p>

Each public name sits on one rung of the GeoStack role ladder, by the job it does.

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

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_ecosystem/pyrox_ecosystem-dark.svg">
    <img alt="The packages that import pyrox's core: pyrox-gp (_core and likelihoods), pyrox-nn (_core and inference), pyrox-lgm and pyrox-ssm (likelihoods), all in the pyrox workspace in GeoML; pyrox builds on JAX, Equinox, NumPyro and einx, and on optax through an extra" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_ecosystem/pyrox_ecosystem-light.svg" width="100%">
  </picture>
</p>

pyrox is the core of a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) in the GeoML tier of the [GeoStack](https://github.com/jejjohnson/geostack). It owns the bridge between a module and its sites, and composes the rest in:

| Concern | Owner |
|---|---|
| Modules as pytrees, and the transforms over them (`PyroxModule` subclasses `eqx.Module`) | [`equinox`](https://github.com/patrick-kidger/equinox) |
| Sample and param sites, handlers, NUTS, SVI, autoguides and `Predictive` | [`numpyro`](https://github.com/pyro-ppl/numpyro) |
| Optimisers for the ensemble MAP path (the `optax` extra) | [`optax`](https://github.com/google-deepmind/optax) |
| Array programming, `jit`, `vmap` and `grad` | [`jax`](https://github.com/jax-ml/jax) |
| Named-axis contractions | [`einx`](https://github.com/fferflo/einx) |

Dependencies point one way. Four packages of the workspace import the core: [pyrox-gp](../pyrox-gp) (`_core` and `likelihoods`), [pyrox-nn](../pyrox-nn) (`_core` and the ensemble runners), [pyrox-lgm](../pyrox-lgm) (`AbstractObservation`) and [pyrox-ssm](../pyrox-ssm) (`likelihoods`); the workspace's other packages do not import it. The core imports none of them, and no gaussx, lineax, kernellib or geonnax either, which [`test_core_import_guard.py`](tests/test_core_import_guard.py) checks.

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
