<div align="center">

<img alt="pyrox-nn" src="../../docs/assets/icon-nn.svg" width="120">

# pyrox-nn

**Bayesian and uncertainty-aware neural network layers for JAX: weights as NumPyro sites, Gaussian processes as random-feature layers, a post-hoc Laplace posterior and an estimator for neural fields, on pyrox, geonnax, functax, kernellib, argminax, gaussx and optax.**

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
📚 [NN API](https://jejjohnson.github.io/pyrox/api/nn/) ·
🗺️ [Estimator API](https://jejjohnson.github.io/pyrox/api/api/) ·
🗂️ [Capability index](https://jejjohnson.github.io/pyrox/capabilities/)

</div>

---

A neural network fitted to a hundred points says nothing about where it is guessing. Outside the data it extrapolates with the same confidence as inside, and the usual remedies (a Bayesian network, a Gaussian process, a Laplace posterior, an ensemble) each need their own priors, guides and bookkeeping. pyrox-nn ships them as layers: each layer is a `PyroxModule` that declares its weights as NumPyro sites with their priors, so stacking layers writes the joint density of the whole network,

```math
p(\mathbf{y}, \boldsymbol{\theta} \mid \mathbf{X}) = p(\mathbf{y} \mid f(\mathbf{X}; \boldsymbol{\theta})) \ \prod_{l=1}^{L} p(\boldsymbol{\theta}_l)
```

where $`\mathbf{X} \in \mathbb{R}^{N \times D_x}`$ holds the $`N`$ inputs, $`\mathbf{y} \in \mathbb{R}^{N}`$ the observations, $`f = g_L \circ \ldots \circ g_1`$ is the network of $`L`$ layers and $`\boldsymbol{\theta}_l`$ are the weights layer $`g_l`$ declares, with its own prior $`p(\boldsymbol{\theta}_l)`$. A dense layer puts the prior on its weight matrix, a random-feature layer on its spectral frequencies, so that the network is a Gaussian process, and an output head on its noise. NUTS, SVI and `Predictive` read the sites as in any NumPyro model.

```python
# Shapes: N samples; D_x inputs; E ensemble members
# (inputs, targets, mlp, predict, is_last and siren as in the quickstart)

# today (pyrox-nn)
# W₁ ~ 𝒩(0, I), b₁ ~ 𝒩(0, I), a Bayesian dense layer
hidden = DenseReparameterization(1, 16, pyrox_name="hidden")
# today (pyrox-nn)
# φ(X), random Fourier features of an RBF kernel, frequencies Ω ~ 𝒩(0, I)
features = RBFFourierFeatures.init(1, 32, lengthscale=0.5)
# today (pyrox-nn)
# q(w), a Laplace posterior over the last layer of a network already fitted
posterior = last_layer_laplace(
    predict, mlp, inputs, targets, sigma=0.1, param_filter=is_last
)
# today (pyrox-nn)
# a Bayesian SIREN as a neural field f(x), to be fitted E times
estimator = NeuralFieldEstimator(model_fn=siren, ensemble_size=8)
# today (pyrox-nn)
# the fitted members, each a MAP fit from its own start
fitted = estimator.fit(inputs, targets, seed=0)
```

None of it is new machinery. Each layer is a [`pyrox`](../pyrox) module, so its sites are scoped by `pyrox_name`, cached per call and handed to [`numpyro`](https://github.com/pyro-ppl/numpyro). The deterministic network inside most layers is a [`geonnax`](https://github.com/jejjohnson/geonnax) core whose arrays the layer swaps for sites with `eqx.tree_at`, and geonnax's encoders are re-exported unchanged. Bases come from [`functax`](https://github.com/jejjohnson/functax), spectral densities from [`kernellib`](https://github.com/jejjohnson/kernellib), the Laplace curvature and evidence from [`argminax`](https://github.com/jejjohnson/argminax) and its linear algebra from [`gaussx`](https://github.com/jejjohnson/gaussx). The estimator fits its ensembles with `pyrox.inference`, which steps [`optax`](https://github.com/google-deepmind/optax) optimisers.

pyrox-nn is one package of the pyrox workspace; the [workspace README](../../README.md) introduces the others.

## 🌟 Highlights

- 🧱 **Layers that own their weights**: every layer is a `PyroxModule` that registers its weights as `<pyrox_name>.<site>`, so a stack of layers is one NumPyro model and NUTS, SVI and `Predictive` need no adapter.
- 🔁 **Deterministic cores, made Bayesian**: the SNGP, VSSGP, rank-1, BatchEnsemble, multiplicative-filter, conditioning, output-head and BNF layers hold a geonnax core and splice sampled or learned arrays into it, so the architecture lives in geonnax once.
- 🌊 **Gaussian processes as layers**: random Fourier features for the RBF, Matérn, Laplace and arc-cosine kernels, `HSGPFeatures` and `SphericalHarmonicFeatures` for a given kernel's spectral density on a bounded box or on the sphere, `DeepVSSGP` for stacks of them, and `RandomFeatureGaussianProcess` as an SNGP output head.
- 🎯 **Uncertainty after the fit**: `last_layer_laplace` puts a generalised Gauss-Newton Laplace posterior on any group of weights of a fitted pytree, and `optimize_prior_precision` sets its prior by maximising the evidence.
- 🗺️ **Neural fields in one call**: `NeuralFieldEstimator` fits any field (a `BayesianSIREN`, a Fourier or Gabor network, the Bayesian Neural Field) as an ensemble of MAP or VI fits, with fixed, learned or input-dependent noise, padded rows masked out, and `fit_xarray` for gridded data.
- 🌐 **Geographic inputs**: longitude and latitude scaling, cyclic, Cartesian, spherical-harmonic and Slepian encoders, re-exported from geonnax, and pandas preprocessing for space-time tables in `pyrox_nn.preprocessing`.
- 🚧 **No GP package needed**: pyrox-nn never imports pyrox-gp, which [`test_nn_import_guard.py`](tests/test_nn_import_guard.py) checks, yet `HSGPFeatures` and `SphericalHarmonicFeatures` accept a pyrox-gp kernel with priors when you pass one.

## 📐 Where the uncertainty goes

A layer can be uncertain in four places, and the package groups its layers by them. All four can sit in one network.

| Where | Layers | What is random | Reach for it when |
|---|---|---|---|
| Weights | `DenseReparameterization`, `DenseFlipout`, `DenseHierarchical`, `BayesianSIREN`, … | $`\boldsymbol{\theta} \sim p(\boldsymbol{\theta})`$ | a Bayesian neural network |
| Features | `RBFFourierFeatures`, `HSGPFeatures`, `RandomFeatureGaussianProcess`, `DeepVSSGP`, … | the frequencies $`\boldsymbol{\Omega}`$ and the readout $`\mathbf{w}`$ | a Gaussian-process prior at the cost of a layer |
| Output | `GaussianRegressionHead`, `MCSoftmaxDenseFA`, `NCPNormalOutput`, … | the noise $`\sigma(\mathbf{x})`$ | noise that changes with the input |
| After the fit | `last_layer_laplace` | $`\mathbf{w} \sim \mathcal{N}(\hat{\mathbf{w}}, \boldsymbol{\Sigma}_w)`$ | a network you have already trained |

A random-feature layer with $`M`$ frequencies maps an input to

```math
\boldsymbol{\phi}(\mathbf{x}) = \frac{1}{\sqrt{M}} \left[ \cos(\boldsymbol{\Omega}^\top \mathbf{x} / \ell), \ \sin(\boldsymbol{\Omega}^\top \mathbf{x} / \ell) \right]
```

where $`\boldsymbol{\Omega} \in \mathbb{R}^{D_x \times M}`$ is drawn from the kernel's spectral density (a standard normal for the RBF kernel) and $`\ell`$ is the lengthscale. A linear readout $`f(\mathbf{x}) = \boldsymbol{\phi}(\mathbf{x})^\top \mathbf{w}`$ with $`\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})`$ is then a Gaussian process whose covariance approaches the kernel as $`M`$ grows. The Laplace posterior over a group of $`P`$ weights $`\mathbf{w}`$ keeps every other weight at its fitted value and has

```math
\boldsymbol{\Sigma}_w^{-1} = \frac{1}{\sigma_\varepsilon^2} \mathbf{J}^\top \mathbf{J} + \lambda \mathbf{I}
```

with $`\mathbf{J} \in \mathbb{R}^{N \times P}`$ the Jacobian of the network's outputs on the training inputs with respect to $`\mathbf{w}`$, $`\sigma_\varepsilon`$ the noise standard deviation and $`\lambda`$ the prior precision. Its predictive variance at a new input is $`\mathbf{J}_* \boldsymbol{\Sigma}_w \mathbf{J}_*^\top`$, and for a last layer that is linear in $`\mathbf{w}`$ it is exact Bayesian linear regression on the network's features.

## 📥 Install

We have not put pyrox-nn on PyPI, and the name `pyrox` there belongs to an unrelated project, so install it from GitHub with [uv](https://docs.astral.sh/uv/). It brings in the pyrox core from the same repository, geonnax, functax, kernellib, argminax, gaussx, lineax and einx, and needs Python 3.12 or newer.

```bash
uv add "pyrox-nn @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-nn"
```

kernellib v0.0.22 pins geonnax v0.0.8 and argminax v0.0.6 pins gaussx v0.6.13, while pyrox-nn needs geonnax v0.0.11 and gaussx v0.7.0, so add this override to your project's `pyproject.toml`; without it uv cannot resolve the install.

```toml
[tool.uv]
override-dependencies = [
    "gaussx @ git+https://github.com/jejjohnson/gaussx.git@v0.7.0",
    "geonnax @ git+https://github.com/jejjohnson/geonnax.git@v0.0.11",
]
```

Four extras add the rest: `optax` for `pyrox.inference`'s ensemble MAP path, `bnf` for `pyrox_nn.api` and `pyrox_nn.preprocessing` (pandas and optax), `xarray` for `fit_xarray` (the `bnf` dependencies plus xarray), and `colab` for what the example notebooks need (matplotlib, watermark and pandas). `import pyrox_nn` itself never imports pandas. The quickstart needs the `bnf` extra.

```bash
uv add "pyrox-nn[bnf] @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-nn"
```

## 🧠 Quickstart: one regression, three kinds of uncertainty

The data are 64 noisy points of $`y_n = \sin(3 x_n) + \varepsilon_n`$ on $`[-1, 1]`$, with $`\varepsilon_n \sim \mathcal{N}(0, \sigma_\varepsilon^2)`$ and $`\sigma_\varepsilon = 0.1`$, and three test inputs $`\mathbf{X}_* \in \mathbb{R}^{N_* \times D_x}`$: $`x = 0`$ and $`0.5`$ inside the data, and $`x = 2`$ outside it. We give the same regression its uncertainty three ways (in the weights, in random features, and after a plain fit) and compare what each says at $`\mathbf{X}_*`$. The setup is shared: the data, one NumPyro model that takes any network $`f`$ and records $`f(\mathbf{X})`$ as the site `f`, and `fit_and_predict`, which fits a mean-field normal guide $`q(\boldsymbol{\theta})`$ by SVI and returns $`S = 256`$ draws of $`f(\mathbf{X}_*)`$ under it.

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal


# The package's tests run in double precision, so the quickstart does too.
jax.config.update("jax_enable_x64", True)

# Shapes: N = 64 samples; D_x = 1 input; N_* = 3 test inputs; S = 256 draws

# X evenly spaced on [−1, 1], and y = sin(3 X) + ε
# → (N, D_x), (N,)
inputs = jnp.linspace(-1.0, 1.0, 64)[:, None]
targets = jnp.sin(3.0 * inputs[:, 0]) + 0.1 * jr.normal(jr.key(1), (64,))
# X_*, two inputs inside the data and one outside
# → (N_*, D_x)
test_inputs = jnp.array([[0.0], [0.5], [2.0]])


# user
# p(y | X, θ) = 𝒩(y; f(X; θ), σ_ε² I), with σ_ε = 0.1, and f(X; θ) kept as a site
def model(f, inputs: Float[Array, "N D_x"], targets: Array | None = None) -> None:
    mean = numpyro.deterministic("f", f(inputs))
    numpyro.sample("obs", dist.Normal(mean, 0.1).to_event(1), obs=targets)


# user
# q(θ) by SVI, then f(X_*; θ) for S draws θ ~ q(θ)
# → (S, N_*)
def fit_and_predict(f) -> Float[Array, "S N_*"]:
    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(1e-2), Trace_ELBO())
    fit = svi.run(jr.key(2), 3000, f, inputs, targets, progress_bar=False)
    predictive = Predictive(model, guide=guide, params=fit.params, num_samples=256)
    return predictive(jr.key(3), f, test_inputs)["f"]
```

**A. Uncertain weights.** Two `DenseReparameterization` layers make a Bayesian network $`f(\mathbf{x}; \boldsymbol{\theta}) = \tanh(\mathbf{x}^\top \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2`$ with $`H = 16`$ hidden units and a standard normal prior on $`\boldsymbol{\theta} = (\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2)`$. Each layer records its weights under its own `pyrox_name`, as the sites `hidden.weight`, `hidden.bias`, `output.weight` and `output.bias`.

```python
from pyrox_nn import DenseReparameterization


# Shapes: N samples; D_x inputs; H = 16 hidden units

# today (pyrox-nn)
# W₁ ~ 𝒩(0, I), b₁ ~ 𝒩(0, I), with W₁ ∈ ℝ^{D_x × H}
hidden = DenseReparameterization(1, 16, pyrox_name="hidden")
# today (pyrox-nn)
# W₂ ~ 𝒩(0, I), b₂ ~ 𝒩(0, I), with W₂ ∈ ℝ^{H × 1}
output = DenseReparameterization(16, 1, pyrox_name="output")


# user
# f(X; θ) = tanh(X W₁ + b₁) W₂ + b₂
# (N, D_x) → (N,)
def bnn(inputs: Float[Array, "N D_x"]) -> Float[Array, " N"]:
    return output(jnp.tanh(hidden(inputs)))[..., 0]


# user
# f(X_*; θ) for θ ~ q(θ)
# → (S, N_*)
bnn_draws = fit_and_predict(bnn)
```

**B. Uncertain features.** `RBFFourierFeatures` maps $`\mathbf{x}`$ to $`2M`$ random Fourier features $`\boldsymbol{\phi}(\mathbf{x})`$ of an RBF kernel, with $`M = 32`$ frequencies $`\boldsymbol{\Omega} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})`$ and a lengthscale $`\ell \sim \mathrm{LogNormal}(\log 0.5, 1)`$, and a Bayesian readout makes $`f(\mathbf{x}) = \boldsymbol{\phi}(\mathbf{x})^\top \mathbf{w} + b`$ a sparse-spectrum Gaussian process. SVI fits a posterior over the frequencies, the lengthscale and the readout together.

```python
from pyrox_nn import RBFFourierFeatures


# Shapes: N samples; D_x inputs; M = 32 frequencies, so 2M = 64 features

# today (pyrox-nn)
# φ(X) = [cos(X Ω / ℓ), sin(X Ω / ℓ)] / √M, sites RBFFourierFeatures.W and .lengthscale
features = RBFFourierFeatures.init(1, 32, lengthscale=0.5)
# today (pyrox-nn)
# w ~ 𝒩(0, I) and b ~ 𝒩(0, 1), the sites head.weight and head.bias
head = DenseReparameterization(64, 1, pyrox_name="head")


# user
# f(X) = φ(X) w + b
# (N, D_x) → (N,)
def ssgp(inputs: Float[Array, "N D_x"]) -> Float[Array, " N"]:
    return head(features(inputs))[..., 0]


# user
# f(X_*) for (Ω, ℓ, w, b) ~ q
# → (S, N_*)
ssgp_draws = fit_and_predict(ssgp)
```

**C. Uncertainty after the fit.** An ordinary `eqx.nn.MLP`, fitted by maximum likelihood with `ensemble_map` from the pyrox core (one member, the prior term zero), gets a Laplace posterior over its last layer, the $`P = 17`$ weights $`\mathbf{w}`$ of the readout. `param_filter` picks those weights by their key path, and `optimize_prior_precision` sets $`\lambda`$ by maximising the evidence.

```python
import equinox as eqx
from pyrox.inference import ensemble_map
from pyrox_nn import last_layer_laplace


# Shapes: N samples; D_x inputs; H = 16 hidden units; P = 17 last-layer weights


# user
# log p(y | X, θ), and a zero log prior, so the fit is maximum likelihood
def log_joint(mlp: eqx.nn.MLP, inputs: Array, targets: Array) -> tuple[Array, Array]:
    log_likelihood = dist.Normal(jax.vmap(mlp)(inputs), 0.1).log_prob(targets).sum()
    return log_likelihood, jnp.zeros(())


# user
# a fresh MLP from a key, f(x; θ) = W₂ tanh(W₁ x + b₁) + b₂
def init_fn(key: Array) -> eqx.nn.MLP:
    return eqx.nn.MLP(1, "scalar", 16, 1, activation=jnp.tanh, key=key)


# user
# w, the weights whose key path starts with the readout, .layers[1]
def is_last(path: tuple, _leaf: Array) -> bool:
    return jax.tree_util.keystr(path).startswith(".layers[1]")


# today (pyrox)
# θ̂ = argmax_θ log p(y | X, θ), one member under a leading axis of size 1
members, _ = ensemble_map(
    log_joint,
    init_fn,
    ensemble_size=1,
    num_epochs=2000,
    data=(inputs, targets),
    seed=jr.key(4),
    learning_rate=1e-2,
)
# θ̂ without the ensemble axis
mlp = jax.tree.map(lambda leaf: leaf[0] if eqx.is_array(leaf) else leaf, members)
# today (pyrox-nn)
# Σ_w⁻¹ = Jᵀ J / σ_ε² + λ I at ŵ, with λ set by maximising the evidence
posterior = last_layer_laplace(
    lambda net, x: jax.vmap(net)(x),
    mlp,
    inputs,
    targets,
    sigma=0.1,
    param_filter=is_last,
).optimize_prior_precision()
# today (pyrox-nn)
# 𝔼[f(X_*)] and Var[f(X_*)] = J_* Σ_w J_*ᵀ
# (N_*, D_x) → (N_*,), (N_*,)
laplace_mean, laplace_var = posterior.predict(test_inputs)
```

**Compare.** Each route gives a mean and a standard deviation of $`f(\mathbf{X}_*)`$.

```python
# Shapes: N_* test inputs (the draws and the Laplace moments as above)

routes = {
    "weights": (bnn_draws.mean(axis=0), bnn_draws.std(axis=0)),
    "features": (ssgp_draws.mean(axis=0), ssgp_draws.std(axis=0)),
    "laplace": (laplace_mean, jnp.sqrt(laplace_var)),
}
for name, (mean, std) in routes.items():
    print(f"{name:>8}", mean.round(2), std.round(2))
```

This prints

```
 weights [-0.06  0.91 -0.83] [0.08 0.09 0.14]
features [0.07 1.03 0.13] [0.08 0.1  0.25]
 laplace [ 0.03  0.99 -1.44] [0.02 0.03 0.25]
```

that is, for each route, the mean of $`f`$ at $`x = 0`$, 0.5 and 2 and its standard deviation. Inside the data all three come within 0.1 of $`\sin(0) = 0`$ and $`\sin(1.5) = 1.00`$. At $`x = 2`$ they part ways. The random-feature GP reverts towards its prior mean of zero, within two standard deviations of $`\sin(6) = -0.28`$. The Bayesian network and the Laplace posterior extrapolate the trend of the data to $`-0.83`$ and $`-1.44`$, and their standard deviations of 0.14 and 0.25 are far smaller than how far off they are, which is the known weakness of weight-space posteriors away from the data. The [kernel approximation](https://jejjohnson.github.io/pyrox/notebooks/kernel_approximation/) and [random features as networks](https://jejjohnson.github.io/pyrox/notebooks/rff_as_neural_networks/) tutorials go further with the feature layers.

## 🗺️ A neural field in one call

The quickstart wrote the model, the guide and the fit by hand. `NeuralFieldEstimator` in `pyrox_nn.api` does all three for any field, here a Bayesian SIREN $`f(\mathbf{x})`$ with sine activations, fitted as an ensemble of $`E = 8`$ MAP fits under the same Gaussian noise. `predict` returns the mean and the quantiles of the predictive mixture, noise included. The block continues from the quickstart and needs the `bnf` extra.

```python
from pyrox_nn import BayesianSIREN
from pyrox_nn.api import NeuralFieldEstimator


# Shapes: E = 8 members; N_* test inputs (the data as above)

# today (pyrox-nn)
# y ~ 𝒩(f(x), σ_ε²), with f a Bayesian SIREN of 32 units and depth 3, fitted E times
estimator = NeuralFieldEstimator(
    model_fn=lambda: BayesianSIREN.init(1, 32, 1, depth=3, first_omega=3.0),
    sigma_obs=0.1,
    ensemble_size=8,
    num_epochs=2000,
)
fitted = estimator.fit(inputs, targets, seed=0)
# today (pyrox-nn)
# the predictive mean and its 5 % and 95 % quantiles at X_*
# (N_*, D_x) → (N_*,), (N_*,), (N_*,)
mean, (lower, upper) = fitted.predict(test_inputs, quantiles=(0.05, 0.95))

print(mean.round(2), lower.round(2), upper.round(2))
```

This prints `[-0.02  0.95  0.01] [-0.28  0.75 -0.45] [0.2  1.14 0.41]`: the mean at $`x = 0`$, 0.5 and 2 and its 90 % interval. Inside the data the interval is about $`\pm 0.2`$, twice the noise, and at $`x = 2`$ it widens to $`[-0.45, 0.41]`$, which covers $`\sin(6) = -0.28`$. `first_omega=3.0` lowers the frequency of the SIREN's first layer from its default of 30, to suit a slow signal on $`[-1, 1]`$. The same estimator takes an `encoder` for longitude and latitude, per-row noise and a mask for padded rows, and `fit_xarray` fits a gridded `xarray.Dataset`; the [SIREN tutorial](https://jejjohnson.github.io/pyrox/notebooks/siren_inr/) calibrates the Bayesian SIREN's prior and fits an ensemble of MAP fits by hand.

## 🔬 What else is in the box

`pyrox_nn` holds the layers, and two namespaces behind the `bnf` extra hold the estimators and the preprocessing; each has its own `__all__`.

| Area | Symbols | Notes |
|---|---|---|
| Bayesian dense layers | `DenseReparameterization`, `DenseFlipout`, `DenseVariational`, `DenseHierarchical`, `DenseVariationalDropout`, `DenseDVI`, `DenseNCP`, `NCPNormalOutput` | Weight-space priors and guides for $`\mathbf{x}^\top \mathbf{W} + \mathbf{b}`$ |
| Random features and GP layers | `RBFFourierFeatures`, `RBFCosineFeatures`, `MaternFourierFeatures`, `MaternCosineFeatures`, `LaplaceFourierFeatures`, `LaplaceCosineFeatures`, `ArcCosineFourierFeatures`, `RandomKitchenSinks`, `VariationalFourierFeatures`, `HSGPFeatures`, `SphericalHarmonicFeatures`, `RandomFeatureGaussianProcess`, `DeepVSSGP` | Kernels as layers; `HSGPFeatures` and `SphericalHarmonicFeatures` take a kernel and use its spectral density |
| Neural fields | `BayesianSIREN`, `BayesianFourierNet`, `BayesianGaborNet`, `BayesianSlepianEncoder`, `BayesianNeuralField`, `FourierFeatures`, `SeasonalFeatures`, `InteractionFeatures`, `Standardization` | Coordinate networks; the last five are the layers of the Bayesian Neural Field |
| Conditioning | `BayesianAffineModulation`, `BayesianConcatConditioner`, `BayesianHyperLinear`, `ConditionedRFFNet`, `HyperFourierFeatures` | A field conditioned on a code, by FiLM, concatenation or a hypernetwork |
| Ensembles and heads | `DenseRank1`, `LayerNormEnsemble`, `MultiHeadAttentionBE`, `GaussianRegressionHead`, `MCSoftmaxDenseFA`, `MCSigmoidDenseFA` | Rank-1 and BatchEnsemble members; heteroscedastic regression and classification heads |
| After the fit | `last_layer_laplace`, `LastLayerLaplace` | The GGN Laplace posterior, its evidence and its prior-precision search |
| Feature functions | `fourier_features`, `seasonal_features`, `seasonal_frequencies`, `interaction_features`, `standardize`, `unstandardize` | Pure-JAX helpers the BNF layers wrap |
| From geonnax | `SIREN`, `SirenDense`, `FourierNet`, `GaborNet`, `FourierFilter`, `GaborFilter`, `mfn_forward`, `OrthogonalRandomFeatures`, `LaplaceRandomFeatureCovariance`, `BasisTransform`, `GeoFNO`, `ManifoldNO`, `LearnedDeformation`, `NCPContinuousPerturb`, … | Deterministic layers and encoders, re-exported unchanged |
| `pyrox_nn.api` | `NeuralFieldEstimator`, `FittedNeuralField`, `BNFEstimator`, `BNFEstimatorMLE`, `BNFEstimatorVI`, `FittedBNF`, `EstimatorBase`, `FittedEstimator` | Fit and predict on arrays, pandas tables or xarray (the `bnf` or `xarray` extra) |
| `pyrox_nn.preprocessing` | `fit_spatiotemporal`, `SpatiotemporalFit`, `encode_time_column`, `fit_standardization` | pandas tables to arrays for the Bayesian Neural Field (the `bnf` extra) |

The geonnax row lists 14 of the 33 deterministic names re-exported from geonnax; the others are 14 encoders and conditioners (`SphericalHarmonicEncoder`, `SlepianEncoder`, `HybridSphericalSlepianEncoder`, `Cartesian3DEncoder`, `CyclicEncoder`, `LonLatScale`, `Deg2Rad`, `AbstractConditioner`, `ConcatConditioner`, `AffineModulation`, `FiLM`, `HyperLinear`, `HyperSIREN`, `ConditionedINR`) and five functions (`cyclic_encode`, `deg2rad`, `lonlat_scale`, `lonlat_to_cartesian3d`, `spherical_harmonic_encode`).

## 📓 Tutorials

| # | Tutorial | What it shows |
|---|---|---|
| 1 | [SIREN](https://jejjohnson.github.io/pyrox/notebooks/siren_inr/) | `SIREN` against a ReLU network in 1-D and 2-D, then `BayesianSIREN` with a calibrated prior and an ensemble of MAP fits |
| 2 | [Kernel approximation](https://jejjohnson.github.io/pyrox/notebooks/kernel_approximation/) | the random Fourier and cosine feature layers against their exact kernels |
| 3 | [Spectral kernel models](https://jejjohnson.github.io/pyrox/notebooks/spectral_kernel_models/) | RBF, Matérn, Laplace and arc-cosine features as priors for Earth-system fields |
| 4 | [Random features as neural networks](https://jejjohnson.github.io/pyrox/notebooks/rff_as_neural_networks/) | random features fixed, learned and in an ensemble |
| 5 | [Deep random feature expansions](https://jejjohnson.github.io/pyrox/notebooks/deep_random_fourier_features/) | stacked spectral GPs with `DeepVSSGP` |
| 6 | [Conditional neural fields](https://jejjohnson.github.io/pyrox/notebooks/conditioning/) | FiLM, hypernetworks and conditioned random-feature networks |
| 7 | [Spatiotemporal preprocessing](https://jejjohnson.github.io/pyrox/notebooks/spatiotemporal_preprocessing/) | geographic encoders, seasonal features and `fit_spatiotemporal` on a pandas table |

## 📖 Documentation

- **[Documentation](https://jejjohnson.github.io/pyrox/)**: the whole workspace, with the tutorials and examples of every package.
- **[NN API](https://jejjohnson.github.io/pyrox/api/nn/)**: every layer, generated from the docstrings, with the pages on [geo encoders](https://jejjohnson.github.io/pyrox/api/nn/geo_encoders/), [conditioning](https://jejjohnson.github.io/pyrox/api/nn/conditioning/) and [multiplicative filter networks](https://jejjohnson.github.io/pyrox/api/nn/mfn/).
- **[Estimator API](https://jejjohnson.github.io/pyrox/api/api/)** and **[Preprocessing API](https://jejjohnson.github.io/pyrox/api/preprocessing/)**: `pyrox_nn.api` and `pyrox_nn.preprocessing`.
- **[Capability index](https://jejjohnson.github.io/pyrox/capabilities/)**: every public name in the workspace and in the libraries it builds on, to check before writing something new.
- **[`AGENTS.md`](AGENTS.md)**: the contracts every layer keeps (the two shapes of layer, static fields, no pyrox-gp import) and the tests that hold them.

## 🧩 Ecosystem

pyrox-nn is a package of the pyrox [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/), in the GeoML tier of the [GeoStack](https://github.com/jejjohnson/geostack). It owns the probabilistic wrappers, and composes the rest in:

| Concern | Owner |
|---|---|
| Sites, the per-call cache and the ensemble MAP and VI fits | [`pyrox`](../pyrox) (`_core` and `inference`) |
| Deterministic network cores and geographic encoders | [`geonnax`](https://github.com/jejjohnson/geonnax) |
| Fourier, spherical-harmonic and Slepian bases, and seasonal features | [`functax`](https://github.com/jejjohnson/functax) |
| Kernels, spectral densities and Funk–Hecke coefficients | [`kernellib`](https://github.com/jejjohnson/kernellib) |
| The Laplace curvature, the evidence and the prior-precision search | [`argminax`](https://github.com/jejjohnson/argminax) |
| Linear algebra on PSD operators, and the GGN diagonal | [`gaussx`](https://github.com/jejjohnson/gaussx) and [`lineax`](https://github.com/patrick-kidger/lineax) |
| Sample sites, NUTS, SVI, autoguides and `Predictive` | [`numpyro`](https://github.com/pyro-ppl/numpyro) |
| Modules as pytrees, and `eqx.tree_at` | [`equinox`](https://github.com/patrick-kidger/equinox) |
| Optimisers for the estimator's fits (the `optax` and `bnf` extras) | [`optax`](https://github.com/google-deepmind/optax) |
| Tables and grids (the `bnf` and `xarray` extras) | [`pandas`](https://pandas.pydata.org/) and [`xarray`](https://xarray.dev/) |

Dependencies point one way. pyrox-nn imports the pyrox core and never pyrox-gp, its sibling over the core; no other package of the workspace imports pyrox-nn, and nothing outside it does either. The core's `pyrox.inference` still resolves `LastLayerLaplace` and `last_layer_laplace`, which moved here, with a `DeprecationWarning`, and looks them up only when asked, so the core does not depend on pyrox-nn.

## 🌳 Repository layout

```
packages/pyrox-nn/
├── src/pyrox_nn/
│   ├── __init__.py        ← the layers, last_layer_laplace and the geonnax re-exports
│   ├── _dense.py          ← Bayesian dense layers
│   ├── _features.py       ← random features, HSGP and spherical-harmonic layers
│   ├── _sngp.py, _vssgp.py ← the SNGP head and deep VSSGP
│   ├── _siren.py, _mfn.py, _bnf.py ← neural fields
│   ├── _laplace.py        ← the post-hoc last-layer Laplace posterior
│   ├── api/               ← NeuralFieldEstimator and the BNF estimators (bnf extra)
│   └── preprocessing/     ← pandas preprocessing (bnf extra)
├── tests/                 ← site sets under trace, SVI and Predictive runs, the import guard
└── AGENTS.md              ← the contracts and the tests that hold them
```

## ⚙️ Development

The package is developed inside the workspace, so clone the repository and run the tasks from its root.

```bash
git clone https://github.com/jejjohnson/pyrox.git
cd pyrox
make install                                                     # uv sync --all-groups + pre-commit hooks
uv run pytest --no-cov packages/pyrox-nn/tests -m "not slow"     # this package's fast tests
make lint                                                        # ruff check on the whole repo
make typecheck                                                   # ty check on every package
```

`make help` lists every task. The root [`AGENTS.md`](../../AGENTS.md) holds the rules for the whole workspace, and this package's [`AGENTS.md`](AGENTS.md) adds the contracts of the layers.

## 🎓 References

- Hafner, D., Tran, D., Lillicrap, T., Irpan, A., & Davidson, J. (2018). [Noise contrastive priors for functional uncertainty](https://arxiv.org/abs/1807.09289). *UAI*.
- Lee, J., et al. (2018). [Deep neural networks as Gaussian processes](https://arxiv.org/abs/1711.00165). *ICLR*.
- Wu, A., Nowozin, S., Meeds, E., Turner, R. E., Hernández-Lobato, J. M., & Gaunt, A. L. (2018). [Deterministic variational inference for robust Bayesian neural networks](https://arxiv.org/abs/1810.03958). *ICLR*.
- Louizos, C., Ullrich, K., & Welling, M. (2017). [Bayesian compression for deep learning](https://arxiv.org/abs/1705.08665). *NeurIPS*.
- Liu, J. Z., et al. (2020). [Simple and principled uncertainty estimation with deterministic deep learning via distance awareness](https://arxiv.org/abs/2006.10108). *NeurIPS*.
- Wen, Y., Tran, D., & Ba, J. (2020). [BatchEnsemble: an alternative approach to efficient ensemble and lifelong learning](https://arxiv.org/abs/2002.06715). *ICLR*.
- Dusenberry, M. W., et al. (2020). [Efficient and scalable Bayesian neural nets with rank-1 factors](https://arxiv.org/abs/2005.07186). *ICML*.
- Nix, D. A., & Weigend, A. S. (1994). [Estimating the mean and variance of the target probability distribution](https://doi.org/10.1109/ICNN.1994.374138). *ICNN*.
- Collier, M., Mustafa, B., Kokiopoulou, E., Jenatton, R., & Berent, J. (2021). [Correlated input-dependent label noise in large-scale image classification](https://arxiv.org/abs/2105.10305). *CVPR*.
- Saad, F. A., et al. (2024). [Scalable spatiotemporal prediction with Bayesian neural fields](https://doi.org/10.1038/s41467-024-51477-5). *Nature Communications* 15, 7942.

## 📄 License

MIT. See [LICENSE](LICENSE).
