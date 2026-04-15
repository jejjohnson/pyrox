---
status: draft
version: 0.1.0
---

# Models -- Composition Patterns

Layer 2 entry points: how the core abstractions (Layer 0) and probabilistic layers (Layer 1) compose into full probabilistic models for NumPyro inference.

pyrox does **not** provide model classes. The user writes a NumPyro model function; pyrox provides the building blocks. This document catalogs the three canonical patterns.

---

## Pattern A: PyroxModule Ownership

The module owns its probabilistic semantics. Parameters and samples are declared inside `__call__` via `pyrox_sample` / `pyrox_param`. The model function simply calls the module.

```python
from pyrox._core import PyroxModule
from pyrox.nn import DenseVariational

class BayesianMLP(PyroxModule):
    layer1: DenseVariational
    layer2: DenseVariational

    def __init__(self, *, key):
        k1, k2 = jr.split(key)
        self.layer1 = DenseVariational(output_dim=50, prior_std=1.0, key=k1)
        self.layer2 = DenseVariational(output_dim=1, prior_std=1.0, key=k2)

    def __call__(self, x):
        h = jax.nn.tanh(self.layer1(x))
        return self.layer2(h)


def model(x, y=None):
    net = BayesianMLP(key=jr.PRNGKey(0))
    f = net(x)
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    numpyro.sample("obs", dist.Normal(f.squeeze(), sigma), obs=y)
```

**When to use:** When the module is inherently probabilistic (all its uncertainty is in its own weights).

**Inference:** Works with `MCMC(NUTS(model))`, `SVI(model, AutoNormal(model), ...)`, `Predictive(model, ...)`.

---

## Pattern B: Pure Equinox + `eqx.tree_at`

Architecture and probabilistic model are separate. The module is a plain `eqx.Module`; the model function samples parameters from priors and injects them via `eqx.tree_at`.

```python
class MLP(eqx.Module):
    W1: jax.Array
    b1: jax.Array
    W2: jax.Array
    b2: jax.Array

    def __init__(self, *, key):
        k1, k2 = jr.split(key)
        self.W1 = jr.normal(k1, (1, 50))
        self.b1 = jnp.zeros(50)
        self.W2 = jr.normal(k2, (50, 1))
        self.b2 = jnp.zeros(1)

    def __call__(self, x):
        h = jax.nn.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


def model(x, y=None):
    net = MLP(key=jr.PRNGKey(0))
    W1 = numpyro.sample("W1", dist.Normal(jnp.zeros((1, 50)), 1.0))
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(50), 1.0))
    net = eqx.tree_at(lambda m: (m.W1, m.b1), net, (W1, b1))
    f = numpyro.deterministic("f", net(x))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    numpyro.sample("obs", dist.Normal(f.squeeze(), sigma), obs=y)
```

**When to use:** When you want to keep existing Equinox modules unchanged and add priors externally. Good for selective Bayesian treatment (priors on some weights, not all).

**Inference:** Same as Pattern A -- standard NumPyro inference.

---

## Pattern C: Parameterized (GP-Style)

For modules where parameters need priors, constraint transforms, variational guides, and model/guide mode switching. Essential for GP kernels and any module with constrained parameters.

Note: kernel implementations now live in `pyrox.gp`, but the `Parameterized` base class comes from `pyrox._core`.

```python
from pyrox._core import Parameterized

class RBFKernel(Parameterized):
    def setup(self):
        self.register_param("variance", jnp.array(1.0), constraint=positive)
        self.register_param("lengthscale", jnp.array(1.0), constraint=positive)
        self.set_prior("variance", dist.LogNormal(0.0, 1.0))
        self.set_prior("lengthscale", dist.LogNormal(0.0, 1.0))

    def __call__(self, X1, X2):
        v = self.get_param("variance")
        l = self.get_param("lengthscale")
        sq_dist = jnp.sum((X1[:, None] - X2[None, :]) ** 2 / l**2, axis=-1)
        return v * jnp.exp(-0.5 * sq_dist)


def model(X, y=None):
    kernel = RBFKernel()
    kernel.set_mode("model")
    kernel.load_pyro_samples()
    K = kernel(X, X) + 1e-4 * jnp.eye(X.shape[0])
    numpyro.sample("y", dist.MultivariateNormal(jnp.zeros(X.shape[0]), K), obs=y)
```

**When to use:** GP kernels, mean functions, or any module with constrained parameters that need priors and variational posteriors.

**Inference:** SVI with `autoguide()` or custom guides. Also works with MCMC.

---

## Inference Methods

All three patterns work with NumPyro's inference machinery. No pyrox-specific inference code.

| Method | NumPyro API | Posterior | Speed |
|---|---|---|---|
| MAP | `SVI(model, AutoDelta(model), ...)` | Point estimate | Fast |
| Mean-field VI | `SVI(model, AutoNormal(model), ...)` | Diagonal Gaussian | Fast |
| Full-rank VI | `SVI(model, AutoMultivariateNormal(model), ...)` | Full Gaussian | Medium |
| MCMC (NUTS) | `MCMC(NUTS(model), ...)` | Exact (asymptotic) | Slow |
| Predictive | `Predictive(model, posterior_samples=..., num_samples=...)` | Forward samples | -- |

---

## Model Progression

The 9-model regression masterclass demonstrates how the patterns compose into increasingly sophisticated models:

| # | Model | Pattern | Layers Used | Uncertainty |
|---|---|---|---|---|
| 1 | Linear regression | B | -- | None (MAP) |
| 2 | Bayesian linear | B | -- | Weight posterior |
| 3 | Neural network | B | -- | None (MAP) |
| 4 | MC-Dropout NN | A | `MCDropout` | Stochastic dropout |
| 5 | Bayesian NN | A | `DenseVariational` | Weight posterior |
| 6 | NCP network | A | `DenseNCP`, `NCPContinuousPerturb` | Noise contrastive prior |
| 7 | SVR (RFF) | B | `RandomFourierFeatures` | None (MAP) |
| 8 | Approx GP (RFF) | B | `RandomFourierFeatures` | Weight posterior |
| 9 | Deep GP (RFF) | B | `RandomFourierFeatures` (stacked) | Weight posterior |

See [regression_masterclass_eqx.md](regression_masterclass_eqx.md) for the full tutorial.

---

## SSGP (Sparse Spectrum GP Regression)

Bayesian linear regression in random Fourier feature space. The GP is
approximated by D features: f(x) = phi(x)^T w, w ~ N(0, I).

```python
from pyrox.nn import RandomFourierFeatures

def ssgp_model(X, y=None):
    rff = RandomFourierFeatures(n_features=256, kernel_type="rbf", key=key)
    phi = rff(X)  # (N, 512)

    w = numpyro.sample("w", dist.Normal(jnp.zeros(512), 1.0))
    f = phi @ w

    noise = numpyro.sample("noise", dist.HalfNormal(1.0))
    numpyro.sample("y", dist.Normal(f, noise), obs=y)
```

Training: O(D^2 N). Prediction: O(D^2) per point.

---

## SNGP (Spectral Normalized GP)

Spectrally normalized feature extractor + RFF GP head for distance-aware
uncertainty.

```python
from pyrox.nn import RandomFourierFeatures

def sngp_model(x, y=None):
    # Distance-preserving feature extractor
    h = SpectralNormalization(eqx.nn.Linear(D_in, 256), coeff=0.95)(x)
    h = jax.nn.relu(h)
    h = SpectralNormalization(eqx.nn.Linear(256, 256), coeff=0.95)(h)
    h = jax.nn.relu(h)

    # RFF + Bayesian last layer
    phi = RandomFourierFeatures(n_features=128, key=key)(h)
    w = numpyro.sample("w", dist.Normal(jnp.zeros(256), 1.0))
    logits = phi @ w

    numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)
```

---

*For Layer 0 core abstractions, see [core.md](core.md). For Layer 1 layers, see [nn.md](nn.md). For GP-specific models, see [gp_models.md](gp_models.md).*
