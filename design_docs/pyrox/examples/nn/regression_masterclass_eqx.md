---
status: stable
version: 0.1.0
---

# A Regression Masterclass with Equinox + NumPyro + einops

*A pedagogical reimplementation of the
[Aboleth Regression Master Class](https://aboleth.readthedocs.io/en/stable/tutorials/some_regressors.html)
using Equinox for architecture, NumPyro for probabilistic inference,
and einops for all tensor contractions.*

---

## Prerequisites and Background

This guide assumes familiarity with:
- Basic Python and NumPy array programming
- The idea of regression (fitting a function to data)
- Elementary probability (what a prior and likelihood are)

No prior experience with JAX, Equinox, NumPyro, or Bayesian deep learning is required.
Each concept is introduced when it first appears.

---

## Why Equinox + NumPyro + einops?

The original Aboleth tutorial uses TensorFlow to compose layers into
computation graphs.  We replace this with three libraries that have clean,
complementary roles:

**Equinox** defines the *deterministic architecture* — modules with
`__call__` methods, immutable PyTrees, shapes, activations, feature maps.

**NumPyro** defines the *probabilistic semantics* — priors over parameters,
likelihoods, and inference (MCMC, SVI, Predictive).

**einops** replaces every `@` operator and manual broadcasting with
explicit `einsum("n feat, feat -> n", Phi, w)` patterns.  You can read
every contraction's dimension names at a glance instead of mentally
tracking shapes through `@` chains.

### The Three-Library Architecture in Detail

The key insight behind this design is **separation of concerns**. Each library
handles exactly one responsibility, and they compose cleanly:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        NumPyro Model Function                        │
│                                                                      │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────┐ │
│  │   NumPyro    │    │    Equinox       │    │      einops          │ │
│  │              │    │                  │    │                      │ │
│  │ • Priors     │───>│ • Module defs    │───>│ • Matrix multiplies  │ │
│  │ • Likelihoods│    │ • PyTree structs │    │ • Outer products     │ │
│  │ • Inference  │    │ • Forward pass   │    │ • Contractions       │ │
│  │   (MCMC/SVI) │    │ • tree_at inject │    │ • Named axes         │ │
│  └─────────────┘    └─────────────────┘    └──────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

**Why not just use NumPyro alone?** You *could* write all the forward-pass
logic inline inside the NumPyro model function. But as models grow, you want
reusable, testable modules. Equinox gives you standard Python classes that
JAX can trace, differentiate, and compile — without any special "variable"
abstractions. An Equinox module is just a PyTree (a nested structure of arrays
and static metadata) with a `__call__` method.

**Why not just use `@` for matrix math?** The `@` operator works fine for
simple 2D matrix multiplications, but it hides which axes are contracted and
which survive. When you have batched inputs, singleton dimensions, and outer
products, `@` requires `None`-indexing, `.squeeze()`, and mental shape tracking.
`einsum` makes every operation a self-documenting specification.

**Why not use Flax or Haiku?** Equinox modules are plain Python classes that
are also valid JAX PyTrees. There is no `apply()` indirection, no separate
parameter dictionary, and no `init()` function. This makes them trivially
compatible with NumPyro's `sample`/`deterministic` API — you just swap leaves
in the PyTree via `eqx.tree_at`.

### The `@` to `einsum` translation

| Before (numpy / JAX)  | After (einops)  | What it does |
|---|---|---|
| `Phi @ w` | `einsum(Phi, w, "n feat, feat -> n")` | features x weights -> predictions |
| `x[:, None] @ W1` | `einsum(x, W1, "n, one h -> n h")` | input x hidden weights -> pre-activations |
| `h @ W2` then `.squeeze(-1)` | `einsum(h, W2, "n h, h one -> n")` | hidden x output weights -> scalar output |
| `x[:, None] * omega[None, :]` | `einsum(x, omega, "n, d -> n d")` | outer product for RFF projection |

The einops version makes three things explicit that `@` hides: which axes
are being contracted, which are being broadcast, and what the output shape is.

---

## The `eqx.tree_at` Pattern: Bridging Deterministic and Probabilistic Worlds

This pattern is central to every model in the tutorial and deserves a thorough
explanation up front.

### The Problem

Equinox modules are **frozen PyTrees**. In JAX, a PyTree is a nested tree
structure of arrays (leaves) and containers (nodes). Equinox modules use
Python dataclass-style attributes as their tree structure. Crucially, they
are **immutable** — you cannot do `net.weight = new_value` because Equinox
modules are frozen after construction.

But in a Bayesian model, we need to:
1. Define the module architecture (shapes, activations, structure)
2. Sample parameter values from priors using NumPyro
3. Plug those sampled values into the module to run the forward pass

### The Solution: `eqx.tree_at`

`eqx.tree_at` creates a **new** PyTree that is identical to the original
except at specified leaves. It takes three arguments:

```python
new_tree = eqx.tree_at(
    where,        # a function that selects which leaf/leaves to replace
    pytree,       # the original PyTree
    replace,      # the new value(s) to put at those positions
)
```

The `where` argument is a callable that takes the PyTree and returns the
leaf (or tuple of leaves) to be replaced. Equinox uses this function
internally to figure out *which position* in the tree structure you mean,
then constructs a fresh copy with the new values at those positions.

### Single-Leaf Replacement

```python
# net.weight is a placeholder array of zeros
net = LinearRegressor(degree=3)

# w is sampled from a prior — a fresh JAX array
w = numpyro.sample("w", dist.Normal(jnp.zeros(4), 1.0))

# Create a new module identical to net, but with net.weight replaced by w
net = eqx.tree_at(lambda m: m.weight, net, w)
```

This is equivalent to the (impossible) `net.weight = w`, but respects
immutability. The lambda `lambda m: m.weight` tells Equinox *where* in the
tree to make the replacement.

### Multi-Leaf Replacement

For modules with multiple parameter arrays, return a **tuple** from the
selector function and pass a corresponding tuple of replacements:

```python
net = eqx.tree_at(
    lambda m: (m.W1, m.b1, m.W2, m.b2),   # select four leaves
    net,                                      # original module
    (W1, b1, W2, b2),                        # four replacement values
)
```

This atomically replaces all four parameter arrays in one call, producing
a new module ready for the forward pass.

### Why Not Just Use a Plain Function?

You might wonder: why not skip the module entirely and write:

```python
def forward(x, w):
    return einsum(features(x), w, "n feat, feat -> n")
```

You *can*, and for simple models it works fine. But modules give you:
- **Encapsulation**: the feature-construction logic lives with the weights
- **Composability**: modules can contain sub-modules (see `DeepRFFRegressor`)
- **Testability**: you can unit-test a module's forward pass independently
- **Reuse**: the same `MLP` class works for MAP, MCMC, and dropout models

---

## Inference Methods: MAP, MCMC, and SVI

This tutorial uses three inference strategies. Here is what each one does,
how it works, and when to choose it.

### MAP (Maximum A Posteriori)

**What it does:** Finds the single most probable parameter setting given the
data and priors. Formally, it solves:

θ̂_MAP = argmax_θ  p(θ | 𝒟) = argmax_θ  [log p(𝒟 | θ) + log p(θ)]

**How it works in NumPyro:** We use SVI (stochastic variational inference)
with an `AutoDelta` guide — a variational family that places all its mass
on a single point. Optimizing the ELBO with a delta guide is equivalent
to maximizing the log-joint, i.e., MAP estimation.

**When to use it:**
- When you want a single best-fit prediction (no uncertainty)
- As a fast sanity check before running full Bayesian inference
- When the model has many parameters and MCMC is too slow

**Limitations:** Provides no uncertainty quantification. The "posterior" is a
single point, so you cannot compute credible intervals or predictive
distributions that reflect parameter uncertainty.

### MCMC (Markov Chain Monte Carlo)

**What it does:** Draws samples θ⁽¹⁾, θ⁽²⁾, …, θ⁽ˢ⁾
from the exact posterior p(θ | 𝒟). Each sample is a full
set of model parameters. Predictions are made by averaging over these samples:

p(y* | x*, 𝒟) ≈ 1/S ∑_{s=1}^S p(y* | x*, θ⁽ˢ⁾)

**How it works in NumPyro:** We use the NUTS (No-U-Turn Sampler) algorithm
(Hoffman & Gelman, 2014), which is an adaptive variant of Hamiltonian Monte
Carlo. NUTS uses gradient information to explore the posterior efficiently
and automatically tunes its step size and trajectory length.

**When to use it:**
- When you need faithful uncertainty estimates
- When the model has a moderate number of parameters (up to ~thousands)
- When you need the gold-standard posterior (no approximation)

**Limitations:** Scales poorly to very high-dimensional parameter spaces
(e.g., large neural networks with millions of weights). Each sample
requires many gradient evaluations.

### SVI (Stochastic Variational Inference)

**What it does:** Approximates the posterior with a simpler distribution
q_φ(θ) (the "guide") by minimizing the KL divergence, or
equivalently maximizing the Evidence Lower Bound (ELBO):

ELBO(φ) = 𝔼_{q_φ(θ)}[log p(𝒟, θ) - log q_φ(θ)]

**How it works in NumPyro:** You choose a guide (variational family) and
optimize its parameters φ using stochastic gradient descent on the ELBO.
Common guides include:
- `AutoDelta`: point mass (equivalent to MAP)
- `AutoNormal`: independent Gaussian per parameter (mean-field)
- `AutoMultivariateNormal`: full-covariance Gaussian

**When to use it:**
- When the model is too large for MCMC
- When you want approximate uncertainty (with `AutoNormal` or similar)
- When you need fast iteration during model development

**Limitations:** The quality of the approximation depends on the guide family.
A mean-field guide cannot capture posterior correlations. The ELBO can have
local optima.

---

## Setup

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from einops import einsum
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, autoguide

import regression_masterclass_eqx as rm
```

## The Dataset

Same as the Aboleth tutorial: f(x) = sin(x)/x with Gaussian noise,
100 training points in [-10, 10], evaluated on a grid over [-20, 20].

The generative process for the data is:

x_i ~ Uniform(-10, 10), i = 1, …, 100

y_i = sin(x_i)/x_i + ε_i, ε_i ~ 𝒩(0, 0.05²)

The sinc function is a useful test case because it is nonlinear,
oscillatory, and decays to zero — properties that challenge simple linear
models but are well-captured by kernel methods and neural networks.

```python
key = jr.PRNGKey(666)
k_data, k_infer, k_pred = jr.split(key, 3)

data = rm.make_dataset(k_data, n_train=100, noise_std=0.05)
```

---

## Model 1: Bayesian Linear Regression

### Mathematical Formulation

**Generative model:**

w ~ 𝒩(0_{d+1}, I_{d+1})

σ ~ HalfNormal(1)

f(x) = Φ(x) w, where Φ(x)_i = [1, x_i, x_i², …, x_i^d]

y_i ~ 𝒩(f(x_i), σ²)

Here d is the polynomial degree and Φ ∈ ℝ^{N × (d+1)}
is the Vandermonde (polynomial feature) matrix.

**Forward pass equation:**

ŷ = Φ w  ⟺  ŷ_i = ∑_{k=0}^{d} x_i^k w_k

### The Equinox Module

```python
class LinearRegressor(eqx.Module):
    weight: jax.Array              # (degree+1,)
    degree: int = eqx.field(static=True)

    def __init__(self, degree: int = 1):
        self.degree = degree
        self.weight = jnp.zeros(degree + 1)

    def features(self, x):
        """Polynomial features  [1, x, x^2, ..., x^d]."""
        # x: (N,) -> output: (N, degree+1)
        return jnp.stack([x ** p for p in range(self.degree + 1)], axis=-1)

    def __call__(self, x):
        # x: (N,)
        # self.features(x): (N, degree+1)
        # self.weight: (degree+1,)
        # einsum: contract over the feature axis -> output: (N,)
        return einsum(self.features(x), self.weight, "n feat, feat -> n")
```

Compare with the `@` version: `self.features(x) @ self.weight`.  The einsum
spells out that `feat` is the contracted axis, `n` (samples) survives.

### The NumPyro Model

```python
def model_linear(x, y=None, *, degree=1):
    # x: (N,), y: (N,) or None
    net = rm.LinearRegressor(degree=degree)

    # Prior: weight vector w ~ N(0, I), shape (degree+1,)
    w = numpyro.sample("w", dist.Normal(jnp.zeros(degree + 1), 1.0))

    # eqx.tree_at: replace placeholder weight with the NumPyro sample
    # net.weight was jnp.zeros(degree+1); now it becomes the sampled w
    net = eqx.tree_at(lambda m: m.weight, net, w)

    # Forward pass: f shape (N,)
    f = numpyro.deterministic("f", net(x))

    # Likelihood
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    numpyro.sample("obs", dist.Normal(f, sigma), obs=y)
```

### Inference and Prediction

```python
mcmc = rm.run_mcmc(rm.model_linear, k_infer,
                   data["x_train"], data["y_train"],
                   num_warmup=1_000, num_samples=2_000, degree=1)
samples = mcmc.get_samples()

preds = rm.predict(rm.model_linear, k_pred, samples, data["x_test"], degree=1)
summary = rm.summarise_predictions(preds, data["y_test"])
print(f"R^2 = {summary['r2']:.4f}")   # expect ~0  (line can't fit sinc)
```

A degree-1 polynomial is a straight line, so it cannot capture the oscillations
of sinc(x). The near-zero R² confirms this. Increasing the degree
helps, but polynomial regression becomes numerically unstable for high degrees
and generalizes poorly outside the training range.

---

## Model 2: Neural Network (MAP)

### Mathematical Formulation

**Generative model (single hidden layer MLP):**

W_1 ~ 𝒩(0, I) ∈ ℝ^{1 × H}, b_1 ~ 𝒩(0, I) ∈ ℝ^H

W_2 ~ 𝒩(0, I) ∈ ℝ^{H × 1}, b_2 ~ 𝒩(0, 1) ∈ ℝ

σ ~ HalfNormal(1)

h_i = tanh(x_i W_1 + b_1) ∈ ℝ^H

f(x_i) = h_i W_2 + b_2 ∈ ℝ

y_i ~ 𝒩(f(x_i), σ²)

Here H is the hidden dimension. The input x_i is a scalar broadcast
across the hidden units via W_1 ∈ ℝ^{1 × H}.

### The Equinox Module

```python
class MLP(eqx.Module):
    W1: jax.Array    # (1, hidden_dim)
    b1: jax.Array    # (hidden_dim,)
    W2: jax.Array    # (hidden_dim, 1)
    b2: jax.Array    # ()
    hidden_dim: int = eqx.field(static=True)

    def __call__(self, x):
        # x: (N,)
        # Layer 1:  x (n,) x W1 (one, h) -> pre-activation (n, h)
        h = jnp.tanh(
            einsum(x, self.W1, "n, one h -> n h") + self.b1  # (N, H) + (H,) -> (N, H)
        )
        # Layer 2:  h (n, h) x W2 (h, one) -> output (n,)
        return einsum(h, self.W2, "n h, h one -> n") + self.b2  # (N,) + () -> (N,)
```

The einsum patterns make the MLP's two matrix multiplications self-documenting:
`"n, one h -> n h"` says "broadcast the scalar input across the hidden dim",
and `"n h, h one -> n"` says "contract hidden, collapse the singleton output dim".

**Shape walkthrough for N=100, H=50:**

| Expression | Shape | Notes |
|---|---|---|
| `x` | `(100,)` | Raw input |
| `self.W1` | `(1, 50)` | First-layer weight |
| `einsum(x, W1, "n, one h -> n h")` | `(100, 50)` | Broadcast x across hidden |
| `+ self.b1` | `(100, 50)` | Bias broadcast over N |
| `jnp.tanh(...)` | `(100, 50)` | Hidden activations |
| `self.W2` | `(50, 1)` | Second-layer weight |
| `einsum(h, W2, "n h, h one -> n")` | `(100,)` | Contract H, collapse 1 |
| `+ self.b2` | `(100,)` | Scalar bias broadcast |

### The NumPyro Model

```python
def model_nnet(x, y=None, *, hidden_dim=50):
    # x: (N,), y: (N,) or None
    net = rm.MLP(hidden_dim=hidden_dim, key=jr.PRNGKey(0))

    # Priors over all weight matrices and bias vectors
    W1 = numpyro.sample("W1", dist.Normal(jnp.zeros((1, hidden_dim)), 1.0))   # (1, H)
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(hidden_dim), 1.0))        # (H,)
    W2 = numpyro.sample("W2", dist.Normal(jnp.zeros((hidden_dim, 1)), 1.0))   # (H, 1)
    b2 = numpyro.sample("b2", dist.Normal(0.0, 1.0))                          # ()

    # Tuple selector: replace all four weight leaves at once
    net = eqx.tree_at(lambda m: (m.W1, m.b1, m.W2, m.b2), net, (W1, b1, W2, b2))

    f = numpyro.deterministic("f", net(x))   # (N,)
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    numpyro.sample("obs", dist.Normal(f, sigma), obs=y)
```

For MAP estimation, use SVI with an `AutoDelta` guide:

```python
guide = autoguide.AutoDelta(rm.model_nnet)
svi_result = rm.run_svi(rm.model_nnet, guide, k_infer,
                        data["x_train"], data["y_train"],
                        num_steps=5_000, hidden_dim=50)
```

`AutoDelta` places a learnable point mass on each latent variable. Maximizing
the ELBO with delta distributions is equivalent to maximizing the log-joint
log p(y | θ) + log p(θ), which is MAP estimation. This is
fast but provides no uncertainty — the "posterior" is a single point.

---

## Model 3: MC-Dropout Neural Network

### Mathematical Formulation

MC-Dropout (Gal & Ghahramani, 2016) reinterprets dropout as approximate
Bayesian inference. At both train and test time, each hidden unit is
randomly masked:

z_ij ~ Bernoulli(1 - p_drop), i = 1, …, N, j = 1, …, H

h_i = tanh(x_i W_1 + b_1) ∈ ℝ^H

h̃_i = (h_i ⊙ z_i) / (1 - p_drop) ∈ ℝ^H  (inverted dropout)

f(x_i) = h̃_i W_2 + b_2

The division by (1 - p_drop) is "inverted dropout": it rescales
the surviving activations so the expected value of h̃_i equals h_i.
This means the same weights work at train time (with dropout) and test time
(without dropout) without any rescaling.

By running multiple forward passes with different dropout masks and collecting
the predictions, we obtain an approximate predictive distribution. In our
framework, the dropout mask is an explicit `numpyro.sample` site, so
`Predictive` automatically draws fresh masks for each posterior sample.

### The Equinox Module

The dropout mask is an **explicit argument** — not managed internally.
This lets NumPyro sample it as a `numpyro.sample` site.

```python
class MLPDropout(eqx.Module):
    # ... same weights as MLP ...
    dropout_rate: float = eqx.field(static=True)

    def __call__(self, x, mask):
        """mask: (N, hidden_dim) — explicit binary dropout mask."""
        # x: (N,), mask: (N, H)
        h = jnp.tanh(
            einsum(x, self.W1, "n, one h -> n h") + self.b1   # (N, H)
        )
        h = h * mask / (1.0 - self.dropout_rate)   # inverted dropout, (N, H)
        return einsum(h, self.W2, "n h, h one -> n") + self.b2   # (N,)
```

### The NumPyro Model

```python
def model_nnet_dropout(x, y=None, *, hidden_dim=50, dropout_rate=0.1):
    N = x.shape[0]
    # x: (N,), y: (N,) or None
    net = rm.MLPDropout(hidden_dim=hidden_dim, dropout_rate=dropout_rate, key=jr.PRNGKey(0))

    W1 = numpyro.sample("W1", dist.Normal(jnp.zeros((1, hidden_dim)), 1.0))   # (1, H)
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(hidden_dim), 1.0))        # (H,)
    W2 = numpyro.sample("W2", dist.Normal(jnp.zeros((hidden_dim, 1)), 1.0))   # (H, 1)
    b2 = numpyro.sample("b2", dist.Normal(0.0, 1.0))                          # ()
    net = eqx.tree_at(lambda m: (m.W1, m.b1, m.W2, m.b2), net, (W1, b1, W2, b2))

    # The dropout mask is a RANDOM VARIABLE — Predictive draws fresh masks
    # mask: (N, H) of 0s and 1s
    mask = numpyro.sample(
        "dropout_mask",
        dist.Bernoulli(probs=(1.0 - dropout_rate) * jnp.ones((N, hidden_dim))),
    )

    f = numpyro.deterministic("f", net(x, mask))   # (N,)
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    numpyro.sample("obs", dist.Normal(f, sigma), obs=y)
```

The key design choice here is making the dropout mask a first-class
`numpyro.sample` site rather than using JAX's random number generator
internally. This means:
- During **MAP training** via SVI, the mask is optimized (effectively ignored)
- During **prediction** via `Predictive`, fresh masks are drawn for each
  forward pass, giving Monte Carlo uncertainty estimates

---

## Model 4: Bayesian Neural Network

### Mathematical Formulation

A fully Bayesian neural network places priors on all weights and performs
posterior inference via MCMC. The generative model is identical to Model 2:

W_1 ~ 𝒩(0, σ_p² I), b_1 ~ 𝒩(0, σ_p² I)

W_2 ~ 𝒩(0, σ_p² I), b_2 ~ 𝒩(0, σ_p²)

σ ~ HalfNormal(1)

f(x) = tanh(x W_1 + b_1) W_2 + b_2

y_i ~ 𝒩(f(x_i), σ²)

The difference from Model 2 is the inference method: instead of MAP (a single
point estimate), we run NUTS to draw correlated samples from the full posterior
p(W_1, b_1, W_2, b_2, σ | 𝒟). The `prior_scale`
parameter σ_p controls how spread out the prior is — larger values
allow more complex functions but may slow MCMC mixing.

Same `MLP` architecture, but with configurable `prior_scale` and intended
for full MCMC over all weights.

```python
mcmc = rm.run_mcmc(rm.model_bayesian_nnet, k_infer,
                   data["x_train"], data["y_train"],
                   num_warmup=1_000, num_samples=2_000,
                   hidden_dim=50, prior_scale=1.0)
```

The `num_warmup` phase lets NUTS adapt its step size and mass matrix. After
warmup, `num_samples` draws are collected from the (approximate) stationary
distribution. With 2000 posterior samples, each containing all network weights,
we can compute predictive means and credible intervals that reflect genuine
parameter uncertainty.

---

## Model 5: SVR via Random Fourier Features

### Mathematical Formulation

Random Fourier Features (RFF) (Rahimi & Recht, 2007) provide a
finite-dimensional approximation to a shift-invariant kernel. For the RBF
(squared-exponential) kernel with lengthscale ℓ:

k(x, x') = exp(-(x - x')² / (2ℓ²))

the RFF approximation constructs random features φ: ℝ → ℝ^D:

ω_j ~ 𝒩(0, ℓ⁻²), b_j ~ Uniform(0, 2π), j = 1, …, D

φ(x)_j = √(2/D) cos(ω_j x + b_j)

such that φ(x)ᵀ φ(x') ≈ k(x, x'). This is a consequence of
Bochner's theorem: the Fourier transform of a positive-definite shift-invariant
kernel is a non-negative measure, so we can approximate the kernel by
Monte Carlo sampling from its spectral density.

**Generative model:**

w ~ 𝒩(0_D, I_D), σ ~ HalfNormal(1)

f(x_i) = φ(x_i)ᵀ w = ∑_{j=1}^D φ(x_i)_j w_j

y_i ~ 𝒩(f(x_i), σ²)

The frequencies ω and phases b are **fixed** (sampled once at module
construction), while the weights w are the only learnable/inferrable
parameters. This makes the model linear in w — we get the expressiveness
of a kernel method with the computational cost of linear regression.

### The Equinox Module

```python
class RFFFeatureMap(eqx.Module):
    omega: jax.Array     # (D,) — random frequencies, FIXED
    bias: jax.Array      # (D,) — random phases, FIXED
    n_features: int = eqx.field(static=True)

    def __call__(self, x):
        # x: (N,)
        D = self.n_features
        # einsum outer product: x_n * omega_d -> projection (N, D)
        projection = einsum(x, self.omega, "n, d -> n d")
        # output: (N, D) — the D-dimensional feature vector for each of the N points
        return jnp.sqrt(2.0 / D) * jnp.cos(projection + self.bias)
```

Compare with the old version: `x[:, None] * self.omega[None, :]`.  The
einsum `"n, d -> n d"` says "outer product" without any `None` indexing
gymnastics.

### The NumPyro Model

```python
def model_svr_rff(x, y=None, *, rff_module):
    # x: (N,), y: (N,) or None
    D = rff_module.n_features
    Phi = rff_module(x)  # deterministic Equinox call, shape (N, D)

    w = numpyro.sample("w", dist.Normal(jnp.zeros(D), 1.0))        # (D,)
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))           # ()

    # einsum: contract over the feature dimension d -> output (N,)
    f = numpyro.deterministic("f", einsum(Phi, w, "n d, d -> n"))
    numpyro.sample("obs", dist.Normal(f, sigma), obs=y)
```

Note that the RFF module is passed in as a keyword argument and is not
modified by `eqx.tree_at`. Its parameters (ω, b) are fixed random
draws — only the linear weights w are inferred. This is the key to the
computational savings: inference is over a D-dimensional space rather than
the N × N kernel matrix.

### Usage

```python
rff = rm.RFFFeatureMap(n_features=50, lengthscale=1.0, key=jr.PRNGKey(0))

mcmc = rm.run_mcmc(rm.model_svr_rff, k_infer,
                   data["x_train"], data["y_train"],
                   rff_module=rff)
```

---

## Model 6: Approximate Gaussian Process (RFF)

### Mathematical Formulation

This model extends Model 5 with a **hierarchical prior** on the weight
amplitude, making it a closer approximation to a full Gaussian process.
In a GP with RBF kernel, the function prior is:

f ~ GP(0, α² k_ℓ(x, x'))

where α is the signal amplitude and k_ℓ is the unit-variance
RBF kernel with lengthscale ℓ. The RFF approximation with a learned
amplitude becomes:

α ~ HalfNormal(1)

w ~ 𝒩(0_D, α² I_D)

σ ~ HalfNormal(0.1)

f(x_i) = φ(x_i)ᵀ w

y_i ~ 𝒩(f(x_i), σ²)

The amplitude α controls the marginal variance of f. By placing a
prior on α and inferring it alongside w, the model can learn how
much signal variance the data supports — a form of automatic relevance
determination. The tighter prior HalfNormal(0.1) on σ
reflects an assumption that the observation noise is small.

Like Model 5, but with a hierarchical prior on the weight amplitude.

```python
def model_gp_rff(x, y=None, *, rff_module):
    # x: (N,), y: (N,) or None
    D = rff_module.n_features
    Phi = rff_module(x)   # (N, D)

    amplitude = numpyro.sample("amplitude", dist.HalfNormal(1.0))          # ()
    w = numpyro.sample("w", dist.Normal(jnp.zeros(D), amplitude))          # (D,)
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))                  # ()

    f = numpyro.deterministic("f", einsum(Phi, w, "n d, d -> n"))          # (N,)
    numpyro.sample("obs", dist.Normal(f, sigma), obs=y)
```

---

## Model 7: Deep Gaussian Process (Two-Layer RFF)

### Mathematical Formulation

A deep Gaussian process (Cutajar et al., 2017; Damianou & Lawrence, 2013)
stacks GP layers, where the output of one GP feeds into the input of the next.
Using RFF approximations at each layer makes this tractable:

**Layer 1:** Map the scalar input x to an intermediate representation.

φ⁽¹⁾(x) ∈ ℝ^{D_1}  (RFF features with frequencies ω⁽¹⁾)

h(x) = φ⁽¹⁾(x) W_1 ∈ ℝ^{d_inner}

where W_1 ∈ ℝ^{D_1 × d_inner}.

**Layer 2:** Apply a second set of RFF features to each dimension of
h independently, then average:

φ⁽²⁾_j = φ⁽²⁾(h_j) ∈ ℝ^{D_2}, j = 1, …, d_inner

Φ̄⁽²⁾ = 1/d_inner ∑_{j=1}^{d_inner} φ⁽²⁾_j ∈ ℝ^{N × D_2}

f(x) = Φ̄⁽²⁾ w_2, w_2 ∈ ℝ^{D_2}

The averaging over inner dimensions acts as a form of regularization and
makes the second layer's features a smooth function of the first layer's
multi-dimensional output.

### The Equinox Module

```python
class DeepRFFRegressor(eqx.Module):
    rff1: RFFFeatureMap    # layer 1 (fixed frequencies/phases)
    W1: jax.Array          # (D1, inner_dim) — learnable
    rff2: RFFFeatureMap    # layer 2 (fixed frequencies/phases)
    w2: jax.Array          # (D2,) — learnable
    inner_dim: int = eqx.field(static=True)

    def __call__(self, x):
        # x: (N,)
        Phi1 = self.rff1(x)                                             # (N, D1)
        h = einsum(Phi1, self.W1, "n d1, d1 inner -> n inner")          # (N, inner_dim)

        # Apply rff2 to each inner dimension separately, then average
        Phi2 = jnp.mean(
            jnp.stack([self.rff2(h[:, j]) for j in range(self.inner_dim)]),
            axis=0,
        )                                                                # (N, D2)
        return einsum(Phi2, self.w2, "n d2, d2 -> n")                   # (N,)
```

**Shape walkthrough for N=100, D1=20, D2=10, inner_dim=5:**

| Expression | Shape | Notes |
|---|---|---|
| `x` | `(100,)` | Raw input |
| `self.rff1(x)` | `(100, 20)` | Layer-1 RFF features |
| `self.W1` | `(20, 5)` | Layer-1 to inner-dim projection |
| `einsum(..., "n d1, d1 inner -> n inner")` | `(100, 5)` | Inner representation |
| `h[:, j]` for each j | `(100,)` | One slice of inner representation |
| `self.rff2(h[:, j])` | `(100, 10)` | Layer-2 RFF features for slice j |
| `jnp.stack([...])` | `(5, 100, 10)` | Stack over inner_dim |
| `jnp.mean(..., axis=0)` | `(100, 10)` | Average over inner_dim |
| `self.w2` | `(10,)` | Final linear weights |
| `einsum(..., "n d2, d2 -> n")` | `(100,)` | Scalar prediction |

### Usage

```python
deep = rm.DeepRFFRegressor(
    n_features_1=20, n_features_2=10, inner_dim=5, key=jr.PRNGKey(0)
)

mcmc = rm.run_mcmc(rm.model_deep_gp_rff, k_infer,
                   data["x_train"], data["y_train"],
                   deep_rff_module=deep)
```

---

## The Three-Library Pattern

Every model in this tutorial follows the same structure:

```python
from einops import einsum

def model_*(x, y=None, **kwargs):
    # 1. Create Equinox module (shape template)
    net = SomeModule(**kwargs)

    # 2. Sample weights from priors (NumPyro)
    w = numpyro.sample("w", some_prior)

    # 3. Inject samples into the module (eqx.tree_at)
    net = eqx.tree_at(lambda m: m.weight, net, w)

    # 4. Forward pass — all matmuls use einsum (einops)
    f = numpyro.deterministic("f", net(x))

    # 5. Likelihood (NumPyro)
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    numpyro.sample("obs", dist.Normal(f, sigma), obs=y)
```

**Step 1 (Equinox):** The module is constructed with placeholder weights
(typically zeros or random initialization). This defines the architecture —
shapes, activations, and the forward-pass computation graph.

**Step 2 (NumPyro):** `numpyro.sample` declares a random variable with a
name and a prior distribution. During MCMC, NumPyro draws values from the
posterior. During SVI, it optimizes the guide's parameters. The returned
array has the same shape as the distribution.

**Step 3 (eqx.tree_at):** The sampled values are injected into the module,
replacing the placeholders. This is the bridge between the probabilistic
world (NumPyro) and the deterministic world (Equinox).

**Step 4 (einops + Equinox):** The module's `__call__` runs the forward
pass. All linear algebra uses `einsum` with named axes. The result is
wrapped in `numpyro.deterministic` so it appears in the trace (useful for
extracting predictions later).

**Step 5 (NumPyro):** The likelihood connects the model's predictions to
the observed data. When `obs=y` is provided (training), NumPyro conditions
on the data. When `obs=None` (prediction), it samples from the likelihood,
generating predictive samples.

**Equinox** owns the architecture.  **einops** owns the linear algebra.
**NumPyro** owns the probability.  Each library does one thing.

---

## Why einops over `@`?

The `@` operator is concise but opaque for anything beyond 2D x 2D:

```python
# What does this do?  Which axis is contracted?
(h @ self.W2).squeeze(-1) + self.b2
```

The einsum version is self-documenting:

```python
# Contract hidden dim (h), collapse singleton (one), keep samples (n)
einsum(h, self.W2, "n h, h one -> n") + self.b2
```

For the RFF outer product, the difference is even starker:

```python
# Before: what do these None indices mean?
x[:, None] * self.omega[None, :]

# After: outer product, named axes
einsum(x, self.omega, "n, d -> n d")
```

The einops version reads like a specification of the operation rather
than an implementation of it.

---

## Summary Table

| Model | Equinox Module | einsum patterns | NumPyro Model | Inference |
|---|---|---|---|---|
| Bayesian linear | `LinearRegressor` | `"n feat, feat -> n"` | `model_linear` | MCMC |
| Neural net | `MLP` | `"n, one h -> n h"` + `"n h, h one -> n"` | `model_nnet` | MAP (SVI + AutoDelta) |
| MC-Dropout NN | `MLPDropout` | same as MLP | `model_nnet_dropout` | MAP + MC sampling |
| Bayesian NN | `MLP` | same as MLP | `model_bayesian_nnet` | MCMC (NUTS) |
| SVR (RFF) | `RFFFeatureMap` | `"n, d -> n d"` + `"n d, d -> n"` | `model_svr_rff` | MCMC |
| Approx GP | `RFFFeatureMap` | same as SVR | `model_gp_rff` | MCMC |
| Deep GP | `DeepRFFRegressor` | `"n d1, d1 inner -> n inner"` + `"n d2, d2 -> n"` | `model_deep_gp_rff` | MCMC |

## References

1. Rasmussen & Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006. The foundational text on GP regression and classification. Models 5-7 approximate GP inference using random features.
2. Rahimi & Recht, "Random Features for Large-Scale Kernel Machines", *NeurIPS*, 2007. Introduced Random Fourier Features (RFF) — the theoretical basis for Models 5, 6, and 7. Shows that shift-invariant kernels can be uniformly approximated by random cosine features via Bochner's theorem.
3. Cutajar et al., "Random Feature Expansions for Deep Gaussian Processes", *ICML*, 2017. Extends RFF to deep GP architectures by stacking RFF layers. Model 7 follows this approach.
4. Gal & Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning", *ICML*, 2016. Shows that dropout training approximates inference in a deep Gaussian process. Model 3 implements this idea, with the dropout mask as an explicit random variable.
5. Damianou & Lawrence, "Deep Gaussian Processes", *AISTATS*, 2013. The original deep GP formulation using variational inference with inducing points. Model 7 uses the RFF approximation instead.
6. Hoffman & Gelman, "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo", *JMLR*, 2014. The NUTS algorithm used by NumPyro for MCMC inference throughout this tutorial.
7. Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations", 2021. The Equinox library used for all module definitions.
8. Rogozhnikov, "einops: Clear and Reliable Tensor Manipulations with Einstein-like Notation", *ICLR*, 2022. The einops library used for all tensor contractions.
9. Aboleth tutorial: https://aboleth.readthedocs.io/en/stable/tutorials/some_regressors.html. The original tutorial that this document reimplements.
