# NN API

The `pyrox.nn` subpackage ships uncertainty-aware neural network layers in three families:

1. **Dense / Bayesian-linear layers** (`pyrox.nn._layers`) — twelve layers covering reparameterization, Flipout, NCP, MC-Dropout, and several random-feature variants.
2. **Bayesian Neural Field stack** (`pyrox.nn._bnf`) — five layers that together implement the BNF architecture (Saad et al., Nat. Comms. 2024).
3. **Pure-JAX feature helpers** (`pyrox.nn._features`) — pandas-free building blocks the BNF layers wrap.

## Dense / Bayesian-linear layers

::: pyrox.nn.DenseReparameterization

::: pyrox.nn.DenseFlipout

::: pyrox.nn.DenseVariational

::: pyrox.nn.MCDropout

::: pyrox.nn.DenseNCP

::: pyrox.nn.NCPContinuousPerturb

::: pyrox.nn.RBFFourierFeatures

::: pyrox.nn.RBFCosineFeatures

::: pyrox.nn.MaternFourierFeatures

::: pyrox.nn.LaplaceFourierFeatures

::: pyrox.nn.ArcCosineFourierFeatures

::: pyrox.nn.RandomKitchenSinks

## Wave-4 spectral layers (#41)

::: pyrox.nn.VariationalFourierFeatures

::: pyrox.nn.OrthogonalRandomFeatures

::: pyrox.nn.HSGPFeatures

## SIREN — Sinusoidal Representation Networks

SIREN (Sitzmann, Martel, Bergman, Lindell, Wetzstein — NeurIPS 2020) replaces
ReLU/GELU with `sin` and prescribes a three-regime initialisation scheme that
keeps pre-activation variance stable across depth.

### Three-regime weight initialisation (Theorem 1)

| Layer | `W` init | Activation |
|-------|----------|------------|
| `"first"` | `U(-1/d_in, 1/d_in)` | `sin(ω₀ · ·)` |
| `"hidden"` | `U(-√(c/d_in)/ω, √(c/d_in)/ω)` | `sin(ω · ·)` |
| `"last"` | `U(-√(c/d_in), √(c/d_in))` | none (linear) |

Bias `b` is initialised `U(-1/√d_in, 1/√d_in)` for every regime.
Typical choice: `ω₀ = ω = 30` for image / high-frequency INR tasks.

### Usage

```python
import jax.random as jr, jax.numpy as jnp
from pyrox.nn import SirenDense, SIREN, BayesianSIREN

# Single layer
layer = SirenDense.init(3, 64, key=jr.PRNGKey(0), layer_type="first")
y = layer(jnp.ones((5, 3)))  # (5, 64)

# Multi-layer network (depth=5 → first + 3 hidden + last)
net = SIREN.init(2, 64, 1, depth=5, key=jr.PRNGKey(0))
y = net(jnp.zeros((100, 2)))  # (100, 1)

# Bayesian variant (no key needed — weights come from the prior)
from numpyro import handlers
bnet = BayesianSIREN.init(2, 32, 1, depth=3)
with handlers.seed(rng_seed=0):
    y = bnet(jnp.zeros((10, 2)))  # (10, 1)
```

!!! note "Alternative INR backbone"
    `SIREN` and `GaborNet` / `FourierNet` (MFN, #87) are complementary INR
    backbones: SIREN composes nonlinearities deeply, while MFN uses a product
    of Gabor filters.  Choose based on the signal's smoothness profile.

::: pyrox.nn.SirenDense

::: pyrox.nn.SIREN

::: pyrox.nn.BayesianSIREN

## Bayesian Neural Field stack

::: pyrox.nn.Standardization

::: pyrox.nn.FourierFeatures

::: pyrox.nn.SeasonalFeatures

::: pyrox.nn.InteractionFeatures

::: pyrox.nn.BayesianNeuralField

## Pure-JAX feature helpers

::: pyrox.nn.fourier_features

::: pyrox.nn.seasonal_features

::: pyrox.nn.seasonal_frequencies

::: pyrox.nn.interaction_features

::: pyrox.nn.standardize

::: pyrox.nn.unstandardize
