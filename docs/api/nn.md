# NN API

The `pyrox_nn` subpackage ships uncertainty-aware neural network layers in four families:

1. **Geographic / spherical encoders** (re-exported from `geonnax`) — degree/radian, lon/lat, cyclic, spherical-harmonic, and Slepian preprocessing for geophysical inputs.
2. **Dense / Bayesian-linear layers** (`pyrox_nn._dense`) — reparameterization, Flipout, hierarchical, NCP, DVI, rank-1 ensemble, and variational-dropout variants of `Wx + b`.
3. **Spectral / GP-flavoured layers** — random-feature kernel maps, SNGP and VSSGP heads, deep random-feature expansions.
4. **Ensembles & output heads** — BatchEnsemble layers, heteroscedastic Monte-Carlo output heads.
5. **Bayesian Neural Field stack** (`pyrox_nn._bnf`) — five layers that together implement the BNF architecture (Saad et al., Nat. Comms. 2024).
6. **Pure-JAX feature helpers** (re-exported from `geonnax.basis`) — pandas-free building blocks the BNF layers wrap.

See also: [Geo encoders](nn/geo_encoders.md) for the longitude/latitude and spherical-harmonic API surface.

## Dense / Bayesian-linear layers

::: pyrox_nn.DenseReparameterization

::: pyrox_nn.DenseFlipout

::: pyrox_nn.DenseVariational

::: pyrox_nn.DenseDVI

::: pyrox_nn.DenseHierarchical

::: pyrox_nn.DenseVariationalDropout

::: pyrox_nn.DenseNCP

::: pyrox_nn.NCPContinuousPerturb

::: pyrox_nn.NCPNormalOutput

::: pyrox_nn.RBFFourierFeatures

::: pyrox_nn.RBFCosineFeatures

::: pyrox_nn.MaternFourierFeatures

::: pyrox_nn.MaternCosineFeatures

::: pyrox_nn.LaplaceFourierFeatures

::: pyrox_nn.LaplaceCosineFeatures

::: pyrox_nn.ArcCosineFourierFeatures

::: pyrox_nn.RandomKitchenSinks

## Wave-4 spectral layers (#41)

::: pyrox_nn.VariationalFourierFeatures

::: pyrox_nn.OrthogonalRandomFeatures

::: pyrox_nn.HSGPFeatures

## SIREN — Sinusoidal Representation Networks

SIREN (Sitzmann, Martel, Bergman, Lindell, Wetzstein — NeurIPS 2020) replaces
ReLU/GELU with `sin` and prescribes a three-regime initialisation scheme that
keeps pre-activation variance stable across depth.

### Three-regime weight initialisation (Theorem 1)

| Layer | `W` init | Activation |
|-------|----------|------------|
| `"first"` | `U(-1/d_in, 1/d_in)` | `sin(ω₀ · (W x + b))` |
| `"hidden"` | `U(-√(c/d_in)/ω, √(c/d_in)/ω)` | `sin(ω · (W x + b))` |
| `"last"` | `U(-√(c/d_in), √(c/d_in))` | none (linear) — `W x + b` |

Bias `b` is initialised `U(-1/√d_in, 1/√d_in)` for every regime.
Typical choice: `ω₀ = ω = 30` for image / high-frequency INR tasks.

### Usage

```python
import jax.random as jr, jax.numpy as jnp
from pyrox_nn import SirenDense, SIREN, BayesianSIREN

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

::: pyrox_nn.SirenDense

::: pyrox_nn.SIREN

::: pyrox_nn.BayesianSIREN

## SNGP — spectral-normalised GP head

The SNGP output layer (Liu et al., 2020): a random-feature GP last layer
whose posterior covariance comes from a Laplace approximation
(`LaplaceRandomFeatureCovariance`, re-exported from `geonnax`), giving
distance-aware uncertainty from a single deterministic forward pass.

::: pyrox_nn.RandomFeatureGaussianProcess

::: pyrox_nn.LaplaceRandomFeatureCovariance

## Deep spectral GPs

::: pyrox_nn.DeepVSSGP

## Ensembles — BatchEnsemble / rank-1

Efficient deep ensembles that share one weight matrix and learn
per-member rank-1 perturbations (Wen et al., 2020; Dusenberry et al.,
2020).

::: pyrox_nn.DenseRank1

::: pyrox_nn.LayerNormEnsemble

::: pyrox_nn.MultiHeadAttentionBE

## Heteroscedastic output heads

Monte-Carlo sigmoid / softmax output layers with factor-analysis noise
(Collier et al., 2021) for input-dependent label noise.

::: pyrox_nn.MCSigmoidDenseFA

::: pyrox_nn.MCSoftmaxDenseFA

## Bayesian Neural Field stack

::: pyrox_nn.Standardization

::: pyrox_nn.FourierFeatures

::: pyrox_nn.SeasonalFeatures

::: pyrox_nn.InteractionFeatures

::: pyrox_nn.BayesianNeuralField

## Pure-JAX feature helpers

::: pyrox_nn.fourier_features

::: pyrox_nn.seasonal_features

::: pyrox_nn.seasonal_frequencies

::: pyrox_nn.interaction_features

::: pyrox_nn.standardize

::: pyrox_nn.unstandardize
