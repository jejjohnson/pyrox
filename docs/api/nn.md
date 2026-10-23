# NN API

The `pyrox_nn` subpackage ships uncertainty-aware neural network layers in four families:

1. **Geographic / spherical encoders** (re-exported from `geonnax`) — degree/radian, lon/lat, cyclic, spherical-harmonic, and Slepian preprocessing for geophysical inputs.
2. **Dense / Bayesian-linear layers** (`pyrox_nn._dense`) — reparameterization, Flipout, hierarchical, NCP, DVI, rank-1 ensemble, and variational-dropout variants of `Wx + b`.
3. **Spectral / GP-flavoured layers** — random-feature kernel maps, SNGP and VSSGP heads, deep random-feature expansions.
4. **Ensembles & output heads** — BatchEnsemble layers, heteroscedastic Monte-Carlo output heads.
5. **Bayesian Neural Field stack** (`pyrox_nn._bnf`) — five layers that together implement the BNF architecture (Saad et al., Nat. Comms. 2024); `BayesianNeuralField` wraps the deterministic `geonnax.NeuralFieldMLP` and takes an optional coordinate `encoder` (for example spherical harmonics on lon/lat).
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

## Spherical-harmonic GP features

`SphericalHarmonicFeatures` is the $S^2$ analogue of `HSGPFeatures`: real
spherical harmonics up to degree $L$ with $\mathcal{N}(0, a_l)$ weights, where
$a_l$ are the kernel's Funk-Hecke coefficients. With fixed kernel
hyperparameters it is a GP with the truncated kernel, linear in the number of
observations, so it is an optimal-interpolation baseline that runs through the
same inference code as a neural field. The sketch below fits it and a
`BayesianNeuralField` on the same synthetic global field with the same
`pyrox.inference.ensemble_map` loop, so the comparison isolates the network.

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from numpyro import handlers
from pyrox.inference import ensemble_map
from pyrox_gp import Matern
from pyrox_nn import BayesianNeuralField, SphericalHarmonicFeatures

# Synthetic global field: 2,000 noisy observations at random lon / lat (degrees).
k_lon, k_lat, k_eps = jr.split(jr.key(0), 3)
lon = jr.uniform(k_lon, (2000,), minval=-180.0, maxval=180.0)
lat = jnp.rad2deg(jnp.arcsin(jr.uniform(k_lat, (2000,), minval=-1.0, maxval=1.0)))
x = jnp.stack([lon, lat], axis=-1)  # (N, 2)
lam, phi = jnp.deg2rad(lon), jnp.deg2rad(lat)
y = jnp.sin(2 * lam) * jnp.cos(phi) + 0.5 * jnp.sin(phi) ** 2
y = y + 0.1 * jr.normal(k_eps, y.shape)


def fit(layer, sigma=0.1):
    """MAP-fit every sample site of ``layer`` under a Normal likelihood."""

    def log_joint(params, xb, yb):
        with handlers.trace() as tr, handlers.substitute(data=params):
            f = layer(xb)
        logprior = sum(
            s["fn"].log_prob(s["value"]).sum()
            for s in tr.values()
            if s["type"] == "sample"
        )
        return dist.Normal(f, sigma).log_prob(yb).sum(), logprior

    def init_fn(key):
        with handlers.trace() as tr, handlers.seed(rng_seed=key):
            layer(x[:1])
        return {n: s["value"] for n, s in tr.items() if s["type"] == "sample"}

    params, _ = ensemble_map(
        log_joint, init_fn, ensemble_size=4, num_epochs=2000,
        data=(x, y), seed=jr.key(1),
    )
    # Ensemble-mean prediction at the training points.
    return jax.vmap(lambda p: handlers.substitute(layer, data=p)(x))(params).mean(0)


gp = SphericalHarmonicFeatures.init(
    l_max=30,
    kernel=Matern(nu=1.5, init_lengthscale=0.5),  # fixed hyperparameters
    input_mode="lonlat",
    input_unit="degrees",
)
bnf = BayesianNeuralField(
    input_scales=(180.0, 90.0),
    fourier_degrees=(4, 4),
    interactions=((0, 1),),
    seasonality_periods=(),
    num_seasonal_harmonics=(),
    width=64,
    depth=2,
)
for name, layer in [("SH-GP", gp), ("BNF", bnf)]:
    rmse = jnp.sqrt(jnp.mean((fit(layer) - y) ** 2))
    print(f"{name}: train RMSE {rmse:.3f}")
```

The kernel's hyperparameters are `param` sites here, so they stay fixed; give
the kernel priors (`kernel.set_prior(...)`) to fit them as sample sites too.

::: pyrox_nn.SphericalHarmonicFeatures

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

## Neural operators and basis transforms (re-exported from `geonnax`)

Deterministic operator cores from `geonnax`, importable from `pyrox_nn` so
that one import surface covers the stack (`from pyrox_nn import GeoFNO`).

::: pyrox_nn.GeoFNO

::: pyrox_nn.LearnedDeformation

::: pyrox_nn.ManifoldNO

::: pyrox_nn.BasisTransform
