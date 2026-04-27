# Conditioning

A *conditioner* is a layer `c(h, z) → y` that transforms an inner activation `h` based on a context vector `z`. `pyrox.nn` ships three concrete conditioners that cover the literature in one consistent API, plus Bayesian variants and a composite that wraps any inner network with per-layer conditioning.

## Decision rubric

| Pattern | Use when | Cost | Where it shows up |
|---|---|---|---|
| `ConcatConditioner` | You want a cheap baseline; `cond_dim` is small. | `(C + K) · C + C` per layer | DiffeqMLP-style CNF vector fields |
| `AffineModulation` (`FiLM`) | Feature-wise modulation is enough; you want low generator cost regardless of `cond_dim`. | `2C · K + 2C` per layer | π-GAN, Modulated SIREN, conditional INRs |
| `HyperLinear` | You need the full target weight matrix to depend on `z`; want NIF/MetaSDF. | `K · (C·C_in + C)` per layer | NIF (Pan et al. 2023), MetaSDF, Ha et al. hypernets |

The Bayesian variants put `Normal(0, prior_std)` priors on the **generator** weights only. Posterior cost scales with the *generator* size, not the target network — that's the whole point of Bayesian amortised inference.

## End-to-end use

```python
import jax.random as jr
import jax.numpy as jnp
from pyrox.nn import SIREN, AffineModulation, ConditionedINR, HyperSIREN

key = jr.key(0)

# 1) FiLM-modulate every hidden layer of a SIREN
inner = SIREN.init(2, 32, 1, depth=4, key=key)
wrapped = ConditionedINR.init(
    inner, conditioner_cls=AffineModulation, cond_dim=4, key=key
)
y = wrapped(jnp.ones((10, 2)), jnp.ones((10, 4)))   # (10, 1)

# 2) Full NIF stack — ParameterNet → per-layer HyperLinear → ShapeNet (SIREN)
import equinox as eqx
class IdentityNet(eqx.Module):
    def __call__(self, mu): return mu

nif = HyperSIREN(
    in_features=2, hidden_features=32, out_features=1,
    depth=5, cond_dim=3, parameter_net=IdentityNet(), key=key,
)
y = nif(jnp.ones((10, 2)), jnp.ones((3,)))           # (10, 1)
```

For a hands-on walkthrough see the [Conditional Neural Fields notebook](../../notebooks/conditioning.ipynb).

## Protocol

::: pyrox.nn.AbstractConditioner

## Concrete conditioners

::: pyrox.nn.ConcatConditioner

::: pyrox.nn.AffineModulation

::: pyrox.nn.FiLM

::: pyrox.nn.HyperLinear

## Bayesian variants

::: pyrox.nn.BayesianConcatConditioner

::: pyrox.nn.BayesianAffineModulation

::: pyrox.nn.BayesianHyperLinear

## Spectral hyper-conditioning

`HyperFourierFeatures` is the conditional analogue of `RBFFourierFeatures`: instead of sampling the random Fourier features' `(W, b, lengthscale)` from a fixed prior, a user-supplied parameter network produces them from the context vector. `ConditionedRFFNet` adds a learnable linear readout — the conditional analogue of `RandomKitchenSinks`.

::: pyrox.nn.HyperFourierFeatures

::: pyrox.nn.ConditionedRFFNet

## Composites

::: pyrox.nn.ConditionedINR

::: pyrox.nn.HyperSIREN
