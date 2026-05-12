# Multiplicative Filter Networks (MFN)

Multiplicative Filter Networks (Fathony, Sahu, Willmott, Kolter — ICLR 2021) replace
MLP composition with multiplicative filter chaining. Instead of deeply composing
nonlinearities, each layer **multiplies** the previous activation by a new filter
evaluated directly on the **original input**:

$$
z_1 = g_1(x), \qquad
z_{i+1} = g_{i+1}(x) \odot \bigl(W_i z_i + b_i\bigr), \qquad
y = W_L z_L + b_L.
$$

Two filter families ship in `pyrox.nn`:

- **FourierNet** — $g_i(x) = \sin(\Omega_i x + \varphi_i)$, frequency-domain filters.
  Products of sinusoids span exponentially many frequencies with depth $L$.
- **GaborNet** — $g_i(x) = \sin(\Omega_i x + \varphi_i) \odot \exp(-\tfrac{\gamma_i}{2}\|x - \mu_i\|^2)$,
  Gabor atoms with learned frequency $\Omega_i$, phase $\varphi_i$, location $\mu_i$,
  and bandwidth $\gamma_i$.

**Connection to [`RBFFourierFeatures`][pyrox.nn.RBFFourierFeatures]:**
A `GaborNet` with `depth=1` and $\mu = 0$ is a localized variant of random Fourier features.
As $\gamma \to 0$ (very wide envelope) it recovers the plain RBF-RFF feature map.
See [`HSGPFeatures`][pyrox.nn.HSGPFeatures] for the related Hilbert-space GP basis.

## Quick example

```python
import jax.random as jr
from pyrox.nn import GaborNet
from numpyro import handlers

key = jr.PRNGKey(0)

# Deterministic GaborNet
net = GaborNet.init(in_features=2, hidden_features=64, out_features=1, depth=3, key=key)

import jax.numpy as jnp
x = jnp.ones((100, 2))
y = net(x)          # (100, 1)

# Bayesian GaborNet — sample sites registered for every parameter
from pyrox.nn import BayesianGaborNet
bnet = BayesianGaborNet.init(
    in_features=2, hidden_features=64, out_features=1, depth=3, key=key,
    pyrox_name="gabor",
)
with handlers.seed(rng_seed=1):
    y_sample = bnet(x)  # weights sampled from prior
```

## Filter primitives

::: pyrox.nn.FourierFilter

::: pyrox.nn.GaborFilter

## Composite networks

::: pyrox.nn.FourierNet

::: pyrox.nn.GaborNet

## Bayesian variants

::: pyrox.nn.BayesianFourierNet

::: pyrox.nn.BayesianGaborNet

## Pure-JAX helper

::: pyrox.nn.mfn_forward
