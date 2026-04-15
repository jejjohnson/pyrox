---
status: draft
version: 0.1.0
---

# Layer 0 -- Core Abstraction Examples

Two patterns for making Equinox modules probabilistic, plus the Parameterized base for GP-style modules.

All core abstractions live in `pyrox._core`.

---

## Pattern A: PyroxModule with pyrox_sample / pyrox_param

### Module owns its probabilistic semantics

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from pyrox._core import PyroxModule

class BayesianLinear(PyroxModule):
    in_features: int
    out_features: int

    def __call__(self, x):
        W = self.pyrox_sample(
            "weight",
            dist.Normal(0, 1).expand([self.in_features, self.out_features]).to_event(2),
        )
        b = self.pyrox_param("bias", jnp.zeros(self.out_features))
        return x @ W + b
```

### Dependent priors -- prior on scale depends on sampled loc

```python
class LocationScale(PyroxModule):
    def __call__(self, x):
        loc = self.pyrox_sample("loc", dist.Normal(0, 1))
        scale = self.pyrox_sample(
            "scale",
            lambda self_: dist.LogNormal(self_.loc, 0.1),  # depends on sampled loc
        )
        return numpyro.sample("obs", dist.Normal(loc, scale), obs=x)
```

---

## Pattern B: Pure Equinox + eqx.tree_at

### Architecture and probabilistic model are separate

```python
import equinox as eqx
import jax.random as jr
import numpyro
import numpyro.distributions as dist

# Pure deterministic module -- no NumPyro dependency
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

# Probabilistic model -- priors + tree_at injection
def model(x, y=None):
    net = MLP(key=jr.PRNGKey(0))
    W1 = numpyro.sample("W1", dist.Normal(jnp.zeros((1, 50)), 1.0))
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(50), 1.0))
    net = eqx.tree_at(lambda m: (m.W1, m.b1), net, (W1, b1))
    f = numpyro.deterministic("f", net(x))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    numpyro.sample("obs", dist.Normal(f.squeeze(), sigma), obs=y)
```

---

## Parameterized (GP-style)

### Register params, attach priors, auto-generate guides

```python
from pyrox._core import Parameterized
import numpyro.distributions as dist

class RBFKernel(Parameterized):
    def setup(self):
        self.register_param("variance", jnp.array(1.0), constraint=dist.constraints.positive)
        self.register_param("lengthscale", jnp.array(1.0), constraint=dist.constraints.positive)
        self.set_prior("variance", dist.LogNormal(0.0, 1.0))
        self.set_prior("lengthscale", dist.LogNormal(0.0, 1.0))

    def __call__(self, X1, X2):
        v = self.get_param("variance")
        l = self.get_param("lengthscale")
        sq_dist = jnp.sum((X1[:, None] - X2[None, :]) ** 2 / l ** 2, axis=-1)
        return v * jnp.exp(-0.5 * sq_dist)
```

---

*For how these patterns compose into full models, see [nn_models.md](nn_models.md). For GP kernel implementations using `Parameterized`, see [gp_primitives.md](gp_primitives.md).*
