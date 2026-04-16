# pyrox

> Probabilistic modeling with Equinox and NumPyro: Bayesian neural networks, Gaussian processes, and composable GP building blocks.

pyrox bridges [Equinox](https://docs.kidger.site/equinox/) modules and [NumPyro](https://num.pyro.ai) traces so one module class can host deterministic parameters, random sample sites, priors, guides, and mode switching — without duplicating inference logic. NumPyro owns inference; pyrox just makes modules visible to it.

## Package layout

- **`pyrox._core`** — the Equinox-to-NumPyro bridge. `PyroxModule`, `PyroxParam`, `PyroxSample`, `Parameterized`, `pyrox_method`.
- **`pyrox.gp`** — Gaussian process building blocks and protocols (Wave 2+).
- **`pyrox.nn`** — Bayesian and uncertainty-aware NN layers (Wave 3+).

Wave 1 ships `pyrox._core`. GP and NN subpackages are scaffolded as placeholders until their dedicated waves land.

## Installation

```bash
pip install pyrox
```

Or with `uv`:

```bash
uv add pyrox
```

## Three modeling patterns

### Pattern A — pure Equinox module injected into a NumPyro model

```python
import equinox as eqx
import numpyro


def model(x, y=None):
    net = MLP(key=key)                      # any eqx.Module
    W = numpyro.sample("W", prior)
    net = eqx.tree_at(lambda m: m.W, net, W)
    f = numpyro.deterministic("f", net(x))
    numpyro.sample("obs", dist.Normal(f, 0.1), obs=y)
```

### Pattern B — `PyroxModule` owns its probabilistic semantics

```python
import jax.numpy as jnp
import numpyro.distributions as dist
from pyrox._core import PyroxModule, pyrox_method


class BayesianLinear(PyroxModule):
    pyrox_name = "BayesianLinear"
    in_features: int
    out_features: int

    @pyrox_method
    def __call__(self, x):
        W = self.pyrox_sample(
            "weight",
            dist.Normal(0, 1)
                .expand([self.in_features, self.out_features])
                .to_event(2),
        )
        b = self.pyrox_param("bias", jnp.zeros(self.out_features))
        return x @ W + b
```

### Pattern C — `Parameterized` for constrained params, priors, and guides

```python
import jax.numpy as jnp
import numpyro.distributions as dist
from pyrox._core import Parameterized, pyrox_method


class RBFKernel(Parameterized):
    pyrox_name = "RBFKernel"

    def setup(self):
        self.register_param(
            "variance", jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        self.register_param(
            "lengthscale", jnp.array(1.0),
            constraint=dist.constraints.positive,
        )
        self.set_prior("variance", dist.LogNormal(0.0, 1.0))
        self.autoguide("variance", "normal")

    @pyrox_method
    def __call__(self, X1, X2):
        v = self.get_param("variance")
        ls = self.get_param("lengthscale")
        sq = jnp.sum((X1[:, None] - X2[None, :]) ** 2 / ls ** 2, axis=-1)
        return v * jnp.exp(-0.5 * sq)
```

Switch `kernel.set_mode("guide")` to draw variational params instead of sampling the prior. `"normal"` autoguides respect the registered constraint via `TransformedDistribution`.

## Links

- [Core API](api/core.md)
- [GP API](api/gp.md) *(scaffold)*
- [NN API](api/nn.md) *(scaffold)*
- [Contributing](contributing.md)
- [Changelog](CHANGELOG.md)
- [GitHub](https://github.com/jejjohnson/pyrox)
