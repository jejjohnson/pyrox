---
status: draft
version: 0.1.0
---

# Layer 1 -- Probabilistic Layer Examples

Edward2-style Bayesian layers that compose like normal Equinox layers. All layers live in `pyrox.nn`.

---

## DenseVariational -- Full Bayesian Dense Layer

```python
from pyrox.nn import DenseVariational

# Stack Bayesian dense layers for a variational BNN
class BayesianMLP(eqx.Module):
    layer1: DenseVariational
    layer2: DenseVariational

    def __init__(self, *, key):
        k1, k2 = jr.split(key)
        self.layer1 = DenseVariational(output_dim=50, prior_std=1.0, key=k1)
        self.layer2 = DenseVariational(output_dim=1, prior_std=1.0, key=k2)

    def __call__(self, x):
        h = jax.nn.tanh(self.layer1(x))
        return self.layer2(h)
```

---

## MCDropout -- Uncertainty via Stochastic Forward Passes

```python
from pyrox.nn import MCDropout

class DropoutMLP(eqx.Module):
    linear1: eqx.nn.Linear
    dropout: MCDropout
    linear2: eqx.nn.Linear

    def __call__(self, x, *, key):
        h = jax.nn.relu(self.linear1(x))
        h = self.dropout(h, key=key)  # always stochastic -- training AND inference
        return self.linear2(h)

# Uncertainty: run N forward passes, collect predictions
keys = jr.split(jr.PRNGKey(0), 100)
preds = jax.vmap(lambda k: model(x_test, key=k))(keys)
mean, std = preds.mean(0), preds.std(0)
```

---

## RandomFourierFeatures -- Approximate GP Layer

### RBF kernel approximation with optionally learnable lengthscale

```python
from pyrox.nn import RandomFourierFeatures

# RFF: z(x) = sqrt(2/D) [cos(Wx + b)] where W ~ N(0, 1/l^2)
rff = RandomFourierFeatures(
    n_features=100,
    learn_lengthscale=True,
    init_lengthscale=1.0,
    key=jr.PRNGKey(0),
)

# Output: (batch, 2 * n_features) -- cos and sin features
features = rff(x)  # (N, 200)
```

---

## DenseNCP -- Noise Contrastive Prior

```python
from pyrox.nn import NCPContinuousPerturb, DenseNCP

# NCP: train on original + perturbed inputs, penalize divergence
perturb = NCPContinuousPerturb(input_noise=0.1)
ncp_dense = DenseNCP(output_dim=1, prior_std=1.0)

def forward(x, *, key):
    x_doubled = perturb(x, key=key)  # (2N, D) -- original + noisy
    return ncp_dense(x_doubled)        # registers ncp_kl deterministic site
```

---

*For core abstractions (PyroxModule, Parameterized), see [core.md](core.md). For composition patterns and full models, see [nn_models.md](nn_models.md).*
