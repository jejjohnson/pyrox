---
status: draft
version: 0.1.0
---

# Integration Examples

Ecosystem composition: gaussx, xtremax, internal cross-subpackage, somax, and data assimilation.

---

## With gaussx (Structured Covariance Backend)

### Solvers delegate to gaussx for structure-exploiting linear algebra

```python
from pyrox.gp.kernels import RBF
from pyrox.gp.solvers import CholeskySolver, CGSolver
from gaussx.ops import solve, logdet
from gaussx.operators import KroneckerOperator

# pyrox.gp's Solver protocol maps to gaussx operations:
#   solver.solve(K, y)  ->  gaussx.ops.solve(K_operator, y)
#   solver.logdet(K)    ->  gaussx.ops.logdet(K_operator)

# For Kronecker GPs, gaussx decomposes automatically:
K_space = rbf_kernel(X_space, X_space, 1.0, l_space)
K_time = rbf_kernel(X_time, X_time, 1.0, l_time)
K = KroneckerOperator(K_space, K_time)

# O(n_space^3 + n_time^3) instead of O((n_space * n_time)^3)
alpha = solve(K, y)
lml = -0.5 * (y @ alpha + logdet(K) + N * jnp.log(2 * jnp.pi))
```

### CovarianceRepresentation -> GaussX Operator Mapping

| CovarianceRepresentation | GaussX Operator |
|---|---|
| `Dense` | `lineax.MatrixLinearOperator` |
| `Diagonal` | `lineax.DiagonalLinearOperator` |
| `LowRank` | `gaussx.operators.LowRankUpdate` |
| `Woodbury` | `gaussx.operators.LowRankUpdate` (general base) |
| `Kronecker` | `gaussx.operators.KroneckerOperator` |

---

## With xtremax (Spatial Extreme Value + GP)

### GP-based spatial pooling for extreme value parameters

```python
from pyrox.gp.models import GPPrior, gp_sample
from pyrox.gp.kernels import RBF
from pyrox.gp.solvers import CholeskySolver
from xtremax.distributions import GEVD

def spatial_gev_model(X_stations, annual_maxima, coords):
    # GP priors on GEV parameters (vary smoothly over space)
    kernel_loc = RBF(variance=1.0, lengthscale=100.0)  # km
    prior_loc = GPPrior(kernel=kernel_loc, solver=CholeskySolver(), X=coords)
    loc = gp_sample("loc", prior_loc)

    scale = numpyro.sample("scale", dist.HalfNormal(5.0))
    shape = numpyro.sample("shape", dist.Normal(0, 0.3))

    # GEV likelihood at each station
    numpyro.sample("obs", GEVD(loc, scale, shape), obs=annual_maxima)
```

---

## Internal: Deep Kernel Learning with pyrox.nn

### Neural net feature extractor + GP (both from the same package)

Since pyrox merges NN and GP subpackages, deep kernel learning is an internal composition pattern rather than a cross-package integration.

```python
import equinox as eqx
from pyrox.gp.kernels import DeepKernel, RBF
from pyrox.gp.solvers import CholeskySolver
from pyrox.gp.models import GPPrior

# Equinox neural net as feature extractor
feature_net = eqx.nn.MLP(in_size=D, out_size=10, width_size=50, depth=2, key=key)

# Deep kernel: k(x, x') = k_RBF(phi(x), phi(x')) where phi is the neural net
kernel = DeepKernel(feature_net=feature_net, base_kernel=RBF())
prior = GPPrior(kernel=kernel, solver=CholeskySolver(), X=X_train)
```

For BNN feature extractors (uncertainty in the feature map), combine with `pyrox.nn.DenseVariational`:

```python
from pyrox.nn import DenseVariational
from pyrox._core import PyroxModule

class BayesianFeatureExtractor(PyroxModule):
    layer1: DenseVariational
    layer2: DenseVariational

    def __init__(self, *, key):
        k1, k2 = jr.split(key)
        self.layer1 = DenseVariational(output_dim=50, prior_std=1.0, key=k1)
        self.layer2 = DenseVariational(output_dim=10, prior_std=1.0, key=k2)

    def __call__(self, x):
        h = jax.nn.tanh(self.layer1(x))
        return self.layer2(h)

# BNN features -> GP kernel
feature_net = BayesianFeatureExtractor(key=jr.PRNGKey(0))
kernel = DeepKernel(feature_net=feature_net, base_kernel=RBF())
```

---

## With somax (Ocean State Estimation)

### GP components as priors for ocean model parameters

pyrox.gp can provide spatially smooth GP priors for ocean model fields in somax. The GP encodes spatial correlation structure, while somax handles the ocean dynamics.

```python
from pyrox.gp.kernels import Matern
from pyrox.gp.solvers import CholeskySolver
from pyrox.gp.models import GPPrior, gp_sample

def ocean_param_model(coords, observations):
    """GP prior on a spatially varying ocean parameter (e.g., diffusivity)."""
    # Smooth spatial prior (Matern-5/2 for twice-differentiable fields)
    kernel = Matern(variance=1.0, lengthscale=200.0, nu=2.5)  # km
    prior = GPPrior(kernel=kernel, solver=CholeskySolver(), X=coords)

    # Latent diffusivity field (log-space for positivity)
    log_kappa = gp_sample("log_kappa", prior)
    kappa = jnp.exp(log_kappa)

    # kappa feeds into somax ocean model as a parameter field
    # somax.forward(state, kappa=kappa, ...)
    ...
```

---

## With vardax / ekalmX (GP Dynamics in Data Assimilation)

### GP-learned dynamics models for state estimation

pyrox.gp can learn transition dynamics from data, which then plug into Kalman-based data assimilation frameworks (vardax, ekalmX).

```python
from pyrox.gp.kernels import RBF
from pyrox.gp.solvers import CholeskySolver
from pyrox.gp.models import GPPrior

def gp_dynamics_model(X_states, X_next_states):
    """Learn f: x_t -> x_{t+1} as a GP, then use for state estimation."""
    kernel = RBF(variance=1.0, lengthscale=1.0)
    prior = GPPrior(kernel=kernel, solver=CholeskySolver(), X=X_states)

    # Train GP on observed transitions
    noise = numpyro.sample("noise", dist.HalfNormal(0.1))
    gp_factor("dynamics", prior, X_next_states, noise_var=noise)

# After training, the GP posterior provides:
#   - Mean prediction: E[x_{t+1} | x_t] for the forecast step
#   - Uncertainty: Var[x_{t+1} | x_t] for the process noise Q
# These plug into ekalmX's EnKF or vardax's 4D-Var as the dynamics model
```

---

## Composition Patterns

| Pattern | Components | Use Case |
|---|---|---|
| Collapsed GP regression | `RBF` + `CholeskySolver` + `gp_factor` | Standard GP, Gaussian likelihood |
| Sparse variational GP | `RBF` + `WoodburySolver` + `InducingPointGuide` + `gp_sample` | Large N, non-Gaussian |
| Temporal GP | `Matern` + `KalmanSolver` + `gp_factor` | Time series, O(NS^3) |
| Kronecker GP | `RBF` + `KroneckerSolver` (via gaussx) | Gridded spatiotemporal |
| Spatial extremes | `GPPrior` + `GEVD` (via xtremax) | GEV parameters as GP fields |
| Deep kernel learning | `DeepKernel` + Equinox MLP + `gp_sample` | Neural feature extractor + GP |
| BNN + GP | `pyrox.nn.DenseVariational` + `DeepKernel` | Uncertain features + GP |
| Unscented Kalman smoother | `KalmanSolver` + `PosteriorLinearisation` + `SigmaPoints` | Non-conjugate temporal |
| Ocean state estimation | `pyrox.gp.GPPrior` + somax dynamics | Spatially smooth ocean parameters |
| GP dynamics + DA | `pyrox.gp.GPPrior` + ekalmX/vardax | Learned transition model |

---

*For GP model workflows, see [gp_models.md](gp_models.md). For NN composition patterns, see [nn_models.md](nn_models.md).*
