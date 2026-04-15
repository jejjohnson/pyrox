---
status: draft
version: 0.1.0
---

# API Overview

pyrox merges pyrox-nn and pyrox-gp into a single package with three submodules: `pyrox._core` (shared probabilistic primitives), `pyrox.nn` (Bayesian neural network layers), and `pyrox.gp` (Gaussian process components and models). The GP subpackage follows a three-layer stack (primitives, components, models) with three subsystem deep-dives (moments, state-space, integration).

---

## Surface Inventory

### Core (`pyrox._core`)

Shared probabilistic module infrastructure, bridging Equinox and NumPyro.

| Export | Type | Description |
|---|---|---|
| `PyroxModule` | `eqx.Module` subclass | Base class bridging Equinox ↔ NumPyro tracing |
| `PyroxParam` | `NamedTuple` | Deterministic learnable parameter descriptor |
| `PyroxSample` | frozen `dataclass` | Random variable descriptor (supports dependent priors) |
| `Parameterized` | extends `PyroxModule` | GP-style base class with `register_param()`, `set_prior()`, `autoguide()`, mode switching |
| `_Context` | internal | Per-call cache preventing duplicate sample sites |
| `pyrox_method` | decorator | Gives non-`__call__` methods their own context scope |

See: [core.md](core.md)

### NN (`pyrox.nn`)

Probabilistic neural network layers built on the core abstractions.

| Module | Key Exports | Description |
|---|---|---|
| `nn.dense` | `DenseVariational`, `DenseReparameterization`, `DenseFlipout`, `DenseBatchEnsemble` | Bayesian dense layers |
| `nn.random_feature` | `RandomFourierFeatures`, `RandomKitchenSinks` | Kernel approximation layers |
| `nn.dropout` | `MCDropout` | Monte Carlo dropout (always stochastic) |
| `nn.noise` | `NCPContinuousPerturb`, `DenseNCP` | Noise contrastive prior layers |

See: [nn/layers.md](nn/layers.md)

### GP Layer 0 — Primitives (`pyrox.gp._src`)

Pure JAX functions. No protocols, no NumPyro.

| Submodule | Key Functions | Description |
|---|---|---|
| `_src.kernels` | `rbf_kernel`, `matern_kernel`, `periodic_kernel`, `linear_kernel` | Kernel evaluation on arrays |
| `_src.covariance` | `cholesky_solve`, `log_marginal_likelihood`, `predictive_mean_and_var` | GP math (solve + logdet) |
| `_src.kalman` | `kalman_predict`, `kalman_update`, `rts_smooth` | Kalman filter/smoother primitives |
| `_src.quadrature` | `gauss_hermite_points_and_weights`, `sigma_points`, `cubature_points` | Quadrature rules |
| `_src.sde` | `matern_to_sde` | Kernel → LTI SDE conversion |
| `_src.pathwise` | `rff_prior_draw`, `pathwise_update` | Pathwise posterior sampling |

See: [gp/primitives.md](gp/primitives.md)

### GP Layer 1 — Components (protocols)

| Protocol | Module | Key Method | Detail |
|---|---|---|---|
| `Kernel` | `pyrox.gp.kernels` | `__call__(X1, X2)` | [gp/moments.md](gp/moments.md) |
| `Solver` | `pyrox.gp.solvers` | `solve`, `logdet`, `log_marginal` | [gp/moments.md](gp/moments.md) |
| `InferenceStrategy` | `pyrox.gp.inference` | `compute_sites(f, y, lik)` | [gp/state_space.md](gp/state_space.md) |
| `Integrator` | `pyrox.gp.integrators` | `integrate(fn, mean, var)` | [gp/integration.md](gp/integration.md) |
| `Guide` | `pyrox.gp.guides` | `sample(key)`, `log_prob(f)` | [gp/moments.md](gp/moments.md) |
| `InducingFeatures` | `pyrox.gp.inducing_features` | `K_uu(kernel)`, `k_u(X, kernel)` | [../features/gp/inducing_features.md](../features/gp/inducing_features.md) |
| `MultiOutputKernel` | `pyrox.gp.multi_output` | `Kgg(X1, X2)`, `mixing_matrix()` | [../features/gp/multi_output.md](../features/gp/multi_output.md) |

See: [gp/components.md](gp/components.md)

### GP Layer 2 — Models (entry points)

| Export | Purpose |
|---|---|
| `gp_sample(name, prior, guide)` | Latent GP as NumPyro sample site (non-Gaussian likelihood) |
| `gp_factor(name, prior, y, noise)` | Collapsed GP as NumPyro factor (Gaussian likelihood) |
| `GPPrior(kernel, solver, X)` | GP prior at observed locations |
| `ConditionedGP` | GP conditioned on observations (posterior predictions) |
| `MarkovGPPrior(kernel, solver, times)` | Temporal GP via state-space/Kalman |
| `multi_output_gp_sample(name, mo_kernel, X)` | Multi-output latent GP (L latent GPs + mixing) |
| `multi_output_gp_factor(name, mo_kernel, Y, noise)` | Multi-output collapsed GP (OILMM) |
| `PathwiseSampler(conditioned_gp)` | Efficient posterior function samples via Matheron's rule |

See: [gp/models.md](gp/models.md)

---

## Subsystem Deep-Dives

| Subsystem | Design Doc | Focus |
|---|---|---|
| **Moments** (core) | [gp/moments.md](gp/moments.md) | Kernels, solvers, GPPrior, guides, covariance representations |
| **State-Space** (temporal) | [gp/state_space.md](gp/state_space.md) | Markovian GPs, Kalman, Bayes-Newton, InferenceStrategy |
| **Integration** | [gp/integration.md](gp/integration.md) | Gaussian expectations, sigma points, cubature, uncertain inputs |

---

## Import Conventions

```python
# Core (pyrox._core)
from pyrox._core import PyroxModule, PyroxParam, PyroxSample, Parameterized

# NN layers (pyrox.nn)
from pyrox.nn.dense import DenseVariational, DenseFlipout
from pyrox.nn.dropout import MCDropout
from pyrox.nn.random_feature import RandomFourierFeatures
from pyrox.nn.noise import DenseNCP, NCPContinuousPerturb

# GP Layer 0 — Pure JAX functions (pyrox.gp._src)
from pyrox.gp._src.kernels import rbf_kernel, matern_kernel
from pyrox.gp._src.covariance import log_marginal_likelihood
from pyrox.gp._src.kalman import kalman_predict, kalman_update

# GP Layer 1 — Protocols (pyrox.gp)
from pyrox.gp.kernels import RBF, Matern, Periodic
from pyrox.gp.solvers import CholeskySolver, CGSolver, KalmanSolver, WoodburySolver
from pyrox.gp.inference import VariationalInference, PosteriorLinearisation
from pyrox.gp.integrators import GaussHermite, SigmaPoints, Cubature
from pyrox.gp.guides import WhitenedGuide, InducingPointGuide, SparseGuide
from pyrox.gp.inducing_features import SphericalHarmonicFeatures, FourierFeatures
from pyrox.gp.multi_output import (
    LinearCoregionalizationKernel, OILMMKernel,
    multi_output_gp_sample, mix_latent_predictions,
)

# GP Layer 2 — Entry points (pyrox.gp)
from pyrox.gp.models import GPPrior, MarkovGPPrior, gp_sample, gp_factor
from pyrox.gp.sampling import PathwiseSampler, DecoupledPathwiseSampler

# Inference (from NumPyro — not pyrox)
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal, AutoDelta
```

---

## Detail Files

| File | Covers |
|---|---|
| [core.md](core.md) | Core — PyroxModule, PyroxParam, PyroxSample, Parameterized, _Context |
| [nn/layers.md](nn/layers.md) | NN — Dense, RFF, Dropout, NCP layers |
| [gp/primitives.md](gp/primitives.md) | GP Layer 0 — pure JAX kernel eval, covariance math, Kalman, quadrature, SDE conversion |
| [gp/components.md](gp/components.md) | GP Layer 1 — Kernel, Solver, InferenceStrategy, Integrator, Guide protocols |
| [gp/models.md](gp/models.md) | GP Layer 2 — gp_sample, gp_factor, GPPrior, ConditionedGP, MarkovGPPrior |
| [gp/moments.md](gp/moments.md) | Subsystem detail — kernels, solvers, guides, covariance representations |
| [gp/state_space.md](gp/state_space.md) | Subsystem detail — temporal inference, Kalman, Bayes-Newton |
| [gp/integration.md](gp/integration.md) | Subsystem detail — Gaussian expectations, quadrature methods |

---

*For usage patterns, see [../examples/](../examples/) — organized by layer with subsystem deep-dives.*
