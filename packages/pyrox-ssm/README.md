# pyrox-ssm

State-space models as NumPyro models for JAX: temporal GPs and Gaussian state-space models whose inference recurses in time.
It builds on [`pyrox`](../pyrox) and [gaussx](https://github.com/jejjohnson/gaussx) (Kalman filtering and smoothing, Gaussian algebra).

The boundary between the pyrox packages is the inference: [`pyrox-gp`](../pyrox-gp) solves over all N points at once, [`pyrox-lgm`](../pyrox-lgm) solves one sparse precision, and pyrox-ssm recurses in time.
This package imports nothing from pyrox-gp, pyrox-lgm or pyrox-nn.

## What's inside

- **Markov GPs**: `MarkovGPPrior`, `ConditionedMarkovGP`, `markov_gp_factor`, `markov_gp_sample`. Any SDE kernel over a sorted time grid; the Kalman filter gives the log marginal likelihood and the RTS smoother the posterior, both in O(N d³).
- **Sparse Markov GPs**: `SparseMarkovGPPrior`, `SparseConditionedMarkovGP`, `sparse_markov_elbo`, `sparse_markov_factor`.
- **Non-Gaussian likelihoods**: `LaplaceMarkovInference`, `GaussNewtonMarkovInference`, `PosteriorLinearizationMarkov`, `ExpectationPropagationMarkov`, each returning a `NonGaussConditionedMarkovGP`.
- **SDE kernels**, re-exported from gaussx: `MaternSDE`, `SumSDE`, `ProductSDE`, `PeriodicSDE`, `QuasiPeriodicSDE`, `CosineSDE`, `ConstantSDE`, `IntegratedWienerSDE`, `SDEKernel`, `SDEParams`.
- **Normalizing Kalman filter** (`[flows]` extra): `pyrox_ssm.flows.NormalizingKalmanPrior`, `normalizing_kalman_factor`.

These names moved here from pyrox-gp unchanged; `from pyrox_gp import MarkovGPPrior` now raises an `ImportError` naming `pyrox_ssm`. Likelihoods and guides come from `pyrox.likelihoods` or pyrox-gp.

```python
import jax.numpy as jnp
from pyrox_ssm import MarkovGPPrior, MaternSDE

times = jnp.linspace(0.0, 10.0, 200)
y = jnp.sin(times)
prior = MarkovGPPrior(MaternSDE(variance=1.0, lengthscale=0.5, order=1), times)
log_marginal = prior.log_marginal(y, jnp.asarray(0.01))
mean, var = prior.condition(y, jnp.asarray(0.01)).predict(times)
```

## Install

pyrox-ssm is not on PyPI yet; install it from GitHub with uv:

```bash
uv add "pyrox-ssm @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-ssm"
# normalizing-flow observation warps:
uv add "pyrox-ssm[flows] @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-ssm"
```

Until kernellib pins gaussx v0.6.15, the project also needs the gaussx override described in the [root README](../../README.md#installation).
