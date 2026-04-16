# GP API

Wave 2 ships the dense-GP foundation: kernel *math functions*, concrete
`Parameterized` kernel classes, abstract component protocols, and the
model-facing entry points (`GPPrior`, `ConditionedGP`, `gp_factor`,
`gp_sample`). Scalable matrix construction and solver strategies
(numerically stable assembly, implicit operators, batched matvec,
Cholesky / CG / BBMM / LSMR / SLQ) live in
[`gaussx`](https://github.com/jejjohnson/gaussx).

!!! note "Split with gaussx"
    pyrox owns the kernel *function* side — closed-form math primitives
    readable in a dozen lines — plus the NumPyro-aware model shell
    (`GPPrior`, `gp_factor`, `gp_sample`). `gaussx` owns every piece of
    linear algebra: stable matrix construction, solver strategies, and
    the underlying `MultivariateNormal` distribution. The model entry
    points accept any `gaussx.AbstractSolverStrategy` (default
    `gaussx.DenseSolver()`).

## Model entry points

```python
import jax.numpy as jnp
import numpyro
from pyrox.gp import GPPrior, RBF, gp_factor, gp_sample

def regression_model(X, y):
    """Collapsed Gaussian-likelihood GP regression."""
    kernel = RBF()
    prior = GPPrior(kernel=kernel, X=X)
    gp_factor("obs", prior, y, noise_var=jnp.array(0.05))


def latent_model(X):
    """Latent-function GP for non-conjugate likelihoods."""
    kernel = RBF()
    prior = GPPrior(kernel=kernel, X=X)
    f = gp_sample("f", prior)
    # ... attach any likelihood to f here, e.g. Bernoulli or Poisson.
```

Swap the solver strategy at construction time:

```python
from gaussx import CGSolver, ComposedSolver, DenseLogdet, DenseSolver
prior = GPPrior(kernel=RBF(), X=X, solver=CGSolver())
# Or compose — CG for solve, dense Cholesky for logdet:
prior = GPPrior(
    kernel=RBF(), X=X,
    solver=ComposedSolver(solve_strategy=CGSolver(), logdet_strategy=DenseLogdet()),
)
```

::: pyrox.gp.GPPrior
::: pyrox.gp.ConditionedGP
::: pyrox.gp.gp_factor
::: pyrox.gp.gp_sample

## Concrete kernels

Each `Parameterized` kernel registers its hyperparameters with positivity
constraints where appropriate. Attach priors with `set_prior`, autoguides
with `autoguide`, and flip `set_mode("model" | "guide")`.

::: pyrox.gp.RBF
::: pyrox.gp.Matern
::: pyrox.gp.Periodic
::: pyrox.gp.Linear
::: pyrox.gp.RationalQuadratic
::: pyrox.gp.Polynomial
::: pyrox.gp.Cosine
::: pyrox.gp.White
::: pyrox.gp.Constant

## Component protocols

Abstract pyrox-local bases for the orthogonal component stack. Wave 2
ships only the contracts for `Guide`, `Integrator`, and `Likelihood` —
concrete implementations land in later waves. Solver strategies live in
[`gaussx._strategies`](https://github.com/jejjohnson/gaussx).

::: pyrox.gp.Kernel
::: pyrox.gp.Guide
::: pyrox.gp.Integrator
::: pyrox.gp.Likelihood

## Math primitives

Pure JAX kernel functions. Stateless, differentiable, composable —
``(Array, ..., hyperparams) -> Gram``. No NumPyro, no protocols.

::: pyrox.gp._src.kernels
    options:
      show_root_heading: false
