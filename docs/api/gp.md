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

## Sparse-GP inducing features (#49)

Inter-domain inducing-feature families used to build scalable sparse GPs
where the inducing-prior covariance ``K_uu`` becomes diagonal. Pass any
of these to :class:`SparseGPPrior` via the ``inducing=`` keyword in
place of a raw point matrix ``Z``.

```python
from pyrox.gp import RBF, FourierInducingFeatures, SparseGPPrior

kernel   = RBF(init_lengthscale=0.3, init_variance=1.0)
features = FourierInducingFeatures.init(in_features=1, num_basis_per_dim=64, L=5.0)
prior    = SparseGPPrior(kernel=kernel, inducing=features)   # K_uu is diagonal!
```

::: pyrox.gp.InducingFeatures
::: pyrox.gp.FourierInducingFeatures
::: pyrox.gp.SphericalHarmonicInducingFeatures
::: pyrox.gp.LaplacianInducingFeatures
::: pyrox.gp.DecoupledInducingFeatures
::: pyrox.gp.funk_hecke_coefficients

## Sparse GP prior

::: pyrox.gp.SparseGPPrior

## Pathwise posterior samplers (#39)

Callable posterior function draws via Matheron's rule. Each sampled
path is a :class:`PathwiseFunction` that evaluates in
``O(N_* · F · D + N_* · N_corr)`` per path — ``N_* · F · D`` for the
RFF prior draw and ``N_* · N_corr`` for the kernel correction against
the ``N_corr`` training (exact) or inducing (decoupled) points — so the
same draw can be reused at arbitrary test sets without rebuilding a
test-set covariance. Standard enabler for Thompson sampling, Bayesian
optimization, and posterior visualization.

```python
from pyrox.gp import (
    RBF,
    GPPrior,
    PathwiseSampler,
    DecoupledPathwiseSampler,
    FullRankGuide,
    SparseGPPrior,
)
import jax
import jax.numpy as jnp

# Exact GP:
posterior = GPPrior(kernel=RBF(), X=X).condition(y, jnp.array(0.05))
paths = PathwiseSampler(posterior, n_features=512).sample_paths(
    jax.random.PRNGKey(0), n_paths=32
)
draws = paths(X_star)            # (32, N_star)

# Sparse / decoupled:
sparse  = SparseGPPrior(kernel=RBF(), Z=Z)
guide   = FullRankGuide.init(Z.shape[0])
paths   = DecoupledPathwiseSampler(sparse, guide).sample_paths(key, n_paths=16)
samples = paths(X_star)
```

Currently supports RBF and Matern kernels. Point-inducing
``SparseGPPrior`` only — inducing-feature priors raise at construction.

::: pyrox.gp.PathwiseSampler
::: pyrox.gp.DecoupledPathwiseSampler
::: pyrox.gp.PathwiseFunction

## State-space (SDE) kernels

Stationary 1-D kernels expressed as linear time-invariant SDEs. Once in
state-space form, GP inference on a 1-D grid reduces to Kalman filtering
in ``O(N d^3)`` instead of ``O(N^3)`` Cholesky. The protocol exposes
``sde_params() -> (F, L, H, Q_c, P_inf)`` and ``discretise(dt) -> (A_k, Q_k)``
for downstream Kalman / RTS use.

```python
import jax.numpy as jnp
from pyrox.gp import (
    ConstantSDE, CosineSDE, MaternSDE, PeriodicSDE,
    ProductSDE, QuasiPeriodicSDE, SumSDE,
)

# Primitive kernels
matern = MaternSDE(variance=1.0, lengthscale=0.5, order=1)  # nu = 3/2
cos    = CosineSDE(variance=1.0, frequency=2.0)
const  = ConstantSDE(variance=0.3)
per    = PeriodicSDE(variance=1.0, lengthscale=1.0, period=2.0, n_harmonics=7)

# Composition: trend + offset
trend = SumSDE((matern, const))                   # state dim = 2 + 1 = 3

# Composition: damped oscillation (Matern x Cosine)
damped = ProductSDE(matern, cos)                  # state dim = 2 * 2 = 4

# Quasi-periodic (Matern x Periodic) — convenience wrapper around ProductSDE
qp = QuasiPeriodicSDE(matern, per)                # state dim = 2 * 15 = 30
```

::: pyrox.gp.SDEKernel
::: pyrox.gp.MaternSDE
::: pyrox.gp.ConstantSDE
::: pyrox.gp.CosineSDE
::: pyrox.gp.PeriodicSDE
::: pyrox.gp.SumSDE
::: pyrox.gp.ProductSDE
::: pyrox.gp.QuasiPeriodicSDE

## Markov GP — Kalman / RTS workflow

`MarkovGPPrior` consumes any [`SDEKernel`][pyrox.gp.SDEKernel] over a sorted
1-D grid and gives `O(N d^3)` marginal likelihood (forward Kalman filter)
and posterior smoothing (backward RTS), where `d` is the SDE state
dimension. Use it for temporal GP regression / forecasting when the
training grid lives on a single time axis. Predictions at arbitrary
test times — including forecasting, backcasting, and within-window
interpolation — re-run the filter+smoother over the merged grid with the
test points masked out of the update step.

```python
import jax.numpy as jnp
from pyrox.gp import MaternSDE, MarkovGPPrior, markov_gp_factor

times = jnp.linspace(0.0, 5.0, 200)
y     = jnp.sin(times) + 0.05 * jnp.cos(7.0 * times)

prior = MarkovGPPrior(
    MaternSDE(variance=1.0, lengthscale=0.5, order=1),  # Matern-3/2
    times,
)
log_marg = prior.log_marginal(y, jnp.asarray(0.01))     # Kalman forward
cond     = prior.condition(y, jnp.asarray(0.01))        # filter + RTS smoother
mean, var = cond.predict(jnp.linspace(-0.5, 6.0, 50))   # arbitrary test times
```

Inside a NumPyro model, swap `gp_factor` for `markov_gp_factor`:

```python
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from pyrox.gp import MarkovGPPrior, MaternSDE, markov_gp_factor

def temporal_model(times, y):
    sigma2 = numpyro.sample("variance",  dist.LogNormal(0.0, 1.0))
    ell    = numpyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))
    sde    = MaternSDE(variance=sigma2, lengthscale=ell, order=1)
    prior  = MarkovGPPrior(sde, times)
    markov_gp_factor("obs", prior, y, jnp.array(0.01))
```

Currently scoped to Gaussian-likelihood regression on a single time axis.
Non-Gaussian likelihoods on top of the Markov path (CVI, EP) and
spatio-temporal Markov priors land in later waves.

::: pyrox.gp.MarkovGPPrior
::: pyrox.gp.ConditionedMarkovGP
::: pyrox.gp.markov_gp_factor
::: pyrox.gp.markov_gp_sample

## Component protocols

Abstract pyrox-local bases for the orthogonal component stack. Wave 2
ships only the contracts for `Guide` and `Likelihood` — concrete
implementations land in later waves. Cubature integrators (Gauss-Hermite,
Monte Carlo) come from `gaussx.AbstractIntegrator` and its concrete
subclasses; solver strategies live in
[`gaussx._strategies`](https://github.com/jejjohnson/gaussx).

::: pyrox.gp.Kernel
::: pyrox.gp.Guide
::: pyrox.gp.Likelihood

## Math primitives

Pure JAX kernel functions. Stateless, differentiable, composable —
``(Array, ..., hyperparams) -> Gram``. No NumPyro, no protocols.

::: pyrox.gp._src.kernels
    options:
      show_root_heading: false
