# GP API

The full GP stack: kernel *math functions*, concrete `Parameterized`
kernel classes, model-facing entry points (`GPPrior`, `ConditionedGP`,
`gp_factor`, `gp_sample`), sparse variational GPs with inter-domain
inducing features, variational guides and likelihoods, non-Gaussian
inference strategies (Laplace, Gauss-Newton, EP, posterior
linearization, quasi-Newton — dense and Markov), pathwise (Matheron)
posterior samplers, state-space (Kalman) GPs, and multi-output kernels.
Scalable matrix construction and solver strategies (numerically stable
assembly, implicit operators, batched matvec,
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
from pyrox_gp import GPPrior, RBF, gp_factor, gp_sample

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

::: pyrox_gp.GPPrior
::: pyrox_gp.ConditionedGP
::: pyrox_gp.gp_factor
::: pyrox_gp.gp_sample

## Concrete kernels

Each `Parameterized` kernel registers its hyperparameters with positivity
constraints where appropriate. Attach priors with `set_prior`, autoguides
with `autoguide`, and flip `set_mode("model" | "guide")`.

::: pyrox_gp.RBF
::: pyrox_gp.Matern
::: pyrox_gp.Periodic
::: pyrox_gp.Linear
::: pyrox_gp.RationalQuadratic
::: pyrox_gp.Polynomial
::: pyrox_gp.Cosine
::: pyrox_gp.White
::: pyrox_gp.Constant

## Sparse-GP inducing features (#49)

Inter-domain inducing-feature families used to build scalable sparse GPs
where the inducing-prior covariance ``K_uu`` becomes diagonal. Pass any
of these to `SparseGPPrior` via the ``inducing=`` keyword in
place of a raw point matrix ``Z``.

```python
from pyrox_gp import RBF, FourierInducingFeatures, SparseGPPrior

kernel   = RBF(init_lengthscale=0.3, init_variance=1.0)
features = FourierInducingFeatures.init(in_features=1, num_basis_per_dim=64, L=5.0)
prior    = SparseGPPrior(kernel=kernel, inducing=features)   # K_uu is diagonal!
```

::: pyrox_gp.InducingFeatures
::: pyrox_gp.FourierInducingFeatures
::: pyrox_gp.SphericalHarmonicInducingFeatures
::: pyrox_gp.SlepianInducingFeatures
::: pyrox_gp.LaplacianInducingFeatures
::: pyrox_gp.DecoupledInducingFeatures
::: pyrox_gp.funk_hecke_coefficients

## Sparse GP prior

::: pyrox_gp.SparseGPPrior

## Variational guides

Variational families `q(u)` over the inducing values of a
`SparseGPPrior`. All five expose the same building-block interface —
`sample(key)`, `log_prob(u)`, `kl_divergence(prior_cov)`, and
`predict(K_xz, K_zz_op, K_xx_diag)` — so they swap freely inside the
SVGP ELBO. `WhitenedGuide` parameterizes in whitened coordinates
`u = L_zz v` (the standard choice for stable optimization);
`NaturalGuide` parameterizes in natural form for natural-gradient / CVI
workflows; `DeltaGuide` is a point mass for MAP-style training.

::: pyrox_gp.FullRankGuide
::: pyrox_gp.MeanFieldGuide
::: pyrox_gp.WhitenedGuide
::: pyrox_gp.NaturalGuide
::: pyrox_gp.DeltaGuide

## Likelihoods

Observation models for latent-GP workflows. Each maps latent function
values to a summed log-density `log_prob(f, y)`; `DistLikelihood` wraps
any `numpyro.distributions.Distribution` factory for one-off models.

::: pyrox_gp.GaussianLikelihood
::: pyrox_gp.HeteroscedasticGaussianLikelihood
::: pyrox_gp.BernoulliLikelihood
::: pyrox_gp.PoissonLikelihood
::: pyrox_gp.SoftmaxLikelihood
::: pyrox_gp.StudentTLikelihood
::: pyrox_gp.DistLikelihood

### Warped (transformed-GP) likelihood

A transformed Gaussian process (Maronas et al., AISTATS 2021) is an SVGP
whose likelihood composes an elementwise monotone warp `G`; the warp
never appears in the KL term, so every existing inference path composes
with it unchanged. Needs an integrator (Gauss-Hermite recommended) and,
for the smooth recommended warp `MixtureGaussianCDF`, the
`pyrox-gp[flows]` extra.

::: pyrox_gp.WarpedGaussianLikelihood
::: pyrox_gp.warped_predictive_moments

## SVGP inference

The structured SVGP ELBO as a differentiable scalar (`svgp_elbo`), its
NumPyro registration (`svgp_factor`), and the natural-gradient / CVI
update loop (`ConjugateVI`) that exploits the `NaturalGuide`
parameterization for conjugate-style coordinate ascent.

::: pyrox_gp.svgp_elbo
::: pyrox_gp.svgp_factor
::: pyrox_gp.ConjugateVI

## Non-Gaussian inference strategies

Site-based Gaussian approximations `q(f) = N(m, V)` of the posterior
under a non-conjugate likelihood. All five strategies share the same
diagonal-site view and differ only in where the per-site curvature
comes from: exact Hessian at the mode (Laplace), PSD-projected
Gauss-Newton curvature, statistical linearization under the cavity
(posterior linearization / CVI), moment matching against the tilted
distribution (EP), or L-BFGS to the MAP with a Laplace covariance at
convergence (quasi-Newton). Each `fit(prior, likelihood, y)` returns a
`NonGaussConditionedGP` that quacks like `ConditionedGP`.

::: pyrox_gp.LaplaceInference
::: pyrox_gp.GaussNewtonInference
::: pyrox_gp.PosteriorLinearization
::: pyrox_gp.ExpectationPropagation
::: pyrox_gp.QuasiNewtonInference
::: pyrox_gp.NonGaussConditionedGP

## Multi-output GPs

Vector-valued GPs via coregionalization: the linear model of
coregionalization (LMC) mixes `Q` independent latent processes through
a learned matrix, the intrinsic coregionalization model (ICM) shares
one latent kernel, and the orthogonal instantaneous linear mixing model
(OILMM) constrains the mixing to be orthogonal so inference decouples
per latent process (the projections delegate to `gaussx.oilmm_project`
/ `gaussx.oilmm_back_project`).

::: pyrox_gp.LMCKernel
::: pyrox_gp.ICMKernel
::: pyrox_gp.OILMMKernel
::: pyrox_gp.MultiOutputInducingVariables
::: pyrox_gp.SharedInducingPoints

## Latent factor

Collapsed latent-factor regression: the linear decoder (mixing matrix)
carries a fixed unit-normal prior and is marginalized analytically, so
the likelihood factorizes only a `Q x Q` capacitance matrix and costs
`O(NQP)` in the output dimension. Contrast the coregionalization
kernels above, which hold the mixing matrix as a concrete array.

::: pyrox_gp.LatentFactorGPPrior
::: pyrox_gp.ConditionedLatentFactorGP
::: pyrox_gp.lfr_model
::: pyrox_gp.lfr_factor
::: pyrox_gp.latent_total_correlation
::: pyrox_gp.collapsed_lfr_log_prob
::: pyrox_gp.decoder_posterior
::: pyrox_gp.lfr_predictive_moments
::: pyrox_gp.warp_to_base
::: pyrox_gp.warped_lfr_log_prob
::: pyrox_gp.warped_decoder_posterior

## Pathwise posterior samplers (#39)

Callable posterior function draws via Matheron's rule. Each sampled
path is a `PathwiseFunction` that evaluates in
``O(N_* · F · D + N_* · N_corr)`` per path — ``N_* · F · D`` for the
RFF prior draw and ``N_* · N_corr`` for the kernel correction against
the ``N_corr`` training (exact) or inducing (decoupled) points — so the
same draw can be reused at arbitrary test sets without rebuilding a
test-set covariance. Standard enabler for Thompson sampling, Bayesian
optimization, and posterior visualization.

```python
from pyrox_gp import (
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

::: pyrox_gp.PathwiseSampler
::: pyrox_gp.DecoupledPathwiseSampler
::: pyrox_gp.PathwiseFunction

## State-space (SDE) kernels

Stationary 1-D kernels expressed as linear time-invariant SDEs. Once in
state-space form, GP inference on a 1-D grid reduces to Kalman filtering
in ``O(N d^3)`` instead of ``O(N^3)`` Cholesky. The protocol exposes
``sde_params() -> (F, L, H, Q_c, P_inf)`` and ``discretise(dt) -> (A_k, Q_k)``
for downstream Kalman / RTS use.

```python
import jax.numpy as jnp
from pyrox_gp import (
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

::: pyrox_gp.SDEKernel
::: pyrox_gp.SDEParams
::: pyrox_gp.MaternSDE
::: pyrox_gp.ConstantSDE
::: pyrox_gp.CosineSDE
::: pyrox_gp.PeriodicSDE
::: pyrox_gp.SumSDE
::: pyrox_gp.ProductSDE
::: pyrox_gp.QuasiPeriodicSDE

## Markov GP — Kalman / RTS workflow

`MarkovGPPrior` consumes any [`SDEKernel`][pyrox_gp.SDEKernel] over a sorted
1-D grid and gives `O(N d^3)` marginal likelihood (forward Kalman filter)
and posterior smoothing (backward RTS), where `d` is the SDE state
dimension. Use it for temporal GP regression / forecasting when the
training grid lives on a single time axis. Predictions at arbitrary
test times — including forecasting, backcasting, and within-window
interpolation — re-run the filter+smoother over the merged grid with the
test points masked out of the update step.

```python
import jax.numpy as jnp
from pyrox_gp import MaternSDE, MarkovGPPrior, markov_gp_factor

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
from pyrox_gp import MarkovGPPrior, MaternSDE, markov_gp_factor

def temporal_model(times, y):
    sigma2 = numpyro.sample("variance",  dist.LogNormal(0.0, 1.0))
    ell    = numpyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))
    sde    = MaternSDE(variance=sigma2, lengthscale=ell, order=1)
    prior  = MarkovGPPrior(sde, times)
    markov_gp_factor("obs", prior, y, jnp.array(0.01))
```

For non-Gaussian likelihoods on the Markov path, see the
[Markov non-Gaussian strategies](#non-gaussian-inference-markov) below;
for inducing-grid scalability, the [sparse Markov GP](#sparse-markov-gp).

::: pyrox_gp.MarkovGPPrior
::: pyrox_gp.ConditionedMarkovGP
::: pyrox_gp.markov_gp_factor
::: pyrox_gp.markov_gp_sample

## Normalizing Kalman Filter

`NormalizingKalmanPrior` wraps a `gaussx.LGSSM` (or `gaussx.MaskedLGSSM`)
base and an optional per-timestep observation warp into the same
model surface: exact collapsed marginal via `log_marginal`, NumPyro
registration via `normalizing_kalman_factor`, and observation-space
predictive moments via `predict` (RTS smoothing followed by a
Gauss-Hermite pushforward through the warp — the mean is `E[G(z)]`,
not `G(E[z])`).

The unwarped model (`warp=None`) works with the base install — `LGSSM`
is a hard dependency. **Passing a `warp` requires the `flows` extra**:
`pip install 'pyrox-gp[flows]'`. Because the warp acts on observations,
the log-det term is independent of the latent state, the Kalman
recursion stays exact, and none of the non-Gaussian Markov strategies
below are involved.

```python
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from gaussx import LGSSM
from pyrox_gp import NormalizingKalmanPrior, normalizing_kalman_factor

def nkf_model(y, warp=None, mask=None):
    T, M = y.shape
    log_q = numpyro.sample("log_q", dist.Normal(0.0, 1.0).expand([M]).to_event(1))
    log_r = numpyro.sample("log_r", dist.Normal(0.0, 1.0).expand([M]).to_event(1))
    base = LGSSM(0.9 * jnp.eye(M), jnp.eye(M),
                 jnp.diag(jnp.exp(jnp.asarray(log_q))),
                 jnp.diag(jnp.exp(jnp.asarray(log_r))),
                 jnp.zeros(M), jnp.eye(M), n_steps=T)
    prior = NormalizingKalmanPrior(base, warp=warp)
    normalizing_kalman_factor("nkf", prior, y, mask)
```

::: pyrox_gp.NormalizingKalmanPrior
::: pyrox_gp.normalizing_kalman_factor

## Non-Gaussian inference (Markov)

The Markov-aware counterparts of the site-based strategies above: same
diagonal-site math, but the global posterior recomputation runs through
the Kalman filter / RTS smoother in `O(N d^3)` instead of a dense
`O(N^3)` solve. Each `fit(prior, likelihood, y)` returns a
`NonGaussConditionedMarkovGP` with the same `predict` API as the
Gaussian-likelihood `ConditionedMarkovGP`.

::: pyrox_gp.LaplaceMarkovInference
::: pyrox_gp.GaussNewtonMarkovInference
::: pyrox_gp.PosteriorLinearizationMarkov
::: pyrox_gp.ExpectationPropagationMarkov
::: pyrox_gp.NonGaussConditionedMarkovGP

## Sparse Markov GP

Sparse variational GP over an SDE kernel and an inducing *time* grid:
the variational family lives on the inducing times while predictions
exploit the Markov structure between them.

::: pyrox_gp.SparseMarkovGPPrior
::: pyrox_gp.SparseConditionedMarkovGP
::: pyrox_gp.sparse_markov_elbo
::: pyrox_gp.sparse_markov_factor

## Component protocols

Abstract pyrox-local bases for the orthogonal component stack — the
contracts that the concrete kernels, guides, and likelihoods above
implement. Cubature integrators (Gauss-Hermite,
Monte Carlo) come from `gaussx.AbstractIntegrator` and its concrete
subclasses; solver strategies live in
[`gaussx`](https://github.com/jejjohnson/gaussx).

::: pyrox_gp.Kernel
::: pyrox_gp.Guide
::: pyrox_gp.Likelihood

## Math primitives

Pure JAX kernel functions. Stateless, differentiable, composable —
``(Array, ..., hyperparams) -> Gram``. No NumPyro, no protocols.

::: pyrox_gp._src.kernels
    options:
      show_root_heading: false
