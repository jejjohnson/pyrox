# GP API

The full GP stack: kernel *math functions*, concrete `Parameterized`
kernel classes, model-facing entry points (`GPPrior`, `ConditionedGP`,
`gp_factor`, `gp_sample`), sparse variational GPs with inter-domain
inducing features, variational guides and likelihoods, non-Gaussian
inference strategies (Laplace, Gauss-Newton, EP, posterior
linearization, quasi-Newton), pathwise (Matheron)
posterior samplers, and multi-output kernels.
Scalable matrix construction and solver strategies (numerically stable
assembly, implicit operators, batched matvec,
Cholesky / CG / BBMM / LSMR / SLQ) live in
[`gaussx`](https://github.com/jejjohnson/gaussx).

Temporal GPs on a 1-D time grid (the SDE kernels, `MarkovGPPrior`, the
Kalman-based non-Gaussian and sparse Markov strategies, the normalizing
Kalman filter) live in pyrox-ssm: see the [SSM API](ssm.md).

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

`prior.log_marginal(y, noise_var)` returns the collapsed log marginal likelihood that `gp_factor` registers, for use outside a NumPyro model (an optimiser's loss, a model comparison); `MultiOutputGPPrior.log_marginal(Y, noise_var)` is the same for `mo_gp_factor`, Kronecker-exact where the Gram allows, and `OILMMGPPrior.log_marginal(Y, noise_var)` is the same for `oilmm_factor`, exact at the projected cost under isotropic noise.

For a non-Gaussian likelihood, `gp_laplace_factor(name, prior, likelihood, y)` registers the Laplace approximate evidence instead, the `log_marginal_approx` of a `LaplaceInference` (or `GaussNewtonInference`) fit. The mode is implicitly differentiable, so NUTS or SVI can run over the kernel hyperparameters with the latent function marginalised, as `pyrox_ssm.markov_laplace_factor` does on a Markov prior.

The conditioned results (`ConditionedGP`, `ConditionedMultiOutputGP`, `ConditionedOILMMGP`) predict the latent function by default; `predict(X_star, include_noise=True)` adds the noise variance they were conditioned on, for a band on a new observation.

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

The solver, the preconditioner and the log-determinant are three separate fields of `GPPrior` (and of `SparseGPPrior`, `MultiOutputGPPrior` and `OILMMGPPrior`), so each can be changed on its own. `GPPrior.solver_strategy()` composes them with gaussx's `compose_solver`, and every quantity of the prior uses the result:

| Quantity | Solve | Log-determinant |
| --- | --- | --- |
| `log_marginal`, `gp_factor`, `log_prob` | yes | yes |
| `condition`, `predict` | yes | no |
| pathwise sampling | yes | no |

```python
import gaussx as gx

preconditioner = gx.NystromPreconditioner.lazy(rank=300, key=key)
# BBMM with a preconditioner, which its shared pass also uses for the log-determinant
prior = GPPrior(kernel, X, solver=gx.BBMMSolver(num_probes=50), preconditioner=preconditioner, matrix_free=True)
# CG with the preconditioner, plus a separate stochastic log-determinant
prior = GPPrior(kernel, X, solver=gx.CGSolver(rtol=1e-4), preconditioner=preconditioner, logdet=gx.SLQLogdet(num_probes=50), matrix_free=True)
```

`preconditioner` and `logdet` are keyword-only. A combination that cannot work, such as a preconditioner with the default `DenseSolver`, raises `ValueError` at construction, naming the three fields. A stochastic log-determinant without a key reuses its probes at every call, so NUTS then targets a fixed approximation of the posterior; `gx.KeyedSolver` binds a key per step.

For a large exact GP there are two presets, each one call that returns an ordinary gaussx strategy. `matfree_solver` is lineax conjugate gradients preconditioned by a low-rank approximation of $K$ plus a log-determinant through matfree: `logdet="nystrom"` (the default) is gaussx's `NystromLogdet`, the exact log-determinant of a Nyström preconditioner plus stochastic Lanczos quadrature on the preconditioned remainder, and `logdet="slq"` is stochastic Lanczos quadrature on the system itself. The library matfree has no conjugate gradients of its own, so "matfree" means lineax CG plus the matfree-backed log-determinant. `bbmm_solver` is gaussx's `BBMMSolver` with the same preconditioner, whose one batched pass gives both the solve and the log-determinant from probes drawn from the preconditioner. Both read the diagonal from the system with gaussx's `split_diagonal`: the scalar noise of a `matrix_free=True` system, or a diagonal that varies by point, such as site variances or heteroscedastic noise, for which the preconditioner is built by gaussx's per-point shift. `preconditioner="fitc"` is gaussx's `FITCPreconditioner`, the Nyström approximation at `rank` training inputs drawn by the `key` plus the prior variance it misses on the diagonal, so that $P$ matches $A$ there; it reads the kernel part and the diagonal from the system, evaluating only the inducing blocks and the diagonal of a `matrix_free=True` prior. To choose the inducing inputs yourself, pass `gaussx.FITCPreconditioner.from_kernel(...)` as the prior's `preconditioner`. From gaussx v0.8.4 the gradient of BBMM's log-determinant, and of a `gaussx.SLQLogdet` given a preconditioner, carries $P$ as a control variate ([gaussx#826](https://github.com/jejjohnson/gaussx/issues/826)). `preconditioned_cg_solver` is a deprecated alias of `matfree_solver` with its old defaults, kept for one release; its `shift=` is used only for a system that is one dense matrix.

Which of these to pick for a given size and kernel, with the errors and times measured on both likelihoods, is the guide [choosing the solver, preconditioner and log-determinant](../../gp-solver-choices/).

::: pyrox_gp.GPPrior
::: pyrox_gp.ConditionedGP
::: pyrox_gp.gp_factor
::: pyrox_gp.gp_laplace_factor
::: pyrox_gp.gp_sample
::: pyrox_gp.matfree_solver
::: pyrox_gp.bbmm_solver
::: pyrox_gp.preconditioned_cg_solver

## Derivatives

The gradient of a GP is a GP, so a `ConditionedGP` also gives the exact
Gaussian posterior over $\nabla f(x_*)$, reusing the cached training solve
(no re-solve). With $\hat K = K(X, X) + (\text{jitter} + \sigma^2) I$,
$\alpha = \hat K^{-1}(y - m(X))$ and
$G(x_*) = \partial_1 K(x_*, X) \in \mathbb{R}^{D \times N}$:

$$
\mathbb{E}[\nabla f(x_*) \mid y] = \nabla m(x_*) + G(x_*)\,\alpha ,
\qquad
\operatorname{Cov}[\nabla f \mid y]
= \nabla_x \nabla_{x'}^\top k(X_*, X_*) - G\,\hat K^{-1} G^\top .
$$

The mean is the derivative of the predictive mean used for sensitivity maps
by Johnson et al. (2020, §3); the covariance follows from the joint Gaussian
of values and derivatives (Solak et al. 2003; Rasmussen & Williams 2006,
§9.4). The derivative cross-covariances are kernellib's `Derivative` /
`DerivativeIndexed` kernels; the solves go through gaussx (triangular solves
with the cached Cholesky factor, or the prior's `solver` on the matrix-free
operator).

```text
k    = freeze(kernel)                          # one draw per call
G    = [Derivative(k, dx=d)(X*, X) for d]      # (D*M, N), kind-major rows
mean = (G @ alpha) as (M, D) + grad m(X*)
G K^-1 G^T = P^T Q                             # P = Q = L^-1 G^T with a factor
cov  = H - P^T Q                               # marginal: (M, D, D) blocks only
```

- `predict_gradient(X_star, cov=...)` returns a `GradientPosterior`
  (`mean`, `cov`); `cov="marginal"` (default) gives the per-point
  `(M, D, D)` blocks, `"full"` the joint `(M, D, M, D)`, `"none"` skips the
  covariance.
- `predict_var_gradient(X_star)` gives $\nabla_x \sigma^2_*(x)$, the
  acquisition gradient for uncertainty sampling.
- `predict_mean_hessian(X_star)` gives $\nabla \nabla^\top \mu_*(x)$,
  `(M, D, D)`.

**Layout.** Internally the derivative rows are *kind-major*, as
`kernellib.derivative_inputs(X_star, values=False)` builds them: row
`d * M + m` is $\partial_d f(x_{*,m})$. The outputs are *point-major*:
`mean[m, d]`, `cov[m, i, j]` (marginal) and `cov[a, i, b, j]` (full).
Flatten the joint with `einx.id("a i b j -> (a i) (b j)", cov)`.

```python
import jax.numpy as jnp
import pyrox_gp as px

cond = px.GPPrior(px.RBF(init_lengthscale=0.5), X).condition(y, jnp.array(0.01))
post = cond.predict_gradient(X_star)            # X_star: (M, D)
sensitivity = jnp.sqrt(jnp.sum(post.mean**2, axis=-1))       # (M,) |E grad f|
std = jnp.sqrt(jnp.diagonal(post.cov, axis1=-2, axis2=-1))   # (M, D)
step = cond.predict_var_gradient(X_star)        # (M, D) uncertainty ascent
```

The kernel must be mean-square differentiable (`RBF`, `RationalQuadratic`,
`Matern(nu >= 1.5)`, `Periodic`); `Matern(nu=0.5)` raises kernellib's
`ValueError`. `predict_mean_hessian` with `Matern(nu=1.5)` is defined only
away from the training inputs.

References: Johnson, Laparra, Pérez-Suay, Mahecha & Camps-Valls (2020),
*Kernel methods and their derivatives: Concept and perspectives for the
earth system sciences*, PLOS ONE 15(10): e0235885,
[doi:10.1371/journal.pone.0235885](https://doi.org/10.1371/journal.pone.0235885);
Solak, Murray-Smith, Leithead, Leith & Rasmussen (2003), *Derivative
observations in Gaussian process models of dynamic systems*, NeurIPS 15;
Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*, §9.4.

::: pyrox_gp.GradientPosterior

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

## Inducing-point initialisation

Pick `Z` as a subset of the training inputs. `"rpcholesky"` (the default)
targets the trace error that bounds the SVGP ELBO gap (Burt et al. 2020),
so clusters of rare points still get inducing points.

```python
from pyrox_gp import RBF, SparseGPPrior, init_inducing

kernel = RBF(init_lengthscale=0.3)
Z      = init_inducing(X, 64, kernel=kernel, key=jax.random.key(0))
prior  = SparseGPPrior(kernel=kernel, Z=Z)
```

::: pyrox_gp.init_inducing

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

After fitting a guide over the inducing values (`svgp_factor`, `svgp_elbo` or `ConjugateVI`), `prior.condition(guide)` returns the `ConditionedSparseGP`, whose `predict(X_star)` gives the SVGP predictive mean and variance, mean function included. `MultiOutputSparseGPPrior.condition(guide)` does the same for the multi-output prior, in the `(M, P)` layout, and replaces its deprecated `predict(guide, X_star)`.

::: pyrox_gp.SparseGPPrior
::: pyrox_gp.ConditionedSparseGP

## Variational guides

Variational families `q(u)` over the inducing values of a
`SparseGPPrior`. All five expose the same building-block interface —
`sample(key)`, `log_prob(u)`, `kl_divergence(prior_cov)`, and
`predict(K_xz, K_zz_op, K_xx_diag)` — so they swap freely inside the
SVGP ELBO. `WhitenedGuide` parameterizes in whitened coordinates
`u = L_zz v` (the standard choice for stable optimization);
`NaturalGuide` parameterizes in natural form for natural-gradient / CVI
workflows; `DeltaGuide` is a point mass for MAP-style training.

A guide's `solver` is the solver of every solve against $K_{ZZ}$ in `predict` and `kl_divergence` for `FullRankGuide`, `MeanFieldGuide` and `DeltaGuide` ([#560](https://github.com/jejjohnson/pyrox/issues/560)): `predict` forms $B = K_{ZZ}^{-1} K_{ZX}$ from one matrix right-hand side and needs no root, and the KL of the first two is `gaussx.gaussian_kl(..., solver=)` (`DeltaGuide`'s $-\log p(\mathrm{loc})$ already went through `gaussx.gaussian_log_prob` with it). With `solver=None` both keep the Cholesky route, and a diagonal $K_{ZZ}$ of inter-domain features stays $O(M)$; an explicit solver overrides that structure. An iterative solver's tolerance enters the ELBO, since a relative error $\delta$ in $B$ moves each predictive variance by about $\delta k_{nn}$, and its own log-determinant (stochastic Lanczos for CG and BBMM) moves the KL by half its error, so pass `logdet=gaussx.DenseLogdet()` through `gaussx.compose_solver` for an exact KL at moderate $M$. `WhitenedGuide` has nothing to choose there: $u = L_{ZZ} v$ needs the Cholesky factor, and its KL is against $\mathcal{N}(0, I)$. `ConjugateVI.step` solves against $K_{ZZ}$ with the prior's `solver` when it sets one.

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

The warp may also be **input-dependent** (`G_{phi(x)}`) — a
non-stationary process from a stationary kernel. Pass a conditional
bijection (e.g. `gauss_flows.Conditioner` wrapping an unconditional
warp); `svgp_elbo` threads the inputs through
`Likelihood.log_prob(f, y, X)` automatically. Cost note: the expected
log-likelihood evaluates the warp at every quadrature node, so a
conditional warp pays `order` conditioner forward passes per point.

Only `svgp_elbo` threads the inputs through. The CVI / natural-gradient
strategies, `pyrox_ssm.sparse_markov_elbo` and the multi-output ELBO do not,
so a conditional warp raises there rather than silently dropping the
condition — use an unconditional warp on those paths.

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
`ConditionedNonGaussGP` that quacks like `ConditionedGP`.

The Laplace and Gauss-Newton loops run on `argminax.solve.laplace_mode_covariance`, the quasi-Newton one on `optimistix.LBFGS`, and the posterior-linearisation and EP loops on `argminax.solve.site_fixed_point`, so all five fits run under `jax.jit` and `jax.vmap` (over datasets, say) and their `n_iter` and `converged` are arrays. Close over the prior rather than passing it through the transform, since its kernel's priors live in a registry that a rebuilt module loses.

`PosteriorLinearization` and `ExpectationPropagation` build the posterior of their sites with `gaussx.site_posterior` and report `gaussx.site_log_evidence`, the Laplace-type evidence at the posterior mean (not the EP evidence). They take `solver`, `preconditioner` and `logdet`, and each one left `None` comes from the prior, so they work on a `matrix_free=True` prior with an iterative solver. Unlike Laplace, they need the posterior marginal variances at every iteration, because each cavity removes its site from its marginal: with a dense solver that is one Cholesky of $K + \mathrm{diag}(1 / \Lambda)$ per iteration, and with an iterative one it is $N$ solves per iteration, `batch_size` at a time, so the exact route suits a moderate $N$. For a large $N$ they take the stochastic variances of gaussx's `SitePosterior.variance_and_error` ([gaussx#827](https://github.com/jejjohnson/gaussx/issues/827)), `num_probes` solves per iteration whatever $N$: `variance="stochastic"`, or `variance="auto"` with an iterative solver above $N = 2048$. The `key` of `fit` is then required, and one key serves every iteration, so the update is a fixed function of the sites and `converged` refers to that surrogate. Each estimate is clamped for its cavity to at most $(1 - 10^{-3}) / \lambda_n$, so that every cavity precision stays positive, and to at least $10^{-6} K_{nn}$. The `estimator` defaults to SBPV, which draws from the prior and the sites, on a dense prior, and to SPV, which needs only products with $K$, on a matrix-free one, since SBPV would densify the prior for its draws. SPV is much the noisier: on a 200-point Bernoulli problem with 200 probes its standard error was about 4.7 times the variance it estimates, against 0.67 for SBPV, so it needs many more probes. CG's own log-determinant is stochastic Lanczos, so pass `logdet=gaussx.DenseLogdet()` beside it for an exact evidence at moderate $N$.

`LaplaceInference` and `GaussNewtonInference` take the same `solver`, `preconditioner` and `logdet`, each falling back to the prior's ([#551](https://github.com/jejjohnson/pyrox/issues/551)). Every Newton step solves against $A = K + \mathrm{diag}(1 / \Lambda)$ through that solver on the prior's own operator, so a `matrix_free=True` prior with CG or BBMM never forms the $N \times N$ kernel matrix, and unlike the cavity loops they need no marginal variances until `q_var` is read. Their `variance`, `estimator`, `num_probes` and `batch_size` mean what they mean for the cavity loops, and only `q_var` uses them: a stochastic `q_var` of a fit given no `key` raises. Give the solver a tolerance at least ten times tighter than the strategy's `tol`; an inner solve that misses its tolerance makes `converged` false, and inside a sampler the solver should be built with `throw=False`. With a stochastic log-determinant the evidence is an estimate whose probes come from the `key` of `fit`, `condition_nongauss` or `gp_laplace_factor`. `QuasiNewtonInference` still raises a `ValueError` on a matrix-free prior, since it whitens its search with a Cholesky root of the kernel matrix.

Every fit keeps its posterior as a `gaussx.SitePosterior` (`site_posterior`), with `q_mean`, `site_nat1` and `site_nat2` read from it, and predicts from its weights with the hyperparameters the fit used: a kernel with priors is drawn once per fit, and `predict` registers no new site (before #551 it drew again). Refit to resample.

::: pyrox_gp.LaplaceInference
::: pyrox_gp.GaussNewtonInference
::: pyrox_gp.PosteriorLinearization
::: pyrox_gp.ExpectationPropagation
::: pyrox_gp.QuasiNewtonInference
::: pyrox_gp.ConditionedNonGaussGP

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
::: pyrox_gp.oilmm_factor
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
::: pyrox_gp.latent_init
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

The pure JAX kernel functions (`rbf_kernel`, `matern_kernel`, ...) live in
kernellib's functional API: `from kernellib.functional import rbf_kernel`.
Stateless, differentiable, composable — `(Array, ..., hyperparams) -> Gram`.
See the [kernellib documentation](https://github.com/jejjohnson/kernellib)
for the reference.
