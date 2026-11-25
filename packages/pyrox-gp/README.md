<div align="center">

<img alt="pyrox-gp" src="../../docs/assets/icon-gp.svg" width="120">

# pyrox-gp

**Gaussian processes for JAX and NumPyro: kernels with priors, exact, sparse, multi-output and non-Gaussian GPs, and pathwise posterior samples, on pyrox, pyrox-gauss, gaussx, kernellib, functax, lineax and flowjax.**

[![Tests](https://github.com/jejjohnson/pyrox/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/pyrox/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/pyrox/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/pyrox/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/pyrox/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/pyrox/actions/workflows/typecheck.yml)
[![Docs](https://github.com/jejjohnson/pyrox/actions/workflows/pages.yml/badge.svg)](https://jejjohnson.github.io/pyrox/)
[![codecov](https://codecov.io/gh/jejjohnson/pyrox/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/pyrox)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

📖 [Documentation](https://jejjohnson.github.io/pyrox/) ·
📚 [GP API](https://jejjohnson.github.io/pyrox/reference/gp/) ·
🚀 [Exact GP regression](https://jejjohnson.github.io/pyrox/exact-gp-regression/) ·
🗂️ [Capability index](https://jejjohnson.github.io/pyrox/capabilities/)

</div>

---

A Gaussian process is a prior over functions, and most of the work of using one is not the prior. The kernel's hyperparameters need priors of their own, the latent function has to be integrated out where the noise is Gaussian and approximated where it is not, large datasets need inducing points, and every solve and log-determinant has to stay stable. pyrox-gp writes each of these as a piece of a NumPyro model. Its kernels declare their hyperparameters as NumPyro sites with priors, and its GP priors add the latent function, so a model is the joint density

```math
p(\mathbf{y}, \mathbf{f}, \boldsymbol{\theta} \mid \mathbf{X}) = \prod_{n=1}^{N} p(y_n \mid f_n) \ \mathcal{N}(\mathbf{f} \mid \boldsymbol{\mu}(\mathbf{X}), \mathbf{K}_{\boldsymbol{\theta}}) \ p(\boldsymbol{\theta})
```

where $`\mathbf{X} \in \mathbb{R}^{N \times D_x}`$ holds the $`N`$ inputs, $`\mathbf{y} \in \mathbb{R}^{N}`$ the observations, $`\mathbf{f} = f(\mathbf{X}) \in \mathbb{R}^{N}`$ the latent function at the inputs, $`\boldsymbol{\mu}(\mathbf{X}) \in \mathbb{R}^{N}`$ the mean function, $`\mathbf{K}_{\boldsymbol{\theta}} = k_{\boldsymbol{\theta}}(\mathbf{X}, \mathbf{X}) \in \mathbb{R}^{N \times N}`$ the kernel's Gram matrix, $`\boldsymbol{\theta}`$ its hyperparameters with their prior $`p(\boldsymbol{\theta})`$, and $`p(y_n \mid f_n)`$ the likelihood of one observation. The kernel records $`\boldsymbol{\theta}`$ as sites such as `RBF.variance` and `RBF.lengthscale`, so NUTS, SVI and `Predictive` treat them as in any NumPyro model.

```python
# Shapes: N samples; D_x inputs; M inducing inputs; S paths
# (kernel, inputs, targets, likelihood, counts and the rest as in the quickstart)

# today (pyrox-gp)
# log p(y | X, θ) = log 𝒩(y; 0, K_θ + σ² I), f integrated out in closed form
gp_factor("y", GPPrior(kernel, inputs), targets, noise_var)
# today (pyrox-gp)
# p(f_* | y), the exact posterior at fixed θ
posterior = GPPrior(kernel, inputs).condition(targets, noise_var)
# today (pyrox-gp)
# S whole functions f⁽ˢ⁾(·) from that posterior, by Matheron's rule
paths = PathwiseSampler(posterior).sample_paths(key, n_paths=256)
# today (pyrox-gp)
# q(u) over M inducing values u = f(Z), for large N
q_u = ConjugateVI().step(
    SparseGPPrior(kernel, Z=inducing), guide, likelihood, inputs, targets
)
# today (pyrox-gp)
# q(f) ≈ p(f | y) for a likelihood that is not Gaussian
approx = GPPrior(kernel, inputs).condition_nongauss(
    likelihood, counts, strategy=strategy
)
```

None of it is new machinery. Each kernel is a [`pyrox`](../pyrox) `Parameterized` module whose numbers come from [`kernellib`](https://github.com/jejjohnson/kernellib) (`kernel.frozen()` returns the plain kernellib kernel at the current hyperparameters), and its sites are scoped, cached per call and handed to [`numpyro`](https://github.com/pyro-ppl/numpyro). The exact and sparse GPs do every solve, log-determinant, Gaussian KL and prediction cache with [`gaussx`](https://github.com/jejjohnson/gaussx) on [`lineax`](https://github.com/patrick-kidger/lineax) operators, and the model's `solver=` picks the gaussx strategy (dense Cholesky by default). The non-Gaussian strategies take their cavities, moment matching and natural-parameter updates from gaussx but still factor $`\mathbf{K}`$ with `jax.scipy`, which [`AGENTS.md`](AGENTS.md) lists for conversion. The multivariate normals are [`pyrox-gauss`](../pyrox-gauss)'s, the Fourier, spherical-harmonic and Slepian inducing bases are [`functax`](https://github.com/jejjohnson/functax)'s, and the warps of the warped likelihood are [`flowjax`](https://github.com/danielward27/flowjax) bijections.

pyrox-gp is one package of the pyrox workspace; the [workspace README](../../README.md) introduces the others.

<p align="center"><img src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_gp_overview/pyrox_gp_overview.png" alt="pyrox-gp in one picture: you supply data, a kernel with priors, a likelihood and a NumPyro model; the model writes one joint density with priors on the kernel, whose sites are named by pyrox_name; pyrox-gp conditions it exactly, by sparse variational inference, by site-based approximations for other likelihoods or by pathwise draws, and for many outputs; out come the hyperparameters, the posterior at new inputs, whole functions and the gradient; pyrox and pyrox-gauss, kernellib, functax and flowjax, and gaussx and lineax underneath" width="100%"></p>

## 🌟 Highlights

- 🎛️ **Kernels with priors**: `RBF`, `Matern`, `Periodic`, `RationalQuadratic`, `Linear`, `Polynomial`, `Cosine`, `White` and `Constant` register their hyperparameters with `set_prior` and `autoguide`, and `kernel.frozen()` hands kernellib the plain kernel for spectral densities and feature maps.
- 🎯 **Exact GPs**: `gp_factor` adds the log marginal likelihood to a NumPyro model, `GPPrior.condition` caches the training solve, and `ConditionedGP` predicts means, variances, joint samples and the gradient $`\nabla f(\mathbf{x}_*)`$ with its covariance.
- 🧮 **Exact GPs without the Gram matrix**: `GPPrior(..., matrix_free=True)` evaluates $`\mathbf{K}`$ one row block per matrix-vector product, with gradients, and `preconditioned_cg_solver` solves it by preconditioned conjugate gradients with a stochastic Lanczos log-determinant.
- 🪶 **Sparse variational GPs**: `SparseGPPrior` over inducing points or over inducing features (Fourier on a box, spherical harmonics and Slepian functions on the sphere, graph Laplacian eigenvectors, decoupled bases), five guides, `svgp_elbo` and `svgp_factor`, and `ConjugateVI` for natural-gradient steps.
- 🎲 **Non-Gaussian likelihoods**: Bernoulli, Poisson, Student-t, softmax, heteroscedastic, warped and any NumPyro distribution, and for the ones with one latent per point, Laplace, Gauss-Newton, posterior linearisation, expectation propagation and quasi-Newton strategies behind one `condition_nongauss` call.
- 🔀 **Many outputs**: LMC, ICM and OILMM kernels with exact and sparse models, and collapsed latent-factor regression, whose decoder is integrated out so the cost is linear in the number of outputs.
- 🌊 **Whole functions from the posterior**: `PathwiseSampler` and `DecoupledPathwiseSampler` draw posterior functions by Matheron's rule on random-feature prior draws, callable at any input after one solve.

## 📐 One prior, four ways to condition it

A `GPPrior` (or a `SparseGPPrior`) is the prior, and the package offers four routes from it to a posterior, chosen by the likelihood, the size of the data and what you need out.

| Route | Entry points | Posterior | Reach for it when |
|---|---|---|---|
| Exact | `gp_factor`, `GPPrior.condition` | `ConditionedGP`, Gaussian in closed form | the noise is Gaussian |
| Sparse variational | `SparseGPPrior`, the guides, `svgp_elbo`, `ConjugateVI` | $`q(\mathbf{u})`$ over $`M`$ inducing values | $`N`$ is large, or the likelihood is not Gaussian and you want SVI |
| Site-based | `condition_nongauss` with a strategy, `gp_laplace_factor` | `ConditionedNonGaussGP`, a Gaussian $`q(\mathbf{f})`$ | counts, labels or heavy tails at moderate $`N`$ |
| Pathwise | `PathwiseSampler`, `DecoupledPathwiseSampler` | `PathwiseFunction`, callable draws | you need whole functions, for Thompson sampling or a derived quantity |

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_gp_hub/pyrox_gp_hub-dark.svg">
    <img alt="One GPPrior fans out to seven uses: gp_factor as a NumPyro factor for NUTS and SVI, log_marginal for the collapsed evidence under jit and grad, condition and predict for the mean and variance with include_noise for a band, predict_gradient for the gradient and its covariance, PathwiseSampler for callable posterior draws, condition_nongauss with Laplace, Gauss-Newton, quasi-Newton, posterior linearisation and EP, and gp_laplace_factor for the Laplace evidence under NUTS" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_gp_hub/pyrox_gp_hub-light.svg" width="100%">
  </picture>
</p>

The figure draws the exact GP, the one pyrox-gp prior with every part.

With Gaussian noise of variance $`\sigma^2`$ and a zero mean, the exact posterior at test inputs $`\mathbf{X}_* \in \mathbb{R}^{N_* \times D_x}`$ is

```math
p(\mathbf{f}_* \mid \mathbf{y}) = \mathcal{N}\left(\mathbf{K}_{*X} \hat{\mathbf{K}}^{-1} \mathbf{y}, \ \mathbf{K}_{**} - \mathbf{K}_{*X} \hat{\mathbf{K}}^{-1} \mathbf{K}_{X*}\right)
```

with $`\hat{\mathbf{K}} = \mathbf{K}_{\boldsymbol{\theta}} + \sigma^2 \mathbf{I}`$, $`\mathbf{K}_{*X} = k_{\boldsymbol{\theta}}(\mathbf{X}_*, \mathbf{X}) \in \mathbb{R}^{N_* \times N}`$ and $`\mathbf{K}_{**} = k_{\boldsymbol{\theta}}(\mathbf{X}_*, \mathbf{X}_*)`$. The sparse route puts a Gaussian $`q(\mathbf{u})`$ on the function's values $`\mathbf{u} = f(\mathbf{Z}) \in \mathbb{R}^{M}`$ at $`M \ll N`$ inducing inputs $`\mathbf{Z}`$ (or on $`M`$ inducing features) and maximises

```math
\mathcal{L}(q) = \sum_{n=1}^{N} \mathbb{E}_{q(f_n)}\left[\log p(y_n \mid f_n)\right] - \mathrm{KL}\left[q(\mathbf{u}) \Vert p(\mathbf{u})\right]
```

where $`q(f_n)`$ is the Gaussian that $`q(\mathbf{u})`$ implies at $`\mathbf{x}_n`$. The site-based strategies replace each $`p(y_n \mid f_n)`$ by a Gaussian site in $`f_n`$ and differ only in how they choose its curvature (the exact Hessian at the mode, its Gauss-Newton part, a statistical linearisation or moment matching). A pathwise draw adds a correction to a random-feature prior draw $`\tilde{f}`$ with noise $`\tilde{\boldsymbol{\varepsilon}} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})`$,

```math
f^{(s)}(\cdot) = \tilde{f}(\cdot) + k_{\boldsymbol{\theta}}(\cdot, \mathbf{X}) \hat{\mathbf{K}}^{-1} \left(\mathbf{y} - \tilde{f}(\mathbf{X}) - \tilde{\boldsymbol{\varepsilon}}\right)
```

which has the exact posterior's distribution as the number of random features grows.

## 📥 Install

We have not put pyrox-gp on PyPI, and the name `pyrox` there belongs to an unrelated project, so install it from GitHub with [uv](https://docs.astral.sh/uv/). It brings in the pyrox core and pyrox-gauss from the same repository, gaussx, kernellib, functax, lineax, einx and flowjax, and needs Python 3.12 or newer.

```bash
uv add "pyrox-gp @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-gp"
```

kernellib v0.0.22 pins gaussx v0.6.5, while pyrox-gp needs gaussx v0.8.2, so add this override to your project's `pyproject.toml`; without it uv cannot resolve the install.

```toml
[tool.uv]
override-dependencies = [
    "gaussx @ git+https://github.com/jejjohnson/gaussx.git@v0.8.2",
]
```

Two extras add the rest: `flows` for `latent_total_correlation` (gauss-flows, which brings scikit-learn, diffrax and matfree), and `colab` for what the example notebooks need (matplotlib, watermark and pandas). The quickstart needs none of them.

```bash
uv add "pyrox-gp[flows] @ git+https://github.com/jejjohnson/pyrox.git#subdirectory=packages/pyrox-gp"
```

## 🧠 Quickstart: one regression, three posteriors

The data are 64 noisy points of $`y_n = \sin(3 x_n) + \varepsilon_n`$ on $`[-1, 1]`$, with $`\varepsilon_n \sim \mathcal{N}(0, 0.1^2)`$, and three test inputs $`\mathbf{X}_* \in \mathbb{R}^{N_* \times D_x}`$: $`x = 0`$ and $`0.5`$ inside the data, and $`x = 2`$ outside it. The model is ours to write: an RBF kernel $`k_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp(-\lVert \mathbf{x} - \mathbf{x}' \rVert^2 / 2\ell^2)`$ with log-normal priors on $`\boldsymbol{\theta} = (\sigma_f^2, \ell)`$, a log-normal prior on the noise variance $`\sigma^2`$, and `gp_factor` for the marginal likelihood. SVI with a point-mass guide finds the MAP hyperparameters $`\hat{\boldsymbol{\theta}}`$.

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from pyrox_gp import RBF, GPPrior, gp_factor


# The package's tests run in double precision, so the quickstart does too.
jax.config.update("jax_enable_x64", True)

# Shapes: N = 64 samples; D_x = 1 input; N_* = 3 test inputs

# X evenly spaced on [−1, 1], and y = sin(3 X) + ε
# → (N, D_x), (N,)
inputs = jnp.linspace(-1.0, 1.0, 64)[:, None]
targets = jnp.sin(3.0 * inputs[:, 0]) + 0.1 * jr.normal(jr.key(1), (64,))
# X_*, two inputs inside the data and one outside
# → (N_*, D_x)
test_inputs = jnp.array([[0.0], [0.5], [2.0]])

# today (pyrox-gp)
# k_θ(x, x') = σ_f² exp(−‖x − x'‖² / 2ℓ²), sites RBF.variance and RBF.lengthscale
kernel = RBF(init_lengthscale=0.5)
# σ_f² ~ LogNormal(0, 1) and ℓ ~ LogNormal(0, 1)
kernel.set_prior("variance", dist.LogNormal(0.0, 1.0))
kernel.set_prior("lengthscale", dist.LogNormal(0.0, 1.0))
# today (pyrox-gp)
# f ~ GP(0, k_θ) at the training inputs X
prior = GPPrior(kernel, inputs)


# user
# log p(y | X, θ, σ²) = log 𝒩(y; 0, K_θ + σ² I), with σ² ~ LogNormal(log 0.01, 1)
def model(targets: Float[Array, " N"]) -> None:
    noise_var = numpyro.sample("noise_var", dist.LogNormal(jnp.log(0.01), 1.0))
    gp_factor("y", prior, targets, noise_var)


# θ̂ = argmax_θ p(y | X, θ, σ²) p(θ) p(σ²), the MAP of the three sites
guide = AutoDelta(model)
svi = SVI(model, guide, numpyro.optim.Adam(1e-2), Trace_ELBO())
fit = svi.run(jr.key(2), 2000, targets, progress_bar=False)
# {"RBF.variance", "RBF.lengthscale", "noise_var"} at θ̂
theta = guide.median(fit.params)
```

**A. The exact posterior.** `condition` solves $`\hat{\mathbf{K}}^{-1} \mathbf{y}`$ once and caches it, and `predict` returns the mean and the variance of $`f(\mathbf{X}_*)`$. `handlers.substitute` fixes the kernel's sites at $`\hat{\boldsymbol{\theta}}`$; without it each call would draw the hyperparameters from their priors.

```python
from numpyro import handlers


# Shapes: N samples; N_* test inputs

with handlers.substitute(data=theta):
    # today (pyrox-gp)
    # p(f_* | y) at θ̂, with K̂⁻¹ y cached
    posterior = prior.condition(targets, theta["noise_var"])
    # today (pyrox-gp)
    # 𝔼[f_*] = K_*X K̂⁻¹ y and Var[f_*] = diag(K_** − K_*X K̂⁻¹ K_X*)
    # (N_*, D_x) → (N_*,), (N_*,)
    exact_mean, exact_var = posterior.predict(test_inputs)
```

**B. Whole functions.** `PathwiseSampler` draws $`S = 256`$ posterior functions by Matheron's rule, each a random-feature prior draw with $`F = 1024`$ features plus its correction. A draw is a function: it can be evaluated at any input, here at $`\mathbf{X}_*`$.

```python
from pyrox_gp import PathwiseSampler


# Shapes: S = 256 paths; F = 1024 random features; N_* test inputs

with handlers.substitute(data=theta):
    # today (pyrox-gp)
    # f⁽ˢ⁾(·) = f̃⁽ˢ⁾(·) + k(·, X) K̂⁻¹ (y − f̃⁽ˢ⁾(X) − ε̃⁽ˢ⁾), for s = 1, …, S
    sampler = PathwiseSampler(posterior, n_features=1024)
    paths = sampler.sample_paths(jr.key(3), n_paths=256)
# today (pyrox-gp)
# f⁽ˢ⁾(X_*) for every path
# (N_*, D_x) → (S, N_*)
path_draws = paths(test_inputs)
```

**C. Eight inducing points.** A sparse GP keeps $`M = 8`$ of the 64 inputs as inducing inputs $`\mathbf{Z}`$, picked by randomly pivoted Cholesky on $`\mathbf{K}_{\boldsymbol{\theta}}`$, and puts a Gaussian $`q(\mathbf{u})`$ in natural parameters on $`\mathbf{u} = f(\mathbf{Z})`$. With Gaussian noise one natural-gradient step of full length reaches the optimal $`q(\mathbf{u})`$; further steps leave $`\mathcal{L}`$ where it is.

```python
from pyrox_gp import (
    ConjugateVI,
    GaussianLikelihood,
    NaturalGuide,
    SparseGPPrior,
    init_inducing,
)


# Shapes: N samples; M = 8 inducing inputs; N_* test inputs

with handlers.substitute(data=theta):
    # today (pyrox-gp)
    # Z, M rows of X picked by randomly pivoted Cholesky on K_θ
    # → (M, D_x)
    inducing = init_inducing(inputs, 8, kernel=kernel, key=jr.key(4))
    # today (pyrox-gp)
    # p(u) = 𝒩(0, K_ZZ), with u = f(Z)
    sparse = SparseGPPrior(kernel, Z=inducing)
    # today (pyrox-gp)
    # p(y_n | f_n) = 𝒩(y_n; f_n, σ²) at the fitted σ²
    likelihood = GaussianLikelihood(theta["noise_var"])
    # today (pyrox-gp)
    # q(u) = argmax_q 𝓛(q), in one step from 𝒩(0, I)
    q_u = ConjugateVI(damping=1.0).step(
        sparse, NaturalGuide.init(8), likelihood, inputs, targets
    )
    # today (pyrox-gp)
    # 𝔼_q[f_*] and Var_q[f_*] under q(f) = ∫ p(f | u) q(u) du
    # → (N_*,), (N_*,)
    sparse_mean, sparse_var = sparse.condition(q_u).predict(test_inputs)
```

**Compare.** Each route gives a mean and a standard deviation of $`f(\mathbf{X}_*)`$.

```python
# Shapes: N_* test inputs (the moments and the draws as above)

routes = {
    "exact": (exact_mean, jnp.sqrt(exact_var)),
    "pathwise": (path_draws.mean(axis=0), path_draws.std(axis=0)),
    "sparse": (sparse_mean, jnp.sqrt(sparse_var)),
}
for name, (mean, std) in routes.items():
    print(f"{name:>8}", mean.round(2), std.round(2))
```

This prints

```
   exact [ 0.04  1.   -0.28] [0.03 0.03 0.73]
pathwise [ 0.04  1.   -0.25] [0.03 0.03 0.75]
  sparse [ 0.04  1.   -0.28] [0.03 0.03 0.73]
```

that is, for each route, the mean of $`f`$ at $`x = 0`$, 0.5 and 2 and its standard deviation. SVI put $`\hat{\sigma}_f^2`$ at 0.59, $`\hat{\ell}`$ at 0.55 and $`\hat{\sigma}^2`$ at 0.012, close to the true noise variance of 0.01. Inside the data the exact posterior is within 0.04 of $`\sin(0) = 0`$ and $`\sin(1.5) = 1.00`$, with a standard deviation of 0.03. At $`x = 2`$, almost two lengthscales from the nearest point, the mean of $`-0.28`$ happens to match $`\sin(6) = -0.28`$, but the standard deviation has grown to 0.73, close to the prior's $`\hat{\sigma}_f = 0.77`$: the data no longer pin $`f`$ down there. The 256 pathwise draws reproduce those moments to within Monte Carlo error, and the eight inducing points reproduce them to two decimals, because a lengthscale of 0.55 on $`[-1, 1]`$ leaves the function few degrees of freedom. The [exact GP regression](https://jejjohnson.github.io/pyrox/exact-gp-regression/) and [pathwise sampling](https://jejjohnson.github.io/pyrox/gp-pathwise/) tutorials go further.

## 🎲 Counts: swap the strategy, keep the prior

With counts the latent function no longer integrates out. The prior stays a `GPPrior`, the likelihood becomes `PoissonLikelihood`, $`y_n \sim \mathrm{Poisson}(e^{f_n})`$, and `condition_nongauss` takes the strategy that builds the Gaussian approximation $`q(\mathbf{f})`$. Here the counts come from the log-rate $`1 + \sin(3 x_n)`$ on the same inputs, and the kernel has fixed hyperparameters ($`\sigma_f^2 = 1`$, $`\ell = 0.5`$), so the block needs no handler.

```python
from pyrox_gp import ExpectationPropagation, LaplaceInference, PoissonLikelihood


# Shapes: N samples; N_* test inputs

# y_n ~ Poisson(exp(1 + sin(3 x_n)))
# → (N,)
log_rate = 1.0 + jnp.sin(3.0 * inputs[:, 0])
counts = jr.poisson(jr.key(5), jnp.exp(log_rate)).astype(inputs.dtype)
# today (pyrox-gp)
# f ~ GP(0, k), with σ_f² = 1 and ℓ = 0.5 fixed
count_prior = GPPrior(RBF(init_lengthscale=0.5), inputs)

for strategy in (LaplaceInference(), ExpectationPropagation()):
    # today (pyrox-gp)
    # q(f) = 𝒩(m, V) ∝ p(f) ∏ₙ tₙ(fₙ), a Gaussian site tₙ per count
    approx = count_prior.condition_nongauss(
        PoissonLikelihood(), counts, strategy=strategy
    )
    # today (pyrox-gp)
    # 𝔼_q[f_*] and Var_q[f_*]
    # (N_*, D_x) → (N_*,), (N_*,)
    mean, var = approx.predict(test_inputs)
    print(f"{type(strategy).__name__:>22}", mean.round(2), jnp.sqrt(var).round(2))
```

This prints

```
      LaplaceInference [ 1.12  1.9  -0.  ] [0.15 0.11 0.99]
ExpectationPropagation [ 1.11  1.9  -0.01] [0.15 0.11 0.99]
```

that is, the posterior mean and standard deviation of the log-rate at $`x = 0`$, 0.5 and 2, against the true values of 1.00, 2.00 and 0.72. Laplace and expectation propagation agree to 0.01. Inside the data both are within 0.12 of the truth, with standard deviations of 0.11 to 0.15, and at $`x = 2`$, where there are no counts, both return to the prior: a mean of zero and a standard deviation of 0.99 against the prior's 1. Gauss-Newton, posterior linearisation and quasi-Newton slot into the same call; the [Laplace](https://jejjohnson.github.io/pyrox/advanced-gp-laplace/) and [expectation propagation](https://jejjohnson.github.io/pyrox/advanced-gp-ep/) tutorials compare them on a classification task.

## 🔬 What else is in the box

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_gp_roles/pyrox_gp_roles-dark.svg">
    <img alt="pyrox-gp on the GeoStack role ladder: conditioning, the evidence factors, the variational objectives and the samplers as algorithms; the kernels, GP priors, likelihoods, guides, posteriors and pathwise samplers as models; the inference strategies and the CG solver as strategies; the predict and density methods and the functions on arrays as primitives; the inducing features as structure; the setup functions and gaussx's solver strategies beside the ladder" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_gp_roles/pyrox_gp_roles-light.svg" width="100%">
  </picture>
</p>

Each public name sits on one rung of the GeoStack role ladder, by the job it does; the figure shows a few names per group, and the table below lists them all.

`pyrox_gp` is one flat namespace; its `__all__` holds every name below.

| Area | Symbols | Notes |
|---|---|---|
| Kernels | `RBF`, `Matern`, `Periodic`, `RationalQuadratic`, `Linear`, `Polynomial`, `Cosine`, `White`, `Constant` | `Parameterized` wrappers of `kernellib.functional`, with priors, guides and `frozen()` |
| Exact GP | `GPPrior`, `ConditionedGP`, `GradientPosterior`, `gp_factor`, `gp_sample`, `preconditioned_cg_solver` | The marginal likelihood, the cached posterior, the gradient posterior, a latent sample site, and CG with a low-rank preconditioner |
| Sparse prior and inducing features | `SparseGPPrior`, `ConditionedSparseGP`, `init_inducing`, `InducingFeatures`, `FourierInducingFeatures`, `SphericalHarmonicInducingFeatures`, `SlepianInducingFeatures`, `LaplacianInducingFeatures`, `DecoupledInducingFeatures`, `funk_hecke_coefficients` | Inducing points picked from the data, or inter-domain features; `funk_hecke_coefficients` is a deprecated alias of kernellib's |
| Guides | `FullRankGuide`, `MeanFieldGuide`, `WhitenedGuide`, `NaturalGuide`, `DeltaGuide` | $`q(\mathbf{u})`$ in moment, whitened, natural or point-mass form |
| SVGP inference | `svgp_elbo`, `svgp_factor`, `ConjugateVI` | The ELBO as a scalar or a NumPyro factor, and natural-gradient steps |
| Likelihoods | `GaussianLikelihood`, `BernoulliLikelihood`, `PoissonLikelihood`, `StudentTLikelihood`, `SoftmaxLikelihood`, `HeteroscedasticGaussianLikelihood`, `DistLikelihood`, `WarpedGaussianLikelihood`, `warped_predictive_moments` | Observation models over gaussx, any NumPyro distribution, and a Gaussian through a flowjax warp |
| Non-Gaussian inference | `LaplaceInference`, `GaussNewtonInference`, `PosteriorLinearization`, `ExpectationPropagation`, `QuasiNewtonInference`, `ConditionedNonGaussGP`, `gp_laplace_factor` | Strategies for `condition_nongauss`, whose Newton and quasi-Newton fits run under `jax.jit` and `jax.vmap`, and the Laplace evidence as a NumPyro factor |
| Multi-output | `LMCKernel`, `ICMKernel`, `OILMMKernel`, `SharedInducingPoints`, `MultiOutputInducingVariables`, `MultiOutputGPPrior`, `ConditionedMultiOutputGP`, `MultiOutputSparseGPPrior`, `ConditionedMultiOutputSparseGP`, `OILMMGPPrior`, `ConditionedOILMMGP`, `mo_gp_factor`, `mo_svgp_elbo`, `mo_svgp_factor`, `oilmm_factor` | Mixing kernels, and exact, sparse and orthogonal-projection models over them |
| Latent factor | `LatentFactorGPPrior`, `ConditionedLatentFactorGP`, `collapsed_lfr_log_prob`, `decoder_posterior`, `lfr_predictive_moments`, `lfr_factor`, `lfr_model`, `latent_init`, `latent_total_correlation`, `warp_to_base`, `warped_decoder_posterior`, `warped_lfr_log_prob` | GP latent factors with the linear decoder integrated out; `latent_total_correlation` needs the `flows` extra |
| Pathwise | `PathwiseSampler`, `DecoupledPathwiseSampler`, `PathwiseFunction` | Posterior function draws for the exact and the sparse GP |
| Protocols | `Kernel`, `Guide`, `Likelihood` | `Kernel` is kernellib's `AbstractKernel`; `Guide` and `Likelihood` come from `pyrox.likelihoods` |

The Markov GPs on a time grid (`MarkovGPPrior`, the SDE kernels and the Kalman-based non-Gaussian and sparse strategies) moved to [`pyrox-ssm`](../pyrox-ssm); importing one of those names from `pyrox_gp` raises an `ImportError` that names its `pyrox_ssm` home.

## 📓 Tutorials

| # | Tutorial | What it shows |
|---|---|---|
| 1 | [Exact GP regression](https://jejjohnson.github.io/pyrox/exact-gp-regression/) | three ways to put priors on a kernel, fitted by SVI, then the posterior at the fitted hyperparameters |
| 2 | [Latent GP classification](https://jejjohnson.github.io/pyrox/latent-gp-classification/) | a Bernoulli likelihood on a whitened latent GP, fitted by SVI |
| 3 | [Pathwise sampling](https://jejjohnson.github.io/pyrox/gp-pathwise/) and [with NumPyro](https://jejjohnson.github.io/pyrox/gp-pathwise-numpyro/) | Matheron's rule for exact and sparse GPs, and inside NumPyro models |
| 4 | [GP derivatives](https://jejjohnson.github.io/pyrox/gp-derivatives/) | the gradient posterior and sensitivity maps with error bars |
| 5 | [Multi-output GPs](https://jejjohnson.github.io/pyrox/multioutput-gp/) | LMC, ICM and OILMM |
| 6 | [Laplace](https://jejjohnson.github.io/pyrox/advanced-gp-laplace/), [Gauss-Newton](https://jejjohnson.github.io/pyrox/advanced-gp-gauss-newton/), [posterior linearisation](https://jejjohnson.github.io/pyrox/advanced-gp-pl/), [EP](https://jejjohnson.github.io/pyrox/advanced-gp-ep/) and [quasi-Newton](https://jejjohnson.github.io/pyrox/advanced-gp-qn/) | one notebook per non-Gaussian strategy |
| 7 | [Regression masterclass](https://jejjohnson.github.io/pyrox/regression-masterclass-parameterized/) | seven regressors on one dataset, the exact GP among them |

## 📖 Documentation

- **[Documentation](https://jejjohnson.github.io/pyrox/)**: the whole workspace, with the tutorials and examples of every package.
- **[GP API](https://jejjohnson.github.io/pyrox/reference/gp/)**: the public names, generated from the docstrings, with the derivations of the gradient posterior and the inducing features; the multi-output model entry points (`MultiOutputGPPrior` and the rest) are not on it yet.
- **[Capability index](https://jejjohnson.github.io/pyrox/capabilities/)**: every public name in the workspace and in the libraries it builds on, to check before writing something new.
- **[`AGENTS.md`](AGENTS.md)**: the contracts every kernel, guide, likelihood and strategy keeps (the kernel context, gaussx for the linear algebra) and the tests that hold them.

## 🧩 Ecosystem

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_gp_ecosystem/pyrox_gp_ecosystem-dark.svg">
    <img alt="No package imports pyrox-gp; pyrox-nn, pyrox-lgm, pyrox-ssm, pyrox-gauss and pyrox-xtreme forbid importing it in their import-guard tests; pyrox-gp builds on pyrox, pyrox-gauss, kernellib, gaussx, lineax, functax, flowjax, JAX, Equinox, NumPyro and einx, and on optax and gauss-flows through extras" src="https://raw.githubusercontent.com/jejjohnson/geostack/main/docs/images/libraries/pyrox_gp_ecosystem/pyrox_gp_ecosystem-light.svg" width="100%">
  </picture>
</p>

pyrox-gp is a package of the pyrox [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/), in the GeoML tier of the [GeoStack](https://github.com/jejjohnson/geostack). It owns the GP models and their NumPyro surface, and composes the rest in:

| Concern | Owner |
|---|---|
| Sites, `Parameterized`, the per-call cache and the `Likelihood` and `Guide` protocols | [`pyrox`](../pyrox) (`_core` and `likelihoods`) |
| Multivariate normals over gaussx operators | [`pyrox-gauss`](../pyrox-gauss) |
| Kernel functions, spectral densities and landmark selection | [`kernellib`](https://github.com/jejjohnson/kernellib) |
| Solves, log-determinants, KLs, prediction caches, EP and natural-parameter updates, solver strategies | [`gaussx`](https://github.com/jejjohnson/gaussx) and [`lineax`](https://github.com/patrick-kidger/lineax) |
| Fourier, spherical-harmonic and Slepian bases for the inducing features | [`functax`](https://github.com/jejjohnson/functax) |
| Bijections for the warped likelihood and the warped latent-factor model | [`flowjax`](https://github.com/danielward27/flowjax), and [`gauss_flows`](https://github.com/jejjohnson/gauss_flows) (the `flows` extra) |
| Sample sites, NUTS, SVI, autoguides and `Predictive` | [`numpyro`](https://github.com/pyro-ppl/numpyro) |
| Modules as pytrees | [`equinox`](https://github.com/patrick-kidger/equinox) |
| The Newton loop of `LaplaceInference` and `GaussNewtonInference`, and L-BFGS for `QuasiNewtonInference` | [`argminax`](https://github.com/jejjohnson/argminax) and [`optimistix`](https://github.com/patrick-kidger/optimistix) |

Dependencies point one way. pyrox-gp imports the pyrox core and pyrox-gauss, and no other package of the workspace imports pyrox-gp: pyrox-nn, pyrox-lgm, pyrox-ssm, pyrox-gauss and pyrox-xtreme each forbid it in an import-guard test, and nothing outside the workspace imports it either. A pyrox-gp kernel still works where a kernellib kernel is expected, so pyrox-nn's `HSGPFeatures` and `SphericalHarmonicFeatures` take one when you pass it.

## 🌳 Repository layout

```
packages/pyrox-gp/
├── src/pyrox_gp/
│   ├── __init__.py          ← the public names, and the ImportErrors for names moved to pyrox-ssm
│   ├── _kernels.py          ← kernels with priors
│   ├── _models.py           ← GPPrior, ConditionedGP, gp_factor, gp_sample
│   ├── _sparse.py, _inducing.py, _guides.py, _inference.py ← the sparse variational GP
│   ├── _likelihoods.py, _warped.py ← observation models
│   ├── _inference_nongauss.py ← Laplace, Gauss-Newton, PL, EP and quasi-Newton
│   ├── _multi_output*.py, _latent_factor*.py ← many outputs
│   └── _pathwise.py         ← Matheron samplers
├── tests/                   ← dense references, site sets under trace, the moved-name errors
└── AGENTS.md                ← the contracts and the tests that hold them
```

## ⚙️ Development

The package is developed inside the workspace, so clone the repository and run the tasks from its root.

```bash
git clone https://github.com/jejjohnson/pyrox.git
cd pyrox
make install                                                     # uv sync --all-groups + pre-commit hooks
uv run pytest --no-cov packages/pyrox-gp/tests -m "not slow"     # this package's fast tests
make lint                                                        # ruff check on the whole repo
make typecheck                                                   # ty check on every package
```

`make help` lists every task. The root [`AGENTS.md`](../../AGENTS.md) holds the rules for the whole workspace, and this package's [`AGENTS.md`](AGENTS.md) adds the contracts of the GP components.

## 🎓 References

- Rasmussen, C. E., & Williams, C. K. I. (2006). [Gaussian processes for machine learning](https://gaussianprocess.org/gpml/). *MIT Press*.
- Minka, T. P. (2001). [Expectation propagation for approximate Bayesian inference](https://arxiv.org/abs/1301.2294). *UAI*.
- Hensman, J., Matthews, A., & Ghahramani, Z. (2015). [Scalable variational Gaussian process classification](https://arxiv.org/abs/1411.2005). *AISTATS*.
- Hensman, J., Durrande, N., & Solin, A. (2017). [Variational Fourier features for Gaussian processes](https://arxiv.org/abs/1611.06740). *JMLR*.
- Cheng, C.-A., & Boots, B. (2017). [Variational inference for Gaussian process models with linear complexity](https://arxiv.org/abs/1711.10127). *NeurIPS*.
- Dutordoir, V., Durrande, N., & Hensman, J. (2020). [Sparse Gaussian processes with spherical harmonic features](https://arxiv.org/abs/2006.16649). *ICML*.
- Burt, D. R., Rasmussen, C. E., & van der Wilk, M. (2020). [Convergence of sparse variational inference in Gaussian processes regression](https://arxiv.org/abs/2008.00323). *JMLR*.
- Wilson, J. T., Borovitskiy, V., Terenin, A., Mostowsky, P., & Deisenroth, M. P. (2020). [Efficiently sampling functions from Gaussian process posteriors](https://arxiv.org/abs/2002.09309). *ICML*.
- Bruinsma, W. P., et al. (2020). [Scalable exact inference in multi-output Gaussian processes](https://arxiv.org/abs/1911.06287). *ICML*.
- Johnson, J. E., Laparra, V., Pérez-Suay, A., Mahecha, M. D., & Camps-Valls, G. (2020). [Kernel methods and their derivatives: concept and perspectives for the earth system sciences](https://doi.org/10.1371/journal.pone.0235885). *PLOS ONE* 15(10), e0235885.
- Borovitskiy, V., Azangulov, I., Terenin, A., Mostowsky, P., Deisenroth, M. P., & Durrande, N. (2021). [Matérn Gaussian processes on graphs](https://arxiv.org/abs/2010.15538). *AISTATS*.
- Maroñas, J., Hamelijnck, O., Knoblauch, J., & Damoulas, T. (2021). [Transforming Gaussian processes with normalizing flows](https://arxiv.org/abs/2011.01596). *AISTATS*.
- Stevenson, E. T., et al. (2026). [Gaussian process latent factor regression for low-data, high-dimensional output problems](https://arxiv.org/abs/2606.06576). *arXiv:2606.06576*.

## 📄 License

MIT. See [LICENSE](LICENSE).
