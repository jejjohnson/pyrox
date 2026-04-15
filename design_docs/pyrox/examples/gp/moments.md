---
status: stable
version: 0.1.0
---

# pyrox.gp — Examples & Supported Patterns

> Worked model examples showing how pyrox.gp components compose into
> hierarchical Bayesian models.  For core architecture, protocols,
> and solver mathematics, see [`../../api/gp/moments.md`](../../api/gp/moments.md).

---

## Table of Contents

1. [Collapsed GP Regression (Gaussian Likelihood)](#1-collapsed-gp-regression)
2. [Latent GP + Non-Gaussian Likelihood](#2-latent-gp--non-gaussian-likelihood)
3. [BHM with Extreme Value Distribution + Spatial GPs](#3-bhm-with-extreme-value-distribution--spatial-gps)
4. [Heteroscedastic GP (Input-Dependent Noise)](#4-heteroscedastic-gp)
5. [Additive / Multi-Component GPs](#5-additive--multi-component-gps)
6. [Multi-Output GP with Correlated Outputs](#6-multi-output-gp-with-correlated-outputs)
7. [Spatiotemporal GP with Kronecker Structure](#7-spatiotemporal-gp-with-kronecker-structure)
8. [Log-Gaussian Cox Process (Point Process)](#8-log-gaussian-cox-process)
9. [GP Classification (Binary and Multi-Class)](#9-gp-classification)
10. [Deep Kernel Learning](#10-deep-kernel-learning)
11. [Warped GP (Non-Gaussian Targets)](#11-warped-gp)
12. [Sparse Variational GP (Large N)](#12-sparse-variational-gp)
13. [Temporal GP via State-Space (Streaming)](#13-temporal-gp-via-state-space)
14. [GP with Inducing Features (VISH / VFF)](#14-gp-with-inducing-features-vish--vff)
15. [Missing Data / Partial Observations](#15-missing-data--partial-observations)
16. [Marked Temporal Point Process + GP Intensity](#16-marked-temporal-point-process--gp-intensity)
17. [Pathwise Posterior Sampling (Matheron's Rule)](#17-pathwise-posterior-sampling-matherons-rule)

---

## Conventions

Throughout this document:

- `gp_sample(name, kernel, X, solver)` — GP is an explicit latent; requires a guide for `q(f)`.
- `gp_factor(name, kernel, X, y, noise_var, solver)` — GP is marginalized out analytically; only works with Gaussian likelihood.
- `ComposedGuide` — wires GP-specific guides with generic autoguides.
- Kernels, solvers, and representations follow the protocols in `pyrox_gp/protocols.py`.

Inference recommendations use these shorthands:

| Shorthand | Meaning |
|-----------|---------|
| **MCMC** | NUTS on all latent variables (hyperparams + latent functions if explicit) |
| **SVI-collapsed** | SVI with `gp_factor`; only hyperparams need a guide |
| **SVI-structured** | SVI with `gp_sample` + GP-aware guides (`WhitenedMeanFieldGuide`, etc.) |

---

## 1. Collapsed GP Regression

**When**: Gaussian likelihood, moderate N, full Bayesian over hyperparameters.

The GP latent function f is marginalized out analytically. Only kernel
hyperparameters θ = {σ², ℓ, σ_n²} remain as latent variables.

### Model

```python
def model(X, y):
    # Hyperpriors
    variance = numpyro.sample("variance", dist.LogNormal(0.0, 1.0))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))
    noise_var = numpyro.sample("noise_var", dist.LogNormal(-2.0, 1.0))

    kernel = RBF(variance=variance, lengthscale=lengthscale)
    solver = CholeskySolver()

    # Marginal likelihood as a factor (f integrated out)
    gp_factor("gp", kernel, X, y, noise_var, solver)
```

### Inference

```python
# MCMC — clean, no guide needed
mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=1000)
mcmc.run(rng_key, X, y)

# SVI — Delta guide on hyperparams (= Type-II ML / MAP)
guide = AutoDelta(model)
svi = SVI(model, guide, Adam(1e-2), Trace_ELBO())
```

### Prediction

After MCMC, integrate predictions over posterior hyperparameter samples:
```python
for s in range(S):
    kernel_s = RBF(variance=samples["variance"][s], ...)
    posterior_s = GPPrior(kernel_s, X, solver).condition(y, samples["noise_var"][s])
    mu_s, var_s = posterior_s.predict(X_test)
```

### Notes

- O(N³) per MCMC step (Cholesky).
- No guide complexity — NUTS explores the low-dimensional θ space.
- Best for N ≲ 5000.

---

## 2. Latent GP + Non-Gaussian Likelihood

**When**: The observation model is not Gaussian (counts, binary, heavy-tailed, etc.).

The GP latent f must be explicitly represented because p(y|f) is non-conjugate
and f cannot be marginalized analytically.

### Model

```python
def model(X, y_counts):
    variance = numpyro.sample("variance", dist.LogNormal(0.0, 1.0))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))

    kernel = RBF(variance=variance, lengthscale=lengthscale)
    solver = CholeskySolver()

    # Explicit GP latent
    f = gp_sample("f", kernel, X, solver)

    # User's likelihood (Poisson here, could be anything)
    numpyro.sample("y", dist.Poisson(jnp.exp(f)), obs=y_counts)
```

### Guide

```python
guide = ComposedGuide(
    gp_guides={"f": WhitenedMeanFieldGuide(kernel_init, X)},
    model=model,
    auto_guide_cls=AutoNormal,  # handles variance, lengthscale
)
```

### Inference

```python
# SVI (recommended for latent GPs — MCMC on N-dim f is expensive)
svi = SVI(model, guide, Adam(1e-2), Trace_ELBO())

# MCMC (works but expensive — N + dim(θ) latent variables)
mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=1000)
```

### Notes

- SVI with whitened guides is usually much faster than MCMC for large N.
- The whitened parameterization (f = Lv, guide on v) makes the optimization
  landscape isotropic — critical for convergence.
- For very large N, switch to `InducingPointGuide`.

---

## 3. BHM with Extreme Value Distribution + Spatial GPs

**When**: Modeling spatial extremes (e.g., precipitation maxima, wind speeds,
temperature records, methane emission magnitudes) where GEV/GPD parameters
vary smoothly over space.

This is a fully Bayesian hierarchical model where each parameter of the
extreme value distribution has its own GP prior encoding spatial dependence.
The key insight: the GEV parameters (μ, σ, ξ) are latent functions of
spatial location, not fixed constants.

### Generative Story

```
For spatial locations s₁, ..., s_S:
  μ(s) ~ GP(m_μ(s), k_μ(s,s'))      location parameter (real-valued)
  log σ(s) ~ GP(m_σ(s), k_σ(s,s'))  log-scale parameter (ensures σ > 0)
  g(ξ(s)) ~ GP(m_ξ(s), k_ξ(s,s'))  transformed shape (bounded or real)

For each location s_i, for time replicates t = 1..T:
  z_{i,t} ~ GEV(μ(s_i), σ(s_i), ξ(s_i))
```

### Model

```python
def model_spatial_extremes(X_spatial, z_obs):
    """
    Spatial extremes with GP priors on GEV parameters.

    Parameters
    ----------
    X_spatial : (S, D)   spatial coordinates (lon, lat, elevation, ...)
    z_obs : (S, T)       block maxima at S locations over T time periods
    """
    S, T = z_obs.shape
    solver = CholeskySolver()

    # ── Hyperpriors on GP kernels ──
    # Location μ: smooth, long-range spatial dependence
    var_mu = numpyro.sample("var_mu", dist.LogNormal(0.0, 1.0))
    ls_mu = numpyro.sample("ls_mu", dist.LogNormal(3.0, 1.0))   # long lengthscale
    k_mu = RBF(variance=var_mu, lengthscale=ls_mu)

    # Log-scale log(σ): moderate spatial variation
    var_sigma = numpyro.sample("var_sigma", dist.LogNormal(-1.0, 1.0))
    ls_sigma = numpyro.sample("ls_sigma", dist.LogNormal(2.0, 1.0))
    k_sigma = RBF(variance=var_sigma, lengthscale=ls_sigma)

    # Shape ξ: weakly varying, short lengthscale prior
    var_xi = numpyro.sample("var_xi", dist.LogNormal(-2.0, 1.0))
    ls_xi = numpyro.sample("ls_xi", dist.LogNormal(2.0, 1.5))
    k_xi = RBF(variance=var_xi, lengthscale=ls_xi)

    # ── GP latent functions over space ──
    # Mean functions encode prior knowledge (e.g., elevation dependence)
    mu_intercept = numpyro.sample("mu_intercept", dist.Normal(0.0, 10.0))
    f_mu = gp_sample("f_mu", k_mu, X_spatial, solver,
                      mean_fn=lambda X: mu_intercept * jnp.ones(X.shape[0]))

    sigma_intercept = numpyro.sample("sigma_intercept", dist.Normal(0.0, 2.0))
    f_log_sigma = gp_sample("f_log_sigma", k_sigma, X_spatial, solver,
                             mean_fn=lambda X: sigma_intercept * jnp.ones(X.shape[0]))

    xi_intercept = numpyro.sample("xi_intercept", dist.Normal(0.0, 0.5))
    f_xi = gp_sample("f_xi", k_xi, X_spatial, solver,
                      mean_fn=lambda X: xi_intercept * jnp.ones(X.shape[0]))

    # ── Transform to GEV parameter space ──
    mu = f_mu                             # (S,)  real-valued
    sigma = jnp.exp(f_log_sigma)          # (S,)  positive
    xi = f_xi                             # (S,)  real-valued (or apply tanh for bounded)

    # ── Likelihood: GEV at each location, T replicates ──
    # Broadcast: mu/sigma/xi are (S,), z_obs is (S, T)
    with numpyro.plate("time", T, dim=-1):
        with numpyro.plate("space", S, dim=-2):
            numpyro.sample(
                "z",
                GEV(loc=mu[:, None], scale=sigma[:, None], shape=xi[:, None]),
                obs=z_obs,
            )
```

### Guide (SVI)

```python
# Each GP latent gets a structured guide; hyperparams get AutoNormal
guide = ComposedGuide(
    gp_guides={
        "f_mu":        WhitenedMeanFieldGuide(k_mu_init, X_spatial),
        "f_log_sigma": WhitenedMeanFieldGuide(k_sigma_init, X_spatial),
        "f_xi":        WhitenedMeanFieldGuide(k_xi_init, X_spatial),
    },
    model=model_spatial_extremes,
    auto_guide_cls=AutoNormal,
)
```

### Inference

```python
# ── SVI (recommended for S > ~200) ──
svi = SVI(model_spatial_extremes, guide, Adam(1e-3), Trace_ELBO())
svi_state = svi.init(rng_key, X_spatial, z_obs)
for step in range(5000):
    svi_state, loss = svi.update(svi_state, X_spatial, z_obs)

# ── MCMC (gold standard for S < ~100) ──
# NUTS explores the full posterior: 3 × S latent function values
# + ~12 hyperparameters
mcmc = MCMC(
    NUTS(model_spatial_extremes, max_tree_depth=10),
    num_warmup=1000,
    num_samples=2000,
)
mcmc.run(rng_key, X_spatial, z_obs)
```

### Prediction at New Spatial Locations

```python
# After SVI: extract GP posteriors, condition, predict at new locations
params = svi.get_params(svi_state)
# ... extract f_mu posterior from guide, condition on training,
# predict at X_new to get mu(s_new), sigma(s_new), xi(s_new)

# After MCMC: for each posterior sample, compute GP conditional at X_new
samples = mcmc.get_samples()
# Use predict_mcmc pattern: for each sample of (hyperparams, f_values),
# compute the conditional GP at new locations.
```

### Variants

**GPD tail model** — replace GEV with Generalized Pareto Distribution for
threshold exceedances:
```python
# Same GP structure on log_sigma and xi, but no mu (GPD has no location)
# Threshold u is fixed or has its own spatial model
numpyro.sample("z", GPD(scale=sigma, shape=xi), obs=z_obs - threshold)
```

**Nonstationary in time** — add temporal covariates to the GP mean functions:
```python
# μ(s, t) = β₀(s) + β₁(s) · t  where β₀, β₁ are GPs
f_beta0 = gp_sample("f_beta0", k_spatial, X_spatial, solver)
f_beta1 = gp_sample("f_beta1", k_spatial, X_spatial, solver)
mu = f_beta0[:, None] + f_beta1[:, None] * time_covariate[None, :]
```

**With spatial covariates** — elevation, distance to coast, etc. enter
the GP mean function directly:
```python
mean_fn_mu = lambda X: beta_elev * X[:, 2]  # elevation in column 2
f_mu = gp_sample("f_mu", k_mu, X_spatial, solver, mean_fn=mean_fn_mu)
```

### Notes

- Three GPs × S spatial locations = 3S latent variables.  MCMC is
  feasible for S ≲ 100; beyond that, use SVI or inducing points.
- The whitened parameterization is critical here: without it, the GPs
  on ξ (which has small variance) would have very different scales
  from μ, causing NUTS step-size issues.
- Consider `InducingPointGuide` for the less-informative GPs (ξ) to
  reduce cost: ξ varies slowly so few inducing points suffice.
- The GEV distribution must handle the ξ → 0 (Gumbel) limit numerically;
  a custom NumPyro distribution with stable log_prob is recommended.

---

## 4. Heteroscedastic GP

**When**: Observation noise varies with input (e.g., measurement precision
changes across a sensor's range).

Two GPs: one for the mean function, one for the log noise variance.

### Model

```python
def model_heteroscedastic(X, y):
    # Mean GP
    k_f = RBF(
        variance=numpyro.sample("var_f", dist.LogNormal(0, 1)),
        lengthscale=numpyro.sample("ls_f", dist.LogNormal(0, 1)),
    )
    f = gp_sample("f", k_f, X, CholeskySolver())

    # Log-noise GP
    k_g = RBF(
        variance=numpyro.sample("var_g", dist.LogNormal(-1, 1)),
        lengthscale=numpyro.sample("ls_g", dist.LogNormal(0, 1)),
    )
    g = gp_sample("g", k_g, X, CholeskySolver())

    # Observation model: y ~ N(f, exp(g))
    noise_std = jnp.exp(0.5 * g)
    numpyro.sample("y", dist.Normal(f, noise_std), obs=y)
```

### Guide

```python
guide = ComposedGuide(
    gp_guides={
        "f": WhitenedMeanFieldGuide(k_f_init, X),
        "g": WhitenedMeanFieldGuide(k_g_init, X),
    },
    model=model_heteroscedastic,
    auto_guide_cls=AutoNormal,
)
```

### Notes

- Two N-dimensional latents; SVI is preferred for N > ~200.
- The noise GP g should have a negative mean prior (so that exp(g)
  doesn't explode during initialization).

---

## 5. Additive / Multi-Component GPs

**When**: The latent function decomposes into components with different
scales / smoothness (trend + seasonal + residual, global + local, etc.).

### Model

```python
def model_additive(X, y):
    noise_var = numpyro.sample("noise_var", dist.LogNormal(-2, 1))

    # Smooth long-range trend
    k_trend = RBF(
        variance=numpyro.sample("var_trend", dist.LogNormal(0, 1)),
        lengthscale=numpyro.sample("ls_trend", dist.LogNormal(2, 1)),  # long
    )
    f_trend = gp_sample("f_trend", k_trend, X, CholeskySolver())

    # Short-range local variation
    k_local = Matern32(
        variance=numpyro.sample("var_local", dist.LogNormal(-1, 1)),
        lengthscale=numpyro.sample("ls_local", dist.LogNormal(-1, 1)),  # short
    )
    f_local = gp_sample("f_local", k_local, X, CholeskySolver())

    # Additive combination
    f = f_trend + f_local
    numpyro.sample("y", dist.Normal(f, jnp.sqrt(noise_var)), obs=y)
```

### Inference

```python
# SVI with structured guides for each component
guide = ComposedGuide(
    gp_guides={
        "f_trend": WhitenedMeanFieldGuide(k_trend_init, X),
        "f_local": WhitenedMeanFieldGuide(k_local_init, X),
    },
    model=model_additive,
    auto_guide_cls=AutoNormal,
)

# Or MCMC if you can afford it
mcmc = MCMC(NUTS(model_additive), ...)
```

### Notes

- Identifiability: the decomposition is not unique without informative
  priors on the lengthscales.  Strong separation between ls_trend and
  ls_local helps.
- Can also collapse one component using gp_factor (e.g., the trend)
  and sample the other.

---

## 6. Multi-Output GP with Correlated Outputs

**When**: Multiple related output variables that share latent structure
(e.g., temperature and pressure at the same locations, or methane
concentration measured by different instruments).

Uses the Linear Model of Coregionalization (LMC): shared latent GPs
mixed through a weight matrix.

### Model

```python
def model_multi_output(X, Y):
    """
    Parameters
    ----------
    X : (N, D)
    Y : (N, P)   P output channels
    """
    N, P = Y.shape
    R = 3  # number of latent GPs (rank of cross-output covariance)

    solver = CholeskySolver()

    # Shared latent GPs
    latent_fns = []
    for r in range(R):
        k_r = RBF(
            variance=numpyro.sample(f"var_{r}", dist.LogNormal(0, 1)),
            lengthscale=numpyro.sample(f"ls_{r}", dist.LogNormal(0, 1)),
        )
        f_r = gp_sample(f"f_{r}", k_r, X, solver)    # (N,)
        latent_fns.append(f_r)

    F_latent = jnp.stack(latent_fns, axis=-1)          # (N, R)

    # Mixing matrix: W ∈ ℝ^{P×R}
    W = numpyro.sample("W", dist.Normal(0, 1).expand([P, R]).to_event(2))

    # Mixed output: F_out = F_latent @ Wᵀ  →  (N, P)
    F_out = F_latent @ W.T

    # Per-output noise
    noise = numpyro.sample("noise", dist.LogNormal(-2, 1).expand([P]).to_event(1))

    with numpyro.plate("obs", N):
        numpyro.sample("Y", dist.Normal(F_out, jnp.sqrt(noise)), obs=Y)
```

### Guide

```python
guide = ComposedGuide(
    gp_guides={f"f_{r}": WhitenedMeanFieldGuide(k_init, X) for r in range(R)},
    model=model_multi_output,
    auto_guide_cls=AutoNormal,
)
```

---

## 7. Spatiotemporal GP with Kronecker Structure

**When**: Data lives on a grid (space × time) and the kernel factorizes
as k((s,t), (s',t')) = k_space(s,s') · k_time(t,t').

Kronecker structure reduces O((S·T)³) to O(S³ + T³).

### Model

```python
def model_spatiotemporal(X_space, X_time, Y_grid):
    """
    Parameters
    ----------
    X_space : (S, D_s)
    X_time  : (T, D_t)
    Y_grid  : (S, T)     observations on the grid
    """
    S, T = Y_grid.shape

    k_space = Matern52(
        variance=numpyro.sample("var_s", dist.LogNormal(0, 1)),
        lengthscale=numpyro.sample("ls_s", dist.LogNormal(0, 1)),
    )
    k_time = RBF(
        variance=numpyro.sample("var_t", dist.LogNormal(0, 1)),
        lengthscale=numpyro.sample("ls_t", dist.LogNormal(0, 1)),
    )

    kernel = KroneckerKernel(k_space, k_time)
    solver = KroneckerSolver()
    noise_var = numpyro.sample("noise_var", dist.LogNormal(-2, 1))

    # Collapsed (Gaussian likelihood on grid)
    y_flat = Y_grid.ravel()                       # (S*T,)
    X_kron = (X_space, X_time)                     # solver understands tuple
    gp_factor("gp", kernel, X_kron, y_flat, noise_var, solver)
```

### Notes

- KroneckerSolver eigendecomposes each factor independently:
  K_s = U_s Λ_s U_sᵀ, K_t = U_t Λ_t U_tᵀ.
  Then K_y⁻¹ and log|K_y| use Kronecker eigenvalue identities.
- Breaks down if data is not on a complete grid (use sparse GP instead).

---

## 8. Log-Gaussian Cox Process

**When**: Modeling intensity of a point process (event locations, species
occurrences, methane plume detections).

### Model

```python
def model_lgcp(X_grid, counts, cell_area):
    """
    Parameters
    ----------
    X_grid : (G, D)    grid cell centers
    counts : (G,)      event counts per cell
    cell_area : float   area of each grid cell
    """
    k = RBF(
        variance=numpyro.sample("variance", dist.LogNormal(0, 1)),
        lengthscale=numpyro.sample("lengthscale", dist.LogNormal(0, 1)),
    )
    solver = CholeskySolver()

    # GP on log-intensity
    log_lambda = gp_sample("log_lambda", k, X_grid, solver)

    # Poisson likelihood with intensity = exp(log_lambda) * cell_area
    rate = jnp.exp(log_lambda) * cell_area
    numpyro.sample("counts", dist.Poisson(rate), obs=counts)
```

### Notes

- Grid resolution trades off between approximation quality and cost.
- For continuous-domain LGCP (no grid), use thinning or inducing points.

---

## 9. GP Classification

**When**: Binary or multi-class labels with spatial/feature dependence.

### Model (Binary)

```python
def model_classification(X, y_binary):
    k = RBF(
        variance=numpyro.sample("variance", dist.LogNormal(0, 1)),
        lengthscale=numpyro.sample("lengthscale", dist.LogNormal(0, 1)),
    )

    f = gp_sample("f", k, X, CholeskySolver())

    numpyro.sample("y", dist.Bernoulli(logits=f), obs=y_binary)
```

### Model (Multi-Class, K classes)

```python
def model_multiclass(X, y_labels, K):
    solver = CholeskySolver()

    # One GP per class (minus one for identifiability, or all K with softmax)
    logits = []
    for c in range(K):
        k_c = RBF(
            variance=numpyro.sample(f"var_{c}", dist.LogNormal(0, 1)),
            lengthscale=numpyro.sample(f"ls_{c}", dist.LogNormal(0, 1)),
        )
        f_c = gp_sample(f"f_{c}", k_c, X, solver)
        logits.append(f_c)

    logits = jnp.stack(logits, axis=-1)    # (N, K)
    numpyro.sample("y", dist.Categorical(logits=logits), obs=y_labels)
```

---

## 10. Deep Kernel Learning

**When**: Raw features are high-dimensional or unstructured (images, text
embeddings); a neural network learns a low-dimensional representation
where GP structure is meaningful.

### Model

```python
def model_dkl(X_raw, y):
    # Neural net feature extractor (parameters as numpyro.param)
    W1 = numpyro.param("W1", init_W1)
    b1 = numpyro.param("b1", init_b1)
    W2 = numpyro.param("W2", init_W2)
    b2 = numpyro.param("b2", init_b2)

    h = jax.nn.relu(X_raw @ W1 + b1)
    X_feat = h @ W2 + b2                              # (N, D_feat)

    # GP in learned feature space
    variance = numpyro.sample("variance", dist.LogNormal(0, 1))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0, 1))
    noise_var = numpyro.sample("noise_var", dist.LogNormal(-2, 1))

    kernel = RBF(variance=variance, lengthscale=lengthscale)
    gp_factor("gp", kernel, X_feat, y, noise_var, CholeskySolver())
```

### Notes

- `numpyro.param` sites are optimized (not sampled) even in MCMC.
  For fully Bayesian NN weights, replace with `numpyro.sample`.
- The Equinox path is cleaner here: define an `eqx.Module` feature
  extractor and compose with the GP.

---

## 11. Warped GP

**When**: Targets are non-negative, skewed, or bounded; a transformation
(warping function) maps them to a space where a GP + Gaussian likelihood
is appropriate.

### Model

```python
def model_warped(X, y_positive):
    # Warp: y_positive → z via Box-Cox or log
    # Here: simple log transform (for strictly positive y)
    z = jnp.log(y_positive)

    variance = numpyro.sample("variance", dist.LogNormal(0, 1))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0, 1))
    noise_var = numpyro.sample("noise_var", dist.LogNormal(-2, 1))

    kernel = RBF(variance=variance, lengthscale=lengthscale)
    gp_factor("gp", kernel, X, z, noise_var, CholeskySolver())
```

### Learnable Warp (Box-Cox)

```python
def model_boxcox(X, y):
    lam = numpyro.sample("lambda", dist.Normal(0, 1))

    # Box-Cox transform
    z = jnp.where(jnp.abs(lam) > 1e-6, (y**lam - 1) / lam, jnp.log(y))

    # Jacobian correction for the change of variables
    log_jac = (lam - 1) * jnp.sum(jnp.log(y))
    numpyro.factor("warp_jacobian", log_jac)

    kernel = RBF(...)
    noise_var = numpyro.sample("noise_var", dist.LogNormal(-2, 1))
    gp_factor("gp", kernel, X, z, noise_var, CholeskySolver())
```

---

## 12. Sparse Variational GP

**When**: N is too large for exact O(N³) Cholesky (N > ~5000).

Introduces M << N inducing points Z with inducing values u.

### Model

```python
def model_svgp(X, y, Z):
    """
    Parameters
    ----------
    X : (N, D)
    y : (N,)
    Z : (M, D)   inducing locations (fixed or learned)
    """
    variance = numpyro.sample("variance", dist.LogNormal(0, 1))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0, 1))
    noise_var = numpyro.sample("noise_var", dist.LogNormal(-2, 1))

    kernel = RBF(variance=variance, lengthscale=lengthscale)

    # Inducing prior: u ~ N(0, K_uu)
    K_uu = kernel(Z).matrix
    u = numpyro.sample("u", dist.MultivariateNormal(
        jnp.zeros(Z.shape[0]),
        K_uu + 1e-6 * jnp.eye(Z.shape[0]),
    ))

    # Conditional: f | u ~ N(K_fu K_uu⁻¹ u, ...)
    K_fu = kernel(X, Z).matrix                            # (N, M)
    L_uu = jnp.linalg.cholesky(K_uu + 1e-6 * jnp.eye(Z.shape[0]))
    alpha_u = solve_triangular(L_uu.T, solve_triangular(L_uu, u, lower=True), lower=False)
    f_mean = K_fu @ alpha_u                               # (N,)

    # For VFE/FITC, the conditional variance matters;
    # for simplicity here we use the mean-field approximation
    numpyro.sample("y", dist.Normal(f_mean, jnp.sqrt(noise_var)), obs=y)
```

### Guide

```python
guide = ComposedGuide(
    gp_guides={"u": InducingPointGuide(kernel_init, X, num_inducing=64)},
    model=model_svgp,
    auto_guide_cls=AutoNormal,
)
```

### Notes

- Cost: O(NM²) per step instead of O(N³).
- Supports mini-batching over N (subsample rows of X, y).
- Inducing locations Z can be learned (include in guide as params).

---

## 13. Temporal GP via State-Space

**When**: 1-D temporal data where the kernel has a state-space representation
(Matérn, periodic + Matérn, etc.).  O(N) instead of O(N³).

### Model

```python
def model_temporal(t, y):
    """t : (N,) sorted time points"""
    variance = numpyro.sample("variance", dist.LogNormal(0, 1))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0, 1))
    noise_var = numpyro.sample("noise_var", dist.LogNormal(-2, 1))

    kernel = Matern32(variance=variance, lengthscale=lengthscale)
    solver = KalmanSolver()

    # Collapsed: Kalman filter computes exact marginal likelihood in O(N)
    gp_factor("gp", kernel, t.reshape(-1, 1), y, noise_var, solver)
```

### Notes

- KalmanSolver converts the Matérn kernel to state-space form internally.
- Can be composed with other temporal kernels (sum of state-space models).
- The smoother provides the full posterior — useful as a guide for
  the temporal component in spatiotemporal models.

---

## 14. GP with Inducing Features (VISH / VFF)

**When**: The domain has natural eigenfunctions (sphere, bounded interval)
and you want sparse variational inference with diagonal K_uu.

### Model (VISH on S^2)

```python
def model_vish(X, y, L=10):
    """GP on the unit sphere with spherical harmonic inducing features.

    X: (N, 3) points on S^2
    L: max spherical harmonic degree -> M = (L+1)^2 inducing features
    """
    variance = numpyro.sample("variance", dist.LogNormal(0, 1))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0, 1))
    noise_var = numpyro.sample("noise_var", dist.LogNormal(-2, 1))

    kernel = RBF(variance=variance, lengthscale=lengthscale)
    M = (L + 1) ** 2
    features = SphericalHarmonicFeatures(max_degree=L)

    prior = GPPrior(kernel=kernel, solver=CholeskySolver(), X=X)
    f = gp_sample("f", prior, guide=SparseGuide(
        inducing_features=features,
        loc=jnp.zeros(M),
        scale_tril=0.1 * jnp.eye(M),
    ))
    numpyro.sample("y", dist.Normal(f, jnp.sqrt(noise_var)), obs=y)
```

### Model (VFF on bounded interval)

```python
def model_vff(X, y, M=50, L=0.6):
    """GP on [-L, L] with variational Fourier features."""
    kernel = Matern(nu=2.5)
    features = FourierFeatures(num_features=M, domain_half_width=L)

    prior = GPPrior(kernel=kernel, solver=CholeskySolver(), X=X)
    f = gp_sample("f", prior, guide=SparseGuide(
        inducing_features=features,
        loc=jnp.zeros(M),
        scale_tril=0.1 * jnp.eye(M),
    ))
    numpyro.sample("y", dist.Normal(f, 0.1), obs=y)
```

### Notes

- Inducing features use RKHS inner products u_m = <f, phi_m>_H as inducing
  variables. The full GP prior is preserved (no truncation).
- K_uu is diagonal for orthogonal eigenbases -> O(M) inversion, O(NM) cost.
- Unlike weight-space methods (RFF, HSGP), adding more features monotonically
  improves the variational bound (Titsias 2009).
- See [features/inducing_features.md](../features/inducing_features.md) for
  full design.

---

## 15. Missing Data / Partial Observations

**When**: Some entries of the observation vector are missing (sensor
dropouts, irregular coverage).

### Model

```python
def model_missing(X, y_partial, obs_mask):
    """
    Parameters
    ----------
    X : (N, D)
    y_partial : (N,)     observed values (NaN or 0 where missing)
    obs_mask : (N,)      boolean, True where observed
    """
    variance = numpyro.sample("variance", dist.LogNormal(0, 1))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0, 1))
    noise_var = numpyro.sample("noise_var", dist.LogNormal(-2, 1))

    kernel = RBF(variance=variance, lengthscale=lengthscale)
    solver = CholeskySolver()

    # GP latent at ALL locations (observed + missing)
    f = gp_sample("f", kernel, X, solver)

    # Likelihood only at observed locations
    with numpyro.handlers.mask(mask=obs_mask):
        numpyro.sample("y", dist.Normal(f, jnp.sqrt(noise_var)), obs=y_partial)
```

### Notes

- The GP prior automatically imputes at unobserved locations through
  the covariance structure.
- After inference, the posterior over f at missing locations gives
  the imputed values with uncertainty.

---

## 16. Marked Temporal Point Process + GP Intensity

**When**: Events arrive at irregular times with associated marks (magnitudes,
categories).  The intensity and/or mark distribution vary smoothly over time.

Relevant for: methane plume detection as a thinned marked temporal point
process, seismology, neuroscience spike trains.

### Model

```python
def model_marked_tpp(event_times, marks, T_max):
    """
    Parameters
    ----------
    event_times : (N_events,)    observed event times in [0, T_max]
    marks : (N_events,)          event magnitudes (e.g., emission rates)
    T_max : float                observation window length
    """
    # Grid for intensity approximation
    G = 200
    t_grid = jnp.linspace(0, T_max, G).reshape(-1, 1)

    # GP on log-intensity
    k_lambda = RBF(
        variance=numpyro.sample("var_lambda", dist.LogNormal(0, 1)),
        lengthscale=numpyro.sample("ls_lambda", dist.LogNormal(1, 1)),
    )
    log_lambda = gp_sample("log_lambda", k_lambda, t_grid, CholeskySolver())
    lambda_grid = jnp.exp(log_lambda)                     # (G,)

    # Log-likelihood of observed events: Σ log λ(t_i)
    # Interpolate grid to event times
    lambda_at_events = jnp.interp(event_times, t_grid.ravel(), lambda_grid)
    numpyro.factor("events", jnp.sum(jnp.log(lambda_at_events)))

    # Compensator: -∫₀ᵀ λ(t) dt  ≈  -(T_max/G) Σ λ(t_grid)
    integral = (T_max / G) * jnp.sum(lambda_grid)
    numpyro.factor("compensator", -integral)

    # Mark model: GP on log-scale of mark distribution
    k_mark = RBF(
        variance=numpyro.sample("var_mark", dist.LogNormal(-1, 1)),
        lengthscale=numpyro.sample("ls_mark", dist.LogNormal(1, 1)),
    )
    # Evaluate mark GP at event times
    log_mark_scale = gp_sample(
        "log_mark_scale", k_mark,
        event_times.reshape(-1, 1), CholeskySolver(),
    )

    numpyro.sample(
        "marks",
        dist.LogNormal(0, jnp.exp(log_mark_scale)),
        obs=marks,
    )
```

---

## Inference Decision Guide

When choosing between MCMC and SVI for any pattern:

| Criterion | MCMC | SVI |
|-----------|------|-----|
| Number of GP latent dimensions | < ~500 total | Any |
| Need exact posterior | Yes | Approximate OK |
| Non-conjugate likelihood | Works (but slow) | Fast with good guide |
| Multiple GP components | Expensive | Scales with good guides |
| Hyperparameter posterior | Exact | Point or approximate |
| Mini-batching needed | No (some exceptions) | Yes (via plates) |
| Wall time budget | Hours OK | Minutes preferred |

**Rule of thumb**: If the total number of latent GP function values
(sum of N across all GP components) is < 500, try MCMC first.
Otherwise, use SVI with structured guides.

**Hybrid approach**: Use SVI to find a good initialization, then run
short MCMC chains from the SVI optimum for proper uncertainty
quantification.

---

## 17. Pathwise Posterior Sampling (Matheron's Rule)

**When**: You need many independent posterior function samples evaluated
at many test points — Thompson sampling, posterior visualization, uncertainty
propagation through downstream models.

### Generative Story

Standard GP posterior sampling via `ConditionedGP.sample()` requires
forming the posterior covariance at test points and Cholesky-factorizing
it: $O(N_*^3)$. Matheron's rule avoids this by decomposing each posterior
sample into a prior draw + a linear correction that reuses the training-time
factorization.

### Model (exact GP)

```python
def model_and_sample(X, y, X_star, key):
    # 1. Standard GP training
    k = RBF(
        variance=numpyro.sample("var", dist.LogNormal(0, 1)),
        lengthscale=numpyro.sample("ls", dist.LogNormal(0, 1)),
    )
    noise = numpyro.sample("noise", dist.LogNormal(-2, 1))
    prior = GPPrior(kernel=k, solver=CholeskySolver(), X=X)
    gp_factor("gp", prior, y, noise)

    # 2. Condition and build pathwise sampler
    posterior = prior.condition(y, noise_var=noise)
    sampler = PathwiseSampler(posterior, n_features=512)

    # 3. Draw 100 posterior function samples — O(D + N) each
    paths = sampler.sample_paths(key, n_paths=100)
    f_samples = paths(X_star)          # (100, N_star)
    return f_samples
```

### Model (sparse GP, decoupled)

```python
def sparse_model_and_sample(X, y, X_star, key, Z):
    k = RBF(
        variance=numpyro.sample("var", dist.LogNormal(0, 1)),
        lengthscale=numpyro.sample("ls", dist.LogNormal(0, 1)),
    )
    prior = GPPrior(kernel=k, solver=WoodburySolver(), X=X)
    guide = SparseGuide(Z=Z, covariance="full_rank", whiten=True)
    f = gp_sample("f", prior, guide=guide)
    numpyro.sample("y", dist.Normal(f, 0.1), obs=y)

    # After SVI: decoupled pathwise sampling
    # Prior basis: 512 RFF features (stationary kernel)
    # Update basis: M inducing-point kernel evaluations
    sampler = DecoupledPathwiseSampler(guide, k, n_features=512)
    paths = sampler.sample_paths(key, n_paths=50)
    f_samples = paths(X_star)          # (50, N_star) — O(D + M) each
    return f_samples
```

### Thompson Sampling for Bayesian Optimization

```python
def thompson_sampling_step(posterior, key, X_candidates):
    # One posterior function draw
    sampler = PathwiseSampler(posterior, n_features=1024)
    path = sampler.sample_paths(key, n_paths=1)

    # Evaluate at all candidates — O(D + M) per candidate
    f_values = path(X_candidates)      # (1, N_candidates)

    # Select the argmax as next query point
    return X_candidates[jnp.argmax(f_values[0])]
```

### Notes

- **Stationary kernels only** for the RFF prior draw (RBF, Matern, Periodic).
  For non-stationary kernels, fall back to `ConditionedGP.sample()`.
- **More features = better approximation** of the prior. 512 is a good
  default; use 1024+ for high-dimensional inputs.
- The Cholesky factorization from training is reused — no additional
  cubic cost at prediction time.
- **gaussX dependency:** `gaussx.matheron_update` for the conditioning step.
  See [gaussx/features/pathwise_sampling.md](../../gaussx/features/pathwise_sampling.md).

---

## Guide Selection Guide

| GP Structure | Recommended Guide | When |
|-------------|-------------------|------|
| Exact GP, moderate N | `WhitenedMeanFieldGuide` | Default choice |
| Exact GP, MAP only | `WhitenedDeltaGuide` | Fast, no uncertainty on f |
| Exact GP, multimodal posterior | `WhitenedFlowGuide` | Rare, expensive |
| Large N (> 5000) | `InducingPointGuide` | Sparse approximation |
| Temporal (1-D, Matérn) | `KalmanGuide` | O(N) via smoother |
| Grid-structured | `KroneckerGuide` | Exploits grid structure |
| Multiple GP components | `ComposedGuide` | One sub-guide per GP site |

---

## Extension Checklist

To add a new pattern to pyrox.gp:

1. **Write the NumPyro model** using `gp_sample` and/or `gp_factor`.
2. **Choose or implement a guide** for each GP site.
3. **Compose with `ComposedGuide`** if there are non-GP latents.
4. **Test both MCMC and SVI** paths on synthetic data.
5. **Document** the pattern here with generative story, model code,
   guide code, inference options, and notes.
