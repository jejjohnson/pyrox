---
status: stable
version: 0.1.0
---

# pyrox.gp — Integrator Examples

> Worked examples showing how the Integrator protocol composes with
> inference strategies, solvers, and GP prediction routines.
>
> For mathematical foundations, see [`../../api/gp/integration.md`](../../api/gp/integration.md).

---

## Table of Contents

1. [Basic: Switching Integrators for VI](#1-switching-integrators-for-vi)
2. [Posterior Linearisation = PL + SigmaPoints + Kalman](#2-posterior-linearisation--pl--sigmapoints--kalman)
3. [Extended Kalman Smoother = PL + Taylor(1) + Kalman](#3-extended-kalman-smoother--pl--taylor1--kalman)
4. [Cubature Kalman Smoother = PL + Cubature + Kalman](#4-cubature-kalman-smoother--pl--cubature--kalman)
5. [GP Prediction with Uncertain Inputs (Sigma Points)](#5-gp-prediction-with-uncertain-inputs-sigma-points)
6. [GP Prediction with Uncertain Inputs (Exact RBF)](#6-gp-prediction-with-uncertain-inputs-exact-rbf)
7. [Multi-Step-Ahead Propagation (PILCO-style)](#7-multi-step-ahead-propagation-pilco-style)
8. [Heteroscedastic GP with PL + Cubature](#8-heteroscedastic-gp-with-pl--cubature)
9. [Expected Improvement for Bayesian Optimisation](#9-expected-improvement-for-bayesian-optimisation)
10. [Custom Integrator: Importance-Weighted MC](#10-custom-integrator-importance-weighted-mc)

---

## 1. Switching Integrators for VI

The simplest demonstration: same model, same strategy (VI), different
integrators.  Only the quadrature method changes.

### Model

```python
def model_vi_demo(t, y_binary):
    variance = numpyro.sample("variance", dist.LogNormal(0, 1))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0, 1))
    ss = Matern32(variance, lengthscale).to_state_space(t)
    solver = KalmanSolver()

    log_lik = lambda f, y: y * jax.nn.log_sigmoid(f) + (1 - y) * jax.nn.log_sigmoid(-f)

    # --- Swap integrator here, everything else identical ---
    integrator = GaussHermiteIntegrator(num_points=20)
    # integrator = SigmaPointIntegrator()
    # integrator = MonteCarloIntegrator(num_samples=200)
    # integrator = TaylorIntegrator(order=2)

    strategy = VIStrategy(integrator=integrator)
    result = temporal_inference(ss, y_binary, strategy, solver, log_lik)
    numpyro.factor("approx_lml", result.log_marginal_likelihood)
```

### What Changes

| Integrator | Site update quality | Cost per site | Notes |
|---|---|---|---|
| GaussHermite(20) | Excellent | 20 fn evals | Default for 1-D |
| SigmaPoints | Good (3rd order) | 3 fn evals | Overkill for 1-D, useful for multi-output |
| MC(200) | Noisy | 200 fn evals | Introduces gradient variance |
| Taylor(2) | Moderate | 1 fn eval + Hessian | Cheapest, but biased for strong nonlinearity |

For Bernoulli log-likelihood (smooth, log-concave), all integrators
converge.  The differences show up for heavy-tailed or multimodal
likelihoods.

---

## 2. Posterior Linearisation = PL + SigmaPoints + Kalman

The unscented Kalman smoother, derived as a composition of three
pyrox.gp components.

### Model

```python
def model_uks(t, y_counts):
    """Poisson counts via unscented Kalman smoother."""
    variance = numpyro.sample("variance", dist.LogNormal(0, 1))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(1, 1))
    ss = Matern32(variance, lengthscale).to_state_space(t)

    strategy = PLStrategy(
        integrator=SigmaPointIntegrator(),
        obs_mean_fn=jnp.exp,               # E[y|f] = exp(f) for Poisson
        obs_var_fn=jnp.exp,                 # Var[y|f] = exp(f) for Poisson
    )

    result = temporal_inference(
        ss_model=ss,
        y=y_counts,
        strategy=strategy,
        solver=KalmanSolver(),
        max_iters=20,
    )

    numpyro.factor("gp_lml", result.log_marginal_likelihood)
```

### Why PL + SigmaPoints

PL needs E_q[E[y|f]], E_q[Var[y|f]], and Cov_q[E[y|f], f].  The
sigma point integrator computes all three from the same 2P+1 points:

```python
# Inside PLStrategy.compute_sites:
pts, w_m, w_c = integrator.points_and_weights(m_n, C_n)

h_vals = vmap(obs_mean_fn)(pts)      # E[y|f] at each sigma point
h_bar = sum(w_m * h_vals)             # E_q[E[y|f]]

R_vals = vmap(obs_var_fn)(pts)       # Var[y|f] at each sigma point
R_bar = sum(w_m * R_vals)             # E_q[Var[y|f]]

# Cross-covariance from the same points
Cov_hf = sum(w_c * (h_vals - h_bar) * (pts - m_n))
```

No redundant point generation.

---

## 3. Extended Kalman Smoother = PL + Taylor(1) + Kalman

The cheapest non-conjugate temporal GP method.  Linearises the
observation model around the posterior mean using a first-order Taylor
expansion.

### Model

```python
def model_eks(t, y_binary):
    """Binary classification via extended Kalman smoother."""
    variance = numpyro.sample("variance", dist.LogNormal(0, 1))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0, 1))
    ss = Matern32(variance, lengthscale).to_state_space(t)

    strategy = PLStrategy(
        integrator=TaylorIntegrator(order=1),    # <-- this makes it an EKS
        obs_mean_fn=jax.nn.sigmoid,              # E[y|f] = σ(f)
        obs_var_fn=lambda f: jax.nn.sigmoid(f) * (1 - jax.nn.sigmoid(f)),
    )

    result = temporal_inference(ss, y_binary, strategy, KalmanSolver())
    numpyro.factor("gp_lml", result.log_marginal_likelihood)
```

### Why Taylor(1) Gives the EKS

With first-order Taylor, E_q[h(f)] ≈ h(m) and the cross-covariance
reduces to Cov_q[h(f), f] ≈ h'(m) · C.  This is exactly the
Jacobian-based linearisation of the extended Kalman filter.

---

## 4. Cubature Kalman Smoother = PL + Cubature + Kalman

Cubature points are simpler than sigma points (no tuning parameters)
and give the same 3rd-order accuracy.

### Model

```python
def model_cks(t, y_counts):
    """Poisson counts via cubature Kalman smoother."""
    variance = numpyro.sample("variance", dist.LogNormal(0, 1))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(1, 1))
    ss = Matern32(variance, lengthscale).to_state_space(t)

    strategy = PLStrategy(
        integrator=CubatureIntegrator(),         # 2P points, no tuning
        obs_mean_fn=jnp.exp,
        obs_var_fn=jnp.exp,
    )

    result = temporal_inference(ss, y_counts, strategy, KalmanSolver())
    numpyro.factor("gp_lml", result.log_marginal_likelihood)
```

---

## 5. GP Prediction with Uncertain Inputs (Sigma Points)

**When**: The test input x* is not known exactly but has a distribution
x* ~ N(μ_x, Σ_x).  Common in robotics (state estimation), sensor fusion,
and multi-step forecasting where outputs feed back as inputs.

### Setup

```python
# Train a standard GP
kernel = RBF(variance=1.0, lengthscale=1.0)
solver = CholeskySolver()
gp = ExactGP(kernel=kernel, fixed_noise=0.1)
gp.fit(X_train, y_train)
```

### Predict with uncertain input

```python
def predict_uncertain(gp, mu_x, Sigma_x, integrator):
    """
    Propagate input uncertainty through a GP posterior.

    Parameters
    ----------
    gp : ExactGP (fitted)
    mu_x : (D,) mean of input distribution
    Sigma_x : (D, D) covariance of input distribution
    integrator : IntegratorProtocol

    Returns
    -------
    E_f : scalar    expected predictive mean
    V_f : scalar    total predictive variance (aleatoric + epistemic from input)
    """
    def gp_mean(x):
        mu, _ = gp.predict(x.reshape(1, -1), full_cov=False)
        return mu.squeeze()

    def gp_var(x):
        _, var = gp.predict(x.reshape(1, -1), full_cov=False)
        return var.squeeze()

    # E_{p(x*)}[μ_f(x*)]
    E_f = integrator.integrate(gp_mean, mu_x, Sigma_x)

    # E_{p(x*)}[σ²_f(x*)]  — expected aleatoric + model uncertainty
    E_var = integrator.integrate(gp_var, mu_x, Sigma_x)

    # Var_{p(x*)}[μ_f(x*)]  — uncertainty from input noise
    E_mean_sq = integrator.integrate(lambda x: gp_mean(x)**2, mu_x, Sigma_x)
    V_mean = E_mean_sq - E_f**2

    # Law of total variance
    V_f = E_var + V_mean

    return E_f, V_f


# --- Usage ---
mu_x = jnp.array([2.5])
Sigma_x = jnp.array([[0.3]])

# Sigma points (recommended for D ≤ 5)
integrator = SigmaPointIntegrator()
E_f, V_f = predict_uncertain(gp, mu_x, Sigma_x, integrator)
print(f"E[f*] = {E_f:.4f}, Var[f*] = {V_f:.4f}")

# Compare: Taylor(1) — cheapest, ignores curvature
integrator_taylor = TaylorIntegrator(order=1)
E_f_t, V_f_t = predict_uncertain(gp, mu_x, Sigma_x, integrator_taylor)
print(f"Taylor(1): E[f*] = {E_f_t:.4f}, Var[f*] = {V_f_t:.4f}")

# Compare: MC — reference
integrator_mc = MonteCarloIntegrator(num_samples=10000)
E_f_mc, V_f_mc = predict_uncertain(gp, mu_x, Sigma_x, integrator_mc)
print(f"MC(10k):   E[f*] = {E_f_mc:.4f}, Var[f*] = {V_f_mc:.4f}")
```

### What the Different Integrators Capture

Consider a GP posterior with a curved mean function near μ_x:

- **Taylor(1)**: E[f*] ≈ μ_f(μ_x).  Completely ignores the curvature of μ_f
  and the spread of Σ_x.  Var[f*] ≈ σ²_f(μ_x) — no input uncertainty contribution.
- **Taylor(2)**: E[f*] ≈ μ_f(μ_x) + ½ tr(H_μ Σ_x).  Captures curvature bias
  but misses higher-order effects.
- **SigmaPoints**: Evaluates μ_f at 2D+1 points spread by Σ_x.  Captures
  mean shift from curvature + variance inflation from nonlinearity.
  Exact for quadratic μ_f.
- **GaussHermite**: More points, higher accuracy.  Exact for polynomial μ_f
  up to degree 2K−1.
- **MC**: Converges to the truth but noisy.

For GP mean functions (which are smooth but generally nonlinear), sigma
points are usually sufficient and much cheaper than MC.

---

## 6. GP Prediction with Uncertain Inputs (Exact RBF)

For the RBF kernel specifically, the expected GP mean and variance under
Gaussian input uncertainty have **closed-form expressions** (Girard et al.,
2003; Deisenroth & Rasmussen, 2011).

### Exact Expected Mean

$$
\mathbb{E}_{p(x_*)}[\mu_f(x_*)] = \alpha^\top \tilde{k}
$$

where α = K_y⁻¹ y and:

$$
\tilde{k}_i = \sigma^2\,|I + \Sigma_x \Lambda^{-1}|^{-1/2}\,\exp\!\left(-\frac{1}{2}(\bar{x}_i)^\top(\Lambda + \Sigma_x)^{-1}\bar{x}_i\right)
$$

with x̄ᵢ = xᵢ − μ_x and Λ = diag(ℓ₁², ..., ℓ_D²).

### Exact Expected Variance (Law of Total Variance)

$$
\text{Var}[f_*] = \sigma^2 - \text{tr}\!\left((K_y^{-1} - \alpha\alpha^\top)\,\tilde{K}\right) + \mathbb{E}[\mu_f^2] - (\mathbb{E}[\mu_f])^2
$$

where K̃ has entries involving the expected product of two kernel evaluations:

$$
\tilde{K}_{ij} = \sigma^4\,|I + 2\Sigma_x\Lambda^{-1}|^{-1/2}\,\exp\!\left(-\frac{1}{2}\bar{z}_{ij}^\top(0.5\Lambda + \Sigma_x)^{-1}\bar{z}_{ij} - \frac{1}{4}\Delta_{ij}^\top \Lambda^{-1} \Delta_{ij}\right)
$$

with z̄ᵢⱼ = ½(xᵢ + xⱼ) − μ_x and Δᵢⱼ = xᵢ − xⱼ.

### Implementation

```python
def predict_uncertain_rbf_exact(
    alpha,           # (N,)     K_y⁻¹ y
    K_y_inv,         # (N,N)    K_y⁻¹
    X_train,         # (N, D)
    mu_x,            # (D,)
    Sigma_x,         # (D, D)
    variance,        # scalar   kernel σ²
    lengthscale,     # (D,)     kernel ℓ
):
    """Closed-form E[f*] and Var[f*] for RBF kernel with uncertain inputs."""
    N, D = X_train.shape
    Lambda = jnp.diag(lengthscale**2)

    # --- E[μ_f(x*)] ---
    Lambda_plus_Sigma = Lambda + Sigma_x
    LS_inv = jnp.linalg.inv(Lambda_plus_Sigma)
    det_factor = jnp.linalg.det(jnp.eye(D) + Sigma_x @ jnp.linalg.inv(Lambda))

    diffs = X_train - mu_x[None, :]                              # (N, D)
    exponents = -0.5 * jnp.sum(diffs @ LS_inv * diffs, axis=1)   # (N,)
    k_tilde = variance / jnp.sqrt(det_factor) * jnp.exp(exponents)
    E_mean = jnp.dot(alpha, k_tilde)

    # --- E[μ_f(x*)²] via K̃ matrix ---
    half_Lambda_plus_Sigma = 0.5 * Lambda + Sigma_x
    hLS_inv = jnp.linalg.inv(half_Lambda_plus_Sigma)
    det_factor_2 = jnp.linalg.det(jnp.eye(D) + 2 * Sigma_x @ jnp.linalg.inv(Lambda))
    prefactor_2 = variance**2 / jnp.sqrt(det_factor_2)

    # Pairwise computations
    z_bar = 0.5 * (X_train[:, None, :] + X_train[None, :, :]) - mu_x  # (N, N, D)
    delta = X_train[:, None, :] - X_train[None, :, :]                   # (N, N, D)

    exp_z = -0.5 * jnp.sum((z_bar @ hLS_inv) * z_bar, axis=-1)
    exp_d = -0.25 * jnp.sum((delta @ jnp.linalg.inv(Lambda)) * delta, axis=-1)
    K_tilde = prefactor_2 * jnp.exp(exp_z + exp_d)                     # (N, N)

    E_mean_sq = alpha @ K_tilde @ alpha

    # --- Law of total variance ---
    # E[σ²_f(x*)] = σ² - tr((K_y⁻¹ - ααᵀ) K̃)
    E_model_var = variance - jnp.trace((K_y_inv - jnp.outer(alpha, alpha)) @ K_tilde)
    Var_mean = E_mean_sq - E_mean**2
    total_var = E_model_var + Var_mean

    return E_mean, total_var
```

### When to Use Exact vs Integrator

| Scenario | Recommendation |
|----------|---------------|
| RBF kernel, D ≤ 5 | Exact (this function) — no approximation error |
| Matérn or other kernels | SigmaPointIntegrator or GaussHermite |
| ARD RBF with D > 10 | MC integrator (K̃ matrix is N² × O(D) per entry) |
| Need cross-covariance E[f* f*'] | Exact formula exists (Deisenroth 2011) |

---

## 7. Multi-Step-Ahead Propagation (PILCO-style)

**When**: Autoregressive forecasting where each prediction's uncertainty
feeds into the next step's input distribution.  The hallmark of
model-based reinforcement learning (Deisenroth & Rasmussen, 2011).

### The Propagation Loop

At each step h = 1, ..., H:

1. **Input distribution**: p(x_h) = N(μ_h, Σ_h)
2. **Propagate through GP**: E[f_h], Var[f_h] via uncertain input prediction
3. **Next input**: μ_{h+1} = E[f_h], Σ_{h+1} = Var[f_h]

The uncertainty **grows** at each step because the output uncertainty of
step h becomes the input uncertainty of step h+1.

### Implementation

```python
def multi_step_propagation(
    gp,                   # fitted ExactGP
    x0: jnp.ndarray,     # (D,) initial state (known exactly)
    H: int,              # prediction horizon
    integrator: IntegratorProtocol,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Autoregressive multi-step prediction with uncertainty propagation.

    Returns
    -------
    means : (H, D)   predicted means at each step
    vars : (H, D)    predicted variances at each step
    """
    mu = x0
    Sigma = 1e-8 * jnp.eye(x0.shape[0])   # start with ~zero uncertainty

    means, variances = [], []
    for h in range(H):
        E_f, V_f = predict_uncertain(gp, mu, Sigma, integrator)
        means.append(E_f)
        variances.append(V_f)

        # Next step: output becomes input
        mu = jnp.atleast_1d(E_f)
        Sigma = jnp.atleast_2d(V_f)

    return jnp.stack(means), jnp.stack(variances)


# Usage
integrator = SigmaPointIntegrator()   # good balance of speed and accuracy
means, vars = multi_step_propagation(gp, x0=jnp.array([1.0]), H=20, integrator)
```

### As a `jax.lax.scan`

```python
def multi_step_scan(gp, x0, H, integrator):
    """Scan-based multi-step propagation (JIT-friendly)."""

    def step(carry, _):
        mu, Sigma = carry
        E_f, V_f = predict_uncertain(gp, mu, Sigma, integrator)
        mu_next = jnp.atleast_1d(E_f)
        Sigma_next = jnp.atleast_2d(V_f)
        return (mu_next, Sigma_next), (E_f, V_f)

    init = (x0, 1e-8 * jnp.eye(x0.shape[0]))
    _, (means, vars) = jax.lax.scan(step, init, jnp.arange(H))
    return means, vars
```

### Comparison of Integrators for Multi-Step Propagation

The error **compounds** across steps, so integrator quality matters
more here than in single-step prediction:

| Integrator | 1-step error | 20-step error | Notes |
|---|---|---|---|
| Taylor(1) | Small | Large (bias accumulates) | Underestimates uncertainty growth |
| Taylor(2) | Moderate | Moderate | Better, but still local |
| SigmaPoints | Small | Small | Recommended for D ≤ 5 |
| Exact RBF | Zero | Small (moment-matching error) | Best for RBF, accumulates Gaussian approx error |
| MC(1000) | Small (noisy) | Moderate (variance grows) | Reference, but expensive |

Taylor(1) is particularly dangerous for multi-step propagation because
it systematically underestimates the variance growth, leading to
overconfident long-horizon predictions.

---

## 8. Heteroscedastic GP with PL + Cubature

**When**: Two coupled latent GPs (mean and noise), with uncertainty
propagation through the noise model.

### Model

```python
def model_hetero_pl(t, y):
    """Heteroscedastic temporal GP using posterior linearisation."""
    # Mean GP
    ss_f = Matern32(
        numpyro.sample("var_f", dist.LogNormal(0, 1)),
        numpyro.sample("ls_f", dist.LogNormal(0, 1)),
    ).to_state_space(t)

    # Log-noise GP
    ss_g = Matern32(
        numpyro.sample("var_g", dist.LogNormal(-1, 1)),
        numpyro.sample("ls_g", dist.LogNormal(0, 1)),
    ).to_state_space(t)

    # Stack into joint state-space model
    ss_joint = stack_state_space(ss_f, ss_g)
    # State: [x_f; x_g],  Emission: H_joint extracts [f, g] from state

    # Joint observation model: y ~ N(f, exp(g))
    # This is a 2-D latent → 1-D observation model
    def obs_mean_fn(fg):
        f, g = fg[0], fg[1]
        return f                    # E[y | f, g] = f

    def obs_var_fn(fg):
        f, g = fg[0], fg[1]
        return jnp.exp(g)           # Var[y | f, g] = exp(g)

    strategy = PLStrategy(
        integrator=CubatureIntegrator(),   # 2D latent → 4 cubature points
        obs_mean_fn=obs_mean_fn,
        obs_var_fn=obs_var_fn,
    )

    result = temporal_inference(ss_joint, y, strategy, KalmanSolver())
    numpyro.factor("gp_lml", result.log_marginal_likelihood)
```

### Why Cubature Here

The latent is 2-dimensional (f, g), so Gauss-Hermite would use K² points.
Cubature uses exactly 4 points (2 × 2D) with no tuning parameters,
which is perfect for this low-dimensional case.

---

## 9. Expected Improvement for Bayesian Optimisation

**When**: Using a GP surrogate for Bayesian optimisation.  The acquisition
function (expected improvement) is an expectation under the GP posterior.

With uncertain input x* ~ N(μ_x, Σ_x) (e.g., from noisy optimisation
of the acquisition function, or averaging over input noise):

### Standard EI (known x*)

```python
def expected_improvement(gp, x_star, f_best):
    """Standard EI at a known point x*."""
    mu, var = gp.predict(x_star.reshape(1, -1))
    sigma = jnp.sqrt(var.squeeze())
    z = (f_best - mu.squeeze()) / sigma
    ei = sigma * (z * jax.scipy.stats.norm.cdf(z) + jax.scipy.stats.norm.pdf(z))
    return ei
```

### EI with Uncertain Input

```python
def expected_improvement_uncertain(gp, mu_x, Sigma_x, f_best, integrator):
    """E_{p(x*)}[EI(x*)] — expected improvement averaged over input uncertainty."""
    def ei_at_x(x):
        return expected_improvement(gp, x, f_best)

    return integrator.integrate(ei_at_x, mu_x, Sigma_x)
```

### EI as a Nested Expectation (Predictive EI)

The "full" expected improvement under GP posterior uncertainty AND
input uncertainty:

```python
def predictive_ei(gp, mu_x, Sigma_x, f_best, integrator_outer, integrator_inner):
    """
    Nested expectation:
    E_{p(x*)}[ E_{p(f*|x*)}[ max(f_best - f*, 0) ] ]

    outer: over input uncertainty (sigma points or exact RBF)
    inner: over GP posterior (Gauss-Hermite on p(f*|x*))
    """
    def inner_ei(x):
        mu, var = gp.predict(x.reshape(1, -1))
        mu, sigma = mu.squeeze(), jnp.sqrt(var.squeeze())

        def improvement(f):
            return jnp.maximum(f_best - f, 0.0)

        return integrator_inner.integrate(improvement, mu, sigma**2)

    return integrator_outer.integrate(inner_ei, mu_x, Sigma_x)
```

---

## 10. Custom Integrator: Importance-Weighted MC

Showing that the protocol is extensible: a custom integrator for
cases where the Gaussian proposal is a poor match for the integrand.

### Implementation

```python
class ImportanceWeightedMCIntegrator:
    """
    Importance-weighted MC using a heavier-tailed proposal.

    Useful when g(f) has significant mass in the tails of q(f),
    e.g., rare event probabilities under a GP.

    Proposal: Student-t with ν degrees of freedom, matching
    the mean and scale of q(f).
    """

    def __init__(self, num_samples: int = 500, nu: float = 5.0, key=None):
        self.S = num_samples
        self.nu = nu
        self.key = key or jax.random.PRNGKey(0)

    def integrate(self, fn, mean, cov):
        std = jnp.sqrt(cov) if cov.ndim == 0 else jnp.sqrt(jnp.diag(cov))

        # Sample from Student-t proposal
        key1, key2 = jax.random.split(self.key)
        # t(ν) = Normal / sqrt(χ²(ν)/ν)
        z = jax.random.normal(key1, (self.S,))
        chi2 = jax.random.gamma(key2, self.nu / 2, (self.S,)) * 2
        t_samples = z / jnp.sqrt(chi2 / self.nu)
        samples = mean + std * t_samples

        # Importance weights: q(f) / proposal(f)
        log_q = -0.5 * ((samples - mean) / std)**2 - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
        log_prop = jax.scipy.stats.t.logpdf(t_samples, self.nu)
        log_weights = log_q - log_prop
        weights = jax.nn.softmax(log_weights)  # self-normalised

        vals = jax.vmap(fn)(samples)
        return jnp.sum(weights * vals)
```

### Usage: Rare Event Probability

```python
# P(f > threshold) under posterior — relevant for exceedance probabilities
def exceedance_prob(gp, x, threshold, integrator):
    mu, var = gp.predict(x.reshape(1, -1))
    mu, sigma = mu.squeeze(), jnp.sqrt(var.squeeze())

    indicator = lambda f: (f > threshold).astype(float)
    return integrator.integrate(indicator, mu, sigma**2)

# Standard MC undersamples the tail
prob_mc = exceedance_prob(gp, x, 3.0, MonteCarloIntegrator(1000))

# Importance-weighted MC concentrates samples in the tail
prob_is = exceedance_prob(gp, x, 3.0, ImportanceWeightedMCIntegrator(1000, nu=3.0))
```

---

## Integrator Selection Quick Reference

| Use case | Recommended | Why |
|----------|------------|-----|
| 1-D site updates (VI, EP) | GaussHermite(20) | High accuracy, fast |
| Multi-output sites (P=2–5) | Cubature or SigmaPoints | Linear in P |
| Uncertain input prediction (D ≤ 3) | SigmaPoints | Good accuracy |
| Uncertain input prediction (RBF kernel) | ExactRBFIntegrator | Zero approximation error |
| Multi-step propagation (PILCO) | SigmaPoints or Exact RBF | Error compounds; need accuracy |
| High-D latent (P > 10) | MC | Only option that doesn't explode |
| Cheapest possible | Taylor(1) | 1 evaluation, poor for nonlinear g |
| Tail probabilities | ImportanceWeightedMC | Standard MC misses tails |
| EKS / EKF equivalent | Taylor(1) + PL + Kalman | Classical signal processing |
| UKS / UKF equivalent | SigmaPoints + PL + Kalman | Better than EKS for nonlinear |
| CKS / CKF equivalent | Cubature + PL + Kalman | Simpler than UKS, same accuracy |
