---
status: draft
version: 0.1.0
---

# pyrox.gp — Temporal Inference: Markovian GPs, Kalman Methods, and the Bayes-Newton Integration

> Design document covering the mathematical foundations of temporal GP inference,
> the Bayes-Newton perspective on approximate inference as Newton's method, and
> how these algorithms integrate into pyrox.gp via the `InferenceStrategy` protocol.
>
> For worked NumPyro examples with `scan`, see [`../../examples/gp/state_space.md`](../../examples/gp/state_space.md).

---

## Table of Contents

1. [The Markovian GP Perspective](#1-the-markovian-gp-perspective)
2. [Conjugate Case: Kalman Filter & RTS Smoother](#2-conjugate-case-kalman-filter--rts-smoother)
3. [Non-Conjugate Case: The Core Problem](#3-non-conjugate-case-the-core-problem)
4. [The Bayes-Newton Unification](#4-the-bayes-newton-unification)
5. [Inference Algorithm Catalog](#5-inference-algorithm-catalog)
6. [The InferenceStrategy Protocol](#6-the-inferencestrategy-protocol)
7. [Integration with numpyro.scan](#7-integration-with-numpyroscan)
8. [Sparse Markov GPs](#8-sparse-markov-gps)
9. [Parallel Kalman via Associative Scan](#9-parallel-kalman-via-associative-scan)
10. [Class Hierarchy & Module Layout](#10-class-hierarchy--module-layout)

---

## 1. The Markovian GP Perspective

Any GP with a stationary covariance function on ℝ (i.e., a temporal GP)
can be exactly represented as a linear time-invariant (LTI) stochastic
differential equation (SDE):

$$
dx(t) = F\,x(t)\,dt + L\,d\beta(t)
$$

$$
f(t) = H\,x(t)
$$

where x(t) ∈ ℝˢ is the state, β(t) is a Brownian motion with diffusion
matrix Q_c, and the GP value f(t) is a linear readout of the state.

Discretising at observation times t₁ < t₂ < ··· < t_N with
Δtₙ = tₙ₊₁ − tₙ gives the discrete state-space model:

$$
\text{(SS1)} \quad x_{n+1} = A_n\,x_n + q_n, \qquad q_n \sim \mathcal{N}(0, Q_n)
$$

$$
\text{(SS2)} \quad f_n = H\,x_n
$$

where the transition matrix and process noise are:

$$
A_n = \exp(F\,\Delta t_n), \qquad Q_n = P_\infty - A_n\,P_\infty\,A_n^\top
$$

and P_∞ is the stationary state covariance (solution of the continuous
Lyapunov equation FP + PFᵀ + LQ_cLᵀ = 0).

### State dimensions for common kernels

| Kernel | State dim S | F, H structure |
|--------|:-----------:|----------------|
| Matérn-1/2 (Exponential) | 1 | F = −1/ℓ, H = σ |
| Matérn-3/2 | 2 | Companion form |
| Matérn-5/2 | 3 | Companion form |
| RBF (approx.) | ~12–20 | Taylor/spectral approx (Hartikainen & Särkkä, 2010) |
| Periodic × Matérn | 2p (p harmonics × Matérn state dim) | Block-diagonal |
| Sum of kernels | S₁ + S₂ + ··· | Block-diagonal concatenation |

The key insight: once in state-space form, all GP operations (inference,
prediction, sampling) reduce to **Kalman filtering and smoothing**, which
are O(NS³) instead of O(N³).

### Multi-output / coupled state-space models

For P output dimensions with a shared temporal structure:

$$
x_n \in \mathbb{R}^{PS}, \qquad H \in \mathbb{R}^{P \times PS}
$$

The state dimension grows linearly in P, and the Kalman operations scale
as O(N(PS)³).  For large P, sparse Markov methods (§8) become necessary.

---

## 2. Conjugate Case: Kalman Filter & RTS Smoother

When the observation model is Gaussian:

$$
\text{(SS3)} \quad y_n = f_n + \varepsilon_n, \qquad \varepsilon_n \sim \mathcal{N}(0, R_n)
$$

the posterior p(x₁:N | y₁:N) is jointly Gaussian and computable exactly
via the Kalman filter (forward) and RTS smoother (backward).

### 2.1 Forward Pass: Kalman Filter

Initialise: x̂₀|₀ = x₀, P₀|₀ = P₀.

For n = 1, ..., N:

**Predict** (propagate prior through dynamics):
$$
\hat{x}_{n|n-1} = A_{n-1}\,\hat{x}_{n-1|n-1}
$$
$$
P_{n|n-1} = A_{n-1}\,P_{n-1|n-1}\,A_{n-1}^\top + Q_{n-1}
$$

**Update** (incorporate observation yₙ):
$$
v_n = y_n - H\,\hat{x}_{n|n-1} \qquad \text{(innovation)}
$$
$$
S_n = H\,P_{n|n-1}\,H^\top + R_n \qquad \text{(innovation covariance)}
$$
$$
K_n = P_{n|n-1}\,H^\top\,S_n^{-1} \qquad \text{(Kalman gain)}
$$
$$
\hat{x}_{n|n} = \hat{x}_{n|n-1} + K_n\,v_n
$$
$$
P_{n|n} = P_{n|n-1} - K_n\,S_n\,K_n^\top
$$

**Log marginal likelihood** (accumulated during filtering):
$$
\log p(y_{1:N}) = -\frac{1}{2}\sum_{n=1}^N \left[ v_n^\top S_n^{-1} v_n + \log|S_n| + \dim(y_n)\log(2\pi) \right]
$$

### 2.2 Backward Pass: RTS Smoother

The smoother computes the full posterior p(xₙ | y₁:N) for all n.

For n = N−1, ..., 1:

$$
G_n = P_{n|n}\,A_n^\top\,P_{n+1|n}^{-1} \qquad \text{(smoother gain)}
$$
$$
\hat{x}_{n|N} = \hat{x}_{n|n} + G_n\,(\hat{x}_{n+1|N} - A_n\,\hat{x}_{n|n})
$$
$$
P_{n|N} = P_{n|n} + G_n\,(P_{n+1|N} - P_{n+1|n})\,G_n^\top
$$

The function posterior is then:

$$
\mathbb{E}[f_n | y_{1:N}] = H\,\hat{x}_{n|N}, \qquad \text{Var}[f_n | y_{1:N}] = H\,P_{n|N}\,H^\top
$$

### 2.3 Complexity Summary (Conjugate)

| Operation | Cost | Storage |
|-----------|------|---------|
| Filter (forward) | O(NS³) | O(NS²) for stored states |
| Smoother (backward) | O(NS³) | O(NS²) |
| Log marginal likelihood | O(NS³) | Free (accumulated in filter) |
| Prediction at new times | O(N*S³) | Interpolation between filter states |

Compare with dense GP: O(N³) for Cholesky.  For Matérn-3/2 (S=2), the
Kalman approach is O(8N) vs O(N³) — a massive win for long time series.

---

## 3. Non-Conjugate Case: The Core Problem

When the likelihood is not Gaussian — e.g., Poisson, Bernoulli,
Student-t, GEV — the posterior p(f₁:N | y₁:N) is no longer Gaussian
and the Kalman filter cannot be applied directly.

The key idea shared by all approximate inference methods: replace the
true likelihood p(yₙ | fₙ) with a **local Gaussian approximation**
(called a "site"):

$$
\text{(Site)} \quad \tilde{p}(y_n | f_n) \propto \mathcal{N}(f_n \mid \tilde{y}_n, \tilde{R}_n)
$$

characterised by pseudo-observation ỹₙ and pseudo-noise R̃ₙ.
With these sites in place, the problem becomes a linear-Gaussian
state-space model with observations ỹₙ and noise R̃ₙ, which the
Kalman filter solves exactly.

**All temporal inference algorithms amount to different strategies for
computing and iterating the sites (ỹₙ, R̃ₙ).**

In natural parameter form, each site contributes:

$$
\tilde{\lambda}_n = \tilde{R}_n^{-1}\,\tilde{y}_n \qquad \text{(natural mean / site precision × pseudo-obs)}
$$
$$
\tilde{\Pi}_n = \tilde{R}_n^{-1} \qquad \text{(site precision)}
$$

The posterior natural parameters are then the prior natural parameters
plus the sum of all site natural parameters (by Gaussian conjugacy of
the approximate model).

---

## 4. The Bayes-Newton Unification

Wilkinson, Särkkä & Solin (JMLR 2023) show that VI, EP, Laplace, and
posterior linearisation are all instances of the following template:

### The General Update Rule

Given the current posterior approximation q(fₙ) = N(mₙ, Cₙ) at site n,
compute new site natural parameters:

$$
\tilde{\Pi}_n^{\text{new}} = -\nabla^2_n \qquad \text{(negative Hessian of some target)}
$$
$$
\tilde{\lambda}_n^{\text{new}} = \nabla_n + \tilde{\Pi}_n^{\text{new}}\,m_n \qquad \text{(gradient + precision × mean)}
$$

where ∇ₙ and ∇²ₙ are the gradient and Hessian of a **target function**
evaluated at the current posterior mean mₙ.  The different inference
algorithms differ only in their choice of target function and Hessian
approximation:

| Algorithm | Target ℒₙ | Hessian ∇²ₙ |
|-----------|-----------|-------------|
| **Newton / Laplace** | log p(yₙ \| fₙ) | Exact: ∂²log p(yₙ\|fₙ)/∂fₙ² |
| **Variational Inference** | E_q[log p(yₙ \| fₙ)] | Exact Hessian of expected log-lik |
| **Expectation Propagation** | log Zₙ (cavity marginal likelihood) | Moment matching ⇒ effective Hessian |
| **Posterior Linearisation** | log N(yₙ; E_q[E[yₙ\|fₙ]], Ωₙ) | Linearised observation Hessian |
| **Gauss-Newton** | Same as parent | J^T W J (Fisher / GGN approx) |
| **Quasi-Newton** | Same as parent | Low-rank BFGS update |
| **Taylor / EKS** | log p(yₙ \| fₙ) | Jacobian of observation function |

### Why This Matters

1. **PSD guarantees**: The Gauss-Newton Hessian approximation is always
   PSD (it's JᵀWJ with W ≥ 0).  Standard Newton/Laplace can produce
   indefinite Hessians for non-log-concave likelihoods, causing the
   posterior covariance to go negative.  Gauss-Newton variants prevent this.

2. **Unified implementation**: The entire algorithm is:
   ```
   for iteration in range(max_iters):
       for n in 1..N:
           (∇, ∇²) = compute_site_gradients(method, likelihood, m_n, C_n, y_n)
           update sites (ỹₙ, R̃ₙ) from (∇, ∇²)
       run Kalman filter + smoother with sites → new q(f)
   ```
   Swapping the inference method changes only `compute_site_gradients`.

3. **The Kalman filter is the inner solver**: Regardless of which
   inference algorithm computes the sites, the linear algebra that
   combines sites with the prior is always the Kalman filter/smoother.
   This is exactly the separation pyrox.gp needs: the `KalmanSolver`
   handles the linear algebra, and the `InferenceStrategy` handles
   the site computations.

---

## 5. Inference Algorithm Catalog

### 5.1 Newton / Laplace

The classical Laplace approximation.  Find the MAP of the posterior, then
approximate it with a Gaussian centered at the MAP with covariance equal
to the inverse Hessian.

**Site update**:
$$
\tilde{\Pi}_n = -\frac{\partial^2 \log p(y_n | f_n)}{\partial f_n^2}\bigg|_{f_n = m_n}
$$
$$
\tilde{\lambda}_n = \frac{\partial \log p(y_n | f_n)}{\partial f_n}\bigg|_{f_n = m_n} + \tilde{\Pi}_n\,m_n
$$

**Properties**:
- Converges in 3–10 iterations for log-concave likelihoods.
- Can fail (negative Π̃ₙ) for non-log-concave likelihoods.
- No PSD guarantee.
- Cheapest per iteration: only needs ∂log p/∂f and ∂²log p/∂f².

### 5.2 Variational Inference (Natural Gradient VI)

Uses the expected log-likelihood under q(fₙ) as the target.

**Site update**:
$$
\tilde{\Pi}_n = -\frac{\partial^2}{\partial m_n^2} \mathbb{E}_{q(f_n)}\!\left[\log p(y_n | f_n)\right]
$$
$$
\tilde{\lambda}_n = \frac{\partial}{\partial m_n} \mathbb{E}_{q(f_n)}\!\left[\log p(y_n | f_n)\right] + \tilde{\Pi}_n\,m_n
$$

The expectations are computed via Gauss-Hermite quadrature or
Bonnet's/Price's theorems (for certain likelihoods):

$$
\frac{\partial}{\partial m_n}\mathbb{E}_{q}\!\left[\log p(y_n|f_n)\right] = \mathbb{E}_{q}\!\left[\frac{\partial \log p(y_n|f_n)}{\partial f_n}\right] \qquad \text{(Bonnet)}
$$

$$
\frac{\partial^2}{\partial m_n^2}\mathbb{E}_{q}\!\left[\log p(y_n|f_n)\right] = \mathbb{E}_{q}\!\left[\frac{\partial^2 \log p(y_n|f_n)}{\partial f_n^2}\right] \qquad \text{(Price)}
$$

**Properties**:
- Equivalent to natural gradient descent on the ELBO.
- No PSD guarantee (the expected Hessian can be indefinite).
- Requires quadrature (adds cost per site update).
- Minimizes KL[q ‖ p] — mode-seeking.

### 5.3 Expectation Propagation

Matches moments of the tilted distribution pₙ(fₙ) ∝ q₋ₙ(fₙ) p(yₙ|fₙ)
where q₋ₙ is the cavity distribution (posterior with site n removed).

**Site update** (via moment matching):

Compute cavity: q₋ₙ(fₙ) = N(fₙ | m₋ₙ, C₋ₙ) where
$$
C_{-n}^{-1} = C_n^{-1} - \tilde{\Pi}_n, \qquad m_{-n} = C_{-n}(C_n^{-1}m_n - \tilde{\lambda}_n)
$$

Compute tilted moments via quadrature:
$$
\hat{m}_n = \mathbb{E}_{p_n}[f_n], \qquad \hat{C}_n = \text{Var}_{p_n}[f_n], \qquad \hat{Z}_n = \int q_{-n}(f_n)\,p(y_n|f_n)\,df_n
$$

Update sites:
$$
\tilde{\Pi}_n^{\text{new}} = \hat{C}_n^{-1} - C_{-n}^{-1}, \qquad \tilde{\lambda}_n^{\text{new}} = \hat{C}_n^{-1}\hat{m}_n - C_{-n}^{-1}m_{-n}
$$

**Properties**:
- Minimizes KL[p ‖ q] — mass-covering (better calibrated uncertainty).
- No PSD guarantee (site precision can go negative).
- Most expensive per iteration (cavity computation + quadrature).
- Often gives the best posterior approximation empirically.

### 5.4 Posterior Linearisation

Linearise the observation model around the current posterior mean:

$$
y_n \approx H_n^{\text{lin}}\,f_n + r_n, \qquad r_n \sim \mathcal{N}(0, \Omega_n)
$$

where:
$$
H_n^{\text{lin}} = \frac{\mathbb{E}_{q(f_n)}[\text{Cov}[y_n, f_n | f_n]]}{\text{Var}_{q}[f_n]}
$$
$$
\Omega_n = \mathbb{E}_{q(f_n)}[\text{Var}[y_n | f_n]] + \text{Var}_{q(f_n)}[\mathbb{E}[y_n | f_n]] - (H_n^{\text{lin}})^2\,\text{Var}_q[f_n]
$$

This directly gives pseudo-observations for the Kalman filter:
$$
\tilde{y}_n = y_n, \qquad \tilde{R}_n = \Omega_n / (H_n^{\text{lin}})^2
$$

**Properties**:
- Closely related to the unscented Kalman smoother (UKS).
- PSD guaranteed (Ωₙ is a covariance, always PSD).
- Natural for likelihoods where E[y|f] and Var[y|f] are known.
- The Taylor expansion variant (∂E[y|f]/∂f evaluated at mₙ) gives
  the extended Kalman smoother (EKS).

### 5.5 Gauss-Newton Variants

Replace the exact Hessian in any of the above with the Gauss-Newton
(Fisher information / generalized Gauss-Newton) approximation:

$$
-\nabla^2 \log p(y_n | f_n) \approx J_n^\top\,W_n\,J_n
$$

where Jₙ = ∂g/∂f (Jacobian of the observation function) and
Wₙ = −∂²L/∂z² (Hessian of the loss w.r.t. the predicted output z = g(f)).

Since JᵀWJ is always PSD (W ≥ 0 for proper losses), this guarantees
PSD site precisions → PSD posterior covariance.

Applicable to: Newton-GN, VI-GN, EP-GN (PEP), PL-GN (2nd order PL).

### 5.6 Quasi-Newton (BFGS) Variants

Replace the Hessian with a rank-2 BFGS update:

$$
B_{k+1} = B_k + \frac{\delta_k \delta_k^\top}{\delta_k^\top \gamma_k} - \frac{B_k \gamma_k \gamma_k^\top B_k}{\gamma_k^\top B_k \gamma_k}
$$

where δₖ = mₖ₊₁ − mₖ and γₖ = ∇ₖ₊₁ − ∇ₖ.

**Properties**:
- PSD guaranteed (BFGS preserves PSD).
- Only needs first derivatives (no Hessian computation).
- Slower convergence than Newton but cheaper per step.

### 5.7 Algorithm Comparison Summary

| Algorithm | Hessian cost | PSD? | Convergence | Best for |
|-----------|:----------:|:----:|:-----------:|----------|
| Newton/Laplace | ∂²log p | No | Fast (quadratic) | Log-concave likelihoods |
| VI (natural grad) | E_q[∂²log p] | No | Linear | General, ELBO-based |
| EP | Moment matching | No | Superlinear | Best uncertainty estimates |
| PL | Linearised | Yes | Linear | Known E[y\|f], Var[y\|f] |
| Gauss-Newton | JᵀWJ | **Yes** | Superlinear | Non-log-concave safety |
| Quasi-Newton | BFGS | **Yes** | Superlinear | Cheap Hessian-free |

---

## 6. The InferenceStrategy Protocol

This is the new protocol layer that mediates between the likelihood and
the solver.  It computes site parameters; the solver does the linear algebra.

### Contract

```python
@runtime_checkable
class InferenceStrategy(Protocol):
    """
    Computes Gaussian site approximations for non-conjugate likelihoods.

    Given the current posterior marginals q(fₙ) = N(mₙ, Cₙ) and
    the observations yₙ, produce pseudo-observations (ỹₙ, R̃ₙ)
    that make the problem conjugate.
    """

    def compute_sites(
        self,
        log_lik: Callable,    # log p(yₙ | fₙ) as a callable
        m: jnp.ndarray,       # (N,) or (N, P) posterior means
        C: jnp.ndarray,       # (N,) or (N, S, S) posterior marginal covs
        y: jnp.ndarray,       # (N,) or (N, P) observations
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns
        -------
        pseudo_y : (N,)    pseudo-observations ỹₙ
        pseudo_R : (N,)    pseudo-noise variances R̃ₙ (diagonal)
        """
        ...

    @property
    def requires_iteration(self) -> bool:
        """True if the strategy needs outer-loop iteration (all non-conjugate)."""
        ...

    @property
    def name(self) -> str:
        ...
```

### Instances

```python
class ConjugateStrategy:
    """Gaussian likelihood — no site approximation needed."""
    requires_iteration = False

    def compute_sites(self, log_lik, m, C, y):
        # Sites are just the real observations with real noise
        return y, noise_var * jnp.ones_like(y)


class LaplaceStrategy:
    """Newton's method on log p(yₙ | fₙ)."""
    requires_iteration = True

    def compute_sites(self, log_lik, m, C, y):
        # Evaluate gradient and Hessian at current mean
        grad_fn = jax.grad(log_lik)
        hess_fn = jax.hessian(log_lik)
        g = jax.vmap(grad_fn)(m, y)          # (N,)
        H = jax.vmap(hess_fn)(m, y)          # (N,)  (diagonal)
        precision = jnp.maximum(-H, 1e-8)    # clip for stability
        pseudo_R = 1.0 / precision
        pseudo_y = m + pseudo_R * g
        return pseudo_y, pseudo_R


class VIStrategy:
    """Natural gradient VI via Gauss-Hermite quadrature."""
    requires_iteration = True

    def compute_sites(self, log_lik, m, C, y):
        # Gauss-Hermite quadrature for E_q[∂log p/∂f] and E_q[∂²log p/∂f²]
        ...


class EPStrategy:
    """Expectation propagation via moment matching."""
    requires_iteration = True

    def compute_sites(self, log_lik, m, C, y):
        # Cavity computation, tilted moment matching
        ...


class GaussNewtonStrategy:
    """Wraps any base strategy with JᵀWJ Hessian approximation."""

    def __init__(self, base: InferenceStrategy):
        self.base = base
    ...


class PosteriorLinearisationStrategy:
    """Statistical linearisation of the observation model."""
    requires_iteration = True

    def compute_sites(self, log_lik, m, C, y):
        # Compute E_q[y|f], Var_q[y|f], linearise
        ...
```

### How it composes with the Solver

The outer inference loop for a temporal GP becomes:

```python
def temporal_inference(
    ss_model: StateSpaceRep,       # from kernel.to_state_space()
    y: jnp.ndarray,                # (N,) observations
    strategy: InferenceStrategy,    # site computation method
    solver: KalmanSolver,           # linear algebra backend
    log_lik: Callable,              # log p(yₙ | fₙ)
    max_iters: int = 20,
    tol: float = 1e-4,
):
    N = y.shape[0]

    # Initialise sites: uninformative
    pseudo_y = jnp.zeros(N)
    pseudo_R = 1e6 * jnp.ones(N)

    for iteration in range(max_iters):
        # 1. Run Kalman filter + smoother with current sites
        filter_result = solver.filter(ss_model, pseudo_y, pseudo_R)
        smooth_result = solver.smooth(filter_result)

        m = smooth_result.means      # (N,)  posterior means
        C = smooth_result.variances   # (N,)  posterior marginal variances

        # 2. Compute new sites from current posterior
        if not strategy.requires_iteration:
            break
        pseudo_y_new, pseudo_R_new = strategy.compute_sites(log_lik, m, C, y)

        # 3. Check convergence
        if jnp.max(jnp.abs(pseudo_y_new - pseudo_y)) < tol:
            break

        # 4. Damped update (for stability)
        pseudo_y = pseudo_y_new
        pseudo_R = pseudo_R_new

    return smooth_result
```

This is the core algorithm.  The key property: **swapping the strategy
changes the inference method; swapping the solver changes the linear
algebra**.  A KalmanSolver gives O(NS³); a CholeskySolver on the
equivalent dense problem gives O(N³); a parallel Kalman solver gives
O(S³ log N).

---

## 7. Integration with numpyro.scan

NumPyro provides `numpyro.contrib.control_flow.scan` which is a
JAX-compatible scan that correctly handles NumPyro's effect system
(sample/param sites remain visible to inference algorithms).

This is critical: a naive `jax.lax.scan` over a Kalman filter would
hide internal `numpyro.sample` sites from MCMC/SVI.
`numpyro.contrib.control_flow.scan` solves this.

### Kalman filter as scan

The Kalman filter is naturally a sequential scan:

```python
def kalman_step(carry, obs_n):
    """Single Kalman predict-update step."""
    x_filt, P_filt, log_lik_accum, A, Q, H = carry
    y_n, R_n = obs_n

    # Predict
    x_pred = A @ x_filt
    P_pred = A @ P_filt @ A.T + Q

    # Update
    v = y_n - H @ x_pred
    S = H @ P_pred @ H.T + R_n
    K = P_pred @ H.T / S         # scalar S for 1-D obs
    x_new = x_pred + K * v
    P_new = P_pred - K @ S @ K.T

    # Log-likelihood contribution
    ll = -0.5 * (v**2 / S + jnp.log(S) + jnp.log(2 * jnp.pi))
    log_lik_accum = log_lik_accum + ll

    carry_new = (x_new, P_new, log_lik_accum, A, Q, H)
    output = (x_new, P_new, v, S)
    return carry_new, output
```

With `jax.lax.scan`:

```python
init = (x0, P0, 0.0, A, Q, H)
obs = (pseudo_y, pseudo_R)     # (N,) each, stacked
final_carry, outputs = jax.lax.scan(kalman_step, init, obs)
```

### When to use numpyro.scan vs jax.lax.scan

| Scenario | Use |
|----------|-----|
| Kalman filter inside a solver (no numpyro sites) | `jax.lax.scan` — faster, no overhead |
| Temporal model with numpyro.sample at each step | `numpyro.contrib.control_flow.scan` — required for effect visibility |
| Hyperparameter learning via SVI/MCMC | Either — the Kalman filter is called inside `numpyro.factor`, which is a single site |

The typical pattern for pyrox.gp:

```python
def model(t, y):
    # Hyperparameters (sampled — visible to MCMC/SVI)
    variance = numpyro.sample("variance", dist.LogNormal(0, 1))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0, 1))
    noise_var = numpyro.sample("noise_var", dist.LogNormal(-2, 1))

    # Build state-space model
    ss = matern32_to_ss(variance, lengthscale, t)

    # Run Kalman filter (pure JAX — no numpyro sites inside)
    lml = kalman_filter_log_marginal(ss, y, noise_var)

    # Register as factor
    numpyro.factor("gp_lml", lml)
```

Here the Kalman filter is a **pure function** called inside the model.
It uses `jax.lax.scan` internally.  The numpyro sites (variance,
lengthscale, noise_var) are outside the scan, so `jax.lax.scan` is fine.

For the non-conjugate case where f is an explicit latent:

```python
def model(t, y_counts):
    variance = numpyro.sample("variance", dist.LogNormal(0, 1))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0, 1))
    ss = matern32_to_ss(variance, lengthscale, t)

    # Sample f via GP prior (uses Kalman sampling internally)
    f = numpyro.sample("f", MarkovGPPrior(ss))

    # User's likelihood
    numpyro.sample("y", dist.Poisson(jnp.exp(f)), obs=y_counts)
```

Here `MarkovGPPrior.sample()` draws from the state-space model using
`jax.lax.scan`, and `MarkovGPPrior.log_prob()` evaluates via the
Kalman filter.

### numpyro.scan for explicit state-space latent variables

If the user wants each state xₙ as a separate numpyro site (e.g., for
a custom transition model that isn't LTI), then `numpyro.scan` is needed:

```python
def transition(carry, t_n):
    x_prev = carry
    x_n = numpyro.sample(f"x", dist.MultivariateNormal(A @ x_prev, Q))
    f_n = H @ x_n
    numpyro.sample(f"y", dist.Poisson(jnp.exp(f_n)), obs=y[t_n])
    return x_n, f_n

x0 = numpyro.sample("x0", dist.MultivariateNormal(m0, P0))
# numpyro.scan handles the effect system correctly
_, fs = numpyro.contrib.control_flow.scan(transition, x0, jnp.arange(N))
```

This is the most flexible but also the most expensive path — each xₙ
is a separate latent variable, giving NS total latent dimensions.
MCMC would struggle; a structured guide (KalmanGuide) is essential.

---

## 8. Sparse Markov GPs

For very long time series (N > 10⁵), even O(N) Kalman can be slow.
Sparse Markov GPs introduce M ≪ N inducing time points, giving O(NMS²)
cost where M is the number of inducing points per "segment."

The key idea (Wilkinson, Solin & Adam, AISTATS 2021): partition the
time axis into segments, each containing one inducing point.  Within
each segment, use the exact Kalman recursion.  Between segments,
use the inducing point to compress the state.

In pyrox.gp, this is a separate solver:

```python
class SparseKalmanSolver(TemporalSolver):
    """O(NMS²) sparse Markov GP solver."""

    def __init__(self, num_inducing: int, jitter: float = 1e-6):
        ...
```

Compatible with all InferenceStrategy instances (Laplace, VI, EP, PL).

---

## 9. Parallel Kalman via Associative Scan

The Kalman filter is inherently sequential.  However, the
predict-update recursion can be reformulated as an associative
binary operation and parallelised via `jax.lax.associative_scan`
(Särkkä & García-Fernández, 2020; Corenflos, Zhao & Särkkä, 2022).

The trick: define elements eₙ = (Aₙ, bₙ, Cₙ) such that the
composition e₁ ⊕ e₂ ⊕ ··· ⊕ eₙ gives the filter state at time n.
The ⊕ operation is:

$$
(A_2, b_2, C_2) \oplus (A_1, b_1, C_1) = (A_2 A_1,\; A_2 b_1 + b_2,\; A_2 C_1 A_2^\top + C_2)
$$

This is associative, enabling a parallel prefix scan in O(S³ log N) span
on a GPU with N processors.

```python
def parallel_kalman_filter(ss_model, y, noise_var):
    """O(S³ log N) parallel Kalman filter via associative scan."""
    elements = build_elements(ss_model, y, noise_var)  # (N, ...) pytree
    scanned = jax.lax.associative_scan(combine_fn, elements)
    return extract_filter_states(scanned)
```

In pyrox.gp, this is a solver variant:

```python
class ParallelKalmanSolver(TemporalSolver):
    """O(S³ log N) parallel Kalman via associative_scan."""
    ...
```

---

## 10. Class Hierarchy & Module Layout

### Proposed module structure

```
pyrox/gp/solvers/
├── kalman.py
│   ├── KalmanSolver           # Sequential O(NS³) filter + smoother
│   ├── ParallelKalmanSolver   # Associative scan O(S³ log N)
│   └── SparseKalmanSolver     # Sparse Markov O(NMS²)
│
pyrox/gp/inference/
├── __init__.py
├── base.py
│   └── InferenceStrategy      # Protocol
├── conjugate.py
│   └── ConjugateStrategy
├── laplace.py
│   └── LaplaceStrategy
├── vi.py
│   └── VIStrategy             # Natural gradient VI
├── ep.py
│   └── EPStrategy
├── linearisation.py
│   ├── PosteriorLinearisationStrategy
│   └── TaylorStrategy         # EKS
├── gauss_newton.py
│   └── GaussNewtonWrapper     # Wraps any strategy with JᵀWJ Hessian
└── quasi_newton.py
    └── QuasiNewtonWrapper     # BFGS Hessian update

pyrox/gp/kernels/
├── temporal.py
│   ├── Matern12               # Returns StateSpaceRep
│   ├── Matern32
│   ├── Matern52
│   ├── Periodic
│   └── SumTemporalKernel      # Block-diagonal state concatenation
│
pyrox/gp/distributions/
├── markov_gp_prior.py
│   └── MarkovGPPrior          # numpyro.Distribution with Kalman log_prob
│
pyrox/gp/guides/
├── kalman.py
│   └── KalmanGuide            # RTS smoother as variational family
```

### Composition Table

Any (KernelType × Solver × InferenceStrategy) triple that type-checks
is a valid model:

| Kernel → Rep | Solver | Strategy | Result |
|-------------|--------|----------|--------|
| Matern32 → StateSpace | KalmanSolver | Conjugate | Exact temporal GP, O(NS³) |
| Matern32 → StateSpace | KalmanSolver | Laplace | GP classification, O(iter·NS³) |
| Matern32 → StateSpace | KalmanSolver | EP | Best uncertainty, O(iter·NS³) |
| Matern32 → StateSpace | KalmanSolver | VI | ELBO-based, O(iter·NS³) |
| Matern32 → StateSpace | KalmanSolver | PL | UKS-like, O(iter·NS³) |
| Matern32 → StateSpace | ParallelKalman | Any | GPU-parallel, O(S³ log N) |
| Matern32 → StateSpace | SparseKalman | Any | Very long series, O(NMS²) |
| RBF → Dense | CholeskySolver | Laplace | Dense GP classification, O(iter·N³) |
| RBF → Dense | BBMMSolver | Laplace | Scalable, O(iter·kN²) |

The same InferenceStrategy works with any solver — it only computes
site parameters from posterior marginals.  The solver handles all
matrix algebra.

---

## References

- **Hartikainen & Särkkä (2010)**. Kalman Filtering and Smoothing Solutions to Temporal Gaussian Process Regression Models. IEEE MLSP.
- **Wilkinson, Särkkä & Solin (2023)**. Bayes-Newton Methods for Approximate Bayesian Inference with PSD Guarantees. JMLR 24(83):1–50.
- **Wilkinson, Chang, Andersen & Solin (2020)**. State Space Expectation Propagation. ICML.
- **Chang, Wilkinson, Khan & Solin (2020)**. Fast Variational Learning in State Space Gaussian Process Models. IEEE MLSP.
- **Wilkinson, Solin & Adam (2021)**. Sparse Algorithms for Markovian Gaussian Processes. AISTATS.
- **Särkkä & García-Fernández (2020)**. Temporal Parallelization of Bayesian Smoothers. IEEE Trans. Automatic Control.
- **Corenflos, Zhao & Särkkä (2022)**. Gaussian Process Regression in Logarithmic Time.
- **García-Fernández, Tronarp & Särkkä (2019)**. Gaussian Process Classification Using Posterior Linearization. IEEE Signal Processing.
- **Solin, Hensman & Turner (2018)**. Infinite-Horizon Gaussian Processes. NeurIPS.
- **Hamelijnck, Wilkinson, Loppi, Solin & Damoulas (2021)**. Spatio-Temporal Variational Gaussian Processes. NeurIPS.