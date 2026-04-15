---
status: draft
version: 0.1.0
---

# pyrox.gp — Integrator Protocol: Gaussian Expectations as a First-Class Abstraction

> The fourth protocol in the pyrox.gp stack.  Integrators compute expectations
> under Gaussian distributions — the atomic operation underlying all
> non-conjugate GP inference and uncertainty propagation.
>
> For worked examples, see [`../../examples/gp/integration_detail.md`](../../examples/gp/integration_detail.md).

---

## Table of Contents

1. [Motivation: The Universal Bottleneck](#1-motivation-the-universal-bottleneck)
2. [The Four-Layer Stack](#2-the-four-layer-stack)
3. [Integrator Protocol](#3-integrator-protocol)
4. [Integrator Catalog](#4-integrator-catalog)
5. [What Gets Integrated: The Expectation Zoo](#5-what-gets-integrated-the-expectation-zoo)
6. [Composition with InferenceStrategy](#6-composition-with-inferencestrategy)
7. [Uncertain Input Propagation](#7-uncertain-input-propagation)
8. [Multi-Dimensional Expectations](#8-multi-dimensional-expectations)
9. [Module Layout](#9-module-layout)

---

## 1. Motivation: The Universal Bottleneck

Every non-conjugate GP operation eventually requires computing an
expectation of some function g under a Gaussian distribution q:

$$
\mathbb{E}_{q(f)}[g(f)], \qquad q(f) = \mathcal{N}(m, C)
$$

This appears in:

| Context | g(f) |
|---------|------|
| VI site updates | ∂ log p(y\|f) / ∂f  (gradient of log-likelihood) |
| VI site updates | ∂² log p(y\|f) / ∂f²  (Hessian of log-likelihood) |
| EP moment matching | f · p(y\|f)  and  f² · p(y\|f)  (tilted moments) |
| Posterior linearisation | E[y\|f], Var[y\|f], Cov[y, f\|f]  (observation statistics) |
| ELBO computation | log p(y\|f)  (expected log-likelihood) |
| Prediction with uncertain inputs | μ_f(x), k_f(x,x)  (GP posterior at random x) |
| Expected kernel | k(f, f')  (kernel evaluated at random function values) |
| Decision theory | L(f, a)  (expected loss for action a) |

BayesNewton hard-codes the integration method into each inference class.
pyrox.gp separates them: the **InferenceStrategy** says *what* to integrate,
the **Integrator** says *how*.

---

## 2. The Four-Layer Stack

```
┌─────────────────────────────────────────────────────┐
│  Kernel           WHAT covariance structure          │
│  (RBF, Matern, Kronecker, ...)                      │
└──────────────────────┬──────────────────────────────┘
                       │ produces CovarianceRep
┌──────────────────────▼──────────────────────────────┐
│  Solver            HOW to do linear algebra          │
│  (Cholesky, CG, BBMM, Kalman, Woodbury, ...)       │
└──────────────────────┬──────────────────────────────┘
                       │ solve + log_det
┌──────────────────────▼──────────────────────────────┐
│  InferenceStrategy  WHAT expectations define sites   │
│  (VI, EP, Laplace, PL, Gauss-Newton, ...)           │
└──────────────────────┬──────────────────────────────┘
                       │ calls integrator for each expectation
┌──────────────────────▼──────────────────────────────┐
│  Integrator         HOW to compute E_q[g(f)]         │
│  (Gauss-Hermite, Sigma Points, Taylor, MC, ...)     │
└─────────────────────────────────────────────────────┘
```

Layers 1–2 concern the GP **prior** (covariance structure + linear algebra).
Layers 3–4 concern the **posterior approximation** (what to compute + how).
All four are orthogonal: any valid combination works.

---

## 3. Integrator Protocol

### Core Contract

```python
@runtime_checkable
class IntegratorProtocol(Protocol):
    """
    Computes expectations E_{N(m,C)}[g(f)] for arbitrary integrands g.

    This is the atomic operation of non-conjugate GP inference and
    uncertainty propagation.  All integrators provide the same interface;
    they differ in accuracy, cost, and assumptions about g.

    Methods
    -------
    integrate(fn, mean, cov) -> result
        Compute E_{N(m,C)}[fn(f)].

    integrate_with_jacobian(fn, mean, cov) -> (E[fn], E[J_fn])
        Jointly compute E[fn(f)] and E[∂fn/∂f].
        Used by PL and Gauss-Newton strategies.

    points_and_weights(mean, cov) -> (points, weights)
        Return deterministic evaluation points and weights.
        Optional: only meaningful for deterministic quadrature methods.
        Allows strategies to evaluate multiple integrands at the same points.
    """

    def integrate(
        self,
        fn: Callable,             # g: ℝᴾ → ℝᴷ
        mean: jnp.ndarray,        # m: (P,) or scalar
        cov: jnp.ndarray,         # C: (P,P) or scalar (variance)
    ) -> jnp.ndarray:
        """E_{N(m,C)}[g(f)]"""
        ...

    def integrate_with_jacobian(
        self,
        fn: Callable,
        mean: jnp.ndarray,
        cov: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """(E[g(f)], E[∂g/∂f]) jointly."""
        ...
```

### Extended Contract (for deterministic methods)

```python
class DeterministicIntegrator(IntegratorProtocol, Protocol):
    """
    Integrators that produce deterministic evaluation points.
    (Gauss-Hermite, sigma points, cubature — but not MC.)

    The points_and_weights method allows a strategy to:
    1. Generate points once from (m, C).
    2. Evaluate multiple integrands at those same points.
    3. Combine results without redundant point generation.

    This is critical for PL, which needs E[E[y|f]], E[Var[y|f]],
    and Cov_q[E[y|f], f] all from the same set of sigma points.
    """

    def points_and_weights(
        self,
        mean: jnp.ndarray,       # (P,)
        cov: jnp.ndarray,         # (P,P)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns
        -------
        points : (K, P)   evaluation points f^{(k)}
        weights : (K,)    associated weights w_k

        Such that E[g(f)] ≈ Σ_k w_k g(f^{(k)})
        """
        ...

    def integrate(self, fn, mean, cov):
        pts, wts = self.points_and_weights(mean, cov)
        vals = jax.vmap(fn)(pts)          # (K, ...) or (K,)
        return jnp.sum(wts[:, None] * vals, axis=0) if vals.ndim > 1 \
            else jnp.sum(wts * vals)
```

### Design Decisions

**Why scalar (P=1) and vector (P>1)?**  In most temporal GP models,
each site is univariate: fₙ ∈ ℝ, q(fₙ) = N(mₙ, Cₙₙ) with scalar
mean and variance.  Gauss-Hermite quadrature works perfectly here.
For multi-output or coupled models, fₙ ∈ ℝᴾ and the integrator
must handle P-dimensional Gaussians — this is where sigma points
and cubature shine (they scale linearly in P, unlike tensor-product
Gauss-Hermite which scales as Kᴾ).

**Why `integrate_with_jacobian`?**  The Gauss-Newton and PL strategies
need both the function value and its Jacobian.  Some integrators can
compute both more efficiently than two separate calls (e.g., sigma
points evaluate fn at the same points for both).

**Why not just pass fn to the strategy and let it integrate?**  Because
the same strategy (e.g., VI) should work with different integrators
without code changes.  Dependency injection via the Integrator protocol.

---

## 4. Integrator Catalog

### 4.1 Gauss-Hermite Quadrature

The standard for 1-D Gaussian expectations.  Uses K evaluation points
and weights derived from the roots of the K-th Hermite polynomial.

$$
\mathbb{E}_{\mathcal{N}(m, \sigma^2)}[g(f)] \approx \sum_{k=1}^K w_k\,g(m + \sigma\,z_k)
$$

where (z_k, w_k) are the standard Gauss-Hermite nodes and weights
(i.e., E_{N(0,1)}[h(z)] ≈ Σ w_k h(z_k)).

**Multi-dimensional extension**: tensor product of 1-D rules.
For P dimensions with K points each: K^P total evaluations.

| Property | Value |
|----------|-------|
| Points (1-D) | K (typically 20–32) |
| Points (P-D) | Kᴾ (exponential in P) |
| Exact for | Polynomials of degree ≤ 2K−1 |
| Cost | O(K) per 1-D integral |
| Suitable dims | P ≤ 3 (beyond that, use sigma points or MC) |

```python
class GaussHermiteIntegrator:
    """
    Gauss-Hermite quadrature for 1-D (or tensor-product for low-D).

    Parameters
    ----------
    num_points : int
        K quadrature points.  Default 20.
    """
    def __init__(self, num_points: int = 20):
        self.K = num_points
        # Pre-compute standard nodes and weights
        self._nodes, self._weights = jnp.array(
            np.polynomial.hermite.hermgauss(num_points)
        )
        # Normalise weights (hermgauss returns weights for exp(-x²),
        # we need weights for N(0,1) ∝ exp(-x²/2))
        self._weights = self._weights / jnp.sqrt(jnp.pi)
        self._nodes = self._nodes * jnp.sqrt(2.0)

    def points_and_weights(self, mean, cov):
        """For scalar mean, cov (1-D case)."""
        std = jnp.sqrt(cov)
        points = mean + std * self._nodes        # (K,)
        return points, self._weights

    def integrate(self, fn, mean, cov):
        pts, wts = self.points_and_weights(mean, cov)
        return jnp.sum(wts * jax.vmap(fn)(pts))
```

### 4.2 Unscented / Sigma Point Transform

Generates 2P + 1 deterministic sigma points that capture the mean and
covariance of a P-dimensional Gaussian exactly.

Given q(f) = N(m, C) with C = LLᵀ (Cholesky):

$$
\chi^{(0)} = m, \qquad w^{(0)} = \frac{\kappa}{P + \kappa}
$$

$$
\chi^{(i)} = m + \sqrt{P + \kappa}\,L_{:,i}, \qquad w^{(i)} = \frac{1}{2(P + \kappa)}, \quad i = 1, \ldots, P
$$

$$
\chi^{(P+i)} = m - \sqrt{P + \kappa}\,L_{:,i}, \qquad w^{(P+i)} = \frac{1}{2(P + \kappa)}, \quad i = 1, \ldots, P
$$

where κ is a tuning parameter (typically κ = 3 − P for Gaussian, or
κ = max(0, 3 − P) for numerical stability).

| Property | Value |
|----------|-------|
| Points | 2P + 1 |
| Exact for | Polynomials of degree ≤ 3 (mean + cov captured exactly) |
| Cost | O(P³) for Cholesky + O(P) evaluations |
| Suitable dims | P ≤ ~50 |

```python
class SigmaPointIntegrator:
    """
    Unscented transform (sigma points).

    Parameters
    ----------
    alpha : float   Spread parameter (default 1e-3)
    beta : float    Prior knowledge parameter (default 2.0 for Gaussian)
    kappa : float   Secondary scaling (default 0.0)
    """
    def __init__(self, alpha=1e-3, beta=2.0, kappa=0.0):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def points_and_weights(self, mean, cov):
        P = mean.shape[0] if mean.ndim > 0 else 1
        lam = self.alpha**2 * (P + self.kappa) - P

        L = jnp.linalg.cholesky(cov) if cov.ndim == 2 \
            else jnp.sqrt(cov).reshape(1, 1)

        # Sigma points
        scale = jnp.sqrt(P + lam)
        pts = [mean]
        for i in range(P):
            pts.append(mean + scale * L[:, i])
            pts.append(mean - scale * L[:, i])
        points = jnp.stack(pts)                     # (2P+1, P)

        # Weights for mean
        w_m = jnp.zeros(2 * P + 1)
        w_m = w_m.at[0].set(lam / (P + lam))
        w_m = w_m.at[1:].set(0.5 / (P + lam))

        # Weights for covariance (includes beta correction)
        w_c = w_m.at[0].set(lam / (P + lam) + (1 - self.alpha**2 + self.beta))

        return points, w_m, w_c
```

The key advantage over Gauss-Hermite: the number of points scales
**linearly** in P, not exponentially.  The price: only 3rd-order accuracy.

### 4.3 Cubature Integration

Spherical-radial cubature rule: 2P points (no center point),
all equally weighted.

$$
\chi^{(i)} = m + \sqrt{P}\,L_{:,i}, \qquad \chi^{(P+i)} = m - \sqrt{P}\,L_{:,i}
$$

$$
w^{(i)} = \frac{1}{2P}, \quad i = 1, \ldots, 2P
$$

| Property | Value |
|----------|-------|
| Points | 2P |
| Exact for | Polynomials of degree ≤ 3 |
| Cost | O(P³) for Cholesky + O(P) evaluations |
| Note | Simpler than sigma points (no tuning parameters) |

```python
class CubatureIntegrator:
    """Spherical-radial cubature rule (Arasaratnam & Haykin, 2009)."""

    def points_and_weights(self, mean, cov):
        P = mean.shape[0]
        L = jnp.linalg.cholesky(cov)
        scale = jnp.sqrt(P)
        pts_pos = mean[None, :] + scale * L.T        # (P, P)
        pts_neg = mean[None, :] - scale * L.T        # (P, P)
        points = jnp.concatenate([pts_pos, pts_neg])  # (2P, P)
        weights = jnp.ones(2 * P) / (2 * P)
        return points, weights
```

### 4.4 Taylor Expansion (Linearisation)

Not a quadrature method — approximates g(f) as a polynomial around m
and computes the expectation analytically.

**First order** (extended Kalman):

$$
g(f) \approx g(m) + J_g(m)\,(f - m)
$$

$$
\mathbb{E}_q[g(f)] \approx g(m), \qquad \text{Cov}_q[g(f), f] \approx J_g(m)\,C
$$

**Second order**:

$$
g(f) \approx g(m) + J_g(m)\,(f - m) + \tfrac{1}{2}(f - m)^\top H_g(m)\,(f - m)
$$

$$
\mathbb{E}_q[g(f)] \approx g(m) + \tfrac{1}{2}\text{tr}(H_g(m)\,C)
$$

| Property | Value |
|----------|-------|
| Points | 1 (evaluates at m only) |
| Cost | O(1) for 1st order, O(P²) for 2nd (Hessian) |
| Exact for | Linear (1st) / quadratic (2nd) functions |
| Accuracy | Poor for highly nonlinear g far from m |

```python
class TaylorIntegrator:
    """
    Taylor expansion around the mean.

    Parameters
    ----------
    order : int
        1 = first-order (EKF/EKS style)
        2 = second-order (includes Hessian trace correction)
    """
    def __init__(self, order: int = 1):
        self.order = order

    def integrate(self, fn, mean, cov):
        val = fn(mean)
        if self.order == 1:
            return val
        # Second order: add ½ tr(H · C)
        H = jax.hessian(fn)(mean)
        if cov.ndim == 0:
            correction = 0.5 * H * cov
        else:
            correction = 0.5 * jnp.trace(H @ cov)
        return val + correction

    def integrate_with_jacobian(self, fn, mean, cov):
        val = fn(mean)
        J = jax.jacobian(fn)(mean)
        if self.order == 2:
            H = jax.hessian(fn)(mean)
            if cov.ndim == 0:
                val = val + 0.5 * H * cov
            else:
                val = val + 0.5 * jnp.trace(H @ cov)
        return val, J
```

### 4.5 Monte Carlo

The general-purpose fallback.  Draws S samples from q(f) and averages.

$$
\mathbb{E}_q[g(f)] \approx \frac{1}{S}\sum_{s=1}^S g(f^{(s)}), \qquad f^{(s)} \sim \mathcal{N}(m, C)
$$

| Property | Value |
|----------|-------|
| Points | S (user-specified, typically 100–1000) |
| Cost | O(S) evaluations + O(S · P) for sampling |
| Accuracy | O(1/√S) regardless of dimension |
| Note | Only stochastic integrator; introduces gradient variance in SVI |

```python
class MonteCarloIntegrator:
    """
    Monte Carlo integration via reparameterised sampling.

    Parameters
    ----------
    num_samples : int
    key : PRNGKey
    """
    def __init__(self, num_samples: int = 100, key=None):
        self.S = num_samples
        self.key = key if key is not None else jax.random.PRNGKey(0)

    def integrate(self, fn, mean, cov):
        if cov.ndim == 0:
            # Scalar case
            eps = jax.random.normal(self.key, (self.S,))
            samples = mean + jnp.sqrt(cov) * eps
        else:
            L = jnp.linalg.cholesky(cov)
            eps = jax.random.normal(self.key, (self.S, mean.shape[0]))
            samples = mean[None, :] + eps @ L.T
        return jnp.mean(jax.vmap(fn)(samples), axis=0)
```

### 4.6 Comparison Summary

| Integrator | Points (P-dim) | Exact degree | Cost | Best for |
|------------|:--------------:|:------------:|:----:|----------|
| Gauss-Hermite | Kᴾ | 2K−1 | O(Kᴾ) | P=1, high accuracy |
| Sigma Points | 2P+1 | 3 | O(P³+P) | P ≤ 50, UKF/UKS |
| Cubature | 2P | 3 | O(P³+P) | P ≤ 50, CKF/CKS |
| Taylor(1) | 1 | 1 | O(P) | Cheap, mildly nonlinear |
| Taylor(2) | 1 | 2 | O(P²) | Cheap, moderate nonlinearity |
| Monte Carlo | S | ∞ (asymp) | O(SP) | Any P, any nonlinearity |

---

## 5. What Gets Integrated: The Expectation Zoo

The Integrator protocol is agnostic to what g is.  The caller (usually
an InferenceStrategy or a prediction routine) defines g.  Here is the
taxonomy of integrands that appear in GP inference and uncertainty
propagation.

### 5.1 Log-Likelihood Expectations

$$
\mathbb{E}_q[\log p(y_n | f_n)]
$$

Used in: ELBO computation, VI energy.

### 5.2 Log-Likelihood Gradient Expectations

$$
\mathbb{E}_q\!\left[\frac{\partial \log p(y_n | f_n)}{\partial f_n}\right]
$$

Used in: VI natural gradient (Bonnet's theorem says this equals
∂/∂mₙ E_q[log p(yₙ|fₙ)]).

### 5.3 Log-Likelihood Hessian Expectations

$$
\mathbb{E}_q\!\left[\frac{\partial^2 \log p(y_n | f_n)}{\partial f_n^2}\right]
$$

Used in: VI natural gradient site precision (Price's theorem).

### 5.4 Observation Moment Expectations (for PL)

$$
\bar{y}_n = \mathbb{E}_q[\mathbb{E}[y_n | f_n]], \qquad \bar{\Omega}_n = \mathbb{E}_q[\text{Var}[y_n | f_n]]
$$

$$
\text{Cov}_q[\mathbb{E}[y_n | f_n],\; f_n]
$$

Used in: statistical linearisation / posterior linearisation.

The cross-covariance is computed from the same evaluation points:

$$
\text{Cov}_q[h(f), f] \approx \sum_k w_k\,(h(f^{(k)}) - \bar{h})\,(f^{(k)} - m)^\top
$$

This is why `points_and_weights` matters: we generate points once and
evaluate both h(f) = E[y|f] and the identity f at the same points.

### 5.5 Tilted Distribution Moments (for EP)

$$
\hat{m}_n = \frac{\int f_n\,q_{-n}(f_n)\,p(y_n|f_n)\,df_n}{\int q_{-n}(f_n)\,p(y_n|f_n)\,df_n}
$$

Technically these are expectations under the tilted distribution
p̃ₙ(fₙ) ∝ q₋ₙ(fₙ) p(yₙ|fₙ), not under q.  The integrator handles
this by integrating g(f) = f · p(y|f) / Z under q₋ₙ.

### 5.6 GP Posterior Mean and Kernel at Uncertain Inputs

$$
\mathbb{E}_{p(x_*)}[\mu_f(x_*)], \qquad \mathbb{E}_{p(x_*)}[k_f(x_*, x_*)]
$$

where μ_f and k_f are the GP posterior mean and covariance functions,
and x* ~ N(μ_x, Σ_x).  Here the Integrator operates over the **input
space**, not the latent function space.  See §7.

### 5.7 Expected Loss / Cost

$$
\mathbb{E}_q[L(f, a)]
$$

For decision-theoretic applications: expected loss of action a under
posterior uncertainty about f.  Same Integrator, different g.

---

## 6. Composition with InferenceStrategy

The InferenceStrategy receives an Integrator at construction time.
It calls the Integrator to compute whatever expectations it needs.

### VI + Gauss-Hermite (the standard)

```python
class VIStrategy:
    def __init__(self, integrator: IntegratorProtocol = GaussHermiteIntegrator()):
        self.integrator = integrator

    def compute_sites(self, log_lik_fn, m, C, y):
        # Per-site, 1-D integration (most common case)
        def site_update(m_n, C_n, y_n):
            grad_fn = jax.grad(lambda f: log_lik_fn(f, y_n))
            hess_fn = jax.grad(jax.grad(lambda f: log_lik_fn(f, y_n)))

            # E_q[∂log p/∂f]  (Bonnet's theorem)
            E_grad = self.integrator.integrate(grad_fn, m_n, C_n)

            # E_q[∂²log p/∂f²]  (Price's theorem)
            E_hess = self.integrator.integrate(hess_fn, m_n, C_n)

            precision = jnp.maximum(-E_hess, 1e-8)
            pseudo_R = 1.0 / precision
            pseudo_y = m_n + pseudo_R * E_grad
            return pseudo_y, pseudo_R

        return jax.vmap(site_update)(m, C, y)
```

### PL + Sigma Points (= Unscented Kalman Smoother)

```python
class PLStrategy:
    def __init__(self, integrator: DeterministicIntegrator = SigmaPointIntegrator()):
        self.integrator = integrator

    def compute_sites(self, obs_mean_fn, obs_var_fn, m, C, y):
        def site_update(m_n, C_n, y_n):
            # Generate points once
            pts, w_m, w_c = self.integrator.points_and_weights(
                m_n.reshape(-1), C_n.reshape(1, 1) if C_n.ndim == 0 else C_n
            )

            # Evaluate observation mean at each point
            h_vals = jax.vmap(obs_mean_fn)(pts)        # (K,)
            h_bar = jnp.sum(w_m * h_vals)              # E_q[E[y|f]]

            # Observation variance at each point
            R_vals = jax.vmap(obs_var_fn)(pts)           # (K,)
            R_bar = jnp.sum(w_m * R_vals)               # E_q[Var[y|f]]

            # Cross-covariance Cov_q[E[y|f], f]
            f_centered = pts.squeeze() - m_n
            h_centered = h_vals - h_bar
            Cov_hf = jnp.sum(w_c * h_centered * f_centered)

            # Statistical linearisation
            H_lin = Cov_hf / C_n          # linearised gain
            Omega = R_bar + jnp.sum(w_c * h_centered**2) - H_lin**2 * C_n

            # Convert to pseudo-observations
            pseudo_R = Omega / H_lin**2
            pseudo_y = y_n  # PL uses the real observation
            return pseudo_y, pseudo_R

        return jax.vmap(site_update)(m, C, y)
```

### The BayesNewton Model Zoo — Decoded

Every model in BayesNewton is now a triple (Strategy, Integrator, Solver):

| BayesNewton Class | Strategy | Integrator | Solver |
|---|---|---|---|
| `VariationalGP` | VI | GaussHermite | Cholesky |
| `MarkovVariationalGP` | VI | GaussHermite | Kalman |
| `SparseVariationalGP` | VI | GaussHermite | Woodbury |
| `SparseMarkovVariationalGP` | VI | GaussHermite | SparseKalman |
| `ExpectationPropagationGP` | EP | GaussHermite | Cholesky |
| `MarkovExpectationPropagationGP` | EP | GaussHermite | Kalman |
| `LaplaceGP` | Laplace | (none — point eval) | Cholesky |
| `MarkovLaplaceGP` | Laplace | (none) | Kalman |
| `PosteriorLinearisationGP` | PL | SigmaPoints | Cholesky |
| `MarkovPosteriorLinearisationGP` (=UKS) | PL | SigmaPoints | Kalman |
| `TaylorGP` (=EKS) | PL | Taylor(1) | Kalman |
| `GaussNewtonGP` | VI + GN wrapper | GaussHermite | Cholesky |
| `VariationalGaussNewtonGP` | VI + GN wrapper | GaussHermite | Cholesky |

In pyrox.gp, all of these are:

```python
temporal_inference(ss, y, strategy=VIStrategy(GaussHermiteIntegrator()), solver=KalmanSolver(), ...)
temporal_inference(ss, y, strategy=PLStrategy(SigmaPointIntegrator()), solver=KalmanSolver(), ...)
temporal_inference(ss, y, strategy=PLStrategy(TaylorIntegrator(order=1)), solver=KalmanSolver(), ...)
```

Three composable objects instead of ~30 concrete classes.

---

## 7. Uncertain Input Propagation

When predicting at a test point x* whose location is uncertain —
x* ~ N(μ_x, Σ_x) — the predictive distribution is:

$$
p(f_* | y) = \int p(f_* | x_*, y)\,p(x_*)\,dx_*
$$

This is an expectation over the input distribution, not over the
latent function.  The same Integrator protocol handles it.

### The Problem

Given a trained GP posterior with mean function μ_f(x) and variance
function σ²_f(x):

$$
\mathbb{E}[f_*] = \mathbb{E}_{p(x_*)}[\mu_f(x_*)]
$$

$$
\text{Var}[f_*] = \mathbb{E}_{p(x_*)}[\sigma^2_f(x_*)] + \text{Var}_{p(x_*)}[\mu_f(x_*)]
$$

The second equation is the **law of total variance**: total predictive
uncertainty = expected aleatoric uncertainty + epistemic uncertainty
from input noise.

### Using an Integrator

```python
def predict_uncertain_input(
    gp_posterior: ConditionedGP,     # trained GP posterior
    mu_x: jnp.ndarray,              # (D,)  input mean
    Sigma_x: jnp.ndarray,           # (D,D) input covariance
    integrator: IntegratorProtocol,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Propagate input uncertainty through a GP posterior.

    Returns
    -------
    E[f*] : scalar
    Var[f*] : scalar (total predictive variance)
    """
    def gp_mean_fn(x):
        mu, _ = gp_posterior.predict(x.reshape(1, -1), full_cov=False)
        return mu.squeeze()

    def gp_var_fn(x):
        _, var = gp_posterior.predict(x.reshape(1, -1), full_cov=False)
        return var.squeeze()

    # E_{p(x*)}[μ_f(x*)]
    E_mean = integrator.integrate(gp_mean_fn, mu_x, Sigma_x)

    # E_{p(x*)}[σ²_f(x*)]
    E_var = integrator.integrate(gp_var_fn, mu_x, Sigma_x)

    # Var_{p(x*)}[μ_f(x*)] via E[μ²] - (E[μ])²
    E_mean_sq = integrator.integrate(lambda x: gp_mean_fn(x)**2, mu_x, Sigma_x)
    Var_mean = E_mean_sq - E_mean**2

    # Law of total variance
    total_var = E_var + Var_mean

    return E_mean, total_var
```

The choice of Integrator determines the quality:

| Integrator | Effect |
|---|---|
| Taylor(1) | Linearise μ_f around μ_x → E[f*] ≈ μ_f(μ_x), ignores curvature |
| Taylor(2) | Adds ½ tr(H_μ Σ_x) correction — captures curvature |
| SigmaPoints | Evaluates μ_f at 2D+1 points — captures nonlinearity |
| GaussHermite | High accuracy for D ≤ 3 |
| MC | Works for any D, stochastic |

For the common case of D ≤ 3 (spatial, spatiotemporal), sigma points
or low-order Gauss-Hermite are the practical choices.

### Exact Moments for RBF Kernel (Special Case)

For the RBF kernel, E_{N(μ_x, Σ_x)}[μ_f(x*)] and Var have closed-form
expressions (Girard et al., 2003; Deisenroth & Rasmussen, 2011).
These are exact and should be used when available:

$$
\mathbb{E}[\mu_f(x_*)] = \alpha^\top \tilde{k}, \qquad \tilde{k}_i = \sigma^2 |I + \Sigma_x \Lambda^{-1}|^{-1/2} \exp\!\left(-\tfrac{1}{2}(x_i - \mu_x)^\top(\Lambda + \Sigma_x)^{-1}(x_i - \mu_x)\right)
$$

where Λ = diag(ℓ²) and α = K_y⁻¹ y.  This is implemented as:

```python
class ExactRBFInputIntegrator:
    """Exact E[μ_f(x*)] for RBF kernel with Gaussian input uncertainty."""

    def integrate_mean(self, alpha, X_train, mu_x, Sigma_x, variance, lengthscale):
        Lambda = jnp.diag(lengthscale**2)
        M = Lambda + Sigma_x
        M_inv = jnp.linalg.inv(M)
        det_factor = jnp.linalg.det(jnp.eye(mu_x.shape[0]) + Sigma_x @ jnp.linalg.inv(Lambda))
        prefactor = variance / jnp.sqrt(det_factor)

        diffs = X_train - mu_x[None, :]                   # (N, D)
        exponent = -0.5 * jnp.sum(diffs @ M_inv * diffs, axis=1)  # (N,)
        k_tilde = prefactor * jnp.exp(exponent)            # (N,)

        return jnp.dot(alpha, k_tilde)
```

The Integrator protocol accommodates both: use `ExactRBFInputIntegrator`
when the kernel is RBF, fall back to `SigmaPointIntegrator` for arbitrary
kernels.

---

## 8. Multi-Dimensional Expectations

When the latent is vector-valued (fₙ ∈ ℝᴾ, e.g., multi-output GPs),
the integrator must handle P-dimensional Gaussians.

### Scaling properties

| Integrator | 1-D (P=1) | 2-D | 5-D | 10-D | 50-D |
|------------|:---------:|:---:|:---:|:----:|:----:|
| GaussHermite (K=20) | 20 pts | 400 | 3.2M | ∞ | ∞ |
| SigmaPoints | 3 pts | 5 | 11 | 21 | 101 |
| Cubature | 2 pts | 4 | 10 | 20 | 100 |
| Taylor(1) | 1 pt | 1 | 1 | 1 | 1 |
| MC (S=500) | 500 pts | 500 | 500 | 500 | 500 |

The takeaway: Gauss-Hermite is king for P=1 (the dominant case in
temporal GPs with scalar observations), but sigma points / cubature
are the right choice for P ≥ 2.

### Automatic Selection

```python
def default_integrator(dim: int) -> IntegratorProtocol:
    """Choose a sensible default integrator for a given latent dimension."""
    if dim == 1:
        return GaussHermiteIntegrator(num_points=20)
    elif dim <= 5:
        return CubatureIntegrator()
    elif dim <= 50:
        return SigmaPointIntegrator()
    else:
        return MonteCarloIntegrator(num_samples=500)
```

---

## 9. Module Layout

```
pyrox/gp/integrators/
├── __init__.py
├── base.py                    # IntegratorProtocol, DeterministicIntegrator
├── gauss_hermite.py           # GaussHermiteIntegrator
├── sigma_points.py            # SigmaPointIntegrator (unscented transform)
├── cubature.py                # CubatureIntegrator
├── taylor.py                  # TaylorIntegrator(order=1|2)
├── monte_carlo.py             # MonteCarloIntegrator
├── exact_rbf.py               # ExactRBFInputIntegrator (closed-form for RBF)
└── utils.py                   # default_integrator(), shared quadrature utilities
```

---

## References

- **Arasaratnam & Haykin (2009)**. Cubature Kalman Filters. IEEE Trans. Automatic Control.
- **Julier & Uhlmann (2004)**. Unscented Filtering and Nonlinear Estimation. Proceedings of the IEEE.
- **Girard, Rasmussen, Quiñonero-Candela & Murray-Smith (2003)**. Gaussian Process Priors with Uncertain Inputs — Application to Multiple-Step Ahead Time Series Forecasting. NeurIPS.
- **Deisenroth & Rasmussen (2011)**. PILCO: A Model-Based and Data-Efficient Approach to Policy Search. ICML.
- **Steinberg & Bonilla (2014)**. Extended and Unscented Gaussian Processes. NeurIPS.
- **García-Fernández, Tronarp & Särkkä (2019)**. Gaussian Process Classification Using Posterior Linearization. IEEE Signal Processing.
- **Wilkinson, Särkkä & Solin (2023)**. Bayes-Newton Methods for Approximate Bayesian Inference with PSD Guarantees. JMLR.
- **Solin & Särkkä (2020)**. Hilbert Space Methods for Reduced-Rank Gaussian Process Regression. Statistics and Computing.
- **Riutort-Mayol, Bürkner, Andersen, Solin & Vehtari (2020)**. Practical Hilbert Space Approximate Bayesian Gaussian Processes for Probabilistic Programming.
