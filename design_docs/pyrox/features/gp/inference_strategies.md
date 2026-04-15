---
status: draft
version: 0.2.0
---

# pyrox x Inference Strategy Implementations: Gap Analysis

**Subject:** Concrete `InferenceStrategy` implementations sourced from BayesNewton's
unified Newton framework, GPyTorch variational strategies, and MarkovFlow.

**Date:** 2026-04-02  
**Updated:** 2026-04-03

---

## 1  Summary

The `InferenceStrategy` protocol is already defined in pyrox.gp's architecture.
This doc specifies the concrete implementations and the gaussx primitives they
compose. The Bayes-Newton thesis: all non-conjugate GP inference is Newton's
method on site natural parameters, differing only in the Hessian approximation.

Every strategy follows the same outer loop:

1. Form the cavity distribution by removing the current site from the posterior.
2. Compute the expected log-likelihood statistics (mean, Jacobian, Hessian)
   via an `Integrator` (Gauss-Hermite, unscented, Monte Carlo).
3. Newton update on site natural parameters using the strategy-specific
   Hessian approximation.
4. Update the posterior via a `Solver` (dense Cholesky, CG, Kalman).
5. Apply damping and repeat until convergence.

Each strategy is an `eqx.Module` that plugs into NumPyro's inference loop
and composes with any `Solver` and any `Integrator`.

| Gap | Strategy | Hessian | Source | Priority |
|---|---|---|---|---|
| 1 | Variational inference (VI) | Variational (ELBO) | GPJax, GPyTorch | High |
| 2 | Conjugate VI (CVI) | Exact (natural gradient) | Khan & Lin 2017 | High |
| 3 | Expectation propagation (EP) | Moment-matched | Minka 2001 | Medium |
| 4 | Posterior linearization (PL) | Statistical linearization | Garcia-Fernandez 2019 | Medium |
| 5 | Laplace approximation | Exact Hessian | BayesNewton | Medium |
| 6 | Gauss-Newton (EKF-style) | GGN approximation | BayesNewton | Low |
| 7 | Quasi-Newton (L-BFGS sites) | L-BFGS | BayesNewton | Low |

---

## 2  Common Mathematical Framework

All seven strategies are instances of Newton's method applied to
site natural parameters $\{\lambda_i^{(1)}, \Lambda_i^{(2)}\}$
of a site approximation $t_i(f_i) \approx p(y_i \mid f_i)$.

### 2.1  Site parameterization

Each likelihood factor $p(y_i \mid f_i)$ is approximated by an
un-normalized Gaussian site:

$$t_i(f_i) = \mathcal{N}^{-1}(f_i \mid \lambda_i^{(1)}, -2\Lambda_i^{(2)})$$

where $\lambda_i^{(1)} \in \mathbb{R}$ and $\Lambda_i^{(2)} \in \mathbb{R}_{<0}$
are the first and second natural parameters, respectively.

### 2.2  Cavity distribution

The cavity distribution removes site $i$ from the current posterior
$q(f_i) = \mathcal{N}(m_i, v_i)$:

$$q_{\setminus i}(f_i) = \mathcal{N}\!\left(\mu_{\text{cav}}, \sigma^2_{\text{cav}}\right)$$

$$\sigma^{-2}_{\text{cav}} = v_i^{-1} + 2\Lambda_i^{(2)}, \qquad \mu_{\text{cav}} = \sigma^2_{\text{cav}}\!\left(v_i^{-1} m_i - \lambda_i^{(1)}\right)$$

### 2.3  Newton update — the unifying equation

Given the log-likelihood $\ell_i(f) = \log p(y_i \mid f)$, compute:

- **Jacobian** $J_i$: first derivative information (strategy-dependent)
- **Hessian** $H_i$: second derivative information (strategy-dependent, **this is what distinguishes each method**)

The site update is:

$$\boxed{\lambda_i^{(1)} = J_i - H_i \, m_i, \qquad \Lambda_i^{(2)} = -H_i}$$

where $m_i$ is the posterior mean at site $i$. The key insight from
Wilkinson et al. (2021): **every method computes the same update equation** — they
differ only in how $J_i$ and $H_i$ are obtained.

| Strategy | $J_i$ | $H_i$ |
|---|---|---|
| Laplace | $\nabla_f \ell_i(m_i)$ | $\nabla^2_f \ell_i(m_i)$ |
| Gauss-Newton | $\nabla_f \ell_i(m_i)$ | $-(\nabla_f g_i)^\top (\nabla_f g_i)$ where $\ell_i = -\tfrac{1}{2}\|g_i\|^2$ |
| VI (variational) | $\nabla_{\lambda^{(1)}} \mathcal{L}$ | from variational bound curvature |
| CVI | $\mathbb{E}_{q}[\nabla_f \ell_i]$ | $\mathbb{E}_{q}[\nabla^2_f \ell_i]$ |
| EP | from moment matching $\hat{p}(y_i \mid f_i)\, q_{\setminus i}(f_i)$ | from moment matching |
| PL | $\mathbb{E}_{q}[\nabla_f \ell_i]$ via SLR | $-A^\top \Omega^{-1} A$ via SLR |
| Quasi-Newton | $\nabla_f \ell_i(m_i)$ | L-BFGS approximation from gradient history |

### 2.4  Damped updates

All strategies support damping to ensure convergence:

$$\lambda_i^{(1)} \leftarrow (1-\rho)\,\lambda_{i,\text{old}}^{(1)} + \rho\,\lambda_{i,\text{new}}^{(1)}$$

$$\Lambda_i^{(2)} \leftarrow (1-\rho)\,\Lambda_{i,\text{old}}^{(2)} + \rho\,\Lambda_{i,\text{new}}^{(2)}$$

where $\rho \in (0, 1]$ is the damping factor. Power EP uses $\rho < 1$;
Laplace and CVI typically use $\rho = 1$.

---

## 3  Gap Catalog

### Gap 1: Variational Inference (VI)

**Hessian approximation:** Variational — optimizes the ELBO
$\mathcal{L} = \mathbb{E}_q[\log p(y \mid f)] - \text{KL}[q(f) \| p(f)]$
via gradient descent on variational parameters.

**Math:** The ELBO decomposes as:

$$\mathcal{L}(m, S) = \sum_{i=1}^N \mathbb{E}_{q(f_i)}\!\left[\log p(y_i \mid f_i)\right] - \text{KL}\!\left[q(\mathbf{u}) \| p(\mathbf{u})\right]$$

The expected log-likelihood term is approximated by an `Integrator`.
The KL term has a closed-form Gaussian expression:

$$\text{KL} = \tfrac{1}{2}\!\left[\text{tr}(K_{uu}^{-1} S) + (m_u - m_0)^\top K_{uu}^{-1} (m_u - m_0) - M + \log\frac{|K_{uu}|}{|S|}\right]$$

Gradients are taken w.r.t. the variational parameters $(m_u, S)$ directly —
no site natural parameter update. This is the **only strategy** that does
not use the Newton site update form.

**Complexity:** $O(NM^2 + M^3)$ per iteration (standard SVGP).

```python
class VariationalInference(eqx.Module):
    """Standard variational inference (ELBO optimization).

    Optimizes variational parameters directly via gradient descent.
    Does not use Newton-style site updates — included for completeness
    as the baseline strategy.
    """
    n_samples: int = 1                          # MC samples for ELL (0 = quadrature)
    integrator: Integrator = GaussHermite(deg=20)
    learning_rate: float = 0.01

    def elbo(
        self,
        likelihood: Likelihood,
        prior_mean: Float[Array, " N"],
        K_uu: Float[Array, "M M"],
        K_uf: Float[Array, "M N"],
        m_u: Float[Array, " M"],
        S_u: Float[Array, "M M"],
        y: Float[Array, " N"],
    ) -> Float[Array, ""]: ...

    def update(
        self,
        posterior: GaussianPosterior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
    ) -> GaussianPosterior: ...
```

**When to use:** Default choice when using stochastic mini-batching with
non-conjugate likelihoods. Well-understood, broadly supported. Can underestimate
posterior variance.

**Ref:** Hensman, Matthews & Ghahramani (2015) *Scalable Variational Gaussian Process Classification.* AISTATS.

---

### Gap 2: Conjugate-computation Variational Inference (CVI)

**Hessian approximation:** Exact expected Hessian under the variational
posterior — natural gradient descent in the natural parameter space.

**Math:** CVI computes the expected derivatives under $q(f_i) = \mathcal{N}(m_i, v_i)$:

$$J_i = \mathbb{E}_{q(f_i)}\!\left[\nabla_f \log p(y_i \mid f_i)\right], \qquad H_i = \mathbb{E}_{q(f_i)}\!\left[\nabla^2_f \log p(y_i \mid f_i)\right]$$

These expectations are computed via the `Integrator`. The site update is:

$$\lambda_i^{(1)} = J_i - H_i \, m_i, \qquad \Lambda_i^{(2)} = -H_i$$

This is equivalent to natural gradient descent on the ELBO with step size
$\rho = 1$. The key advantage: **conjugate-form updates** — no optimizer
state, no learning rate tuning, and guaranteed to converge for log-concave
likelihoods.

**Complexity:** $O(N \cdot Q + M^3)$ per iteration, where $Q$ is the
integrator cost per site.

```python
class ConjugateVI(eqx.Module):
    """Conjugate-computation variational inference.

    Natural gradient descent on the ELBO in natural parameter space.
    Equivalent to Newton's method with the expected Hessian.
    """
    integrator: Integrator = GaussHermite(deg=20)
    damping: float = 1.0                        # rho in (0, 1]

    def compute_sites(
        self,
        likelihood: Likelihood,
        cavity_mean: Float[Array, " N"],
        cavity_var: Float[Array, " N"],
        y: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Return (nat1, nat2) site parameters."""
        ...

    def update(
        self,
        posterior: GaussianPosterior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
    ) -> GaussianPosterior: ...
```

**When to use:** Preferred over standard VI when the full dataset fits in
memory (no mini-batching needed). Faster convergence, no optimizer
hyperparameters. Works well with Kalman solvers for temporal models.

**Ref:** Khan & Lin (2017) *Conjugate-Computation Variational Inference: Converting Variational Inference in Non-Conjugate Models to Inferences in Conjugate Models.* AISTATS.

---

### Gap 3: Expectation Propagation (EP)

**Hessian approximation:** Moment-matched — matches the first two moments of
the tilted distribution $\hat{p}_i(f_i) \propto p(y_i \mid f_i)\, q_{\setminus i}(f_i)$.

**Math:** EP computes the tilted distribution moments:

$$\mu_{\text{tilt}} = \mathbb{E}_{\hat{p}_i}[f_i], \qquad \sigma^2_{\text{tilt}} = \text{Var}_{\hat{p}_i}[f_i]$$

where $\hat{p}_i(f_i) \propto p(y_i \mid f_i) \, q_{\setminus i}(f_i)$ and expectations
are computed via the `Integrator`. The site update is:

$$\Lambda_i^{(2)} = -\frac{1}{2}\!\left(\sigma_{\text{tilt}}^{-2} - \sigma_{\text{cav}}^{-2}\right), \qquad \lambda_i^{(1)} = \sigma_{\text{tilt}}^{-2}\,\mu_{\text{tilt}} - \sigma_{\text{cav}}^{-2}\,\mu_{\text{cav}}$$

Expressed in Newton form: $H_i = -(\sigma_{\text{tilt}}^{-2} - \sigma_{\text{cav}}^{-2})$
and $J_i = \lambda_i^{(1)} + H_i \, m_i$.

Power EP generalizes this with a power parameter $\alpha \in (0, 1]$,
replacing $p(y_i \mid f_i)$ with $p(y_i \mid f_i)^\alpha$ in the tilted
distribution.

**Complexity:** $O(N \cdot Q + M^3)$ per iteration. EP typically converges
in fewer iterations than VI but each iteration is more expensive due to
cavity computation.

```python
class ExpectationPropagation(eqx.Module):
    """Expectation propagation with moment matching.

    Iteratively refines site approximations by matching moments of the
    tilted distribution. Power EP (alpha < 1) improves stability.
    """
    integrator: Integrator = GaussHermite(deg=20)
    damping: float = 0.5                        # rho — EP often needs damping
    power: float = 1.0                          # alpha — power EP parameter

    def compute_tilted_moments(
        self,
        likelihood: Likelihood,
        cavity_mean: Float[Array, " N"],
        cavity_var: Float[Array, " N"],
        y: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Return (tilted_mean, tilted_var) via moment matching."""
        ...

    def compute_sites(
        self,
        tilted_mean: Float[Array, " N"],
        tilted_var: Float[Array, " N"],
        cavity_mean: Float[Array, " N"],
        cavity_var: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Return (nat1, nat2) site parameters from tilted moments."""
        ...

    def update(
        self,
        posterior: GaussianPosterior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
    ) -> GaussianPosterior: ...
```

**When to use:** Best posterior approximation quality among all methods —
minimizes reverse KL. Preferred for classification and count data where
accurate uncertainty is critical. Requires damping for stability.

**Ref:** Minka (2001) *Expectation Propagation for Approximate Bayesian Inference.* UAI.

---

### Gap 4: Posterior Linearization (PL)

**Hessian approximation:** Statistical linear regression (SLR) — linearizes the
likelihood observation model around the current posterior using sigma points.

**Math:** Given an observation model $y_i = h(f_i) + \varepsilon_i$,
SLR computes the affine approximation:

$$h(f_i) \approx A_i\, f_i + b_i, \qquad A_i = C_{fh}\,C_{ff}^{-1}$$

where the cross-covariance $C_{fh} = \text{Cov}_q[f_i, h(f_i)]$ and
$C_{ff} = \text{Var}_q[f_i]$ are computed via sigma points (unscented transform).
The residual covariance is:

$$\Omega_i = C_{hh} - A_i\,C_{ff}\,A_i^\top + R_i$$

The Newton quantities are:

$$H_i = -A_i^\top \Omega_i^{-1} A_i, \qquad J_i = A_i^\top \Omega_i^{-1}(y_i - b_i - A_i\,m_i) + H_i\,m_i$$

yielding site updates:

$$\lambda_i^{(1)} = A_i^\top \Omega_i^{-1}(y_i - b_i), \qquad \Lambda_i^{(2)} = -\tfrac{1}{2} A_i^\top \Omega_i^{-1} A_i$$

**Complexity:** $O(N \cdot S \cdot d_y^2)$ per iteration, where $S$ is the
number of sigma points and $d_y$ is the observation dimension. Well-suited
for multi-output likelihoods.

```python
class PosteriorLinearization(eqx.Module):
    """Posterior linearization via statistical linear regression.

    Uses sigma-point methods (unscented, cubature) to linearize the
    observation model around the current posterior. Natural generalization
    to multi-output likelihoods.
    """
    integrator: Integrator = UnscentedTransform()
    damping: float = 1.0

    def statistical_linear_regression(
        self,
        observation_model: Callable,
        mean: Float[Array, " N"],
        var: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"], Float[Array, " N"]]:
        """Return (A, b, Omega) — SLR coefficients."""
        ...

    def compute_sites(
        self,
        A: Float[Array, " N"],
        b: Float[Array, " N"],
        Omega: Float[Array, " N"],
        y: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Return (nat1, nat2) site parameters from SLR."""
        ...

    def update(
        self,
        posterior: GaussianPosterior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
    ) -> GaussianPosterior: ...
```

**When to use:** Natural choice for state-space models where the likelihood
is a nonlinear observation model (sensor fusion, tracking). Extends
gracefully to multi-output. Pairs naturally with Kalman solvers. Equivalent
to the iterated extended/unscented Kalman smoother.

**Ref:** Garcia-Fernandez, Tronarp & Sarkka (2019) *Gaussian Process Classification Using Posterior Linearization.* IEEE Signal Processing Letters.

---

### Gap 5: Laplace Approximation

**Hessian approximation:** Exact Hessian of the log-likelihood evaluated at the
posterior mean (MAP estimate).

**Math:** Evaluate the log-likelihood derivatives at the current posterior mean $m_i$:

$$J_i = \nabla_f \log p(y_i \mid f)\big|_{f=m_i}, \qquad H_i = \nabla^2_f \log p(y_i \mid f)\big|_{f=m_i}$$

Site update:

$$\lambda_i^{(1)} = J_i - H_i\,m_i, \qquad \Lambda_i^{(2)} = -H_i$$

For a Bernoulli likelihood with probit link $\Phi$:

$$J_i = \frac{(2y_i - 1)\,\phi(m_i)}{(2y_i - 1)\,\Phi(m_i) + (1 - y_i)}, \qquad H_i = -J_i(J_i + m_i)$$

The Laplace approximation is a **point estimate** at the mode — it does not
integrate over posterior uncertainty when computing the Hessian. This makes
it fast but potentially overconfident.

**Complexity:** $O(N)$ per site update (no integration needed). Total
$O(N + M^3)$ per iteration.

```python
class LaplaceApproximation(eqx.Module):
    """Laplace approximation — exact Hessian at the posterior mean.

    The simplest Newton method. Fast (no quadrature) but uses a point
    estimate, so it can be overconfident for highly non-Gaussian
    likelihoods.
    """
    damping: float = 1.0

    def compute_sites(
        self,
        likelihood: Likelihood,
        posterior_mean: Float[Array, " N"],
        y: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Return (nat1, nat2) from exact Hessian at the mean.

        Uses jax.grad and jax.hessian on the log-likelihood.
        """
        ...

    def update(
        self,
        posterior: GaussianPosterior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
    ) -> GaussianPosterior: ...
```

**When to use:** When speed matters and the likelihood is close to
log-concave (classification with probit/logit link). Good starting point
for iterative refinement. Widely used in spatial statistics (INLA).

**Ref:** Rasmussen & Williams (2006) *Gaussian Processes for Machine Learning.* Ch. 3.4. Also: Wilkinson et al. (2021).

---

### Gap 6: Gauss-Newton (EKF-style)

**Hessian approximation:** Generalized Gauss-Newton (GGN) — replaces the exact
Hessian with $J_g^\top J_g$, where $J_g$ is the Jacobian of the residual
function.

**Math:** Decompose the log-likelihood as a least-squares residual:

$$\log p(y_i \mid f) = -\tfrac{1}{2}\|g_i(f)\|^2 + c$$

where $g_i(f)$ is a residual function. The GGN Hessian approximation is:

$$H_i^{\text{GGN}} = -\left(\frac{\partial g_i}{\partial f}\right)^\top\!\! \left(\frac{\partial g_i}{\partial f}\right)\bigg|_{f=m_i}$$

This drops the second-order term $\sum_j g_j \nabla^2 g_j$ from the exact
Hessian. The GGN Hessian is always negative semi-definite, guaranteeing that
$\Lambda_i^{(2)} \geq 0$ — the site precision is always non-negative.

For Gaussian observation noise $y_i = f_i + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2)$:
$g_i(f) = (y_i - f) / \sigma$, so $H_i^{\text{GGN}} = -\sigma^{-2} = H_i^{\text{exact}}$.

Site update:

$$\lambda_i^{(1)} = J_i - H_i^{\text{GGN}}\,m_i, \qquad \Lambda_i^{(2)} = -H_i^{\text{GGN}}$$

**Complexity:** $O(N \cdot d_y)$ per iteration (no integration, just Jacobian
evaluation via `jax.jacfwd`). Total $O(N \cdot d_y + M^3)$.

```python
class GaussNewton(eqx.Module):
    """Gauss-Newton approximation (EKF-style).

    Uses the generalized Gauss-Newton Hessian, which is always PSD.
    Equivalent to the extended Kalman smoother when paired with a
    Kalman solver. More stable than Laplace for non-log-concave
    likelihoods.
    """
    damping: float = 1.0

    def compute_sites(
        self,
        likelihood: Likelihood,
        posterior_mean: Float[Array, " N"],
        y: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Return (nat1, nat2) from GGN Hessian.

        Decomposes log p(y|f) into residual form and computes J_g^T J_g.
        """
        ...

    def update(
        self,
        posterior: GaussianPosterior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
    ) -> GaussianPosterior: ...
```

**When to use:** When the Laplace Hessian is not negative semi-definite
(non-log-concave likelihoods like Student-t, heteroscedastic noise). The
GGN guarantee of PSD sites avoids numerical instability. Equivalent to the
extended Kalman filter/smoother.

**Ref:** Wilkinson et al. (2021) *Bayes-Newton Methods for Approximate Bayesian Inference with PSD Guarantees.* JMLR.

---

### Gap 7: Quasi-Newton (L-BFGS Sites)

**Hessian approximation:** L-BFGS — builds a low-rank approximation to the
inverse Hessian from gradient history, avoiding explicit Hessian computation.

**Math:** Maintain a history of $k$ gradient differences
$\{s_j, y_j\}_{j=1}^k$ where:

$$s_j = m_i^{(j+1)} - m_i^{(j)}, \qquad y_j = J_i^{(j+1)} - J_i^{(j)}$$

The L-BFGS inverse Hessian approximation is:

$$(-H_i)^{-1} \approx B_k = \left(\prod_{j=1}^k V_j^\top\right) B_0 \left(\prod_{j=1}^k V_j\right) + \sum_{j=1}^k \rho_j s_j s_j^\top \cdots$$

where $\rho_j = 1/(y_j^\top s_j)$ and $V_j = I - \rho_j y_j s_j^\top$.
In practice, use the two-loop recursion to compute $H_i^{-1} \nabla \ell_i$
without forming the matrix.

For sites: compute $H_i^{\text{LBFGS}}$ from the L-BFGS approximation, then:

$$\lambda_i^{(1)} = J_i - H_i^{\text{LBFGS}}\,m_i, \qquad \Lambda_i^{(2)} = -H_i^{\text{LBFGS}}$$

**Complexity:** $O(N \cdot k)$ per iteration where $k$ is the history
length (typically 5-20). No Hessian evaluation or quadrature.

```python
class QuasiNewton(eqx.Module):
    """Quasi-Newton (L-BFGS) site updates.

    Builds a low-rank Hessian approximation from gradient history.
    Avoids explicit Hessian computation — useful when the Hessian is
    expensive or unavailable.
    """
    history_size: int = 10                      # k — number of (s, y) pairs
    damping: float = 1.0

    def compute_sites(
        self,
        likelihood: Likelihood,
        posterior_mean: Float[Array, " N"],
        y: Float[Array, " N"],
        grad_history: tuple[Float[Array, "K N"], Float[Array, "K N"]],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Return (nat1, nat2) from L-BFGS Hessian approximation."""
        ...

    def update(
        self,
        posterior: GaussianPosterior,
        likelihood: Likelihood,
        y: Float[Array, " N"],
        state: QuasiNewtonState,
    ) -> tuple[GaussianPosterior, QuasiNewtonState]: ...
```

**When to use:** When the Hessian is expensive to compute (complex
multi-output likelihoods) or when `jax.hessian` is prohibitive. Also useful
as a fallback when the likelihood does not decompose into residual form
(ruling out Gauss-Newton).

**Ref:** Wilkinson et al. (2021) *Bayes-Newton Methods for Approximate Bayesian Inference with PSD Guarantees.* JMLR.

---

## 4  Shared Infrastructure

All strategies share the Newton update loop and compose with the same
gaussx building blocks:

| Component | gaussx / pyrox function | Used by |
|---|---|---|
| Newton site update: $\lambda^{(1)} = J - Hm$, $\Lambda^{(2)} = -H$ | `gaussx.newton_site_update` | CVI, EP, PL, Laplace, GN, QN |
| Damped natural parameter update | `gaussx.damped_natural_update` | All |
| Cavity distribution | `gaussx.compute_cavity` | CVI, EP, PL |
| Moment matching (tilted distribution) | `gaussx.moment_match` | EP |
| Statistical linear regression | `gaussx.statistical_linear_regression` | PL |
| Expected log-likelihood | `gaussx.expected_log_likelihood` | VI, CVI |
| ELBO (trace + logdet + ELL) | `gaussx.svgp_elbo` | VI |
| KL divergence (Gaussian) | `gaussx.kl_divergence` | VI |
| Posterior update (natural params) | `gaussx.natural_posterior_update` | All |
| Cholesky solve | `gaussx.solve` / `lineax.Cholesky` | Dense solver |
| CG solve | `gaussx.solve` / `lineax.CG` | CG solver |
| Kalman smoother | `gaussx.kalman_smoother` | Temporal solver |
| Log-determinant | `gaussx.logdet` | All (marginal likelihood) |
| Diagonal operator | `gaussx.DiagonalLinearOperator` | Site precision matrices |

### Composition with Solvers and Integrators

```
InferenceStrategy  x  Solver       x  Integrator
-------------------   -----------     ----------------
LaplaceApproximation  DenseSolver     (none — point eval)
GaussNewton           DenseSolver     (none — point eval)
QuasiNewton           DenseSolver     (none — grad history)
ConjugateVI           CGSolver        GaussHermite
ExpectationPropagation KalmanSolver   UnscentedTransform
PosteriorLinearization KalmanSolver   UnscentedTransform
VariationalInference   DenseSolver    MonteCarlo
```

Any combination is valid — the table shows the most natural pairings.

---

## 5  References

1. Wilkinson, W. J., Solin, A., & Adam, V. (2021). *Bayes-Newton Methods for Approximate Bayesian Inference with PSD Guarantees.* JMLR, 24(83), 1-50.
2. Khan, M. E. & Lin, W. (2017). *Conjugate-Computation Variational Inference: Converting Variational Inference in Non-Conjugate Models to Inferences in Conjugate Models.* AISTATS.
3. Minka, T. P. (2001). *Expectation Propagation for Approximate Bayesian Inference.* UAI.
4. Garcia-Fernandez, A. F., Tronarp, F., & Sarkka, S. (2019). *Gaussian Process Classification Using Posterior Linearization.* IEEE Signal Processing Letters.
5. Hensman, J., Matthews, A. G. de G., & Ghahramani, Z. (2015). *Scalable Variational Gaussian Process Classification.* AISTATS.
6. Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press.
7. Opper, M. & Archambeau, C. (2009). *The Variational Gaussian Approximation Revisited.* Neural Computation.
8. Titsias, M. (2009). *Variational Learning of Inducing Variables in Sparse Gaussian Processes.* AISTATS.
