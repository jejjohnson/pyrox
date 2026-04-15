---
status: draft
version: 0.1.0
---

# Layer 2 — Models

High-level entry points that compose Layer 1 protocols into complete GP workflows. Users reach for these first.

---

## Entry Points: `gp_sample` vs `gp_factor`

The two ways to use a GP in a NumPyro model:

| Entry point | When to use | What it does |
|---|---|---|
| `gp_sample("f", prior)` | Latent GP — non-conjugate likelihood | Declares `f` as a NumPyro sample site. Requires a guide. |
| `gp_factor("gp", prior, y)` | Collapsed GP — Gaussian likelihood | Adds marginal log-likelihood as a NumPyro factor. No guide needed. |

### `gp_sample`

**Mathematical definition:** Declares $f \sim \mathcal{GP}(\mu, k)$ as a latent variable in the NumPyro trace.

```python
def gp_sample(
    name: str,
    prior: GPPrior,
    *,
    guide: Guide | None = None,
) -> Array:
    """Sample a GP realization as a NumPyro sample site.

    Use when the likelihood is non-Gaussian (Poisson, Bernoulli, etc.)
    and f cannot be marginalized analytically.
    """
```

### `gp_factor`

**Mathematical definition:** Adds the collapsed marginal log-likelihood:

$$\log p(y \mid X, \theta) = -\frac{1}{2}y^T(K + \sigma^2 I)^{-1}y - \frac{1}{2}\log|K + \sigma^2 I| - \frac{N}{2}\log(2\pi)$$

```python
def gp_factor(
    name: str,
    prior: GPPrior,
    y: Array,
    noise_var: float | Array,
) -> None:
    """Add collapsed GP marginal likelihood as a NumPyro factor.

    Use when the likelihood is Gaussian — f is marginalized analytically.
    No guide needed. Supports all Solver types.
    """
```

---

## GPPrior

**Mathematical definition:** A Gaussian process $f \sim \mathcal{GP}(m, k)$ evaluated at finite inputs $X$:

$$f \mid X \sim \mathcal{N}(m(X),\; K(X, X))$$

```python
class GPPrior(eqx.Module):
    """GP prior at observed locations.

    Composes a Kernel + Solver. The kernel defines the covariance
    structure; the solver determines how solve and logdet are computed.
    """
    kernel: Kernel
    solver: Solver
    X: Array                    # (N, D) observed inputs
    mean_fn: Callable | None    # optional mean function

    def log_prob(self, f: Array) -> Scalar:
        """Log-density of f under the GP prior."""

    def condition(self, y: Array, noise_var: float) -> "ConditionedGP":
        """Condition on observations. Returns posterior GP."""
```

---

## ConditionedGP

**Mathematical definition (GP posterior):**

$$f_* \mid y, X, X_* \sim \mathcal{N}(\mu_*, \Sigma_*)$$

$$\mu_* = K_{*f}(K_{ff} + \sigma^2 I)^{-1}y$$

$$\Sigma_* = K_{**} - K_{*f}(K_{ff} + \sigma^2 I)^{-1}K_{f*}$$

```python
class ConditionedGP(eqx.Module):
    """GP conditioned on observations. Provides posterior predictions."""

    def predict_mean(self, X_star: Array) -> Array: ...
    def predict_var(self, X_star: Array) -> Array: ...
    def predict(self, X_star: Array) -> tuple[Array, Array]: ...
    def sample(self, key: PRNGKey, X_star: Array, n_samples: int) -> Array: ...
```

---

## MarkovGPPrior

For temporal GPs with Matérn kernels — uses the state-space representation.

**Mathematical definition:** The GP is encoded as a Linear Time-Invariant SDE:

$$dx(t) = A\,x(t)\,dt + L\,d\beta(t), \qquad y_t = H\,x(t_n) + \varepsilon_t$$

```python
class MarkovGPPrior(eqx.Module):
    """Temporal GP prior using state-space (SDE/Kalman) representation.

    Uses KalmanSolver instead of dense linear algebra.
    Complexity: O(NS^3) instead of O(N^3).
    """
    kernel: Kernel              # must support to_state_space()
    solver: KalmanSolver
    times: Array                # (N,) observation times
```

---

## PathwiseSampler

**Mathematical definition (Matheron's rule):** For a conditioned GP, posterior
samples can be drawn without forming the posterior covariance:

$$f_\text{post}(\cdot) = f_\text{prior}(\cdot) + K(\cdot, X)(K_{XX} + \sigma^2 I)^{-1}(y - f_\text{prior}(X))$$

where $f_\text{prior}$ is a prior function draw. The prior draw is represented
via kernel features (RFF for stationary kernels); the update term reuses
the training-time Cholesky. Each posterior sample is a callable function
that can be evaluated at arbitrary test points in $O(M + D)$ time.

**When to use:** Drawing many posterior samples at many test points —
prediction, Thompson sampling, posterior visualization. Not for MCMC/SVI
inference (HMC operates in parameter space, not function space).

```python
class PathwiseSampler(eqx.Module):
    """Function-valued posterior samples via Matheron's rule.

    Decomposes each posterior sample into:
      prior_draw(x) + update(x)

    where the prior is represented via kernel features (RFF) and
    the update uses gaussx.matheron_update with the pre-computed
    training Cholesky.
    """

    def __init__(
        self,
        conditioned_gp: ConditionedGP,
        n_features: int = 512,
    ):
        """Build sampler from a conditioned GP.

        Args:
            conditioned_gp: GP posterior (contains Cholesky, alpha, kernel).
            n_features: Number of RFF features for prior draws.
        """

    def sample_paths(
        self, key: PRNGKey, n_paths: int = 1
    ) -> "PathwiseFunction":
        """Draw n_paths independent posterior function samples.

        Returns a callable: path(X_star) -> Array of shape (n_paths, N_star).
        Each evaluation is O(n_features + M) per test point.
        """

    def __call__(
        self, key: PRNGKey, X_star: Array, n_paths: int = 1
    ) -> Float[Array, "S N_star"]:
        """Convenience: draw and evaluate in one call."""
```

**Decoupled variant:** For sparse GPs (SVGP), the prior draw uses RFF
features while the update uses the inducing-point basis. This avoids
the cubic cost in test points entirely:

```python
class DecoupledPathwiseSampler(eqx.Module):
    """Decoupled pathwise sampler for sparse GPs.

    Prior basis: D random Fourier features (kernel-dependent).
    Update basis: M inducing-point kernel evaluations.

    Total cost per sample per test point: O(D + M).
    """
```

**gaussX dependency:** Uses `gaussx.matheron_update` for the conditioning
step and `gaussx.sample_joint_conditional` for joint prior draws.
See [gaussx/features/gp_pathwise_sampling.md](../../gaussx/features/gp_pathwise_sampling.md).

---

*For detailed subsystem APIs, see [moments.md](moments.md), [state_space.md](state_space.md), [integration.md](integration.md).*
*For usage patterns, see [../examples/](../examples/).*
