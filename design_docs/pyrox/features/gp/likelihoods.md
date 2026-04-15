---
status: draft
version: 0.1.0
---

# pyrox x Likelihood Implementations: Gap Analysis

**Subject:** Likelihood implementations needed for `pyrox.gp`, sourced from
BayesNewton (~105K lines), GPJax, and GPyTorch.

**Date:** 2026-04-02

---

## 1  Scope

Likelihoods live in `pyrox.gp`. gaussx provides the integration primitives
(`expected_log_likelihood`, `moment_match`, `statistical_linear_regression`)
and the integrator protocol (`GaussHermiteIntegrator`, `UnscentedIntegrator`,
etc.). `pyrox.gp` implements the likelihood functions that plug into those
primitives.

The key contract: a likelihood is a callable `log p(y | f)` with optional
structured metadata for analytical fast paths.

**In scope:** Scalar observation likelihoods (Gaps 1-7), multi-class (Gap 8),
heteroscedastic (Gap 9), multi-latent wrappers (Gap 10).

**Out of scope:** Deep likelihood networks, copula likelihoods, likelihood-free
methods.

---

## 2  Common Likelihood Protocol

Every likelihood implements the following interface contract. Newton-based
inference methods (Laplace, EP) require the gradient and Hessian of
$\log p(y \mid f)$ with respect to $f$. Conjugate likelihoods additionally
provide closed-form posterior updates.

```python
class Likelihood(eqx.Module):
    """Protocol for all pyrox.gp likelihoods."""

    def log_prob(
        self,
        y: Float[Array, " N"],
        f: Float[Array, " N"],
    ) -> Float[Array, " N"]:
        """Evaluate log p(y_n | f_n) element-wise."""
        ...

    def grad_log_prob(
        self,
        y: Float[Array, " N"],
        f: Float[Array, " N"],
    ) -> Float[Array, " N"]:
        r"""First derivative: $\partial \log p(y|f) / \partial f$.

        Default: JAX autodiff of `log_prob`. Override for analytical form.
        """
        ...

    def hessian_log_prob(
        self,
        y: Float[Array, " N"],
        f: Float[Array, " N"],
    ) -> Float[Array, " N"]:
        r"""Second derivative: $\partial^2 \log p(y|f) / \partial f^2$.

        Must be **negative** for log-concave likelihoods (required by Laplace).
        Default: JAX autodiff. Override for analytical form.
        """
        ...

    @property
    def is_conjugate(self) -> bool:
        """True if analytical posterior updates are available."""
        return False

    @property
    def n_latents(self) -> int:
        """Number of latent GPs consumed per observation (1 for scalar)."""
        return 1
```

For variational inference, the expected log-likelihood $\mathbb{E}_{q(f)}[\log p(y \mid f)]$
is computed by the `Integrator` protocol:

```python
class Integrator(eqx.Module):
    """Protocol for numerical integration against a Gaussian measure."""

    def __call__(
        self,
        fn: Callable[[Float[Array, " N"]], Float[Array, " N"]],
        mean: Float[Array, " N"],
        var: Float[Array, " N"],
    ) -> Float[Array, " N"]:
        """Compute E_q[fn(f)] where q(f) = N(mean, var)."""
        ...
```

---

## 3  Gap Catalog

### Gap 1: GaussianLikelihood

**Domain:** Continuous observations with additive i.i.d. noise.

**Math:** Link function is the identity, $y = f + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2)$:

$$\log p(y \mid f) = -\tfrac{1}{2}\log(2\pi\sigma^2) - \frac{(y - f)^2}{2\sigma^2}$$

$$\frac{\partial \log p}{\partial f} = \frac{y - f}{\sigma^2}, \qquad \frac{\partial^2 \log p}{\partial f^2} = -\frac{1}{\sigma^2}$$

**Conjugate:** Yes. The posterior is available in closed form:
$p(f \mid y) = \mathcal{N}\!\bigl(\sigma^{-2}(K^{-1} + \sigma^{-2}I)^{-1}y, \;(K^{-1} + \sigma^{-2}I)^{-1}\bigr)$.
The expected log-likelihood under $q(f) = \mathcal{N}(\mu, s^2)$ is also analytical:

$$\mathbb{E}_q[\log p(y \mid f)] = -\tfrac{1}{2}\log(2\pi\sigma^2) - \frac{(y - \mu)^2 + s^2}{2\sigma^2}$$

**Complexity:** $O(N)$ per evaluation — constant Hessian, no link inversion.

```python
class GaussianLikelihood(eqx.Module):
    """Gaussian likelihood with learned observation noise."""
    obs_stddev: Float[Array, ""]               # sigma (positive, use softplus)

    def log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...
    def grad_log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...
    def hessian_log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...

    @property
    def is_conjugate(self) -> bool: return True

    def analytical_ell(
        self,
        y: Float[Array, " N"],
        mean: Float[Array, " N"],
        var: Float[Array, " N"],
    ) -> Float[Array, " N"]:
        """Closed-form E_q[log p(y|f)] — bypasses numerical integration."""
        ...
```

**Ref:** Rasmussen & Williams (2006), Ch. 2. GPJax `Gaussian`. BayesNewton `Gaussian`.

---

### Gap 2: BernoulliLikelihood (logit link)

**Domain:** Binary classification, $y \in \{0, 1\}$.

**Math:** Logit link, $p(y=1 \mid f) = \sigma(f) = (1 + e^{-f})^{-1}$:

$$\log p(y \mid f) = y \, f - \log(1 + e^f)$$

$$\frac{\partial \log p}{\partial f} = y - \sigma(f)$$

$$\frac{\partial^2 \log p}{\partial f^2} = -\sigma(f)(1 - \sigma(f))$$

The Hessian is always negative (log-concave), so Laplace approximation is well-defined.

**Conjugate:** No. Requires numerical integration or Laplace/EP.

**Complexity:** $O(N)$ per evaluation. Numerically stable via `jax.nn.log_sigmoid`.

```python
class BernoulliLikelihood(eqx.Module):
    """Bernoulli likelihood with logit link."""

    def log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...
    def grad_log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...
    def hessian_log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...

    @property
    def is_conjugate(self) -> bool: return False
```

**Ref:** Rasmussen & Williams (2006), Ch. 3. GPJax `Bernoulli`. BayesNewton `Bernoulli`.

---

### Gap 3: PoissonLikelihood (log link)

**Domain:** Count data, $y \in \{0, 1, 2, \ldots\}$.

**Math:** Log link, $\lambda = e^f$:

$$\log p(y \mid f) = y \, f - e^f - \log(y!)$$

$$\frac{\partial \log p}{\partial f} = y - e^f$$

$$\frac{\partial^2 \log p}{\partial f^2} = -e^f$$

Log-concave (Hessian always negative). The mean-variance relationship is
$\text{Var}[y \mid f] = \mathbb{E}[y \mid f] = e^f$.

**Conjugate:** No.

**Complexity:** $O(N)$ per evaluation.

```python
class PoissonLikelihood(eqx.Module):
    """Poisson likelihood with log link."""
    binsize: Float[Array, ""] = 1.0            # optional exposure/binsize scaling

    def log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...
    def grad_log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...
    def hessian_log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...

    @property
    def is_conjugate(self) -> bool: return False
```

**Ref:** GPJax `Poisson`. BayesNewton `Poisson`. Rasmussen & Williams (2006), Ch. 3.

---

### Gap 4: StudentTLikelihood

**Domain:** Continuous observations with heavy-tailed noise (outlier robustness).

**Math:** Observation model $y = f + \varepsilon$, $\varepsilon \sim t_\nu(0, \sigma^2)$:

$$\log p(y \mid f) = \log\Gamma\!\left(\frac{\nu+1}{2}\right) - \log\Gamma\!\left(\frac{\nu}{2}\right) - \frac{1}{2}\log(\nu\pi\sigma^2) - \frac{\nu+1}{2}\log\!\left(1 + \frac{(y-f)^2}{\nu\sigma^2}\right)$$

$$\frac{\partial \log p}{\partial f} = \frac{(\nu+1)(y-f)}{\nu\sigma^2 + (y-f)^2}$$

$$\frac{\partial^2 \log p}{\partial f^2} = (\nu+1)\frac{(y-f)^2 - \nu\sigma^2}{\bigl(\nu\sigma^2 + (y-f)^2\bigr)^2}$$

**Not log-concave** in general — the Hessian can be positive far from the mode.
Laplace approximation may not converge; EP or variational inference preferred.

**Conjugate:** No.

**Complexity:** $O(N)$ per evaluation.

```python
class StudentTLikelihood(eqx.Module):
    """Student-t likelihood for robust regression."""
    df: Float[Array, ""]                       # nu (degrees of freedom, > 2)
    obs_stddev: Float[Array, ""]               # sigma (scale)

    def log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...
    def grad_log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...
    def hessian_log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...

    @property
    def is_conjugate(self) -> bool: return False
```

**Ref:** GPflow `StudentT`. Vanhatalo et al. (2009) *Gaussian process regression with Student-t likelihood.* NeurIPS.

---

### Gap 5: ExponentialLikelihood

**Domain:** Positive continuous observations (survival analysis, wait times).

**Math:** Log link, rate $\lambda = e^f$:

$$\log p(y \mid f) = f - y \, e^f, \qquad y > 0$$

$$\frac{\partial \log p}{\partial f} = 1 - y \, e^f$$

$$\frac{\partial^2 \log p}{\partial f^2} = -y \, e^f$$

Log-concave (Hessian always negative for $y > 0$).

**Conjugate:** No.

**Complexity:** $O(N)$ per evaluation.

```python
class ExponentialLikelihood(eqx.Module):
    """Exponential likelihood with log link."""

    def log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...
    def grad_log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...
    def hessian_log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...

    @property
    def is_conjugate(self) -> bool: return False
```

**Ref:** GPflow `Exponential`.

---

### Gap 6: BetaLikelihood

**Domain:** Observations in $(0, 1)$ (proportions, fractions).

**Math:** Logit link, $\mu = \sigma(f)$. Parameterized by mean $\mu$ and precision $\phi$
with $\alpha = \mu\phi$, $\beta = (1-\mu)\phi$:

$$\log p(y \mid f) = \log\Gamma(\phi) - \log\Gamma(\alpha) - \log\Gamma(\beta) + (\alpha - 1)\log y + (\beta - 1)\log(1 - y)$$

$$\frac{\partial \log p}{\partial f} = \phi\,\sigma(f)(1-\sigma(f))\bigl[\psi(\beta) - \psi(\alpha) + \log y - \log(1-y)\bigr]$$

$$\frac{\partial^2 \log p}{\partial f^2} = \phi\,\sigma'(f)\bigl[\psi(\beta) - \psi(\alpha) + \log y - \log(1-y)\bigr] - \phi^2\,\sigma(f)^2(1-\sigma(f))^2\bigl[\psi_1(\alpha) + \psi_1(\beta)\bigr]$$

where $\psi$ is the digamma function, $\psi_1$ the trigamma, and $\sigma'(f) = \sigma(f)(1-\sigma(f))(1-2\sigma(f))$.

**Conjugate:** No.

**Complexity:** $O(N)$ per evaluation. Requires digamma/trigamma (available via `jax.scipy.special`).

```python
class BetaLikelihood(eqx.Module):
    """Beta likelihood with logit link for (0,1)-valued data."""
    precision: Float[Array, ""]                # phi (> 0)

    def log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...
    def grad_log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...
    def hessian_log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...

    @property
    def is_conjugate(self) -> bool: return False
```

**Ref:** GPflow `Beta`. Ferrari & Cribari-Neto (2004) *Beta regression for modelling rates and proportions.* JASA.

---

### Gap 7: GammaLikelihood

**Domain:** Positive continuous observations (precipitation, energy, variance).

**Math:** Log link, rate $\beta = e^f$ with fixed shape $\alpha$:

$$\log p(y \mid f) = \alpha \, f - y \, e^f + (\alpha - 1)\log y - \log\Gamma(\alpha), \qquad y > 0$$

$$\frac{\partial \log p}{\partial f} = \alpha - y \, e^f$$

$$\frac{\partial^2 \log p}{\partial f^2} = -y \, e^f$$

Log-concave (Hessian always negative for $y > 0$).

**Conjugate:** No.

**Complexity:** $O(N)$ per evaluation.

```python
class GammaLikelihood(eqx.Module):
    """Gamma likelihood with log link."""
    shape: Float[Array, ""]                    # alpha (> 0, fixed or learnable)

    def log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...
    def grad_log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...
    def hessian_log_prob(self, y: Float[Array, " N"], f: Float[Array, " N"]) -> Float[Array, " N"]: ...

    @property
    def is_conjugate(self) -> bool: return False
```

**Ref:** GPflow `Gamma`.

---

### Gap 8: SoftmaxLikelihood (multi-class)

**Domain:** Multi-class classification, $y \in \{1, \ldots, C\}$ with $C$ latent GPs.

**Math:** Softmax link over $C$ latent functions $\mathbf{f} = [f_1, \ldots, f_C]$:

$$\log p(y = c \mid \mathbf{f}) = f_c - \log\!\sum_{j=1}^C e^{f_j}$$

$$\frac{\partial \log p}{\partial f_j} = \delta_{jc} - \frac{e^{f_j}}{\sum_k e^{f_k}} = \delta_{jc} - \pi_j$$

$$\frac{\partial^2 \log p}{\partial f_j \partial f_k} = -\pi_j(\delta_{jk} - \pi_k)$$

The Hessian is a $C \times C$ matrix per observation, $H = \text{diag}(\boldsymbol{\pi}) - \boldsymbol{\pi}\boldsymbol{\pi}^\top$,
which is negative semi-definite (log-concave).

**Conjugate:** No.

**Complexity:** $O(NC^2)$ for the full Hessian. $O(NC)$ for log-prob and gradient.

```python
class SoftmaxLikelihood(eqx.Module):
    """Softmax likelihood for multi-class classification."""
    n_classes: int                             # C

    def log_prob(
        self,
        y: Int[Array, " N"],
        f: Float[Array, "N C"],
    ) -> Float[Array, " N"]: ...

    def grad_log_prob(
        self,
        y: Int[Array, " N"],
        f: Float[Array, "N C"],
    ) -> Float[Array, "N C"]: ...

    def hessian_log_prob(
        self,
        y: Int[Array, " N"],
        f: Float[Array, "N C"],
    ) -> Float[Array, "N C C"]: ...

    @property
    def is_conjugate(self) -> bool: return False

    @property
    def n_latents(self) -> int: return self.n_classes
```

**Ref:** GPflow `Softmax`. Williams & Barber (1998) *Bayesian classification with Gaussian processes.* IEEE TPAMI.

---

### Gap 9: HeteroscedasticGaussian

**Domain:** Continuous observations with input-dependent noise — two latent GPs,
one for the mean and one for the log-variance.

**Math:** Two latent functions: $f_1$ (mean), $f_2$ (log-variance), so
$\sigma^2(x) = e^{f_2(x)}$:

$$\log p(y \mid f_1, f_2) = -\tfrac{1}{2}\log(2\pi) - \tfrac{1}{2}f_2 - \frac{(y - f_1)^2}{2 e^{f_2}}$$

Gradients (stacked as $\mathbf{f} = [f_1, f_2]$):

$$\frac{\partial \log p}{\partial f_1} = \frac{y - f_1}{e^{f_2}}, \qquad \frac{\partial \log p}{\partial f_2} = -\frac{1}{2} + \frac{(y - f_1)^2}{2 e^{f_2}}$$

Hessian (block-diagonal per observation, $2 \times 2$):

$$H = \begin{bmatrix} -e^{-f_2} & -(y - f_1)e^{-f_2} \\ -(y - f_1)e^{-f_2} & -\frac{(y-f_1)^2}{2}e^{-f_2} \end{bmatrix}$$

Not guaranteed negative definite (the $(2,2)$ block can change sign), so
Newton methods may need damping.

**Conjugate:** No.

**Complexity:** $O(N)$ per evaluation. Requires two independent latent GPs.

```python
class HeteroscedasticGaussianLikelihood(eqx.Module):
    """Heteroscedastic Gaussian: two latent GPs for mean and log-variance."""

    def log_prob(
        self,
        y: Float[Array, " N"],
        f: Float[Array, "N 2"],
    ) -> Float[Array, " N"]: ...

    def grad_log_prob(
        self,
        y: Float[Array, " N"],
        f: Float[Array, "N 2"],
    ) -> Float[Array, "N 2"]: ...

    def hessian_log_prob(
        self,
        y: Float[Array, " N"],
        f: Float[Array, "N 2"],
    ) -> Float[Array, "N 2 2"]: ...

    @property
    def is_conjugate(self) -> bool: return False

    @property
    def n_latents(self) -> int: return 2
```

**Ref:** BayesNewton `HeteroscedasticNoise`. Saul et al. (2016) *Chained Gaussian Processes.* AISTATS. Lazaro-Gredilla & Titsias (2011).

---

### Gap 10: MultiLatentLikelihood (wrapper)

**Domain:** General multi-output observations where each output has its own
scalar likelihood and latent GP.

**Math:** Factorized across outputs:

$$\log p(\mathbf{y} \mid \mathbf{f}) = \sum_{d=1}^{D} \log p_d(y_d \mid f_d)$$

Gradients and Hessians are block-diagonal — each block is the scalar
likelihood's gradient/Hessian:

$$\frac{\partial \log p}{\partial f_d} = \frac{\partial \log p_d}{\partial f_d}, \qquad H = \text{blockdiag}\!\left(\frac{\partial^2 \log p_d}{\partial f_d^2}\right)$$

This is a structural wrapper, not a new distribution.

**Conjugate:** Only if all wrapped likelihoods are conjugate.

**Complexity:** $O(ND)$ — sum of per-output costs.

```python
class MultiLatentLikelihood(eqx.Module):
    """Wraps D scalar likelihoods into a multi-output likelihood."""
    likelihoods: tuple[Likelihood, ...]        # D scalar likelihoods

    def log_prob(
        self,
        y: Float[Array, "N D"],
        f: Float[Array, "N D"],
    ) -> Float[Array, " N"]: ...

    def grad_log_prob(
        self,
        y: Float[Array, "N D"],
        f: Float[Array, "N D"],
    ) -> Float[Array, "N D"]: ...

    def hessian_log_prob(
        self,
        y: Float[Array, "N D"],
        f: Float[Array, "N D"],
    ) -> Float[Array, "N D D"]:
        """Block-diagonal Hessian (off-diagonal blocks are zero)."""
        ...

    @property
    def is_conjugate(self) -> bool:
        return all(lik.is_conjugate for lik in self.likelihoods)

    @property
    def n_latents(self) -> int: return len(self.likelihoods)
```

**Ref:** BayesNewton `MultiLatentLikelihood`. Bruinsma et al. (2020) *Scalable Exact Inference in Multi-Output Gaussian Processes.* ICML.

---

## 4  Shared Infrastructure

All likelihoods plug into the same inference machinery — they only provide
$\log p(y \mid f)$ and optionally its derivatives. The integration and
inference strategies are reused unchanged.

| Component | Source | Notes |
|---|---|---|
| $\mathbb{E}_q[\log p(y \mid f)]$ via quadrature | `gaussx.expected_log_likelihood` | Accepts any `Integrator` + `Likelihood.log_prob` |
| Gauss-Hermite integration | `gaussx.GaussHermiteIntegrator` | Default for scalar likelihoods, $Q$ quadrature points |
| Unscented integration | `gaussx.UnscentedIntegrator` | Better for multi-latent ($D > 1$) |
| Moment matching (EP) | `gaussx.moment_match` | Computes cavity $\to$ tilted $\to$ site updates |
| Statistical linear regression | `gaussx.statistical_linear_regression` | Posterior linearization (PL) method |
| Analytical Gaussian ELL | `gaussx.gaussian_expected_log_lik` | Fast path for `GaussianLikelihood` only |
| Autodiff fallback | `jax.grad`, `jax.hessian` | Default when analytical derivatives not provided |
| Numerical stability | `jax.nn.log_sigmoid`, `jax.scipy.special.gammaln` | Used inside likelihood implementations |

### Integrator protocol

For variational inference with non-conjugate likelihoods, the expected
log-likelihood is computed numerically:

$$\mathbb{E}_{q(f)}[\log p(y \mid f)] \approx \sum_{i=1}^{Q} w_i \, \log p(y \mid f_i), \qquad f_i = \mu + \sqrt{2 s^2}\,x_i$$

where $(x_i, w_i)$ are Gauss-Hermite nodes/weights. For multi-latent
likelihoods (Gaps 8-10), the integration extends to multiple dimensions
via product rules or the unscented transform.

### Moment matching for EP

EP requires computing the zeroth, first, and second moments of the tilted
distribution $\tilde{p}(f) \propto p(y \mid f) \, \mathcal{N}(f \mid \mu_\text{cav}, \sigma^2_\text{cav})$.
For log-concave likelihoods (Gaps 1-3, 5, 7, 8), these moments are
well-defined and can be computed via Gauss-Hermite quadrature. For
non-log-concave likelihoods (Gap 4: Student-t), moment matching may
require additional damping or fail to converge.

---

## 5  gaussx Building Blocks Used

| Component | gaussx function |
|---|---|
| $\mathbb{E}_q[\log p(y \mid f)]$ via any integrator | `gaussx.expected_log_likelihood` |
| Moment matching (EP) | `gaussx.moment_match` |
| Statistical linear regression (PL) | `gaussx.statistical_linear_regression` |
| Analytical Gaussian ELL | `gaussx.gaussian_expected_log_lik` |
| `GaussianLogLikelihood` (analytical fast path) | `gaussx.GaussianLogLikelihood` |

---

## 6  References

1. Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press.
2. Williams, C. K. I. & Barber, D. (1998). *Bayesian classification with Gaussian processes.* IEEE TPAMI.
3. Vanhatalo, J., Jylanki, P., & Vehtari, A. (2009). *Gaussian process regression with Student-t likelihood.* NeurIPS.
4. Ferrari, S. & Cribari-Neto, F. (2004). *Beta regression for modelling rates and proportions.* JASA.
5. Saul, A. D., Hensman, J., Vehtari, A., & Lawrence, N. D. (2016). *Chained Gaussian Processes.* AISTATS.
6. Lazaro-Gredilla, M. & Titsias, M. K. (2011). *Variational Heteroscedastic Gaussian Process Regression.* ICML.
7. Bruinsma, W., et al. (2020). *Scalable Exact Inference in Multi-Output Gaussian Processes.* ICML.
8. Hensman, J., Matthews, A. G., & Ghahramani, Z. (2015). *Scalable Variational Gaussian Process Classification.* AISTATS.
9. Minka, T. P. (2001). *Expectation Propagation for approximate Bayesian inference.* UAI.
