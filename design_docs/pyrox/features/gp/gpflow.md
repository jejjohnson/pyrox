---
status: draft
version: 0.1.0
---

# pyrox x Novel GPflow Components: Gap Analysis

**Subject:** Components from GPflow relevant to pyrox.gp, focusing on items
not already covered by the BayesNewton, GPJax, and GPyTorch gap analyses.

**Date:** 2026-04-02

---

## 1  Summary

GPflow shares significant overlap with GPyTorch and GPJax. Most of its
kernels, likelihoods, and model patterns are already cataloged in the other
pyrox feature docs (likelihoods, variational_families, models, inference_strategies,
inducing_features, sde_kernels). This doc catalogs only the novel GPflow
components not already covered elsewhere.

---

## 2  Gap Catalog

### Gap 1: CGLB (Conjugate Gradient Lower Bound)

**Category:** Scalable ELBO

**Math:** Replace the Cholesky-based ELBO with a CG-based bound. The standard
sparse GP ELBO requires $\log|K_{uu}|$ and solves $K_{uu}^{-1} \mathbf{a}$.
CGLB replaces these with:

- **logdet via stochastic Lanczos quadrature:**
  $\log|K_{uu}| \approx n \cdot \mathbf{z}^\top \log(T) \mathbf{e}_1$
  where $T$ is the tridiagonal matrix from Lanczos on $K_{uu}$ with probe vector $\mathbf{z}$.

- **solve via conjugate gradients:**
  $K_{uu}^{-1} \mathbf{a} \approx \mathbf{x}_J$ after $J$ CG iterations.

- **CG-based lower bound on the ELBO:**
  $\mathcal{L}_{\text{CGLB}} = \mathcal{L}_{\text{ELBO}} - \frac{1}{2\sigma^2} \|\mathbf{r}_J\|^2_{K_{uu}^{-1}}$
  where $\mathbf{r}_J$ is the CG residual, providing a *tighter* bound than
  truncating CG without correction.

**Complexity:** $O(NMJ)$ per iteration where $J$ is CG steps, vs $O(NM^2 + M^3)$ for Cholesky. Memory $O(NM)$ vs $O(M^2)$.

```python
class CGLBModel(eqx.Module):
    """Conjugate-gradient lower bound for sparse GP."""
    kernel: Kernel
    inducing_points: Float[Array, "M D"]
    noise_variance: Float[Array, ""]
    max_cg_iters: int
    max_lanczos_iters: int
    n_probes: int

    def elbo(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, "N 1"],
    ) -> Float[Array, ""]:
        """CG-based ELBO with residual correction term."""
        ...

    def _cg_solve(
        self,
        K_uu: Float[Array, "M M"],
        rhs: Float[Array, "M K"],
    ) -> tuple[Float[Array, "M K"], Float[Array, "M K"]]:
        """CG solve returning (solution, residual)."""
        ...

    def _stochastic_logdet(
        self,
        K_uu: Float[Array, "M M"],
    ) -> Float[Array, ""]:
        """Stochastic Lanczos quadrature for log-determinant."""
        ...
```

**Ref:** Artemev, M., Burt, D. R., & van der Wilk, M. (2021). *Tighter Bounds on the Log Marginal Likelihood of Gaussian Process Regression Using Conjugate Gradients.* ICML.
[GPflow source: `gpflow/models/cglb.py`](https://github.com/GPflow/GPflow/blob/develop/gpflow/models/cglb.py)

---

### Gap 2: ArcCosine Kernel

**Category:** Kernel

**Math:** The ArcCosine kernel of order $n$ maps to an infinite-width neural
network with step/ReLU/squared-ReLU activations:

$$k_n(x, x') = \frac{2}{\pi} \|x\|^n \|x'\|^n \, J_n(\theta)$$

where $\theta = \arccos\!\left(\frac{x \cdot x'}{\|x\| \, \|x'\|}\right)$ and the $J_n$ functions are:

$$J_0(\theta) = \pi - \theta$$
$$J_1(\theta) = \sin\theta + (\pi - \theta)\cos\theta$$
$$J_2(\theta) = 3\sin\theta\cos\theta + (\pi - \theta)(1 + 2\cos^2\theta)$$

Order 0 corresponds to a step activation (Heaviside), order 1 to ReLU, order 2
to squared ReLU. The kernel is stationary in angle space and positive definite
on $\mathbb{R}^D$ for all $D$.

**Complexity:** $O(N^2 D)$ for the full Gram matrix (same as any dot-product kernel).

```python
class ArcCosineKernel(eqx.Module):
    """ArcCosine kernel (neural network correspondence)."""
    variance: Float[Array, ""]
    order: int                                  # 0, 1, or 2

    def __call__(
        self,
        x: Float[Array, " D"],
        x_prime: Float[Array, " D"],
    ) -> Float[Array, ""]:
        """Evaluate k_n(x, x')."""
        ...

    def _J_n(
        self,
        theta: Float[Array, ""],
        order: int,
    ) -> Float[Array, ""]:
        """Compute J_n(theta) for n = 0, 1, 2."""
        ...
```

**Ref:** Cho, Y. & Saul, L. K. (2009). *Kernel Methods for Deep Learning.* NeurIPS.
[GPflow source: `gpflow/kernels/misc.py`](https://github.com/GPflow/GPflow/blob/develop/gpflow/kernels/misc.py)

---

### Gap 3: Ordinal Likelihood

**Category:** Likelihood

**Math:** Cumulative probit model with ordered cutpoints
$c_0 = -\infty < c_1 < c_2 < \cdots < c_{K-1} < c_K = +\infty$ for $K$ ordinal categories:

$$p(y = k \mid f) = \Phi(c_k - f) - \Phi(c_{k-1} - f)$$

where $\Phi$ is the standard normal CDF. The cutpoints are constrained to be
monotonically increasing via a softplus-based parameterization:

$$c_k = c_1 + \sum_{j=2}^{k} \text{softplus}(\delta_j), \quad k \geq 2$$

The variational expectation under $q(f) = \mathcal{N}(\mu, \sigma^2)$:

$$\mathbb{E}_{q(f)}[\log p(y=k|f)] = \mathbb{E}\!\left[\log\!\left(\Phi\!\left(\frac{c_k - f}{1}\right) - \Phi\!\left(\frac{c_{k-1} - f}{1}\right)\right)\right]$$

computed via Gauss-Hermite quadrature.

**Complexity:** $O(NK)$ per evaluation, $O(NKQ)$ for quadrature with $Q$ nodes.

```python
class OrdinalLikelihood(eqx.Module):
    """Ordinal regression via cumulative probit."""
    n_classes: int                              # K
    cutpoint_deltas: Float[Array, "K-2"]        # unconstrained increments

    def log_prob(
        self,
        f: Float[Array, " N"],
        y: Int[Array, " N"],
    ) -> Float[Array, " N"]:
        """log p(y | f) for ordinal observations y in {0, ..., K-1}."""
        ...

    def cutpoints(self) -> Float[Array, "K-1"]:
        """Monotonic cutpoints from unconstrained deltas."""
        ...
```

**Ref:** Chu, W. & Ghahramani, Z. (2005). *Gaussian Processes for Ordinal Regression.* JMLR.
[GPflow source: `gpflow/likelihoods/ordinal.py`](https://github.com/GPflow/GPflow/blob/develop/gpflow/likelihoods/ordinal.py)

---

### Gap 4: Psi-statistics ($\mathbb{E}_{q(X)}[k]$)

**Category:** Variational (Bayesian GPLVM)

**Math:** Psi-statistics are expectations of kernel products under a variational
distribution $q(X) = \prod_n \mathcal{N}(x_n | \mu_n, S_n)$ over latent inputs.
Required for Bayesian GPLVM:

$$\Psi_0 = \sum_n \mathbb{E}_{q(x_n)}[k(x_n, x_n)]$$

$$\Psi_1 \in \mathbb{R}^{N \times M}: \quad [\Psi_1]_{nm} = \mathbb{E}_{q(x_n)}[k(x_n, z_m)]$$

$$\Psi_2 \in \mathbb{R}^{M \times M}: \quad [\Psi_2]_{mm'} = \sum_n \mathbb{E}_{q(x_n)}[k(x_n, z_m)\,k(x_n, z_{m'})]$$

For the RBF kernel $k(x, x') = \sigma^2 \exp\!\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$,
all three have closed-form expressions. For example:

$$[\Psi_1]_{nm} = \sigma^2 \prod_d \frac{1}{\sqrt{1 + S_{nd}/\ell_d^2}} \exp\!\left(-\frac{(\mu_{nd} - z_{md})^2}{2(\ell_d^2 + S_{nd})}\right)$$

**Complexity:** $\Psi_0$: $O(ND)$. $\Psi_1$: $O(NMD)$. $\Psi_2$: $O(NM^2D)$.

```python
def psi0(
    kernel: RBFKernel,
    q_mu: Float[Array, "N D"],
    q_var: Float[Array, "N D"],
) -> Float[Array, ""]:
    """Psi_0 = sum_n E_q[k(x_n, x_n)]."""
    ...

def psi1(
    kernel: RBFKernel,
    inducing: Float[Array, "M D"],
    q_mu: Float[Array, "N D"],
    q_var: Float[Array, "N D"],
) -> Float[Array, "N M"]:
    """Psi_1: E_q[k(x_n, z_m)]."""
    ...

def psi2(
    kernel: RBFKernel,
    inducing: Float[Array, "M D"],
    q_mu: Float[Array, "N D"],
    q_var: Float[Array, "N D"],
) -> Float[Array, "M M"]:
    """Psi_2: sum_n E_q[k(x_n, z_m) k(x_n, z_m')]."""
    ...
```

**Ref:** Titsias, M. & Lawrence, N. D. (2010). *Bayesian Gaussian Process Latent Variable Model.* AISTATS.
[GPflow source: `gpflow/expectations/`](https://github.com/GPflow/GPflow/tree/develop/gpflow/expectations)

---

### Gap 5: Changepoint Kernel

**Category:** Kernel

**Math:** A kernel that smoothly transitions between two (or more) base kernels
at learned changepoints:

$$k(x, x') = \sigma_1(x)\, k_1(x, x')\, \sigma_1(x') + \sigma_2(x)\, k_2(x, x')\, \sigma_2(x')$$

where the mixing functions are complementary sigmoids:

$$\sigma_1(x) = \text{sigmoid}\!\left(\frac{x - t}{s}\right), \qquad \sigma_2(x) = 1 - \sigma_1(x)$$

with changepoint location $t$ and steepness $s > 0$. Generalizes to $P$ segments:

$$k(x, x') = \sum_{p=1}^{P} \sigma_p(x)\, k_p(x, x')\, \sigma_p(x')$$

where $\sum_p \sigma_p(x) = 1$ via softmax over sigmoid activations.

**Complexity:** $O(PN^2D)$ — linear in the number of segments $P$, each requiring one base kernel evaluation.

```python
class ChangepointKernel(eqx.Module):
    """Kernel with learned changepoints between base kernels."""
    kernels: list[Kernel]                       # P base kernels
    locations: Float[Array, "P-1"]              # changepoint locations t
    steepness: Float[Array, "P-1"]              # sigmoid steepness s

    def __call__(
        self,
        x: Float[Array, " D"],
        x_prime: Float[Array, " D"],
    ) -> Float[Array, ""]:
        """k(x, x') = sum_p sigma_p(x) k_p(x, x') sigma_p(x')."""
        ...

    def _mixing_weights(
        self,
        x: Float[Array, " D"],
    ) -> Float[Array, " P"]:
        """Compute sigma_p(x) mixing weights summing to 1."""
        ...
```

**Ref:** Saatchi, Y., Turner, R. D., & Rasmussen, C. E. (2010). *Gaussian Process Change Point Models.* ICML.
[GPflow source: `gpflow/kernels/changepoints.py`](https://github.com/GPflow/GPflow/blob/develop/gpflow/kernels/changepoints.py)

---

### Gap 6: Coregion Kernel

**Category:** Kernel (multi-output)

**Math:** The intrinsic coregionalization model (ICM) factorizes the multi-output
kernel as:

$$K\bigl((x, d),\, (x', d')\bigr) = B_{dd'} \cdot k(x, x')$$

where $B \in \mathbb{R}^{P \times P}$ is a positive semi-definite mixing matrix
parameterizing correlations between $P$ outputs, and $k(x, x')$ is a shared
base kernel. $B$ is parameterized as:

$$B = W W^\top + \text{diag}(\kappa)$$

where $W \in \mathbb{R}^{P \times R}$ (rank $R$ factor) and $\kappa \in \mathbb{R}^P_{>0}$.
The full multi-output covariance is the Kronecker product $K = B \otimes K_{xx}$.

Input format: each observation is $(x, d)$ where $d \in \{0, \ldots, P-1\}$ is the
output index (integer).

**Complexity:** $O(P^2 N^2 D)$ for the full multi-output Gram matrix. Exploiting
Kronecker structure: $O(P^3 + N^3)$ for independent inputs.

```python
class CoregionKernel(eqx.Module):
    """Intrinsic coregionalization kernel for multi-output GPs."""
    W: Float[Array, "P R"]                      # low-rank factor
    kappa: Float[Array, " P"]                   # diagonal (positive)
    base_kernel: Kernel                         # shared spatial kernel

    def __call__(
        self,
        x: Float[Array, " D"],
        x_prime: Float[Array, " D"],
        d: Int[Array, ""],
        d_prime: Int[Array, ""],
    ) -> Float[Array, ""]:
        """k((x,d), (x',d')) = B[d,d'] * k_base(x, x')."""
        ...

    def mixing_matrix(self) -> Float[Array, "P P"]:
        """B = W W^T + diag(kappa)."""
        ...
```

**Ref:** Alvarez, M. A., Rosasco, L., & Lawrence, N. D. (2012). *Kernels for Vector-Valued Functions: A Review.* Foundations and Trends in ML.
[GPflow source: `gpflow/kernels/multioutput/kernels.py`](https://github.com/GPflow/GPflow/blob/develop/gpflow/kernels/multioutput/kernels.py)

---

### Gap 7: Heteroscedastic Likelihood

**Category:** Likelihood

**Math:** Input-dependent observation noise modeled by a second GP:

$$y_i \mid f_i, g_i \sim \mathcal{N}(f_i,\, \sigma^2(x_i)), \qquad \sigma^2(x_i) = \text{softplus}(g_i)$$

where $f \sim \mathcal{GP}(0, k_f)$ is the mean function and $g \sim \mathcal{GP}(0, k_g)$
is the log-noise function. The joint model has latent vector $\mathbf{h} = [f, g]^\top$
with block-diagonal prior:

$$p(\mathbf{h}) = \mathcal{GP}\!\left(\mathbf{0},\, \begin{bmatrix} K_f & 0 \\ 0 & K_g \end{bmatrix}\right)$$

The log-likelihood for a single observation:

$$\log p(y_i \mid f_i, g_i) = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log\,\text{softplus}(g_i) - \frac{(y_i - f_i)^2}{2\,\text{softplus}(g_i)}$$

Inference requires variational methods (SVGP with $2M$ inducing points) or
Laplace approximation over the joint $[f, g]$.

**Complexity:** $O(N(2M)^2)$ for SVGP with $M$ inducing points per latent function.

```python
class HeteroscedasticLikelihood(eqx.Module):
    """Gaussian likelihood with input-dependent noise from a second GP."""
    min_variance: float = 1e-6                  # numerical floor

    def log_prob(
        self,
        f: Float[Array, " N"],
        g: Float[Array, " N"],
        y: Float[Array, " N"],
    ) -> Float[Array, " N"]:
        """log p(y | f, g) with variance = softplus(g)."""
        ...

    def expected_log_prob(
        self,
        y: Float[Array, " N"],
        f_mu: Float[Array, " N"],
        f_var: Float[Array, " N"],
        g_mu: Float[Array, " N"],
        g_var: Float[Array, " N"],
    ) -> Float[Array, ""]:
        """E_q[log p(y | f, g)] via Gauss-Hermite quadrature."""
        ...
```

**Ref:** Goldberg, P. W., Williams, C. K. I., & Bishop, C. M. (1998). *Regression with Input-Dependent Noise: A Gaussian Process Treatment.* NeurIPS.
[GPflow source: `gpflow/likelihoods/heteroscedastic.py`](https://github.com/GPflow/GPflow/blob/develop/gpflow/likelihoods/)

---

### Gap 8: SGPR Marginal Likelihood Bound

**Category:** Model (sparse GP regression)

**Math:** Titsias' collapsed variational bound for sparse GP regression with
$M$ inducing points $Z$:

$$\mathcal{L}_{\text{SGPR}} = \log \mathcal{N}(\mathbf{y} \mid \mathbf{0},\, Q_{ff} + \sigma^2 I) - \frac{1}{2\sigma^2} \text{tr}(K_{ff} - Q_{ff})$$

where $Q_{ff} = K_{fu} K_{uu}^{-1} K_{uf}$ is the Nystrom approximation.
Expanding using the Woodbury identity:

$$\log \mathcal{N}(\mathbf{y} \mid \mathbf{0},\, Q_{ff} + \sigma^2 I) = -\frac{N}{2}\log(2\pi) - \frac{1}{2}\log|K_{uu}| - \frac{1}{2}\log|\Lambda| - \frac{1}{2}\left(\frac{\|\mathbf{y}\|^2}{\sigma^2} - \mathbf{c}^\top \Lambda^{-1} \mathbf{c}\right)$$

where $\Lambda = K_{uu} + \sigma^{-2} K_{uf} K_{fu}$ and $\mathbf{c} = \sigma^{-2} K_{uf} \mathbf{y}$.
The trace penalty $\frac{1}{2\sigma^2}\text{tr}(K_{ff} - Q_{ff})$ acts as a
regularizer driving the inducing points toward the data.

**Complexity:** $O(NM^2)$ time, $O(NM)$ memory. Avoids the $O(N^3)$ cost of exact GP regression.

```python
class SGPRModel(eqx.Module):
    """Sparse GP regression with Titsias' collapsed bound."""
    kernel: Kernel
    inducing_points: Float[Array, "M D"]
    noise_variance: Float[Array, ""]

    def log_marginal_likelihood(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, "N 1"],
    ) -> Float[Array, ""]:
        """Collapsed variational bound (SGPR ELBO)."""
        ...

    def predict(
        self,
        X_train: Float[Array, "N D"],
        y_train: Float[Array, "N 1"],
        X_test: Float[Array, "Nt D"],
    ) -> tuple[Float[Array, "Nt 1"], Float[Array, "Nt 1"]]:
        """Posterior mean and variance at test points."""
        ...
```

**Ref:** Titsias, M. (2009). *Variational Learning of Inducing Variables in Sparse Gaussian Processes.* AISTATS.
[GPflow source: `gpflow/models/sgpr.py`](https://github.com/GPflow/GPflow/blob/develop/gpflow/models/sgpr.py)

---

## 3  Notes

- **CGLB** replaces Cholesky-based ELBO with CG-based bound using gaussx's `CGSolver`
- **Psi-statistics** are analytical $\mathbb{E}_{q(X)}[k(X, Z)]$ needed for Bayesian GPLVM
- **ArcCosine kernel** gives neural-network-equivalent GPs (deep kernel connection)
- **Coregion kernel** decomposes naturally as Kronecker product, integrates with gaussx's `Kronecker` operator
- **SGPR** and **CGLB** both leverage gaussx primitives (`solve`, `logdet`, `cholesky`)

---

## 4  References

1. Artemev, M., Burt, D. R., & van der Wilk, M. (2021). *Tighter Bounds on the Log Marginal Likelihood of Gaussian Process Regression Using Conjugate Gradients.* ICML.
2. Cho, Y. & Saul, L. K. (2009). *Kernel Methods for Deep Learning.* NeurIPS.
3. Chu, W. & Ghahramani, Z. (2005). *Gaussian Processes for Ordinal Regression.* JMLR.
4. Titsias, M. & Lawrence, N. D. (2010). *Bayesian Gaussian Process Latent Variable Model.* AISTATS.
5. Saatchi, Y., Turner, R. D., & Rasmussen, C. E. (2010). *Gaussian Process Change Point Models.* ICML.
6. Alvarez, M. A., Rosasco, L., & Lawrence, N. D. (2012). *Kernels for Vector-Valued Functions: A Review.* Foundations and Trends in ML.
7. Goldberg, P. W., Williams, C. K. I., & Bishop, C. M. (1998). *Regression with Input-Dependent Noise: A Gaussian Process Treatment.* NeurIPS.
8. Titsias, M. (2009). *Variational Learning of Inducing Variables in Sparse Gaussian Processes.* AISTATS.
