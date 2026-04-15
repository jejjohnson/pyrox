---
status: draft
version: 0.1.0
---

# pyrox.gp x Pathwise Posterior Sampling

**Subject:** Efficient GP posterior function samples via Matheron's rule,
random Fourier features, and decoupled sparse representations
(Wilson et al. 2020, 2021; Cheng & Boots 2017).

**Date:** 2026-04-02

---

## 1  The Problem

Given a GP posterior $f \mid \mathcal{D} \sim \mathcal{GP}(\mu_\text{post}, k_\text{post})$, we want
to draw *function* samples $\hat{f} \sim p(f \mid \mathcal{D})$ that can be evaluated at
arbitrary test points $x_*$.

**Standard approach.** Given $N_*$ test points, form the posterior covariance
$K_{**}^\text{post} \in \mathbb{R}^{N_* \times N_*}$ and compute its Cholesky:

$$\hat{f}(X_*) = \mu_\text{post}(X_*) + L_\text{post}\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

This costs $O(N_*^3)$ per draw and $O(N_*^2)$ memory. Worse, the sample is
defined only at the fixed grid $X_*$ -- evaluating at a new point requires
refactoring the entire covariance.

**Why this matters:**

- **Thompson sampling** (Bayesian optimization): draw a function, maximize it.
  Needs evaluation at candidate points not known in advance.
- **Posterior visualization:** dense function draws over a grid.
- **Uncertainty propagation:** push posterior samples through a downstream model.

**Goal:** $O(1)$ cost per test-point evaluation (after a one-time $O(N^3)$
or $O(M^3)$ setup), with samples that are *consistent functions* -- the same
random draw evaluated at different inputs.

---

## 2  Matheron's Rule

### 2.1  Conditional Gaussian Identity

Let $(f_*, f_X)$ be jointly Gaussian:

$$\begin{pmatrix} f_* \\ f_X \end{pmatrix} \sim \mathcal{N}\!\left(\begin{pmatrix} \mu_* \\ \mu_X \end{pmatrix}, \begin{pmatrix} K_{**} & K_{*X} \\ K_{X*} & K_{XX} \end{pmatrix}\right)$$

The conditional is:

$$f_* \mid f_X \sim \mathcal{N}\!\bigl(\mu_* + K_{*X} K_{XX}^{-1}(f_X - \mu_X),\; K_{**} - K_{*X} K_{XX}^{-1} K_{X*}\bigr)$$

A sample from the conditional can be written as:

$$\hat{f}_* \mid f_X = \hat{f}_* + K_{*X} K_{XX}^{-1}(f_X - \hat{f}_X)$$

where $\hat{f}_* \sim \mathcal{N}(\mu_*, K_{**})$ is a *prior* sample. This is
**Matheron's rule** (Journel & Huijbregts 1978): a conditional sample equals a
prior sample plus a deterministic correction.

### 2.2  Applying to GP Posteriors

For a GP with observations $y = f(X) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$:

$$\hat{f}_\text{post}(\cdot) = \hat{f}_\text{prior}(\cdot) + k(\cdot, X)\,(K_{XX} + \sigma^2 I)^{-1}\bigl(y - \hat{f}_\text{prior}(X) - \hat{\varepsilon}\bigr)$$

where $\hat{f}_\text{prior} \sim \mathcal{GP}(0, k)$ and $\hat{\varepsilon} \sim \mathcal{N}(0, \sigma^2 I)$.

**Derivation.** Augment the joint with noise:

$$\begin{pmatrix} f(\cdot) \\ y \end{pmatrix} \sim \mathcal{N}\!\left(0, \begin{pmatrix} k(\cdot, \cdot) & k(\cdot, X) \\ k(X, \cdot) & K_{XX} + \sigma^2 I \end{pmatrix}\right)$$

Apply the conditional identity with $f_X \to y$. A joint prior sample is
$(\hat{f}_\text{prior}(\cdot),\; \hat{f}_\text{prior}(X) + \hat\varepsilon)$.
Substituting:

$$\hat{f}_\text{post}(\cdot) = \hat{f}_\text{prior}(\cdot) + k(\cdot, X)(K_{XX} + \sigma^2 I)^{-1}\bigl(y - \hat{f}_\text{prior}(X) - \hat\varepsilon\bigr)$$

### 2.3  Structure of the Update

Define the **correction weights**:

$$v = (K_{XX} + \sigma^2 I)^{-1}\bigl(y - \hat{f}_\text{prior}(X) - \hat\varepsilon\bigr) \in \mathbb{R}^N$$

These are computed once ($O(N^3)$ via Cholesky, or $O(N^2)$ if the
training-time factorization $L L^\top = K_{XX} + \sigma^2 I$ is cached).
Then evaluation at any $x_*$:

$$\hat{f}_\text{post}(x_*) = \hat{f}_\text{prior}(x_*) + k(x_*, X)^\top v$$

Cost per point: $O(N)$ for the kernel vector $k(x_*, X)$ plus $O(D)$ for
the prior draw (see Section 3). No additional Cholesky.

---

## 3  Prior Function Samples via Random Fourier Features

### 3.1  Bochner's Theorem

For a stationary kernel $k(x, x') = k(x - x')$, Bochner's theorem gives:

$$k(\tau) = \int_{\mathbb{R}^d} e^{i\omega^\top \tau}\, S(\omega)\, d\omega$$

where $S(\omega) \geq 0$ is the spectral density (the Fourier transform of $k$).
Normalizing: $p(\omega) = S(\omega) / \sigma_k^2$ is a valid density.

### 3.2  Random Fourier Feature (RFF) Approximation

Draw $D$ frequencies and phases:

$$\omega_j \sim p(\omega), \qquad b_j \sim \mathrm{Uniform}(0, 2\pi), \qquad j = 1, \ldots, D$$

The RFF feature map:

$$\phi(x) = \sqrt{\frac{2\sigma_k^2}{D}} \begin{pmatrix} \cos(\omega_1^\top x + b_1) \\ \vdots \\ \cos(\omega_D^\top x + b_D) \end{pmatrix} \in \mathbb{R}^D$$

satisfies $\mathbb{E}[\phi(x)^\top \phi(x')] = k(x, x')$.

### 3.3  Prior Function Sample

A sample from $\mathcal{GP}(0, k)$ is approximated as:

$$\hat{f}_\text{prior}(x) = \phi(x)^\top w, \qquad w \sim \mathcal{N}(0, I_D)$$

Expanding:

$$\hat{f}_\text{prior}(x) = \sqrt{\frac{2\sigma_k^2}{D}} \sum_{j=1}^{D} \cos(\omega_j^\top x + b_j)\, w_j$$

This is a *consistent function* -- the same $(\omega, b, w)$ triple gives the
same function at any input $x$. Cost per evaluation: $O(Dd)$ where $d$ is the
input dimension.

### 3.4  Spectral Densities for Common Kernels

| Kernel | $S(\omega)$ | Sampling |
|---|---|---|
| RBF / Squared Exponential | $\sigma^2 (2\pi)^{d/2} \ell^d \exp(-2\pi^2 \ell^2 \|\omega\|^2)$ | $\omega \sim \mathcal{N}(0, \ell^{-2} I)$ |
| Matern-$\nu$ | $\sigma^2 \frac{2^d \pi^{d/2} \Gamma(\nu + d/2) (2\nu)^\nu}{\Gamma(\nu) \ell^{2\nu}} (2\nu/\ell^2 + 4\pi^2\|\omega\|^2)^{-(\nu+d/2)}$ | $\omega \sim \text{Student-}t_{2\nu}(0, \ell^{-2}I)$ |

### 3.5  Connection to `pyrox.nn.RandomFourierFeatures`

The `pyrox.nn.RandomFourierFeatures` module already implements the RFF feature
map $\phi(x)$. For pathwise sampling, we reuse the same spectral-density
sampling logic from `pyrox._core.spectral` to draw $\omega$ and $b$, then
draw the weight vector $w \sim \mathcal{N}(0, I)$ to produce the prior
function sample.

---

## 4  Sparse Pathwise Sampling (SVGP)

### 4.1  Variational Sparse GP Setup

In the SVGP framework (Titsias 2009, Hensman et al. 2013), we place $M$
inducing points at locations $Z$ with inducing values $u = f(Z)$.
The variational posterior is:

$$q(u) = \mathcal{N}(m, S)$$

where $m \in \mathbb{R}^M$ and $S \in \mathbb{R}^{M \times M}$ are variational
parameters. The approximate posterior predictive is:

$$q(f_*) = \mathcal{N}\!\bigl(\mu_q(x_*),\; \sigma_q^2(x_*)\bigr)$$

$$\mu_q(x_*) = k(x_*, Z)\,K_{ZZ}^{-1}\,m$$

$$\sigma_q^2(x_*) = k(x_*, x_*) - k(x_*, Z)\,K_{ZZ}^{-1}\bigl(K_{ZZ} - S\bigr)\,K_{ZZ}^{-1}\,k(Z, x_*)$$

### 4.2  Pathwise Formulation (Wilson et al. 2021)

Apply Matheron's rule to the variational posterior. A sample from $q(f)$:

$$\hat{f}_\text{post}(x_*) = \hat{f}_\text{prior}(x_*) + k(x_*, Z)\,K_{ZZ}^{-1}\bigl(m - \hat{f}_\text{prior}(Z)\bigr) + k(x_*, Z)\,K_{ZZ}^{-1}\,L_S\,\varepsilon$$

where:

- $\hat{f}_\text{prior}(\cdot)$ is an RFF prior draw (Section 3)
- $L_S = \mathrm{chol}(S)$
- $\varepsilon \sim \mathcal{N}(0, I_M)$

**Derivation.** The variational posterior implies $u \sim \mathcal{N}(m, S)$.
The conditional $f_* \mid u$ under the prior is:

$$f_* \mid u = f_\text{prior}(x_*) + k(x_*, Z) K_{ZZ}^{-1}(u - f_\text{prior}(Z))$$

Substituting a sample $\hat{u} = m + L_S \varepsilon$:

$$\hat{f}_\text{post}(x_*) = \hat{f}_\text{prior}(x_*) + k(x_*, Z) K_{ZZ}^{-1}(m - \hat{f}_\text{prior}(Z)) + k(x_*, Z) K_{ZZ}^{-1} L_S \varepsilon$$

### 4.3  Precomputable Quantities

Define:

$$\alpha = K_{ZZ}^{-1}\bigl(m - \hat{f}_\text{prior}(Z)\bigr) \in \mathbb{R}^M$$

$$\beta = K_{ZZ}^{-1} L_S\,\varepsilon \in \mathbb{R}^M$$

$$v = \alpha + \beta \in \mathbb{R}^M$$

All three are computed once ($O(M^3)$ for $K_{ZZ}^{-1}$, cached). Then:

$$\hat{f}_\text{post}(x_*) = \hat{f}_\text{prior}(x_*) + k(x_*, Z)^\top v$$

Cost per point: $O(Md + Dd)$ -- the kernel vector plus the prior draw.

---

## 5  Decoupled Representation

### 5.1  Motivation

In the sparse pathwise formula (Section 4.2), the prior sample
$\hat{f}_\text{prior}$ appears in two places:

1. As the "base" function $\hat{f}_\text{prior}(x_*)$
2. Inside the correction via $\hat{f}_\text{prior}(Z)$

Both use the same RFF basis with $D$ features. Cheng & Boots (2017) observe
that these two roles have different approximation requirements:

- The **mean** correction $k(x_*, Z) K_{ZZ}^{-1} m$ is exact (it uses the
  true kernel, no RFF approximation).
- The **prior cancellation** $\hat{f}_\text{prior}(x_*) - k(x_*, Z) K_{ZZ}^{-1} \hat{f}_\text{prior}(Z)$
  involves the RFF approximation error in both terms. If these errors don't
  cancel well, the sample quality degrades.

### 5.2  Decoupled Bases

Use separate feature sets for the mean and the residual:

$$\hat{f}_\text{post}(x_*) = \underbrace{\phi_\text{mean}(x_*)^\top w_\text{mean}}_{\text{mean basis}} + \underbrace{\phi_\text{res}(x_*)^\top w_\text{res}}_{\text{residual basis}}$$

**Mean basis:** Use the exact inducing-point features (Nystrom approximation):

$$\phi_\text{mean}(x) = K_{ZZ}^{-1/2}\,k(Z, x) \in \mathbb{R}^M$$

This gives the exact posterior mean with $w_\text{mean} = K_{ZZ}^{-1/2} m$.

**Residual basis:** Use RFF features $\phi_\text{res}(x) \in \mathbb{R}^D$ for the
prior draw component, with weights chosen so that the residual covariance
is correct.

### 5.3  Benefits

- The posterior mean is represented *exactly* (no RFF error).
- The residual captures the remaining variance and can tolerate more RFF error.
- Empirically, $D$ can be much smaller for equivalent sample quality (Cheng &
  Boots 2017 report 2-5x reduction).

---

## 6  Complexity

| Method | Setup | Per-point eval | Memory | Sample consistency |
|---|---|---|---|---|
| Standard Cholesky | $O(N_*^3)$ | -- (grid-locked) | $O(N_*^2)$ | Grid only |
| Matheron (exact GP) | $O(N^3)$ train Cholesky | $O(N + D)$ | $O(N + D)$ | Anywhere |
| Sparse pathwise (SVGP) | $O(M^3)$ | $O(M + D)$ | $O(M + D)$ | Anywhere |
| Decoupled sparse | $O(M^3)$ | $O(M + D)$ | $O(M + D)$ | Anywhere |

Where:

- $N$ = number of training points
- $N_*$ = number of test points
- $M$ = number of inducing points ($M \ll N$)
- $D$ = number of RFF features

**Typical settings:** $D \in [500, 2000]$ for RBF in $d \leq 10$. $M \in [64, 1024]$.

---

## 7  API

### Layer 0 -- Primitives (`pyrox.gp._src.pathwise`)

Pure functions with no state.

```python
def rff_prior_sample(
    key: PRNGKeyArray,
    kernel: Kernel,
    n_features: int = 1024,
) -> tuple[Float[Array, "D d"], Float[Array, " D"], Float[Array, " D"]]:
    """Draw RFF parameters for a prior function sample.

    Samples frequencies omega ~ S(omega), phases b ~ Uniform(0, 2pi),
    and weights w ~ N(0, I).

    Returns (omega, b, w) tuple.

    Complexity: O(D * d) where D = n_features, d = input dim.
    """
    ...

def eval_rff_sample(
    x: Float[Array, "... d"],
    omega: Float[Array, "D d"],
    b: Float[Array, " D"],
    w: Float[Array, " D"],
    variance: float = 1.0,
) -> Float[Array, "..."]:
    """Evaluate an RFF prior sample at points x.

    f(x) = sqrt(2 * variance / D) * sum_j cos(omega_j @ x + b_j) * w_j

    Complexity: O(... * D * d) where ... is the batch shape.
    """
    ...

def matheron_correction(
    x_star: Float[Array, "N_star d"],
    x_train: Float[Array, "N d"],
    y_train: Float[Array, " N"],
    kernel: Kernel,
    noise_var: float,
    f_prior_train: Float[Array, " N"],
    noise_sample: Float[Array, " N"],
) -> Float[Array, " N_star"]:
    """Compute Matheron's posterior correction at test points.

    correction = k(x_star, x_train) @ (K_XX + sigma^2 I)^{-1} (y - f_prior(X) - eps)

    Complexity: O(N^3) for the solve (O(N^2) if Cholesky is cached).
    """
    ...

def matheron_weights(
    x_train: Float[Array, "N d"],
    y_train: Float[Array, " N"],
    kernel: Kernel,
    noise_var: float,
    f_prior_train: Float[Array, " N"],
    noise_sample: Float[Array, " N"],
) -> Float[Array, " N"]:
    """Precompute correction weights v = (K_XX + sigma^2 I)^{-1} (y - f_prior(X) - eps).

    Complexity: O(N^3) for the Cholesky + solve (O(N^2) if factorization cached).
    """
    ...

def sparse_pathwise_weights(
    z: Float[Array, "M d"],
    kernel: Kernel,
    m: Float[Array, " M"],
    S: Float[Array, "M M"],
    f_prior_z: Float[Array, " M"],
    key: PRNGKeyArray,
) -> Float[Array, " M"]:
    """Precompute sparse pathwise weights v = alpha + beta.

    alpha = K_ZZ^{-1} (m - f_prior(Z))
    beta  = K_ZZ^{-1} L_S epsilon,   epsilon ~ N(0, I)

    Complexity: O(M^3) for K_ZZ factorization.
    """
    ...
```

### Layer 1 -- Sampler Module (`pyrox.gp.samplers`)

```python
class PathwiseSampler(eqx.Module):
    """Callable posterior function sample.

    Stores the RFF parameters (omega, b, w) for the prior draw and the
    precomputed correction weights v. Evaluating the sampler at new
    points is O(N + D) per point (exact GP) or O(M + D) (sparse).
    """
    omega: Float[Array, "D d"]
    b: Float[Array, " D"]
    w: Float[Array, " D"]
    variance: float

    # Correction
    ref_points: Float[Array, "R d"]       # X (exact) or Z (sparse)
    correction_weights: Float[Array, " R"]
    kernel: Kernel

    @staticmethod
    def from_exact_gp(
        key: PRNGKeyArray,
        kernel: Kernel,
        x_train: Float[Array, "N d"],
        y_train: Float[Array, " N"],
        noise_var: float,
        n_features: int = 1024,
    ) -> "PathwiseSampler":
        """Build a pathwise sampler for an exact GP posterior.

        1. Draw RFF prior sample parameters (omega, b, w).
        2. Evaluate f_prior(X) at training points.
        3. Compute correction weights v.

        Complexity: O(N^3) one-time, then O(N + D) per evaluation.
        """
        ...

    @staticmethod
    def from_svgp(
        key: PRNGKeyArray,
        kernel: Kernel,
        z: Float[Array, "M d"],
        m: Float[Array, " M"],
        S: Float[Array, "M M"],
        n_features: int = 1024,
    ) -> "PathwiseSampler":
        """Build a pathwise sampler for a sparse variational GP.

        1. Draw RFF prior sample parameters (omega, b, w).
        2. Evaluate f_prior(Z) at inducing points.
        3. Compute sparse pathwise weights v = alpha + beta.

        Complexity: O(M^3) one-time, then O(M + D) per evaluation.
        """
        ...

    def __call__(
        self, x: Float[Array, "... d"],
    ) -> Float[Array, "..."]:
        """Evaluate posterior sample at arbitrary points.

        f_post(x) = f_prior(x) + k(x, ref_points) @ correction_weights

        Complexity: O(R * d + D * d) per point, where R = N or M.
        """
        f_prior = eval_rff_sample(x, self.omega, self.b, self.w, self.variance)
        k_vec = self.kernel(x, self.ref_points)  # [..., R]
        return f_prior + k_vec @ self.correction_weights


class DecoupledPathwiseSampler(eqx.Module):
    """Decoupled pathwise sampler (Cheng & Boots 2017).

    Uses exact inducing-point features for the mean and RFF for the
    residual, improving sample quality for a given D.
    """
    # Mean component (Nystrom)
    z: Float[Array, "M d"]
    mean_weights: Float[Array, " M"]      # K_ZZ^{-1/2} m
    kernel: Kernel

    # Residual component (RFF)
    omega: Float[Array, "D d"]
    b: Float[Array, " D"]
    res_weights: Float[Array, " D"]
    variance: float

    @staticmethod
    def from_svgp(
        key: PRNGKeyArray,
        kernel: Kernel,
        z: Float[Array, "M d"],
        m: Float[Array, " M"],
        S: Float[Array, "M M"],
        n_features: int = 512,
    ) -> "DecoupledPathwiseSampler":
        """Build a decoupled sampler for an SVGP posterior.

        Complexity: O(M^3) one-time.
        """
        ...

    def __call__(
        self, x: Float[Array, "... d"],
    ) -> Float[Array, "..."]:
        """Evaluate posterior sample.

        f(x) = phi_mean(x) @ w_mean + phi_res(x) @ w_res

        Complexity: O(M * d + D * d) per point.
        """
        ...
```

### Layer 2 -- NumPyro Integration

```python
def pathwise_sample(
    key: PRNGKeyArray,
    kernel: Kernel,
    x_train: Float[Array, "N d"],
    y_train: Float[Array, " N"],
    noise_var: float,
    n_samples: int = 1,
    n_features: int = 1024,
) -> PathwiseSampler | list[PathwiseSampler]:
    """Draw pathwise posterior sample(s) for use in NumPyro models.

    Suitable for Thompson sampling in Bayesian optimization:

        def bo_model(x_candidates):
            kernel = ...  # from hyperparameter posterior
            sampler = pathwise_sample(key, kernel, X, y, noise)
            f_vals = jax.vmap(sampler)(x_candidates)
            x_best = x_candidates[jnp.argmax(f_vals)]

    Complexity: O(N^3) setup per sample, O(N + D) per evaluation.
    """
    ...

def sparse_pathwise_sample(
    key: PRNGKeyArray,
    kernel: Kernel,
    z: Float[Array, "M d"],
    m: Float[Array, " M"],
    S: Float[Array, "M M"],
    n_samples: int = 1,
    n_features: int = 1024,
    decoupled: bool = False,
) -> PathwiseSampler | DecoupledPathwiseSampler:
    """Draw pathwise posterior sample(s) from an SVGP.

    If decoupled=True, uses the Cheng & Boots (2017) representation.

    Complexity: O(M^3) setup per sample, O(M + D) per evaluation.
    """
    ...
```

---

## 8  Implementation Notes

### RFF vs HSGP for prior draws

| Approach | Pros | Cons |
|---|---|---|
| RFF | Works in any $d$; simple; no domain bounds | Convergence $O(1/\sqrt{D})$; high $D$ for smooth kernels |
| HSGP (Hilbert space GP) | Exponential convergence in 1-D/2-D; basis is deterministic | Requires bounded domain; cost grows fast with $d$ |

**Recommendation:** Use RFF as the default. Offer HSGP as an option for $d \leq 3$
when the domain is known, via a `prior_basis="rff"` / `prior_basis="hsgp"` flag
on `PathwiseSampler`.

### Caching correction weights

The correction weights $v = (K_{XX} + \sigma^2 I)^{-1} r$ are the most expensive
part. Key caching strategies:

1. **Cache the Cholesky factor.** If $L L^\top = K_{XX} + \sigma^2 I$ is already
   computed during training, each new sample only requires two triangular
   solves ($O(N^2)$) rather than a full factorization ($O(N^3)$).

2. **Batch multiple samples.** Draw $S$ sets of $(\omega^{(s)}, b^{(s)}, w^{(s)})$,
   stack the residuals $r^{(s)} = y - f_\text{prior}^{(s)}(X) - \hat\varepsilon^{(s)}$
   into a matrix $R \in \mathbb{R}^{N \times S}$, and solve $L L^\top V = R$ once.

3. **For SVGP:** $K_{ZZ}$ factorization is $O(M^3)$ and can be reused across
   all samples. Store $L_{ZZ}^{-1}$ and $L_S$ as part of the sampler state.

### Batched samples via `vmap`

```python
keys = jax.random.split(key, n_samples)
samplers = jax.vmap(
    lambda k: PathwiseSampler.from_exact_gp(k, kernel, X, y, noise, D)
)(keys)

# Evaluate all samples at test points
f_samples = jax.vmap(lambda s: s(x_test))(samplers)  # [n_samples, N_test]
```

This requires `PathwiseSampler` to be a valid PyTree (guaranteed by `eqx.Module`).

### Numerical considerations

- **RFF feature count.** For RBF kernels in $d$ dimensions, $D \approx 10 d$ to
  $100 d$ is typically sufficient. For Matern-1/2, fewer features suffice
  (rougher kernels have flatter spectral densities).

- **Conditioning.** The solve $(K_{XX} + \sigma^2 I)^{-1} r$ is well-conditioned
  when $\sigma^2 > 0$. For near-noiseless settings, use a jitter $\sim 10^{-6}$.

- **JIT compilation.** All operations (RFF sampling, kernel evaluations, solves)
  are pure JAX and fully JIT-compatible. The `PathwiseSampler.__call__` method
  traces cleanly through `jax.jit`.

---

## 9  References

1. Wilson, J. T., Borovitskiy, V., Terenin, A., Mostowsky, P. & Deisenroth, M. P. (2020). *Efficiently Sampling Functions from Gaussian Process Posteriors.* ICML.

2. Wilson, J. T., Borovitskiy, V., Terenin, A., Mostowsky, P. & Deisenroth, M. P. (2021). *Pathwise Conditioning of Gaussian Processes.* JMLR.

3. Cheng, C.-A. & Boots, B. (2017). *Variational Inference for Gaussian Process Models with Linear Complexity.* NeurIPS.

4. Rahimi, A. & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines.* NeurIPS.

5. Journel, A. G. & Huijbregts, C. J. (1978). *Mining Geostatistics.* Academic Press.

6. Hensman, J., Fusi, N. & Lawrence, N. D. (2013). *Gaussian Processes for Big Data.* UAI.

7. Titsias, M. (2009). *Variational Learning of Inducing Variables in Sparse Gaussian Processes.* AISTATS.

8. Solin, A. & Sarkka, S. (2020). *Hilbert Space Methods for Reduced-Rank Gaussian Process Regression.* Statistics and Computing.
