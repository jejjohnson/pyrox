---
status: draft
version: 0.1.0
---

# pyrox x Random Feature Regression: Gap Analysis

**Subject:** Bayesian linear regression in random feature spaces -- Sparse
Spectrum GP (SSGP), Variational Fourier Features (VFF), and related
weight-space GP approximations.

**Date:** 2026-04-02

---

## 1  Scope

Random Fourier Features (RFF) approximate a stationary kernel as an
inner product in feature space: $k(x, x') \approx \phi(x)^T \phi(x')$.
This turns GP inference into Bayesian linear regression on the features,
reducing $O(N^3)$ to $O(D^2 N)$ where $D$ is the number of features.

`pyrox.nn` already provides `RandomFourierFeatures` as a Layer 1 component.
This doc catalogs the model-level patterns that compose RFF with Bayesian
inference, and identifies gaps in the feature layer itself.

**Ownership within the merged pyrox package:**
- **pyrox.nn** owns: `RandomFourierFeatures` layer, `RandomKitchenSinks`,
  feature map implementations, spectral density samplers
- **pyrox.gp** owns: GP-specific pathwise sampling that *uses* RFF
  (`rff_prior_draw`, `PathwiseSampler`)
- **gaussx** owns: Matheron update identity, structured MVN sampling

---

## 2  What pyrox.nn Already Provides

| Component | Status | Notes |
|---|---|---|
| `RandomFourierFeatures` | Designed | RBF kernel approximation, optionally learnable lengthscale |
| `RandomKitchenSinks` | Designed | Fixed-frequency variant |

---

## 3  Gap Catalog

### Gap 1: Extended Spectral Densities

**Source:** Lazaro-Gredilla et al. (2010), Mutny & Krause (2018)

**Priority:** High

The current `RandomFourierFeatures` only supports the RBF kernel (Gaussian
spectral density). Other stationary kernels have known spectral densities
that should be supported.

```python
class RandomFourierFeatures(PyroxModule):
    """Extended RFF layer supporting multiple stationary kernels.

    The spectral density determines which kernel is approximated:
    - Gaussian -> RBF kernel
    - Student-t(2*nu, 0, 1/l) -> Matern(nu) kernel
    - Sum of Gaussians -> mixture kernel
    - Learned (normalizing flow) -> flexible kernel

    Parameters
    ----------
    n_features : int
        Number of random features D. Output dim is 2D (cos + sin).
    kernel_type : str
        One of "rbf", "matern12", "matern32", "matern52".
    learn_lengthscale : bool
        If True, lengthscale is a learnable parameter.
    learn_frequencies : bool
        If True, frequencies are variational parameters (VSSGP).
    """
    n_features: int
    kernel_type: str
    learn_lengthscale: bool
    learn_frequencies: bool

    def spectral_density(self, key: PRNGKey) -> Float[Array, "D d"]:
        """Sample frequencies from the kernel's spectral density."""
        ...

    def __call__(self, x: Float[Array, "N d"]) -> Float[Array, "N 2D"]:
        """Compute feature map phi(x) = sqrt(2/D) [cos(Wx+b), sin(Wx+b)]."""
        ...
```

**Kernel spectral densities:**

| Kernel | Spectral density $S(\omega)$ |
|---|---|
| RBF | $\mathcal{N}(0, \ell^{-2} I)$ |
| Matern-1/2 | Cauchy $(0, \ell^{-1})$ |
| Matern-3/2 | Student-t $(3, 0, \sqrt{3}/\ell)$ |
| Matern-5/2 | Student-t $(5, 0, \sqrt{5}/\ell)$ |
| Periodic | Discrete: $\omega_k = 2\pi k / p$, weights from $I_k(\ell^{-2})$ |

**Estimated lines:** ~80 (extend existing layer)

---

### Gap 2: Variational Spectral Features (VSSGP)

**Source:** Gal & Turner (2015) *Improving the Gaussian Process Sparse
Spectrum Approximation by Representing Uncertainty in Frequency Domain*,
Mutny & Krause (2018)

**Priority:** Medium

In standard SSGP, frequencies $\omega_d$ are drawn once and fixed. VSSGP
treats them as variational parameters with learned posteriors, avoiding the
variance starvation problem.

```python
class VariationalFourierFeatures(PyroxModule):
    """RFF with variational frequency posteriors (VSSGP).

    Instead of fixing frequencies omega ~ S(omega), learns a
    variational posterior q(omega) = N(mu_omega, sigma_omega^2)
    over each frequency. This is a PyroxModule -- frequencies are
    NumPyro sample sites.
    """
    n_features: int
    input_dim: int

    def __call__(self, x: Float[Array, "N d"]) -> Float[Array, "N 2D"]:
        # Sample frequencies from variational posterior
        omega = self.pyrox_sample(
            "omega",
            dist.Normal(self.omega_loc, jnp.exp(self.omega_log_scale)),
        )
        # Compute features with sampled frequencies
        proj = x @ omega.T  # (N, D)
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1) / jnp.sqrt(self.n_features)
```

**Key design consideration:** VSSGP needs the KL divergence between the
frequency posterior and the spectral density prior. For RBF this is
KL(Normal || Normal), which NumPyro handles automatically via `Trace_ELBO`.

**Estimated lines:** ~60

---

### Gap 3: Orthogonal Random Features

**Source:** Yu et al. (2016) *Orthogonal Random Features*, Choromanski
et al. (2017)

**Priority:** Low

Replace i.i.d. frequency samples with structured orthogonal matrices
(via QR decomposition of random Gaussian matrices). Reduces variance
of the kernel approximation without increasing the number of features.

```python
class OrthogonalRandomFeatures(eqx.Module):
    """RFF with orthogonal frequency matrix for lower-variance approximation.

    Uses blocks of orthogonal matrices: W = S * Q where Q is from
    QR decomposition of a random Gaussian matrix and S contains
    chi-distributed norms.
    """
    n_features: int
    input_dim: int

    def __call__(self, x: Float[Array, "N d"]) -> Float[Array, "N 2D"]:
        ...
```

**Estimated lines:** ~40

---

## 4  Model Patterns (User-Level, No Classes Needed)

These patterns compose `pyrox.nn`'s RFF layers with NumPyro inference.
They should be documented as examples, not implemented as classes.

### 4.1  SSGP (Sparse Spectrum Gaussian Process)

Standard Bayesian linear regression in RFF space:

```python
def ssgp_model(X, y=None):
    rff = RandomFourierFeatures(n_features=256, kernel_type="rbf", key=key)
    phi = rff(X)  # (N, 512)

    # Bayesian linear regression in feature space
    w = numpyro.sample("w", dist.Normal(jnp.zeros(512), 1.0))
    f = phi @ w

    noise = numpyro.sample("noise", dist.HalfNormal(1.0))
    numpyro.sample("y", dist.Normal(f, noise), obs=y)
```

### 4.2  Deep GP via RFF Stacking

Stack RFF layers for a deep kernel machine:

```python
def deep_rff_model(X, y=None):
    rff1 = RandomFourierFeatures(n_features=100, key=k1)
    rff2 = RandomFourierFeatures(n_features=50, key=k2)

    h1 = jax.nn.tanh(rff1(X))
    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros(200), 1.0))
    h2 = jax.nn.tanh(rff2((h1 @ w1)[:, None]))
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros(100), 1.0))
    f = h2 @ w2

    numpyro.sample("y", dist.Normal(f, 0.1), obs=y)
```

### 4.3  Heteroscedastic Regression via Dual RFF

Two-headed RFF: one for mean, one for noise:

```python
def heteroscedastic_rff_model(X, y=None):
    rff = RandomFourierFeatures(n_features=128, key=key)
    phi = rff(X)  # (N, 256)

    w_mean = numpyro.sample("w_mean", dist.Normal(jnp.zeros(256), 1.0))
    w_noise = numpyro.sample("w_noise", dist.Normal(jnp.zeros(256), 1.0))

    f_mean = phi @ w_mean
    f_noise = jax.nn.softplus(phi @ w_noise)

    numpyro.sample("y", dist.Normal(f_mean, f_noise), obs=y)
```

---

## 5  HSGP: Hilbert Space Features (Deterministic Eigenfunctions)

**Source:** Solin & Särkkä (2020), Riutort-Mayol et al. (2023),
[numpyro.contrib.hsgp](https://github.com/pyro-ppl/numpyro/tree/master/numpyro/contrib/hsgp)

**Priority:** High

HSGP replaces random Fourier features with deterministic Laplacian
eigenfunctions on a bounded domain. Same pattern — feature map +
Bayesian linear regression — but with ordered, deterministic bases
instead of random frequencies. NumPyro already provides a full
implementation in `numpyro.contrib.hsgp`.

### 5.1  Why HSGP alongside RFF

| Property | RFF | HSGP |
|---|---|---|
| Feature map | Random: $\omega \sim S(\omega)$ | Deterministic: eigenfunctions of $\Delta$ |
| Domain | $\mathbb{R}^d$ (unbounded) | $[-L, L]^d$ (bounded) or $S^{d-1}$ (sphere) |
| Kernel approx | Stochastic (seed-dependent) | Deterministic (truncated Mercer expansion) |
| Hyperparameter gradients | Require resampling or REINFORCE | Analytical (spectral density is differentiable) |
| Basis ordering | Unordered (all frequencies equal) | Ordered by eigenvalue (low-freq first) |
| Scaling in $d$ | $O(D)$ features regardless of $d$ | $O(\prod m_d)$ — curse of dimensionality |
| Sweet spot | High-$d$, unbounded inputs | Low-$d$ ($\leq 3$), bounded/periodic inputs |

Both produce $f(x) \approx \phi(x)^T \beta$, $\beta \sim \mathcal{N}(0, I)$.
The NumPyro model code is identical — only the feature layer changes.

### 5.2  Math

**Eigenfunctions** on $[-L_1, L_1] \times \cdots \times [-L_D, L_D]$ (Dirichlet Laplacian):

$$\phi_j(x) = \prod_{d=1}^{D} \frac{1}{\sqrt{L_d}} \sin\!\left(\frac{j_d \pi (x_d + L_d)}{2 L_d}\right)$$

**Eigenvalues:**

$$\lambda_j = \sum_{d=1}^{D} \left(\frac{j_d \pi}{2 L_d}\right)^2$$

**GP approximation** (non-centered parameterization, Eq. 8 Riutort-Mayol et al.):

$$f(x) \approx \Phi(x) \cdot \text{diag}\!\bigl(\sqrt{S(\sqrt{\lambda_j})}\bigr) \cdot \beta, \quad \beta \sim \mathcal{N}(0, I_M)$$

where $S(\omega)$ is the spectral density of the kernel and $M = \prod m_d$.

**Supported kernels** (via spectral density):

| Kernel | $S(\omega)$ | Parameters |
|---|---|---|
| Squared exponential | $\alpha^2 (2\pi)^{D/2} \prod \ell_d \exp(-\tfrac{1}{2}\sum \ell_d^2 \omega_d^2)$ | $\alpha$, $\ell$ (per-dim) |
| Matérn-$\nu$ | $\alpha^2 \cdot C_\nu \cdot (2\nu/\ell^2 + \|\omega\|^2)^{-(\nu+D/2)}$ | $\alpha$, $\ell$, $\nu$ |
| Rational quadratic | Bessel $K_\nu$ expression (requires TFP) | $\alpha$, $\ell$, $\alpha_{\text{mix}}$ |
| Periodic | Modified Bessel $I_j(1/\ell^2)$ coefficients | $\alpha$, $\ell$, $\omega_0$ |

**Periodic kernel** uses a separate basis (cos + sin instead of sin-only):

$$\phi_j^c(x) = \cos(j \omega_0 x), \quad \phi_j^s(x) = \sin(j \omega_0 x)$$

### 5.3  Complexity

| Operation | Cost |
|---|---|
| Eigenfunction evaluation $\Phi(X)$ | $O(N \cdot D \cdot M)$ where $M = \prod m_d$ |
| Spectral density vector $\sqrt{S(\lambda)}$ | $O(M)$ |
| Forward pass $\Phi \cdot (\sqrt{S} \circ \beta)$ | $O(NM)$ |
| Total training (per MCMC/SVI step) | $O(NM)$ — no matrix inversion |

### 5.4  API

Three layers, mirroring the numpyro.contrib.hsgp structure but as
Equinox modules that return features (decoupled from NumPyro sampling):

```python
class HilbertSpaceFeatures(eqx.Module):
    """Laplacian eigenfunction features on [-L, L]^D.

    Deterministic, ordered basis functions for GP approximation.
    Drop-in replacement for RandomFourierFeatures — same output
    shape, same BLR pattern, but deterministic and hyperparameter-
    differentiable.
    """
    ell: Float[Array, " D"]          # half-lengths L_d of bounding box
    m: tuple[int, ...]               # basis count per dimension
    # precomputed:
    sqrt_eigenvalues: Float[Array, "D M"]  # sqrt(lambda_j)
    n_features: int                  # M = prod(m)

    @staticmethod
    def from_domain(
        ell: float | list[float],
        m: int | list[int],
        dim: int = 1,
    ) -> "HilbertSpaceFeatures": ...

    def __call__(self, x: Float[Array, "N D"]) -> Float[Array, "N M"]:
        """Evaluate eigenfunctions phi_j(x) at input locations."""
        ...

    def spectral_density(
        self,
        kernel: str,                  # "se" | "matern" | "rq"
        alpha: float,
        length: float | Float[Array, " D"],
        **kwargs,                     # nu for matern, scale_mixture for rq
    ) -> Float[Array, " M"]:
        """Evaluate sqrt(S(sqrt(lambda_j))) for the given kernel."""
        ...


class PeriodicFeatures(eqx.Module):
    """Cos + sin basis for periodic kernel HSGP (1-D only)."""
    w0: float                         # fundamental frequency 2*pi/period
    m: int                            # number of basis functions

    def __call__(self, x: Float[Array, "N 1"]) -> tuple[Float[Array, "N m"], Float[Array, "N m"]]:
        """Returns (cosines, sines) basis matrices."""
        ...

    def spectral_density(
        self, alpha: float, length: float,
    ) -> Float[Array, " m"]:
        """Modified Bessel I_j coefficients."""
        ...


class SphericalHarmonicFeatures(eqx.Module):
    """Spherical harmonic basis for GP on S^{d-1}.

    Same pattern as HilbertSpaceFeatures but with Laplace-Beltrami
    eigenfunctions on the sphere instead of Dirichlet eigenfunctions
    on a box.
    """
    max_degree: int                   # L -> M = (L+1)^2 features

    def __call__(
        self,
        x: Float[Array, "N 2"],       # (colatitude, longitude)
    ) -> Float[Array, "N M"]: ...

    def spectral_density(
        self,
        kernel: str,
        alpha: float,
        length: float,
        **kwargs,
    ) -> Float[Array, " M"]: ...
```

**Usage in a NumPyro model** (same pattern as RFF):

```python
def model(x, y=None):
    alpha = numpyro.sample("alpha", dist.LogNormal(0, 1))
    length = numpyro.sample("length", dist.LogNormal(0, 1))
    noise = numpyro.sample("noise", dist.HalfNormal(1))

    # Feature layer (deterministic, precomputed at init)
    hsf = HilbertSpaceFeatures.from_domain(ell=5.0, m=20, dim=1)
    phi = hsf(x)                                  # (N, 20)
    spd = hsf.spectral_density("se", alpha, length)  # (20,)

    # BLR in feature space (identical to RFF pattern)
    beta = numpyro.sample("beta", dist.Normal(0, 1).expand([hsf.n_features]))
    f = phi @ (spd * beta)

    numpyro.sample("y", dist.Normal(f, noise), obs=y)
```

**Compatibility with numpyro.contrib.hsgp:** The primitives
(`eigenfunctions`, `sqrt_eigenvalues`, `diag_spectral_density_*`)
can be imported directly from `numpyro.contrib.hsgp.laplacian` and
`numpyro.contrib.hsgp.spectral_densities`. The pyrox layers wrap
these into Equinox modules with a unified `__call__` + `spectral_density`
interface, but the underlying computation is identical.

### 5.5  Relationship to pyrox.gp inducing features

| | pyrox.nn HSGP | pyrox.gp VISH/VFF |
|---|---|---|
| **What it is** | Feature map layer → BLR | Inducing features in SVGP |
| **Prior** | Truncated (approximate) | Full (exact) |
| **Posterior** | Exact under truncated prior | Variational approximation |
| **Inference** | MCMC or SVI on weights $\beta$ | ELBO optimization on $m, S$ |
| **Shared code** | Eigenfunctions, spectral densities | Same eigenfunctions, same spectral densities |
| **When to use** | $N \lesssim 10^5$, want full posterior | $N \gtrsim 10^5$, mini-batching needed |

Both share the basis function primitives in `pyrox.gp._src.basis` and the
spectral density functions in `pyrox._core.spectral`.

---

## 6  Relationship to pyrox.gp Pathwise Sampling

`pyrox.gp` uses RFF to represent prior function draws for pathwise posterior
sampling (Matheron's rule). The key distinction:

| | pyrox.nn RFF | pyrox.gp RFF |
|---|---|---|
| **Role** | Feature map for Bayesian linear regression | Prior function draw for pathwise conditioning |
| **Inference** | On the weights $w$ (BLR in feature space) | On the GP (true kernel, not approximated) |
| **Accuracy** | GP is approximate (D features) | GP posterior is exact; sample representation is approximate |
| **Entry point** | `RandomFourierFeatures` layer | `rff_prior_draw` primitive |
| **Shared code** | Spectral density samplers could be shared | Same spectral density logic |

**Resolved sharing question:** Since `pyrox.nn` and `pyrox.gp` are now
submodules of the same `pyrox` package, the spectral density sampling logic
(Gap 1) can live in a shared internal location and be imported by both
submodules. The recommended location is `pyrox._core.spectral` -- both
`pyrox.nn` and `pyrox.gp` import from this shared internal module. This
eliminates the previous cross-package dependency question entirely.

---

## 7  Integration Plan

| Component | File | Priority | Dependencies |
|---|---|---|---|
| Extended spectral densities | `_core/spectral.py` | High | None |
| `HilbertSpaceFeatures` | `nn/hilbert_space.py` | **High** | `numpyro.contrib.hsgp` |
| `PeriodicFeatures` | `nn/hilbert_space.py` | **High** | `numpyro.contrib.hsgp` |
| `SphericalHarmonicFeatures` (nn layer) | `nn/hilbert_space.py` | **High** | `pyrox.gp._src.basis` |
| `VariationalFourierFeatures` | `nn/random_feature.py` | Medium | `PyroxModule` |
| `OrthogonalRandomFeatures` | `nn/random_feature.py` | Low | None |
| SSGP example | `examples/nn/models.md` | Medium | `RandomFourierFeatures` |
| HSGP example | `examples/nn/models.md` | **High** | `HilbertSpaceFeatures` |
| Deep RFF example | `examples/nn/models.md` | Medium | `RandomFourierFeatures` |

---

## 8  References

1. Rahimi, A. & Recht, B. (2007). *Random Features for Large-Scale
   Kernel Machines.* NeurIPS.

2. Lazaro-Gredilla, M., Quinonero-Candela, J., Rasmussen, C. E., &
   Figueiras-Vidal, A. R. (2010). *Sparse Spectrum Gaussian Process
   Regression.* JMLR 11:1865-1881.

3. Gal, Y. & Turner, R. (2015). *Improving the Gaussian Process
   Sparse Spectrum Approximation by Representing Uncertainty in
   Frequency Domain.* ICML Workshop.

4. Yu, F., et al. (2016). *Orthogonal Random Features.* NeurIPS.

5. Mutny, M. & Krause, A. (2018). *Efficient High Dimensional
   Bayesian Optimization with Additivity and Quadrature Fourier
   Features.* NeurIPS.

6. Wilson, J., et al. (2020). *Efficiently Sampling Functions from
   Gaussian Process Posteriors.* ICML. (Pathwise sampling context)

7. Solin, A. & Särkkä, S. (2020). *Hilbert Space Methods for
   Reduced-Rank Gaussian Process Regression.* Statistics and Computing.

8. Riutort-Mayol, G., et al. (2023). *Practical Hilbert Space
   Approximate Bayesian Gaussian Processes for Probabilistic
   Programming.* Statistics and Computing.

9. Dutordoir, V., Durrande, N., & Hensman, J. (2020). *Sparse
   Gaussian Processes with Spherical Harmonic Features.* ICML.
