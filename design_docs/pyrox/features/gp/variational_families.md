---
status: draft
version: 0.1.0
---

# pyrox x Variational Families: Gap Analysis

**Subject:** Structured variational posterior families (guides) sourced from GPJax
(8 families), GPyTorch (Natural, CIQ, LMC), and BayesNewton (site-based).

**Date:** 2026-04-02

---

## 1  Summary

Variational families (guides) live in `pyrox.gp`. gaussx provides the linear
algebra primitives that guides use internally (whitening, Cholesky, Woodbury,
Kalman), but the guide protocol and concrete implementations belong here.

The `Guide` protocol is already defined in pyrox.gp's architecture. This doc
specifies concrete implementations.

---

## 2  Common Mathematical Framework

The Sparse Variational GP (SVGP) framework (Titsias 2009, Hensman et al. 2013)
approximates the intractable posterior $p(f \mid y)$ through a set of $M$
inducing variables $\mathbf{u} = f(\mathbf{Z})$ at inducing locations $\mathbf{Z}$:

$$q(\mathbf{f}) = \int p(\mathbf{f} \mid \mathbf{u})\, q(\mathbf{u})\, d\mathbf{u}$$

where $q(\mathbf{u}) = \mathcal{N}(\mathbf{m}, \mathbf{S})$ is the variational
posterior over the inducing variables. The ELBO is:

$$\mathcal{L} = \sum_{n=1}^{N} \mathbb{E}_{q(f_n)}\!\bigl[\log p(y_n \mid f_n)\bigr] - \mathrm{KL}\!\bigl[q(\mathbf{u}) \,\|\, p(\mathbf{u})\bigr]$$

**Predictive distribution.** For a test point $x_*$:

$$\mu_* = \mathbf{k}_*^\top \mathbf{K}_{uu}^{-1} \mathbf{m}, \qquad \sigma_*^2 = k_{**} - \mathbf{k}_*^\top \mathbf{K}_{uu}^{-1}(\mathbf{K}_{uu} - \mathbf{S})\mathbf{K}_{uu}^{-1}\mathbf{k}_*$$

where $\mathbf{k}_* = k(\mathbf{Z}, x_*)$ and $k_{**} = k(x_*, x_*)$.

**KL divergence.** The standard MVN KL between $q(\mathbf{u}) = \mathcal{N}(\mathbf{m}, \mathbf{S})$ and the prior $p(\mathbf{u}) = \mathcal{N}(\mathbf{0}, \mathbf{K}_{uu})$:

$$\mathrm{KL} = \tfrac{1}{2}\bigl[\mathrm{tr}(\mathbf{K}_{uu}^{-1}\mathbf{S}) + \mathbf{m}^\top \mathbf{K}_{uu}^{-1}\mathbf{m} - M + \log|\mathbf{K}_{uu}| - \log|\mathbf{S}|\bigr]$$

**Role of guides.** Each guide family provides a different parameterization of $(\mathbf{m}, \mathbf{S})$
and corresponding methods to (a) compute or bound the KL term, (b) sample
$q(f_n)$ for the expected log-likelihood, and (c) predict at test points.
The choice of guide affects optimization geometry, memory cost, and
compatibility with different inference strategies.

---

## 3  Gap Catalog

### Gap 1: FullRankGuide

**Parameterization:** Dense Cholesky factor. Optimize $\mathbf{m} \in \mathbb{R}^M$ and lower-triangular $\mathbf{L} \in \mathbb{R}^{M \times M}$, with $\mathbf{S} = \mathbf{L}\mathbf{L}^\top$.

**Math:**

$$q(\mathbf{u}) = \mathcal{N}(\mathbf{m},\, \mathbf{L}\mathbf{L}^\top)$$

KL divergence uses the standard formula with $\log|\mathbf{S}| = 2\sum_i \log L_{ii}$. Sampling: $\mathbf{u} = \mathbf{m} + \mathbf{L}\boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

Predictive mean and variance follow from the general SVGP formulas in Section 2.

**Complexity:** $O(M^2)$ parameters, $O(M^3)$ for KL (Cholesky solve), $O(NM^2)$ for prediction.

```python
class FullRankGuide(eqx.Module):
    """Full-rank Gaussian variational posterior q(u) = N(m, LL^T)."""
    mean: Float[Array, " M"]                  # m
    scale_tril: Float[Array, "M M"]           # L (lower triangular)

    def sample(
        self, key: PRNGKeyArray, n_samples: int = 1
    ) -> Float[Array, "n_samples M"]:
        """Sample u ~ q(u) via reparameterization."""
        ...

    def kl_divergence(
        self, prior_cov: Float[Array, "M M"]
    ) -> Float[Array, ""]:
        """KL[q(u) || p(u)] with p(u) = N(0, K_uu)."""
        ...

    def predict(
        self,
        Kuf: Float[Array, "M N"],
        Kuu: Float[Array, "M M"],
        Kff_diag: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Predictive mean and variance."""
        ...
```

**Ref:** Hensman, Matthews & Ghahramani (2015) *Scalable Variational Gaussian Process Classification.* AISTATS.

---

### Gap 2: MeanFieldGuide

**Parameterization:** Diagonal covariance. Optimize $\mathbf{m} \in \mathbb{R}^M$ and $\boldsymbol{\sigma} \in \mathbb{R}^M_{>0}$, with $\mathbf{S} = \mathrm{diag}(\boldsymbol{\sigma}^2)$.

**Math:**

$$q(\mathbf{u}) = \prod_{j=1}^{M} \mathcal{N}(m_j,\, \sigma_j^2)$$

KL decomposes into per-dimension terms:

$$\mathrm{KL} = \tfrac{1}{2}\sum_{j=1}^{M}\!\left[\frac{\sigma_j^2}{[K_{uu}^{-1}]_{jj}^{-1}} + \frac{m_j^2}{[K_{uu}]_{jj}} - 1 - \log\sigma_j^2 + \log[K_{uu}]_{jj}\right]$$

When $\mathbf{K}_{uu}$ is not diagonal, the full KL from Section 2 is used with $\mathbf{S} = \mathrm{diag}(\boldsymbol{\sigma}^2)$.

Sampling: $u_j = m_j + \sigma_j \epsilon_j$, $\epsilon_j \sim \mathcal{N}(0, 1)$.

**Complexity:** $O(M)$ parameters, $O(M^3)$ for KL (still need $\mathbf{K}_{uu}^{-1}$, but $\mathrm{tr}$ and $\log|\mathbf{S}|$ are $O(M)$), $O(NM)$ for prediction.

```python
class MeanFieldGuide(eqx.Module):
    """Mean-field (diagonal) variational posterior q(u) = prod N(m_j, sigma_j^2)."""
    mean: Float[Array, " M"]                  # m
    log_scale: Float[Array, " M"]             # log(sigma), unconstrained

    @property
    def scale(self) -> Float[Array, " M"]:
        """sigma = exp(log_scale)."""
        ...

    def sample(
        self, key: PRNGKeyArray, n_samples: int = 1
    ) -> Float[Array, "n_samples M"]:
        ...

    def kl_divergence(
        self, prior_cov: Float[Array, "M M"]
    ) -> Float[Array, ""]:
        ...

    def predict(
        self,
        Kuf: Float[Array, "M N"],
        Kuu: Float[Array, "M M"],
        Kff_diag: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        ...
```

**Ref:** Hensman, Matthews & Ghahramani (2015) *Scalable Variational Gaussian Process Classification.* AISTATS.

---

### Gap 3: WhitenedGuide

**Parameterization:** Whitened space. Instead of optimizing $(m, S)$ directly,
reparameterize $\mathbf{u} = \mathbf{L}_{uu}\mathbf{v} + \boldsymbol{\mu}_{uu}$
where $\mathbf{K}_{uu} = \mathbf{L}_{uu}\mathbf{L}_{uu}^\top$, and place
the variational distribution on $\mathbf{v}$:

$$q(\mathbf{v}) = \mathcal{N}(\tilde{\mathbf{m}},\, \tilde{\mathbf{S}})$$

**Math:**

The whitened KL simplifies because the prior on $\mathbf{v}$ is $\mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$\mathrm{KL}\bigl[q(\mathbf{v}) \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I})\bigr] = \tfrac{1}{2}\bigl[\mathrm{tr}(\tilde{\mathbf{S}}) + \|\tilde{\mathbf{m}}\|^2 - M - \log|\tilde{\mathbf{S}}|\bigr]$$

Prediction maps back through $\mathbf{L}_{uu}$:

$$\mu_* = \mathbf{k}_*^\top \mathbf{L}_{uu}^{-\top} \tilde{\mathbf{m}}, \qquad \sigma_*^2 = k_{**} - \mathbf{k}_*^\top \mathbf{K}_{uu}^{-1} \mathbf{k}_* + \mathbf{k}_*^\top \mathbf{L}_{uu}^{-\top} \tilde{\mathbf{S}} \mathbf{L}_{uu}^{-1} \mathbf{k}_*$$

**Advantage:** Better optimization conditioning. The KL no longer depends on
$\mathbf{K}_{uu}$, decoupling kernel hyperparameter gradients from variational
parameter gradients.

**Complexity:** Same as the wrapped guide ($O(M^2)$ for full-rank, $O(M)$ for mean-field), plus one-time $O(M^3)$ Cholesky of $\mathbf{K}_{uu}$.

```python
class WhitenedGuide(eqx.Module):
    """Whitened variational posterior: q(v) where u = L_uu v."""
    inner_guide: FullRankGuide | MeanFieldGuide   # guide on whitened v
    whiten: bool = True                           # toggle whitening

    def sample(
        self, key: PRNGKeyArray, Kuu: Float[Array, "M M"], n_samples: int = 1
    ) -> Float[Array, "n_samples M"]:
        """Sample u = L_uu @ v_sample."""
        ...

    def kl_divergence(self) -> Float[Array, ""]:
        """KL[q(v) || N(0, I)] — no dependence on K_uu."""
        ...

    def predict(
        self,
        Kuf: Float[Array, "M N"],
        Kuu: Float[Array, "M M"],
        Kff_diag: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        ...
```

**Ref:** Matthews et al. (2017) *GPflow: A Gaussian Process Library using TensorFlow.* JMLR. (Section on whitened parameterization.)

---

### Gap 4: NaturalGuide

**Parameterization:** Natural (information) parameters of the Gaussian:

$$\boldsymbol{\eta}_1 = \mathbf{S}^{-1}\mathbf{m}, \qquad \boldsymbol{\eta}_2 = -\tfrac{1}{2}\mathbf{S}^{-1}$$

**Math:**

Recover moment parameters:

$$\mathbf{S} = (-2\boldsymbol{\eta}_2)^{-1}, \qquad \mathbf{m} = \mathbf{S}\,\boldsymbol{\eta}_1$$

KL uses the standard formula from Section 2, reconstructing $(\mathbf{m}, \mathbf{S})$ from $(\boldsymbol{\eta}_1, \boldsymbol{\eta}_2)$.

**Key advantage:** Natural gradient updates reduce to vanilla SGD on natural parameters:

$$\boldsymbol{\eta}_1^{(t+1)} = (1 - \rho)\,\boldsymbol{\eta}_1^{(t)} + \rho\,\hat{\boldsymbol{\eta}}_1, \qquad \boldsymbol{\eta}_2^{(t+1)} = (1 - \rho)\,\boldsymbol{\eta}_2^{(t)} + \rho\,\hat{\boldsymbol{\eta}}_2$$

where $\hat{\boldsymbol{\eta}}$ are the stochastic natural gradient estimates and $\rho$ is the learning rate. This avoids computing the Fisher information matrix.

**Complexity:** $O(M^2)$ parameters (same as full-rank), $O(M^3)$ for recovering $\mathbf{S}$ and KL computation. Natural gradient step itself is $O(M^2)$.

```python
class NaturalGuide(eqx.Module):
    """Natural-parameter variational posterior q(u) in exponential family form."""
    nat1: Float[Array, " M"]                  # eta_1 = S^{-1} m
    nat2: Float[Array, "M M"]                 # eta_2 = -0.5 S^{-1}

    @property
    def mean(self) -> Float[Array, " M"]:
        """m = S @ eta_1."""
        ...

    @property
    def covariance(self) -> Float[Array, "M M"]:
        """S = (-2 eta_2)^{-1}."""
        ...

    def sample(
        self, key: PRNGKeyArray, n_samples: int = 1
    ) -> Float[Array, "n_samples M"]:
        ...

    def kl_divergence(
        self, prior_cov: Float[Array, "M M"]
    ) -> Float[Array, ""]:
        ...

    def natural_update(
        self,
        nat1_target: Float[Array, " M"],
        nat2_target: Float[Array, "M M"],
        learning_rate: float,
    ) -> "NaturalGuide":
        """Damped natural gradient step: eta <- (1-rho)*eta + rho*target."""
        ...
```

**Ref:** Hensman, Fusi & Lawrence (2012) *Gaussian Processes for Big Data.* UAI. Salimbeni, Eleftheriadis & Hensman (2018) *Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models.* AISTATS.

---

### Gap 5: DeltaGuide

**Parameterization:** Point estimate (MAP). $\mathbf{S} = \mathbf{0}$ (zero covariance), so $q(\mathbf{u}) = \delta(\mathbf{u} - \mathbf{m})$.

**Math:**

The ELBO collapses to the log-likelihood at the MAP estimate (no expectation needed):

$$\mathcal{L}_{\text{MAP}} = \sum_{n=1}^{N} \log p(y_n \mid f_n(\mathbf{m})) - \tfrac{1}{2}\mathbf{m}^\top \mathbf{K}_{uu}^{-1}\mathbf{m} - \tfrac{1}{2}\log|\mathbf{K}_{uu}| - \tfrac{M}{2}\log(2\pi)$$

The KL term reduces to the log-prior (up to a constant):

$$\mathrm{KL}\bigl[\delta(\mathbf{u} - \mathbf{m}) \,\|\, p(\mathbf{u})\bigr] = -\log p(\mathbf{m}) + \text{const}$$

Prediction gives a point estimate with no posterior variance from $q$:

$$\mu_* = \mathbf{k}_*^\top \mathbf{K}_{uu}^{-1}\mathbf{m}, \qquad \sigma_*^2 = k_{**} - \mathbf{k}_*^\top \mathbf{K}_{uu}^{-1}\mathbf{k}_*$$

**Complexity:** $O(M)$ parameters, $O(M^3)$ for solve/logdet of $\mathbf{K}_{uu}$, $O(NM)$ for prediction.

```python
class DeltaGuide(eqx.Module):
    """Point-estimate (MAP) guide: q(u) = delta(u - m)."""
    mean: Float[Array, " M"]                  # m

    def __call__(
        self, Kuu: Float[Array, "M M"]
    ) -> Float[Array, " M"]:
        """Return the MAP inducing values."""
        ...

    def log_prior(
        self, prior_cov: Float[Array, "M M"]
    ) -> Float[Array, ""]:
        """log p(m) = log N(m; 0, K_uu)."""
        ...

    def predict(
        self,
        Kuf: Float[Array, "M N"],
        Kuu: Float[Array, "M M"],
        Kff_diag: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        """Predictive mean + prior variance only (no posterior uncertainty)."""
        ...
```

**Ref:** Titsias (2009) *Variational Learning of Inducing Variables in Sparse Gaussian Processes.* AISTATS.

---

### Gap 6: SiteGuide

**Parameterization:** Per-observation natural parameter sites. Each data point
$y_n$ contributes a site $t_n(\mathbf{u}) \propto \exp\!\bigl(\boldsymbol{\eta}_{1,n}^\top \mathbf{u} + \mathbf{u}^\top \boldsymbol{\eta}_{2,n}\, \mathbf{u}\bigr)$ in the inducing space.

**Math:**

The posterior is the product of the prior and all sites:

$$q(\mathbf{u}) \propto p(\mathbf{u}) \prod_{n=1}^{N} t_n(\mathbf{u})$$

Aggregating natural parameters:

$$\boldsymbol{\eta}_1^{\text{post}} = \sum_{n=1}^{N} \boldsymbol{\eta}_{1,n}, \qquad \boldsymbol{\eta}_2^{\text{post}} = -\tfrac{1}{2}\mathbf{K}_{uu}^{-1} + \sum_{n=1}^{N} \boldsymbol{\eta}_{2,n}$$

Sites are updated via conjugate computation variational inference (CVI) or
power EP updates:

$$\boldsymbol{\eta}_{n}^{\text{new}} = (1 - \rho)\,\boldsymbol{\eta}_{n}^{\text{old}} + \rho\,\hat{\boldsymbol{\eta}}_{n}$$

where $\hat{\boldsymbol{\eta}}_n$ comes from the cavity distribution and the tilted likelihood.

**Complexity:** $O(NM)$ site parameters ($\boldsymbol{\eta}_{1,n} \in \mathbb{R}^M$, $\boldsymbol{\eta}_{2,n} \in \mathbb{R}^{M \times M}$ or rank-1), $O(M^3)$ for posterior reconstruction.

```python
class SiteGuide(eqx.Module):
    """Per-observation site-based guide for EP / PL / CVI inference."""
    site_nat1: Float[Array, "N M"]            # per-observation eta_1
    site_nat2: Float[Array, "N M M"] | Float[Array, "N M"]  # full or rank-1

    def posterior_natural_params(
        self, prior_precision: Float[Array, "M M"]
    ) -> tuple[Float[Array, " M"], Float[Array, "M M"]]:
        """Aggregate sites + prior into posterior natural params."""
        ...

    def cavity(
        self, idx: Int[Array, " B"], prior_precision: Float[Array, "M M"]
    ) -> tuple[Float[Array, " M"], Float[Array, "M M"]]:
        """Cavity distribution: posterior with site idx removed."""
        ...

    def update_sites(
        self,
        idx: Int[Array, " B"],
        new_nat1: Float[Array, "B M"],
        new_nat2: Float[Array, "B M M"],
        learning_rate: float,
    ) -> "SiteGuide":
        """Damped site update."""
        ...

    def kl_divergence(
        self, prior_cov: Float[Array, "M M"]
    ) -> Float[Array, ""]:
        """KL of the aggregated posterior vs the prior."""
        ...
```

**Ref:** Khan & Lin (2017) *Conjugate-Computation Variational Inference.* AISTATS. Wilkinson et al. (2021) *BayesNewton: A Practical Framework for Approximate Bayesian Inference.* JMLR.

---

### Gap 7: KalmanGuide

**Parameterization:** Markov-structured variational distribution with banded
precision. For temporal GPs where the prior has Markov structure (state-space
form), the posterior precision is block-tridiagonal:

$$q(\mathbf{u}) = \mathcal{N}(\mathbf{m},\, \mathbf{S}), \qquad \mathbf{S}^{-1} \text{ is block-tridiagonal}$$

**Math:**

The state-space GP prior gives a Markov chain:

$$\mathbf{u}_t = \mathbf{A}_t \mathbf{u}_{t-1} + \boldsymbol{\epsilon}_t, \qquad \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_t)$$

The variational posterior is computed by the Kalman smoother:

1. **Forward (filter):** $p(\mathbf{u}_t \mid y_{1:t})$ via Kalman filter updates.
2. **Backward (smoother):** $q(\mathbf{u}_t) = p(\mathbf{u}_t \mid y_{1:T})$ via Rauch-Tung-Striebel smoother.

KL divergence decomposes over the Markov chain:

$$\mathrm{KL} = \mathrm{KL}\bigl[q(\mathbf{u}_1) \,\|\, p(\mathbf{u}_1)\bigr] + \sum_{t=2}^{T} \mathbb{E}_{q(\mathbf{u}_{t-1})}\!\bigl[\mathrm{KL}\bigl[q(\mathbf{u}_t \mid \mathbf{u}_{t-1}) \,\|\, p(\mathbf{u}_t \mid \mathbf{u}_{t-1})\bigr]\bigr]$$

**Complexity:** $O(T d^3)$ where $d$ is the state dimension (typically small, e.g. $d = 2\nu$ for Matern-$\nu$), compared to $O(T^3 d^3)$ for a naive dense guide. Linear in $T$.

```python
class KalmanGuide(eqx.Module):
    """Markov-structured guide using Kalman filtering/smoothing."""
    filter_means: Float[Array, "T D"]         # filtered means
    filter_covs: Float[Array, "T D D"]        # filtered covariances
    smoother_means: Float[Array, "T D"]       # smoothed means
    smoother_covs: Float[Array, "T D D"]      # smoothed covariances

    def forward_filter(
        self,
        observations: Float[Array, "T P"],
        transition_matrices: Float[Array, "T D D"],
        transition_noise: Float[Array, "T D D"],
        emission_matrices: Float[Array, "T P D"],
        emission_noise: Float[Array, "T P P"],
    ) -> "KalmanGuide":
        """Run Kalman filter forward pass."""
        ...

    def backward_smooth(self) -> "KalmanGuide":
        """Run RTS backward smoother."""
        ...

    def kl_divergence(
        self,
        transition_matrices: Float[Array, "T D D"],
        transition_noise: Float[Array, "T D D"],
    ) -> Float[Array, ""]:
        """KL via Markov chain decomposition."""
        ...

    def predict(
        self, t: Int[Array, " N"]
    ) -> tuple[Float[Array, "N D"], Float[Array, "N D D"]]:
        """Marginal posterior at times t."""
        ...
```

**Ref:** Chang, Wilkinson & Solin (2020) *Fast Variational Learning in State-Space Gaussian Process Models.* MLSP. Wilkinson et al. (2021) *BayesNewton.* JMLR.

---

### Gap 8: CIQGuide

**Parameterization:** Same as FullRankGuide ($\mathbf{m}$, $\mathbf{L}$), but
uses contour integral quadrature (CIQ) to approximate the $\log|\mathbf{S}|$
and $\mathrm{tr}(\mathbf{K}_{uu}^{-1}\mathbf{S})$ terms in the KL without
explicit Cholesky decompositions.

**Math:**

CIQ approximates the log-determinant via:

$$\log|\mathbf{A}| \approx N \sum_{q=1}^{Q} w_q \, \mathbf{e}_1^\top (\mathbf{T} + t_q \mathbf{I})^{-1} \mathbf{e}_1$$

where $\mathbf{T}$ is the tridiagonal matrix from Lanczos decomposition of $\mathbf{A}$,
and $(w_q, t_q)$ are the CIQ quadrature weights and nodes from a contour in the complex plane.

The trace term $\mathrm{tr}(\mathbf{K}_{uu}^{-1}\mathbf{S})$ is computed via
stochastic Lanczos quadrature (SLQ).

**Complexity:** $O(M^2 J)$ per ELBO evaluation where $J$ is the number of Lanczos iterations (typically $J \ll M$). Avoids $O(M^3)$ Cholesky, enabling $M \gg 10^3$.

```python
class CIQGuide(eqx.Module):
    """Contour integral quadrature guide for scalable KL computation."""
    mean: Float[Array, " M"]                  # m
    scale_tril: Float[Array, "M M"]           # L (never explicitly formed as dense)
    n_lanczos: int = 20                       # J — Lanczos iterations
    n_contour: int = 15                       # Q — quadrature nodes

    def kl_divergence(
        self, prior_cov: Float[Array, "M M"]
    ) -> Float[Array, ""]:
        """Approximate KL via CIQ (log-det) + SLQ (trace)."""
        ...

    def sample(
        self, key: PRNGKeyArray, n_samples: int = 1
    ) -> Float[Array, "n_samples M"]:
        ...

    def predict(
        self,
        Kuf: Float[Array, "M N"],
        Kuu: Float[Array, "M M"],
        Kff_diag: Float[Array, " N"],
    ) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
        ...
```

**Ref:** Pleiss, Jankowiak, Eriksson, Damle & Gardner (2020) *Fast Matrix Square Roots with Applications to Gaussian Processes and Bayesian Optimization.* NeurIPS.

---

### Gap 9: LMCGuide

**Parameterization:** Multi-output structured variational distribution based on
the Linear Model of Coregionalization (LMC). For $P$ outputs with shared
inducing locations, the variational distribution has Kronecker structure:

$$q(\mathbf{u}) = \mathcal{N}(\mathbf{m},\, \mathbf{S}), \qquad \mathbf{S} = \mathbf{B} \otimes \mathbf{S}_0$$

where $\mathbf{B} \in \mathbb{R}^{P \times P}$ captures inter-task covariance
and $\mathbf{S}_0 \in \mathbb{R}^{M \times M}$ is the spatial covariance.

**Math:**

The mean is reshaped as $\mathbf{M} \in \mathbb{R}^{M \times P}$ (one column per output). KL exploits Kronecker structure:

$$\mathrm{KL} = \tfrac{1}{2}\bigl[\mathrm{tr}\bigl((\mathbf{K}_{uu}^{-1} \otimes \mathbf{B}^{-1})(\mathbf{B} \otimes \mathbf{S}_0)\bigr) + \mathbf{m}^\top (\mathbf{K}_{uu}^{-1} \otimes \mathbf{B}^{-1})\mathbf{m} - MP + P\log|\mathbf{K}_{uu}| + M\log|\mathbf{B}| - P\log|\mathbf{S}_0| - M\log|\mathbf{B}|\bigr]$$

which simplifies via Kronecker identities to $O(M^3 + P^3)$ instead of $O((MP)^3)$.

**Complexity:** $O(MP)$ mean parameters, $O(M^2 + P^2)$ covariance parameters, $O(M^3 + P^3)$ KL computation.

```python
class LMCGuide(eqx.Module):
    """Multi-output LMC-structured variational posterior."""
    mean: Float[Array, "M P"]                 # M — per-output inducing means
    spatial_scale_tril: Float[Array, "M M"]   # L_0 s.t. S_0 = L_0 L_0^T
    task_scale_tril: Float[Array, "P P"]      # L_B s.t. B = L_B L_B^T
    n_latents: int                            # Q — number of latent GPs

    def sample(
        self, key: PRNGKeyArray, n_samples: int = 1
    ) -> Float[Array, "n_samples M P"]:
        """Sample via Kronecker reparameterization."""
        ...

    def kl_divergence(
        self,
        prior_cov: Float[Array, "M M"],
    ) -> Float[Array, ""]:
        """KL with Kronecker structure: O(M^3 + P^3)."""
        ...

    def predict(
        self,
        Kuf: Float[Array, "M N"],
        Kuu: Float[Array, "M M"],
        Kff_diag: Float[Array, " N"],
    ) -> tuple[Float[Array, "N P"], Float[Array, "N P"]]:
        """Multi-output predictive mean and variance."""
        ...
```

**Ref:** Bruinsma et al. (2020) *Scalable Exact Inference in Multi-Output Gaussian Processes.* ICML. Bonilla, Chai & Williams (2007) *Multi-task Gaussian Process Prediction.* NeurIPS.

---

## 4  Shared Infrastructure

All guide families share common gaussx primitives and pyrox infrastructure:

| Component | Source | Used By | Notes |
|---|---|---|---|
| Cholesky decomposition | `gaussx.cholesky` | FullRank, MeanField, Whitened, Natural, Delta, LMC | $O(M^3)$ factorization |
| Triangular solve | `gaussx.solve` | FullRank, Whitened, Natural, CIQ | $\mathbf{L}^{-1}\mathbf{x}$ operations |
| Log-determinant | `gaussx.logdet` | FullRank, MeanField, Natural, KL computation | Via Cholesky or CIQ |
| Diagonal operator | `gaussx.DiagonalLinearOperator` | MeanField, inter-domain features | $O(M)$ instead of $O(M^3)$ |
| Kronecker operator | `gaussx.Kronecker` | LMC | Kronecker-structured solves |
| Whitened SVGP predict | `gaussx.whitened_svgp_predict` | Whitened | Maps whitened params to predictions |
| Natural <-> expectation | `gaussx.GaussianExpFam` | Natural, Site | Parameter conversions |
| Damped natural update | `gaussx.damped_natural_update` | Natural, Site | $(1-\rho)\eta + \rho\hat\eta$ |
| CVI site updates | `gaussx.cvi_update_sites` | Site | Cavity + tilted likelihood |
| Kalman filter | `gaussx.kalman_filter` | Kalman | Forward pass |
| RTS smoother | `gaussx.rts_smoother` | Kalman | Backward pass |
| Stochastic trace | `matfree.stochastic_trace` | CIQ | SLQ for trace term |
| Lanczos tridiag | `matfree.lanczos_tridiag` | CIQ | For CIQ log-det |
| LOVE cache | `gaussx.love_cache` | CIQ (optional) | Fast predictive variance |

---

## 5  References

1. Titsias, M. (2009). *Variational Learning of Inducing Variables in Sparse Gaussian Processes.* AISTATS.
2. Hensman, J., Fusi, N., & Lawrence, N. D. (2012). *Gaussian Processes for Big Data.* UAI.
3. Hensman, J., Matthews, A. G., & Ghahramani, Z. (2015). *Scalable Variational Gaussian Process Classification.* AISTATS.
4. Matthews, A. G., et al. (2017). *GPflow: A Gaussian Process Library using TensorFlow.* JMLR.
5. Salimbeni, H., Eleftheriadis, S., & Hensman, J. (2018). *Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models.* AISTATS.
6. Khan, M. E. & Lin, W. (2017). *Conjugate-Computation Variational Inference: Converting Variational Inference in Non-Conjugate Models to Inferences in Conjugate Models.* AISTATS.
7. Pleiss, G., Jankowiak, M., Eriksson, D., Damle, A., & Gardner, J. (2020). *Fast Matrix Square Roots with Applications to Gaussian Processes and Bayesian Optimization.* NeurIPS.
8. Bruinsma, W., et al. (2020). *Scalable Exact Inference in Multi-Output Gaussian Processes.* ICML.
9. Bonilla, E. V., Chai, K. M., & Williams, C. K. I. (2007). *Multi-task Gaussian Process Prediction.* NeurIPS.
10. Chang, P. E., Wilkinson, W. J., & Solin, A. (2020). *Fast Variational Learning in State-Space Gaussian Process Models.* MLSP.
11. Wilkinson, W. J., et al. (2021). *BayesNewton: A Practical Framework for Approximate Bayesian Inference for Gaussian Process Models and Beyond.* JMLR.
