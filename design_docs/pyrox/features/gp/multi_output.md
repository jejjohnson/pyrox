---
status: draft
version: 0.1.0
---

# pyrox x Multi-Output GP: Gap Analysis

**Subject:** Multi-output GP infrastructure sourced from GPflow's multi-output
system, GPJax OILMM, and GPyTorch LMC/ICM.

**Date:** 2026-04-02

---

## 1  Scope

Multi-output GPs model vector-valued functions $f: \mathbb{R}^d \to \mathbb{R}^P$
by correlating $P$ output channels through a mixing mechanism. GPflow has the most
complete multi-output system but suffers from combinatorial dispatch complexity.

pyrox.gp's approach: separate the mixing matrix $W$ from the kernel and expose
it as a first-class `MixingOperator`, avoiding the dispatch explosion.

**In scope:** LMC, ICM, OILMM, multi-output inducing variables, heterotopic data, convolution processes.
**Out of scope:** Deep multi-output GPs, non-linear mixing, multi-fidelity GPs.

---

## 2  Common Mathematical Framework: Linear Mixing Model

All methods in this catalog share the **linear model of coregionalization** (LMC)
structure. We have $Q$ independent latent GPs $g_q \sim \mathcal{GP}(0, k_q)$ and
a mixing matrix $W \in \mathbb{R}^{P \times Q}$:

$$f_p(x) = \sum_{q=1}^{Q} w_{pq}\, g_q(x)$$

Stacking outputs into a vector $\mathbf{f}(x) = [f_1(x), \ldots, f_P(x)]^\top$:

$$\mathbf{f}(x) = W\, \mathbf{g}(x)$$

### 2.1  Kronecker Structure

Because the latent GPs are independent, the joint covariance over all outputs
at all inputs factorizes. For $N$ inputs and $P$ outputs, the $NP \times NP$
joint covariance is:

$$\text{Cov}[\text{vec}(\mathbf{F})] = \sum_{q=1}^{Q} (w_q w_q^\top) \otimes K_q$$

where $K_q = k_q(X, X) \in \mathbb{R}^{N \times N}$ and $w_q \in \mathbb{R}^P$ is the $q$-th column of $W$.

When all latent GPs share a single kernel $k$, this simplifies to a single
**Kronecker product**:

$$\text{Cov}[\text{vec}(\mathbf{F})] = B \otimes K, \qquad B = W W^\top \in \mathbb{R}^{P \times P}$$

This is the key structural property that gaussx exploits: `gaussx.Kronecker`
handles the full $NP \times NP$ covariance without materializing it, providing
$O(N^3 + P^3)$ solves and log-determinants instead of $O((NP)^3)$.

### 2.2  Observation Model

With i.i.d. Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I_{NP})$:

$$K_y = B \otimes K + \sigma^2 I_{NP}$$

When noise varies per output, $\Sigma_\epsilon = \text{diag}(\sigma_1^2, \ldots, \sigma_P^2) \otimes I_N$,
this remains Kronecker-structured and `gaussx.Kronecker` still applies.

---

## 3  Gap Catalog

### Gap 1: LMC -- Linear Model of Coregionalization

**Mixing structure:** Dense $W \in \mathbb{R}^{P \times Q}$ with $R$ rank-1 components per latent GP.

**Math:** Each output is a linear combination of $Q$ latent GPs, each with its own kernel:

$$k_{pp'}(x, x') = \sum_{q=1}^{Q} b_{pp'}^{(q)}\, k_q(x, x')$$

where $B^{(q)} = w_q w_q^\top$ is the coregionalization matrix for the $q$-th latent GP.
The full covariance is:

$$K_{\text{full}} = \sum_{q=1}^{Q} B^{(q)} \otimes K_q$$

**gaussx factorization:** Each term $B^{(q)} \otimes K_q$ is a `gaussx.Kronecker` operator.
The sum is a `gaussx.LowRankUpdate` or materialized sum. When $Q = 1$, it is a
single `gaussx.Kronecker`. When all $K_q$ are identical, the sum collapses to
$(\sum_q B^{(q)}) \otimes K = B \otimes K$.

**Inducing point structure:** Shared inducing points $Z \in \mathbb{R}^{M \times d}$ across
all latent GPs. The inducing covariance is $K_{uu} = I_Q \otimes k(Z, Z) \in \mathbb{R}^{QM \times QM}$
(block diagonal, one block per latent GP). The cross-covariance is
$K_{uf} = W^\top \otimes k(Z, X) \in \mathbb{R}^{QM \times NP}$.

**Complexity:**
- Exact: $O(Q(N^3 + P^3))$ via eigendecomposition of each $B^{(q)}$ and $K_q$.
- SVGP: $O(QM^2N + QM^3)$ with $M$ shared inducing points.
- Storage: $O(PQ + QN^2)$ for parameters.

```python
class LMCKernel(eqx.Module):
    """Linear Model of Coregionalization kernel."""
    kernels: tuple[Kernel, ...]                     # Q base kernels
    mixing: Float[Array, "P Q"]                     # W mixing matrix

    def coregionalization_matrix(self, q: int) -> Float[Array, "P P"]:
        """B^(q) = w_q w_q^T."""
        ...

    def full_covariance(
        self, X: Float[Array, "N D"]
    ) -> Float[Array, "NP NP"]:
        """Sum_q B^(q) x K_q(X, X) as gaussx operators."""
        ...
```

**Ref:** Alvarez, Rosasco & Lawrence (2012). *Kernels for Vector-Valued Functions: A Review.* Foundations and Trends in ML.

---

### Gap 2: ICM -- Intrinsic Coregionalization Model

**Mixing structure:** LMC restricted to a single shared kernel ($Q$ latent GPs, one kernel).

**Math:** All latent GPs share the same kernel $k$, so:

$$k_{pp'}(x, x') = b_{pp'}\, k(x, x'), \qquad K_{\text{full}} = B \otimes K$$

where $B = W W^\top + \text{diag}(\kappa) \in \mathbb{R}^{P \times P}$ is the
coregionalization matrix (with optional diagonal "nugget" $\kappa$ for per-output variance).

This is the simplest Kronecker structure: the full $NP \times NP$ covariance is
a single Kronecker product.

**gaussx factorization:** Single `gaussx.Kronecker(B_op, K_op)` where `B_op` is a
dense $P \times P$ operator and `K_op` is the $N \times N$ input kernel matrix.
If $\kappa \neq 0$, the diagonal correction is a `gaussx.LowRankUpdate` on the
output side.

**Inducing point structure:** Shared $Z \in \mathbb{R}^{M \times d}$.
$K_{uu} = B \otimes k(Z, Z)$ is itself Kronecker. The SVGP ELBO exploits the
Kronecker structure end-to-end: the Schur complement, log-determinant, and
trace terms all decompose.

**Complexity:**
- Exact: $O(N^3 + P^3)$ via simultaneous eigendecomposition of $B$ and $K$.
- SVGP: $O(M^2 N + M^3 + P^3)$ with $M$ shared inducing points.
- Storage: $O(P^2 + N^2)$.

```python
class ICMKernel(eqx.Module):
    """Intrinsic Coregionalization Model kernel."""
    kernel: Kernel                                  # shared base kernel
    mixing: Float[Array, "P Q"]                     # W mixing matrix
    kappa: Float[Array, " P"] | None = None         # per-output diagonal variance

    def coregionalization_matrix(self) -> Float[Array, "P P"]:
        """B = W W^T + diag(kappa)."""
        ...

    def full_covariance(
        self, X: Float[Array, "N D"]
    ) -> Kronecker:
        """B x K(X, X) as a gaussx.Kronecker operator."""
        ...
```

**Ref:** Bonilla, Chai & Williams (2007). *Multi-task Gaussian Process Prediction.* NeurIPS.

---

### Gap 3: OILMM -- Orthogonal Instantaneous Linear Mixing Model

**Mixing structure:** Orthogonal $W \in \mathbb{R}^{P \times Q}$ with $W^\top W = I_Q$, $Q \leq P$.

**Math:** The orthogonality constraint on $W$ enables exact decoupling of the
multi-output posterior into $Q$ independent single-output problems. Using a
row-major data layout with observations stacked as $Y \in \mathbb{R}^{N \times P}$:

$$\mathbf{f}(x) = W\, \mathbf{g}(x) + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \sigma^2 I_P)$$

the projected observations $\tilde{Y} = Y W \in \mathbb{R}^{N \times Q}$ yield
$Q$ independent scalar GP regression problems, one for each projected column
$\tilde{y}_q \in \mathbb{R}^N$:

$$\tilde{y}_q | g_q \sim \mathcal{N}\bigl(g_q(X), \sigma^2 I_N\bigr)$$

No Kronecker product is needed at inference time -- the mixing matrix $W$
projects the data, and inference proceeds with $Q$ independent GPs.

**gaussx factorization:** At construction, the covariance is $K_{\text{full}} = W \text{diag}(K_1, \ldots, K_Q) W^\top + \sigma^2 I$.
This is a `gaussx.LowRankUpdate`: the low-rank part is $W$ applied to a
`gaussx.BlockDiag` of the $Q$ kernel matrices. At inference time, after
projection, each GP uses a standard dense operator -- no special gaussx structure.

**Inducing point structure:** Each latent GP $g_q$ has its own inducing points
$Z_q \in \mathbb{R}^{M_q \times d}$, trained independently. Total inducing
variables: $\sum_q M_q$.

**Complexity:**
- Exact: $O(QN^3)$ -- $Q$ independent $N \times N$ solves.
- SVGP: $O(Q M^2 N + Q M^3)$ -- $Q$ independent SVGP problems.
- Projection: $O(NPQ)$ one-time cost to compute $\tilde{y} = W^\top y$.
- Key advantage: linear in $P$ (number of outputs), not cubic.

```python
class OILMMKernel(eqx.Module):
    """Orthogonal Instantaneous Linear Mixing Model."""
    kernels: tuple[Kernel, ...]                     # Q independent kernels
    mixing: Float[Array, "P Q"]                     # orthogonal W, W^T W = I_Q
    noise_variance: Float[Array, ""]                # sigma^2

    def project_observations(
        self, Y: Float[Array, "N P"]
    ) -> Float[Array, "N Q"]:
        """Project to latent space: Y_tilde = Y @ W."""
        ...

    def independent_gps(self) -> tuple[Kernel, ...]:
        """Return Q independent GP kernels for decoupled inference."""
        ...
```

**Ref:** Bruinsma, Perim, Tebbutt, Sherrington, Nowozin & Turner (2020). *Scalable Exact Inference in Multi-Output Gaussian Processes.* ICML.

---

### Gap 4: Multi-Output Inducing Variables

**Structure:** Three strategies for placing inducing points across outputs.

**Math:** In multi-output SVGP, the inducing variables $u$ live in the joint
$QM$-dimensional space. Three strategies:

**(a) Shared inducing points.** All latent GPs share the same $Z \in \mathbb{R}^{M \times d}$:

$$K_{uu} = I_Q \otimes k(Z, Z) \in \mathbb{R}^{QM \times QM}$$

This is a `gaussx.BlockDiag` of $Q$ identical blocks. Solves and log-determinants
cost $O(QM^3)$.

**(b) Separate inducing points.** Each latent GP has its own $Z_q \in \mathbb{R}^{M_q \times d}$:

$$K_{uu} = \text{BlockDiag}\bigl(k_1(Z_1, Z_1), \ldots, k_Q(Z_Q, Z_Q)\bigr)$$

This is a `gaussx.BlockDiag` of $Q$ heterogeneous blocks. Solves cost $O(\sum_q M_q^3)$.

**(c) Mixed inducing points (GPflow SharedIndependent/SeparateIndependent).**
Some outputs share points, others have separate. The block structure is
still exploitable via `gaussx.BlockDiag` with appropriate grouping.

**Cross-covariance:** For shared $Z$ with an ICM kernel:

$$K_{uf} = W^\top \otimes k(Z, X) \in \mathbb{R}^{QM \times NP}$$

This is a `gaussx.Kronecker` operator, so the SVGP prediction
$\mu = K_{fu} K_{uu}^{-1} m$ and the ELBO trace term both exploit
Kronecker-BlockDiag structure.

**Complexity:**
- Shared: $O(QM^3 + QM^2 N)$, with block-diagonal $K_{uu}$.
- Separate: $O(\sum_q M_q^3 + \sum_q M_q^2 N)$.
- The mixing overhead is $O(P^2 Q)$ for the coregionalization matrix.

```python
class SharedInducingPoints(eqx.Module):
    """Shared inducing points across Q latent GPs."""
    locations: Float[Array, "M D"]                  # shared Z

    def K_uu(
        self, kernels: tuple[Kernel, ...]
    ) -> BlockDiag:
        """I_Q x k(Z, Z) as gaussx.BlockDiag."""
        ...

class SeparateInducingPoints(eqx.Module):
    """Per-latent-GP inducing points."""
    locations: tuple[Float[Array, "M_q D"], ...]    # Z_1, ..., Z_Q

    def K_uu(
        self, kernels: tuple[Kernel, ...]
    ) -> BlockDiag:
        """BlockDiag(k_1(Z_1,Z_1), ..., k_Q(Z_Q,Z_Q))."""
        ...

class MultiOutputInducingVariables(eqx.Module):
    """Unified interface for multi-output inducing variables."""
    inducing: SharedInducingPoints | SeparateInducingPoints
    mixing: Float[Array, "P Q"]

    def K_uf(
        self, X: Float[Array, "N D"], kernels: tuple[Kernel, ...]
    ) -> Float[Array, "QM NP"]:
        """Cross-covariance exploiting Kronecker structure."""
        ...
```

**Ref:** van der Wilk, Dutordoir, John, Sherrington, Sherlock & Hensman (2020). *A Framework for Interdomain and Multioutput Gaussian Processes.* arXiv:2003.01115.

---

### Gap 5: Heterotopic Observations

**Structure:** Different outputs observed at different input locations.

**Math:** In the general case, output $p$ is observed at inputs $X_p \in \mathbb{R}^{N_p \times d}$,
where $N_p$ varies per output. The total number of observations is $N_{\text{tot}} = \sum_p N_p$.

The joint covariance is no longer a clean Kronecker product. Instead, it is a
**block matrix** indexed by output pairs:

$$[K_{\text{full}}]_{pp'} = b_{pp'}\, k(X_p, X_{p'}) \in \mathbb{R}^{N_p \times N_{p'}}$$

For the ICM case, if $X_p = X$ for all $p$ (isotopic), this reduces to $B \otimes K$.

**gaussx factorization:** When the observation sets are heterotopic, the Kronecker
structure is lost. Two strategies:

1. **Padding to isotopic:** Pad each $X_p$ to a common superset $X_\cup$ with
   masked likelihood. Restores Kronecker structure at the cost of extra
   observations. Use `gaussx.Kronecker` with masking.

2. **Block-structured solve:** Assemble the $P \times P$ block matrix directly.
   Each block is $N_p \times N_{p'}$. When $P$ is small, direct block
   factorization is tractable. No special gaussx operator needed beyond
   dense blocks.

**Inducing point structure:** SVGP with shared or separate inducing points
(Gap 4) handles heterotopic data naturally. The cross-covariance $K_{uf}$
is computed per output:

$$[K_{uf}]_{qp} = k_q(Z_q, X_p) \in \mathbb{R}^{M_q \times N_p}$$

and the ELBO likelihood term decomposes per observation.

**Complexity:**
- Padded isotopic: $O((N_\cup)^3 + P^3)$ where $N_\cup = |X_\cup|$.
- Block solve: $O(N_{\text{tot}}^3)$ worst case, $O(P \max(N_p)^3)$ with structure.
- SVGP (recommended): $O(QM^2 N_{\text{tot}} + QM^3)$ -- heterotopic adds no asymptotic cost.

```python
class HeterotopicData(eqx.Module):
    """Observation data where each output has different input locations."""
    inputs: tuple[Float[Array, "N_p D"], ...]       # X_1, ..., X_P
    outputs: tuple[Float[Array, " N_p"], ...]       # y_1, ..., y_P

    @property
    def n_outputs(self) -> int: ...
    @property
    def total_observations(self) -> int: ...

class HeterotopicLikelihood(eqx.Module):
    """Likelihood that handles per-output observation sets."""
    noise_variances: Float[Array, " P"]             # per-output noise

    def log_likelihood(
        self,
        f_samples: tuple[Float[Array, "S N_p"], ...],
        data: HeterotopicData,
    ) -> Float[Array, ""]:
        """Sum of per-output log-likelihoods."""
        ...
```

**Ref:** Alvarez, Rosasco & Lawrence (2012). *Kernels for Vector-Valued Functions: A Review.* Foundations and Trends in ML, Section 5.

---

### Gap 6: Convolution Processes

**Mixing structure:** Continuous convolution with smoothing kernels (non-instantaneous mixing).

**Math:** Each output is a convolution of latent GPs with output-specific
smoothing kernels:

$$f_p(x) = \sum_{q=1}^{Q} \int_{\mathbb{R}^d} w_{pq}(\tau)\, g_q(x - \tau)\, d\tau$$

where $w_{pq}: \mathbb{R}^d \to \mathbb{R}$ are smoothing kernels (typically Gaussian).
This generalizes the instantaneous LMC mixing ($w_{pq}(\tau) = w_{pq}\,\delta(\tau)$).

The cross-covariance between outputs is:

$$k_{pp'}(x, x') = \sum_{q=1}^{Q} \int \int w_{pq}(\tau)\, w_{p'q}(\tau')\, k_q(x - \tau, x' - \tau')\, d\tau\, d\tau'$$

For Gaussian smoothing kernels $w_{pq}(\tau) = \mathcal{N}(\tau; 0, S_{pq})$ and a
Gaussian base kernel $k_q$, the integral has a closed form:

$$k_{pp'}(x, x') = \sum_{q=1}^{Q} c_{pp'q}\, \mathcal{N}(x - x'; 0, \Sigma_q + S_{pq} + S_{p'q})$$

where $c_{pp'q}$ is a normalizing constant.

**gaussx factorization:** The Kronecker structure is generally lost because the
effective kernel $k_{pp'}(x, x')$ depends on the output pair $(p, p')$. The
covariance matrix is a $P \times P$ block matrix where each block has a different
length scale. For small $P$, assemble as a dense $NP \times NP$ matrix.

For the special case of shared smoothing kernels ($S_{pq} = S_q$ for all $p$),
the kernel structure partially decomposes: $k_{pp'}(x, x') = b_{pp'}\, \tilde{k}_q(x, x')$,
recovering ICM-like Kronecker structure with a modified kernel.

**Inducing point structure:** Standard shared inducing points. The key difference
is that $K_{uf}$ and $K_{uu}$ use the convolved kernel $k_{pp'}$ rather than
the base kernel. Precompute the convolved kernel analytically for Gaussian
smoothing, or use quadrature for general smoothing kernels.

**Complexity:**
- Exact: $O((NP)^3)$ in general (no Kronecker). $O(Q(N^3 + P^3))$ for shared smoothing.
- SVGP: $O(QM^2 NP + QM^3)$.
- Kernel evaluation: $O(N^2 P^2 Q)$ for all pairwise output-input blocks.

```python
class ConvolutionKernel(eqx.Module):
    """Convolution process multi-output kernel."""
    base_kernels: tuple[Kernel, ...]                # Q latent GP kernels
    smoothing_variances: Float[Array, "P Q D"]      # S_{pq} diagonal smoothing

    def cross_covariance(
        self,
        X: Float[Array, "N D"],
        p: int,
        p_prime: int,
    ) -> Float[Array, "N N"]:
        """k_{pp'}(X, X) via analytic convolution integral."""
        ...

    def full_covariance(
        self, X: Float[Array, "N D"]
    ) -> Float[Array, "NP NP"]:
        """Assemble full block covariance matrix."""
        ...
```

**Ref:** Alvarez & Lawrence (2011). *Computationally Efficient Convolved Multiple Output Gaussian Processes.* JMLR.

---

## 4  Shared Infrastructure: gaussx Operators

All multi-output methods above build on gaussx's structured linear algebra
operators to avoid materializing large $NP \times NP$ matrices.

| gaussx Operator | Multi-Output Role | Complexity Gain |
|---|---|---|
| `gaussx.Kronecker` | $B \otimes K$ for ICM / LMC with shared kernel. Provides `solve`, `logdet`, `trace` via eigendecompositions of the two factors. | $O(N^3 + P^3)$ vs $O((NP)^3)$ |
| `gaussx.BlockDiag` | $K_{uu} = \text{BlockDiag}(k_1(Z_1,Z_1), \ldots, k_Q(Z_Q,Z_Q))$ for independent latent GP inducing covariances. Block-diagonal solves and log-dets. | $O(\sum_q M_q^3)$ vs $O((QM)^3)$ |
| `gaussx.LowRankUpdate` | $K_y = B \otimes K + \sigma^2 I$ via Woodbury. Also for OILMM: $W K_g W^\top + \sigma^2 I$ where $K_g$ is block-diagonal. | Woodbury inversion: $O(N^3 + Q^3)$ |
| `gaussx.KroneckerSum` | Noise models where $K_y = B \otimes I + I \otimes \Lambda$ (output correlation + input-diagonal noise). Eigendecomposition of both factors. | $O(N^3 + P^3)$ |
| `gaussx.DiagonalLinearOperator` | Per-output noise $\Sigma_\epsilon = \text{diag}(\sigma_1^2, \ldots, \sigma_P^2)$. Also diagonal $K_{uu}$ for spectral inducing features (see inducing_features.md). | $O(M)$ |

### Integration pattern

The typical multi-output SVGP assembly is:

```
kernel_matrix = gaussx.Kronecker(B_op, K_op)       # B x K
inducing_cov  = gaussx.BlockDiag(K_uu_blocks)      # block-diag K_uu
noise         = gaussx.DiagonalLinearOperator(...)  # per-output noise
full_cov      = gaussx.LowRankUpdate(kernel_matrix, noise)
```

All downstream operations (`solve`, `logdet`, `trace`, `sqrt`) dispatch through
gaussx primitives, which select efficient algorithms based on the operator type.

---

## 5  References

1. Alvarez, M., Rosasco, L. & Lawrence, N. (2012). *Kernels for Vector-Valued Functions: A Review.* Foundations and Trends in Machine Learning, 4(3), 195--266.
2. Bonilla, E., Chai, K. M. & Williams, C. (2007). *Multi-task Gaussian Process Prediction.* NeurIPS.
3. Bruinsma, W., Perim, E., Tebbutt, W., Sherrington, J., Sherlock, A., Nowozin, S. & Turner, R. (2020). *Scalable Exact Inference in Multi-Output Gaussian Processes.* ICML.
4. Alvarez, M. & Lawrence, N. (2011). *Computationally Efficient Convolved Multiple Output Gaussian Processes.* JMLR, 12, 1459--1500.
5. van der Wilk, M., Dutordoir, V., John, S., Sherrington, J., Sherlock, A. & Hensman, J. (2020). *A Framework for Interdomain and Multioutput Gaussian Processes.* arXiv:2003.01115.
6. Titsias, M. (2009). *Variational Learning of Inducing Variables in Sparse Gaussian Processes.* AISTATS.
