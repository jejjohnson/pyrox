---
status: draft
version: 0.1.0
---

# pyrox x SDE Kernel Representations: Gap Analysis

**Subject:** Kernel-to-state-space conversions sourced from BayesNewton (~69K lines
of kernels) and MarkovFlow SDE kernel classes.

**Date:** 2026-04-02

---

## 1  Scope

pyrox.gp supports converting stationary GP kernels into state-space (SDE)
form so that temporal GP inference runs in $O(N)$ via Kalman filtering
instead of $O(N^3)$ via dense Cholesky. The SDE representations live in
`pyrox.gp`; the downstream state-space recipes (Kalman filter, RTS smoother,
SpInGP, DARE) live in gaussx.

The `StateSpaceRep` covariance representation is already defined in pyrox.gp's
Kernel protocol. This doc specifies the concrete kernel implementations that
produce it via `.to_ssm()`.

**In scope:** Stationary kernels with closed-form SDE representations, composition rules (sum, product).
**Out of scope:** Non-stationary kernels, deep kernels, approximate SDE representations via spectral matching.

**Key insight:** Every stationary kernel whose spectral density is a rational
function of $\omega^2$ has an exact finite-dimensional SDE representation.
Composition rules (sum = block-diagonal, product = Kronecker) let complex
kernels be built from primitives.

---

## 2  Common Mathematical Framework

A stationary GP kernel $k(\tau)$ with rational spectral density can be
represented as the steady-state solution of a linear time-invariant (LTI)
stochastic differential equation:

$$\frac{d\mathbf{x}}{dt} = F\,\mathbf{x}(t) + L\,w(t), \qquad f(t) = H\,\mathbf{x}(t)$$

where $w(t)$ is white noise with spectral density $Q_c$, and:

| Symbol | Shape | Meaning |
|--------|-------|---------|
| $F$ | $d \times d$ | Feedback (drift) matrix |
| $L$ | $d \times s$ | Noise input matrix |
| $H$ | $1 \times d$ | Observation (measurement) matrix |
| $Q_c$ | $s \times s$ | Spectral density of the driving noise |
| $P_\infty$ | $d \times d$ | Stationary covariance, solving $F P_\infty + P_\infty F^\top + L Q_c L^\top = 0$ |

Here $d$ is the state dimension (SDE order) and $s$ is the noise dimension
(usually $s = 1$).

**Discretisation.** Given a time step $\Delta t = t_{k+1} - t_k$, the
continuous SDE is discretised to a discrete-time state-space model:

$$\mathbf{x}_{k+1} = A_k\,\mathbf{x}_k + \mathbf{q}_k, \qquad \mathbf{q}_k \sim \mathcal{N}(0, Q_k)$$

where:

$$A_k = \exp(F \Delta t_k), \qquad Q_k = P_\infty - A_k\,P_\infty\,A_k^\top$$

The $Q_k$ identity follows from the Lyapunov equation for $P_\infty$ and
avoids computing the matrix-fraction decomposition. For non-uniform time
steps, $A_k$ and $Q_k$ vary per step.

**Protocol.** Each SDE kernel implements:

```python
class SDEKernel(eqx.Module):
    """Protocol for kernels with SDE representations."""

    def sde_params(self) -> tuple[
        Float[Array, "d d"],    # F
        Float[Array, "d s"],    # L
        Float[Array, "1 d"],    # H
        Float[Array, "s s"],    # Q_c
        Float[Array, "d d"],    # P_inf
    ]: ...

    def discretise(self, dt: Float[Array, " N"]) -> tuple[
        Float[Array, "N d d"],  # A_k
        Float[Array, "N d d"],  # Q_k
    ]: ...
```

---

## 3  Gap Catalog

### Gap 1: Matern SDE ($\nu = 1/2, 3/2, 5/2$)

**Domain:** 1-D temporal or spatial inputs (time series, geophysical transects).

**Math:** The Matern-$\nu$ kernel with $\nu = p + 1/2$ (for $p \in \{0, 1, 2\}$)
has an exact SDE of dimension $d = p + 1$. The companion-form feedback matrix is:

$$F = \begin{pmatrix} 0 & 1 & 0 & \cdots \\ 0 & 0 & 1 & \cdots \\ \vdots & & & \ddots \\ -a_{d} & -a_{d-1} & \cdots & -a_1 \end{pmatrix}, \qquad L = \begin{pmatrix} 0 \\ \vdots \\ 0 \\ 1 \end{pmatrix}, \qquad H = \begin{pmatrix} 1 & 0 & \cdots & 0 \end{pmatrix}$$

where the coefficients $\{a_i\}$ come from the characteristic polynomial
$(\lambda^2 + \lambda_0^2)^{d}$ with $\lambda_0 = \sqrt{2\nu}/\ell$.

Concrete forms:

- **Matern-1/2** ($d=1$): $F = [-\lambda]$, $Q_c = 2\sigma^2\lambda$, $P_\infty = [\sigma^2]$ where $\lambda = 1/\ell$.
- **Matern-3/2** ($d=2$): $F = \begin{psmallmatrix} 0 & 1 \\ -\lambda^2 & -2\lambda \end{psmallmatrix}$, $\lambda = \sqrt{3}/\ell$.
- **Matern-5/2** ($d=3$): $F = \begin{psmallmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ -\lambda^3 & -3\lambda^2 & -3\lambda \end{psmallmatrix}$, $\lambda = \sqrt{5}/\ell$.

**Complexity:** State dimension $d = p+1$. Per-step Kalman cost $O(d^2) = O(1)$ for fixed $\nu$. Total: $O(N)$.

```python
class MaternSDE(eqx.Module):
    """Matern kernel in SDE form (nu = 1/2, 3/2, 5/2)."""
    variance: Float[Array, ""]       # sigma^2
    lengthscale: Float[Array, ""]    # ell
    order: int                       # p: nu = p + 1/2, state dim d = p + 1

    def sde_params(self) -> tuple[
        Float[Array, "d d"],    # F
        Float[Array, "d 1"],    # L
        Float[Array, "1 d"],    # H
        Float[Array, "1 1"],    # Q_c
        Float[Array, "d d"],    # P_inf
    ]: ...

    def discretise(self, dt: Float[Array, " N"]) -> tuple[
        Float[Array, "N d d"],  # A_k
        Float[Array, "N d d"],  # Q_k
    ]: ...
```

**Ref:** Sarkka & Solin (2019) *Applied Stochastic Differential Equations*, Ch. 12.
**Impl ref:** BayesNewton `bayesnewton.kernels.Matern{12,32,52}`; MarkovFlow `markovflow.kernels.Matern`.

---

### Gap 2: Periodic Kernel SDE

**Domain:** Signals with exact periodicity (seasonal cycles, diurnal patterns).

**Math:** The standard periodic kernel $k(\tau) = \sigma^2 \exp\bigl(-2\sin^2(\pi\tau/T)/\ell^2\bigr)$
is approximated by truncating the Fourier series of its spectral density to
$J$ components, yielding a state dimension $d = 2J$. Each Fourier component
$j \in \{1, \ldots, J\}$ contributes a $2 \times 2$ block:

$$F_j = \begin{pmatrix} 0 & -\omega_j \\ \omega_j & 0 \end{pmatrix}, \qquad \omega_j = \frac{2\pi j}{T}$$

The overall system is block-diagonal: $F = \text{blkdiag}(F_1, \ldots, F_J)$.

The coefficients $q_j$ of $Q_c$ come from the modified Bessel function expansion:

$$q_j = 2\sigma^2 \exp(-1/\ell^2)\,I_j(1/\ell^2)$$

where $I_j$ is the modified Bessel function of the first kind.

**Complexity:** State dimension $d = 2J$. Per-step cost $O(J^2)$ due to block-diagonal structure. Typically $J \leq 7$ suffices. Total: $O(NJ^2)$.

```python
class PeriodicSDE(eqx.Module):
    """Periodic kernel in SDE form via Fourier series truncation."""
    variance: Float[Array, ""]       # sigma^2
    lengthscale: Float[Array, ""]    # ell
    period: Float[Array, ""]         # T
    n_harmonics: int                 # J -> state dim d = 2J

    def sde_params(self) -> tuple[
        Float[Array, "d d"],    # F  (block-diagonal, d = 2J)
        Float[Array, "d 1"],    # L
        Float[Array, "1 d"],    # H
        Float[Array, "1 1"],    # Q_c
        Float[Array, "d d"],    # P_inf
    ]: ...

    def discretise(self, dt: Float[Array, " N"]) -> tuple[
        Float[Array, "N d d"],  # A_k (block-diagonal rotation matrices)
        Float[Array, "N d d"],  # Q_k
    ]: ...
```

**Ref:** Solin & Sarkka (2014) *Explicit Link Between Periodic Covariance Functions and State Space Models.* AISTATS.
**Impl ref:** BayesNewton `bayesnewton.kernels.Periodic`.

---

### Gap 3: Quasi-Periodic Kernel SDE (Matern $\times$ Periodic)

**Domain:** Signals with approximately periodic structure whose amplitude varies over
time (quasi-periodic stellar variability, modulated seasonal patterns).

**Math:** The quasi-periodic kernel is the product of a Matern and a periodic kernel:

$$k_{\text{QP}}(\tau) = k_{\text{Mat}}(\tau) \cdot k_{\text{Per}}(\tau)$$

The SDE representation of a product kernel is the Kronecker product of the
constituent SDE parameters:

$$F_{\text{QP}} = F_{\text{Mat}} \oplus F_{\text{Per}} = F_{\text{Mat}} \otimes I_{d_P} + I_{d_M} \otimes F_{\text{Per}}$$

$$H_{\text{QP}} = H_{\text{Mat}} \otimes H_{\text{Per}}, \qquad P_{\infty,\text{QP}} = P_{\infty,\text{Mat}} \otimes P_{\infty,\text{Per}}$$

where $\oplus$ denotes the Kronecker sum. The state dimension is
$d_{\text{QP}} = d_{\text{Mat}} \times d_{\text{Per}}$.

**Complexity:** State dimension $d = d_M \cdot 2J$. For Matern-3/2 with $J = 6$: $d = 24$. Per-step cost $O(d^3)$, total $O(Nd^3)$.

```python
class QuasiPeriodicSDE(eqx.Module):
    """Quasi-periodic kernel: Matern x Periodic in SDE form."""
    matern: MaternSDE
    periodic: PeriodicSDE

    def sde_params(self) -> tuple[
        Float[Array, "d d"],    # F  (d = d_mat * d_per)
        Float[Array, "d s"],    # L
        Float[Array, "1 d"],    # H
        Float[Array, "s s"],    # Q_c
        Float[Array, "d d"],    # P_inf
    ]: ...

    def discretise(self, dt: Float[Array, " N"]) -> tuple[
        Float[Array, "N d d"],  # A_k
        Float[Array, "N d d"],  # Q_k
    ]: ...
```

**Ref:** Sarkka & Solin (2019) *Applied Stochastic Differential Equations*, Ch. 12; Wilkinson et al. (2021) *BayesNewton*.
**Impl ref:** BayesNewton `bayesnewton.kernels.QuasiPeriodicMatern`.

---

### Gap 4: Sum Kernel SDE (Block-Diagonal Composition)

**Domain:** Any additive kernel decomposition (trend + seasonal + noise).

**Math:** For a sum of kernels $k(\tau) = \sum_{i=1}^K k_i(\tau)$, the SDE
representation is the block-diagonal concatenation of each component's SDE:

$$F_{\Sigma} = \text{blkdiag}(F_1, \ldots, F_K), \qquad L_{\Sigma} = \text{blkdiag}(L_1, \ldots, L_K)$$

$$H_{\Sigma} = [H_1, \ldots, H_K], \qquad P_{\infty,\Sigma} = \text{blkdiag}(P_{\infty,1}, \ldots, P_{\infty,K})$$

The total state dimension is $d_\Sigma = \sum_i d_i$.

**Complexity:** State dimension $d = \sum_i d_i$. Per-step cost $O(d^2)$ (exploiting block-diagonal structure via `gaussx.BlockDiag`). Total: $O(Nd^2)$.

```python
class SumSDE(eqx.Module):
    """Sum of SDE kernels via block-diagonal composition."""
    components: tuple[SDEKernel, ...]   # K component kernels

    def sde_params(self) -> tuple[
        Float[Array, "d d"],    # F  (block-diagonal, d = sum of d_i)
        Float[Array, "d s"],    # L  (block-diagonal)
        Float[Array, "1 d"],    # H  (concatenated)
        Float[Array, "s s"],    # Q_c (block-diagonal)
        Float[Array, "d d"],    # P_inf (block-diagonal)
    ]: ...

    def discretise(self, dt: Float[Array, " N"]) -> tuple[
        Float[Array, "N d d"],  # A_k (block-diagonal)
        Float[Array, "N d d"],  # Q_k (block-diagonal)
    ]: ...
```

**Ref:** Sarkka & Solin (2019) *Applied Stochastic Differential Equations*, Sec. 12.3.
**Impl ref:** BayesNewton `bayesnewton.kernels.StackKernel`.

---

### Gap 5: Product Kernel SDE (Kronecker Composition)

**Domain:** Any multiplicative kernel decomposition (modulated kernels, interaction terms).

**Math:** For a product of two kernels $k(\tau) = k_1(\tau) \cdot k_2(\tau)$,
the SDE representation uses the Kronecker sum/product of the constituent
SDE parameters:

$$F_\otimes = F_1 \otimes I_{d_2} + I_{d_1} \otimes F_2$$

$$L_\otimes = L_1 \otimes L_2, \qquad H_\otimes = H_1 \otimes H_2, \qquad P_{\infty,\otimes} = P_{\infty,1} \otimes P_{\infty,2}$$

The state dimension is $d_\otimes = d_1 \cdot d_2$.

**Complexity:** State dimension $d = d_1 \cdot d_2$. Per-step cost $O(d^3)$ in general; $O(d_1^3 d_2^3)$. Total: $O(Nd^3)$.

```python
class ProductSDE(eqx.Module):
    """Product of two SDE kernels via Kronecker composition."""
    left: SDEKernel
    right: SDEKernel

    def sde_params(self) -> tuple[
        Float[Array, "d d"],    # F  (Kronecker sum, d = d1 * d2)
        Float[Array, "d s"],    # L  (Kronecker product)
        Float[Array, "1 d"],    # H  (Kronecker product)
        Float[Array, "s s"],    # Q_c
        Float[Array, "d d"],    # P_inf (Kronecker product)
    ]: ...

    def discretise(self, dt: Float[Array, " N"]) -> tuple[
        Float[Array, "N d d"],  # A_k
        Float[Array, "N d d"],  # Q_k
    ]: ...
```

**Ref:** Sarkka & Solin (2019) *Applied Stochastic Differential Equations*, Sec. 12.3.
**Impl ref:** BayesNewton `bayesnewton.kernels.ProductKernel`.

---

### Gap 6: Cosine Kernel SDE

**Domain:** Purely oscillatory components (carrier signals, spectral peaks).

**Math:** The cosine kernel $k(\tau) = \sigma^2 \cos(\omega_0 \tau)$ has a
minimal 2-D SDE:

$$F = \begin{pmatrix} 0 & -\omega_0 \\ \omega_0 & 0 \end{pmatrix}, \qquad L = \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \qquad H = \begin{pmatrix} 1 & 0 \end{pmatrix}$$

$$P_\infty = \sigma^2 I_2, \qquad Q_c = 0$$

This is a deterministic oscillator (no driving noise). The transition matrix
is a pure rotation:

$$A_k = \begin{pmatrix} \cos(\omega_0 \Delta t_k) & -\sin(\omega_0 \Delta t_k) \\ \sin(\omega_0 \Delta t_k) & \cos(\omega_0 \Delta t_k) \end{pmatrix}, \qquad Q_k = 0$$

**Complexity:** State dimension $d = 2$. Per-step cost $O(1)$. Total: $O(N)$.

```python
class CosineSDE(eqx.Module):
    """Cosine kernel in SDE form (deterministic oscillator)."""
    variance: Float[Array, ""]       # sigma^2
    frequency: Float[Array, ""]      # omega_0

    def sde_params(self) -> tuple[
        Float[Array, "2 2"],    # F
        Float[Array, "2 1"],    # L  (zeros)
        Float[Array, "1 2"],    # H
        Float[Array, "1 1"],    # Q_c (zero)
        Float[Array, "2 2"],    # P_inf
    ]: ...

    def discretise(self, dt: Float[Array, " N"]) -> tuple[
        Float[Array, "N 2 2"],  # A_k (rotation)
        Float[Array, "N 2 2"],  # Q_k (zeros)
    ]: ...
```

**Ref:** MarkovFlow `markovflow.kernels.Cosine`.
**Impl ref:** MarkovFlow `markovflow.kernels.cosine`.

---

### Gap 7: Constant and Linear Kernel SDE

**Domain:** Non-stationary baselines (intercept + linear trend).

**Math:**

*Constant kernel* $k(t, t') = \sigma^2$ has a 1-D SDE with no dynamics:

$$F = [0], \quad L = [0], \quad H = [1], \quad Q_c = [0], \quad P_\infty = [\sigma^2]$$

$$A_k = [1], \quad Q_k = [0]$$

*Linear kernel* $k(t, t') = \sigma^2 t\, t'$ is modeled by augmenting the
state to include position integrated from a constant:

$$F = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}, \quad L = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad H = \begin{pmatrix} 0 & 1 \end{pmatrix}$$

$$P_\infty = \sigma^2 \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} \qquad \text{(degenerate — initialise from prior)}$$

Note: $P_\infty$ is singular for the linear kernel (non-stationary), so
initialisation uses the prior $P_0$ instead.

**Complexity:** State dimension $d = 1$ (constant) or $d = 2$ (linear). Per-step cost $O(1)$. Total: $O(N)$.

```python
class ConstantSDE(eqx.Module):
    """Constant kernel in SDE form."""
    variance: Float[Array, ""]       # sigma^2

    def sde_params(self) -> tuple[
        Float[Array, "1 1"],    # F (zero)
        Float[Array, "1 1"],    # L (zero)
        Float[Array, "1 1"],    # H (one)
        Float[Array, "1 1"],    # Q_c (zero)
        Float[Array, "1 1"],    # P_inf
    ]: ...

    def discretise(self, dt: Float[Array, " N"]) -> tuple[
        Float[Array, "N 1 1"],  # A_k (identity)
        Float[Array, "N 1 1"],  # Q_k (zeros)
    ]: ...


class LinearSDE(eqx.Module):
    """Linear kernel in SDE form (non-stationary, degenerate P_inf)."""
    variance: Float[Array, ""]       # sigma^2

    def sde_params(self) -> tuple[
        Float[Array, "2 2"],    # F
        Float[Array, "2 1"],    # L
        Float[Array, "1 2"],    # H
        Float[Array, "1 1"],    # Q_c
        Float[Array, "2 2"],    # P_inf (singular)
    ]: ...

    def discretise(self, dt: Float[Array, " N"]) -> tuple[
        Float[Array, "N 2 2"],  # A_k
        Float[Array, "N 2 2"],  # Q_k
    ]: ...
```

**Ref:** Sarkka & Solin (2019) *Applied Stochastic Differential Equations*, Sec. 12.2.
**Impl ref:** BayesNewton `bayesnewton.kernels.Constant`, `bayesnewton.kernels.Linear`.

---

### Gap 8: SubbandMatern SDE (Matern $\times$ Cosine)

**Domain:** Narrowband spectral peaks centered at a carrier frequency
(geophysics, audio, oscillatory time series).

**Math:** The subband Matern kernel is the product of a Matern and cosine kernel:

$$k_{\text{SB}}(\tau) = k_{\text{Mat}}(\tau) \cdot \cos(\omega_0 \tau)$$

This is equivalent to shifting the Matern spectral density to be centered at
$\pm\omega_0$. The SDE representation uses the Kronecker-sum construction
from Gap 5:

$$F_{\text{SB}} = F_{\text{Mat}} \otimes I_2 + I_{d_M} \otimes F_{\text{Cos}}$$

However, an analytically simplified form exists. For Matern-3/2 ($d_M = 2$),
the state dimension is $d = 2 \times 2 = 4$ with:

$$F_{\text{SB}} = \begin{pmatrix}
0 & -\omega_0 & 1 & 0 \\
\omega_0 & 0 & 0 & 1 \\
-\lambda^2 & 0 & -2\lambda & -\omega_0 \\
0 & -\lambda^2 & \omega_0 & -2\lambda
\end{pmatrix}$$

The transition matrix $A_k = \exp(F_{\text{SB}} \Delta t_k)$ combines
exponential decay with rotation.

**Complexity:** State dimension $d = 2 d_M$. For Matern-3/2: $d = 4$. Per-step cost $O(d^3) = O(1)$ for fixed $\nu$. Total: $O(N)$.

```python
class SubbandMaternSDE(eqx.Module):
    """Subband Matern kernel (Matern x Cosine) in SDE form."""
    variance: Float[Array, ""]       # sigma^2
    lengthscale: Float[Array, ""]    # ell
    frequency: Float[Array, ""]      # omega_0
    order: int                       # p: nu = p + 1/2

    def sde_params(self) -> tuple[
        Float[Array, "d d"],    # F  (d = 2 * (p+1))
        Float[Array, "d 1"],    # L
        Float[Array, "1 d"],    # H
        Float[Array, "1 1"],    # Q_c
        Float[Array, "d d"],    # P_inf
    ]: ...

    def discretise(self, dt: Float[Array, " N"]) -> tuple[
        Float[Array, "N d d"],  # A_k
        Float[Array, "N d d"],  # Q_k
    ]: ...
```

**Ref:** Sarkka & Solin (2019) *Applied Stochastic Differential Equations*, Sec. 12.4.
**Impl ref:** BayesNewton `bayesnewton.kernels.SubbandMatern`.

---

## 4  Shared Infrastructure

All SDE kernel representations produce the same `(F, L, H, Q_c, P_inf)` tuple
and share the same downstream Kalman machinery:

| Component | Source | Notes |
|---|---|---|
| Matrix exponential | `jax.scipy.linalg.expm` / `gaussx.matrix_exp` (planned) | $A_k = \exp(F \Delta t)$ |
| Continuous Lyapunov solver | `gaussx.solve_lyapunov` (planned) | $P_\infty$ from $FP + PF^\top + LQ_cL^\top = 0$ |
| Discrete process noise | $Q_k = P_\infty - A_k P_\infty A_k^\top$ | Avoids matrix-fraction decomposition |
| Kalman filter | `gaussx.kalman_filter` | Sequential $O(Nd^3)$ forward pass |
| RTS smoother | `gaussx.rts_smoother` | Backward pass |
| Parallel Kalman | `gaussx.parallel_kalman_filter`, `gaussx.parallel_rts_smoother` | Associative-scan $O(\log N)$ depth |
| SpInGP (precision-based) | `gaussx.spingp_posterior` (planned) | Sparse precision Kalman |
| DARE (steady-state) | `gaussx.dare` (planned) | Uniform-$\Delta t$ shortcut |
| Block-diagonal operators | `gaussx.BlockDiag` | Efficient sum-kernel structure |
| Kronecker operators | `gaussx.Kronecker` | Efficient product-kernel structure |

---

## 5  References

1. Sarkka, S. & Solin, A. (2019). *Applied Stochastic Differential Equations.* Cambridge University Press.
2. Wilkinson, W. J., et al. (2021). *BayesNewton: A Scalable Framework for Bayesian Inference with Gaussian Processes.* arXiv:2111.01721.
3. Solin, A. & Sarkka, S. (2014). *Explicit Link Between Periodic Covariance Functions and State Space Models.* AISTATS.
4. MarkovFlow. *MarkovFlow: A Markov GP library.* https://github.com/secondmind-labs/markovflow.
5. Hartikainen, J. & Sarkka, S. (2010). *Kalman Filtering and Smoothing Solutions to Temporal Gaussian Process Regression Models.* MLSP.
6. Grigorievskiy, A., Lawrence, N., & Sarkka, S. (2017). *Parallelizable Sparse Inverse Formulation Gaussian Processes (SpInGP).* IJCNN.
