---
status: draft
version: 0.1.0
---

# pyrox.gp x LogFalkon / GSC-Falkon

**Subject:** LogFalkon (Meanti et al. 2020) — Newton outer loop for non-quadratic
losses with Falkon-preconditioned conjugate gradient inner solve. Extends Nystr\"om
kernel methods from squared loss to logistic, exponential, and other generalized
self-concordant (GSC) losses with convergence guarantees.

**Date:** 2026-04-03

---

## 1  Background -- Falkon

Falkon (Rudi et al. 2017) is a preconditioned Nystr\"om solver for kernel ridge
regression (squared loss). Given $N$ data points and $M \ll N$ Nystr\"om centers,
it solves the kernel ridge regression problem without ever forming the full
$N \times N$ kernel matrix.

### The Nystr\"om approximation

Select $M$ centers $\{z_j\}_{j=1}^M$ (uniformly at random or via leverage-score
sampling) from the training data. Form the rectangular kernel matrix
$K_{NM} \in \mathbb{R}^{N \times M}$ with $(K_{NM})_{ij} = k(x_i, z_j)$ and the
small square $K_{MM} \in \mathbb{R}^{M \times M}$ with $(K_{MM})_{ij} = k(z_i, z_j)$.

The Nystr\"om approximation to the full kernel matrix is:

$$K_{NN} \approx K_{NM}\,K_{MM}^{-1}\,K_{MN}$$

### The Falkon system

Standard kernel ridge regression solves $(K_{NN} + \lambda I)\,\alpha = y$, which
costs $O(N^3)$. The Nystr\"om substitution reduces this to an $M \times M$ system:

$$(K_{MN}\,K_{NM} + \lambda\,K_{MM})\,\beta = K_{MN}\,y$$

where $\alpha = K_{MM}^{-1}\,K_{MN}\,\beta$ recovers the full coefficients. This is
solved by preconditioned conjugate gradient (CG).

### The Falkon preconditioner

Let $B = K_{MM}^{-1/2}\,K_{MN} \in \mathbb{R}^{M \times N}$. The preconditioner is:

$$T = \frac{1}{M}\,B\,B^\top + \lambda\,I_M$$

Compute the Cholesky factorization $T = L\,L^\top$ once (cost $O(NM + M^2)$ to form
$T$, then $O(M^3)$ for Cholesky). Each preconditioned CG iteration costs $O(NM)$
for the matrix-vector product $K_{MN}\,K_{NM}\,v$.

### Overall complexity

| Operation | Cost |
|---|---|
| Center selection | $O(N)$ or $O(NM)$ for leverage scores |
| Kernel matrices $K_{NM}$, $K_{MM}$ | $O(NM + M^2)$ |
| Preconditioner $T$ | $O(NM + M^3)$ |
| Each CG iteration | $O(NM + M^2)$ |
| Total ($t$ CG iterations) | $O(NM \cdot t + M^3)$ |

With $M = O(\sqrt{N})$ and $t = O(\log(1/\epsilon))$ CG iterations, Falkon achieves
statistical optimality with $O(N\sqrt{N}\,\log(1/\epsilon))$ total cost.

---

## 2  The GSC Framework

Standard Falkon is limited to squared loss $\ell(y, f) = \frac{1}{2}(y - f)^2$
because the preconditioner and CG system assume a fixed quadratic objective.
Classification losses (logistic, exponential) break this assumption.

### Generalized Self-Concordant (GSC) losses

A twice-differentiable loss $\ell : \mathbb{R} \to \mathbb{R}$ is **generalized
self-concordant** with parameter $R > 0$ if:

$$|\ell'''(t)| \leq R\,\bigl[\ell''(t)\bigr]^{3/2} \qquad \forall\, t \in \mathbb{R}$$

This condition bounds the rate of change of the curvature. It is the key property
that guarantees **superlinear convergence** of Newton's method without requiring a
line search or trust region.

### Why GSC matters for Newton convergence

For a standard self-concordant function (Nesterov & Nemirovski 1994), Newton's
method converges quadratically once inside a neighborhood of the optimum. GSC
generalizes this: the Newton decrement $\lambda_t$ satisfies

$$\lambda_{t+1} \leq C \cdot \lambda_t^2$$

within a computable convergence radius that depends only on $R$ and the initial
Hessian. This means:

1. **No line search needed** -- the full Newton step is always safe once convergence begins
2. **Predictable iteration count** -- typically $T = O(\log\log(1/\epsilon))$ Newton steps suffice
3. **The convergence rate is independent of the condition number** of the Hessian

### Examples

| Loss | $\ell(y, f)$ | $\ell''(f)$ | GSC? | $R$ |
|---|---|---|---|---|
| Squared | $\frac{1}{2}(y - f)^2$ | $1$ | Yes (trivially) | $0$ |
| Logistic | $\log(1 + e^{-yf})$ | $\sigma(f)(1-\sigma(f))$ | Yes | $1$ |
| Exponential | $e^{-yf}$ | $e^{-yf}$ | Yes | $1$ |
| Huber | piecewise | $\mathbb{1}_{|r| \leq \delta}$ | No | -- |
| Hinge | $\max(0, 1-yf)$ | $0$ a.e. | No | -- |

**Logistic loss derivation.** Let $\sigma(f) = 1/(1 + e^{-f})$. Then:

$$\ell'(f) = \sigma(f) - y, \qquad \ell''(f) = \sigma(f)(1 - \sigma(f))$$

$$\ell'''(f) = \sigma(f)(1 - \sigma(f))(1 - 2\sigma(f))$$

Since $|1 - 2\sigma(f)| \leq 1$ and $\sigma(f)(1-\sigma(f)) \leq 1/4$:

$$|\ell'''(f)| = \ell''(f)\,|1 - 2\sigma(f)| \leq \ell''(f) \leq [\ell''(f)]^{3/2} / [\ell''(f)]^{1/2} \leq [\ell''(f)]^{3/2}$$

because $[\ell''(f)]^{1/2} \leq 1/\sqrt{4} \cdot \sqrt{4} = 1$. More precisely,
$|\ell'''| \leq 1 \cdot [\ell'']^{3/2}$, so $R = 1$.

---

## 3  LogFalkon Algorithm

LogFalkon (Meanti et al. 2020) combines the GSC Newton outer loop with the Falkon
preconditioned CG inner loop. The full algorithm proceeds as follows.

### Initialization

1. Select $M$ Nystr\"om centers $\{z_j\}$ from training data
2. Compute $K_{MM}$ and its Cholesky factor $K_{MM} = L_{MM}\,L_{MM}^\top$
3. Compute $K_{NM}$
4. Initialize predictions $f_0 = 0 \in \mathbb{R}^N$

### Newton outer loop

For $t = 0, 1, 2, \ldots, T-1$:

**Step 1: Compute weights from current predictions.**

$$w_t^{(i)} = \ell''(y_i,\, f_t^{(i)}), \qquad g_t^{(i)} = \frac{\ell'(y_i,\, f_t^{(i)})}{\ell''(y_i,\, f_t^{(i)})}$$

Define $W_t = \text{diag}(w_t) \in \mathbb{R}^{N \times N}$.

**Step 2: Form the weighted Nystr\"om system.**

The Newton step requires solving:

$$(K_{NM}^\top\,W_t\,K_{NM} + \lambda\,N\,K_{MM})\,\beta = K_{NM}^\top\,W_t\,g_t$$

This is the core insight: at each Newton iteration, the loss curvature enters
through the diagonal weight matrix $W_t$. The system has the same $M \times M$
structure as standard Falkon but with a **changing** right-hand side and
**changing** weighting.

**Step 3: Solve the inner system with preconditioned CG.**

Apply preconditioned CG to the system above. The matrix-vector product
$v \mapsto K_{NM}^\top W_t K_{NM}\,v$ costs $O(NM)$ (two rectangular
matrix-vector products and one diagonal scaling). The preconditioner $T_t$
(Section 4) must be recomputed at each Newton step because $W_t$ changes.

**Step 4: Update predictions.**

$$\alpha_t = \beta_t, \qquad f_{t+1} = K_{NM}\,\alpha_t$$

### Convergence

By the GSC property, the Newton iterates $\{f_t\}$ converge superlinearly. In
practice, $T = 5$--$10$ Newton steps suffice for logistic regression. The total
number of CG iterations across all Newton steps is typically $O(T \cdot \log(1/\epsilon))$.

---

## 4  Weighted Preconditioner

The standard Falkon preconditioner $T = (1/M)\,B\,B^\top + \lambda\,I$ is
independent of the data labels and the loss function. LogFalkon requires a
**weighted** preconditioner that changes at each Newton step.

### Construction

Let $B = K_{MM}^{-1/2}\,K_{MN} \in \mathbb{R}^{M \times N}$ (same as standard
Falkon). At Newton step $t$:

$$T_t = \frac{1}{M}\,B\,W_t\,B^\top + \lambda\,I_M$$

where $W_t = \text{diag}(w_t)$ is the diagonal matrix of loss second derivatives.
Expanding:

$$T_t = \frac{1}{M}\,K_{MM}^{-1/2}\,K_{MN}\,W_t\,K_{NM}\,K_{MM}^{-1/2} + \lambda\,I_M$$

### Cholesky factorization

Compute $T_t = L_t\,L_t^\top$ by Cholesky. This is the preconditioner for CG:

$$P_t^{-1} = L_t^{-\top}\,K_{MM}^{-1/2}\,(\cdot)\,K_{MM}^{-1/2}\,L_t^{-1}$$

The preconditioned system has condition number bounded by $O(1/\lambda)$, ensuring
fast CG convergence.

### Cost per Newton step

| Operation | Cost |
|---|---|
| Evaluate $\ell'$, $\ell''$ at $N$ points | $O(N)$ |
| Form $B\,W_t\,B^\top$ | $O(NM^2)$ |
| Cholesky of $T_t$ ($M \times M$) | $O(M^3)$ |
| CG iterations ($t_\text{cg}$ steps) | $O(NM \cdot t_\text{cg})$ |
| Update predictions $f_{t+1}$ | $O(NM)$ |

The dominant cost is $O(NM^2)$ for forming $T_t$ when $M$ is large. An alternative
is to form $T_t$ via $M$ rank-1 updates using batched outer products of the columns
of $B$ weighted by $W_t$, which can be more memory-efficient.

---

## 5  Complexity

| | Standard Kernel Logistic Regression | LogFalkon |
|---|---|---|
| Parameters | $N$ (dual) | $M$ (Nystr\"om) |
| Per Newton step | $O(N^3)$ (solve $N \times N$ system) | $O(NM^2 + NM \cdot t_\text{cg})$ |
| Newton steps | $T$ | $T$ |
| Preconditioner | N/A or $O(N^3)$ | $O(NM^2 + M^3)$ per step |
| Total | $O(T \cdot N^3)$ | $O(T \cdot (NM^2 + NM \cdot t_\text{cg}))$ |
| Memory | $O(N^2)$ | $O(NM + M^2)$ |
| Typical setting | $N \leq 10^4$ | $N \leq 10^7$, $M = O(\sqrt{N})$ |

With $M = O(\sqrt{N})$, $T = O(1)$, and $t_\text{cg} = O(\log(1/\epsilon))$:

$$\text{LogFalkon total} = O(N^2 \log(1/\epsilon))$$

compared to $O(N^3)$ for the standard approach. For $N = 10^6$ with $M = 10^3$,
LogFalkon is $\sim 10^3 \times$ faster.

---

## 6  API

### Layer 0 -- Primitives (`pyrox.gp._src.logfalkon`)

Pure functions for GSC losses, weighted preconditioner, and Newton step.

```python
def gsc_loss_logistic(
    y: Float[Array, " N"],
    f: Float[Array, " N"],
) -> tuple[Scalar, Float[Array, " N"], Float[Array, " N"]]:
    """Logistic loss with first and second derivatives.

    Returns:
        (loss, grad, hessian_diag) where
        loss = sum(log(1 + exp(-y * f))),
        grad[i] = d ell / d f_i = sigma(f_i) - y_i,
        hessian_diag[i] = d^2 ell / d f_i^2 = sigma(f_i)(1 - sigma(f_i)).

    GSC parameter: R = 1.
    """
    ...


def gsc_loss_exponential(
    y: Float[Array, " N"],
    f: Float[Array, " N"],
) -> tuple[Scalar, Float[Array, " N"], Float[Array, " N"]]:
    """Exponential loss with first and second derivatives.

    Returns:
        (loss, grad, hessian_diag) where
        loss = sum(exp(-y * f)),
        grad[i] = -y_i * exp(-y_i * f_i),
        hessian_diag[i] = exp(-y_i * f_i).

    GSC parameter: R = 1.
    """
    ...


def weighted_falkon_preconditioner(
    K_MM_chol: Float[Array, "M M"],        # Cholesky of K_MM (lower triangular)
    K_NM: Float[Array, "N M"],             # rectangular kernel matrix
    weights: Float[Array, " N"],           # ell''(y, f_t) diagonal weights
    penalty: float,                        # regularization lambda
    M: int,                                # number of centers
) -> Float[Array, "M M"]:
    """Weighted Falkon preconditioner T_t = (1/M) B W B^T + lambda I.

    Computes B = K_MM^{-1/2} K_MN via the Cholesky factor, then forms
    the weighted inner product and adds the ridge.

    Returns the Cholesky factor L_t of T_t.

    Complexity: O(NM^2 + M^3).
    """
    ...


def logfalkon_newton_step(
    K_NM: Float[Array, "N M"],             # rectangular kernel matrix
    K_MM_chol: Float[Array, "M M"],        # Cholesky of K_MM
    precond_chol: Float[Array, "M M"],     # Cholesky of T_t (from weighted_falkon_preconditioner)
    grad: Float[Array, " N"],              # ell'(y, f_t)
    hessian_diag: Float[Array, " N"],      # ell''(y, f_t)
    penalty: float,                        # regularization lambda
    N: int,                                # number of data points
    cg_maxiter: int = 100,                 # max CG iterations
    cg_tol: float = 1e-6,                  # CG convergence tolerance
    beta_init: Float[Array, " M"] | None = None,  # warm-start for CG
) -> tuple[Float[Array, " M"], Float[Array, " N"]]:
    """One Newton step of LogFalkon: solve the weighted Nystrom system via CG.

    Solves: (K_NM^T W K_NM + lambda N K_MM) beta = K_NM^T W g
    where W = diag(hessian_diag), g = grad / hessian_diag.

    Returns:
        (beta, f_new) where f_new = K_NM @ beta are the updated predictions.

    Complexity: O(NM * cg_iters) for CG + O(NM) for prediction update.
    """
    ...
```

### Layer 1 -- Solver (`pyrox.gp.solvers`)

```python
class LogFalkonSolver(eqx.Module):
    """Solver for kernel classification via LogFalkon.

    Runs a Newton outer loop with Falkon-preconditioned CG inner solves.
    Supports any GSC loss function.
    """
    centers: Float[Array, "M d"]           # Nystrom centers
    K_MM_chol: Float[Array, "M M"]         # Cholesky of K_MM
    penalty: float                         # regularization lambda
    n_newton: int = 10                     # max Newton iterations
    cg_maxiter: int = 100                  # max CG iterations per Newton step
    cg_tol: float = 1e-6                   # CG convergence tolerance
    newton_tol: float = 1e-8               # Newton convergence tolerance
    momentum: float = 0.0                  # EMA momentum on precision (0 = off)

    @staticmethod
    def from_data(
        X: Float[Array, "N d"],
        kernel_fn: Callable,
        M: int,
        penalty: float,
        *,
        center_selection: str = "uniform",  # "uniform" or "leverage"
        n_newton: int = 10,
        cg_maxiter: int = 100,
        cg_tol: float = 1e-6,
        newton_tol: float = 1e-8,
        momentum: float = 0.0,
        key: PRNGKeyArray,
    ) -> "LogFalkonSolver":
        """Initialize solver: select centers, compute K_MM and its Cholesky.

        Args:
            X: Training features.
            kernel_fn: k(X1, X2) -> kernel matrix.
            M: Number of Nystrom centers.
            penalty: Regularization parameter lambda.
            center_selection: "uniform" (random subset) or "leverage" (RLS).
            key: JAX PRNG key for center selection.

        Complexity: O(NM + M^3) for setup.
        """
        ...

    def solve(
        self,
        X: Float[Array, "N d"],
        y: Float[Array, " N"],
        kernel_fn: Callable,
        loss_fn: Callable = gsc_loss_logistic,
    ) -> "LogFalkonResult":
        """Run the full LogFalkon Newton loop.

        Args:
            X: Training features.
            y: Training labels (in {-1, +1} for classification).
            kernel_fn: k(X1, X2) -> kernel matrix.
            loss_fn: GSC loss function returning (loss, grad, hess_diag).

        Returns:
            LogFalkonResult with coefficients, predictions, and convergence info.

        Complexity: O(T * (NM^2 + NM * cg_iters)) total.
        """
        ...

    def predict(
        self,
        X_new: Float[Array, "N_star d"],
        beta: Float[Array, " M"],
        kernel_fn: Callable,
    ) -> Float[Array, " N_star"]:
        """Predict at new locations: f_* = K(X_*, centers) @ beta.

        Complexity: O(N_star * M).
        """
        ...


class LogFalkonResult(eqx.Module):
    """Result container for LogFalkon solve."""
    beta: Float[Array, " M"]               # Nystrom coefficients
    predictions: Float[Array, " N"]        # final predictions f_T
    n_newton_steps: int                    # actual Newton steps taken
    newton_losses: Float[Array, " T"]      # loss at each Newton step
    converged: bool                        # whether Newton converged
```

### Layer 2 -- Integration with `pyrox.gp`

```python
def logfalkon_classify(
    X_train: Float[Array, "N d"],
    y_train: Float[Array, " N"],
    X_test: Float[Array, "N_star d"],
    kernel: "pyrox.gp.Kernel",
    M: int = 1000,
    penalty: float = 1e-6,
    loss: str = "logistic",
    *,
    key: PRNGKeyArray,
) -> Float[Array, " N_star"]:
    """End-to-end kernel classification via LogFalkon.

    Thin wrapper that:
    1. Extracts kernel_fn from the pyrox.gp.Kernel object.
    2. Initializes LogFalkonSolver.
    3. Runs the Newton loop.
    4. Returns predictions at test points.

    Usage:
        kernel = pyrox.gp.RBFKernel(lengthscale=1.0, variance=1.0)
        f_test = logfalkon_classify(X_train, y_train, X_test, kernel, M=500, key=key)
        probs = jax.nn.sigmoid(f_test)
    """
    ...
```

---

## 7  Implementation Notes

### Momentum on precision (EMA smoothing)

When the Newton weights $W_t$ change rapidly between steps (e.g., early iterations
with poor initial predictions), the preconditioner $T_t$ can oscillate, degrading
CG convergence. An exponential moving average (EMA) on the weights stabilizes this:

$$\bar{w}_t = \gamma\,\bar{w}_{t-1} + (1 - \gamma)\,w_t$$

with momentum $\gamma \in [0, 1)$. Use $\bar{w}_t$ in place of $w_t$ when forming
$T_t$. In practice $\gamma = 0.5$--$0.9$ helps on ill-conditioned problems. Set
$\gamma = 0$ (default) for the standard algorithm.

### Convergence criteria

The Newton loop terminates when any of:

1. **Relative loss decrease:** $|\ell_{t} - \ell_{t-1}| / |\ell_{t-1}| < \texttt{newton\_tol}$
2. **Gradient norm:** $\|K_{NM}^\top W_t\,g_t\|_\infty < \texttt{newton\_tol}$
3. **Maximum iterations:** $t \geq \texttt{n\_newton}$

Criterion 1 is cheap ($O(1)$ after loss evaluation). Criterion 2 requires an extra
$O(NM)$ matrix-vector product but gives a tighter convergence certificate.

### Warm-starting CG

At Newton step $t > 0$, initialize the CG iterate $\beta^{(0)}_t = \beta_{t-1}$
(the solution from the previous Newton step). Since consecutive Newton systems are
similar (the weights $W_t$ change smoothly near convergence), warm-starting
typically reduces CG iterations by $30$--$50\%$ in later Newton steps.

### Numerical stability

- **Hessian floor:** Clamp $\ell''(y_i, f_i) \geq \epsilon_{\min}$ (e.g.,
  $\epsilon_{\min} = 10^{-12}$) to avoid division by zero in $g_t = \ell'/\ell''$
  when predictions are very confident.
- **Cholesky regularization:** Add a small diagonal jitter $\delta I$ to $T_t$
  before Cholesky if $T_t$ is near-singular (can happen when $\lambda$ is very
  small and $M$ is large).
- **Mixed precision:** $K_{NM}$ storage in `float32` with CG accumulation in
  `float64` can improve convergence on large problems. JAX supports this via
  `jnp.float64` promotion.

### JAX / `jax.lax.while_loop` considerations

The Newton outer loop has a data-dependent iteration count (convergence-based
termination), so it must be implemented with `jax.lax.while_loop` for JIT
compatibility. The CG inner loop can similarly use `jax.lax.while_loop` or a fixed
`jax.lax.fori_loop` with early-exit via conditional no-ops.

---

## 8  References

1. Meanti, G., Carratino, L., Rosasco, L. & Rudi, A. (2020). *Kernel Methods through the Roof: Handling Billions of Points Efficiently.* NeurIPS. -- LogFalkon algorithm, GSC-Falkon extension, weighted preconditioner.

2. Rudi, A., Carratino, L. & Rosasco, L. (2017). *FALKON: An Optimal Large Scale Kernel Method.* NeurIPS. -- Original Falkon for squared loss, Nystr\"om preconditioner, statistical optimality.

3. Marteau-Ferey, U., Ostrovskii, D., Bach, F. & Rudi, A. (2019). *Beyond Least-Squares: Fast Rates for Regularized Empirical Risk Minimization through Self-Concordance.* COLT. -- GSC loss framework, convergence theory for Newton with GSC losses.

4. Rudi, A., Calandriello, D., Carratino, L. & Rosasco, L. (2018). *On Fast Leverage Score Sampling and Optimal Learning.* NeurIPS. -- Leverage score sampling for Nystr\"om center selection.

5. Nesterov, Y. & Nemirovski, A. (1994). *Interior-Point Polynomial Algorithms in Convex Optimization.* SIAM. -- Original self-concordance theory.
