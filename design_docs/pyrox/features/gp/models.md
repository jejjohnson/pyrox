---
status: draft
version: 0.1.0
---

# pyrox x Model Patterns: Gap Analysis

**Subject:** Model-level compositions sourced from GPyTorch (ExactGP, ApproximateGP,
Deep GP, GPLVM), MarkovFlow (spatio-temporal), and GPJax (OILMM model).

**Date:** 2026-04-02

---

## 1  Summary

pyrox.gp's philosophy: **the user owns the model, pyrox.gp owns the GP math.**
Unlike GPyTorch/GPJax which provide monolithic model classes, pyrox.gp provides
composable building blocks (`gp_sample`, `gp_factor`, `Kernel`, `Solver`, `Guide`)
that the user wires together in a NumPyro model.

This doc catalogs the model *patterns* that pyrox.gp should make easy -- not as
classes to implement, but as compositions that the API must support cleanly.
Each gap shows the mathematical model, the NumPyro model function that composes
pyrox.gp primitives, complexity, and what it demonstrates about API composability.

---

## 2  Common Framework

Every GP model in pyrox.gp enters the NumPyro trace through exactly one of two
entry points. The choice is determined by conjugacy:

### `gp_factor` -- collapsed (Gaussian likelihood)

Adds the marginal log-likelihood directly as a NumPyro factor site. The latent
function $f$ is analytically marginalized out. No guide is needed.

$$\log p(y \mid X, \theta) = -\tfrac{1}{2} y^\top (K + \sigma^2 I)^{-1} y
  - \tfrac{1}{2} \log |K + \sigma^2 I| - \tfrac{N}{2} \log 2\pi$$

```python
def model(X, y):
    kernel = pyrox.gp.RBF(variance=..., lengthscale=...)
    solver = pyrox.gp.CholeskySolver()
    prior = pyrox.gp.GPPrior(kernel=kernel, solver=solver, X=X)
    noise_var = numpyro.param("noise_var", 0.1, constraint=positive)
    pyrox.gp.gp_factor("gp", prior, y, noise_var)
```

### `gp_sample` -- latent (non-conjugate likelihood)

Declares $f \sim \mathcal{GP}(\mu, k)$ as a NumPyro sample site. Requires a
guide to provide the variational posterior. The user then passes $f$ to any
likelihood.

```python
def model(X, y):
    kernel = pyrox.gp.RBF(variance=..., lengthscale=...)
    solver = pyrox.gp.CholeskySolver()
    prior = pyrox.gp.GPPrior(kernel=kernel, solver=solver, X=X)
    f = pyrox.gp.gp_sample("f", prior)          # latent GP draw
    numpyro.sample("y", dist.Bernoulli(logits=f), obs=y)
```

**Key design principle:** pyrox.gp never touches the likelihood. It provides
$f$; the user writes `numpyro.sample("y", likelihood(f), obs=y)`. This means
any likelihood NumPyro supports (Poisson, Bernoulli, StudentT, custom) works
without pyrox.gp knowing about it.

---

## 3  Gap Catalog

### Gap 1: Exact GP Regression

**Source:** GPyTorch `ExactGP`, GPJax `ConjugatePosterior`, GPflow `GPR`.

**Math:** The conjugate Gaussian model:

$$f \sim \mathcal{GP}(0, k), \qquad y \mid f \sim \mathcal{N}(f, \sigma^2 I)$$

Marginal likelihood:

$$\log p(y \mid \theta) = -\tfrac{1}{2} y^\top (K_{ff} + \sigma^2 I)^{-1} y
  - \tfrac{1}{2} \log |K_{ff} + \sigma^2 I| - \tfrac{N}{2} \log 2\pi$$

Posterior predictive:

$$f_* \mid y \sim \mathcal{N}\!\bigl(K_{*f}(K_{ff} + \sigma^2 I)^{-1} y,\;
  K_{**} - K_{*f}(K_{ff} + \sigma^2 I)^{-1} K_{f*}\bigr)$$

**Complexity:** $O(N^3)$ for Cholesky; $O(N^2 k)$ with CG/BBMM solver.

**Demo code:**

```python
def model(X, y):
    # --- hyperparameters ---
    variance = numpyro.param("variance", 1.0, constraint=positive)
    lengthscale = numpyro.param("lengthscale", 1.0, constraint=positive)
    noise_var = numpyro.param("noise_var", 0.1, constraint=positive)

    # --- GP prior ---
    kernel = pyrox.gp.RBF(variance=variance, lengthscale=lengthscale)
    solver = pyrox.gp.CholeskySolver()
    prior = pyrox.gp.GPPrior(kernel=kernel, solver=solver, X=X)

    # --- collapsed marginal likelihood ---
    pyrox.gp.gp_factor("gp", prior, y, noise_var)
```

**Composability demonstrated:** Simplest possible pattern. `gp_factor` is the
single line that connects the GP to the NumPyro trace. Kernel and solver are
independent choices.

**Ref:** Rasmussen & Williams (2006) *Gaussian Processes for Machine Learning.* MIT Press.

---

### Gap 2: SVGP (Sparse Variational GP)

**Source:** GPyTorch `ApproximateGP` + `VariationalStrategy`, GPJax `SVGP`, GPflow `SVGP`.

**Math:** Introduce $M \ll N$ inducing variables $u = f(Z)$ with variational
posterior $q(u) = \mathcal{N}(m, S)$. The ELBO is:

$$\mathcal{L} = \sum_{n=1}^{N} \mathbb{E}_{q(f_n)}[\log p(y_n \mid f_n)]
  - \mathrm{KL}[q(u) \| p(u)]$$

where the conditional is:

$$q(f_n) = \mathcal{N}\!\bigl(K_{nZ} K_{ZZ}^{-1} m,\;
  k_{nn} - K_{nZ} K_{ZZ}^{-1}(K_{ZZ} - S) K_{ZZ}^{-1} K_{Zn}\bigr)$$

**Complexity:** $O(NM^2)$ per ELBO evaluation. Minibatch-compatible.

**Demo code:**

```python
def model(X, y):
    variance = numpyro.param("variance", 1.0, constraint=positive)
    lengthscale = numpyro.param("lengthscale", 1.0, constraint=positive)

    kernel = pyrox.gp.RBF(variance=variance, lengthscale=lengthscale)
    solver = pyrox.gp.CholeskySolver()
    prior = pyrox.gp.GPPrior(kernel=kernel, solver=solver, X=X)

    # --- inducing points + guide ---
    Z = numpyro.param("Z", kmeans_init(X, M=100))
    guide = pyrox.gp.InducingPointGuide(
        Z=Z,
        variational_family=pyrox.gp.FullRankGuide(M=100),
        whiten=True,
    )

    # --- latent GP ---
    f = pyrox.gp.gp_sample("f", prior, guide=guide)
    numpyro.sample("y", dist.Poisson(rate=jnp.exp(f)), obs=y)
```

**Composability demonstrated:** The guide is a separate object plugged into
`gp_sample`. Swapping `FullRankGuide` for `MeanFieldGuide` or `NaturalGuide`
changes the variational family without touching the model. Any non-conjugate
likelihood works.

**Ref:** Hensman, Matthews & Ghahramani (2015) *Scalable Variational Gaussian Process Classification.* AISTATS.

---

### Gap 3: VGP (Full Variational GP)

**Source:** GPJax `VariationalGaussian`, GPflow `VGP`.

**Math:** Variational inference with $q(f) = \mathcal{N}(m, S)$ over all $N$
function values (no inducing points). The ELBO is:

$$\mathcal{L} = \sum_{n=1}^{N} \mathbb{E}_{q(f_n)}[\log p(y_n \mid f_n)]
  - \mathrm{KL}[\mathcal{N}(m, S) \| \mathcal{N}(0, K_{ff})]$$

**Complexity:** $O(N^3)$ -- no sparsity. Useful when $N$ is moderate and exact
posterior covariance is desired.

**Demo code:**

```python
def model(X, y):
    variance = numpyro.param("variance", 1.0, constraint=positive)
    lengthscale = numpyro.param("lengthscale", 1.0, constraint=positive)

    kernel = pyrox.gp.RBF(variance=variance, lengthscale=lengthscale)
    solver = pyrox.gp.CholeskySolver()
    prior = pyrox.gp.GPPrior(kernel=kernel, solver=solver, X=X)

    # --- full-rank variational guide (no inducing points) ---
    guide = pyrox.gp.FullRankGuide(M=X.shape[0], whiten=True)

    f = pyrox.gp.gp_sample("f", prior, guide=guide)
    numpyro.sample("y", dist.Bernoulli(logits=f), obs=y)
```

**Composability demonstrated:** Same `gp_sample` entry point as SVGP. The only
difference is the guide: no `InducingPointGuide` wrapper, just a direct
`FullRankGuide` over all $N$ points.

**Ref:** Opper & Archambeau (2009) *The Variational Gaussian Approximation Revisited.* Neural Computation.

---

### Gap 4: Temporal GP (Kalman)

**Source:** MarkovFlow `GaussMarkovModel`, BayesNewton `InfiniteHorizonModel`.

**Math:** A Matern GP is represented as a Linear Time-Invariant SDE:

$$dx(t) = A\,x(t)\,dt + L\,d\beta(t), \qquad y_n = H\,x(t_n) + \varepsilon_n$$

Inference uses the Kalman filter (forward) and RTS smoother (backward):

**Predict:** $\hat{m}_{t|t-1} = \Phi\,m_{t-1|t-1}, \quad
  \hat{P}_{t|t-1} = \Phi\,P_{t-1|t-1}\,\Phi^\top + Q_d$

**Update:** $K_t = \hat{P}_{t|t-1} H^\top (H\hat{P}_{t|t-1}H^\top + R)^{-1}, \quad
  m_{t|t} = \hat{m}_{t|t-1} + K_t(y_t - H\hat{m}_{t|t-1})$

where $\Phi = \exp(A\,\Delta t)$ and $Q_d = P_\infty - \Phi\,P_\infty\,\Phi^\top$.

**Complexity:** $O(N S^3)$ where $S$ is the state dimension (e.g., $S=3$ for Matern-5/2).

**Demo code:**

```python
def model(times, y):
    variance = numpyro.param("variance", 1.0, constraint=positive)
    lengthscale = numpyro.param("lengthscale", 1.0, constraint=positive)
    noise_var = numpyro.param("noise_var", 0.1, constraint=positive)

    # --- kernel with state-space representation ---
    kernel = pyrox.gp.Matern(variance=variance, lengthscale=lengthscale, nu=2.5)
    solver = pyrox.gp.KalmanSolver()
    prior = pyrox.gp.MarkovGPPrior(kernel=kernel, solver=solver, times=times)

    # --- collapsed via Kalman filter ---
    pyrox.gp.gp_factor("gp", prior, y, noise_var)
```

**Composability demonstrated:** The kernel is the same `Matern` kernel -- the
solver changes from `CholeskySolver` to `KalmanSolver`, which triggers the
state-space path. The `MarkovGPPrior` variant knows it has sorted 1-D inputs.
Same `gp_factor` entry point.

**Ref:** Sarkka & Solin (2019) *Applied Stochastic Differential Equations.* Cambridge University Press.

---

### Gap 5: Temporal SVGP

**Source:** MarkovFlow `SparseCVI`, BayesNewton `SparseGP` with `KalmanSmoother`.

**Math:** Sparse variational inference on temporal GPs. Inducing points are placed
in time, giving a Markov-structured variational posterior:

$$q(f) = \int p(f \mid u)\,q(u)\,du, \qquad
  q(u) = \mathcal{N}(m_u, S_u)$$

where $p(f \mid u)$ is the conditional from the state-space model and $q(u)$ has
banded (tri-diagonal block) precision. The ELBO is computed via Kalman filtering
through pseudo-observations at the inducing times.

**Complexity:** $O((N + M) S^3)$ -- linear in both data and inducing points.

**Demo code:**

```python
def model(times, y):
    variance = numpyro.param("variance", 1.0, constraint=positive)
    lengthscale = numpyro.param("lengthscale", 1.0, constraint=positive)

    kernel = pyrox.gp.Matern(variance=variance, lengthscale=lengthscale, nu=1.5)
    solver = pyrox.gp.KalmanSolver()
    prior = pyrox.gp.MarkovGPPrior(kernel=kernel, solver=solver, times=times)

    # --- Kalman-structured guide ---
    Z_times = jnp.linspace(times.min(), times.max(), M=50)
    guide = pyrox.gp.KalmanGuide(inducing_times=Z_times)

    f = pyrox.gp.gp_sample("f", prior, guide=guide)
    numpyro.sample("y", dist.Poisson(rate=jnp.exp(f)), obs=y)
```

**Composability demonstrated:** `KalmanGuide` slots into the same `gp_sample`
interface. The temporal structure is exploited automatically because the prior
is a `MarkovGPPrior`. Non-conjugate likelihoods work out of the box.

**Ref:** Adam, Chang, Khan & Solin (2020) *Dual Parameterization of Sparse Variational Gaussian Processes.* NeurIPS.

---

### Gap 6: Deep GP

**Source:** GPyTorch `DeepGP`, Salimbeni & Deisenroth (2017).

**Math:** An $L$-layer composition of GPs:

$$h_0 = X, \qquad h_\ell \sim \mathcal{GP}_\ell(0, k_\ell(h_{\ell-1}, h_{\ell-1})),
  \quad \ell = 1, \ldots, L$$
$$y \mid h_L \sim p(y \mid h_L)$$

Each layer uses SVGP with its own inducing points $Z_\ell$ and guide $q(u_\ell)$.
The ELBO decomposes as:

$$\mathcal{L} = \mathbb{E}_{q(h_{1:L})}[\log p(y \mid h_L)]
  - \sum_{\ell=1}^{L} \mathrm{KL}[q(u_\ell) \| p(u_\ell)]$$

**Complexity:** $O(L \cdot N M^2)$ per layer.

**Demo code:**

```python
def model(X, y, n_layers=3):
    h = X
    for ell in range(n_layers):
        variance = numpyro.param(f"var_{ell}", 1.0, constraint=positive)
        lengthscale = numpyro.param(f"ls_{ell}", 1.0, constraint=positive)

        kernel = pyrox.gp.RBF(variance=variance, lengthscale=lengthscale)
        solver = pyrox.gp.CholeskySolver()
        prior = pyrox.gp.GPPrior(kernel=kernel, solver=solver, X=h)

        Z = numpyro.param(f"Z_{ell}", kmeans_init(h, M=50))
        guide = pyrox.gp.InducingPointGuide(
            Z=Z,
            variational_family=pyrox.gp.MeanFieldGuide(M=50),
            whiten=True,
        )

        h = pyrox.gp.gp_sample(f"h_{ell}", prior, guide=guide)

    numpyro.sample("y", dist.Normal(h, 0.1), obs=y)
```

**Composability demonstrated:** No `DeepGP` class exists. The user writes a
for-loop, feeding each layer's output as the next layer's input. Each
`gp_sample` call is independent. This is pure NumPyro composition -- pyrox.gp
provides each layer, the user owns the architecture.

**Ref:** Salimbeni & Deisenroth (2017) *Doubly Stochastic Variational Inference for Deep Gaussian Processes.* NeurIPS.

---

### Gap 7: GPLVM (GP Latent Variable Model)

**Source:** GPyTorch `GPLVM`, GPJax latent variable GP.

**Math:** Dimensionality reduction via a GP mapping from latent space to
observation space:

$$X_n \sim \mathcal{N}(0, I_Q), \qquad y_n \mid X_n \sim \mathcal{N}(f(X_n), \sigma^2 I)$$

The latent inputs $X$ are point-estimated (MAP) alongside GP hyperparameters.
The marginal likelihood is:

$$\log p(Y \mid X, \theta) = -\tfrac{1}{2} \mathrm{tr}\bigl[(K + \sigma^2 I)^{-1} Y Y^\top\bigr]
  - \tfrac{D}{2} \log |K + \sigma^2 I| - \tfrac{ND}{2} \log 2\pi$$

where $Y \in \mathbb{R}^{N \times D}$ and $K$ depends on the latent $X \in \mathbb{R}^{N \times Q}$.

**Complexity:** $O(N^3)$ (exact) or $O(NM^2)$ (sparse).

**Demo code:**

```python
def model(Y):
    N, D = Y.shape
    Q = 2  # latent dimensionality

    # --- latent inputs (point estimate) ---
    X_latent = numpyro.param("X_latent", pca_init(Y, Q))

    variance = numpyro.param("variance", 1.0, constraint=positive)
    lengthscale = numpyro.param("lengthscale", jnp.ones(Q), constraint=positive)
    noise_var = numpyro.param("noise_var", 0.1, constraint=positive)

    kernel = pyrox.gp.RBF(variance=variance, lengthscale=lengthscale)
    solver = pyrox.gp.CholeskySolver()
    prior = pyrox.gp.GPPrior(kernel=kernel, solver=solver, X=X_latent)

    # --- collapsed per output dimension ---
    for d in range(D):
        pyrox.gp.gp_factor(f"gp_{d}", prior, Y[:, d], noise_var)
```

**Composability demonstrated:** The latent inputs are a plain `numpyro.param` --
pyrox.gp does not know they are latent. The user owns the latent space; pyrox.gp
provides the GP mapping. Multiple output dimensions share the same prior (same
$K$), using `gp_factor` per dimension.

**Ref:** Lawrence (2005) *Probabilistic Non-linear Principal Component Analysis with Gaussian Process Latent Variable Models.* JMLR.

---

### Gap 8: Bayesian GPLVM

**Source:** GPyTorch `BayesianGPLVM`, Titsias & Lawrence (2010).

**Math:** Extends GPLVM by placing a variational posterior on the latent inputs:

$$q(X) = \prod_{n=1}^{N} \mathcal{N}(X_n; \mu_n, \sigma_n^2 I_Q)$$

The collapsed ELBO marginalizes $f$ and bounds $\log p(Y)$:

$$\mathcal{L} = \log \mathcal{N}(Y; 0, K_{ff} + \sigma^2 I)
  - \tfrac{1}{2\sigma^2} \mathrm{tr}(\tilde{K})
  - \mathrm{KL}[q(X) \| p(X)]$$

where $\tilde{K} = \sum_n \mathrm{Var}_{q(X_n)}[k(\cdot, X_n)]$ is the
trace-correction term from uncertain inputs.

**Complexity:** $O(N^3)$ (exact) or $O(NM^2)$ (sparse) plus $O(NQ)$ for the
latent distribution.

**Demo code:**

```python
def model(Y):
    N, D = Y.shape
    Q = 2

    # --- latent inputs (variational) ---
    X_mu = numpyro.param("X_mu", pca_init(Y, Q))
    X_logstd = numpyro.param("X_logstd", -2.0 * jnp.ones((N, Q)))
    X_latent = numpyro.sample(
        "X", dist.Normal(X_mu, jnp.exp(X_logstd)).to_event(1)
    )

    variance = numpyro.param("variance", 1.0, constraint=positive)
    lengthscale = numpyro.param("lengthscale", jnp.ones(Q), constraint=positive)
    noise_var = numpyro.param("noise_var", 0.1, constraint=positive)

    kernel = pyrox.gp.RBF(variance=variance, lengthscale=lengthscale)
    solver = pyrox.gp.CholeskySolver()
    prior = pyrox.gp.GPPrior(kernel=kernel, solver=solver, X=X_latent)

    for d in range(D):
        pyrox.gp.gp_factor(f"gp_{d}", prior, Y[:, d], noise_var)
```

**Composability demonstrated:** The only change from GPLVM (Gap 7) is that
`X_latent` is a `numpyro.sample` site instead of `numpyro.param`. NumPyro's
SVI machinery handles the variational distribution over $X$ -- pyrox.gp is
completely unaware that the inputs are uncertain. The trace-correction term
can be handled by the guide or as an explicit factor.

**Ref:** Titsias & Lawrence (2010) *Bayesian Gaussian Process Latent Variable Model.* AISTATS.

---

### Gap 9: Multi-Output GP (LMC/ICM)

**Source:** GPflow multi-output, GPyTorch `MultitaskGP`.

**Math:** The Linear Model of Coregionalization models $P$ correlated outputs
via $Q$ independent latent GPs mixed by a weight matrix $W \in \mathbb{R}^{P \times Q}$:

$$f_p(x) = \sum_{q=1}^{Q} W_{pq}\,g_q(x), \qquad g_q \sim \mathcal{GP}(0, k_q)$$

The joint covariance is:

$$\mathrm{cov}[f_p(x), f_{p'}(x')] = \sum_{q=1}^{Q} W_{pq}\,W_{p'q}\,k_q(x, x')
  = [W W^\top]_{pp'} \, k(x, x') \quad \text{(ICM, shared } k \text{)}$$

$$K_{\text{joint}} = W W^\top \otimes K_{xx} \quad \text{(ICM)}$$

**Complexity:** $O(Q \cdot N^3)$ for LMC, or $O(P^3 + N^3)$ for ICM via
Kronecker structure.

**Demo code:**

```python
def model(X, Y):
    # Y: (N, P) multi-output observations
    N, P = Y.shape
    Q = 3  # number of latent GPs

    # --- mixing matrix ---
    W = numpyro.param("W", jnp.eye(P, Q))

    variance = numpyro.param("variance", 1.0, constraint=positive)
    lengthscale = numpyro.param("lengthscale", 1.0, constraint=positive)
    noise_var = numpyro.param("noise_var", 0.1, constraint=positive)

    kernel = pyrox.gp.RBF(variance=variance, lengthscale=lengthscale)
    solver = pyrox.gp.CholeskySolver()

    # --- sample independent latent GPs ---
    prior = pyrox.gp.GPPrior(kernel=kernel, solver=solver, X=X)
    G = jnp.stack([
        pyrox.gp.gp_sample(f"g_{q}", prior) for q in range(Q)
    ], axis=-1)  # (N, Q)

    # --- mix to observations ---
    F = G @ W.T  # (N, P)
    numpyro.sample("Y", dist.Normal(F, jnp.sqrt(noise_var)), obs=Y)
```

**Composability demonstrated:** Multi-output is not a special kernel class.
The user samples $Q$ independent GPs with `gp_sample`, then applies the
mixing matrix manually. pyrox.gp provides each latent GP; the user owns the
output correlation structure. For ICM with Kronecker structure, a
`KroneckerSolver` can be swapped in for efficiency.

**Ref:** Alvarez, Rosasco & Lawrence (2012) *Kernels for Vector-Valued Functions: A Review.* Foundations and Trends in ML.

---

### Gap 10: OILMM (Orthogonal Instantaneous LMM)

**Source:** GPJax OILMM, Bruinsma et al. (2020).

**Math:** LMC with orthogonal mixing $W^\top W = I_Q$ and output-specific
diagonal noise. The orthogonality constraint decouples inference into $Q$
independent 1-D problems:

$$Y = G W^\top + E, \qquad W^\top W = I_Q, \qquad
  E_{np} \sim \mathcal{N}(0, \sigma_p^2)$$

Project to latent space: $\tilde{Y} = Y W$ (now $N \times Q$). Each column
$\tilde{y}_q$ has an independent GP:

$$\tilde{y}_q \sim \mathcal{N}(0,\; K_q + W^\top \Sigma_\varepsilon W)$$

**Complexity:** $O(Q \cdot N^3)$ -- fully decoupled across latent GPs, no
$O((NP)^3)$ joint cost.

**Demo code:**

```python
def model(X, Y):
    N, P = Y.shape
    Q = 3

    # --- orthogonal mixing (Stiefel manifold) ---
    W = numpyro.param("W", ortho_init(P, Q))  # constrained W^T W = I

    noise_std = numpyro.param("noise_std", 0.1 * jnp.ones(P), constraint=positive)
    Sigma_e = jnp.diag(noise_std ** 2)

    # --- project observations to latent space ---
    Y_tilde = Y @ W                            # (N, Q)
    noise_tilde = W.T @ Sigma_e @ W            # (Q, Q) effective noise

    # --- independent GP per latent channel ---
    for q in range(Q):
        variance = numpyro.param(f"var_{q}", 1.0, constraint=positive)
        lengthscale = numpyro.param(f"ls_{q}", 1.0, constraint=positive)

        kernel = pyrox.gp.RBF(variance=variance, lengthscale=lengthscale)
        solver = pyrox.gp.CholeskySolver()
        prior = pyrox.gp.GPPrior(kernel=kernel, solver=solver, X=X)

        pyrox.gp.gp_factor(f"gp_{q}", prior, Y_tilde[:, q], noise_tilde[q, q])
```

**Composability demonstrated:** OILMM is not a special model class. The user
handles the orthogonal projection and loops over independent GPs. pyrox.gp
provides each collapsed GP via `gp_factor`. The Stiefel manifold constraint
on $W$ is the user's responsibility (or a NumPyro constraint).

**Ref:** Bruinsma, Perim, Tebbutt, Sherborne, Requeima & Turner (2020) *Scalable Exact Inference in Multi-Output Gaussian Processes.* ICML.

---

### Gap 11: Spatio-Temporal GP

**Source:** MarkovFlow spatio-temporal, Sarkka & Solin (2019).

**Math:** Separable spatio-temporal kernel as a product of spatial and
temporal components:

$$k\bigl((x,t), (x',t')\bigr) = k_s(x, x') \cdot k_t(t, t')$$

The temporal kernel uses its SDE representation, giving a state-space model
where the state is a vector of spatial function evaluations at each time step:

$$x_t \in \mathbb{R}^{N_s \cdot S}, \qquad \text{state dim } = N_s \times S$$

The Kalman filter runs over time with spatial covariance $K_s$ entering as a
block-Kronecker structure:

$$Q_d = K_s \otimes Q_t, \qquad H = I_{N_s} \otimes H_t$$

**Complexity:** $O(N_t \cdot (N_s S)^3)$ for dense spatial, or
$O(N_t \cdot N_s M^2 S^3)$ with spatial inducing points.

**Demo code:**

```python
def model(X_spatial, times, Y):
    # Y: (N_t, N_s) spatio-temporal observations
    # --- spatial kernel ---
    var_s = numpyro.param("var_s", 1.0, constraint=positive)
    ls_s = numpyro.param("ls_s", 1.0, constraint=positive)
    kernel_s = pyrox.gp.RBF(variance=var_s, lengthscale=ls_s)

    # --- temporal kernel (state-space) ---
    var_t = numpyro.param("var_t", 1.0, constraint=positive)
    ls_t = numpyro.param("ls_t", 1.0, constraint=positive)
    kernel_t = pyrox.gp.Matern(variance=var_t, lengthscale=ls_t, nu=1.5)

    # --- compose: spatial kernel x temporal SDE ---
    kernel = pyrox.gp.ProductKernel(kernel_s, kernel_t)
    solver = pyrox.gp.KalmanSolver()
    prior = pyrox.gp.SpatioTemporalPrior(
        spatial_kernel=kernel_s,
        temporal_kernel=kernel_t,
        solver=solver,
        X_spatial=X_spatial,
        times=times,
    )

    noise_var = numpyro.param("noise_var", 0.1, constraint=positive)
    pyrox.gp.gp_factor("gp", prior, Y.ravel(), noise_var)
```

**Composability demonstrated:** Spatial and temporal kernels are composed
via `ProductKernel`. The `SpatioTemporalPrior` handles the Kronecker
state-space structure internally. The solver choice (`KalmanSolver`)
determines that the temporal axis uses the SDE representation. Adding
spatial inducing points would use `InducingPointGuide` on the spatial
component.

**Ref:** Sarkka, Solin & Hartikainen (2013) *Spatiotemporal Learning via Infinite-Dimensional Bayesian Filtering and Smoothing.* IEEE Signal Processing Magazine.

---

### Gap 12: Deep Kernel Learning

**Source:** GPyTorch `DKL` / `DeepKernelGP`, Wilson et al. (2016).

**Math:** A neural network feature extractor $\phi_\theta: \mathbb{R}^D \to \mathbb{R}^Q$
feeds into a GP:

$$g \sim \mathcal{GP}(0, k), \qquad f(x) = g(\phi_\theta(x)), \qquad
  y \mid f \sim p(y \mid f)$$

The composite kernel is $k_\text{DKL}(x, x') = k(\phi_\theta(x), \phi_\theta(x'))$.
The NN parameters $\theta$ and GP hyperparameters are jointly optimized via
the marginal likelihood (exact GP) or ELBO (SVGP).

**Complexity:** $O(N^3)$ for exact or $O(NM^2)$ for SVGP, plus the NN forward
pass cost.

**Demo code:**

```python
def model(X, y):
    # --- neural network feature extractor (pyrox.nn) ---
    nn_params = numpyro.param("nn_params", init_mlp(D_in=X.shape[1], D_out=8))
    features = mlp_forward(nn_params, X)  # (N, 8) -- user's NN

    # --- GP on learned features ---
    variance = numpyro.param("variance", 1.0, constraint=positive)
    lengthscale = numpyro.param("lengthscale", jnp.ones(8), constraint=positive)
    noise_var = numpyro.param("noise_var", 0.1, constraint=positive)

    kernel = pyrox.gp.RBF(variance=variance, lengthscale=lengthscale)
    solver = pyrox.gp.CholeskySolver()
    prior = pyrox.gp.GPPrior(kernel=kernel, solver=solver, X=features)

    pyrox.gp.gp_factor("gp", prior, y, noise_var)
```

**Composability demonstrated:** pyrox.gp has no `DeepKernel` class. The user
applies their NN to the inputs and passes the features to a standard `GPPrior`.
Joint optimization happens automatically because NumPyro differentiates through
both the NN and the GP. This pattern works with any feature extractor -- CNNs,
transformers, equivariant networks.

**Ref:** Wilson, Hu, Salakhutdinov & Xing (2016) *Deep Kernel Learning.* AISTATS.

---

## 4  Cross-Module Patterns

Several model patterns compose `pyrox.nn` and `pyrox.gp` components. Since both
are submodules of the merged `pyrox` package, these are internal imports.

| Pattern | pyrox.nn component | pyrox.gp component | Entry point | Composition |
|---|---|---|---|---|
| Deep kernel learning | MLP / CNN feature extractor | `GPPrior` + `gp_factor` or `gp_sample` | Gap 12 | NN output $\to$ GP input |
| SNGP | `SpectralNormalization` (Lipschitz NN) | `GPPrior` (RFF last layer) | `gp_factor` | Spectral-normed NN + GP head for calibrated uncertainty |
| DUE | `SpectralNormalization` + residual net | `InducingPointGuide` + `gp_sample` | `gp_sample` | Bi-Lipschitz feature extractor + SVGP output layer |
| RFF-SSGP | `RandomFourierFeatures` | Bayesian linear regression (BLR) | Direct | Weight-space approximation; GP math not used |
| Neural process | `SetEncoder` | `GPPrior` (predictive) | `gp_sample` | Amortized GP-like predictions |
| Attentive GP | `MultiHeadAttention` | `GPPrior` (inducing = keys) | `gp_sample` | Attention-weighted inducing points |

**Example: SNGP (Spectral-Normalized GP)**

```python
def model(X, y):
    # --- spectral-normalized feature extractor ---
    nn_params = numpyro.param("nn_params", init_sn_resnet(D_in=X.shape[1]))
    features = sn_resnet_forward(nn_params, X)  # Lipschitz-bounded

    # --- GP output layer with RFF approximation ---
    variance = numpyro.param("variance", 1.0, constraint=positive)
    lengthscale = numpyro.param("lengthscale", 1.0, constraint=positive)
    noise_var = numpyro.param("noise_var", 0.1, constraint=positive)

    kernel = pyrox.gp.RBF(variance=variance, lengthscale=lengthscale)
    solver = pyrox.gp.CholeskySolver()
    prior = pyrox.gp.GPPrior(kernel=kernel, solver=solver, X=features)

    pyrox.gp.gp_factor("gp", prior, y, noise_var)
```

**Example: DUE (Deterministic Uncertainty Estimation)**

```python
def model(X, y):
    # --- bi-Lipschitz feature extractor (pyrox.nn) ---
    nn_params = numpyro.param("nn_params", init_due_resnet(D_in=X.shape[1]))
    features = due_resnet_forward(nn_params, X)

    # --- SVGP output layer ---
    kernel = pyrox.gp.RBF(variance=1.0, lengthscale=1.0)
    solver = pyrox.gp.CholeskySolver()
    prior = pyrox.gp.GPPrior(kernel=kernel, solver=solver, X=features)

    Z = numpyro.param("Z", kmeans_init(features, M=20))
    guide = pyrox.gp.InducingPointGuide(
        Z=Z,
        variational_family=pyrox.gp.MeanFieldGuide(M=20),
        whiten=True,
    )

    f = pyrox.gp.gp_sample("f", prior, guide=guide)
    numpyro.sample("y", dist.Categorical(logits=f), obs=y)
```

---

## 5  References

1. Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press.
2. Hensman, J., Matthews, A. G. D. G. & Ghahramani, Z. (2015). *Scalable Variational Gaussian Process Classification.* AISTATS.
3. Opper, M. & Archambeau, C. (2009). *The Variational Gaussian Approximation Revisited.* Neural Computation.
4. Sarkka, S. & Solin, A. (2019). *Applied Stochastic Differential Equations.* Cambridge University Press.
5. Adam, V., Chang, P. E., Khan, M. E. & Solin, A. (2020). *Dual Parameterization of Sparse Variational Gaussian Processes.* NeurIPS.
6. Salimbeni, H. & Deisenroth, M. P. (2017). *Doubly Stochastic Variational Inference for Deep Gaussian Processes.* NeurIPS.
7. Lawrence, N. D. (2005). *Probabilistic Non-linear Principal Component Analysis with Gaussian Process Latent Variable Models.* JMLR.
8. Titsias, M. & Lawrence, N. D. (2010). *Bayesian Gaussian Process Latent Variable Model.* AISTATS.
9. Alvarez, M. A., Rosasco, L. & Lawrence, N. D. (2012). *Kernels for Vector-Valued Functions: A Review.* Foundations and Trends in ML.
10. Bruinsma, W., Perim, E., Tebbutt, W., Sherborne, T., Requeima, J. & Turner, R. E. (2020). *Scalable Exact Inference in Multi-Output Gaussian Processes.* ICML.
11. Sarkka, S., Solin, A. & Hartikainen, J. (2013). *Spatiotemporal Learning via Infinite-Dimensional Bayesian Filtering and Smoothing.* IEEE Signal Processing Magazine.
12. Wilson, A. G., Hu, Z., Salakhutdinov, R. & Xing, E. P. (2016). *Deep Kernel Learning.* AISTATS.
13. Liu, J. Z., Lin, Z., Padhy, S., Tran, D., Bedrax-Weiss, T. & Lakshminarayanan, B. (2020). *Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness.* NeurIPS.
14. Van Amersfoort, J., Smith, L., Teh, Y. W. & Gal, Y. (2021). *Improving Deterministic Uncertainty Estimation in Deep Learning for Classification and Regression.* arXiv:2102.11409.
