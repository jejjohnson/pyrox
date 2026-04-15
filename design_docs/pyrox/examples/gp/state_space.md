---
status: stable
version: 0.1.0
---

# pyrox.gp — Temporal GP Examples with NumPyro

> Worked examples of Markovian GP models using pyrox.gp temporal solvers
> and inference strategies, integrated with NumPyro for Bayesian inference.
>
> For mathematical foundations, see [`../../api/gp/state_space.md`](../../api/gp/state_space.md).

---

## Table of Contents

1. [Exact Temporal GP Regression (Conjugate)](#1-exact-temporal-gp-regression)
2. [GP Classification with Laplace + Kalman](#2-gp-classification-with-laplace--kalman)
3. [Poisson Counts with EP + Kalman](#3-poisson-counts-with-ep--kalman)
4. [Changepoint Detection via Additive Temporal GPs](#4-changepoint-detection-via-additive-temporal-gps)
5. [Latent Temporal GP in a BHM (Explicit f with Guides)](#5-latent-temporal-gp-in-a-bhm)
6. [Spatiotemporal GP with Kronecker × Kalman](#6-spatiotemporal-gp-with-kronecker--kalman)
7. [Online / Streaming GP with Filter-Only Mode](#7-online--streaming-gp-with-filter-only-mode)
8. [Parallel Kalman on GPU for Long Series](#8-parallel-kalman-on-gpu-for-long-series)
9. [Non-LTI Temporal Model with numpyro.scan](#9-non-lti-temporal-model-with-numpyroscan)
10. [Temporal Extremes: GEV with Time-Varying Parameters](#10-temporal-extremes-gev-with-time-varying-parameters)

---

## 1. Exact Temporal GP Regression

**When**: Gaussian likelihood, 1-D time series, moderate to very long N.

The simplest case: conjugate likelihood, so a single Kalman filter pass
gives the exact posterior and log marginal likelihood.  No iteration needed.

### Model

```python
def model_temporal_regression(t, y):
    """
    Parameters
    ----------
    t : (N,) sorted observation times
    y : (N,) real-valued observations
    """
    variance = numpyro.sample("variance", dist.LogNormal(0.0, 1.0))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))
    noise_var = numpyro.sample("noise_var", dist.LogNormal(-2.0, 1.0))

    # Build Matérn-3/2 state-space model
    ss = Matern32(variance=variance, lengthscale=lengthscale).to_state_space(t)

    # Kalman filter computes exact log marginal likelihood
    solver = KalmanSolver()
    lml = solver.log_marginal_likelihood(ss, y, noise_var)

    numpyro.factor("gp_lml", lml)
```

### Internal implementation of log_marginal_likelihood

```python
def log_marginal_likelihood(self, ss, y, noise_var):
    """Kalman filter via jax.lax.scan — O(NS³)."""

    def step(carry, yn):
        x, P = carry
        # Predict
        x_pred = ss.A @ x
        P_pred = ss.A @ P @ ss.A.T + ss.Q
        # Update
        f_pred = ss.H @ x_pred
        S = ss.H @ P_pred @ ss.H.T + noise_var
        K = P_pred @ ss.H.T / S
        v = yn - f_pred
        x_new = x_pred + K * v
        P_new = P_pred - K * S * K.T
        # Log-likelihood
        ll = -0.5 * (v**2 / S + jnp.log(S) + jnp.log(2 * jnp.pi))
        return (x_new, P_new), ll

    (_, _), lls = jax.lax.scan(step, (ss.x0, ss.P0), y)
    return jnp.sum(lls)
```

### Inference

```python
# MCMC on hyperparameters only (f is marginalized out)
mcmc = MCMC(NUTS(model_temporal_regression), num_warmup=500, num_samples=1000)
mcmc.run(rng_key, t, y)
# Cost: O((warmup + samples) × NS³) with S=2 for Matérn-3/2

# SVI for MAP estimation
guide = AutoDelta(model_temporal_regression)
svi = SVI(model_temporal_regression, guide, Adam(1e-2), Trace_ELBO())
```

### Prediction

```python
# After inference, run smoother for posterior at all times
samples = mcmc.get_samples()
for s in range(num_samples):
    ss = Matern32(samples["variance"][s], samples["lengthscale"][s]).to_state_space(t)
    solver = KalmanSolver()
    smooth = solver.filter_and_smooth(ss, y, samples["noise_var"][s])
    mean_s = smooth.means     # (N,)
    var_s = smooth.variances   # (N,)

    # Interpolation: predict at new times t_star
    mean_star, var_star = solver.interpolate(smooth, ss, t_star)
```

---

## 2. GP Classification with Laplace + Kalman

**When**: Binary labels, temporal dependence, want fast deterministic
approximation (no sampling of N-dimensional f).

The Laplace approximation iterates Newton steps on the posterior mode.
Each iteration runs a full Kalman filter + smoother with the current
site approximations.

### Model

```python
def model_temporal_classification(t, y_binary):
    """
    Parameters
    ----------
    t : (N,) sorted times
    y_binary : (N,) ∈ {0, 1}
    """
    variance = numpyro.sample("variance", dist.LogNormal(0.0, 1.0))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))

    ss = Matern32(variance=variance, lengthscale=lengthscale).to_state_space(t)
    solver = KalmanSolver()
    strategy = LaplaceStrategy()

    # Bernoulli log-likelihood
    def log_lik(f_n, y_n):
        return y_n * jax.nn.log_sigmoid(f_n) + (1 - y_n) * jax.nn.log_sigmoid(-f_n)

    # Run iterative inference
    result = temporal_inference(
        ss_model=ss,
        y=y_binary,
        strategy=strategy,
        solver=solver,
        log_lik=log_lik,
        max_iters=20,
    )

    # Register the approximate log marginal likelihood
    numpyro.factor("gp_approx_lml", result.log_marginal_likelihood)
```

### Inference

```python
# MCMC on hyperparameters (variance, lengthscale)
# f is handled by the Laplace approximation inside the model
mcmc = MCMC(NUTS(model_temporal_classification), num_warmup=500, num_samples=500)
mcmc.run(rng_key, t, y_binary)
# Cost per MCMC step: O(max_iters × NS³)
```

### Notes

- The Laplace inner loop is fully differentiable (implicit differentiation
  through the fixed-point iteration) so NUTS can compute gradients w.r.t.
  variance and lengthscale.
- For PSD safety, wrap with `GaussNewtonWrapper(LaplaceStrategy())`.

---

## 3. Poisson Counts with EP + Kalman

**When**: Count data with temporal dependence.  EP typically gives
better uncertainty estimates than Laplace for Poisson likelihoods.

### Model

```python
def model_temporal_counts(t, y_counts):
    variance = numpyro.sample("variance", dist.LogNormal(0.0, 1.0))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(1.0, 1.0))

    ss = Matern52(variance=variance, lengthscale=lengthscale).to_state_space(t)
    solver = KalmanSolver()
    strategy = EPStrategy(num_ghq_points=20)  # Gauss-Hermite quadrature

    def log_lik(f_n, y_n):
        rate = jnp.exp(f_n)
        return y_n * f_n - rate - jax.scipy.special.gammaln(y_n + 1)

    result = temporal_inference(
        ss_model=ss,
        y=y_counts,
        strategy=strategy,
        solver=solver,
        log_lik=log_lik,
        max_iters=30,
    )

    numpyro.factor("gp_ep_lml", result.log_marginal_likelihood)
```

### Why EP over Laplace here

- Poisson log-likelihood is log-concave, so Laplace works, but EP's
  moment-matching gives a better approximation to the marginal posterior
  (mass-covering vs mode-seeking).
- The extra cost (cavity computation + quadrature) is small relative
  to the Kalman smoother.

---

## 4. Changepoint Detection via Additive Temporal GPs

**When**: A time series with a smooth trend plus abrupt changes.
Use two GPs with different lengthscales.

### Model

```python
def model_changepoint(t, y):
    noise_var = numpyro.sample("noise_var", dist.LogNormal(-2, 1))

    # Smooth long-range trend (Matérn-5/2, long lengthscale)
    var_trend = numpyro.sample("var_trend", dist.LogNormal(0, 1))
    ls_trend = numpyro.sample("ls_trend", dist.LogNormal(2, 1))
    ss_trend = Matern52(var_trend, ls_trend).to_state_space(t)

    # Short-range / abrupt component (Matérn-1/2 = Ornstein-Uhlenbeck)
    var_fast = numpyro.sample("var_fast", dist.LogNormal(-1, 1))
    ls_fast = numpyro.sample("ls_fast", dist.LogNormal(-2, 1))
    ss_fast = Matern12(var_fast, ls_fast).to_state_space(t)

    # Sum of state-space models = block-diagonal concatenation
    ss_combined = stack_state_space(ss_trend, ss_fast)
    # State dim = 3 (Matérn-5/2) + 1 (Matérn-1/2) = 4
    # H_combined = [H_trend, H_fast]  — sums the two components

    solver = KalmanSolver()
    lml = solver.log_marginal_likelihood(ss_combined, y, noise_var)
    numpyro.factor("gp_lml", lml)
```

### The stack_state_space utility

```python
def stack_state_space(ss1, ss2):
    """Concatenate two independent LTI systems into a block-diagonal system.

    State: x = [x1; x2],  dims S1 + S2
    Transition: A = blkdiag(A1, A2)
    Process noise: Q = blkdiag(Q1, Q2)
    Emission: H = [H1, H2]  (sums the outputs)
    """
    S1, S2 = ss1.A.shape[0], ss2.A.shape[0]
    A = jax.scipy.linalg.block_diag(ss1.A, ss2.A)
    Q = jax.scipy.linalg.block_diag(ss1.Q, ss2.Q)
    H = jnp.concatenate([ss1.H, ss2.H], axis=-1)
    x0 = jnp.concatenate([ss1.x0, ss2.x0])
    P0 = jax.scipy.linalg.block_diag(ss1.P0, ss2.P0)
    return StateSpaceRep(A=A, Q=Q, H=H, x0=x0, P0=P0, dt=ss1.dt)
```

### Decomposition

After inference, run the smoother and extract each component:

```python
smooth = solver.filter_and_smooth(ss_combined, y, noise_var)
# smooth.states: (N, 4) — first 3 dims are trend, last 1 is fast
f_trend = smooth.states[:, :3] @ ss_trend.H.T  # (N,)
f_fast = smooth.states[:, 3:] @ ss_fast.H.T    # (N,)
```

---

## 5. Latent Temporal GP in a BHM

**When**: The GP latent function feeds into a non-Gaussian likelihood
as part of a larger hierarchical model, and you want to use SVI with
a structured guide.

### Model

```python
def model_temporal_bhm(t, y_counts, covariates):
    """
    Poisson counts with a GP temporal trend + linear covariate effect.

    Parameters
    ----------
    t : (N,)
    y_counts : (N,)
    covariates : (N, P)
    """
    # Linear covariate effect
    beta = numpyro.sample("beta", dist.Normal(0, 1).expand([covariates.shape[1]]).to_event(1))
    linear_effect = covariates @ beta   # (N,)

    # GP temporal trend (explicit latent)
    variance = numpyro.sample("variance", dist.LogNormal(0, 1))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(1, 1))
    ss = Matern32(variance, lengthscale).to_state_space(t)

    f = numpyro.sample("f", MarkovGPPrior(ss))   # (N,)

    # Combine
    log_rate = f + linear_effect
    numpyro.sample("y", dist.Poisson(jnp.exp(log_rate)), obs=y_counts)
```

### MarkovGPPrior as a Distribution

```python
class MarkovGPPrior(numpyro.distributions.Distribution):
    """
    GP prior in state-space form.  log_prob uses Kalman filter.
    sample uses forward simulation through the state-space model.
    """

    def __init__(self, ss: StateSpaceRep):
        self.ss = ss
        self.N = ss.dt.shape[0] + 1
        super().__init__(event_shape=(self.N,))

    def sample(self, key, sample_shape=()):
        """Forward simulation: x_{n+1} = A x_n + q_n, f_n = H x_n."""
        def step(carry, key_n):
            x = carry
            q = jax.random.multivariate_normal(key_n, jnp.zeros_like(x), self.ss.Q)
            x_next = self.ss.A @ x + q
            f = (self.ss.H @ x_next).squeeze()
            return x_next, f

        x0 = jax.random.multivariate_normal(key, self.ss.x0, self.ss.P0)
        f0 = (self.ss.H @ x0).squeeze()
        keys = jax.random.split(key, self.N - 1)
        _, fs_rest = jax.lax.scan(step, x0, keys)
        return jnp.concatenate([f0[None], fs_rest])

    def log_prob(self, f):
        """Log probability via Kalman filter on f (zero noise)."""
        # Treat f as noiseless observations of the state-space model
        solver = KalmanSolver()
        return solver.log_marginal_likelihood(self.ss, f, noise_var=1e-6)
```

### Guide (KalmanGuide)

The optimal Gaussian approximation to q(f) for a Markov GP is the
RTS smoother posterior.  The KalmanGuide exploits this:

```python
class KalmanGuide:
    """
    Structured guide for MarkovGPPrior sites.

    Instead of learning N free variational parameters (like WhitenedMeanFieldGuide),
    learns pseudo-observations (ỹ, R̃) and runs the Kalman smoother to
    get q(f).

    This is exactly the Bayes-Newton approach: the guide IS the
    Kalman smoother with learnable sites.

    Variational Parameters
    ----------------------
    {site_name}_pseudo_y : (N,)     pseudo-observations
    {site_name}_pseudo_log_R : (N,) log pseudo-noise (unconstrained)
    """

    def __init__(self, ss: StateSpaceRep):
        self.ss = ss
        self.N = ss.dt.shape[0] + 1

    def __call__(self, site_name, *args, **kwargs):
        import numpyro

        pseudo_y = numpyro.param(f"{site_name}_pseudo_y", jnp.zeros(self.N))
        pseudo_log_R = numpyro.param(
            f"{site_name}_pseudo_log_R", jnp.zeros(self.N)
        )
        pseudo_R = jax.nn.softplus(pseudo_log_R) + 1e-6

        # Run Kalman smoother with pseudo-observations
        solver = KalmanSolver()
        smooth = solver.filter_and_smooth(self.ss, pseudo_y, pseudo_R)

        # Sample f from the smoother posterior
        # (reparameterised: f = mean + L @ eps, eps ~ N(0,I))
        f_mean = smooth.means
        f_std = jnp.sqrt(smooth.variances)

        eps = numpyro.sample(
            f"{site_name}_eps",
            dist.Normal(jnp.zeros(self.N), jnp.ones(self.N)).to_event(1),
            infer={"is_auxiliary": True},
        )
        f = f_mean + f_std * eps

        numpyro.deterministic(site_name, f)
```

### Composed Guide + SVI

```python
guide = ComposedGuide(
    gp_guides={"f": KalmanGuide(ss_init)},
    model=model_temporal_bhm,
    auto_guide_cls=AutoNormal,   # handles beta, variance, lengthscale
)

svi = SVI(model_temporal_bhm, guide, Adam(1e-3), Trace_ELBO())
```

### Why KalmanGuide over WhitenedMeanFieldGuide

- KalmanGuide has 2N parameters (pseudo_y + pseudo_R) but produces a
  **Markov-structured** q(f) — the smoothed posterior respects the
  temporal correlation structure.
- WhitenedMeanFieldGuide has 2N parameters (mean + std in whitened space)
  but q(f) has unstructured diagonal covariance in the whitened space,
  which maps to a dense but poorly structured covariance in function space.
- The KalmanGuide's q(f) is the exact posterior if the true likelihood
  were Gaussian with noise ỹ, R̃ — so the variational family is rich.

---

## 6. Spatiotemporal GP with Kronecker × Kalman

**When**: Data on a spatial grid observed over time.
Spatial covariance is dense (Cholesky); temporal covariance uses Kalman.

### Model

```python
def model_spatiotemporal(X_space, t, Y_grid):
    """
    Parameters
    ----------
    X_space : (S, D)   spatial locations
    t : (T,)           time points
    Y_grid : (S, T)    observations
    """
    S, T = Y_grid.shape

    # Spatial kernel (dense)
    var_s = numpyro.sample("var_s", dist.LogNormal(0, 1))
    ls_s = numpyro.sample("ls_s", dist.LogNormal(0, 1))
    K_space = RBF(var_s, ls_s)(X_space).matrix            # (S, S)
    L_space = jnp.linalg.cholesky(K_space + 1e-5 * jnp.eye(S))

    # Temporal kernel (state-space)
    var_t = numpyro.sample("var_t", dist.LogNormal(0, 1))
    ls_t = numpyro.sample("ls_t", dist.LogNormal(0, 1))
    ss_time = Matern32(var_t, ls_t).to_state_space(t)

    noise_var = numpyro.sample("noise_var", dist.LogNormal(-2, 1))

    # Kalman filter over time, with S-dimensional observations at each step
    # The spatial covariance enters as a "multi-output" state-space model
    # State: x_t ∈ ℝ^{S × state_dim}
    # This is the approach of Hamelijnck et al. (NeurIPS 2021)

    def spatiotemporal_step(carry, obs_t):
        """Kalman step with spatial correlation."""
        X_state, P_state = carry
        y_t = obs_t                                     # (S,)

        # Predict (Kronecker structure: A ⊗ I_S for state, etc.)
        X_pred = ss_time.A @ X_state                    # (state_dim, S)
        # P_pred has Kronecker structure but we work with the factors
        ...
        return (X_new, P_new), ll_t

    # ... scan over time
    _, lls = jax.lax.scan(spatiotemporal_step, init, Y_grid.T)
    numpyro.factor("gp_lml", jnp.sum(lls))
```

### Notes

- The Kronecker structure means we never form the (ST × ST) matrix.
- Spatial operations are O(S³) per time step; temporal operations are O(T)
  via the scan.  Total: O(TS³ + S³) vs O((ST)³) for dense.
- For very large S, combine with inducing points in space.

---

## 7. Online / Streaming GP with Filter-Only Mode

**When**: Observations arrive one at a time; need predictions in real-time
without storing the full history.

### Model (filter-only, no smoother)

```python
class OnlineTemporalGP:
    """Streaming GP predictions using filter-only mode."""

    def __init__(self, kernel, noise_var):
        self.ss = kernel.to_state_space_continuous()
        self.noise_var = noise_var
        # Filter state
        self.x = self.ss.x0
        self.P = self.ss.P0
        self.t_prev = None

    def update(self, t_new, y_new):
        """Incorporate a new observation and return updated prediction."""
        if self.t_prev is not None:
            dt = t_new - self.t_prev
            A = self.ss.discretise_A(dt)
            Q = self.ss.discretise_Q(dt)
            # Predict
            self.x = A @ self.x
            self.P = A @ self.P @ A.T + Q

        # Update
        f_pred = self.ss.H @ self.x
        S = self.ss.H @ self.P @ self.ss.H.T + self.noise_var
        K = self.P @ self.ss.H.T / S
        v = y_new - f_pred
        self.x = self.x + K * v
        self.P = self.P - K * S * K.T
        self.t_prev = t_new

        return f_pred, S  # predictive mean and variance

    def predict_ahead(self, t_future):
        """Predict at a future time without incorporating an observation."""
        dt = t_future - self.t_prev
        A = self.ss.discretise_A(dt)
        Q = self.ss.discretise_Q(dt)
        x_pred = A @ self.x
        P_pred = A @ self.P @ A.T + Q
        f_pred = self.ss.H @ x_pred
        var_pred = self.ss.H @ P_pred @ self.ss.H.T + self.noise_var
        return f_pred, var_pred
```

---

## 8. Parallel Kalman on GPU for Long Series

**When**: N > 10⁵ observations, GPU available, want to exploit parallelism.

### Model

```python
def model_parallel_temporal(t, y):
    variance = numpyro.sample("variance", dist.LogNormal(0, 1))
    lengthscale = numpyro.sample("lengthscale", dist.LogNormal(0, 1))
    noise_var = numpyro.sample("noise_var", dist.LogNormal(-2, 1))

    ss = Matern32(variance, lengthscale).to_state_space(t)

    # Parallel Kalman via associative_scan
    solver = ParallelKalmanSolver()
    lml = solver.log_marginal_likelihood(ss, y, noise_var)
    numpyro.factor("gp_lml", lml)
```

### Internal: associative_scan implementation

```python
def parallel_kalman_filter(ss, y, noise_var):
    """
    Build Kalman elements and reduce via associative_scan.

    Each element e_n = (A_n, b_n, C_n) encodes one predict-update step.
    The associative operation ⊕ combines two elements:
        (A2, b2, C2) ⊕ (A1, b1, C1) = (A2 A1, A2 b1 + b2, A2 C1 A2.T + C2)

    jax.lax.associative_scan computes all prefixes in O(log N) parallel steps.
    """
    S = ss.A.shape[0]
    N = y.shape[0]

    # Build per-observation elements
    def make_element(yn):
        # ... convert (A, Q, H, R, yn) into (A_elem, b_elem, C_elem)
        # See Särkkä & García-Fernández (2020) for the derivation
        ...

    elements = jax.vmap(make_element)(y)   # (N, S, S), (N, S), (N, S, S)

    def combine(e2, e1):
        A2, b2, C2 = e2
        A1, b1, C1 = e1
        return (A2 @ A1, A2 @ b1 + b2, A2 @ C1 @ A2.T + C2)

    scanned = jax.lax.associative_scan(combine, elements)
    # scanned contains filter states at all times

    # Extract log-likelihood from innovations (computed during element construction)
    return jnp.sum(log_liks)
```

### Performance

| N | Sequential Kalman (CPU) | Parallel Kalman (GPU) |
|---|:-----------------------:|:---------------------:|
| 10³ | ~1ms | ~2ms (overhead) |
| 10⁵ | ~100ms | ~10ms |
| 10⁷ | ~10s | ~100ms |

The crossover is around N ≈ 10⁴ — below that, sequential is faster
due to lower overhead.

---

## 9. Non-LTI Temporal Model with numpyro.scan

**When**: The transition dynamics are not time-invariant — e.g., the
lengthscale changes over time, or the state-space model has nonlinear
transitions.  Requires `numpyro.contrib.control_flow.scan` because
each time step has its own numpyro.sample sites.

### Model

```python
from numpyro.contrib.control_flow import scan as numpyro_scan

def model_nonstationary_temporal(t, y):
    """
    GP with time-varying lengthscale (non-stationary).
    The lengthscale itself follows a random walk.
    """
    # Hyperpriors
    var_f = numpyro.sample("var_f", dist.LogNormal(0, 1))
    var_ls = numpyro.sample("var_ls", dist.LogNormal(-2, 1))  # RW innovation variance
    noise_var = numpyro.sample("noise_var", dist.LogNormal(-2, 1))
    ls_init = numpyro.sample("ls_init", dist.LogNormal(0, 1))

    N = t.shape[0]

    def transition(carry, n):
        log_ls_prev, x_prev = carry

        # Lengthscale random walk (in log space)
        log_ls = numpyro.sample(
            "log_ls",
            dist.Normal(log_ls_prev, jnp.sqrt(var_ls)),
        )
        ls = jnp.exp(log_ls)

        # Matérn-3/2 transition with CURRENT lengthscale
        dt = t[n] - t[n - 1]
        A, Q = matern32_discretise(var_f, ls, dt)

        # State transition
        x = numpyro.sample("x", dist.MultivariateNormal(A @ x_prev, Q))

        # Observation
        f = x[0]   # H = [1, 0] for Matérn-3/2
        numpyro.sample("y", dist.Normal(f, jnp.sqrt(noise_var)), obs=y[n])

        return (log_ls, x), f

    # Initial state
    x0 = numpyro.sample("x0", dist.MultivariateNormal(jnp.zeros(2), var_f * jnp.eye(2)))
    numpyro.sample("y_0", dist.Normal(x0[0], jnp.sqrt(noise_var)), obs=y[0])

    init = (jnp.log(ls_init), x0)
    # numpyro.scan — required because numpyro.sample is inside the loop
    _, fs = numpyro_scan(transition, init, jnp.arange(1, N))
```

### Guide

```python
# Each (log_ls_n, x_n) is a latent variable — total 3N dims
# A structured guide is essential
guide = ComposedGuide(
    gp_guides={
        # KalmanGuide won't work here (non-LTI)
        # Use a mean-field guide over the scan variables
        # numpyro_scan produces sites "log_ls" and "x" with plate dim
    },
    model=model_nonstationary_temporal,
    auto_guide_cls=AutoNormal,  # mean-field over all sites
)
```

### Notes

- `numpyro.scan` is slower than `jax.lax.scan` because it must trace
  through the effect system, but it's necessary when sample sites are
  inside the loop.
- For this non-stationary model, MCMC is very expensive (3N latent dims).
  SVI with AutoNormal is the pragmatic choice.
- If the lengthscale changes slowly, consider discretising it to a few
  epochs and using LTI within each epoch (piecewise stationary model).

---

## 10. Temporal Extremes: GEV with Time-Varying Parameters

**When**: Block maxima over time with smoothly varying GEV parameters.
Combines temporal GP priors with extreme value likelihood.

### Model

```python
def model_temporal_extremes(t, z_maxima):
    """
    Block maxima z_t ~ GEV(μ(t), σ(t), ξ(t)) where the parameters
    are temporal GPs.

    Parameters
    ----------
    t : (T,) sorted time indices (e.g., years)
    z_maxima : (T,) observed block maxima
    """
    solver = KalmanSolver()

    # GP prior on location μ(t) — smooth, Matérn-5/2
    var_mu = numpyro.sample("var_mu", dist.LogNormal(0, 1))
    ls_mu = numpyro.sample("ls_mu", dist.LogNormal(2, 1))
    ss_mu = Matern52(var_mu, ls_mu).to_state_space(t)
    mu_intercept = numpyro.sample("mu_intercept", dist.Normal(0, 10))
    f_mu = numpyro.sample("f_mu", MarkovGPPrior(ss_mu))
    mu = mu_intercept + f_mu

    # GP prior on log-scale log σ(t)
    var_sig = numpyro.sample("var_sig", dist.LogNormal(-1, 1))
    ls_sig = numpyro.sample("ls_sig", dist.LogNormal(2, 1))
    ss_sig = Matern32(var_sig, ls_sig).to_state_space(t)
    sig_intercept = numpyro.sample("sig_intercept", dist.Normal(0, 2))
    f_log_sigma = numpyro.sample("f_log_sigma", MarkovGPPrior(ss_sig))
    sigma = jnp.exp(sig_intercept + f_log_sigma)

    # GP prior on shape ξ(t) — weakly varying
    var_xi = numpyro.sample("var_xi", dist.LogNormal(-2, 1))
    ls_xi = numpyro.sample("ls_xi", dist.LogNormal(2, 1))
    ss_xi = Matern32(var_xi, ls_xi).to_state_space(t)
    xi_intercept = numpyro.sample("xi_intercept", dist.Normal(0, 0.3))
    f_xi = numpyro.sample("f_xi", MarkovGPPrior(ss_xi))
    xi = xi_intercept + f_xi

    # GEV likelihood
    numpyro.sample("z", GEV(loc=mu, scale=sigma, shape=xi), obs=z_maxima)
```

### Guide

```python
# Three temporal GP latents — each gets a KalmanGuide
guide = ComposedGuide(
    gp_guides={
        "f_mu":        KalmanGuide(ss_mu_init),
        "f_log_sigma": KalmanGuide(ss_sig_init),
        "f_xi":        KalmanGuide(ss_xi_init),
    },
    model=model_temporal_extremes,
    auto_guide_cls=AutoNormal,
)
```

### Inference

```python
# SVI (recommended — 3T latent dims for the GPs)
svi = SVI(model_temporal_extremes, guide, Adam(5e-4), Trace_ELBO())
for step in range(5000):
    svi_state, loss = svi.update(svi_state, t, z_maxima)

# MCMC (feasible for T < ~100 with long warmup)
mcmc = MCMC(NUTS(model_temporal_extremes, max_tree_depth=12),
            num_warmup=2000, num_samples=2000)
mcmc.run(rng_key, t, z_maxima)
```

### Comparison with Non-Temporal (Spatial) Extremes

| Aspect | Spatial (Pattern 3, see moments examples) | Temporal (this pattern) |
|--------|--------------------------------------|------------------------|
| GP representation | Dense K(X,X) | State-space, Kalman |
| Solver | CholeskySolver O(S³) | KalmanSolver O(TS²) |
| Guide | WhitenedMeanFieldGuide | KalmanGuide |
| Scales to | S ~ 1000 locations | T ~ 10⁵ time points |
| Combined | Spatiotemporal extremes: Kronecker × Kalman |

---

## Inference Decision Guide (Temporal)

| Scenario | Strategy | Solver | NumPyro path |
|----------|----------|--------|--------------|
| Gaussian lik, learn hyperparams | Conjugate | KalmanSolver | MCMC / SVI on θ |
| Non-Gaussian, deterministic approx | Laplace or EP | KalmanSolver | MCMC on θ, Laplace/EP on f |
| Non-Gaussian, full Bayesian | — | — | MCMC on (θ, f) or SVI + KalmanGuide |
| Very long series (N > 10⁵) | Conjugate | ParallelKalmanSolver | MCMC on θ |
| Non-stationary dynamics | — | numpyro.scan | SVI + AutoNormal |
| Spatiotemporal | Conjugate | Kronecker × Kalman | MCMC / SVI on θ |