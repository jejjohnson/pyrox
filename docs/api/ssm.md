# SSM API

`pyrox-ssm` is the package whose inference recurses in time: [pyrox-gp](gp.md) solves over all N points at once, [pyrox-lgm](lgm.md) solves one sparse precision, and `pyrox_ssm` runs a filter and smoother over the time axis. It holds the Markov-GP track (moved from pyrox-gp; the old `pyrox_gp` names raise an `ImportError` naming their new home) and re-exports gaussx's SDE kernels. Its likelihoods are any `pyrox.likelihoods.Likelihood` (the pyrox-gp ones work unchanged) and its sparse guides any `pyrox.likelihoods.Guide` with `predict` and `kl_divergence`.

## State-space (SDE) kernels

Stationary 1-D kernels expressed as linear time-invariant SDEs. Once in
state-space form, GP inference on a 1-D grid reduces to Kalman filtering
in ``O(N d^3)`` instead of ``O(N^3)`` Cholesky. The protocol exposes
``sde_params() -> (F, L, H, Q_c, P_inf)`` and ``discretise(dt) -> (A_k, Q_k)``
for downstream Kalman / RTS use.

```python
import jax.numpy as jnp
from pyrox_ssm import (
    ConstantSDE, CosineSDE, MaternSDE, PeriodicSDE,
    ProductSDE, QuasiPeriodicSDE, SumSDE,
)

# Primitive kernels
matern = MaternSDE(variance=1.0, lengthscale=0.5, order=1)  # nu = 3/2
cos    = CosineSDE(variance=1.0, frequency=2.0)
const  = ConstantSDE(variance=0.3)
per    = PeriodicSDE(variance=1.0, lengthscale=1.0, period=2.0, n_harmonics=7)

# Composition: trend + offset
trend = SumSDE((matern, const))                   # state dim = 2 + 1 = 3

# Composition: damped oscillation (Matern x Cosine)
damped = ProductSDE(matern, cos)                  # state dim = 2 * 2 = 4

# Quasi-periodic (Matern x Periodic) — convenience wrapper around ProductSDE
qp = QuasiPeriodicSDE(matern, per)                # state dim = 2 * 15 = 30
```

### Non-stationary priors

Every kernel above is stationary: the process is started in its
stationary distribution, so the filter seeds from `P_inf` and
`sde_autocovariance` reports `K(tau)`.

[`IntegratedWienerSDE`][pyrox_ssm.IntegratedWienerSDE] is not. At
`order=1` it is the *local linear trend* — smooth, and linear unless the
data push back, with no lengthscale to choose — and its marginal
variance grows without bound, so there is no `P_inf` at all. It reports
`stationary = False` and `sde_params().P_inf = None`, and the filter
starts from an explicit initial covariance instead:

```python
import jax.numpy as jnp
from pyrox_ssm import IntegratedWienerSDE, MarkovGPPrior

trend = IntegratedWienerSDE(diffusion=1e-4)       # state = [level, slope]
prior = MarkovGPPrior(trend, times)               # diffuse default P_0
# ... or state what is known before the first observation:
prior = MarkovGPPrior(trend, times, init_cov=jnp.diag(jnp.array([1e2, 1.0])))
```

Filtering, smoothing, prediction and the site-based non-Gaussian
strategies all work; the *dense* paths ([`log_prob`][pyrox_ssm.MarkovGPPrior.log_prob]
and [`markov_gp_sample`][pyrox_ssm.markov_gp_sample]) raise, because
`K_ij = H exp(F |t_i - t_j|) P_inf H^T` is a function of the lag alone,
while a non-stationary covariance depends on both times.

A `SumSDE` may mix the two — a trend plus a stationary seasonal — and
starts from the block-diagonal of its components' initial covariances.

::: pyrox_ssm.SDEKernel
::: pyrox_ssm.SDEParams
::: pyrox_ssm.MaternSDE
::: pyrox_ssm.ConstantSDE
::: pyrox_ssm.CosineSDE
::: pyrox_ssm.PeriodicSDE
::: pyrox_ssm.IntegratedWienerSDE
::: pyrox_ssm.SumSDE
::: pyrox_ssm.ProductSDE
::: pyrox_ssm.QuasiPeriodicSDE

## Linear-Gaussian state-space models

`LinearGaussianSSMPrior` is the general Gaussian state-space model, given directly by its transitions `A_k`, observation matrices `H_k`, process and observation noise covariances `Q_k` and `R_k`, an initial state `N(m_0, P_0)` and an optional known observation offset `b_k`, so that `x_k = A_k x_{k-1} + eta_k` and `y_k = H_k x_k + b_k + eps_k` for `k = 1, …, T`. Use it when the dynamics are known as matrices rather than as a GP kernel: a local-level model, a multi-output state, a discretised physical model whose matrices carry priors. It follows the `gaussx.kalman_filter` convention of predicting before every update, so `y[0]` is scored against `A_1 x_0`; set `transition[0] = I` and `process_noise[0] = 0` to observe the initial state itself.

Observations are `(T, M)` (a flat `(T,)` series is accepted when `M = 1`), and every method takes a `mask=` either per step, `(T,)`, or per entry, `(T, M)`; a masked entry is marginalised out exactly and may hold `NaN`. `log_marginal`, `smooth` and `condition` give the prediction-error log marginal and the RTS-smoothed state moments, and `lgssm_factor` adds the log marginal to a NumPyro model:

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pyrox_ssm as pss

def local_level(y, mask):
    T = y.shape[0]
    q = numpyro.sample("q", dist.LogNormal(-2.0, 1.0))
    r = numpyro.sample("r", dist.LogNormal(-1.0, 1.0))
    prior = pss.LinearGaussianSSMPrior(
        transition=jnp.ones((T, 1, 1)), obs_model=jnp.eye(1),
        process_noise=q * jnp.ones((T, 1, 1)), obs_noise=r * jnp.eye(1),
        init_mean=jnp.zeros(1), init_cov=10.0 * jnp.eye(1),
    )
    pss.lgssm_factor("y", prior, y, mask=mask)
```

`MarkovGPPrior` is its kernel-built special case: [`as_lgssm(noise_var)`][pyrox_ssm.MarkovGPPrior.as_lgssm] returns the kernel's discretised transitions over the time grid (with a leading zero step, so `A_1 = I` and `Q_1 = 0`), `H`, `R_k = noise_var + obs_noise_floor`, `m_0 = 0`, `P_0 = initial_covariance()` and the offset `b_k = mean_fn(t_k)`. `MarkovGPPrior`'s filtering and smoothing run through this model, so the two agree bit for bit.

::: pyrox_ssm.LinearGaussianSSMPrior
::: pyrox_ssm.ConditionedLinearGaussianSSM
::: pyrox_ssm.lgssm_factor

## Markov GP — Kalman / RTS workflow

`MarkovGPPrior` consumes any [`SDEKernel`][pyrox_ssm.SDEKernel] over a sorted
1-D grid and gives `O(N d^3)` marginal likelihood (forward Kalman filter)
and posterior smoothing (backward RTS), where `d` is the SDE state
dimension. Use it for temporal GP regression / forecasting when the
training grid lives on a single time axis. Predictions at arbitrary
test times — including forecasting, backcasting, and within-window
interpolation — re-run the filter+smoother over the merged grid with the
test points masked out of the update step.

```python
import jax.numpy as jnp
from pyrox_ssm import MaternSDE, MarkovGPPrior, markov_gp_factor

times = jnp.linspace(0.0, 5.0, 200)
y     = jnp.sin(times) + 0.05 * jnp.cos(7.0 * times)

prior = MarkovGPPrior(
    MaternSDE(variance=1.0, lengthscale=0.5, order=1),  # Matern-3/2
    times,
)
log_marg = prior.log_marginal(y, jnp.asarray(0.01))     # Kalman forward
cond     = prior.condition(y, jnp.asarray(0.01))        # filter + RTS smoother
mean, var = cond.predict(jnp.linspace(-0.5, 6.0, 50))   # arbitrary test times
```

Inside a NumPyro model, swap `gp_factor` for `markov_gp_factor`:

```python
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from pyrox_ssm import MarkovGPPrior, MaternSDE, markov_gp_factor

def temporal_model(times, y):
    sigma2 = numpyro.sample("variance",  dist.LogNormal(0.0, 1.0))
    ell    = numpyro.sample("lengthscale", dist.LogNormal(0.0, 1.0))
    sde    = MaternSDE(variance=sigma2, lengthscale=ell, order=1)
    prior  = MarkovGPPrior(sde, times)
    markov_gp_factor("obs", prior, y, jnp.array(0.01))
```

For non-Gaussian likelihoods on the Markov path, see the
[Markov non-Gaussian strategies](#non-gaussian-inference-markov) below;
for inducing-grid scalability, the [sparse Markov GP](#sparse-markov-gp).

::: pyrox_ssm.MarkovGPPrior
::: pyrox_ssm.ConditionedMarkovGP
::: pyrox_ssm.markov_gp_factor
::: pyrox_ssm.markov_gp_sample

### Ragged sites

Every method that reads observations (`filter`, `smooth`, `log_marginal`, `condition`, `condition_nongauss`, `markov_gp_factor`, each non-Gaussian strategy's `fit`, `sparse_markov_elbo` and `sparse_markov_factor`) takes an optional boolean `mask=` of the observations' shape. A masked step runs the Kalman predict only, so it adds nothing to the log marginal, and its observation is replaced before any arithmetic, so a `NaN` placeholder there reaches neither the value nor the gradient. `mask=None` is the unmasked result, bit for bit.

Sites observed on different days become one vmapped factor with `pad_sites`, which pads each series after its last observation (at its median spacing, so the padded times stay strictly increasing) and returns the mask. Padding after the last observation adds predict-only steps, so each site's log marginal is unchanged:

```python
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pyrox_ssm as pss

t, y, mask = pss.pad_sites(times_per_site, y_per_site)           # (S, N) each

def site_log_marginal(ell_s, t_s, y_s, m_s):
    sde = pss.MaternSDE(variance=1.0, lengthscale=ell_s, order=1)
    return pss.MarkovGPPrior(sde, t_s).log_marginal(y_s, jnp.asarray(0.01), mask=m_s)

def pooled_model(t, y, mask):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    with numpyro.plate("site", t.shape[0]):
        ell = numpyro.sample("lengthscale", dist.LogNormal(mu, 0.5))  # (S,)
    numpyro.factor("y", jax.vmap(site_log_marginal)(ell, t, y, mask).sum())
```

::: pyrox_ssm.pad_sites

## Normalizing Kalman Filter

`NormalizingKalmanPrior` wraps a `gaussx.LGSSM` (or `gaussx.MaskedLGSSM`)
base and an optional per-timestep observation warp into the same
model surface: exact collapsed marginal via `log_marginal`, NumPyro
registration via `normalizing_kalman_factor`, and observation-space
predictive moments via `predict` (RTS smoothing followed by a
Gauss-Hermite pushforward through the warp — the mean is `E[G(z)]`,
not `G(E[z])`).

`pyrox_ssm.flows` needs the `flows` extra (`pip install 'pyrox-ssm[flows]'`),
which brings `flowjax` and, for a non-`None` `warp`, `gauss_flows`; the
unwarped model (`warp=None`) needs no `gauss_flows` at runtime. Because the warp acts on observations,
the log-det term is independent of the latent state, the Kalman
recursion stays exact, and none of the non-Gaussian Markov strategies
below are involved.

```python
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from gaussx import LGSSM
from pyrox_ssm.flows import NormalizingKalmanPrior, normalizing_kalman_factor

def nkf_model(y, warp=None, mask=None):
    T, M = y.shape
    log_q = numpyro.sample("log_q", dist.Normal(0.0, 1.0).expand([M]).to_event(1))
    log_r = numpyro.sample("log_r", dist.Normal(0.0, 1.0).expand([M]).to_event(1))
    base = LGSSM(0.9 * jnp.eye(M), jnp.eye(M),
                 jnp.diag(jnp.exp(jnp.asarray(log_q))),
                 jnp.diag(jnp.exp(jnp.asarray(log_r))),
                 jnp.zeros(M), jnp.eye(M), n_steps=T)
    prior = NormalizingKalmanPrior(base, warp=warp)
    normalizing_kalman_factor("nkf", prior, y, mask)
```

::: pyrox_ssm.flows.NormalizingKalmanPrior
::: pyrox_ssm.flows.normalizing_kalman_factor

## Non-Gaussian inference (Markov)

The Markov-aware counterparts of pyrox-gp's site-based strategies
([GP API](gp.md#non-gaussian-inference-strategies)): same
diagonal-site math, but the global posterior recomputation runs through
the Kalman filter / RTS smoother in `O(N d^3)` instead of a dense
`O(N^3)` solve. Each `fit(prior, likelihood, y)` returns a
`NonGaussConditionedMarkovGP` with the same `predict` API as the
Gaussian-likelihood `ConditionedMarkovGP`. `PosteriorLinearizationMarkov` and
`ExpectationPropagationMarkov` take the same `integrator` as their dense twins;
expectation propagation accepts only a `gaussx.GaussHermiteIntegrator`.

::: pyrox_ssm.LaplaceMarkovInference
::: pyrox_ssm.GaussNewtonMarkovInference
::: pyrox_ssm.PosteriorLinearizationMarkov
::: pyrox_ssm.ExpectationPropagationMarkov
::: pyrox_ssm.NonGaussConditionedMarkovGP

## Sparse Markov GP

Sparse variational GP over an SDE kernel and an inducing *time* grid:
the variational family lives on the inducing times while predictions
exploit the Markov structure between them.

::: pyrox_ssm.SparseMarkovGPPrior
::: pyrox_ssm.SparseConditionedMarkovGP
::: pyrox_ssm.sparse_markov_elbo
::: pyrox_ssm.sparse_markov_factor
