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
