# Ecosystem interop: pipekit & NumPyro

This page codifies the outcome of the pipekit/NumPyro reuse audit
(epic [#42](https://github.com/jejjohnson/xtremax/issues/42)) so future
contributions keep the dependency direction intact.

## The stance: xtremax stays pipekit-free

**xtremax never imports pipekit.** This is pipekit's own philosophy:
algorithm and domain libraries plug into its pipelines by *structurally*
satisfying its runtime-checkable `Protocol`s — duck typing checked with
`isinstance` at runtime — not by depending on it. The dependency arrow
points one way:

```
workflow repos / notebooks  ──imports──▶  pipekit  and  xtremax
xtremax                     ──imports──▶  jax, numpyro, xarray, …   (never pipekit)
```

Anything that composes xtremax functions into pipekit graphs
(`pipekit.Lambda`, `Sequential`, `AssertShape`, `Quarantine`, …) lives in
downstream workflow repositories and example notebooks — never in
`src/xtremax/`.

## Protocol conformance table

Verified with runtime `isinstance` checks against pipekit's
runtime-checkable Protocols:

| pipekit Protocol | Required members | xtremax class | Status |
|---|---|---|---|
| `pipekit.Predictor` | `predict(x)` | `XarrayQuantileRegressor` | ✅ conforms with zero changes — the pattern to emulate |
| `pipekit_cycle.ObservationNoise` | `covariance()`, `sample(key, shape)` | `GeneralizedExtremeValueDistribution`, `GeneralizedParetoDistribution` | ✅ conforms — `covariance()` added for this seam; `sample(key, sample_shape)` was already positionally compatible |
| `pipekit.FittableTransformer` | `fit(...)`, `transform(...)` | — | no candidate today; wrap downstream if needed |

### The `ObservationNoise` seam

`pipekit_cycle` data-assimilation cycles accept any observation-error
model exposing `covariance()` and `sample(key, shape)`. The EVT
distributions satisfy this structurally, so DA experiments can use
realistic heavy-tailed observation noise:

```python
import jax
from xtremax.distributions import GeneralizedParetoDistribution

noise = GeneralizedParetoDistribution(scale=0.5, concentration=0.2)
noise.covariance()                       # marginal variance (diagonal view)
noise.sample(jax.random.PRNGKey(0), (64,))        # error draws
noise.sample(jax.random.PRNGKey(0), shape=(64,))  # protocol keyword spelling
```

`sample` keeps NumPyro's `sample_shape` parameter but also accepts the
protocol's `shape` spelling as a keyword-only alias, so both
NumPyro-style and protocol-typed callers work. In both spellings the
argument is the **leading draw batch** (the protocol's "batch shape"),
prepended to the distribution's own batch shape: a heteroscedastic
noise model with `batch_shape == (d,)` asked for `shape=(n,)` returns
`(n, d)` — `n` error vectors of dimension `d`, which is what
perturbed-observation ensemble methods consume. It is *not* the full
output shape.

`covariance()` returns the marginal variance broadcast to the batch
shape — the errors are independent per batch element, so consuming
analysis steps interpret it as a diagonal covariance. Where the moment
does not exist (ξ ≥ 1/2), it is `+inf`; deterministic analyses that
require a finite covariance should restrict ξ accordingly, while
stochastic (perturbed-observation) methods only need `sample`.

### The `Predictor` pattern

`XarrayQuantileRegressor.predict(X)` matches `pipekit.Predictor`
exactly, without xtremax knowing pipekit exists. When adding new
model-like classes, prefer sklearn-style member names (`predict`, `fit`,
`transform`) so pipekit interop comes for free.

## NumPyro reuse

xtremax builds *on* NumPyro rather than beside it: distributions
subclass `numpyro.distributions.Distribution`, declare
`arg_constraints`/`support` so `biject_to` works inside NUTS/SVI, and
delegate wherever NumPyro already has the machinery
(`GumbelType1GEVD` subclasses `numpyro.distributions.Gumbel` directly).
New distribution code should follow the same rule: never re-implement
what NumPyro's `Distribution` base or an existing NumPyro distribution
already provides.

## Upstream gaps (tracked in pipekit)

Meaningful pipekit-train adoption for EVT fitting is blocked on
upstream features — NUTS kernel kwargs, an unsupervised `NumpyroTask`,
and a full-posterior predictive op — tracked at
[jejjohnson/pipekit#51](https://github.com/jejjohnson/pipekit/issues/51).
Nothing in xtremax should work around these gaps by importing pipekit.
