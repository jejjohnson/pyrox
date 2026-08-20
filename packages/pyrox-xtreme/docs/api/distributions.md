# Distributions

NumPyro-native extreme value distributions. Every class subclasses
`numpyro.distributions.Distribution`, so it plugs directly into MCMC / SVI /
`Predictive` and is differentiable end to end. Beyond `sample` / `log_prob`,
each carries the EVT toolkit: `cdf`, `icdf`, `survival_function`, `mean`,
`variance`, `entropy`, and `return_level`.

All of them delegate their math to the [primitives](primitives.md) — reach for
those directly if you want a density without a distribution object.

## Generalized Extreme Value

The distribution for **block maxima** (annual temperature maxima, seasonal
peak discharge). The concentration $\xi$ selects the tail type continuously,
so inference can learn which regime the data is in instead of you committing
up front — this is the default choice unless you have reason to fix $\xi$.

::: xtremax.distributions
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - GeneralizedExtremeValueDistribution

### Ragged records: masking the likelihood

Station networks rarely share a calendar. Annual maxima over $S$ stations and
$T$ years arrive as an $(S, T)$ grid with gaps where a station had no record
that year, and the likelihood should only see the observed station-years.

Use NumPyro's masking distribution — fill the gaps with any in-support value
(the per-station mean works) and let the mask discard them:

```python
import jax.numpy as jnp
import numpyro
from xtremax import GeneralizedExtremeValueDistribution as GEV

filled = jnp.where(mask, maxima, station_mean[:, None])  # (S, T)
gev = GEV(loc=mu[:, None], scale=sigma[:, None], concentration=xi[:, None])
numpyro.sample("obs", gev.mask(mask), obs=filled)
```

`numpyro.handlers.mask(mask=mask)` around a plain `sample` is equivalent, as
is summing `log_prob` yourself into a `numpyro.factor`:

```python
lp = gev.log_prob(filled)
numpyro.factor("obs", jnp.where(mask, lp, 0.0).sum())
```

All three score identically. Prefer the first: `factor` *does* register an
observed site, but an auxiliary one carrying `Unit(log_factor)` rather than the
GEV, so everything downstream sees a summed scalar where the other two keep the
distribution. `log_likelihood` returns one total per posterior draw instead of
the per-observation matrix WAIC and PSIS-LOO need, and `Predictive` has no
distribution left to draw the site from.

!!! note "Why the fill value does not bias the fit"

    Masked entries contribute neither density nor gradient, so what you write
    into the gaps cannot move the posterior. This holds even when the filler
    falls *outside* the support: the GEV endpoint is parameter-dependent
    ($1 + \xi(y-\mu)/\sigma > 0$), so a per-station mean can sit past a
    $\xi < 0$ upper bound at the parameters the sampler happens to be holding.

    Genuinely observed maxima are a different matter — one outside the support
    at a trial parameter draw scores $-\infty$, and `find_valid_initial_params`
    rejects that draw (it wants a finite potential *and* finite gradients) and
    tries another, up to 100 times. `nan` fails the same `isfinite` test, so the
    distinction is not which one gets retried but whether retrying can help: an
    out-of-support observation is $-\infty$ only at *some* parameters, so a later
    draw lands somewhere valid. A `nan` leaking out of a masked-out gap is `nan`
    at *every* parameter, and no number of retries recovers from that.

## Generalized Pareto

The distribution for **threshold exceedances** (peaks-over-threshold). Pair it
with the [extraction](extraction.md) threshold and declustering utilities:
POT uses more of the data than block maxima, at the cost of a threshold
choice.

::: xtremax.distributions
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - GeneralizedParetoDistribution

## Fixed-type GEV families

The three GEV types as standalone distributions, for when theory or prior
evidence pins the tail behaviour and you would rather not estimate $\xi$.
Fixing the type removes a parameter and usually tightens return-level
intervals — at the cost of an assumption the data can no longer overturn.

| Class | Type | Tail |
|-------|------|------|
| `GumbelType1GEVD` | I ($\xi = 0$) | Exponential |
| `FrechetType2GEVD` | II ($\xi > 0$) | Heavy polynomial, lower-bounded |
| `WeibullType3GEVD` | III ($\xi < 0$) | Bounded above |

::: xtremax.distributions
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - GumbelType1GEVD
        - FrechetType2GEVD
        - WeibullType3GEVD
