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
