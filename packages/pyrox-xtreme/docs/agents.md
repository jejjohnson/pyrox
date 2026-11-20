---
title: Building with agents
---

# Building with agents

xtremax exists so that an extreme value model is a NumPyro model built from
distributions that stay accurate and differentiable through the Gumbel
limit ξ → 0 and in the far tail, with point processes and xarray
extraction alongside. Coding agents tend to re-implement what they cannot
see — a GEV density with a `jnp.where` on ξ = 0, `icdf(1 - 1/T)` for a
return level, `1 - cdf` for a tail, a Poisson likelihood or a runs
declustering loop by hand — and each one loses accuracy, gradients or
float32 safety. xtremax ships three things that let them find it.

## The capability index

The [capability index](https://jejjohnson.github.io/xtremax/reference/capabilities/)
lists every public name in `xtremax.distributions`, `xtremax.primitives`,
`xtremax.extraction`, `xtremax.point_processes` (with its primitives,
operators and NumPyro wrappers) and `xtremax.simulations`, each once,
grouped by the module that defines it, with a one-line summary. It is
regenerated from the code and checked in the test suite, so it never
drifts.

## The Claude Code plugin

The repository is a Claude Code plugin marketplace. In any project:

```text
/plugin marketplace add jejjohnson/xtremax
/plugin install xtremax@xtremax
```

The plugin adds:

- **`extreme-value-analysis-with-xtremax`** (skill) — loads whenever a
  task fits a GEV or GPD, computes a return level, models exceedances or
  event times as a point process, or extracts block maxima or peaks over
  threshold: what lives where, the rules (distributions as NumPyro sites,
  initialise NUTS inside the data, scipy's sign convention, stable return
  levels, padded event buffers, explicit keys), a worked example, and a
  "don't write it — use xtremax" table.
- **`xtremax-reuse-reviewer`** (subagent) — a read-only check of a diff for
  extreme value code xtremax already provides, and for misuse.

## llms.txt

For other agents and tools, the docs site serves
[`llms.txt`](https://jejjohnson.github.io/xtremax/llms.txt): a curated map
of xtremax and its key pages.

## Rules for your project's `AGENTS.md`

Paste this into the agent instructions of a project that builds on
xtremax:

```markdown
## Extreme values: build on xtremax

This project uses xtremax for extreme value analysis (NumPyro EVT
distributions, stable return levels, point processes, xarray extraction).
Before writing a GEV / GPD density, a quantile or return level, a Poisson or
Hawkes likelihood, or block-maxima / threshold / declustering code, search
the capability index
(https://jejjohnson.github.io/xtremax/reference/capabilities/) or the
subpackages' `__all__`, and compose what exists:

- Likelihoods are `numpyro.sample(..., xtremax.GeneralizedExtremeValueDistribution(loc, scale, concentration), obs=y)`
  (or `GeneralizedParetoDistribution`); ragged records use `.mask(m)`.
- Return levels and tails come from `xtremax.primitives.gev_return_level` /
  `gpd_return_level` (or `d.return_level(T)` on a GEV or GPD) and
  `*_survival`, never `icdf(1 - 1/T)` or `1 - cdf`.
- Start NUTS with `init_strategy=init_to_median` and data-scale priors; the
  default init can strand a GEV fit at ξ < -1.
- scipy's `genextreme` takes `c = -ξ`.
- Block maxima, thresholds and declustering come from `xtremax.extraction`;
  event-time models from `xtremax.point_processes`.
- Pass an explicit, freshly split PRNG key to every random routine.
```

## Working on xtremax itself

Contributors (and their agents) follow
[`AGENTS.md`](https://github.com/jejjohnson/xtremax/blob/main/AGENTS.md) in
the repository: the layer map, the boundaries with NumPyro, pipekit and the
GP libraries, "reuse before you write", the contracts and the tests that
enforce them, and recipe skills for adding EVT distributions, point
processes, extraction utilities and notebooks.
