# API Reference

xtremax is a JAX/NumPyro-native library for extreme value modeling, built on
[JAX](https://github.com/jax-ml/jax), [NumPyro](https://github.com/pyro-ppl/numpyro),
[equinox](https://github.com/patrick-kidger/equinox), and
[xarray](https://github.com/pydata/xarray). The reference is organised by
subsystem rather than dumped as one flat page:

| Section | Subpackage | What's inside |
|---------|-----------|---------------|
| [Distributions](distributions.md) | `xtremax.distributions` | NumPyro-native EVT distributions — GEV, GPD, Gumbel, Fréchet, Weibull — with `return_level`, survival functions, and full moment support |
| [Primitives](primitives.md) | `xtremax.primitives` | Pure JAX functions per family (`gev_*`, `gpd_*`, …): log-probs, CDFs, quantiles, return levels, plus non-stationary and spatial helpers |
| [Extraction](extraction.md) | `xtremax.extraction` | xarray-native extremes extraction — block maxima, threshold selection, declustering, extremal-index estimation |
| [Point Processes — Overview](point-processes.md) | `xtremax.point_processes` | Shared infrastructure: domains, event histories, sample results, mark/retention builders, log-intensity integration |
| [Point Processes — Primitives](point-process-primitives.md) | `xtremax.point_processes.primitives` | Pure functions: intensities, compensators, log-probs, samplers, and diagnostics for HPP / IPP / renewal / Hawkes / marked processes |
| [Point Processes — Operators](point-process-operators.md) | `xtremax.point_processes.operators` | `equinox.Module` process objects bundling intensity and sampling; temporal operators add time-rescaling goodness-of-fit |
| [Point Processes — Distributions](point-process-distributions.md) | `xtremax.point_processes.distributions` | NumPyro `Distribution` wrappers — point processes as likelihoods in MCMC / SVI |
| [Simulations](simulations.md) | `xtremax.simulations` | Synthetic extremes generators: temperature / precipitation / wind, GMST trajectories, spatial fields |

## Conventions

A few patterns hold across the whole package:

- **NumPyro-native.** Distributions subclass
  `numpyro.distributions.Distribution` and work directly with NUTS, SVI, and
  `Predictive`. There is no custom inference machinery in xtremax.

- **JAX all the way down.** All distribution math is JAX — differentiable,
  jit-able, vmap-able. Point-process operators are immutable `equinox.Module`
  pytrees, safe under `jit` / `grad` / `vmap`.

- **Pure functions underneath.** Every distribution and operator delegates to
  pure primitives (`gev_log_prob`, `hpp_intensity`, …) that you can call
  directly for custom likelihoods and vmapped parameter fields. Across the
  distributions and point-process stack, PRNG keys are explicit arguments for
  every stochastic routine; the `simulations` generators are the exception —
  they are NumPy-based and take integer `seed` arguments instead.

- **xarray as the data interface.** Extraction utilities take and return
  `xr.DataArray` / `xr.Dataset`, preserving coordinates, dimensions, and
  metadata.

- **Theory-aware naming.** Parameters follow EVT conventions — `loc`, `scale`,
  `concentration` (ξ) — and methods like `return_level` and
  `estimate_extremal_index` are first-class citizens.

## See also

- [Vision](../design_docs/vision.md) — why xtremax exists and what it
  deliberately is not.
- [Architecture](../design_docs/architecture.md) — draft roadmap for the
  layered primitives → components → models design; parts are aspirational,
  and this reference reflects what ships today.
- [Interop](../interop.md) — how xtremax composes with the wider ecosystem
  without depending on it.
