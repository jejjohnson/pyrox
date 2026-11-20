---
name: add-evt-distribution
description: Add an extreme value distribution to xtremax — or a new quantity (a moment, a tail function, a return level variant) to an existing one — as a pure-JAX primitive in src/xtremax/primitives plus a NumPyro Distribution class in src/xtremax/distributions that routes to it, smooth through the ξ → 0 limit, with a support NUTS can use, scipy-oracle and gradient tests, docs and the capability index. Use when asked to add, port or fix an EVT family (GEV, GPD, Gumbel, Fréchet, Weibull, a new one), a density / CDF / quantile / moment / return-level computation, or a non-stationary or spatial EVT primitive.
---

# Add an EVT distribution or quantity

Read "1. Primitives" and "2. EVT distributions" under "The contracts" in
`AGENTS.md` first; this is the step-by-step.

## 1. Make sure it does not exist yet

- Search `docs/api/capabilities.md` (`xtremax.primitives`,
  `xtremax.distributions`). Fréchet and Weibull are the GEV restricted to
  ξ > 0 / ξ < 0, Gumbel the ξ = 0 limit and the exponential the GPD's: a
  re-parameterisation of an existing family is a thin wrapper over its
  `gev_*` / `gpd_*` primitives (see `primitives/frechet.py`), not new
  maths.
- `GeneralizedExtremeValueDistribution` (and `WeibullType3GEVD`) already
  carry `survival_function`, `log_survival_function`, `hazard_rate`,
  `cumulative_hazard_rate`, `return_level`, `tail_index`,
  `exceedance_probability`, `conditional_excess_mean`, `mean_residual_life`,
  `percentile_residual_life`, `entropy`, `variance`, `skew`, `kurtosis`;
  the other classes lack a few of these — add a missing one by routing to
  the existing primitive, as the GEV does.
- NumPyro may already have it (`numpyro.distributions.Gumbel`, `Weibull`,
  `Pareto`): extend it, as `GumbelType1GEVD` does, rather than re-deriving.
- Non-stationary return levels and spatial building blocks are in
  `primitives/nonstationary.py` and `primitives/spatial.py`; GP priors on
  parameters belong to a GP library, not here.

## 2. The primitive first (`src/xtremax/primitives/<family>.py`)

- Pure functions, value first, then parameters:
  `<family>_log_prob(x, loc, scale, shape)` (drop `loc` if the family has
  none, like the GPD), `_cdf`, `_survival`, `_log_survival`, `_icdf(q, …)`,
  `_mean(loc, scale, shape)`, `_return_level(period, …)`. ξ is called
  `shape` here.
- Smooth through the limit: write `log(1 + ξz)/ξ` as
  `z * log1p_over_x(ξ z)` and quantile expansions through `expm1_over_x`
  (`primitives/_common.py`); no `jnp.where(ξ == 0, …)` in the kernel.
- Standardise with `clamped_standardize`; tail exponentials with
  `safe_exp_neg` (CDF / survival) or `safe_exp_neg_density` (log-density);
  survival as `-expm1(log F)`; quantiles from `-log q`; return levels via
  `-log1p(-1/T)` (see `_gev_icdf_from_neg_log_q` in `primitives/gev.py`).
- Moments built from `Γ(1 - kξ)` differences go through the reduced forms
  in `primitives/_moments.py`, not the textbook formula (it cancels to
  nothing near ξ = 0, issue #88).
- Support and extended reals: out of support → log-density `-inf`, CDF and
  survival the boundary value; `x = ±inf` → the limit; NaN propagates;
  sanitise every discarded `jnp.where` branch so its gradient is finite.
- Module docstring with the parameterisation (and scipy's, if its sign
  differs); a one-line summary per function (the capability index uses it).

## 3. The class (`src/xtremax/distributions/<family>.py`)

Exemplar: `GeneralizedExtremeValueDistribution` (`gevd.py`).

- Subclass `numpyro.distributions.Distribution` (or the NumPyro class it
  extends); `arg_constraints` for every parameter (`concentration` with the
  sign constraint the family needs), `reparametrized_params`.
- `__init__(loc, scale, concentration=None, shape=None, validate_args=None)`:
  the deprecated `shape=` alias warns with `DeprecationWarning` and raises
  if both are given; `promote_shapes`, then `batch_shape` from
  `lax.broadcast_shapes`.
- `support` as a property: `concrete_sign(self.concentration)` →
  `greater_than_eq(self.lower_bound())`, `less_than_eq(self.upper_bound())`
  or `real` (`distributions/_support.py`). Never `interval` with an
  infinite bound.
- `sample(key, sample_shape=())`: `check_prng_key(key)` (`xtremax._rng`),
  a `Uniform(0, 1)` draw clipped to `[eps, 1 - eps]`, then `self.icdf`.
- `log_prob` decorated with `@validate_sample`; every quantity a one-line
  call to the primitive; class-layer branches (if any remain) use
  `GUMBEL_THRESHOLD` from `primitives/_common.py`.
- `expand(batch_shape)` rebuilds through `type(self)(...)`.
- Docstring: the density and CDF, the support per sign of ξ, an
  `Examples:` block that runs (no test runs it — check it by hand).

## 4. Export, docs, capability index

- Primitive: `primitives/__init__.py` (import + `__all__`) and
  `src/xtremax/__init__.py` (import + `__all__`); the `members:` list of
  the family's section on `docs/api/primitives.md`.
- Class: `distributions/__init__.py`, `src/xtremax/__init__.py`, a section
  on `docs/api/distributions.md`, and a row in the table on
  `docs/api/index.md` if it changes a subpackage's summary.
- `make capabilities`.

## 5. Tests

- `tests/test_oracles.py`: pin every quantity to `scipy.stats`
  (`genextreme(c=-ξ)`, `genpareto(c=ξ)`, …) across `XI_GRID`, which
  straddles 0 — add the family to `_class_and_frozen`, `CLASS_CASES` and
  `SAMPLE_CASES`; a float64 tail check goes into `TestFloat64Isolation`'s
  subprocess script.
- `tests/test_gradients.py`: finite-difference gradients against scipy in
  the small-ξ band; add a batched instance to `_batched_distributions()` so
  every method in `NULLARY` / `UNARY_X` runs with batched parameters.
- `tests/test_primitives.py`: round trips, the limit at ξ = 0, `jit` /
  `vmap`, class ↔ primitive parity (`TestClassPrimitiveParity`).
- `tests/test_distributions.py`: add the class to `TestBijectToSupport._dists`
  (and the NUTS latent-site parametrisation) and to the PRNG-key checks in
  `TestPRNGKeyValidation`; the `shape=` alias (`TestShapeParameterAlias`).
- Monte Carlo checks bounded by the statistic's sampling distribution with
  the bound's source in a comment; `slow` for samplers and sweeps,
  `integration` for MCMC.

## 6. Verify

```bash
# no addopts marker filter: a bare pytest runs every lane of these files
uv run pytest -n auto tests/test_oracles.py tests/test_gradients.py \
  tests/test_primitives.py tests/test_distributions.py
uv run pytest tests/test_docs_api_coverage.py tests/test_capabilities.py
```

then the `pre-pr-check` skill.
