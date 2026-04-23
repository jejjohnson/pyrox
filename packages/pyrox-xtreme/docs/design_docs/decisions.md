---
status: draft
version: 0.1.0
---

# Design Decisions

---

## D1: Point processes under distributions/

**Status:** accepted

**Context:** Point processes (Hawkes, Poisson, renewal) are used for modeling extreme event arrival times. They have log_prob, sample, and can be used in NumPyro models. Are they "distributions" or a separate concept?

**Decision:** They live under `distributions/point_process/`. They are distributions in the NumPyro sense — they implement the `Distribution` interface with `log_prob` and `sample`.

**Consequences:** Clean import path (`from xtremax.distributions.point_process import HawkesProcess`). If the subpackage grows too large, revisit.

---

## D2: GPJax is the GP engine

**Status:** accepted

**Context:** Spatial GEV models need Gaussian processes for spatial pooling. Should we hand-roll GPs or use an existing library?

**Decision:** GPJax. No hand-rolled GPs. The `xtremax.gp` module wraps GPJax with helpers for parameter storage, variational GP, and sparse variational GP with minibatching.

**Consequences:** GPJax is an optional dependency. Users who don't need spatial models don't install it.

---

## D3: Copulas via NumPyro builtins

**Status:** accepted

**Context:** Multivariate extreme value modeling needs copulas. Build custom or use NumPyro's?

**Decision:** Use NumPyro's built-in `numpyro.distributions.copulas`. xtremax provides thin helpers for combining copulas with EVT marginals, not a custom copula implementation.

**Consequences:** Lightweight — no copula math to maintain. Limited to what NumPyro supports.

---

## D4: Max-stable processes in scope

**Status:** accepted

**Context:** Max-stable processes (Smith, Schlather, Brown-Resnick, Extremal-t) are the spatial analog of GEV. Are they core or deferred?

**Decision:** In scope for v0.1. Four models, inference via composite pairwise likelihood.

**Consequences:** Significant API surface in `distributions/max_stable/`. Composite likelihood requires special handling (not standard NumPyro MCMC).

---

## D5: Three-tier threshold selection

**Status:** accepted

**Context:** POT analysis requires choosing a threshold. Should xtremax provide one method or multiple?

**Decision:** Three tiers of increasing sophistication:
1. Constant quantile threshold (simple, most common)
2. Trend-following quantile threshold (tracks warming/trends via covariates)
3. Parametric quantile threshold (GP quantile regression for spatially varying thresholds)

**Consequences:** Users can start simple and upgrade. Each tier builds on the previous.

---

## Resolved Questions

| Question | Resolution |
|---|---|
| Point process location | `distributions/point_process/` |
| GP engine | GPJax (wrapped in `xtremax.gp`) |
| Copula implementation | NumPyro builtins + thin helpers |
| Max-stable scope | In scope for v0.1 |
| Threshold selection | Three-tier system (constant, trend, parametric) |
| Poisson variant consolidation | Four files → single `poisson.py` with four classes |
| Model generalization | Temperature-specific → arbitrary covariates |
| Public API style | Re-export from `__init__.py`, import from `xtremax.distributions` |
