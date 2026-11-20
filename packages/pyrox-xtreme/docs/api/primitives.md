# Primitives

The distribution math as standalone pure JAX functions — for custom
likelihoods, vmapped parameter grids, and non-stationary fields where a
distribution object would be in the way. Every family exposes the same core
set (`log_prob`, `cdf`, `icdf`, `mean`, `return_level`), so switching tail
behaviour is a rename rather than a rewrite.

All functions take parameters positionally as `(x, loc, scale, shape)` and
broadcast over them, so you can evaluate a whole spatial field of parameters
in one call.

## Generalized Extreme Value (GEV)

The block-maxima workhorse. The concentration $\xi$ spans all three tail
types smoothly, and the $\xi \to 0$ Gumbel limit is handled with a numerically
stable branch rather than a discontinuity. Survival functions come in linear
and log forms — use `gev_log_survival` in the far tail where $1 - F(x)$
underflows.

::: xtremax.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - gev_log_prob
        - gev_cdf
        - gev_icdf
        - gev_survival
        - gev_log_survival
        - gev_mean
        - gev_return_level

## Generalized Pareto (GPD)

The peaks-over-threshold counterpart, defined on exceedances above a threshold
(so `loc = 0` by convention). Same structure as the GEV family, including the
log-survival form for deep-tail work.

::: xtremax.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - gpd_log_prob
        - gpd_cdf
        - gpd_icdf
        - gpd_survival
        - gpd_log_survival
        - gpd_mean
        - gpd_return_level

## Gumbel — GEV Type I

The $\xi = 0$ case: exponential tails, closed-form moments
($\mu + \sigma\gamma_E$), and no shape parameter to estimate.

::: xtremax.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - gumbel_log_prob
        - gumbel_cdf
        - gumbel_icdf
        - gumbel_mean
        - gumbel_return_level

## Fréchet — GEV Type II

The $\xi > 0$ case: heavy polynomial tails and lower-bounded support. The mean
diverges once $\xi \ge 1$, and `frechet_mean` returns `+inf` there rather than
a silently wrong number.

::: xtremax.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - frechet_log_prob
        - frechet_cdf
        - frechet_icdf
        - frechet_mean
        - frechet_return_level

## Weibull (reverse) — GEV Type III

The $\xi < 0$ case: a bounded upper tail, for quantities with a physical
ceiling. Moments are always finite.

::: xtremax.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - weibull_log_prob
        - weibull_cdf
        - weibull_icdf
        - weibull_mean
        - weibull_return_level

## Non-stationary EVT

Under a changing climate the return level is no longer a single number — the
GEV parameters become functions of covariates, and "the 100-year event" has to
be re-stated per year or per scenario. These helpers build covariate design
matrices, assemble spatio-temporal parameter fields, and invert between return
levels and return periods for both stationary and non-stationary fields.

::: xtremax.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - design_matrix
        - assemble_nonstationary_gev_fields
        - nonstationary_return_level
        - nonstationary_return_period
        - expected_exceedances

## Spatial helpers

Building blocks for spatial pooling: a distance matrix between sites and a
two-range exponential correlation used as a GP covariance for parameter
fields.

::: xtremax.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - pairwise_distances
        - two_range_correlation
