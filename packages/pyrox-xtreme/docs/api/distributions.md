# Distributions

NumPyro-native extreme value distributions. Every class subclasses
`numpyro.distributions.Distribution`, so it plugs directly into MCMC / SVI /
`Predictive` and is differentiable end to end. Beyond `sample` / `log_prob`,
each carries the EVT toolkit: `cdf`, `icdf`, `survival_function`, `mean`,
`variance`, `entropy`, and `return_level`.

::: xtremax.distributions
    options:
      show_root_heading: false
      show_root_toc_entry: false
