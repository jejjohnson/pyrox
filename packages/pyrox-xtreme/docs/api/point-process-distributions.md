# Point Processes — Distributions

NumPyro `Distribution` wrappers around the point-process
[operators](point-process-operators.md), so a point process can be used
directly as a likelihood site in an MCMC or SVI model:

```python
numpyro.sample("events", HomogeneousPoissonProcess(rate, T), obs=times)
```

Each class mirrors the operator of the same name — the wrapper adds
NumPyro's `Distribution` interface (`log_prob`, `sample`, batch/event shapes)
and nothing else.

## Temporal processes

::: xtremax.point_processes.distributions
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - HomogeneousPoissonProcess
        - InhomogeneousPoissonProcess
        - RenewalProcess
        - ExponentialHawkes
        - GeneralHawkesProcess
        - MarkedTemporalPointProcess
        - ThinningProcess

## Spatial processes

::: xtremax.point_processes.distributions
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - HomogeneousSpatialPP
        - InhomogeneousSpatialPP
        - MarkedSpatialPP

## Spatiotemporal processes

::: xtremax.point_processes.distributions
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - HomogeneousSpatioTemporalPP
        - InhomogeneousSpatioTemporalPP
        - MarkedSpatioTemporalPP
        - SpatioTemporalHawkes
