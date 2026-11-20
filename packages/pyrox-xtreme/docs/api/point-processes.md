# Point Processes — Overview

Shared infrastructure for the point-process stack. The three tiers built on
top of this live on their own pages —
[primitives](point-process-primitives.md) (pure functions),
[operators](point-process-operators.md) (`equinox.Module` objects), and
[distributions](point-process-distributions.md) (NumPyro likelihoods).

Everything here is fixed-size by design: JAX needs static shapes, so event
sequences are stored in padded buffers with a companion boolean mask rather
than as ragged arrays. That mask threads through every likelihood and sampler
in the package.

## Domains

The observation window. Its measure appears directly in every likelihood — the
compensator term is an integral over exactly this region — so getting it right
matters as much as the intensity itself.

::: xtremax.point_processes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - TemporalDomain
        - RectangularDomain

## Event history

The conditioning buffer for self-exciting processes: which events have already
happened, and (optionally) their marks. Hawkes intensities read from this.

::: xtremax.point_processes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - EventHistory

## Sample results

What the samplers return: padded event arrays plus the mask that says which
entries are real. One type per event geometry (temporal / spatial /
spatiotemporal), each with a marked counterpart that carries the mark array
alongside.

::: xtremax.point_processes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - SampleResult
        - SpatialSampleResult
        - SpatiotemporalSampleResult
        - MarkedSampleResult
        - MarkedSpatialSampleResult
        - MarkedSpatiotemporalSampleResult

## Goodness of fit

The bundle returned by the temporal operators' `goodness_of_fit` method —
time-rescaled residuals plus the KS statistic and QQ quantiles computed from
them.

::: xtremax.point_processes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - GoodnessOfFit

## Marks & retention builders

Small adapters that lift a simple callable into the full
`(coords, history) -> Distribution` signature the marked and thinned processes
expect — so a constant mark distribution or a time-varying retention
probability doesn't require writing the whole closure by hand.

::: xtremax.point_processes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - constant_mark_distribution
        - time_varying_marks
        - constant_retention
        - time_varying_retention

## Log-intensity integration

Quadrature for the compensator $\Lambda = \int \exp(\log \lambda)$ over each
domain type. Intensities are parameterised in log space for positivity, so
these integrate the exponential rather than the callable directly.

::: xtremax.point_processes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - cumulative_log_intensity
        - integrate_log_intensity
        - integrate_log_intensity_spatial
        - integrate_log_intensity_spatiotemporal
