# Point Processes — Overview

Shared infrastructure for the point-process stack: observation domains, event
histories, structured sample results, goodness-of-fit containers, mark and
retention builders, and log-intensity integration. The three tiers built on
top of this live on their own pages —
[primitives](point-process-primitives.md),
[operators](point-process-operators.md), and
[distributions](point-process-distributions.md).

::: xtremax.point_processes
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - TemporalDomain
        - RectangularDomain
        - EventHistory
        - SampleResult
        - SpatialSampleResult
        - SpatiotemporalSampleResult
        - MarkedSampleResult
        - MarkedSpatialSampleResult
        - MarkedSpatiotemporalSampleResult
        - GoodnessOfFit
        - constant_mark_distribution
        - time_varying_marks
        - constant_retention
        - time_varying_retention
        - cumulative_log_intensity
        - integrate_log_intensity
        - integrate_log_intensity_spatial
        - integrate_log_intensity_spatiotemporal
