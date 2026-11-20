# Point Processes — Operators

Immutable `equinox.Module` process objects that bundle an intensity model with
sampling. Operators are pytrees, safe under `jit` / `grad` / `vmap`, and
delegate all math to the [primitives](point-process-primitives.md) — so an
operator is a convenience wrapper, never a separate implementation.

Use these when you want a process as a **reusable object** (fit it, sample
from it, check it); use the primitives when you want a single function inside
a larger model.

## Temporal processes

The full temporal family, from constant-rate Poisson through self-exciting
Hawkes. These are the operators that inherit the goodness-of-fit mixin, so
each exposes `residuals`, `goodness_of_fit`, and `compensator_curve` built on
the time-rescaling theorem.

::: xtremax.point_processes.operators
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

Patterns on a rectangular domain with no time dimension. Diagnostics here are
function-based rather than methods — see `csr_ripleys_k` and friends in the
[primitives](point-process-primitives.md#diagnostics).

::: xtremax.point_processes.operators
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - HomogeneousSpatialPP
        - InhomogeneousSpatialPP
        - MarkedSpatialPP

## Spatiotemporal processes

Events carrying both a location and a time, on $D \times [t_0, t_1)$ —
including the ETAS-style Hawkes process for spatially spreading contagion.

::: xtremax.point_processes.operators
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - HomogeneousSpatioTemporalPP
        - InhomogeneousSpatioTemporalPP
        - MarkedSpatioTemporalPP
        - SpatioTemporalHawkes

## Building blocks

Components the processes above are assembled from: excitation kernels for
Hawkes, a piecewise-constant log-intensity for non-parametric IPP fits, and
the diagnostics bundle.

::: xtremax.point_processes.operators
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - ExponentialKernel
        - PiecewiseConstantLogIntensity
        - GoodnessOfFit
