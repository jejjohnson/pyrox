# Point Processes — Operators

Immutable `equinox.Module` process objects that bundle an intensity model with
sampling. Operators are pytrees, safe under `jit` / `grad` / `vmap`, and
delegate all math to the [primitives](point-process-primitives.md).

The temporal operators (`HomogeneousPoissonProcess`,
`InhomogeneousPoissonProcess`, `RenewalProcess`, `ExponentialHawkes`,
`GeneralHawkesProcess`) also expose time-rescaling goodness-of-fit methods
(`residuals`, `goodness_of_fit`, `compensator_curve`). Spatial and
spatiotemporal diagnostics are function-based instead — see
`csr_ripleys_k`, `ipp_spatiotemporal_pearson_residuals`, and
`ipp_spatiotemporal_chi_square_gof` in the
[primitives](point-process-primitives.md).

::: xtremax.point_processes.operators
    options:
      show_root_heading: false
      show_root_toc_entry: false
