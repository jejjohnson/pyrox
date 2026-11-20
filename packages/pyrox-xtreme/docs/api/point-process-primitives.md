# Point Processes — Primitives

Pure functions for every process family. Each family follows the same shape:
an **intensity** $\lambda(\cdot)$, a **compensator** $\Lambda(\cdot)$ (the
integrated intensity), a **log-likelihood** built from
$\sum_i \log \lambda(x_i) - \Lambda$, and a **sampler**. Everything is JAX —
differentiable, `jit`-able, `vmap`-able — and PRNG keys are explicit
arguments.

The [operators](point-process-operators.md) and
[distributions](point-process-distributions.md) are thin wrappers over these;
call the primitives directly when you want a custom likelihood or a vmapped
parameter sweep.

## Homogeneous Poisson — temporal

Constant intensity $\lambda(t) = \lambda$, so every quantity is closed-form:
the compensator is $\lambda t$, inter-event gaps are $\mathrm{Exp}(\lambda)$,
and memorylessness makes the mean residual life exactly $1/\lambda$.

::: xtremax.point_processes.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - hpp_intensity
        - hpp_cumulative_intensity
        - hpp_log_prob
        - hpp_sample
        - hpp_hazard
        - hpp_survival
        - hpp_inter_event_log_prob
        - hpp_mean_residual_life
        - hpp_predict_count
        - hpp_exceedance_log_prob
        - hpp_return_period

## Homogeneous Poisson — spatial

Complete spatial randomness (CSR) on a rectangular domain: the joint Janossy
likelihood, the marginal Poisson count, and the CSR reference quantities that
spatial diagnostics compare against.

::: xtremax.point_processes.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - hpp_spatial_intensity
        - hpp_spatial_log_prob
        - hpp_spatial_count_log_prob
        - hpp_spatial_sample
        - hpp_spatial_predict_count
        - hpp_spatial_nearest_neighbor_distance

## Homogeneous Poisson — spatiotemporal

The same constant-intensity process on $D \times [t_0, t_1)$.

::: xtremax.point_processes.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - hpp_spatiotemporal_intensity
        - hpp_spatiotemporal_log_prob
        - hpp_spatiotemporal_count_log_prob
        - hpp_spatiotemporal_sample
        - hpp_spatiotemporal_predict_count

## Inhomogeneous Poisson — temporal

Intensity supplied as a **log**-intensity callable, so positivity is
structural rather than a constraint. The compensator comes from composite
trapezoid quadrature; sampling is either Lewis–Shedler thinning (general) or
exact inversion when $\Lambda^{-1}$ is available.

::: xtremax.point_processes.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - ipp_intensity
        - ipp_cumulative_intensity
        - ipp_log_prob
        - ipp_sample_thinning
        - ipp_sample_inversion
        - ipp_hazard
        - ipp_cumulative_hazard
        - ipp_survival
        - ipp_inter_event_log_prob
        - ipp_predict_count

## Inhomogeneous Poisson — spatial

::: xtremax.point_processes.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - ipp_spatial_intensity
        - ipp_spatial_log_prob
        - ipp_spatial_sample_thinning
        - ipp_spatial_predict_count

## Inhomogeneous Poisson — spatiotemporal

Beyond the core four, this family carries the marginalisation helpers you need
for plotting — collapsing $\lambda(s, t)$ onto space or time — and the
residual-based goodness-of-fit tests that stand in for the temporal
time-rescaling diagnostics.

::: xtremax.point_processes.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - ipp_spatiotemporal_intensity
        - ipp_spatiotemporal_log_prob
        - ipp_spatiotemporal_sample_thinning
        - ipp_spatiotemporal_predict_count
        - ipp_spatiotemporal_intensity_surface_at_time
        - ipp_spatiotemporal_marginal_spatial_intensity
        - ipp_spatiotemporal_marginal_temporal_intensity
        - ipp_spatiotemporal_pearson_residuals
        - ipp_spatiotemporal_chi_square_gof

## Renewal processes

Parameterised by the **inter-event** distribution rather than an intensity:
the conditional intensity is the hazard evaluated since the last event,
$\lambda^*(t) = h(t - t_{N(t)})$. `renewal_ogata_intensity_fn` adapts a
renewal process into the `(t, history) -> λ*(t)` closure that the thinning
samplers expect.

::: xtremax.point_processes.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - renewal_intensity
        - renewal_log_prob
        - renewal_sample
        - renewal_hazard
        - renewal_cumulative_hazard
        - renewal_survival
        - renewal_inter_event_log_prob
        - renewal_expected_count
        - renewal_ogata_intensity_fn

## Hawkes — temporal

Self-exciting processes where past events raise the intensity. The
exponential-kernel variant has a closed-form compensator and a recursive
intensity; the general-kernel variant takes any excitation function $\phi$ at
the cost of quadrature. Both sample via Ogata thinning, which needs the
`lambda_max` upper bound.

::: xtremax.point_processes.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - exp_hawkes_intensity
        - exp_hawkes_cumulative_intensity
        - exp_hawkes_log_prob
        - exp_hawkes_sample
        - exp_hawkes_lambda_max
        - general_hawkes_intensity
        - general_hawkes_cumulative_intensity
        - general_hawkes_log_prob
        - general_hawkes_sample

## Hawkes — spatiotemporal (ETAS)

Separable exp-Gaussian excitation: an exponential temporal decay times a
Gaussian spatial kernel, the ETAS form used for aftershock and contagion
modelling.

::: xtremax.point_processes.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - stpp_hawkes_intensity
        - stpp_hawkes_compensator
        - stpp_hawkes_log_prob
        - stpp_hawkes_sample
        - stpp_hawkes_lambda_max

## Marks

Marked processes factorise as *ground process × mark distribution*, so the
mark contribution to the likelihood is a separate additive term. These
functions supply that term and the matching samplers for each event geometry.

::: xtremax.point_processes.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - marks_log_prob
        - sample_marks_at_times
        - spatial_marks_log_prob
        - sample_spatial_marks_at_locations
        - spatiotemporal_marks_log_prob
        - sample_spatiotemporal_marks

## Thinning & observation operators

Thinning models *partial observation*: a base process generates events and a
retention probability decides which are recorded. Useful as an observation
operator on top of any ground process.

::: xtremax.point_processes.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - thinning_sample
        - thinning_retention_log_prob
        - retention_compensator

## Diagnostics

Temporal diagnostics rest on the **time-rescaling theorem**: under the correct
model, rescaled inter-event times are $\mathrm{Exp}(1)$, which turns model
checking into a KS test or a QQ plot. Spatial diagnostics instead compare
against CSR through Ripley's $K$ and its variance-stabilised relatives.

::: xtremax.point_processes.primitives
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - time_rescaling_residuals
        - compensator_curve
        - ks_statistic_exp1
        - qq_exp1_quantiles
        - csr_ripleys_k
        - csr_l_function
        - csr_pair_correlation
        - unit_ball_volume
