"""Pure-function primitives for temporal and spatial point processes.

Each submodule is free of class state and free of framework
dependencies (no NumPyro — save that for the outer-layer
wrappers — no equinox). These are the canonical reference
implementations; the operator and distribution layers delegate to them.
"""

from __future__ import annotations

from xtremax.point_processes.primitives.diagnostics import (
    compensator_curve,
    ks_statistic_exp1,
    qq_exp1_quantiles,
    time_rescaling_residuals,
)
from xtremax.point_processes.primitives.diagnostics_spatial import (
    csr_l_function,
    csr_pair_correlation,
    csr_ripleys_k,
    unit_ball_volume,
)
from xtremax.point_processes.primitives.hawkes import (
    exp_hawkes_cumulative_intensity,
    exp_hawkes_intensity,
    exp_hawkes_lambda_max,
    exp_hawkes_log_prob,
    exp_hawkes_sample,
    general_hawkes_cumulative_intensity,
    general_hawkes_intensity,
    general_hawkes_log_prob,
    general_hawkes_sample,
)
from xtremax.point_processes.primitives.hpp import (
    hpp_cumulative_intensity,
    hpp_exceedance_log_prob,
    hpp_hazard,
    hpp_intensity,
    hpp_inter_event_log_prob,
    hpp_log_prob,
    hpp_mean_residual_life,
    hpp_predict_count,
    hpp_return_period,
    hpp_sample,
    hpp_survival,
)
from xtremax.point_processes.primitives.hpp_spatial import (
    hpp_spatial_count_log_prob,
    hpp_spatial_intensity,
    hpp_spatial_log_prob,
    hpp_spatial_nearest_neighbor_distance,
    hpp_spatial_predict_count,
    hpp_spatial_sample,
)
from xtremax.point_processes.primitives.ipp import (
    ipp_cumulative_hazard,
    ipp_cumulative_intensity,
    ipp_hazard,
    ipp_intensity,
    ipp_inter_event_log_prob,
    ipp_log_prob,
    ipp_predict_count,
    ipp_sample_inversion,
    ipp_sample_thinning,
    ipp_survival,
)
from xtremax.point_processes.primitives.ipp_spatial import (
    ipp_spatial_intensity,
    ipp_spatial_log_prob,
    ipp_spatial_predict_count,
    ipp_spatial_sample_thinning,
)
from xtremax.point_processes.primitives.marked import (
    marks_log_prob,
    sample_marks_at_times,
)
from xtremax.point_processes.primitives.marked_spatial import (
    sample_spatial_marks_at_locations,
    spatial_marks_log_prob,
)
from xtremax.point_processes.primitives.renewal import (
    renewal_cumulative_hazard,
    renewal_expected_count,
    renewal_hazard,
    renewal_intensity,
    renewal_inter_event_log_prob,
    renewal_log_prob,
    renewal_ogata_intensity_fn,
    renewal_sample,
    renewal_survival,
)
from xtremax.point_processes.primitives.thinning import (
    retention_compensator,
    thinning_retention_log_prob,
    thinning_sample,
)


__all__ = [
    "compensator_curve",
    "csr_l_function",
    "csr_pair_correlation",
    "csr_ripleys_k",
    "exp_hawkes_cumulative_intensity",
    "exp_hawkes_intensity",
    "exp_hawkes_lambda_max",
    "exp_hawkes_log_prob",
    "exp_hawkes_sample",
    "general_hawkes_cumulative_intensity",
    "general_hawkes_intensity",
    "general_hawkes_log_prob",
    "general_hawkes_sample",
    "hpp_cumulative_intensity",
    "hpp_exceedance_log_prob",
    "hpp_hazard",
    "hpp_intensity",
    "hpp_inter_event_log_prob",
    "hpp_log_prob",
    "hpp_mean_residual_life",
    "hpp_predict_count",
    "hpp_return_period",
    "hpp_sample",
    "hpp_spatial_count_log_prob",
    "hpp_spatial_intensity",
    "hpp_spatial_log_prob",
    "hpp_spatial_nearest_neighbor_distance",
    "hpp_spatial_predict_count",
    "hpp_spatial_sample",
    "hpp_survival",
    "ipp_cumulative_hazard",
    "ipp_cumulative_intensity",
    "ipp_hazard",
    "ipp_intensity",
    "ipp_inter_event_log_prob",
    "ipp_log_prob",
    "ipp_predict_count",
    "ipp_sample_inversion",
    "ipp_sample_thinning",
    "ipp_spatial_intensity",
    "ipp_spatial_log_prob",
    "ipp_spatial_predict_count",
    "ipp_spatial_sample_thinning",
    "ipp_survival",
    "ks_statistic_exp1",
    "marks_log_prob",
    "qq_exp1_quantiles",
    "renewal_cumulative_hazard",
    "renewal_expected_count",
    "renewal_hazard",
    "renewal_intensity",
    "renewal_inter_event_log_prob",
    "renewal_log_prob",
    "renewal_ogata_intensity_fn",
    "renewal_sample",
    "renewal_survival",
    "retention_compensator",
    "sample_marks_at_times",
    "sample_spatial_marks_at_locations",
    "spatial_marks_log_prob",
    "thinning_retention_log_prob",
    "thinning_sample",
    "time_rescaling_residuals",
    "unit_ball_volume",
]
