"""Pure-function primitives for temporal point processes.

Each submodule is free of class state and free of framework
dependencies (no NumPyro, no equinox). These are the canonical
reference implementations; the operator and distribution layers
delegate to them.
"""

from __future__ import annotations

from xtremax.point_processes.primitives.diagnostics import (
    compensator_curve,
    ks_statistic_exp1,
    qq_exp1_quantiles,
    time_rescaling_residuals,
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


__all__ = [
    "compensator_curve",
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
    "ipp_survival",
    "ks_statistic_exp1",
    "qq_exp1_quantiles",
    "time_rescaling_residuals",
]
