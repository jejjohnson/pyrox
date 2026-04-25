"""NumPyro ``Distribution`` wrappers for temporal and spatial point processes."""

from __future__ import annotations

from xtremax.point_processes.distributions.hawkes import (
    ExponentialHawkes,
    GeneralHawkesProcess,
)
from xtremax.point_processes.distributions.hpp_spatial import HomogeneousSpatialPP
from xtremax.point_processes.distributions.ipp_spatial import InhomogeneousSpatialPP
from xtremax.point_processes.distributions.marked import (
    MarkedTemporalPointProcess,
)
from xtremax.point_processes.distributions.marked_spatial import MarkedSpatialPP
from xtremax.point_processes.distributions.renewal import RenewalProcess
from xtremax.point_processes.distributions.temporal import (
    HomogeneousPoissonProcess,
    InhomogeneousPoissonProcess,
)
from xtremax.point_processes.distributions.thinning import ThinningProcess


__all__ = [
    "ExponentialHawkes",
    "GeneralHawkesProcess",
    "HomogeneousPoissonProcess",
    "HomogeneousSpatialPP",
    "InhomogeneousPoissonProcess",
    "InhomogeneousSpatialPP",
    "MarkedSpatialPP",
    "MarkedTemporalPointProcess",
    "RenewalProcess",
    "ThinningProcess",
]
