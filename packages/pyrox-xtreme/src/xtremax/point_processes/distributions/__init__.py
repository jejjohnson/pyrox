"""NumPyro ``Distribution`` wrappers for temporal and spatial point processes."""

from __future__ import annotations

from xtremax.point_processes.distributions.hawkes import (
    ExponentialHawkes,
    GeneralHawkesProcess,
)
from xtremax.point_processes.distributions.hawkes_spatiotemporal import (
    SpatioTemporalHawkes,
)
from xtremax.point_processes.distributions.hpp_spatial import HomogeneousSpatialPP
from xtremax.point_processes.distributions.hpp_spatiotemporal import (
    HomogeneousSpatioTemporalPP,
)
from xtremax.point_processes.distributions.ipp_spatial import InhomogeneousSpatialPP
from xtremax.point_processes.distributions.ipp_spatiotemporal import (
    InhomogeneousSpatioTemporalPP,
)
from xtremax.point_processes.distributions.marked import (
    MarkedTemporalPointProcess,
)
from xtremax.point_processes.distributions.marked_spatial import MarkedSpatialPP
from xtremax.point_processes.distributions.marked_spatiotemporal import (
    MarkedSpatioTemporalPP,
)
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
    "HomogeneousSpatioTemporalPP",
    "InhomogeneousPoissonProcess",
    "InhomogeneousSpatialPP",
    "InhomogeneousSpatioTemporalPP",
    "MarkedSpatialPP",
    "MarkedSpatioTemporalPP",
    "MarkedTemporalPointProcess",
    "RenewalProcess",
    "SpatioTemporalHawkes",
    "ThinningProcess",
]
