"""NumPyro ``Distribution`` wrappers for temporal point processes."""

from __future__ import annotations

from xtremax.point_processes.distributions.temporal import (
    HomogeneousPoissonProcess,
    InhomogeneousPoissonProcess,
)


__all__ = [
    "HomogeneousPoissonProcess",
    "InhomogeneousPoissonProcess",
]
