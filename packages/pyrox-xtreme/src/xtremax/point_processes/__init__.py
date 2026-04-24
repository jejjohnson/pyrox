"""Point processes for xtremax.

Three layers, mirroring the rest of the package:

* :mod:`~xtremax.point_processes.primitives` — pure functions (no
  NumPyro, no equinox).
* :mod:`~xtremax.point_processes.operators` — ``equinox.Module``
  classes that bundle intensity specs with numerical defaults. The
  primary user-facing API.
* :mod:`~xtremax.point_processes.distributions` — thin
  ``numpyro.distributions.Distribution`` wrappers.

This first release covers the temporal module: homogeneous Poisson
(HPP) and inhomogeneous Poisson (IPP). Spatial, marked, self-exciting
(Hawkes), and renewal processes slot into the same three-layer
structure in later releases.
"""

from __future__ import annotations

from xtremax.point_processes import distributions, operators, primitives
from xtremax.point_processes._integration import (
    cumulative_log_intensity,
    integrate_log_intensity,
)


__all__ = [
    "cumulative_log_intensity",
    "distributions",
    "integrate_log_intensity",
    "operators",
    "primitives",
]
