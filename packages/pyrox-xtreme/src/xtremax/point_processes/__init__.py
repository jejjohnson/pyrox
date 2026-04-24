"""Point processes for xtremax.

Three layers, mirroring the rest of the package:

* :mod:`~xtremax.point_processes.primitives` — pure functions (no
  equinox). These take a NumPyro Distribution as the
  inter-event / mark law where relevant; they never depend on
  the NumPyro Distribution base machinery (sample / log_prob / cdf
  is used directly).
* :mod:`~xtremax.point_processes.operators` — ``equinox.Module``
  classes that bundle intensity specs with numerical defaults. The
  primary user-facing API.
* :mod:`~xtremax.point_processes.distributions` — thin
  ``numpyro.distributions.Distribution`` wrappers around the
  operators so these processes can appear inside NumPyro models.

Families currently available:

* Homogeneous Poisson (``HomogeneousPoissonProcess``).
* Inhomogeneous Poisson (``InhomogeneousPoissonProcess``).
* Renewal (``RenewalProcess``) — any NumPyro inter-event distribution.
* Self-exciting Hawkes (``ExponentialHawkes``, ``GeneralHawkesProcess``).
* Marked (``MarkedTemporalPointProcess``) — ground × mark-distribution.
* Thinning (``ThinningProcess``) — base TPP × retention callable.

The shared :class:`~xtremax.point_processes._history.EventHistory`
pytree is the lingua franca for user-supplied retention and mark
callables: every family threads history through in the same shape so
user code need not know the underlying family.
"""

from __future__ import annotations

from xtremax.point_processes import distributions, operators, primitives
from xtremax.point_processes._adapters import (
    constant_mark_distribution,
    constant_retention,
    time_varying_marks,
    time_varying_retention,
)
from xtremax.point_processes._history import EventHistory
from xtremax.point_processes._integration import (
    cumulative_log_intensity,
    integrate_log_intensity,
)


__all__ = [
    "EventHistory",
    "constant_mark_distribution",
    "constant_retention",
    "cumulative_log_intensity",
    "distributions",
    "integrate_log_intensity",
    "operators",
    "primitives",
    "time_varying_marks",
    "time_varying_retention",
]
