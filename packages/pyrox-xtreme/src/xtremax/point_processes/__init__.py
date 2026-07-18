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

Padded-buffer invariant
-----------------------

Every padded event sequence in this package satisfies: **the mask is a
contiguous prefix** (``mask == [True]*n + [False]*(max-n)``) **and
padding times sit at the right edge of the window** (``T``; spatial
padding rows at ``domain.lo``). All samplers emit sequences in this
form — thinning-based samplers compact accepted events before
returning — and consumers with sequential structure
(``time_rescaling_residuals``, ``exp_hawkes_log_prob`` via Ozaki gaps,
``renewal_log_prob`` via gap differences) assume it. Hand-constructed
inputs with hole masks are supported only where a function explicitly
documents mask-robust behaviour.

The shared :class:`~xtremax.point_processes._history.EventHistory`
pytree is the lingua franca for user-supplied retention and mark
callables: every family threads history through in the same shape so
user code need not know the underlying family.

Sample-result convention
------------------------

Every sampler returns a ``NamedTuple`` from
:mod:`~xtremax.point_processes._results` —
:class:`SampleResult` ``(times, mask, n_events)`` for unmarked temporal
processes, :class:`MarkedSampleResult` ``(times, mask, marks)`` for
marked ones, and the spatial/spatiotemporal analogues. These unpack
positionally exactly like the tuples they replaced, are valid JAX
pytrees, and let wrapper operators dispatch on the result *type*
instead of shape/dtype heuristics. The ``n_events`` semantics are
family-specific and documented per sampler (HPP: uncapped event count;
thinning IPPs: uncapped candidate count; Hawkes: proposals consumed;
renewal: retained events). ``sample_shape`` batching exists only on the
temporal HPP; batch every other sampler with ``jax.vmap`` /
``equinox.filter_vmap`` over keys.

Operator architecture
---------------------

Cross-family operator surface lives once in
:mod:`~xtremax.point_processes.operators._base`: the time-rescaling
diagnostics trio (``residuals`` / ``goodness_of_fit`` /
``compensator_curve``) in ``GoodnessOfFitMixin`` (families supply only
``_compensator_fn``), the live-``λ_max`` accessor in
``LiveIntensityMixin``, and the separable marked-process plumbing in
``SeparableMarkedPP``.
"""

from __future__ import annotations

from xtremax.point_processes import distributions, operators, primitives
from xtremax.point_processes._adapters import (
    constant_mark_distribution,
    constant_retention,
    time_varying_marks,
    time_varying_retention,
)
from xtremax.point_processes._domain import RectangularDomain, TemporalDomain
from xtremax.point_processes._history import EventHistory
from xtremax.point_processes._integration import (
    cumulative_log_intensity,
    integrate_log_intensity,
)
from xtremax.point_processes._integration_spatial import (
    integrate_log_intensity_spatial,
)
from xtremax.point_processes._integration_spatiotemporal import (
    integrate_log_intensity_spatiotemporal,
)
from xtremax.point_processes._results import (
    GoodnessOfFit,
    MarkedSampleResult,
    MarkedSpatialSampleResult,
    MarkedSpatiotemporalSampleResult,
    SampleResult,
    SpatialSampleResult,
    SpatiotemporalSampleResult,
)


__all__ = [
    "EventHistory",
    "GoodnessOfFit",
    "MarkedSampleResult",
    "MarkedSpatialSampleResult",
    "MarkedSpatiotemporalSampleResult",
    "RectangularDomain",
    "SampleResult",
    "SpatialSampleResult",
    "SpatiotemporalSampleResult",
    "TemporalDomain",
    "constant_mark_distribution",
    "constant_retention",
    "cumulative_log_intensity",
    "distributions",
    "integrate_log_intensity",
    "integrate_log_intensity_spatial",
    "integrate_log_intensity_spatiotemporal",
    "operators",
    "primitives",
    "time_varying_marks",
    "time_varying_retention",
]
