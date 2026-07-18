"""Shared result types for point-process samplers and diagnostics.

Sample-result convention (decided in issue #65, option A):

Every sampler returns a ``NamedTuple`` instead of a bare tuple. A
``NamedTuple`` unpacks positionally exactly like the tuple it replaces
(``times, mask, n = op.sample(...)`` keeps working verbatim), is a
registered JAX pytree (safe through ``jit`` / ``vmap`` / ``scan``), and
— crucially — lets wrapper operators (:class:`ThinningProcess`,
marked processes) dispatch on the *type* of the result instead of on
fragile ``ndim``/dtype heuristics that batched samplers would silently
break.

The ``n_events`` field's precise semantics are family-specific and
documented per operator:

* temporal/spatial/spatiotemporal **HPP** — the uncapped Poisson event
  count (``> max_events`` signals buffer truncation);
* temporal/spatial/spatiotemporal **IPP** (thinning samplers) — the
  uncapped Poisson *candidate* count (``> max_candidates`` signals a
  truncated candidate pool);
* **Hawkes** (Ogata thinning) — proposals consumed;
* **renewal** — retained (accepted) events.

``sample_shape`` batching is supported only by the temporal HPP sampler;
every other operator is batched with ``jax.vmap`` /
``equinox.filter_vmap`` over keys.
"""

from __future__ import annotations

from typing import NamedTuple

from jaxtyping import Array, Bool, Float, Int


class SampleResult(NamedTuple):
    """Unmarked temporal sample: ``(times, mask, n_events)``."""

    times: Float[Array, ...]
    mask: Bool[Array, ...]
    n_events: Int[Array, ...]


class MarkedSampleResult(NamedTuple):
    """Marked temporal sample: ``(times, mask, marks)``.

    Three fields (not four) so it unpacks identically to the tuple it
    replaces; the retained-event count is ``mask.sum()``.
    """

    times: Float[Array, ...]
    mask: Bool[Array, ...]
    marks: Float[Array, ...]


class SpatialSampleResult(NamedTuple):
    """Unmarked spatial sample: ``(locations, mask, n_events)``."""

    locations: Float[Array, ...]
    mask: Bool[Array, ...]
    n_events: Int[Array, ...]


class MarkedSpatialSampleResult(NamedTuple):
    """Marked spatial sample: ``(locations, mask, marks)``."""

    locations: Float[Array, ...]
    mask: Bool[Array, ...]
    marks: Float[Array, ...]


class SpatiotemporalSampleResult(NamedTuple):
    """Unmarked spatiotemporal sample: ``(locations, times, mask, n_events)``."""

    locations: Float[Array, ...]
    times: Float[Array, ...]
    mask: Bool[Array, ...]
    n_events: Int[Array, ...]


class MarkedSpatiotemporalSampleResult(NamedTuple):
    """Marked spatiotemporal sample: ``(locations, times, mask, marks)``."""

    locations: Float[Array, ...]
    times: Float[Array, ...]
    mask: Bool[Array, ...]
    marks: Float[Array, ...]


class GoodnessOfFit(NamedTuple):
    """Bundle of diagnostics returned by ``goodness_of_fit``.

    Attributes:
        residuals: Time-rescaled inter-event residuals :math:`\\tau_i`.
        mask: Real-event mask aligned with ``residuals``.
        ks_statistic: Kolmogorov–Smirnov statistic versus ``Exp(1)``.
        theoretical_quantiles: QQ-plot theoretical quantiles.
        empirical_quantiles: QQ-plot empirical quantiles.
    """

    residuals: Float[Array, ...]
    mask: Bool[Array, ...]
    ks_statistic: Float[Array, ...]
    theoretical_quantiles: Float[Array, ...]
    empirical_quantiles: Float[Array, ...]
