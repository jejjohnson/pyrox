"""Shared mixins and bases for the point-process operator families.

The operator modules used to copy-paste the diagnostics trio
(``residuals`` / ``goodness_of_fit`` / ``compensator_curve``) five
times, the live-``λ_max`` accessor three times, and the separable
marked-process plumbing three times. That duplication was the direct
cause of two high-severity bugs (fixes that landed in only some of the
copies), so the shared surface now lives here exactly once:

* :class:`GoodnessOfFitMixin` — needs only ``_compensator_fn``.
* :class:`LiveIntensityMixin` — pinned ``λ_max`` → module
  ``.max_intensity()`` → raise.
* :class:`SeparableMarkedPP` — ground × mark-distribution composition
  parameterised by the event-coordinate arity ((t) / (s) / (s, t)).

The mixins are plain classes (no fields), so they compose with
``equinox.Module`` subclasses without affecting the PyTree structure.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from xtremax.point_processes._results import GoodnessOfFit
from xtremax.point_processes.primitives.diagnostics import (
    compensator_curve,
    ks_statistic_exp1,
    qq_exp1_quantiles,
    time_rescaling_residuals,
)


def _n_positional_params(fn: Callable[..., object]) -> int:
    """Count the callable's positional parameters (Python-level, un-traced)."""
    params = inspect.signature(fn).parameters.values()
    return sum(
        1 for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )


def call_sequence_log_prob(
    op: eqx.Module,
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    marks: Float[Array, ...] | None = None,
) -> Float[Array, ...]:
    """Call ``op.log_prob`` with whichever signature the operator exposes.

    Dispatch is by ``inspect.signature`` (a Python-level check, outside
    any traced code): sequence-based operators (IPP, Hawkes, renewal)
    take ``(times, mask)`` — and marked ones ``(times, mask, marks)`` —
    while HPP-style operators take only the event count. The previous
    ``try/except TypeError`` probing swallowed genuine ``TypeError``s
    raised *inside* user code and silently rerouted them to the count
    fallback.
    """
    n_params = _n_positional_params(op.log_prob)
    if marks is not None and n_params >= 3:
        return op.log_prob(event_times, mask, marks)
    if n_params >= 2:
        return op.log_prob(event_times, mask)
    return op.log_prob(jnp.sum(mask, axis=-1))


def call_sampler(
    op: eqx.Module,
    key: PRNGKeyArray,
    max_events: int,
    max_candidates: int | None = None,
    **kwargs,
) -> tuple[Array, ...]:
    """Call ``op.sample`` handling the non-uniform sizing signatures.

    Hawkes-style samplers take ``(key, max_events, max_candidates=...)``
    while IPP-style thinning samplers size their buffer with a single
    ``max_candidates`` positional. Passing ``max_events`` positionally
    into the latter used to collide with a forwarded ``max_candidates``
    kwarg and raise ``TypeError``. Dispatch on the sampler's actual
    parameter names (a Python-level inspect check, outside traced code).
    """
    params = list(inspect.signature(op.sample).parameters)
    sizing_param = params[1] if len(params) > 1 else None
    if sizing_param == "max_candidates":
        n_candidates = max_candidates if max_candidates is not None else max_events
        return op.sample(key, n_candidates, **kwargs)
    if max_candidates is not None:
        kwargs["max_candidates"] = max_candidates
    return op.sample(key, max_events, **kwargs)


class GoodnessOfFitMixin:
    """Time-rescaling diagnostics defined once for every temporal family.

    Requires the concrete operator to provide
    ``_compensator_fn(event_times, mask) -> Callable[[ts], Λ(ts)]`` —
    the (possibly history-conditioned) compensator evaluated at
    arbitrary times.
    """

    def _compensator_fn(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Callable[[Array], Array]:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _compensator_fn to use "
            "the GoodnessOfFitMixin diagnostics."
        )

    def residuals(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> tuple[Float[Array, ...], Bool[Array, ...]]:
        """Time-rescaling residuals under the family's compensator."""
        fn = self._compensator_fn(event_times, mask)
        return time_rescaling_residuals(event_times, mask, fn)

    def goodness_of_fit(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> GoodnessOfFit:
        """Bundle residuals + KS + QQ for plotting / hypothesis testing."""
        residuals, res_mask = self.residuals(event_times, mask)
        ks = ks_statistic_exp1(residuals, res_mask)
        theoretical, empirical = qq_exp1_quantiles(residuals, res_mask)
        return GoodnessOfFit(residuals, res_mask, ks, theoretical, empirical)

    def compensator_curve(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> tuple[Float[Array, ...], Float[Array, ...]]:
        """Pairs ``(t_i, Λ(t_i))`` for a compensator plot."""
        fn = self._compensator_fn(event_times, mask)
        return compensator_curve(event_times, mask, fn)


class LiveIntensityMixin:
    """Shared live-``λ_max`` accessor for the three IPP families.

    Precedence: pinned :attr:`lambda_max` → the intensity module's
    ``.max_intensity()`` → raise. Reading the module every call keeps
    thinning samplers safe after an optimiser update. There is **no**
    automatic fallback (e.g. ``2Λ/|D|``): no derived estimate can
    guarantee a true upper bound for an arbitrary intensity, and an
    under-estimate silently biases the sampler low in intensity peaks.
    """

    def effective_lambda_max(self) -> Float[Array, ...]:
        """Return the current thinning bound (see class docstring)."""
        if self.lambda_max is not None:
            return self.lambda_max
        max_intensity = getattr(self.log_intensity_fn, "max_intensity", None)
        if max_intensity is not None:
            return max_intensity()
        raise ValueError(
            "Cannot sample via thinning: no `lambda_max` pinned on the "
            "operator and `log_intensity_fn` has no `.max_intensity()` "
            "method. Pass a bound at construction (it must be a true "
            "upper bound on the intensity over the domain — too small a "
            "value silently biases the sampler low in intensity peaks)."
        )


class SeparableMarkedPP(eqx.Module):
    """Base for separable marked point processes.

    Holds the ground operator and the mark-distribution callable, and
    implements the coordinate-arity-independent plumbing: joint
    ``sample`` (ground events, then marks) and the joint
    ``mark_intensity``. Concrete subclasses fix the event-coordinate
    tuple — ``(t,)`` temporal, ``(s,)`` spatial, ``(s, t)``
    spatiotemporal — by implementing :meth:`_sample_marks` and (for
    the intensity product) exposing a ground ``intensity``.
    """

    ground: eqx.Module
    mark_distribution_fn: Callable[..., object]
    mark_dim: int | None = eqx.field(static=True, default=None)

    def _sample_marks(
        self,
        key: PRNGKeyArray,
        *event_coords: Array,
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _sample_marks."
        )

    def _sample_joint(
        self,
        key: PRNGKeyArray,
        max_events: int,
        **ground_kwargs,
    ) -> tuple[Array, ...]:
        """Ground events then marks: ``(*coords, mask, marks)``.

        Works for every coordinate arity because all ground samplers
        return ``(*coords, mask, n)`` with the mask in the penultimate
        slot. The ground call goes through :func:`call_sampler` so
        IPP-style (``max_candidates``-sized) grounds compose too.
        """
        key_ground, key_marks = jax.random.split(key)
        max_candidates = ground_kwargs.pop("max_candidates", None)
        ground_result = call_sampler(
            self.ground,
            key_ground,
            max_events,
            max_candidates=max_candidates,
            **ground_kwargs,
        )
        *coords, mask, _ = ground_result
        marks = self._sample_marks(key_marks, *coords, mask=mask)
        return (*coords, mask, marks)

    def mark_intensity(
        self,
        *coords_and_marks: Array,
    ) -> Float[Array, ...]:
        """Joint intensity ``λ(coords, m) = λ(coords) · f(m | coords)``.

        Useful for diagnostic plots (mark-conditional rate maps) and
        for mark-thinned sub-process intensities. The final positional
        argument is the marks array; everything before it is the
        event-coordinate tuple.
        """
        *coords, marks = coords_and_marks
        ground_lam = self.ground.intensity(*coords)

        def mark_density(*args: Array) -> Array:
            *cs, m = args
            return jnp.exp(self.mark_distribution_fn(*cs).log_prob(m))

        densities = jax.vmap(mark_density)(*coords, marks)
        return ground_lam * densities
