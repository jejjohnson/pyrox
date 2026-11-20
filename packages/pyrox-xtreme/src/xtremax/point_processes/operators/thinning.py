"""Equinox operator that thins a base temporal point process.

A :class:`ThinningProcess` wraps any temporal operator (IPP, Hawkes,
renewal, marked, …) together with a retention callable
``p(t | H, m) ∈ [0, 1]``. Its conditional intensity is
:math:`\\lambda_\\text{thin}(t | H) = p(t | H) \\cdot \\lambda_\\text{base}(t | H)`,
so under separability the log-likelihood of *observed* events
decomposes as

.. math::
    \\log L_\\text{thin} = \\sum_i \\log p(t_i) \\lambda(t_i) - \\int_0^T
        p \\lambda = \\log L_\\text{base}(\\{t_i\\})
        + \\sum_i \\log p(t_i | H_i, m_i)
        + \\int_0^T (1 - p(t | H)) \\lambda_\\text{base}(t | H)\\, dt.

The first two terms are exact given the observed sequence; the correction
integral is **added** — the base likelihood subtracts the full compensator
:math:`\\int \\lambda`, but only :math:`\\int p\\lambda` belongs to the
thinned process, so :math:`\\int (1-p)\\lambda` must be restored. It is
computed by trapezoid quadrature through
:func:`~xtremax.point_processes.primitives.thinning.retention_compensator`.

.. warning::
    For **history-dependent** bases (Hawkes, renewal) this ``log_prob``
    is a *pseudo-likelihood*: the base intensity is conditioned on the
    *retained* events only, while :meth:`ThinningProcess.sample` thins a
    full *latent* realisation whose unobserved (thinned-away) events also
    excite the base. The decomposition is exact only for Poisson-family
    bases (HPP/IPP), whose intensity does not depend on history. Fits
    against history-dependent bases should be read as approximate.

The retention callable is stored as a plain PyTree leaf so any
parameters inside a learnable observation operator flow through
``eqx.filter_grad`` / NUTS.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._history import EventHistory
from xtremax.point_processes._results import MarkedSampleResult, SampleResult
from xtremax.point_processes.operators._base import (
    _n_positional_params,
    call_sampler,
    call_sequence_log_prob,
)
from xtremax.point_processes.primitives.thinning import (
    retention_compensator,
    thinning_retention_log_prob,
)


class ThinningProcess(eqx.Module):
    """Thinned TPP: base ground process + retention callable.

    Args:
        base: Any temporal operator exposing ``log_prob(times, mask)``
            / ``sample(key, max_events)``. The operator must also
            expose a ``conditional_intensity_fn`` way for the retention
            compensator to query :math:`\\lambda_\\text{base}(t|H)` —
            see :meth:`_base_intensity_fn`.
        retention_fn: Callable
            ``(t, history, proposed_mark=None) -> p``. Returns the
            retention probability in ``[0, 1]``. ``history`` is the
            observed (retained) history up to ``t``.
        observation_window: Window length. Defaults to
            ``base.observation_window`` when ``None``.
        n_integration_points: Trapezoid grid size for the retention
            compensator.
    """

    base: eqx.Module
    retention_fn: Callable[..., Array]
    observation_window: Float[Array, ...]
    n_integration_points: int = eqx.field(static=True, default=100)

    def __init__(
        self,
        base: eqx.Module,
        retention_fn: Callable[..., Array],
        observation_window: ArrayLike | None = None,
        n_integration_points: int = 100,
    ) -> None:
        self.base = base
        self.retention_fn = retention_fn
        if observation_window is None:
            observation_window = base.observation_window
        self.observation_window = jnp.asarray(observation_window)
        self.n_integration_points = n_integration_points

    # ------------------------------------------------------------
    # Intensity adapters (try to find a base-specific form, fall back
    # to a homogeneous surrogate with rate == expected-count / T)
    # ------------------------------------------------------------

    def _base_intensity_fn(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Callable[[Array, EventHistory], Array]:
        """Return ``(t, history) -> λ_base(t | H_observed)``.

        Tries, in order: the base operator's own ``intensity`` method
        with a ``(t, event_times, mask)`` signature (Hawkes, renewal);
        ``intensity(t)`` (IPP-style); for a
        :class:`MarkedTemporalPointProcess` base, the ground operator's
        intensity (so the compensator term uses the ground-process
        rate, matching the separable factorisation
        :math:`\\lambda(t, m) = \\lambda_g(t) \\cdot f(m \\mid t)`);
        and finally the constant ``base.rate`` for HPP. This lets
        :class:`ThinningProcess` wrap any temporal operator shipped
        with the package without needing a family-specific subclass.
        """
        # If the base is a marked TPP, fall through to its ground
        # operator. Under separable marks the ground intensity is the
        # correct quantity to multiply by the mark-averaged retention.
        target = getattr(self.base, "ground", self.base)

        # Dispatch on the intensity method's signature — a Python-level
        # inspect check outside any traced code. The previous
        # ``try/except TypeError`` probing swallowed genuine
        # ``TypeError``s raised *inside* user intensity code and
        # silently rerouted them to the next fallback.
        intensity = getattr(target, "intensity", None)
        if intensity is not None:
            n_params = _n_positional_params(intensity)
            if n_params >= 3:  # (t, event_times, mask) — Hawkes, renewal

                def _fn_history(t: Array, history: EventHistory) -> Array:
                    return target.intensity(t, event_times, mask)

                return _fn_history
            if n_params >= 1:  # (t,) — IPP-style

                def _fn_ipp(t: Array, history: EventHistory) -> Array:
                    return target.intensity(t)

                return _fn_ipp

        # Fallback: constant rate from ``target.rate`` (HPP).
        rate = getattr(target, "rate", None)
        if rate is None:
            raise AttributeError(
                "ThinningProcess.base must expose `intensity(t, ...)` or "
                "a `rate` attribute (directly, or on a ``ground`` "
                "operator); got a base with neither."
            )

        def _fn_const(t: Array, history: EventHistory) -> Array:
            del t, history
            return jnp.asarray(rate)

        return _fn_const

    # ------------------------------------------------------------
    # Core distribution API
    # ------------------------------------------------------------

    def log_prob(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
        marks: Float[Array, ...] | None = None,
        *,
        mark_sample_key: PRNGKeyArray | None = None,
        n_mark_samples: int = 8,
    ) -> Float[Array, ...]:
        """Log-likelihood of the observed (retained) events.

        See the module docstring for the decomposition. Supports a
        marked base via the optional ``marks`` argument — the retention
        callable is then invoked with ``(t, history, mark)`` so e.g.
        magnitude-gated detectors work out of the box.

        For **mark-dependent** retention against a marked base, pass
        ``mark_sample_key`` so the compensator integral Monte-Carlo-
        averages retention over the base's mark distribution at each
        quadrature node. Without a key, ``retention_fn`` is called with
        ``proposed_mark=None`` in the compensator path (correct only
        for mark-independent retention).
        """
        base_logp = self._call_base_log_prob(event_times, mask, marks=marks)

        # Promote scalar marks to 2-D inside EventHistory so that
        # history-dependent retention callables can uniformly look at
        # ``history.marks`` without branching on dimensionality.
        history_marks = None
        if marks is not None:
            history_marks = (
                jnp.expand_dims(marks, axis=-1) if marks.ndim == mask.ndim else marks
            )
        history = EventHistory(times=event_times, mask=mask, marks=history_marks)
        retention_term = thinning_retention_log_prob(
            self.retention_fn, event_times, mask, history, marks=marks
        )

        intensity_fn = self._base_intensity_fn(event_times, mask)

        # If the user wants MC-marginalised retention and the base is
        # a marked process with a ``mark_distribution_fn``, build the
        # sampler by delegating to it.
        mark_sampler_fn = None
        if mark_sample_key is not None:
            base_mark_fn = getattr(self.base, "mark_distribution_fn", None)
            if base_mark_fn is not None:

                def mark_sampler_fn(
                    t: Array,
                    prefix: EventHistory,
                    k: PRNGKeyArray,
                ) -> Array:
                    d = base_mark_fn(t, prefix)
                    return d.sample(k)

        correction = retention_compensator(
            self.retention_fn,
            intensity_fn,
            history,
            self.observation_window,
            n_points=self.n_integration_points,
            mark_sampler_fn=mark_sampler_fn,
            mark_sample_key=mark_sample_key,
            n_mark_samples=n_mark_samples,
        )
        # The base log_prob subtracted the FULL compensator ∫λ; the thinned
        # process owes only ∫pλ, so the difference ∫(1-p)λ is added back.
        # (Subtracting it here — the old sign — made every fit maximize the
        # wrong objective, off by exactly 2∫(1-p)λ.)
        return base_logp + retention_term + correction

    def _call_base_log_prob(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
        marks: Float[Array, ...] | None,
    ) -> Float[Array, ...]:
        """Forward to the base ``log_prob`` matching its signature.

        Signature-based dispatch (see
        :func:`~xtremax.point_processes.operators._base.call_sequence_log_prob`).
        """
        return call_sequence_log_prob(self.base, event_times, mask, marks=marks)

    def sample(
        self,
        key: PRNGKeyArray,
        max_events: int,
        max_candidates: int | None = None,
        **base_kwargs,
    ) -> (
        tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]
        | tuple[Float[Array, ...], Bool[Array, ...], Float[Array, ...]]
    ):
        """Sample by first drawing from the base and then Bernoulli-thinning.

        Two-pass draw: (1) draw a full latent realisation from the
        base; (2) for each latent event evaluate the retention
        probability given the *retained* history so far and accept
        with that probability. This correctly propagates history- and
        mark-dependent retention. ``max_candidates`` is forwarded to
        the base's own sampler for families (Hawkes, IPP) that use
        thinning internally.

        Returns:
            ``(times, mask, n_retained)`` for an unmarked base, or
            ``(times, mask, retained_marks)`` when the base is a marked
            process. The third element disambiguates the two cases.
        """
        key_base, key_thin = jax.random.split(key)

        # Base sampler signatures are not uniform; the shared helper
        # dispatches on the base's actual parameter names (see
        # :func:`~xtremax.point_processes.operators._base.call_sampler`).
        base_result = call_sampler(
            self.base,
            key_base,
            max_events,
            max_candidates=max_candidates,
            **base_kwargs,
        )

        # Disambiguate the base result by TYPE first — package samplers
        # return :class:`SampleResult` (unmarked) or
        # :class:`MarkedSampleResult` (marked), so the dispatch is
        # unambiguous even for batched samplers. Bare tuples from
        # user-supplied bases fall back to the legacy shape/dtype
        # heuristic (0-D integer third element ⇒ event count).
        latent_marks = None
        if isinstance(base_result, MarkedSampleResult):
            latent_times, latent_mask, latent_marks = base_result
        elif isinstance(base_result, SampleResult):
            latent_times, latent_mask, _ = base_result
        elif len(base_result) == 2:
            latent_times, latent_mask = base_result
        elif len(base_result) == 3:
            latent_times, latent_mask, third = base_result
            third_arr = jnp.asarray(third)
            if third_arr.ndim == 0 and jnp.issubdtype(third_arr.dtype, jnp.integer):
                latent_marks = None
            else:
                latent_marks = third
        else:
            latent_times, latent_mask = base_result[:2]
            latent_marks = base_result[-1]

        keys = jax.random.split(key_thin, latent_times.shape[-1])

        if latent_marks is None:
            # Plain times + mask scan; retention gets no proposed_mark.
            def step_unmarked(
                carry: EventHistory,
                inp: tuple[Array, Array, PRNGKeyArray],
            ) -> tuple[EventHistory, None]:
                t_i, valid_i, key_i = inp
                p = self.retention_fn(t_i, carry, None)
                u = jax.random.uniform(key_i, dtype=t_i.dtype)
                accepted = valid_i & (u < jnp.clip(p, 0.0, 1.0))
                new_history = carry.append(t_i, accepted=accepted)
                return new_history, None

            initial_history = EventHistory.empty(
                max_events=max_events, mark_dim=None, dtype=latent_times.dtype
            )
            final_history, _ = jax.lax.scan(
                step_unmarked,
                initial_history,
                (
                    jnp.moveaxis(latent_times, -1, 0),
                    jnp.moveaxis(latent_mask, -1, 0),
                    keys,
                ),
            )
        else:
            # Marked base: thread both the proposed mark into the
            # retention evaluation and the retained marks into the
            # carry's history buffer.
            latent_marks_2d = (
                latent_marks if latent_marks.ndim == 2 else latent_marks[..., None]
            )
            mark_dim = latent_marks_2d.shape[-1]

            def step_marked(
                carry: EventHistory,
                inp: tuple[Array, Array, PRNGKeyArray, Array],
            ) -> tuple[EventHistory, None]:
                t_i, valid_i, key_i, mark_row = inp
                # Pass the user a mark in their original shape (scalar
                # if marks were originally 1-D).
                proposed_mark = mark_row[0] if latent_marks.ndim == 1 else mark_row
                p = self.retention_fn(t_i, carry, proposed_mark)
                u = jax.random.uniform(key_i, dtype=t_i.dtype)
                accepted = valid_i & (u < jnp.clip(p, 0.0, 1.0))
                new_history = carry.append(t_i, mark=mark_row, accepted=accepted)
                return new_history, None

            initial_history = EventHistory.empty(
                max_events=max_events,
                mark_dim=mark_dim,
                dtype=latent_times.dtype,
                # Marks keep their own dtype — discrete (integer) marks
                # were previously cast to the float time dtype.
                mark_dtype=latent_marks_2d.dtype,
            )
            final_history, _ = jax.lax.scan(
                step_marked,
                initial_history,
                (
                    jnp.moveaxis(latent_times, -1, 0),
                    jnp.moveaxis(latent_mask, -1, 0),
                    keys,
                    jnp.moveaxis(latent_marks_2d, -2, 0),
                ),
            )

        times = jnp.where(
            final_history.mask, final_history.times, self.observation_window
        )
        if latent_marks is None:
            n_retained = jnp.sum(final_history.mask).astype(jnp.int32)
            return SampleResult(times, final_history.mask, n_retained)
        # Marked base: return retained marks in the user's original
        # shape (scalar marks flatten the trailing ``1`` dimension).
        retained_marks = final_history.marks
        if latent_marks.ndim == 1:
            retained_marks = retained_marks[..., 0]
        return MarkedSampleResult(times, final_history.mask, retained_marks)
