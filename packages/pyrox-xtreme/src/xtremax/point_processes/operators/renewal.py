"""Equinox operator for 1-D renewal temporal point processes.

Bundles an inter-event NumPyro Distribution with numerical defaults
and exposes the standard TPP API (log-likelihood, sampling, hazard,
compensator, diagnostics). The inter-event distribution is stored on
the module directly so its parameters participate in the PyTree and
are trainable via ``eqx.filter_grad`` / optax / NUTS.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes.operators.temporal import GoodnessOfFit
from xtremax.point_processes.primitives.diagnostics import (
    compensator_curve,
    ks_statistic_exp1,
    qq_exp1_quantiles,
    time_rescaling_residuals,
)
from xtremax.point_processes.primitives.renewal import (
    renewal_cumulative_hazard,
    renewal_expected_count,
    renewal_hazard,
    renewal_intensity,
    renewal_inter_event_log_prob,
    renewal_log_prob,
    renewal_sample,
    renewal_survival,
)


class RenewalProcess(eqx.Module):
    """Renewal process on ``[0, T]`` with a user-supplied inter-event law.

    Args:
        inter_event_dist: NumPyro Distribution for inter-event times.
            Must implement ``sample``, ``log_prob``, and ``cdf``. Any
            Distribution parameters that are JAX arrays remain live
            PyTree leaves.
        observation_window: Window length ``T > 0``.
        n_integration_points: Grid size used by compensator-based
            diagnostics that cannot be reduced to the hazard closed
            form.
    """

    inter_event_dist: dist.Distribution
    observation_window: Float[Array, ...]
    n_integration_points: int = eqx.field(static=True, default=100)

    def __init__(
        self,
        inter_event_dist: dist.Distribution,
        observation_window: ArrayLike,
        n_integration_points: int = 100,
    ) -> None:
        self.inter_event_dist = inter_event_dist
        self.observation_window = jnp.asarray(observation_window)
        self.n_integration_points = n_integration_points

    # ------------------------------------------------------------
    # Core distribution API
    # ------------------------------------------------------------

    def log_prob(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Joint event-time log-likelihood."""
        return renewal_log_prob(
            event_times, mask, self.observation_window, self.inter_event_dist
        )

    def sample(
        self,
        key: PRNGKeyArray,
        max_events: int,
    ) -> tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
        """Sample event times by accumulating iid gaps."""
        return renewal_sample(
            key, self.inter_event_dist, self.observation_window, max_events
        )

    # ------------------------------------------------------------
    # Intensity, hazard, survival
    # ------------------------------------------------------------

    def intensity(
        self,
        t: Float[Array, ...],
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Conditional intensity ``λ*(t) = h(t − t_{N(t)})``."""
        return renewal_intensity(t, event_times, mask, self.inter_event_dist)

    def hazard(self, tau: Float[Array, ...]) -> Float[Array, ...]:
        """Inter-event hazard ``h(τ) = f(τ)/S(τ)``."""
        return renewal_hazard(tau, self.inter_event_dist)

    def cumulative_hazard(self, tau: Float[Array, ...]) -> Float[Array, ...]:
        """``Λ(τ) = −log S(τ)``."""
        return renewal_cumulative_hazard(tau, self.inter_event_dist)

    def survival(self, tau: Float[Array, ...]) -> Float[Array, ...]:
        """Inter-event survival ``S(τ) = 1 − F(τ)``."""
        return renewal_survival(tau, self.inter_event_dist)

    def inter_event_log_prob(self, tau: Float[Array, ...]) -> Float[Array, ...]:
        """Log density of a single inter-event gap."""
        return renewal_inter_event_log_prob(tau, self.inter_event_dist)

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------

    def predict_count(
        self,
        start_time: Float[Array, ...] | float = 0.0,
        end_time: Float[Array, ...] | None = None,
    ) -> Float[Array, ...]:
        r"""Expected event count on ``[start, end]`` via renewal equation."""
        if end_time is None:
            end_time = self.observation_window
        T = jnp.asarray(end_time) - jnp.asarray(start_time)
        return renewal_expected_count(
            T, self.inter_event_dist, n_points=self.n_integration_points
        )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    def _cumulative_intensity_fn(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ):
        """Build the compensator at observed event times.

        The renewal compensator is a staircase-like cumulative hazard
        evaluated on inter-event gaps; we reconstruct it at the
        supplied events and return a callable for the diagnostics
        helpers, which expect ``cumulative_intensity_fn(event_times)``.
        """
        zero = jnp.zeros_like(event_times[..., :1])
        prev = jnp.concatenate([zero, event_times[..., :-1]], axis=-1)
        gaps = event_times - prev
        per_gap_hazard = renewal_cumulative_hazard(
            jnp.where(mask, gaps, 0.0), self.inter_event_dist
        )
        cum = jnp.cumsum(per_gap_hazard, axis=-1)

        def _fn(ts: Array) -> Array:
            del ts  # the diagnostics helper ignores the value and uses shape
            return cum

        return _fn

    def residuals(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> tuple[Float[Array, ...], Bool[Array, ...]]:
        """Inter-event-gap-based residuals ``τᵢ = Λ_F(Δtᵢ)``."""
        fn = self._cumulative_intensity_fn(event_times, mask)
        return time_rescaling_residuals(event_times, mask, fn)

    def goodness_of_fit(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> GoodnessOfFit:
        """Residuals + KS + QQ against ``Exp(1)``."""
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
        fn = self._cumulative_intensity_fn(event_times, mask)
        return compensator_curve(event_times, mask, fn)
