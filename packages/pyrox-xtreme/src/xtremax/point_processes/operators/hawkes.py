"""Equinox operators for self-exciting (Hawkes) temporal point processes.

Two concrete operators:

* :class:`ExponentialHawkes` — closed-form kernel
  :math:`\\phi(\\tau) = \\alpha e^{-\\beta \\tau}`, O(n) recursion for
  intensities (Ozaki 1979), exact compensator, adaptive thinning
  sampler with envelope :math:`\\mu + \\alpha N(t)`.
* :class:`GeneralHawkesProcess` — kernel supplied as an
  ``eqx.Module`` exposing ``.kernel(dt)`` and ``.kernel_integral(a, b)``
  (both optional — if ``kernel_integral`` is missing the operator
  falls back to trapezoid quadrature over the compensator).

Both follow the same "closed form when available, quadrature
otherwise" pattern used by :class:`InhomogeneousPoissonProcess` in
:mod:`~xtremax.point_processes.operators.temporal`. Parameters live as
PyTree leaves, not static fields, so gradients flow through
``eqx.filter_grad`` and NumPyro NUTS alike.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._history import EventHistory
from xtremax.point_processes._integration import integrate_log_intensity
from xtremax.point_processes.operators.temporal import GoodnessOfFit
from xtremax.point_processes.primitives.diagnostics import (
    compensator_curve,
    ks_statistic_exp1,
    qq_exp1_quantiles,
    time_rescaling_residuals,
)
from xtremax.point_processes.primitives.hawkes import (
    exp_hawkes_cumulative_intensity,
    exp_hawkes_intensity,
    exp_hawkes_log_prob,
    exp_hawkes_sample,
    general_hawkes_cumulative_intensity,
    general_hawkes_intensity,
    general_hawkes_log_prob,
    general_hawkes_sample,
)


class ExponentialKernel(eqx.Module):
    r"""Exponential excitation kernel :math:`\phi(\tau) = \alpha e^{-\beta \tau}`.

    Useful as a ``GeneralHawkesProcess`` kernel when you want to share
    the general-kernel plumbing (e.g. with a learnable multi-peak
    super-kernel made by summing modules) but still get the exact
    integral for the exponential component.
    """

    alpha: Float[Array, ...]
    beta: Float[Array, ...]

    def kernel(self, dt: Array) -> Array:
        return self.alpha * jnp.exp(-self.beta * jnp.clip(dt, 0.0, jnp.inf))

    def kernel_integral(self, a: Array, b: Array) -> Array:
        a_c = jnp.clip(a, 0.0, jnp.inf)
        b_c = jnp.clip(b, 0.0, jnp.inf)
        return (self.alpha / self.beta) * (
            jnp.exp(-self.beta * a_c) - jnp.exp(-self.beta * b_c)
        )

    def max_kernel(self) -> Array:
        return jnp.asarray(self.alpha)


class ExponentialHawkes(eqx.Module):
    r"""Hawkes process with exponential excitation on ``[0, T]``.

    Args:
        mu: Baseline rate ``μ > 0``.
        alpha: Excitation amplitude ``α >= 0``.
        beta: Excitation decay ``β > 0``.
        observation_window: Window length ``T > 0``.

    Notes:
        The process is **sub-critical** (stationary) iff
        :math:`\alpha / \beta < 1`; the sampler and log-likelihood are
        well-defined regardless, but super-critical branching ratios
        make the adaptive thinning envelope blow up and the sampler
        will often truncate at ``max_events``.
    """

    mu: Float[Array, ...]
    alpha: Float[Array, ...]
    beta: Float[Array, ...]
    observation_window: Float[Array, ...]

    def __init__(
        self,
        mu: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        observation_window: ArrayLike,
    ) -> None:
        self.mu = jnp.asarray(mu)
        self.alpha = jnp.asarray(alpha)
        self.beta = jnp.asarray(beta)
        self.observation_window = jnp.asarray(observation_window)

    # ------------------------------------------------------------
    # Core distribution API
    # ------------------------------------------------------------

    def log_prob(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Log-likelihood using Ozaki's O(n) recursion."""
        return exp_hawkes_log_prob(
            event_times, mask, self.observation_window, self.mu, self.alpha, self.beta
        )

    def sample(
        self,
        key: PRNGKeyArray,
        max_events: int,
        max_candidates: int | None = None,
    ) -> tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
        """Adaptive thinning sample; envelope ``μ + α N(t)``."""
        return exp_hawkes_sample(
            key,
            self.observation_window,
            self.mu,
            self.alpha,
            self.beta,
            max_events,
            max_candidates=max_candidates,
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
        """Conditional intensity ``λ*(t | H)``."""
        return exp_hawkes_intensity(
            t, event_times, mask, self.mu, self.alpha, self.beta
        )

    def cumulative_intensity(
        self,
        t: Float[Array, ...],
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Compensator ``Λ(t | H)``."""
        return exp_hawkes_cumulative_intensity(
            t, event_times, mask, self.mu, self.alpha, self.beta
        )

    def hazard(
        self,
        t: Float[Array, ...],
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Hazard equals intensity for a simple point process."""
        return self.intensity(t, event_times, mask)

    # ------------------------------------------------------------
    # Live-value accessors (parallels IPP pattern)
    # ------------------------------------------------------------

    def effective_lambda_max(self, history: EventHistory) -> Float[Array, ...]:
        """Thinning envelope ``μ + α N(t)`` read from live parameters."""
        n = jnp.sum(history.mask).astype(self.mu.dtype)
        return self.mu + self.alpha * n

    def effective_integrated_intensity(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        """Compensator ``Λ(T | H)`` evaluated at the window edge."""
        return self.cumulative_intensity(self.observation_window, event_times, mask)

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    def residuals(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> tuple[Float[Array, ...], Bool[Array, ...]]:
        """Time-rescaling residuals under the conditional compensator."""

        def cum_fn(ts: Array) -> Array:
            return self.cumulative_intensity(ts, event_times, mask)

        return time_rescaling_residuals(event_times, mask, cum_fn)

    def goodness_of_fit(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> GoodnessOfFit:
        residuals, res_mask = self.residuals(event_times, mask)
        ks = ks_statistic_exp1(residuals, res_mask)
        theoretical, empirical = qq_exp1_quantiles(residuals, res_mask)
        return GoodnessOfFit(residuals, res_mask, ks, theoretical, empirical)

    def compensator_curve(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> tuple[Float[Array, ...], Float[Array, ...]]:
        def cum_fn(ts: Array) -> Array:
            return self.cumulative_intensity(ts, event_times, mask)

        return compensator_curve(event_times, mask, cum_fn)


class GeneralHawkesProcess(eqx.Module):
    r"""Hawkes process with a user-supplied kernel module.

    Args:
        mu: Baseline rate ``μ > 0``.
        kernel: ``eqx.Module`` exposing ``.kernel(dt) -> Array``. When
            it also exposes ``.kernel_integral(a, b) -> Array`` the
            compensator uses the closed form (exact, differentiable);
            otherwise the operator falls back to trapezoid quadrature
            with ``n_integration_points`` nodes.
        observation_window: Window length.
        n_integration_points: Quadrature grid size for the fallback
            compensator and for diagnostics.
        max_kernel_value: Optional static upper bound on
            :math:`\phi(\tau)` over ``τ >= 0``. When ``None`` the
            operator tries ``kernel.max_kernel()`` and raises if
            absent. Used only by :meth:`sample` to build the thinning
            envelope ``μ + kernel_max · N(t)``.
    """

    mu: Float[Array, ...]
    kernel: eqx.Module
    observation_window: Float[Array, ...]
    n_integration_points: int = eqx.field(static=True, default=100)
    max_kernel_value: Float[Array, ...] | None = None

    def __init__(
        self,
        mu: ArrayLike,
        kernel: eqx.Module,
        observation_window: ArrayLike,
        n_integration_points: int = 100,
        max_kernel_value: ArrayLike | None = None,
    ) -> None:
        self.mu = jnp.asarray(mu)
        self.kernel = kernel
        self.observation_window = jnp.asarray(observation_window)
        self.n_integration_points = n_integration_points
        self.max_kernel_value = (
            None if max_kernel_value is None else jnp.asarray(max_kernel_value)
        )

    # ------------------------------------------------------------
    # Live accessors
    # ------------------------------------------------------------

    def _kernel_integral_fn(self) -> Callable[[Array, Array], Array]:
        integral = getattr(self.kernel, "kernel_integral", None)
        if integral is not None:
            return integral

        def _quadrature_integral(a: Array, b: Array) -> Array:
            # Build an ``integrate_log_intensity``-compatible log-kernel.
            def log_kernel(tau: Array) -> Array:
                return jnp.log(jnp.clip(self.kernel.kernel(tau), 1e-30, jnp.inf))

            return integrate_log_intensity(
                log_kernel, a, b, n_points=self.n_integration_points
            )

        return _quadrature_integral

    def effective_lambda_max(self, history: EventHistory) -> Float[Array, ...]:
        if self.max_kernel_value is not None:
            max_kernel = self.max_kernel_value
        else:
            fn = getattr(self.kernel, "max_kernel", None)
            if fn is None:
                raise ValueError(
                    "GeneralHawkesProcess.sample requires either "
                    "max_kernel_value on the operator or a "
                    "`.max_kernel()` method on the kernel module."
                )
            max_kernel = fn()
        n = jnp.sum(history.mask).astype(self.mu.dtype)
        return self.mu + max_kernel * n

    # ------------------------------------------------------------
    # Core distribution API
    # ------------------------------------------------------------

    def log_prob(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        return general_hawkes_log_prob(
            event_times,
            mask,
            self.observation_window,
            self.mu,
            self.kernel.kernel,
            self._kernel_integral_fn(),
        )

    def sample(
        self,
        key: PRNGKeyArray,
        max_events: int,
        max_candidates: int | None = None,
    ) -> tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
        # Capture the live bound to keep the sampler safe after updates.
        def lambda_max_fn(history: EventHistory) -> Array:
            return self.effective_lambda_max(history)

        return general_hawkes_sample(
            key=key,
            T=self.observation_window,
            mu=self.mu,
            kernel_fn=self.kernel.kernel,
            kernel_max_fn=lambda_max_fn,
            max_events=max_events,
            max_candidates=max_candidates,
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
        return general_hawkes_intensity(
            t, event_times, mask, self.mu, self.kernel.kernel
        )

    def cumulative_intensity(
        self,
        t: Float[Array, ...],
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        return general_hawkes_cumulative_intensity(
            t, event_times, mask, self.mu, self._kernel_integral_fn()
        )

    def effective_integrated_intensity(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> Float[Array, ...]:
        return self.cumulative_intensity(self.observation_window, event_times, mask)

    # ------------------------------------------------------------
    # Diagnostics (parallels ExponentialHawkes)
    # ------------------------------------------------------------

    def residuals(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> tuple[Float[Array, ...], Bool[Array, ...]]:
        def cum_fn(ts: Array) -> Array:
            return self.cumulative_intensity(ts, event_times, mask)

        return time_rescaling_residuals(event_times, mask, cum_fn)

    def goodness_of_fit(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> GoodnessOfFit:
        residuals, res_mask = self.residuals(event_times, mask)
        ks = ks_statistic_exp1(residuals, res_mask)
        theoretical, empirical = qq_exp1_quantiles(residuals, res_mask)
        return GoodnessOfFit(residuals, res_mask, ks, theoretical, empirical)

    def compensator_curve(
        self,
        event_times: Float[Array, ...],
        mask: Bool[Array, ...],
    ) -> tuple[Float[Array, ...], Float[Array, ...]]:
        def cum_fn(ts: Array) -> Array:
            return self.cumulative_intensity(ts, event_times, mask)

        return compensator_curve(event_times, mask, cum_fn)
