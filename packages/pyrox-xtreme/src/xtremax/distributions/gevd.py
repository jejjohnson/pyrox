"""
Comprehensive Generalized Extreme Value Distribution (GEVD) for NumPyro

This module provides a robust implementation of the GEVD with extensive statistical
methods and proper NumPyro integration.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes, validate_sample

from xtremax.distributions._support import concrete_sign
from xtremax.primitives._common import GUMBEL_THRESHOLD
from xtremax.primitives._moments import (
    gev_excess_kurtosis,
    gev_skewness,
    gev_variance,
)
from xtremax.primitives.gev import (
    gev_cdf,
    gev_icdf,
    gev_log_prob,
    gev_log_survival,
    gev_mean,
    gev_return_level,
    gev_survival,
)


# Minimal scalar-or-array alias for broadcastable GEVD parameters.
ArrayLike = float | jnp.ndarray
from xtremax._rng import check_prng_key


class GeneralizedExtremeValueDistribution(dist.Distribution):
    """
    Generalized Extreme Value Distribution (GEVD) for NumPyro.

    The GEVD is a family of continuous probability distributions developed within
    extreme value theory. It encompasses three types of extreme value distributions:

    - Type I (Gumbel): ξ = 0, exponential tails
    - Type II (Fréchet): ξ > 0, heavy polynomial tails
    - Type III (Weibull): ξ < 0, bounded upper tail

    **Probability Density Function:**

    For ξ ≠ 0:
        f(x) = (1/σ) * t(x)^(-(1/ξ + 1)) * exp(-t(x)^(-1/ξ))
        where t(x) = 1 + ξ(x - μ)/σ

    For ξ = 0 (Gumbel limit):
        f(x) = (1/σ) * exp(-(x - μ)/σ) * exp(-exp(-(x - μ)/σ))

    **Cumulative Distribution Function:**

    For ξ ≠ 0:
        F(x) = exp(-t(x)^(-1/ξ))
        where t(x) = 1 + ξ(x - μ)/σ

    For ξ = 0:
        F(x) = exp(-exp(-(x - μ)/σ))

    **Support:**

    - ξ > 0: x ≥ μ - σ/ξ (lower bounded)
    - ξ = 0: x ∈ ℝ (unbounded)
    - ξ < 0: x ≤ μ - σ/ξ (upper bounded)

    Parameters:
        loc (float): Location parameter μ ∈ ℝ
        scale (float): Scale parameter σ > 0
        shape (float): Shape parameter ξ ∈ ℝ

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> # Create GEVD instances for each type
        >>> gumbel = GeneralizedExtremeValueDistribution(loc=0, scale=1, shape=0)
        >>> frechet = GeneralizedExtremeValueDistribution(loc=0, scale=1, shape=0.2)
        >>> weibull = GeneralizedExtremeValueDistribution(loc=0, scale=1, shape=-0.2)
        >>>
        >>> # Sample and evaluate
        >>> key = jax.random.PRNGKey(42)
        >>> samples = frechet.sample(key, sample_shape=(1000,))
        >>> log_probs = frechet.log_prob(samples)
    """

    # NumPyro distribution interface requirements
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "concentration": constraints.real,
    }
    reparametrized_params = ["loc", "scale", "concentration"]

    def __init__(
        self,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        concentration: ArrayLike | None = None,
        shape: ArrayLike | None = None,
        validate_args: bool | None = None,
    ):
        """
        Initialize the Generalized Extreme Value Distribution.

        Args:
            loc: Location parameter μ (real number)
            scale: Scale parameter σ (positive real number)
            concentration: Shape parameter ξ (real number)
            shape: Deprecated backward-compatible alias for ``concentration``
            validate_args: Whether to validate input arguments

        Raises:
            ValueError: If scale <= 0
        """
        if shape is not None:
            if concentration is not None:
                raise ValueError(
                    "Pass only one of 'concentration' or the deprecated 'shape' alias."
                )
            warnings.warn(
                "'shape' is deprecated; use 'concentration' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            concentration = shape

        if concentration is None:
            concentration = 0.0

        self.loc, self.scale, self.concentration = promote_shapes(
            loc, scale, concentration
        )

        # Determine batch shape from broadcasted parameters
        batch_shape = lax.broadcast_shapes(
            jnp.shape(self.loc), jnp.shape(self.scale), jnp.shape(self.concentration)
        )

        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(
        self,
        key: jnp.ndarray,
        sample_shape: tuple = (),
        *,
        shape: tuple | None = None,
    ) -> jnp.ndarray:
        """
        Generate samples from the GEVD using inverse transform sampling.

        The sampling uses the quantile function (inverse CDF):

        For ξ ≠ 0:
            Q(p) = μ + (σ/ξ) * ((-ln(p))^(-ξ) - 1)

        For ξ = 0:
            Q(p) = μ - σ * ln(-ln(p))

        Args:
            key: JAX random key for sampling
            sample_shape: Shape of samples to generate
            shape: Keyword-only alias for ``sample_shape`` matching the
                ``pipekit_cycle.ObservationNoise`` protocol's parameter
                name (``sample(key, shape)``), so callers typed against
                that protocol can pass it by keyword. Passing both a
                non-empty ``sample_shape`` and ``shape`` raises
                ``ValueError``.

        Returns:
            Array of samples from the GEVD
        """
        if shape is not None:
            if sample_shape != ():
                raise ValueError(
                    "Pass only one of 'sample_shape' (NumPyro spelling) or "
                    "'shape' (ObservationNoise protocol spelling), not both."
                )
            sample_shape = shape
        check_prng_key(key)
        extended_shape = sample_shape + self.batch_shape

        # JAX's Uniform(0, 1) sampler can emit exact 0 or 1 at the
        # endpoints; passing those to icdf yields -inf/+inf and poisons
        # downstream computations. Clamp away from the endpoints.
        uniform_samples = dist.Uniform(0.0, 1.0).sample(key, extended_shape)
        eps = jnp.finfo(uniform_samples.dtype).eps
        uniform_samples = jnp.clip(uniform_samples, eps, 1.0 - eps)

        # Apply inverse CDF transformation
        return self.icdf(uniform_samples)

    @validate_sample
    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        """Log PDF. Thin wrapper for :func:`~xtremax.primitives.gev.gev_log_prob`."""
        return gev_log_prob(value, self.loc, self.scale, self.concentration)

    def cdf(self, value: jnp.ndarray) -> jnp.ndarray:
        """CDF. Thin wrapper for :func:`~xtremax.primitives.gev.gev_cdf`."""
        return gev_cdf(value, self.loc, self.scale, self.concentration)

    def icdf(self, q: jnp.ndarray) -> jnp.ndarray:
        """Quantile function. Thin wrapper for ``gev_icdf``."""
        return gev_icdf(q, self.loc, self.scale, self.concentration)

    @property
    def support(self) -> constraints.Constraint:
        """
        Return the support of the distribution.

        Returns:
            Constraint object reflecting the shape-dependent support:
            - ξ > 0: [μ - σ/ξ, +∞) → ``greater_than_eq(lower_bound)``
              (closed, matching the old ``interval`` membership; the
              density vanishes at the endpoint so ``log_prob`` is -inf)
            - ξ = 0: (-∞, +∞) → ``real``
            - ξ < 0: (-∞, μ - σ/ξ] → ``less_than_eq(upper_bound)`` (closed:
              the endpoint density is finite at ξ = -1 and diverges for
              ξ < -1, so the endpoint is a genuine support point)

        The constraint kind is dispatched on the *statically known* sign of
        the concentration so that ``biject_to(support)`` returns a bijector
        that genuinely handles the half-unbounded geometry — the previous
        ``interval(lo, hi)`` construction mapped to ``Sigmoid ∘ Affine``
        which yields inf/nan when a bound is infinite, breaking every
        NUTS/SVI model that used this distribution as a latent site.

        When the concentration is traced (inside ``jit``/inference) or a
        mixed-sign batch, the sign is not statically known and the support
        falls back to ``constraints.real``: membership then accepts points
        outside the true parameter-dependent support, but ``log_prob``
        returns ``-inf`` there, so inference remains correct (init simply
        rejects such draws).
        """
        sign = concrete_sign(self.concentration)
        if sign == 1:
            return constraints.greater_than_eq(self.lower_bound())
        if sign == -1:
            return constraints.less_than_eq(self.upper_bound())
        return constraints.real

    def upper_bound(self) -> jnp.ndarray:
        """
        Compute the upper bound of the support.

        Returns:
            Upper bound: μ - σ/ξ for ξ < 0, +∞ otherwise
        """
        loc = jnp.asarray(self.loc)
        scale = jnp.asarray(self.scale)
        shape = jnp.asarray(self.concentration)
        # Replace ξ=0 with a safe placeholder (1.0) inside the division
        # so plain Python inputs don't raise `ZeroDivisionError` before
        # `jnp.where` selects the +∞ branch. The non-finite branch value
        # is masked out.
        safe_shape = jnp.where(shape == 0.0, 1.0, shape)
        bounded = loc - scale / safe_shape
        return jnp.where(shape < 0, bounded, jnp.inf)

    def lower_bound(self) -> jnp.ndarray:
        """
        Compute the lower bound of the support.

        Returns:
            Lower bound: μ - σ/ξ for ξ > 0, -∞ otherwise
        """
        loc = jnp.asarray(self.loc)
        scale = jnp.asarray(self.scale)
        shape = jnp.asarray(self.concentration)
        safe_shape = jnp.where(shape == 0.0, 1.0, shape)
        bounded = loc - scale / safe_shape
        return jnp.where(shape > 0, bounded, -jnp.inf)

    @property
    def mean(self) -> jnp.ndarray:
        """Mean. Thin wrapper for :func:`~xtremax.primitives.gev.gev_mean`."""
        return gev_mean(self.loc, self.scale, self.concentration)

    @property
    def mode(self) -> jnp.ndarray:
        """
        Compute the mode of the distribution.

        For ξ ≠ 0:
            mode = μ + (σ/ξ) * ((1+ξ)^(-ξ) - 1)

        This interior stationary point only exists for ξ > -1. For
        ξ ≤ -1 (Weibull branch), the density increases all the way to the
        finite upper endpoint, so the mode is the upper bound itself.

        For ξ = 0:
            mode = μ

        Returns:
            Mode of the distribution
        """
        loc, scale, shape = self.loc, self.scale, self.concentration

        is_gumbel = jnp.abs(shape) < GUMBEL_THRESHOLD
        safe_shape = jnp.where(is_gumbel, 1.0, shape)
        has_interior_mode = shape > -1.0
        safe_base = jnp.where(has_interior_mode, 1.0 + shape, 1.0)

        def gumbel_mode():
            return loc

        def gevd_mode():
            interior_mode = loc + (scale / safe_shape) * (
                jnp.power(safe_base, -shape) - 1.0
            )
            return jnp.where(has_interior_mode, interior_mode, self.upper_bound())

        mode_gumbel = gumbel_mode()
        mode_gevd = gevd_mode()

        return jnp.where(is_gumbel, mode_gumbel, mode_gevd)

    @property
    def variance(self) -> jnp.ndarray:
        """
        Compute the variance of the distribution.

        The variance exists when ξ < 1/2:

        For ξ ≠ 0, ξ < 1/2:
            Var[X] = (σ²/ξ²) * (Γ(1-2ξ) - Γ²(1-ξ))

        For ξ = 0:
            Var[X] = σ² * π²/6

        Thin wrapper for :func:`~xtremax.primitives._moments.gev_variance`,
        which evaluates the Γ-difference through a cancellation-free
        reduced form — the direct subtraction above loses every significant
        digit for small ξ.

        Returns:
            Variance, or +∞ when it doesn't exist (ξ ≥ 1/2)
        """
        return gev_variance(self.scale, self.concentration)

    def covariance(self) -> jnp.ndarray:
        """Marginal variance broadcast to the batch shape.

        Structural seam for ``pipekit_cycle.ObservationNoise`` (which
        requires ``covariance()`` and ``sample(key, shape)``): together
        with the existing :meth:`sample`, this lets the distribution act
        as a non-Gaussian observation-error model in data-assimilation
        experiments — without xtremax importing pipekit (see
        ``docs/interop.md``). The observation errors are treated as
        independent per batch element, so the "covariance" is the
        marginal variance vector rather than a dense matrix; consuming
        analysis steps interpret it as a diagonal.

        Returns:
            Variance broadcast to ``batch_shape`` (``+inf`` where the
            variance does not exist, i.e. ξ >= 1/2).
        """
        return jnp.broadcast_to(jnp.asarray(self.variance), self.batch_shape)

    def kurtosis(self) -> jnp.ndarray:
        """
        Compute the excess kurtosis of the distribution.

        Excess kurtosis exists when ξ < 1/4. The formula involves fourth-order
        moments and gamma functions:

        κ = μ₄/σ⁴ - 3

        where μ₄ is the fourth central moment.

        Thin wrapper for
        :func:`~xtremax.primitives._moments.gev_excess_kurtosis`, which
        evaluates the fourth-moment gamma combination through a
        cancellation-free reduced form (it loses *two* leading orders in
        the naive one) and reduces to the Gumbel value 12/5 at ξ = 0.

        Returns:
            Excess kurtosis, or NaN when it doesn't exist (ξ ≥ 1/4)
        """
        return gev_excess_kurtosis(self.concentration)

    def skew(self) -> jnp.ndarray:
        """
        Compute the skewness of the distribution.

        Skewness exists when ξ < 1/3:

        For Gumbel (ξ = 0): skew ≈ 1.1396
        For general GEVD: involves third-order moments and gamma functions

        Thin wrapper for :func:`~xtremax.primitives._moments.gev_skewness`,
        which evaluates the third-moment gamma combination through a
        cancellation-free reduced form. The reduced ratio carries the sign
        of ξ on its own, so no explicit ``sign(ξ)`` factor is needed.

        Returns:
            Skewness, or NaN when it doesn't exist (ξ ≥ 1/3)
        """
        return gev_skewness(self.concentration)

    def entropy(self) -> jnp.ndarray:
        r"""Differential entropy of the GEV distribution (in nats).

        For any :math:`\xi` (including the Gumbel limit),

        .. math:: H = \log\sigma + 1 + \gamma(1 + \xi)

        where :math:`\gamma` is the Euler–Mascheroni constant. At
        :math:`\xi = 0` this correctly reduces to :math:`\log\sigma + 1 + \gamma`
        (the Gumbel entropy), so no separate branch is needed.
        """
        scale, shape = self.scale, self.concentration
        euler_gamma = 0.5772156649015329
        return jnp.log(scale) + 1.0 + euler_gamma * (1.0 + shape)

    def survival_function(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the survival function S(x) = 1 - F(x).

        Args:
            value: Points at which to evaluate the survival function

        Returns:
            Survival probabilities
        """
        # Delegate to the stable primitive so the class and functional APIs
        # share the smooth ξ→0 handling (no threshold to drift against).
        return gev_survival(value, self.loc, self.scale, self.concentration)

    def log_survival_function(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the log survival function log S(x) = log( 1 - F(x) ).

        Args:
            value: Points at which to evaluate the survival function

        Returns:
            Survival probabilities
        """
        # Delegate to the stable primitive (smooth ξ→0, matches the class cdf).
        return gev_log_survival(value, self.loc, self.scale, self.concentration)

    def hazard_rate(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the hazard rate h(x) = f(x) / S(x).

        The hazard rate represents the instantaneous rate of failure
        given survival up to time x.

        Args:
            value: Points at which to evaluate the hazard rate

        Returns:
            Hazard rate values
        """
        return jnp.exp(self.log_prob(value) - self.log_survival_function(value))

    def cumulative_hazard_rate(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the cumulative hazard rate Λ(x) = -log(S(x)).

        This represents the accumulated hazard up to time x.

        Args:
            value: Points at which to evaluate the cumulative hazard rate

        Returns:
            Cumulative hazard rate values
        """
        return -self.log_survival_function(value)

    def return_level(self, return_period: float | jnp.ndarray) -> jnp.ndarray:
        """Return level. Thin wrapper for ``gev_return_level``."""
        return gev_return_level(return_period, self.loc, self.scale, self.concentration)

    def tail_index(self) -> jnp.ndarray:
        """
        Compute the tail index α = 1/ξ for ξ > 0.

        For Fréchet distributions (ξ > 0), the tail follows a power law:
        P(X > x) ~ x^(-1/ξ) as x → ∞

        Returns:
            Tail index (1/ξ for ξ > 0, ∞ for ξ ≤ 0)
        """
        shape = jnp.asarray(self.concentration)
        # Safe divisor: with Python-float parameters the division happens
        # eagerly in both jnp.where branches, so ξ = 0 raised
        # ZeroDivisionError before the mask applied.
        safe_shape = jnp.where(shape > 0, shape, 1.0)
        return jnp.where(shape > 0, 1.0 / safe_shape, jnp.inf)

    def exceedance_probability(self, threshold: jnp.ndarray) -> jnp.ndarray:
        """
        Compute probability of exceeding a threshold: P(X > threshold).

        This is equivalent to the survival function at the threshold.

        Args:
            threshold: Threshold value

        Returns:
            Exceedance probabilities
        """
        return self.survival_function(threshold)

    def conditional_excess_mean(self, threshold: jnp.ndarray) -> jnp.ndarray:
        r"""Mean excess :math:`E[X - u \mid X > u]` for the GEV distribution.

        Computed via quantile-space quadrature:

        .. math:: E[X \mid X > u]
            = \frac{\int_{F(u)}^{1} F^{-1}(p)\, dp}{1 - F(u)}

        using a per-threshold trapezoidal rule on a grid from
        :math:`F(u)` to :math:`1 - \epsilon`. This is exact up to
        quadrature error across all three GEV branches (Gumbel,
        Fréchet, Weibull).

        The previous implementation used the GPD linear form
        :math:`(\sigma + \xi(u - \mu)) / (1 - \xi)`, which is the
        *Pickands–Balkema–de Haan POT limit* — correct only as
        :math:`u \to` upper endpoint, and notably returning a constant
        :math:`\sigma` for all :math:`u` when :math:`\xi = 0` even
        though the true Gumbel mean excess depends on :math:`u`.

        Returns NaN where the mean does not exist (:math:`\xi \ge 1`).
        """
        threshold_arr = jnp.asarray(threshold)
        shape = self.concentration

        # Log-space tail-probability quadrature:
        #   ∫_{F(u)}^1 F⁻¹(p) dp = ∫_0^{S(u)} F⁻¹(1-y) dy
        #                        = ∫_{-∞}^{log S(u)} F⁻¹(1-eᵛ) · eᵛ dv
        # with v = log y. We truncate the lower end at log(1e-12),
        # which is negligible for any ξ < 1. This replaces the earlier
        # linear-p grid that had two problems: (a) the normaliser used
        # the truncated mass `(1-ε) - F(u)` instead of `S(u)`, and (b)
        # the fixed linear grid systematically underresolved heavy
        # tails where most of the conditional mass lives near p = 1.
        n_grid = 1024
        p0 = self.cdf(threshold_arr)
        # Joint batch of threshold and distribution parameters — the grid
        # axis is appended after this, so every per-batch quantity must be
        # broadcast to it before gaining the trailing axis.
        batch = jnp.shape(p0)
        s_u = 1.0 - p0
        # Floor S(u) so log is finite; the NaN mask below drops results
        # whose threshold is effectively at or beyond the upper support.
        # Also cap just below 1.0 so the integration grid stays strictly
        # inside the open interval (0, 1): ``p_grid`` would otherwise
        # reach 0 when ``S(u) = 1`` (e.g. u far below the location),
        # and ``icdf(0)`` diverges to -∞ for unbounded-below branches.
        s_u_safe = jnp.clip(s_u, 1e-12, 1.0 - 1e-6)
        log_s_u = jnp.log(s_u_safe)
        # Lower endpoint of the log-y integration range: step ~20 e-folds
        # below log_s_u so the grid always captures the bulk of the
        # tail-integrand mass AND stays strictly ascending (``v_grid``
        # dx > 0). Without this, a fixed floor like log(1e-6) would run
        # backward when S(u) < 1e-6 and trapezoid would sign-flip the
        # result. The mass below the truncation point — which behaves
        # like y^{1-ξ} and reached ~14% of the total at ξ = 0.9 — is
        # added back analytically below rather than by widening the
        # span: a 1/(1-ξ)-scaled span would push ``exp(v_grid)`` under
        # the float32 exp underflow (~e⁻¹⁰³) and produce 0·inf = NaN in
        # the integrand. The ``-jnp.log1p(-y_grid)`` computation below
        # is accurate down to float32 subnormals.
        span = jnp.asarray(20.0, dtype=log_s_u.dtype)
        log_y_min = log_s_u - span

        unit = jnp.linspace(0.0, 1.0, n_grid)
        log_s_u_exp = jnp.expand_dims(log_s_u, axis=-1)
        log_y_min_exp = jnp.expand_dims(jnp.broadcast_to(log_y_min, batch), axis=-1)
        # v grid ascending from log_y_min up to log S(u). If log_y_min
        # >= log_s_u (untrustworthy regime; mask below returns NaN),
        # the grid is degenerate but numerically bounded.
        v_grid = log_y_min_exp * (1.0 - unit) + log_s_u_exp * unit
        y_grid = jnp.exp(v_grid)
        # Compute ``x = F⁻¹(1 - y)`` directly from ``y`` using
        # ``-log1p(-y)`` instead of going through ``self.icdf(1 - y)``.
        # For ``y`` near ``eps(float32)`` the round-tripped ``p = 1 - y``
        # in float32 only retains ~1 ulp of precision, and the
        # downstream ``log(p)`` in ``gev_icdf`` loses that — icdf comes
        # out finite but wildly wrong and the quadrature sign-flips.
        # ``log1p(-y)`` is designed for exactly this small-y regime.
        # Parameters are broadcast to the joint batch and given the same
        # trailing grid axis as ``neg_log_q`` — evaluating batched
        # (..., n_grid) grids against unexpanded (batch,) parameters
        # raised a broadcasting error.
        neg_log_q = -jnp.log1p(-y_grid)
        is_gumbel = jnp.abs(shape) < GUMBEL_THRESHOLD
        xi = jnp.where(is_gumbel, 1.0, shape)
        loc_e = jnp.expand_dims(jnp.broadcast_to(self.loc, batch), axis=-1)
        scale_e = jnp.expand_dims(jnp.broadcast_to(self.scale, batch), axis=-1)
        xi_e = jnp.expand_dims(jnp.broadcast_to(xi, batch), axis=-1)
        gum_e = jnp.expand_dims(jnp.broadcast_to(is_gumbel, batch), axis=-1)
        gumbel_x = loc_e - scale_e * jnp.log(neg_log_q)
        gev_x = loc_e + (scale_e / xi_e) * (jnp.power(neg_log_q, -xi_e) - 1.0)
        x_grid = jnp.where(gum_e, gumbel_x, gev_x)

        integrand = x_grid * y_grid  # F⁻¹(1-y) · y, multiplied by dv=dy/y
        numerator = jnp.trapezoid(integrand, x=v_grid, axis=-1)

        # Closed-form remainder for the truncated (0, y_c] tail with
        # y_c = exp(log_y_min) = S(u)·e⁻²⁰. There -log(1-y) ≈ y to ~y_c
        # relative accuracy, so
        #   ξ ≠ 0: x(y) ≈ (μ - σ/ξ) + (σ/ξ)·y^{-ξ}
        #          → ∫₀^{y_c} x dy = (μ - σ/ξ)·y_c + (σ/ξ)·y_c^{1-ξ}/(1-ξ)
        #   ξ = 0: x(y) ≈ μ - σ·log y
        #          → ∫₀^{y_c} x dy = y_c·(μ - σ·(log y_c - 1))
        # For light tails the remainder is negligible; at ξ = 0.9 it is
        # ~14% of the numerator — dropping it was the truncation bias.
        loc_b = jnp.broadcast_to(self.loc, batch)
        scale_b = jnp.broadcast_to(self.scale, batch)
        xi_b = jnp.broadcast_to(xi, batch)
        gum_b = jnp.broadcast_to(is_gumbel, batch)
        shape_b = jnp.broadcast_to(shape, batch)
        y_c = jnp.exp(log_y_min)
        # Sanitize the discarded branch: 1-ξ ≤ 0 (ξ ≥ 1, masked to NaN
        # below) and the Gumbel branch would otherwise divide by zero.
        one_minus_xi = jnp.where(gum_b | (shape_b >= 1.0), 1.0, 1.0 - xi_b)
        gev_rem = (loc_b - scale_b / xi_b) * y_c + (scale_b / xi_b) * jnp.power(
            y_c, one_minus_xi
        ) / one_minus_xi
        gumbel_rem = y_c * (loc_b - scale_b * (log_y_min - 1.0))
        numerator = numerator + jnp.where(gum_b, gumbel_rem, gev_rem)

        mean_conditional = numerator / s_u_safe
        mean_excess = mean_conditional - threshold_arr

        mean_exists = shape < 1.0
        valid = mean_exists & (s_u > 1e-12)
        return jnp.where(valid, mean_excess, jnp.nan)

    def reliability_function(self, time: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the reliability function R(t) = P(X > t).

        This is equivalent to the survival function but emphasizes
        the reliability engineering interpretation for Weibull Type III.

        Args:
            time: Time points for reliability evaluation

        Returns:
            Reliability values (probability of surviving past time t)
        """
        return self.survival_function(time)

    def mean_residual_life(self, time: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the mean residual life function: E[X - t | X > t].

        For reliability applications, this represents the expected
        additional lifetime given survival to time t.

        Args:
            time: Current time/age

        Returns:
            Mean residual life values
        """
        return self.conditional_excess_mean(time)

    def percentile_residual_life(
        self, time: jnp.ndarray, percentile: float
    ) -> jnp.ndarray:
        """
        Compute percentile residual life: (Q(p|X>t) - t).

        This gives the additional time until the p-th percentile failure,
        conditional on survival to time t.

        Args:
            time: Current time/age
            percentile: Percentile level (0 < percentile < 1)

        Returns:
            Percentile residual life values
        """
        # Conditional CDF: F(x | X > t) = p ⟺ F(x) = 1 - (1 - p) S(t).
        # (Previously used `1 - p S(t)`, which mapped p=0 to F^{-1}(1),
        # i.e. the upper endpoint, instead of returning zero residual.)
        survival_prob = self.survival_function(time)
        total_prob = 1.0 - (1.0 - percentile) * survival_prob
        return self.icdf(total_prob) - time

    def expand(self, batch_shape: tuple[int, ...]) -> dist.Distribution:
        """Expand to ``batch_shape`` by reconstructing via ``__init__``.

        Going through the constructor (rather than an ``ExpandedDistribution``
        wrapper) keeps every EVT method available on the returned instance.
        """
        batch_shape = tuple(batch_shape)
        if batch_shape == self.batch_shape:
            return self
        return type(self)(
            loc=jnp.broadcast_to(self.loc, batch_shape),
            scale=jnp.broadcast_to(self.scale, batch_shape),
            concentration=jnp.broadcast_to(self.concentration, batch_shape),
            validate_args=self._validate_args,
        )


# Convenient alias for backward compatibility and shorter imports
GEVD = GeneralizedExtremeValueDistribution
