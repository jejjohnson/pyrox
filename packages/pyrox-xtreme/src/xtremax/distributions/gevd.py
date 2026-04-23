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
from jax.scipy.special import gammaln
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes, validate_sample

from xtremax.primitives.gev import (
    gev_cdf,
    gev_icdf,
    gev_log_prob,
    gev_mean,
    gev_return_level,
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

        # Numerical threshold for Gumbel approximation
        self._gumbel_threshold = 1e-7

        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key: jnp.ndarray, sample_shape: tuple = ()) -> jnp.ndarray:
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

        Returns:
            Array of samples from the GEVD
        """
        check_prng_key(key)
        shape = sample_shape + self.batch_shape

        # JAX's Uniform(0, 1) sampler can emit exact 0 or 1 at the
        # endpoints; passing those to icdf yields -inf/+inf and poisons
        # downstream computations. Clamp away from the endpoints.
        uniform_samples = dist.Uniform(0.0, 1.0).sample(key, shape)
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
            - ξ > 0: [μ - σ/ξ, +∞)
            - ξ = 0: (-∞, +∞)
            - ξ < 0: (-∞, μ - σ/ξ]

        ``constraints.interval(lower_bound, upper_bound)`` resolves to
        the unbounded real line when both bounds are ±∞ (ξ = 0) and to
        the correct half-bounded interval otherwise — so
        ``validate_args=True`` actually rejects samples outside the
        true support instead of letting ``log_prob`` return -∞.
        """
        return constraints.interval(self.lower_bound(), self.upper_bound())

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

        is_gumbel = jnp.abs(shape) < self._gumbel_threshold
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

        Returns:
            Variance or NaN/∞ when it doesn't exist
        """
        _loc, scale, shape = self.loc, self.scale, self.concentration

        is_gumbel = jnp.abs(shape) < self._gumbel_threshold
        var_exists = shape < 0.5

        def gumbel_variance():
            return (scale**2) * (jnp.pi**2) / 6.0

        def gevd_variance():
            gamma1 = jnp.exp(gammaln(1.0 - 2.0 * shape))
            gamma2 = jnp.exp(2.0 * gammaln(1.0 - shape))
            return (scale**2 / shape**2) * (gamma1 - gamma2)

        var_gumbel = gumbel_variance()
        var_gevd = gevd_variance()

        result = jnp.where(is_gumbel, var_gumbel, var_gevd)
        return jnp.where(var_exists, result, jnp.inf)

    def kurtosis(self) -> jnp.ndarray:
        """
        Compute the excess kurtosis of the distribution.

        Excess kurtosis exists when ξ < 1/4. The formula involves fourth-order
        moments and gamma functions:

        κ = μ₄/σ⁴ - 3

        where μ₄ is the fourth central moment.

        Returns:
            Excess kurtosis or NaN/∞ when it doesn't exist
        """
        shape = self.concentration

        is_gumbel = jnp.abs(shape) < self._gumbel_threshold
        kurt_exists = shape < 0.25

        def gumbel_kurtosis():
            # For Gumbel: excess kurtosis = 12/5
            return 12.0 / 5.0

        def gevd_kurtosis():
            # Complex formula involving gamma functions
            g1 = jnp.exp(gammaln(1.0 - shape))
            g2 = jnp.exp(gammaln(1.0 - 2.0 * shape))
            g3 = jnp.exp(gammaln(1.0 - 3.0 * shape))
            g4 = jnp.exp(gammaln(1.0 - 4.0 * shape))

            # Central moments
            mu2 = g2 - g1**2
            mu4 = g4 - 4.0 * g1 * g3 + 6.0 * g1**2 * g2 - 3.0 * g1**4

            return (mu4 / mu2**2) - 3.0

        kurt_gumbel = gumbel_kurtosis()
        kurt_gevd = gevd_kurtosis()

        result = jnp.where(is_gumbel, kurt_gumbel, kurt_gevd)
        return jnp.where(kurt_exists, result, jnp.inf)

    def skew(self) -> jnp.ndarray:
        """
        Compute the skewness of the distribution.

        Skewness exists when ξ < 1/3:

        For Gumbel (ξ = 0): skew ≈ 1.1396
        For general GEVD: involves third-order moments and gamma functions

        Returns:
            Skewness or NaN/∞ when it doesn't exist
        """
        shape = self.concentration

        is_gumbel = jnp.abs(shape) < self._gumbel_threshold
        skew_exists = shape < 1.0 / 3.0

        def gumbel_skewness():
            # Analytical value for Gumbel distribution
            return 1.1395470994046486

        def gevd_skewness():
            g1 = jnp.exp(gammaln(1.0 - shape))
            g2 = jnp.exp(gammaln(1.0 - 2.0 * shape))
            g3 = jnp.exp(gammaln(1.0 - 3.0 * shape))

            mu2 = g2 - g1**2
            mu3 = g3 - 3.0 * g1 * g2 + 2.0 * g1**3

            return mu3 / jnp.power(mu2, 1.5)

        skew_gumbel = gumbel_skewness()
        skew_gevd = gevd_skewness()

        result = jnp.where(is_gumbel, skew_gumbel, skew_gevd)
        return jnp.where(skew_exists, result, jnp.inf)

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
        shape = jnp.asarray(self.concentration)
        is_gumbel = jnp.abs(shape) < self._gumbel_threshold
        xi = jnp.where(is_gumbel, 1.0, shape)
        z = (value - self.loc) / self.scale

        gumbel = -jnp.expm1(-jnp.exp(-z))

        t = 1.0 + xi * z
        valid = t > 0.0
        t_safe = jnp.where(valid, t, 1.0)
        log_cdf = -jnp.power(t_safe, -1.0 / xi)
        gev_inside = -jnp.expm1(log_cdf)
        boundary = jnp.where(shape > 0, 1.0, 0.0)
        gevd = jnp.where(valid, gev_inside, boundary)

        return jnp.where(is_gumbel, gumbel, gevd)

    def log_survival_function(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the log survival function log S(x) = log( 1 - F(x) ).

        Args:
            value: Points at which to evaluate the survival function

        Returns:
            Survival probabilities
        """
        shape = jnp.asarray(self.concentration)
        is_gumbel = jnp.abs(shape) < self._gumbel_threshold
        xi = jnp.where(is_gumbel, 1.0, shape)
        z = (value - self.loc) / self.scale

        gumbel = jnp.log(-jnp.expm1(-jnp.exp(-z)))

        t = 1.0 + xi * z
        valid = t > 0.0
        t_safe = jnp.where(valid, t, 1.0)
        log_cdf = -jnp.power(t_safe, -1.0 / xi)
        gev_inside = jnp.log(-jnp.expm1(log_cdf))
        boundary = jnp.where(shape > 0, 0.0, -jnp.inf)
        gevd = jnp.where(valid, gev_inside, boundary)

        return jnp.where(is_gumbel, gumbel, gevd)

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
        return jnp.where(self.concentration > 0, 1.0 / self.concentration, jnp.inf)

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
        s_u = 1.0 - p0
        # Floor S(u) so log is finite; the NaN mask below drops results
        # whose threshold is effectively at or beyond the upper support.
        # Also cap just below 1.0 so the integration grid stays strictly
        # inside the open interval (0, 1): ``p_grid`` would otherwise
        # reach 0 when ``S(u) = 1`` (e.g. u far below the location),
        # and ``icdf(0)`` diverges to -∞ for unbounded-below branches.
        s_u_safe = jnp.clip(s_u, 1e-12, 1.0 - 1e-6)
        log_s_u = jnp.log(s_u_safe)
        # Lower endpoint of the log-y integration range: step ~20
        # e-folds below log_s_u so the grid always captures the bulk
        # of the tail-integrand mass for ξ < 1 AND stays strictly
        # ascending (``v_grid`` dx > 0). Without this, a fixed floor
        # like log(1e-6) would run backward when S(u) < 1e-6 and
        # trapezoid would sign-flip the result. The downstream
        # ``-jnp.log1p(-y_grid)`` computation is accurate down to
        # float32 subnormals, so we can safely push y well below
        # eps(float32) without the precision blow-up that a raw
        # ``log(1 - y)`` route would hit.
        span = jnp.asarray(20.0, dtype=log_s_u.dtype)
        log_y_min = log_s_u - span

        unit = jnp.linspace(0.0, 1.0, n_grid)
        log_s_u_exp = jnp.expand_dims(log_s_u, axis=-1)
        log_y_min_exp = jnp.expand_dims(log_y_min, axis=-1)
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
        neg_log_q = -jnp.log1p(-y_grid)
        is_gumbel = jnp.abs(shape) < self._gumbel_threshold
        xi = jnp.where(is_gumbel, 1.0, shape)
        gumbel_x = self.loc - self.scale * jnp.log(neg_log_q)
        gev_x = self.loc + (self.scale / xi) * (jnp.power(neg_log_q, -xi) - 1.0)
        x_grid = jnp.where(is_gumbel, gumbel_x, gev_x)

        integrand = x_grid * y_grid  # F⁻¹(1-y) · y, multiplied by dv=dy/y
        numerator = jnp.trapezoid(integrand, x=v_grid, axis=-1)
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

        Going through the constructor repopulates ``_gumbel_threshold`` on
        the returned instance so downstream methods that dispatch on it
        (log_prob/cdf/icdf/mean/entropy) keep working after expansion.
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
