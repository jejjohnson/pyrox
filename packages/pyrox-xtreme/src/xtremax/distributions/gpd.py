"""
Generalized Pareto Distribution (GPD) for NumPyro

This module provides a robust implementation of the GPD with extensive statistical
methods and proper NumPyro integration for threshold exceedance modeling in extreme
value theory and peaks-over-threshold (POT) analysis.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from jax.typing import ArrayLike
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes, validate_sample

from xtremax._rng import check_prng_key
from xtremax.primitives.gpd import (
    gpd_cdf,
    gpd_icdf,
    gpd_log_prob,
    gpd_mean,
    gpd_return_level,
)


class GeneralizedParetoDistribution(dist.Distribution):
    """
    Generalized Pareto Distribution (GPD) for NumPyro.

    The Generalized Pareto Distribution is fundamental in extreme value theory
    for modeling threshold exceedances in the Peaks-Over-Threshold (POT) framework.
    It emerges as the limiting distribution of scaled excesses over high thresholds
    for a wide class of underlying distributions.

    **Key Characteristics:**
    - Models excesses above a threshold: Y = X - u | X > u
    - Three families based on shape parameter ξ:
      * ξ > 0: Pareto-type (heavy tails, power-law decay)
      * ξ = 0: Exponential-type (exponential tails)
      * ξ < 0: Beta-type (bounded support, light tails)
    - Direct connection to GEVD via POT theory
    - Foundation for threshold-based extreme value modeling

    **Probability Density Function:**

    For ξ ≠ 0:
        f(x) = (1/σ) * (1 + ξx/σ)^(-(1/ξ + 1))

    For ξ = 0 (exponential limit):
        f(x) = (1/σ) * exp(-x/σ)

    **Cumulative Distribution Function:**

    For ξ ≠ 0:
        F(x) = 1 - (1 + ξx/σ)^(-1/ξ)

    For ξ = 0:
        F(x) = 1 - exp(-x/σ)

    **Support:**

    - ξ ≥ 0: x ≥ 0 (non-negative)
    - ξ < 0: 0 ≤ x ≤ -σ/ξ (bounded above)

    **Parameters:**

    - scale (σ): Scale parameter > 0
    - shape (ξ): Shape parameter ∈ ℝ

    **Connection to GEVD:**

    If block maxima follow GEVD(μ, σ*, ξ), then threshold excesses
    follow GPD(σ + ξ(u - μ), ξ) where u is the threshold.

    **Applications:**

    - Financial risk: Value-at-Risk, Expected Shortfall modeling
    - Hydrology: Flood frequency analysis above design levels
    - Insurance: Large claim modeling, catastrophe reinsurance
    - Engineering: Structural reliability, extreme load analysis
    - Environmental: Pollution exceedances, extreme weather events
    - Telecommunications: Network traffic bursts, service failures

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> # Heavy-tailed excesses (financial losses)
        >>> heavy_tail_gpd = GeneralizedParetoDistribution(scale=1000, shape=0.25)
        >>>
        >>> # Exponential excesses (reliability applications)
        >>> exponential_gpd = GeneralizedParetoDistribution(scale=500, shape=0.0)
        >>>
        >>> # Bounded excesses (physical constraints)
        >>> bounded_gpd = GeneralizedParetoDistribution(scale=100, shape=-0.2)
        >>>
        >>> # Key properties
        >>> print(f"Mean (heavy tail): {heavy_tail_gpd.mean}")
        >>> print(f"Upper bound (bounded): {bounded_gpd.upper_bound()}")
        >>> print(f"Tail index (heavy): {heavy_tail_gpd.tail_index()}")
        >>>
        >>> # Sample and analyze
        >>> key = jax.random.PRNGKey(42)
        >>> samples = heavy_tail_gpd.sample(key, sample_shape=(1000,))
        >>> log_probs = heavy_tail_gpd.log_prob(samples)
    """

    # NumPyro distribution interface requirements
    arg_constraints = {"scale": constraints.positive, "concentration": constraints.real}
    reparametrized_params = ["scale", "concentration"]

    def __init__(
        self,
        scale: ArrayLike = 1.0,
        concentration: ArrayLike | None = None,
        shape: ArrayLike | None = None,
        validate_args: bool | None = None,
    ):
        """
        Initialize the Generalized Pareto Distribution.

        Args:
            scale: Scale parameter σ > 0
            concentration: Shape parameter ξ ∈ ℝ
                  * ξ > 0: Heavy tails (Pareto-type)
                  * ξ = 0: Exponential tails
                  * ξ < 0: Light tails, bounded support (Beta-type)
            shape: Deprecated backward-compatible alias for ``concentration``.
                Stored as ``self.concentration``; the NumPyro-inherited
                ``Distribution.shape()`` method is kept callable.
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

        self.scale, self.concentration = promote_shapes(scale, concentration)

        # Determine batch shape from broadcasted parameters
        batch_shape = lax.broadcast_shapes(
            jnp.shape(self.scale), jnp.shape(self.concentration)
        )

        # Numerical threshold for exponential approximation (ξ ≈ 0)
        self._exponential_threshold = 1e-8

        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key: jnp.ndarray, sample_shape: tuple = ()) -> jnp.ndarray:
        """
        Generate samples from the GPD using inverse transform sampling.

        The sampling uses the quantile function:

        For ξ ≠ 0:
            Q(p) = (σ/ξ) * ((1-p)^(-ξ) - 1)

        For ξ = 0:
            Q(p) = -σ * ln(1-p)

        Args:
            key: JAX random key for sampling
            sample_shape: Shape of samples to generate

        Returns:
            Array of samples from the GPD (all within support)
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
        """Log PDF. Thin wrapper for :func:`~xtremax.primitives.gpd.gpd_log_prob`."""
        return gpd_log_prob(value, self.scale, self.concentration)

    def cdf(self, value: jnp.ndarray) -> jnp.ndarray:
        """CDF. Thin wrapper for :func:`~xtremax.primitives.gpd.gpd_cdf`."""
        return gpd_cdf(value, self.scale, self.concentration)

    def icdf(self, q: jnp.ndarray) -> jnp.ndarray:
        """Quantile function. Thin wrapper for ``gpd_icdf``."""
        return gpd_icdf(q, self.scale, self.concentration)

    @property
    def support(self) -> constraints.Constraint:
        """
        Return the support constraint for GPD.

        Returns:
            Constraint object reflecting the shape-dependent support:
            - ξ ≥ 0: [0, +∞)
            - ξ < 0: [0, -σ/ξ]

        ``constraints.interval(0, upper_bound)`` resolves to nonnegative
        semantics when the upper bound is +∞ (ξ ≥ 0) and to a proper
        bounded interval when ξ < 0 — so ``validate_args=True`` actually
        rejects samples above the finite upper endpoint instead of
        silently deferring to ``log_prob`` returning -∞.
        """
        return constraints.interval(jnp.zeros_like(self.scale), self.upper_bound())

    def upper_bound(self) -> jnp.ndarray:
        """
        Compute the upper bound of the support.

        Returns:
            Upper bound:
            - ξ ≥ 0: +∞
            - ξ < 0: -σ/ξ
        """
        scale = jnp.asarray(self.scale)
        shape = jnp.asarray(self.concentration)
        # Replace ξ=0 with a safe placeholder inside the division so Python
        # scalars don't raise ZeroDivisionError before jnp.where selects
        # the +∞ branch.
        safe_shape = jnp.where(shape == 0.0, 1.0, shape)
        return jnp.where(shape < 0, -scale / safe_shape, jnp.inf)

    def lower_bound(self) -> jnp.ndarray:
        """
        Compute the lower bound of the support.

        Returns:
            Lower bound: 0 (always non-negative for GPD)
        """
        return jnp.zeros_like(self.scale)

    @property
    def mean(self) -> jnp.ndarray:
        """Mean. Thin wrapper for :func:`~xtremax.primitives.gpd.gpd_mean`."""
        return gpd_mean(self.scale, self.concentration)

    @property
    def mode(self) -> jnp.ndarray:
        """
        Compute the mode of the GPD.

        The mode is always 0 for the GPD, representing the threshold
        (most likely exceedance is just above the threshold).

        Returns:
            Mode (always 0)
        """
        return jnp.zeros_like(self.scale)

    @property
    def variance(self) -> jnp.ndarray:
        """
        Compute the variance of the GPD.

        Variance exists when ξ < 1/2:

        For ξ < 1/2:
            Var[X] = σ² / ((1-ξ)²(1-2ξ))

        For ξ ≥ 1/2:
            Var[X] = +∞ (infinite variance)

        Returns:
            Variance or +∞ when it doesn't exist
        """
        scale, shape = self.scale, self.concentration

        # Variance exists for ξ < 1/2
        var_exists = shape < 0.5

        # Variance formula: σ² / ((1-ξ)²(1-2ξ))
        denominator = (1.0 - shape) ** 2 * (1.0 - 2.0 * shape)
        var_val = (scale**2) / denominator

        return jnp.where(var_exists, var_val, jnp.inf)

    def kurtosis(self) -> jnp.ndarray:
        """
        Compute the excess kurtosis of the GPD.

        Excess kurtosis exists when ξ < 1/4:

        κ = 3(1-2ξ)(2ξ²+ξ+3) / ((1-3ξ)(1-4ξ)) - 3

        Returns:
            Excess kurtosis or +∞ when it doesn't exist
        """
        shape = self.concentration

        # Kurtosis exists for ξ < 1/4
        kurt_exists = shape < 0.25

        # Handle exponential case (ξ = 0) separately
        is_exponential = jnp.abs(shape) < self._exponential_threshold
        exponential_kurtosis = 6.0  # Known value for exponential distribution

        # General formula for ξ ≠ 0
        numerator = 3.0 * (1.0 - 2.0 * shape) * (2.0 * shape**2 + shape + 3.0)
        denominator = (1.0 - 3.0 * shape) * (1.0 - 4.0 * shape)
        general_kurtosis = numerator / denominator - 3.0

        kurtosis_val = jnp.where(is_exponential, exponential_kurtosis, general_kurtosis)

        return jnp.where(kurt_exists, kurtosis_val, jnp.inf)

    def skew(self) -> jnp.ndarray:
        """
        Compute the skewness of the GPD.

        Skewness exists when ξ < 1/3:

        γ₃ = 2(1+ξ)√(1-2ξ) / (1-3ξ)

        Returns:
            Skewness or +∞ when it doesn't exist
        """
        shape = self.concentration

        # Skewness exists for ξ < 1/3
        skew_exists = shape < 1.0 / 3.0

        # Handle exponential case
        is_exponential = jnp.abs(shape) < self._exponential_threshold
        exponential_skewness = 2.0  # Known value for exponential distribution

        # General formula
        numerator = 2.0 * (1.0 + shape) * jnp.sqrt(1.0 - 2.0 * shape)
        denominator = 1.0 - 3.0 * shape
        general_skewness = numerator / denominator

        skewness_val = jnp.where(is_exponential, exponential_skewness, general_skewness)

        return jnp.where(skew_exists, skewness_val, jnp.inf)

    def entropy(self) -> jnp.ndarray:
        """
        Compute the differential entropy of the GPD.

        For the GPD:
        H = log(σ) + ξ + 1

        Returns:
            Differential entropy in nats
        """
        return jnp.log(self.scale) + self.concentration + 1.0

    def survival_function(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the survival function S(x) = 1 - F(x).

        For ξ ≠ 0:
            S(x) = (1 + ξx/σ)^(-1/ξ)

        For ξ = 0:
            S(x) = exp(-x/σ)

        This has direct interpretation in POT models as the probability
        of observing an exceedance larger than x.

        Args:
            value: Points at which to evaluate survival function

        Returns:
            Survival probabilities
        """
        scale = jnp.asarray(self.scale)
        shape = jnp.asarray(self.concentration)
        value = jnp.asarray(value)
        is_exponential = jnp.abs(shape) < self._exponential_threshold
        safe_shape = jnp.where(is_exponential, jnp.ones_like(shape), shape)

        # Below the lower support bound x < 0, survival is exactly 1.
        value_in_support = value >= 0.0

        exponential = jnp.exp(-jnp.maximum(value, 0.0) / scale)

        t = 1.0 + safe_shape * value / scale
        valid = t > 0.0
        t_safe = jnp.where(valid, t, 1.0)
        pareto_inside = jnp.power(t_safe, -1.0 / safe_shape)
        boundary = jnp.where(shape < 0, 0.0, 1.0)
        pareto = jnp.where(valid, pareto_inside, boundary)

        survival = jnp.where(is_exponential, exponential, pareto)
        return jnp.where(value_in_support, survival, 1.0)

    def hazard_rate(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the hazard rate h(x) = f(x) / S(x).

        For GPD:
        h(x) = 1 / (σ + ξx)

        This represents the instantaneous failure rate in reliability
        applications or the rate of threshold exceedance.

        Args:
            value: Points at which to evaluate hazard rate

        Returns:
            Hazard rate values
        """
        scale, shape = self.scale, self.concentration

        # Hazard rate: h(x) = 1 / (σ + ξx). Enforce BOTH the lower support
        # bound (GPD is defined for x ≥ 0; f(x)=0, S(x)=1 ⇒ h(x)=0 below)
        # and the upper bound in the bounded ξ < 0 case (σ + ξx > 0).
        denominator = scale + shape * value
        valid = (denominator > 0.0) & (value >= 0.0)
        hazard_val = 1.0 / jnp.where(valid, denominator, 1.0)

        return jnp.where(valid, hazard_val, 0.0)

    def cumulative_hazard_rate(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the cumulative hazard rate Λ(x) = -log(S(x)).

        For ξ ≠ 0:
            Λ(x) = (1/ξ) * log(1 + ξx/σ)

        For ξ = 0:
            Λ(x) = x/σ

        Args:
            value: Points at which to evaluate cumulative hazard rate

        Returns:
            Cumulative hazard rate values
        """
        scale = jnp.asarray(self.scale)
        shape = jnp.asarray(self.concentration)
        value = jnp.asarray(value)
        is_exponential = jnp.abs(shape) < self._exponential_threshold
        safe_shape = jnp.where(is_exponential, jnp.ones_like(shape), shape)

        exponential = jnp.maximum(value, 0.0) / scale

        t = 1.0 + safe_shape * value / scale
        valid = t > 0.0
        t_safe = jnp.where(valid, t, 1.0)
        pareto_inside = jnp.log(t_safe) / safe_shape
        boundary = jnp.where(shape < 0, jnp.inf, 0.0)
        pareto = jnp.where(valid, pareto_inside, boundary)

        cumulative_hazard = jnp.where(is_exponential, exponential, pareto)
        return jnp.where(value >= 0.0, cumulative_hazard, 0.0)

    def return_level(self, return_period: float | jnp.ndarray) -> jnp.ndarray:
        """Return level. Thin wrapper for ``gpd_return_level``."""
        return gpd_return_level(return_period, self.scale, self.concentration)

    def tail_index(self) -> jnp.ndarray:
        """
        Compute the tail index for GPD.

        For GPD with ξ > 0: tail index α = 1/ξ
        For ξ ≤ 0: tail index is not defined (no power-law tail)

        Returns:
            Tail index (1/ξ for ξ > 0, ∞ otherwise)
        """
        return jnp.where(self.concentration > 0, 1.0 / self.concentration, jnp.inf)

    def exceedance_probability(self, threshold: jnp.ndarray) -> jnp.ndarray:
        """
        Compute probability of exceeding a threshold: P(X > threshold).

        This is the survival function and fundamental for POT analysis.

        Args:
            threshold: Threshold value

        Returns:
            Exceedance probabilities
        """
        return self.survival_function(threshold)

    def conditional_excess_mean(self, threshold: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the mean excess function: E[X - u | X > u].

        For GPD, this has the simple form:
        E[X - u | X > u] = (σ + ξu) / (1 - ξ)

        This linear relationship is a key property of GPD and forms
        the basis for threshold selection in POT modeling.

        Args:
            threshold: Threshold value u

        Returns:
            Conditional excess mean values
        """
        scale, shape = self.scale, self.concentration

        # Only valid for ξ < 1 and threshold within support
        mean_exists = shape < 1.0
        within_support = (threshold >= 0.0) & (threshold < self.upper_bound())
        valid = mean_exists & within_support

        # Mean excess: (σ + ξu) / (1 - ξ)
        excess_mean = (scale + shape * threshold) / (1.0 - shape)

        return jnp.where(valid, excess_mean, jnp.inf)

    def threshold_stability_plot_data(self, thresholds: jnp.ndarray) -> dict:
        """
        Generate data for threshold stability plots in POT analysis.

        Returns modified scale parameters σ* = σ + ξ(u - u₀) for different
        thresholds, which should be approximately constant if GPD fits well.

        Args:
            thresholds: Array of threshold values

        Returns:
            Dictionary with threshold stability metrics
        """
        scale, shape = self.scale, self.concentration

        # Reference threshold (typically the lowest)
        u0 = thresholds[0]

        # Modified scale parameters for each threshold
        modified_scales = scale + shape * (thresholds - u0)

        # Shape parameters (should remain constant)
        shapes = jnp.full_like(thresholds, shape)

        return {
            "thresholds": thresholds,
            "modified_scales": modified_scales,
            "shapes": shapes,
            "reference_threshold": u0,
        }

    def hill_plot_data(
        self, order_statistics: jnp.ndarray, k_values: jnp.ndarray
    ) -> dict:
        """
        Generate data for Hill plots (tail index estimation).

        Computes Hill estimator: α̂_k = (1/k) * Σᵢ₌₁ᵏ log(X_{n-i+1,n} / X_{n-k,n})

        Args:
            order_statistics: Sorted sample in descending order
            k_values: Numbers of upper order statistics to use

        Returns:
            Dictionary with Hill plot data
        """
        n = len(order_statistics)

        hill_estimates = []
        for k in k_values:
            if k >= n or k <= 0:
                hill_estimates.append(jnp.nan)
                continue

            # Hill estimator
            log_ratios = jnp.log(order_statistics[:k] / order_statistics[k])
            hill_est = jnp.mean(log_ratios)
            hill_estimates.append(1.0 / hill_est if hill_est > 0 else jnp.inf)

        return {
            "k_values": k_values,
            "hill_estimates": jnp.array(hill_estimates),
            "theoretical_tail_index": self.tail_index(),
        }

    def expand(self, batch_shape: tuple[int, ...]) -> dist.Distribution:
        """Expand to ``batch_shape`` by reconstructing via ``__init__``.

        We deliberately go through the constructor so every cached
        attribute set by ``__init__`` (e.g. ``_exponential_threshold``) is
        present on the returned distribution. Bypassing ``__init__`` (as
        an earlier version did) broke ``cdf``/``skew``/``kurtosis`` on
        expanded instances.
        """
        batch_shape = tuple(batch_shape)
        if batch_shape == self.batch_shape:
            return self
        return type(self)(
            scale=jnp.broadcast_to(self.scale, batch_shape),
            concentration=jnp.broadcast_to(self.concentration, batch_shape),
            validate_args=self._validate_args,
        )


# Convenient aliases
GPD = GeneralizedParetoDistribution
ParetoDistribution = GeneralizedParetoDistribution
