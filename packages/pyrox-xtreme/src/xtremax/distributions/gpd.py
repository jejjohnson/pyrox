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
from xtremax.distributions._support import concrete_sign
from xtremax.primitives.gpd import (
    gpd_cdf,
    gpd_icdf,
    gpd_log_prob,
    gpd_log_survival,
    gpd_mean,
    gpd_return_level,
    gpd_survival,
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

        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(
        self,
        key: jnp.ndarray,
        sample_shape: tuple = (),
        *,
        shape: tuple | None = None,
    ) -> jnp.ndarray:
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
            shape: Keyword-only alias for ``sample_shape`` matching the
                ``pipekit_cycle.ObservationNoise`` protocol's parameter
                name (``sample(key, shape)``), so callers typed against
                that protocol can pass it by keyword. Passing both a
                non-empty ``sample_shape`` and ``shape`` raises
                ``ValueError``.

        Returns:
            Array of samples from the GPD (all within support)
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
            - ξ ≥ 0: [0, +∞) → ``nonnegative``
            - ξ < 0: [0, -σ/ξ] → ``interval(0, upper_bound)``

        Only the ξ < 0 case uses ``interval`` — its bounds are finite, so
        ``biject_to`` maps it to a well-behaved ``Sigmoid ∘ Affine``. For
        ξ ≥ 0 the old ``interval(0, inf)`` construction made ``biject_to``
        return inf, breaking NUTS/SVI models with a GPD latent site;
        ``constraints.nonnegative`` has a registered exp-bijector and
        accepts the closed endpoint x = 0.

        When the concentration is traced (inside ``jit``/inference) or a
        mixed-sign batch, the support falls back to ``nonnegative`` — the
        lower bound is parameter-independent, and points above a finite
        ξ < 0 upper endpoint are handled by ``log_prob`` returning -inf.
        """
        if concrete_sign(self.concentration) == -1:
            return constraints.interval(jnp.zeros_like(self.scale), self.upper_bound())
        return constraints.nonnegative

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

        For ξ > -1 the density is monotone decreasing from x = 0, so the
        mode is 0 (the threshold). For ξ < -1 the density *increases*
        toward the finite upper endpoint -σ/ξ, so the mode is the upper
        endpoint itself (at ξ = -1, the uniform case, every point is
        modal and 0 is returned by convention).

        Returns:
            Mode: 0 for ξ ≥ -1, the upper endpoint -σ/ξ for ξ < -1
        """
        return jnp.where(
            self.concentration < -1.0, self.upper_bound(), jnp.zeros_like(self.scale)
        )

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
        Compute the excess kurtosis of the GPD.

        Excess kurtosis exists when ξ < 1/4:

        κ = 3(1-2ξ)(2ξ²+ξ+3) / ((1-3ξ)(1-4ξ)) - 3

        Returns:
            Excess kurtosis or +∞ when it doesn't exist
        """
        shape = self.concentration

        # Kurtosis exists for ξ < 1/4
        kurt_exists = shape < 0.25

        # The rational formula is exact and smooth at ξ = 0 (evaluating to
        # the exponential value 6), so no separate branch or threshold is
        # needed — the old class-local 1e-8 threshold is gone.
        numerator = 3.0 * (1.0 - 2.0 * shape) * (2.0 * shape**2 + shape + 3.0)
        denominator = (1.0 - 3.0 * shape) * (1.0 - 4.0 * shape)
        kurtosis_val = numerator / denominator - 3.0

        # NaN, not inf: beyond ξ = 1/4 the standardized fourth moment is
        # undefined, unlike the variance which genuinely diverges to +inf.
        return jnp.where(kurt_exists, kurtosis_val, jnp.nan)

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

        # The closed-form expression is exact and smooth at ξ = 0
        # (evaluating to the exponential value 2), so no separate branch or
        # threshold is needed — the old class-local 1e-8 threshold is gone.
        numerator = 2.0 * (1.0 + shape) * jnp.sqrt(1.0 - 2.0 * shape)
        denominator = 1.0 - 3.0 * shape
        skewness_val = numerator / denominator

        # NaN, not inf: beyond ξ = 1/3 the standardized third moment is
        # undefined, unlike the variance which genuinely diverges to +inf.
        return jnp.where(skew_exists, skewness_val, jnp.nan)

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
        # Delegate to the stable primitive so the class and functional APIs
        # share the same smooth ξ→0 handling (no threshold to drift against).
        return gpd_survival(value, self.scale, self.concentration)

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
        # Λ(x) = -log S(x); delegate to the stable log-survival primitive.
        return -gpd_log_survival(value, self.scale, self.concentration)

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

        Vectorized over ``k_values`` via a cumulative sum of log order
        statistics, so it is safe under ``jax.jit`` (the previous Python
        loop branched on traced values and raised a ``TracerBoolConversion``
        error). Invalid ``k`` (≤ 0 or ≥ n) yield NaN.

        Args:
            order_statistics: Sorted sample in descending order
            k_values: Numbers of upper order statistics to use

        Returns:
            Dictionary with Hill plot data
        """
        order_statistics = jnp.asarray(order_statistics)
        k_arr = jnp.asarray(k_values)
        n = order_statistics.shape[0]

        # (1/k)·Σ_{i<k} log X_i − log X_k, for every k at once.
        logs = jnp.log(order_statistics)
        cumsum = jnp.cumsum(logs)
        valid = (k_arr > 0) & (k_arr < n)
        k_safe = jnp.clip(k_arr, 1, max(n - 1, 1))
        mean_log_top = cumsum[k_safe - 1] / k_safe
        hill_est = mean_log_top - logs[k_safe]
        estimates = jnp.where(hill_est > 0, 1.0 / hill_est, jnp.inf)
        estimates = jnp.where(valid, estimates, jnp.nan)

        return {
            "k_values": k_values,
            "hill_estimates": estimates,
            "theoretical_tail_index": self.tail_index(),
        }

    def expand(self, batch_shape: tuple[int, ...]) -> dist.Distribution:
        """Expand to ``batch_shape`` by reconstructing via ``__init__``.

        We deliberately go through the constructor (rather than an
        ``ExpandedDistribution`` wrapper) so every EVT method stays
        available on the returned distribution.
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
