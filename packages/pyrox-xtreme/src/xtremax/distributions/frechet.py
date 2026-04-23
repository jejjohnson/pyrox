"""
Fréchet Type II Generalized Extreme Value Distribution for NumPyro

This module provides a robust implementation of the GEVD Type II (Fréchet)
with extensive statistical methods and proper NumPyro integration for heavy-tailed
extreme value modeling with power-law tail behavior.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from jax.scipy.special import gammaln
from jax.typing import ArrayLike
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes, validate_sample

from xtremax._rng import check_prng_key
from xtremax.primitives.frechet import (
    frechet_cdf,
    frechet_icdf,
    frechet_log_prob,
    frechet_mean,
    frechet_return_level,
)


class FrechetType2GEVD(dist.Distribution):
    """
    Fréchet Type II Generalized Extreme Value Distribution for NumPyro.

    The Fréchet Type II is the GEVD with shape parameter ξ > 0, characterized by
    heavy polynomial tails and lower-bounded support. This distribution is
    fundamental in extreme value theory for modeling phenomena with power-law
    tail behavior, such as financial extreme losses, natural disaster magnitudes,
    and network traffic bursts.

    **Key Characteristics:**
    - Lower bounded support: x ≥ μ - σ/ξ (since ξ > 0)
    - Heavy polynomial tails: P(X > x) ~ x^(-1/ξ) as x → ∞
    - Power-law tail index α = 1/ξ
    - Infinite moments when ξ ≥ 1/k for k-th moment
    - Models phenomena with "fat tails" and extreme outliers

    **Probability Density Function:**

    f(x) = (1/σ) * t(x)^(-(1/ξ + 1)) * exp(-t(x)^(-1/ξ))
    where t(x) = 1 + ξ(x - μ)/σ and ξ > 0

    **Cumulative Distribution Function:**

    F(x) = exp(-t(x)^(-1/ξ)) for t(x) > 0
    F(x) = 0 for t(x) ≤ 0 (below lower bound)
    where t(x) = 1 + ξ(x - μ)/σ

    **Support:**

    x ∈ [μ - σ/ξ, +∞) where μ - σ/ξ is the lower bound

    **Tail Behavior:**

    P(X > x) ~ C * x^(-1/ξ) as x → ∞, where C is a constant
    This gives tail index α = 1/ξ, with smaller ξ meaning heavier tails

    **Parameters:**

    - loc (μ): Location parameter ∈ ℝ
    - scale (σ): Scale parameter > 0
    - shape (ξ): Shape parameter > 0 (enforced for Type II)

    **Statistical Properties:**

    - Mean exists for ξ < 1: E[X] = μ + (σ/ξ)(Γ(1-ξ) - 1)
    - Variance exists for ξ < 1/2: Var[X] = (σ²/ξ²)(Γ(1-2ξ) - Γ²(1-ξ))
    - k-th moment exists for ξ < 1/k
    - Heavy right tail makes it suitable for extreme event modeling

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> # Create Fréchet Type II GEVD with different tail heaviness
        >>> light_tail = FrechetType2GEVD(loc=0.0, scale=1.0, shape=0.1)  # α = 10
        >>> heavy_tail = FrechetType2GEVD(loc=0.0, scale=1.0, shape=0.8)  # α = 1.25
        >>>
        >>> # Key properties
        >>> print(f"Lower bound: {heavy_tail.lower_bound()}")
        >>> print(f"Tail index: {heavy_tail.tail_index()}")
        >>> print(f"Mean exists: {heavy_tail.concentration < 1}")
        >>>
        >>> # Sample and evaluate
        >>> key = jax.random.PRNGKey(42)
        >>> samples = heavy_tail.sample(key, sample_shape=(1000,))
        >>> log_probs = heavy_tail.log_prob(samples)
    """

    # NumPyro distribution interface requirements
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "concentration": constraints.positive,  # Enforce ξ > 0 for Type II
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
        Initialize the Fréchet Type II GEVD.

        Args:
            loc: Location parameter μ (real number)
            scale: Scale parameter σ (positive real number)
            concentration: Shape parameter ξ (positive real number, ξ > 0)
            shape: Deprecated backward-compatible alias for ``concentration``.
                Stored as ``self.concentration``; the NumPyro-inherited
                ``Distribution.shape()`` method is kept callable.
            validate_args: Whether to validate input arguments

        Raises:
            ValueError: If ``validate_args=True`` and ``concentration <= 0``
                (enforced by the ``arg_constraints`` entry). The
                previous Python branch ``if jnp.any(shape <= 0)`` was
                tracer-unsafe: building the distribution inside a
                ``jit``-compiled NumPyro model would raise a tracer
                concretization error before sampling/log-prob could
                run. ``arg_constraints`` already expresses the domain.
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
            concentration = 0.1

        self.loc, self.scale, self.concentration = promote_shapes(
            loc, scale, concentration
        )

        # Determine batch shape from broadcasted parameters
        batch_shape = lax.broadcast_shapes(
            jnp.shape(self.loc), jnp.shape(self.scale), jnp.shape(self.concentration)
        )

        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key: jnp.ndarray, sample_shape: tuple = ()) -> jnp.ndarray:
        """
        Generate samples from the Fréchet Type II GEVD using inverse transform sampling.

        The sampling uses the quantile function:
        Q(p) = μ + (σ/ξ) * ((-ln(p))^(-ξ) - 1)

        Since ξ > 0, this generates samples with heavy right tails.

        Args:
            key: JAX random key for sampling
            sample_shape: Shape of samples to generate

        Returns:
            Array of samples from the Fréchet Type II GEVD, all above lower bound
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
        """Log PDF. Thin wrapper for ``frechet_log_prob``."""
        return frechet_log_prob(value, self.loc, self.scale, self.concentration)

    def cdf(self, value: jnp.ndarray) -> jnp.ndarray:
        """CDF. Thin wrapper for :func:`~xtremax.primitives.frechet.frechet_cdf`."""
        return frechet_cdf(value, self.loc, self.scale, self.concentration)

    def icdf(self, q: jnp.ndarray) -> jnp.ndarray:
        """Quantile function. Thin wrapper for ``frechet_icdf``."""
        return frechet_icdf(q, self.loc, self.scale, self.concentration)

    @property
    def support(self) -> constraints.Constraint:
        """
        Return the support constraint for Fréchet Type II GEVD.

        Returns:
            Constraint representing x ≥ μ - σ/ξ (lower bounded)
        """
        lower_bound = self.lower_bound()
        return constraints.interval(lower_bound, self.upper_bound())

    def upper_bound(self) -> jnp.ndarray:
        """
        Compute the upper bound of the support.

        For Fréchet Type II: upper bound = +∞

        Returns:
            Upper bound (+∞ for Type II)
        """
        return jnp.full_like(self.loc, jnp.inf)

    def lower_bound(self) -> jnp.ndarray:
        """
        Compute the lower bound of the support.

        For Fréchet Type II (ξ > 0): lower bound = μ - σ/ξ

        Returns:
            Lower bound of the distribution support
        """
        return self.loc - self.scale / self.concentration

    @property
    def mean(self) -> jnp.ndarray:
        """Mean. Thin wrapper for :func:`~xtremax.primitives.frechet.frechet_mean`."""
        return frechet_mean(self.loc, self.scale, self.concentration)

    @property
    def mode(self) -> jnp.ndarray:
        """
        Compute the mode of the Fréchet Type II GEVD.

        The mode is:
        mode = μ + (σ/ξ) * ((1+ξ)^(-ξ) - 1)

        This represents the most likely value in the heavy-tailed distribution.

        Returns:
            Mode of the distribution
        """
        loc, scale, shape = self.loc, self.scale, self.concentration

        return loc + (scale / shape) * (jnp.power(1.0 + shape, -shape) - 1.0)

    @property
    def variance(self) -> jnp.ndarray:
        """
        Compute the variance of the Fréchet Type II GEVD.

        The variance exists when ξ < 1/2:
        Var[X] = (σ²/ξ²) * (Γ(1-2ξ) - Γ²(1-ξ))

        For very heavy tails (ξ ≥ 1/2), the variance is infinite.

        Returns:
            Variance or +∞ when it doesn't exist (ξ ≥ 1/2)
        """
        _loc, scale, shape = self.loc, self.scale, self.concentration

        # Variance exists for ξ < 1/2
        var_exists = shape < 0.5

        # Compute variance using gamma functions
        gamma1 = jnp.exp(gammaln(1.0 - 2.0 * shape))  # Γ(1-2ξ)
        gamma2 = jnp.exp(2.0 * gammaln(1.0 - shape))  # Γ²(1-ξ)

        var_val = (scale**2 / shape**2) * (gamma1 - gamma2)

        return jnp.where(var_exists, var_val, jnp.inf)

    def kurtosis(self) -> jnp.ndarray:
        """
        Compute the excess kurtosis of the Fréchet Type II GEVD.

        Excess kurtosis exists when ξ < 1/4 and involves fourth-order moments.
        For heavy-tailed distributions, kurtosis captures the tail heaviness.

        Returns:
            Excess kurtosis or +∞ when it doesn't exist (ξ ≥ 1/4)
        """
        shape = self.concentration

        # Kurtosis exists for ξ < 1/4
        kurt_exists = shape < 0.25

        # Complex formula involving gamma functions for fourth moment
        g1 = jnp.exp(gammaln(1.0 - shape))  # Γ(1-ξ)
        g2 = jnp.exp(gammaln(1.0 - 2.0 * shape))  # Γ(1-2ξ)
        g3 = jnp.exp(gammaln(1.0 - 3.0 * shape))  # Γ(1-3ξ)
        g4 = jnp.exp(gammaln(1.0 - 4.0 * shape))  # Γ(1-4ξ)

        # Central moments
        mu2 = g2 - g1**2
        mu4 = g4 - 4.0 * g1 * g3 + 6.0 * g1**2 * g2 - 3.0 * g1**4

        excess_kurt = (mu4 / mu2**2) - 3.0

        return jnp.where(kurt_exists, excess_kurt, jnp.inf)

    def skew(self) -> jnp.ndarray:
        """
        Compute the skewness of the Fréchet Type II GEVD.

        Skewness exists when ξ < 1/3 and involves third-order moments.

        For Fréchet Type II, skewness is typically positive due to the
        heavy right tail creating right-skewed distributions.

        Returns:
            Skewness or +∞ when it doesn't exist (ξ ≥ 1/3)
        """
        shape = self.concentration

        # Skewness exists for ξ < 1/3
        skew_exists = shape < 1.0 / 3.0

        # Compute using gamma functions
        g1 = jnp.exp(gammaln(1.0 - shape))  # Γ(1-ξ)
        g2 = jnp.exp(gammaln(1.0 - 2.0 * shape))  # Γ(1-2ξ)
        g3 = jnp.exp(gammaln(1.0 - 3.0 * shape))  # Γ(1-3ξ)

        # Central moments
        mu2 = g2 - g1**2
        mu3 = g3 - 3.0 * g1 * g2 + 2.0 * g1**3

        skewness = mu3 / jnp.power(mu2, 1.5)

        return jnp.where(skew_exists, skewness, jnp.inf)

    def entropy(self) -> jnp.ndarray:
        """
        Compute the differential entropy of the Fréchet Type II GEVD.

        The entropy is:
        H = log(σ) + 1 + γ * (1 + ξ)

        where γ is the Euler-Mascheroni constant ≈ 0.5772. This is the
        standard GEV entropy formula evaluated at the Fréchet branch
        (ξ > 0); at ξ = 0 it reduces to the Gumbel entropy `log σ + 1 + γ`.

        Returns:
            Differential entropy in nats
        """
        scale, shape = self.scale, self.concentration
        euler_gamma = 0.5772156649015329

        return jnp.log(scale) + 1.0 + euler_gamma * (1.0 + shape)

    def survival_function(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the survival function S(x) = 1 - F(x).

        For Fréchet Type II, this exhibits power-law decay:
        S(x) ~ C * x^(-1/ξ) for large x, where C is a constant.

        Args:
            value: Points at which to evaluate the survival function

        Returns:
            Survival probabilities (power-law tail behavior)
        """
        # Computed as ``-expm1(log F)`` rather than ``1 - F`` so the
        # far right tail stays resolvable: once ``F ≈ 1 - 1e-8``, float32
        # rounds ``1 - F`` to zero while ``-expm1(-t^(-1/ξ))`` preserves
        # the tail mass down to subnormals, keeping
        # ``exceedance_probability`` / ``hazard_rate`` finite.
        loc, scale, shape = self.loc, self.scale, self.concentration
        z = (value - loc) / scale
        t = 1.0 + shape * z
        valid = t > 0.0
        log_cdf = -jnp.power(jnp.where(valid, t, 1.0), -1.0 / shape)
        surv_inside = -jnp.expm1(log_cdf)
        # Below the lower support bound (t ≤ 0 with ξ > 0), F(x) = 0 so
        # S(x) = 1.
        return jnp.where(valid, surv_inside, 1.0)

    def log_survival_function(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the log survival function log S(x) = log(1 - F(x)).

        For heavy-tailed distributions, this provides better numerical
        stability for extreme quantiles.

        Args:
            value: Points at which to evaluate log survival function

        Returns:
            Log survival probabilities
        """
        loc, scale, shape = self.loc, self.scale, self.concentration
        z = (value - loc) / scale
        t = 1.0 + shape * z

        valid = t > 0.0
        # With F(x) = exp(-t^(-1/ξ)), log F(x) = -t^(-1/ξ). The survival
        # is 1 - F(x) = -expm1(-t^(-1/ξ)), and log S = log(-expm1(...)).
        log_f = -jnp.power(jnp.where(valid, t, 1.0), -1.0 / shape)
        log_surv_inside = jnp.log(-jnp.expm1(log_f))
        # Below the lower bound (t <= 0), F(x) = 0 so log S = 0.
        return jnp.where(valid, log_surv_inside, 0.0)

    def hazard_rate(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the hazard rate h(x) = f(x) / S(x).

        For Fréchet Type II, the hazard rate approaches zero as x → ∞,
        indicating decreasing failure rates for extreme values.

        Args:
            value: Points at which to evaluate the hazard rate

        Returns:
            Hazard rate values
        """
        log_hazard = self.log_prob(value) - self.log_survival_function(value)
        return jnp.exp(log_hazard)

    def cumulative_hazard_rate(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the cumulative hazard rate Λ(x) = -log(S(x)).

        This represents the accumulated hazard up to time x.
        For Fréchet Type II, this grows slower than exponentially.

        Args:
            value: Points at which to evaluate the cumulative hazard rate

        Returns:
            Cumulative hazard rate values
        """
        return -self.log_survival_function(value)

    def return_level(self, return_period: float | jnp.ndarray) -> jnp.ndarray:
        """Return level. Thin wrapper for ``frechet_return_level``."""
        return frechet_return_level(
            return_period, self.loc, self.scale, self.concentration
        )

    def tail_index(self) -> jnp.ndarray:
        """
        Compute the tail index α = 1/ξ for Fréchet Type II GEVD.

        The tail index characterizes the power-law tail behavior:
        P(X > x) ~ x^(-α) as x → ∞

        Lower values of α (higher ξ) indicate heavier tails.

        Returns:
            Tail index α = 1/ξ
        """
        return 1.0 / self.concentration

    def exceedance_probability(self, threshold: jnp.ndarray) -> jnp.ndarray:
        """
        Compute probability of exceeding a threshold: P(X > threshold).

        For large thresholds, this exhibits power-law decay:
        P(X > x) ~ C * x^(-1/ξ)

        Args:
            threshold: Threshold value

        Returns:
            Exceedance probabilities (power-law tail behavior)
        """
        return self.survival_function(threshold)

    def conditional_excess_mean(self, threshold: jnp.ndarray) -> jnp.ndarray:
        r"""Mean excess :math:`E[X - u \mid X > u]` for Fréchet Type II GEVD.

        Computed via quantile-space quadrature, same machinery as the
        general GEV class. The previous GPD linear approximation
        :math:`(\sigma + \xi(u - \mu)) / (1 - \xi)` is only the POT
        asymptote, not the finite-threshold mean excess.

        Returns NaN for :math:`\xi \ge 1` (heavy tails where the mean
        does not exist).
        """
        threshold_arr = jnp.asarray(threshold)
        shape = self.concentration

        # Log-space tail-probability quadrature (see GEVD implementation
        # for the derivation). This replaces the earlier linear-p grid
        # whose (a) normaliser used truncated mass instead of S(u), and
        # (b) linear spacing underresolved heavy Fréchet tails.
        n_grid = 1024
        p0 = self.cdf(threshold_arr)
        s_u = 1.0 - p0
        # Cap away from 1.0 so p_grid stays strictly in (0, 1). See
        # GEVD.conditional_excess_mean.
        s_u_safe = jnp.clip(s_u, 1e-12, 1.0 - 1e-6)
        log_s_u = jnp.log(s_u_safe)
        # Lower log-y endpoint: step ~20 e-folds below log_s_u so the
        # grid always captures the bulk of the tail-integrand mass
        # (ξ < 1) AND stays strictly ascending. A fixed floor like
        # log(1e-6) would run backward when S(u) < 1e-6 and trapezoid
        # would sign-flip. The ``-jnp.log1p(-y)`` step below is
        # accurate to subnormals in float32, so y can safely go below
        # eps(float32). See GEVD.conditional_excess_mean for details.
        span = jnp.asarray(20.0, dtype=log_s_u.dtype)
        log_y_min = log_s_u - span

        unit = jnp.linspace(0.0, 1.0, n_grid)
        log_s_u_exp = jnp.expand_dims(log_s_u, axis=-1)
        log_y_min_exp = jnp.expand_dims(log_y_min, axis=-1)
        v_grid = log_y_min_exp * (1.0 - unit) + log_s_u_exp * unit
        y_grid = jnp.exp(v_grid)
        # Compute ``x = F⁻¹(1 - y)`` directly from ``y`` using
        # ``-log1p(-y)``. For ``y`` near ``eps(float32)`` the
        # round-tripped ``p = 1 - y`` keeps only ~1 ulp of precision and
        # ``log(p)`` inside ``frechet_icdf`` loses that entirely, so the
        # quadrature sign-flips at very high thresholds. See
        # GEVD.conditional_excess_mean for details. ξ > 0 is enforced
        # by ``arg_constraints``, so the Gumbel branch is unreachable.
        neg_log_q = -jnp.log1p(-y_grid)
        x_grid = self.loc + (self.scale / shape) * (jnp.power(neg_log_q, -shape) - 1.0)

        integrand = x_grid * y_grid
        numerator = jnp.trapezoid(integrand, x=v_grid, axis=-1)
        mean_conditional = numerator / s_u_safe
        mean_excess = mean_conditional - threshold_arr

        mean_exists = shape < 1.0
        valid = mean_exists & (s_u > 1e-12)
        return jnp.where(valid, mean_excess, jnp.nan)

    def hill_estimator_threshold(self, quantile: float = 0.95) -> jnp.ndarray:
        """
        Suggest threshold for Hill tail index estimation.

        For Fréchet distributions, the threshold should be high enough
        to be in the tail region but low enough to have sufficient data.

        Args:
            quantile: Quantile level for threshold selection

        Returns:
            Suggested threshold for tail analysis
        """
        return self.icdf(quantile)

    def pareto_tail_approx(
        self, threshold: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generalized Pareto approximation for excesses over threshold.

        For Fréchet GEVD, excesses over high thresholds approximately
        follow GPD(σ*, ξ) where:
        σ* = σ + ξ(u - μ) (modified scale)
        ξ remains the same (shape parameter)

        Args:
            threshold: Threshold value for POT modeling

        Returns:
            Tuple of (GPD scale parameter, GPD shape parameter)
        """
        loc, scale, shape = self.loc, self.scale, self.concentration

        # GPD parameters for excesses over threshold
        gpd_scale = scale + shape * (threshold - loc)
        gpd_shape = shape

        return gpd_scale, gpd_shape

    def value_at_risk(self, alpha: float) -> jnp.ndarray:
        """
        Compute Value-at-Risk (VaR) at confidence level α.

        VaR_α = Q(α) where Q is the quantile function.
        For heavy-tailed distributions, VaR can be very large.

        Args:
            alpha: Confidence level (e.g., 0.95, 0.99, 0.999)

        Returns:
            Value-at-Risk at confidence level α
        """
        return self.icdf(alpha)

    def expected_shortfall(self, alpha: float) -> jnp.ndarray:
        """
        Compute Expected Shortfall (Conditional VaR) at confidence level α.

        ES_α = E[X | X > VaR_α] for Fréchet distributions.
        This captures the expected loss beyond the VaR threshold.

        Args:
            alpha: Confidence level (e.g., 0.95, 0.99, 0.999)

        Returns:
            Expected Shortfall at confidence level α
        """
        var_alpha = self.value_at_risk(alpha)
        excess_mean = self.conditional_excess_mean(var_alpha)

        return var_alpha + excess_mean

    def power_law_scaling(self, scale_factor: float) -> jnp.ndarray:
        """
        Compute scaling relationship for power-law tails.

        For Fréchet distributions: if X ~ Fréchet(μ,σ,ξ),
        then aX has tail behavior ~ (ax)^(-1/ξ) = a^(-1/ξ) * x^(-1/ξ)

        Args:
            scale_factor: Scaling factor a > 0

        Returns:
            Relative change in tail probabilities under scaling
        """
        return jnp.power(scale_factor, -1.0 / self.concentration)

    def expand(self, batch_shape: tuple[int, ...]) -> dist.Distribution:
        """Expand to ``batch_shape`` by reconstructing via ``__init__``."""
        batch_shape = tuple(batch_shape)
        if batch_shape == self.batch_shape:
            return self
        return type(self)(
            loc=jnp.broadcast_to(self.loc, batch_shape),
            scale=jnp.broadcast_to(self.scale, batch_shape),
            concentration=jnp.broadcast_to(self.concentration, batch_shape),
            validate_args=self._validate_args,
        )


# Convenient aliases
FrechetGEVD = FrechetType2GEVD
HeavyTailGEVD = FrechetType2GEVD
PowerLawGEVD = FrechetType2GEVD
