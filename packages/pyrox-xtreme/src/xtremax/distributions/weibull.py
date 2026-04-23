"""
Weibull Type III Generalized Extreme Value Distribution for NumPyro

This module provides a robust implementation of the GEVD Type III (reversed Weibull)
with extensive statistical methods and proper NumPyro integration for bounded upper
tail extreme value modeling.
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
from xtremax.primitives.weibull import (
    weibull_cdf,
    weibull_icdf,
    weibull_log_prob,
    weibull_mean,
    weibull_return_level,
)


class WeibullType3GEVD(dist.Distribution):
    """
    Weibull Type III Generalized Extreme Value Distribution for NumPyro.

    The Weibull Type III (reversed Weibull) is the GEVD with shape parameter ξ < 0,
    characterized by a bounded upper tail. This distribution is fundamental in extreme
    value theory for modeling bounded maxima and reliability applications with upper
    failure limits.

    **Key Characteristics:**
    - Bounded upper tail: x ≤ μ - σ/ξ (since ξ < 0)
    - Light tail behavior with exponential decay
    - Often used for modeling material strength limits, maximum system capacity
    - Opposite tail behavior to Fréchet (Type II)

    **Probability Density Function:**

    f(x) = (1/σ) * t(x)^(-(1/ξ + 1)) * exp(-t(x)^(-1/ξ))
    where t(x) = 1 + ξ(x - μ)/σ and ξ < 0

    **Cumulative Distribution Function:**

    F(x) = exp(-t(x)^(-1/ξ)) for t(x) > 0
    F(x) = 1 for t(x) ≤ 0 (above upper bound)
    where t(x) = 1 + ξ(x - μ)/σ

    **Support:**

    x ∈ (-∞, μ - σ/ξ] where μ - σ/ξ is the upper bound

    **Parameters:**

    - loc (μ): Location parameter ∈ ℝ
    - scale (σ): Scale parameter > 0
    - shape (ξ): Shape parameter < 0 (enforced for Type III)

    **Statistical Properties:**

    - Mean exists for ξ > -1: E[X] = μ + (σ/ξ)(Γ(1-ξ) - 1)
    - Variance exists for ξ > -1/2: Var[X] = (σ²/ξ²)(Γ(1-2ξ) - Γ²(1-ξ))
    - Upper bounded support makes it suitable for reliability analysis

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> # Create Weibull Type III GEVD
        >>> weibull_gevd = WeibullType3GEVD(loc=10.0, scale=2.0, shape=-0.3)
        >>>
        >>> # Key properties
        >>> print(f"Upper bound: {weibull_gevd.upper_bound()}")
        >>> print(f"Mean: {weibull_gevd.mean}")
        >>> print(f"Mode: {weibull_gevd.mode}")
        >>>
        >>> # Sample and evaluate
        >>> key = jax.random.PRNGKey(42)
        >>> samples = weibull_gevd.sample(key, sample_shape=(1000,))
        >>> log_probs = weibull_gevd.log_prob(samples)
    """

    # NumPyro distribution interface requirements
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "concentration": constraints.less_than(0.0),  # Enforce ξ < 0 for Type III
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
        Initialize the Weibull Type III GEVD.

        Args:
            loc: Location parameter μ (real number)
            scale: Scale parameter σ (positive real number)
            concentration: Shape parameter ξ (negative real number, ξ < 0)
            shape: Deprecated backward-compatible alias for ``concentration``.
                Stored as ``self.concentration``; the NumPyro-inherited
                ``Distribution.shape()`` method is kept callable.
            validate_args: Whether to validate input arguments

        Raises:
            ValueError: If ``validate_args=True`` and ``concentration >= 0``
                (enforced by the ``arg_constraints`` entry). The
                previous Python branch ``if jnp.any(shape >= 0)`` was
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
            concentration = -0.1

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
        Generate samples from the Weibull Type III GEVD via inverse transform sampling.

        The sampling uses the quantile function:
        Q(p) = μ + (σ/ξ) * ((-ln(p))^(-ξ) - 1)

        Since ξ < 0, this becomes:
        Q(p) = μ + (σ/|ξ|) * (1 - (-ln(p))^(|ξ|))

        Args:
            key: JAX random key for sampling
            sample_shape: Shape of samples to generate

        Returns:
            Array of samples from the Weibull Type III GEVD, all bounded above
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
        """Log PDF. Thin wrapper for ``weibull_log_prob``."""
        return weibull_log_prob(value, self.loc, self.scale, self.concentration)

    def cdf(self, value: jnp.ndarray) -> jnp.ndarray:
        """CDF. Thin wrapper for :func:`~xtremax.primitives.weibull.weibull_cdf`."""
        return weibull_cdf(value, self.loc, self.scale, self.concentration)

    def icdf(self, q: jnp.ndarray) -> jnp.ndarray:
        """Quantile function. Thin wrapper for ``weibull_icdf``."""
        return weibull_icdf(q, self.loc, self.scale, self.concentration)

    @property
    def support(self) -> constraints.Constraint:
        """
        Return the support constraint for Weibull Type III GEVD.

        Returns:
            Constraint representing x ≤ μ - σ/ξ (upper bounded)
        """
        upper_bound = self.upper_bound()
        return constraints.interval(self.lower_bound(), upper_bound)

    def upper_bound(self) -> jnp.ndarray:
        """
        Compute the upper bound of the support.

        For Weibull Type III (ξ < 0): upper bound = μ - σ/ξ

        Returns:
            Upper bound of the distribution support
        """
        # Since ξ < 0, -σ/ξ = σ/|ξ| > 0
        return self.loc - self.scale / self.concentration

    def lower_bound(self) -> jnp.ndarray:
        """
        Compute the lower bound of the support.

        For Weibull Type III: lower bound = -∞

        Returns:
            Lower bound (-∞ for Type III)
        """
        return jnp.full_like(self.loc, -jnp.inf)

    @property
    def mean(self) -> jnp.ndarray:
        """Mean. Thin wrapper for :func:`~xtremax.primitives.weibull.weibull_mean`."""
        return weibull_mean(self.loc, self.scale, self.concentration)

    @property
    def mode(self) -> jnp.ndarray:
        """
        Compute the mode of the Weibull Type III GEVD.

        The mode is:
        mode = μ + (σ/ξ) * ((1+ξ)^(-ξ) - 1)

        This interior stationary point only exists for -1 < ξ < 0. For
        ξ ≤ -1, the density increases all the way to the finite upper
        endpoint, so the mode is the upper bound itself.

        Returns:
            Mode of the distribution
        """
        loc, scale, shape = self.loc, self.scale, self.concentration
        has_interior_mode = shape > -1.0
        safe_base = jnp.where(has_interior_mode, 1.0 + shape, 1.0)

        interior_mode = loc + (scale / shape) * (jnp.power(safe_base, -shape) - 1.0)
        return jnp.where(has_interior_mode, interior_mode, self.upper_bound())

    @property
    def variance(self) -> jnp.ndarray:
        """
        Compute the variance of the Weibull Type III GEVD.

        Var[X] = (σ²/ξ²) * (Γ(1-2ξ) - Γ²(1-ξ))

        For Weibull Type III, ξ < 0 means the distribution is bounded
        above, so *all* moments exist for every valid ξ in the Type III
        range (no guard is needed). The earlier guard ``ξ > -1/2`` was
        mistakenly inherited from the Fréchet moment-existence condition
        reflected to negative ξ; it silently returned NaN for valid
        shapes like ξ = -1.

        Returns:
            Variance (always finite on the Type III domain).
        """
        _loc, scale, shape = self.loc, self.scale, self.concentration

        # Γ(1 - 2ξ) and Γ(1 - ξ) are finite for every ξ < 0 (argument > 1).
        gamma1 = jnp.exp(gammaln(1.0 - 2.0 * shape))  # Γ(1-2ξ)
        gamma2 = jnp.exp(2.0 * gammaln(1.0 - shape))  # Γ²(1-ξ)
        return (scale**2 / shape**2) * (gamma1 - gamma2)

    def kurtosis(self) -> jnp.ndarray:
        """
        Compute the excess kurtosis of the Weibull Type III GEVD.

        κ = μ₄/σ⁴ - 3

        As with variance, every moment exists for every ξ < 0 (bounded
        support), so no ``ξ > -1/4`` guard is applied.

        Returns:
            Excess kurtosis (always finite on the Type III domain).
        """
        shape = self.concentration

        g1 = jnp.exp(gammaln(1.0 - shape))  # Γ(1-ξ)
        g2 = jnp.exp(gammaln(1.0 - 2.0 * shape))  # Γ(1-2ξ)
        g3 = jnp.exp(gammaln(1.0 - 3.0 * shape))  # Γ(1-3ξ)
        g4 = jnp.exp(gammaln(1.0 - 4.0 * shape))  # Γ(1-4ξ)

        mu2 = g2 - g1**2
        mu4 = g4 - 4.0 * g1 * g3 + 6.0 * g1**2 * g2 - 3.0 * g1**4
        return (mu4 / mu2**2) - 3.0

    def skew(self) -> jnp.ndarray:
        """
        Compute the skewness of the Weibull Type III GEVD.

        For Weibull Type III, skewness is typically negative due to the
        upper bound creating left-skewed distributions. All moments
        exist across the full valid ξ < 0 range.

        Returns:
            Skewness (always finite on the Type III domain).
        """
        shape = self.concentration

        g1 = jnp.exp(gammaln(1.0 - shape))  # Γ(1-ξ)
        g2 = jnp.exp(gammaln(1.0 - 2.0 * shape))  # Γ(1-2ξ)
        g3 = jnp.exp(gammaln(1.0 - 3.0 * shape))  # Γ(1-3ξ)

        mu2 = g2 - g1**2
        mu3 = g3 - 3.0 * g1 * g2 + 2.0 * g1**3

        return mu3 / jnp.power(mu2, 1.5)

    def entropy(self) -> jnp.ndarray:
        """
        Compute the differential entropy of the Weibull Type III GEVD.

        The entropy is:
        H = log(σ) + 1 + γ * (1 + ξ)

        where γ is the Euler-Mascheroni constant ≈ 0.5772. This is the
        standard GEV entropy formula evaluated at the Weibull branch
        (ξ < 0); at ξ = 0 it reduces to the Gumbel entropy
        ``log σ + 1 + γ``.

        Returns:
            Differential entropy in nats
        """
        scale, shape = self.scale, self.concentration
        euler_gamma = 0.5772156649015329

        return jnp.log(scale) + 1.0 + euler_gamma * (1.0 + shape)

    def survival_function(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the survival function S(x) = 1 - F(x).

        For Weibull Type III, this represents the probability of not
        exceeding the value x, which is relevant for reliability analysis.

        Args:
            value: Points at which to evaluate the survival function

        Returns:
            Survival probabilities
        """
        # Computed as ``-expm1(log F)`` rather than ``1 - F`` so the
        # near-upper-endpoint regime stays resolvable: once
        # ``F ≈ 1 - 1e-8``, float32 rounds ``1 - F`` to zero and
        # downstream ``log(S)`` in hazard_rate/cumulative_hazard_rate
        # returns ``-inf``/``nan`` even for valid in-support inputs.
        loc, scale, shape = self.loc, self.scale, self.concentration
        z = (value - loc) / scale
        t = 1.0 + shape * z
        valid = t > 0.0
        log_cdf = -jnp.power(jnp.where(valid, t, 1.0), -1.0 / shape)
        surv_inside = -jnp.expm1(log_cdf)
        # Above the upper support bound (t ≤ 0 with ξ < 0), F(x) = 1 so
        # S(x) = 0.
        return jnp.where(valid, surv_inside, 0.0)

    def log_survival_function(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the log survival function log S(x) = log(1 - F(x)).

        Directly evaluates ``log(-expm1(log F))`` to keep the near-upper-
        endpoint regime numerically stable for hazard/cumulative-hazard
        consumers.

        Args:
            value: Points at which to evaluate log survival function

        Returns:
            Log survival probabilities
        """
        loc, scale, shape = self.loc, self.scale, self.concentration
        z = (value - loc) / scale
        t = 1.0 + shape * z
        valid = t > 0.0
        log_cdf = -jnp.power(jnp.where(valid, t, 1.0), -1.0 / shape)
        log_surv_inside = jnp.log(-jnp.expm1(log_cdf))
        # Above the upper support bound, S(x) = 0 ⇒ log S = -∞.
        return jnp.where(valid, log_surv_inside, -jnp.inf)

    def hazard_rate(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the hazard rate h(x) = f(x) / S(x).

        The hazard rate represents the instantaneous failure rate
        given survival up to time x. For Weibull Type III, the
        hazard rate approaches infinity as x approaches the upper bound.

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
        For Weibull Type III, this is finite at the upper bound.

        Args:
            value: Points at which to evaluate the cumulative hazard rate

        Returns:
            Cumulative hazard rate values
        """
        return -self.log_survival_function(value)

    def return_level(self, return_period: float | jnp.ndarray) -> jnp.ndarray:
        """Return level. Thin wrapper for ``weibull_return_level``."""
        return weibull_return_level(
            return_period, self.loc, self.scale, self.concentration
        )

    def tail_index(self) -> jnp.ndarray:
        """
        Compute the tail index for Weibull Type III GEVD.

        For Weibull Type III (ξ < 0), there is no power law tail behavior.
        Instead, the tail is exponentially bounded. The concept of tail
        index (1/ξ) is not meaningful for bounded distributions.

        Returns:
            NaN (tail index not defined for bounded distributions)
        """
        return jnp.full_like(self.concentration, jnp.nan)

    def exceedance_probability(self, threshold: jnp.ndarray) -> jnp.ndarray:
        """
        Compute probability of exceeding a threshold: P(X > threshold).

        For Weibull Type III, this is 0 when threshold ≥ upper bound.

        Args:
            threshold: Threshold value

        Returns:
            Exceedance probabilities (0 above upper bound)
        """
        return self.survival_function(threshold)

    def conditional_excess_mean(self, threshold: jnp.ndarray) -> jnp.ndarray:
        r"""Mean excess :math:`E[X - u \mid X > u]` for Weibull Type III GEVD.

        Computed via quantile-space quadrature, same as the general GEV
        class. The previous GPD linear approximation
        :math:`(\sigma + \xi(u - \mu)) / (1 - \xi)` is the POT asymptote
        and is wrong for finite thresholds. As :math:`u` approaches the
        upper endpoint :math:`\mu - \sigma/\xi`, the quadrature
        correctly decays to zero.
        """
        threshold_arr = jnp.asarray(threshold)

        # Log-space tail-probability quadrature (see GEVD implementation
        # for the derivation). Correctly normalises by S(u) rather than
        # the truncated mass `(1-ε) - F(u)`.
        n_grid = 1024
        p0 = self.cdf(threshold_arr)
        s_u = 1.0 - p0
        # Cap away from 1.0 so p_grid stays strictly in (0, 1). See
        # GEVD.conditional_excess_mean.
        s_u_safe = jnp.clip(s_u, 1e-12, 1.0 - 1e-6)
        log_s_u = jnp.log(s_u_safe)
        # Lower log-y endpoint: step ~20 e-folds below log_s_u so the
        # grid always captures the bulk of the tail-integrand mass
        # AND stays strictly ascending. A fixed floor like log(1e-6)
        # would run backward when S(u) < 1e-6 (near the upper
        # endpoint) and trapezoid would sign-flip. The
        # ``-jnp.log1p(-y)`` step below is accurate to subnormals in
        # float32. See GEVD.conditional_excess_mean for details.
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
        # ``log(p)`` inside ``weibull_icdf`` loses that entirely. See
        # GEVD.conditional_excess_mean for details. ξ < 0 is enforced
        # by ``arg_constraints``, so the Gumbel branch is unreachable.
        shape = self.concentration
        neg_log_q = -jnp.log1p(-y_grid)
        x_grid = self.loc + (self.scale / shape) * (jnp.power(neg_log_q, -shape) - 1.0)

        integrand = x_grid * y_grid
        numerator = jnp.trapezoid(integrand, x=v_grid, axis=-1)
        mean_conditional = numerator / s_u_safe
        mean_excess = mean_conditional - threshold_arr

        return jnp.where(s_u > 1e-12, mean_excess, jnp.nan)

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
WeibullGEVD = WeibullType3GEVD
BoundedGEVD = WeibullType3GEVD
