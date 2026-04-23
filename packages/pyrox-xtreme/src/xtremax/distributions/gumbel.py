"""
Gumbel Type I Generalized Extreme Value Distribution for NumPyro

This module provides a robust implementation of the GEVD Type I (Gumbel)
with extensive statistical methods and proper NumPyro integration for exponential-tailed
extreme value modeling, particularly suited for classical meteorological applications.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import scipy.special as _sp_special
from jax import lax
from jax.scipy.special import gammaln
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes, validate_sample

from xtremax._rng import check_prng_key
from xtremax.primitives.gumbel import (
    gumbel_cdf,
    gumbel_icdf,
    gumbel_log_prob,
    gumbel_mean,
    gumbel_return_level,
)


def _host_complex_loggamma(z: np.ndarray) -> np.ndarray:
    """Complex-valued log-gamma on host via SciPy.

    ``jax.scipy.special.gammaln`` is real-only, so the characteristic
    function cannot use it for the complex argument ``1 - iσt``. We hop
    to SciPy via ``jax.pure_callback`` instead — accurate and covers
    both complex64 and complex128 inputs.
    """
    return np.asarray(_sp_special.loggamma(z), dtype=z.dtype)


class GumbelType1GEVD(dist.Distribution):
    """
    Gumbel Type I Generalized Extreme Value Distribution for NumPyro.

    The Gumbel Type I is the GEVD with shape parameter ξ = 0, characterized by
    exponential tails and unbounded support. This distribution is the limiting
    case of the GEVD and represents the "double exponential" distribution for
    extreme values. It's the most commonly used extreme value distribution in
    meteorology, hydrology, and engineering applications.

    **Key Characteristics:**
    - Unbounded support: x ∈ (-∞, +∞)
    - Exponential tails: P(X > x) ~ exp(-exp((x-μ)/σ)) for large x
    - Moderate extreme behavior (neither heavy nor light tails)
    - Asymmetric with positive skewness ≈ 1.14
    - Often emerges from block maxima of exponential-type distributions

    **Probability Density Function:**

    f(x) = (1/σ) * exp(-z - exp(-z))
    where z = (x - μ)/σ

    **Cumulative Distribution Function:**

    F(x) = exp(-exp(-z))
    where z = (x - μ)/σ

    **Quantile Function:**

    Q(p) = μ - σ * ln(-ln(p))

    **Support:**

    x ∈ (-∞, +∞) (unbounded)

    **Tail Behavior:**

    Upper tail: P(X > x) ~ exp(-exp((x-μ)/σ)) (exponential decay)
    Lower tail: P(X < x) ~ exp(exp((μ-x)/σ)) (double exponential growth)

    **Parameters:**

    - loc (μ): Location parameter ∈ ℝ (mode of the distribution)
    - scale (σ): Scale parameter > 0 (controls spread)

    **Statistical Properties:**

    - Mean: E[X] = μ + σγ (where γ ≈ 0.5772 is Euler-Mascheroni constant)
    - Mode: μ (location parameter)
    - Variance: Var[X] = σ²π²/6
    - Skewness: γ₃ ≈ 1.1396 (constant, independent of parameters)
    - Kurtosis: γ₄ = 12/5 = 2.4 (excess kurtosis, constant)

    **Applications in Geosciences:**

    - Daily maximum/minimum temperatures (temperate climates)
    - Wind speed extremes (non-hurricane conditions)
    - River discharge peaks (regulated systems)
    - Wave height extremes (moderate sea states)
    - Precipitation extremes (stable atmospheric conditions)
    - Seismic background noise levels
    - Atmospheric pressure extremes

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> # Create Gumbel distribution for temperature extremes
        >>> temp_extremes = GumbelType1GEVD(loc=35.0, scale=5.0)  # °C
        >>>
        >>> # Key properties
        >>> print(f"Mean temperature: {temp_extremes.mean:.2f}°C")
        >>> print(f"Mode (most likely): {temp_extremes.mode:.2f}°C")
        >>> print(f"Standard deviation: {jnp.sqrt(temp_extremes.variance):.2f}°C")
        >>> print(f"Skewness: {temp_extremes.skew():.3f}")
        >>>
        >>> # Sample and evaluate
        >>> key = jax.random.PRNGKey(42)
        >>> samples = temp_extremes.sample(key, sample_shape=(1000,))
        >>> log_probs = temp_extremes.log_prob(samples)
        >>>
        >>> # Return level analysis
        >>> return_50yr = temp_extremes.return_level(50)
        >>> print(f"50-year return level: {return_50yr:.2f}°C")
    """

    # NumPyro distribution interface requirements
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    reparametrized_params = ["loc", "scale"]

    def __init__(
        self, loc: float = 0.0, scale: float = 1.0, validate_args: bool | None = None
    ):
        """
        Initialize the Gumbel Type I GEVD.

        Args:
            loc: Location parameter μ (mode of distribution)
            scale: Scale parameter σ (positive real number)
            validate_args: Whether to validate input arguments

        Raises:
            ValueError: If scale <= 0
        """
        self.loc, self.scale = promote_shapes(loc, scale)

        # Determine batch shape from broadcasted parameters
        batch_shape = lax.broadcast_shapes(jnp.shape(self.loc), jnp.shape(self.scale))

        # Mathematical constants for Gumbel distribution
        self._euler_gamma = 0.5772156649015329  # Euler-Mascheroni constant
        self._pi_squared_over_six = (jnp.pi**2) / 6.0  # π²/6 for variance
        self._gumbel_skewness = 1.1395470994046486  # 12*ζ(3)/π² where ζ(3) ≈ 1.202
        self._gumbel_kurtosis = 12.0 / 5.0  # Excess kurtosis = 12/5

        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key: jnp.ndarray, sample_shape: tuple = ()) -> jnp.ndarray:
        """
        Generate samples from the Gumbel Type I GEVD using inverse transform sampling.

        The sampling uses the quantile function (inverse CDF):
        Q(p) = μ - σ * ln(-ln(p))

        This is numerically stable and provides exact samples from the distribution.

        Args:
            key: JAX random key for sampling
            sample_shape: Shape of samples to generate

        Returns:
            Array of samples from the Gumbel distribution
        """
        check_prng_key(key)
        shape = sample_shape + self.batch_shape

        # JAX's Uniform(0, 1) sampler can emit exact 0 or 1 at the
        # endpoints; passing those to icdf yields -inf/+inf and poisons
        # downstream computations. Clamp away from the endpoints.
        uniform_samples = dist.Uniform(0.0, 1.0).sample(key, shape)
        eps = jnp.finfo(uniform_samples.dtype).eps
        uniform_samples = jnp.clip(uniform_samples, eps, 1.0 - eps)

        # Apply inverse CDF transformation: Q(U) = μ - σ * ln(-ln(U))
        return self.icdf(uniform_samples)

    @validate_sample
    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        """Log PDF. Thin wrapper for ``gumbel_log_prob``."""
        return gumbel_log_prob(value, self.loc, self.scale)

    def cdf(self, value: jnp.ndarray) -> jnp.ndarray:
        """CDF. Thin wrapper for :func:`~xtremax.primitives.gumbel.gumbel_cdf`."""
        return gumbel_cdf(value, self.loc, self.scale)

    def icdf(self, q: jnp.ndarray) -> jnp.ndarray:
        """Quantile function. Thin wrapper for ``gumbel_icdf``."""
        return gumbel_icdf(q, self.loc, self.scale)

    @property
    def support(self) -> constraints.Constraint:
        """
        Return the support constraint for Gumbel Type I GEVD.

        Returns:
            Constraint representing x ∈ (-∞, +∞) (unbounded real line)
        """
        return constraints.real

    def upper_bound(self) -> jnp.ndarray:
        """
        Compute the upper bound of the support.

        For Gumbel Type I: upper bound = +∞

        Returns:
            Upper bound (+∞ for Gumbel)
        """
        return jnp.full_like(self.loc, jnp.inf)

    def lower_bound(self) -> jnp.ndarray:
        """
        Compute the lower bound of the support.

        For Gumbel Type I: lower bound = -∞

        Returns:
            Lower bound (-∞ for Gumbel)
        """
        return jnp.full_like(self.loc, -jnp.inf)

    @property
    def mean(self) -> jnp.ndarray:
        """Mean. Thin wrapper for :func:`~xtremax.primitives.gumbel.gumbel_mean`."""
        return gumbel_mean(self.loc, self.scale)

    @property
    def mode(self) -> jnp.ndarray:
        """
        Compute the mode of the Gumbel Type I GEVD.

        The mode is simply the location parameter:
        mode = μ

        This is the most likely extreme value and represents the peak
        of the probability density function.

        Returns:
            Mode of the distribution (equals location parameter)
        """
        return self.loc

    @property
    def variance(self) -> jnp.ndarray:
        """
        Compute the variance of the Gumbel Type I GEVD.

        The variance is:
        Var[X] = σ² * π²/6

        This is proportional to the square of the scale parameter,
        with the proportionality constant π²/6 ≈ 1.6449.

        Returns:
            Variance of the distribution
        """
        return (self.scale**2) * self._pi_squared_over_six

    def kurtosis(self) -> jnp.ndarray:
        """
        Compute the excess kurtosis of the Gumbel Type I GEVD.

        The excess kurtosis is constant:
        κ = 12/5 = 2.4

        This value is independent of the distribution parameters and indicates
        heavier tails than the normal distribution (which has κ = 0).

        Returns:
            Excess kurtosis (constant value 12/5)
        """
        return jnp.full_like(self.loc, self._gumbel_kurtosis)

    def skew(self) -> jnp.ndarray:
        """
        Compute the skewness of the Gumbel Type I GEVD.

        The skewness is constant:
        γ₃ ≈ 1.1396470994

        This positive value indicates right-skewness (longer right tail),
        which is characteristic of extreme value distributions for maxima.
        The value is independent of distribution parameters.

        Returns:
            Skewness (constant positive value ≈ 1.14)
        """
        return jnp.full_like(self.loc, self._gumbel_skewness)

    def entropy(self) -> jnp.ndarray:
        """
        Compute the differential entropy of the Gumbel Type I GEVD.

        The entropy is:
        H = log(σ) + γ + 1
        where γ is the Euler-Mascheroni constant.

        This measures the uncertainty in the distribution and increases
        with the scale parameter.

        Returns:
            Differential entropy in nats
        """
        return jnp.log(self.scale) + self._euler_gamma + 1.0

    def survival_function(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the survival function S(x) = 1 - F(x).

        For Gumbel Type I:
        S(x) = 1 - exp(-exp(-z)) where z = (x - μ)/σ

        This represents the probability of exceeding a given value,
        which is fundamental for risk analysis and return level calculations.

        Args:
            value: Points at which to evaluate the survival function

        Returns:
            Survival probabilities
        """
        # Computed as `-expm1(-exp(-z))` rather than `1 - cdf` so the upper
        # tail stays numerically resolvable (1 - exp(-small) catastrophically
        # cancels to 0 once S(x) is below ~1e-7).
        z = (value - self.loc) / self.scale
        return -jnp.expm1(-jnp.exp(-z))

    def log_survival_function(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the log survival function log S(x) = log(1 - F(x)).

        For extreme quantiles, this provides better numerical stability
        than computing log(1 - F(x)) directly.

        Args:
            value: Points at which to evaluate log survival function

        Returns:
            Log survival probabilities
        """
        loc, scale = self.loc, self.scale
        z = (value - loc) / scale
        # S(x) = 1 - F(x) = -expm1(-exp(-z)).
        # log S(x) = log(-expm1(-exp(-z))) — numerically stable across both
        # tails. For large z: exp(-z) → 0, -expm1(-u) → u, so log S → -z
        # (the correct Gumbel upper-tail asymptote). The previous branch
        # returned ≈ -exp(-z) at large z, which tends to 0 instead of the
        # large negative log-survival.
        return jnp.log(-jnp.expm1(-jnp.exp(-z)))

    def hazard_rate(self, value: jnp.ndarray) -> jnp.ndarray:
        r"""Hazard rate :math:`h(x) = f(x) / S(x)` for the Gumbel.

        With :math:`F(x) = \exp(-e^{-z})` and :math:`z = (x - \mu)/\sigma`,
        :math:`S(x) = 1 - F(x) = -\mathrm{expm1}(-e^{-z})`, so the hazard
        does *not* simplify to :math:`e^{z}/\sigma` — the upper-tail limit
        is :math:`1/\sigma`, not diverging.
        """
        z = (value - self.loc) / self.scale
        log_pdf = gumbel_log_prob(value, self.loc, self.scale)
        # log S(x) = log(-expm1(-exp(-z))) — numerically stable across both tails.
        log_surv = jnp.log(-jnp.expm1(-jnp.exp(-z)))
        return jnp.exp(log_pdf - log_surv)

    def cumulative_hazard_rate(self, value: jnp.ndarray) -> jnp.ndarray:
        r"""Cumulative hazard :math:`\Lambda(x) = -\log S(x)` for the Gumbel.

        With :math:`S(x) = -\mathrm{expm1}(-e^{-z})`,
        :math:`\Lambda(x) = -\log(-\mathrm{expm1}(-e^{-z}))`.
        """
        z = (value - self.loc) / self.scale
        return -jnp.log(-jnp.expm1(-jnp.exp(-z)))

    def return_level(self, return_period: float | jnp.ndarray) -> jnp.ndarray:
        """Return level. Thin wrapper for ``gumbel_return_level``."""
        return gumbel_return_level(return_period, self.loc, self.scale)

    def tail_index(self) -> jnp.ndarray:
        """
        Compute the tail index for Gumbel Type I GEVD.

        For Gumbel distributions, there is no power-law tail behavior.
        The tail index concept (from extreme value theory) is not applicable
        since Gumbel has exponential (not polynomial) tails.

        Returns:
            NaN (tail index not defined for exponential tails)
        """
        return jnp.full_like(self.loc, jnp.nan)

    def exceedance_probability(self, threshold: jnp.ndarray) -> jnp.ndarray:
        """
        Compute probability of exceeding a threshold: P(X > threshold).

        This is equivalent to the survival function and follows the
        double exponential decay characteristic of Gumbel distributions.

        Args:
            threshold: Threshold value

        Returns:
            Exceedance probabilities
        """
        return self.survival_function(threshold)

    def conditional_excess_mean(self, threshold: jnp.ndarray) -> jnp.ndarray:
        r"""Mean excess :math:`E[X - u \mid X > u]` for the Gumbel.

        Computed from the identity

        .. math:: E[X - u \mid X > u] \, S(u) = \int_u^\infty S(x)\,dx

        via a trapezoidal quadrature over a grid from ``u`` to ``u + 50σ``
        (well past the Gumbel tail). Works for scalar, batched, and
        broadcasted ``threshold`` / ``scale`` — the grid axis is always
        placed last so trapezoidal integration stays on the quadrature
        axis rather than the batch axis.

        Returns ``NaN`` where the survival probability is effectively zero.
        """
        threshold_arr = jnp.asarray(threshold)
        scale_arr = jnp.asarray(self.scale)

        # Trapezoidal quadrature of the IBP form
        #   E[X - u | X > u] = (1/S(u)) ∫_u^∞ S(x) dx
        # with an *adaptive* upper cap chosen as the further of
        # ``u + 50σ`` (enough for exponential-tail decay from u) and
        # ``icdf(1 - 1e-6)`` (absolute far-tail quantile). The fixed
        # ``u + 50σ`` cap used previously truncated substantial mass
        # when u sat far below the location (e.g. u = μ - 1000σ),
        # biasing the mean excess low by O(|u|). The adaptive cap
        # handles both far-left and far-right thresholds correctly.
        n_grid = 1024
        q_top = jnp.asarray(1.0 - 1e-6, dtype=scale_arr.dtype)
        x_top = self.icdf(q_top)
        upper = jnp.maximum(threshold_arr + 50.0 * scale_arr, x_top)

        unit = jnp.linspace(0.0, 1.0, n_grid)
        t_exp = jnp.expand_dims(threshold_arr, axis=-1)
        u_exp = jnp.expand_dims(upper, axis=-1)
        x_grid = t_exp + (u_exp - t_exp) * unit  # (..., n_grid)

        integrand = self.survival_function(x_grid)
        integral = jnp.trapezoid(integrand, x=x_grid, axis=-1)
        s_u = self.survival_function(threshold_arr)
        return jnp.where(s_u > 1e-15, integral / s_u, jnp.nan)

    def median(self) -> jnp.ndarray:
        """
        Compute the median (50th percentile) of the Gumbel Type I GEVD.

        The median is:
        Q(0.5) = μ - σ * ln(-ln(0.5)) = μ - σ * ln(ln(2))

        Returns:
            Median value of the distribution
        """
        return self.icdf(0.5)

    def interquartile_range(self) -> jnp.ndarray:
        """
        Compute the interquartile range (IQR) of the Gumbel Type I GEVD.

        IQR = Q(0.75) - Q(0.25)

        Returns:
            Interquartile range
        """
        q25 = self.icdf(0.25)
        q75 = self.icdf(0.75)
        return q75 - q25

    def coefficient_of_variation(self) -> jnp.ndarray:
        """
        Compute the coefficient of variation (CV = σ/μ) for risk assessment.

        For Gumbel distributions:
        CV = (σπ/√6) / (μ + σγ)

        Returns:
            Coefficient of variation
        """
        std_dev = jnp.sqrt(self.variance)
        return std_dev / jnp.abs(self.mean)

    def moment_generating_function(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the moment generating function M(t) = E[exp(tX)].

        For Gumbel Type I with t < 1/σ:
        M(t) = exp(μt) * Γ(1 - σt)
        where Γ is the gamma function.

        Args:
            t: Values at which to evaluate MGF (must satisfy t < 1/σ)

        Returns:
            Moment generating function values
        """
        loc, scale = self.loc, self.scale

        # MGF exists for t < 1/σ
        valid = t < (1.0 / scale)

        # M(t) = exp(μt) * Γ(1 - σt)
        mgf_val = jnp.exp(loc * t) * jnp.exp(gammaln(1.0 - scale * t))

        return jnp.where(valid, mgf_val, jnp.inf)

    def characteristic_function(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the characteristic function φ(t) = E[exp(itX)].

        For Gumbel Type I:
        φ(t) = exp(iμt) * Γ(1 - iσt)

        ``jax.scipy.special.gammaln`` is real-only, so the complex argument
        ``1 - iσt`` is evaluated via ``jax.pure_callback`` to
        ``scipy.special.loggamma``. This is host-hop but numerically
        correct — a naive JAX call would have failed or returned garbage.

        Args:
            t: Real values at which to evaluate the characteristic function

        Returns:
            Complex-valued characteristic function
        """
        loc, scale = self.loc, self.scale
        t_arr = jnp.asarray(t)
        scale_arr = jnp.asarray(scale)
        loc_arr = jnp.asarray(loc)

        complex_arg = 1.0 - 1j * scale_arr * t_arr
        log_gamma = jax.pure_callback(
            _host_complex_loggamma,
            jax.ShapeDtypeStruct(complex_arg.shape, complex_arg.dtype),
            complex_arg,
        )
        return jnp.exp(1j * loc_arr * t_arr) * jnp.exp(log_gamma)

    def gumbel_probability_paper_coordinates(
        self, value: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Convert values to Gumbel probability paper coordinates for graphical analysis.

        Transforms (x, F(x)) to coordinates where the Gumbel CDF appears linear:
        y-coordinate: -ln(-ln(F(x)))
        x-coordinate: x (unchanged)

        Args:
            value: Data values

        Returns:
            Tuple of (x-coordinates, y-coordinates) for probability paper
        """
        cdf_vals = self.cdf(value)

        # Avoid numerical issues at boundaries
        cdf_vals = jnp.clip(cdf_vals, 1e-15, 1 - 1e-15)

        # Gumbel probability paper transformation: y = -ln(-ln(F))
        y_coords = -jnp.log(-jnp.log(cdf_vals))

        return value, y_coords

    def expand(self, batch_shape: tuple[int, ...]) -> dist.Distribution:
        """Expand to ``batch_shape`` by reconstructing via ``__init__``.

        Going through the constructor re-populates all cached constants
        (``_pi_squared_over_six``, ``_gumbel_skewness``,
        ``_gumbel_kurtosis``, ``_euler_gamma``) on the returned instance.
        Bypassing ``__init__`` broke ``variance``/``skew``/``kurtosis``/
        ``entropy`` on expanded distributions.
        """
        batch_shape = tuple(batch_shape)
        if batch_shape == self.batch_shape:
            return self
        return type(self)(
            loc=jnp.broadcast_to(self.loc, batch_shape),
            scale=jnp.broadcast_to(self.scale, batch_shape),
            validate_args=self._validate_args,
        )


# Convenient aliases for backward compatibility and clarity
GumbelGEVD = GumbelType1GEVD
GumbelDistribution = GumbelType1GEVD
ExponentialTailGEVD = GumbelType1GEVD
