"""Transformed Gaussian processes: a GP composed with a monotone warp.

A transformed GP (Maronas et al., AISTATS 2021) is an SVGP whose likelihood
composes an elementwise monotone map $G$:

$$
\\mathcal{L} = \\sum_n \\mathbb{E}_{q(f_n)}[\\log p(y_n \\mid G(f_n))]
             - \\mathrm{KL}[q(u) \\Vert p(u)]
$$

The warp appears only in the expected log-likelihood -- the KL term is
untouched -- so this composes with every existing inference path in
`pyrox_gp` without modification.

The ELBO is valid for **any** map $G$ -- the training path never inverts
the warp, so neither ``inverse`` nor ``log_abs_det_jacobian`` is required.
Monotonicity buys interpretation and identifiability (it stops $G$ folding
the latent space), not correctness.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as nd
from flowjax.bijections import AbstractBijection
from jaxtyping import Array, Float

from pyrox_gp._protocols import Likelihood


def _apply_warp(warp: AbstractBijection, f: Float[Array, " ..."]) -> Array:
    """Apply a scalar bijection elementwise, preserving shape.

    Shape-agnostic on purpose: `pyrox_gp.svgp_elbo` integrates
    per point and calls ``log_prob`` with ``f`` of shape ``(1,)``, while the
    predictive path and the advanced inference strategies pass full ``(N,)``
    arrays.
    """
    if warp.cond_shape is not None:
        raise ValueError(
            "Conditional warps are not supported here: this likelihood never "
            "supplies a condition, so `transform` would fail at training and "
            "prediction time. Use an unconditional bijection."
        )
    if warp.shape == ():
        fn = warp.transform
    elif warp.shape == (1,):
        fn = lambda v: warp.transform(v[None])[0]
    else:
        raise ValueError(
            f"Warp event shape must be () or (1,); got {warp.shape}. "
            "A transformed GP warps each scalar function value independently."
        )
    return jnp.reshape(jax.vmap(fn)(jnp.reshape(f, (-1,))), jnp.shape(f))


class WarpedGaussianLikelihood(Likelihood):
    r"""$p(y \mid f) = N(y \mid G(f), \sigma^2)$ -- the transformed-GP model.

    The warp is a **child module**, so its parameters are ordinary trainable
    leaves. Passing a warp through a lambda to
    [`DistLikelihood`][pyrox_gp.DistLikelihood] does *not* work: its
    ``dist_fn`` is a static field, so the warp is silently frozen and never
    trains.

    The expected log-likelihood has no closed form, so
    [`svgp_elbo`][pyrox_gp.svgp_elbo] requires an integrator.

    !!! warning "Prefer Gauss-Hermite, and prefer a smooth warp"
        The warped integrand has heavy tails, so Monte Carlo integration is
        high variance -- roughly 4M samples to match Gauss-Hermite at order
        20. Gauss-Hermite in turn converges spectrally only for *analytic*
        warps: piecewise ones such as ``RationalQuadraticSpline`` stall
        around a ``3e-3`` error floor and their error is **non-monotone** in
        quadrature order, so raising the order is not a valid convergence
        check. ``gauss_flows.MixtureGaussianCDF`` is smooth and reaches
        machine precision.

    Note ``eqx.filter_jit`` is required over plain ``jax.jit`` when jitting
    around this likelihood -- flowjax spline pytrees carry string leaves.

    Attributes:
        warp: Any ``flowjax.bijections.AbstractBijection`` with event shape
            ``()`` or ``(1,)``.
        noise_var: Observation noise variance $\sigma^2$.

    Examples:
        >>> import jax.numpy as jnp
        >>> from flowjax.bijections import RationalQuadraticSpline
        >>> lik = WarpedGaussianLikelihood(
        ...     warp=RationalQuadraticSpline(knots=8, interval=4.0),
        ...     noise_var=jnp.asarray(0.1),
        ... )
        >>> float(lik.log_prob(jnp.zeros(3), jnp.zeros(3))) < 0.0
        True
    """

    warp: AbstractBijection
    noise_var: Float[Array, ""]

    def log_prob(
        self,
        f: Float[Array, " ..."],
        y: Float[Array, " ..."],
    ) -> Float[Array, ""]:
        r"""Sum of per-point Gaussian log-densities about $G(f)$."""
        g = _apply_warp(self.warp, f)
        return nd.Normal(g, jnp.sqrt(self.noise_var)).log_prob(y).sum()


def warped_predictive_moments(
    lik: WarpedGaussianLikelihood,
    f_loc: Float[Array, " N"],
    f_var: Float[Array, " N"],
    *,
    order: int = 32,
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    r"""Moments of $y = G(f) + \epsilon$ under $q(f)$, by quadrature.

    $$
    m_1 = \mathbb{E}_{q(f)}[G(f)], \qquad
    m_2 = \sigma^2 + \mathbb{E}_{q(f)}[G(f)^2] - m_1^2
    $$

    Note $m_1 \neq G(\mathbb{E}[f])$ whenever $G$ is nonlinear -- warping the
    posterior mean is the natural-looking mistake and is badly biased for a
    skewed warp.

    Args:
        lik: The warped likelihood.
        f_loc: Posterior means of the base GP, shape ``(N,)``.
        f_var: Posterior marginal variances of the base GP, shape ``(N,)``.
        order: Gauss-Hermite nodes. Values above ~256 are numerically
            unreliable (``hermegauss`` overflows to ``NaN`` by order 512).

    Returns:
        Tuple of ``(mean, variance)`` in observation space, both ``(N,)``.
    """
    if not 1 <= order <= 256:
        raise ValueError(f"order must be in [1, 256]; got {order}.")
    x, w = np.polynomial.hermite_e.hermegauss(order)
    x = jnp.asarray(x)
    w = jnp.asarray(w) / np.sqrt(2.0 * np.pi)
    fs = f_loc[None, :] + jnp.sqrt(f_var)[None, :] * x[:, None]
    g = _apply_warp(lik.warp, fs)
    m1 = jnp.sum(w[:, None] * g, axis=0)
    # Centered second moment. Subtracting m1**2 from the raw second moment
    # cancels catastrophically once G's output carries a large offset
    # relative to its spread (an Affine warp centred near 1e4 loses the
    # whole variance in float32, and can go negative).
    m2 = lik.noise_var + jnp.sum(w[:, None] * (g - m1[None, :]) ** 2, axis=0)
    return m1, m2
