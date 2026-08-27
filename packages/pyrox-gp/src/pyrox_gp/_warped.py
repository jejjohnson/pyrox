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


def _apply_warp(
    warp: AbstractBijection,
    f: Float[Array, " ..."],
    X: Float[Array, " ..."] | None = None,
) -> Array:
    """Apply a scalar bijection elementwise, preserving the shape of ``f``.

    Shape-agnostic on purpose: `pyrox_gp.svgp_elbo` integrates
    per point and calls ``log_prob`` with ``f`` of shape ``(1,)``, while the
    predictive path and the advanced inference strategies pass full ``(N,)``
    arrays.

    For conditional warps, ``X`` is broadcast against ``f``: `svgp_elbo`
    integrates per point, so ``f`` has shape ``(1,)`` and ``X`` is that
    point's ``(D,)`` input, while the predictive path passes ``(N,)`` and
    ``(N, D)``.
    """
    if warp.shape == ():
        fn = lambda v, c: warp.transform(v, c)
    elif warp.shape == (1,):
        fn = lambda v, c: warp.transform(v[None], c)[0]
    else:
        raise ValueError(
            f"Warp event shape must be () or (1,); got {warp.shape}. "
            "A transformed GP warps each scalar function value independently."
        )

    flat = jnp.reshape(f, (-1,))
    if warp.cond_shape is None:
        out = jax.vmap(lambda v: fn(v, None))(flat)
    else:
        if X is None:
            raise ValueError(
                "A conditional warp (cond_shape is not None) needs the "
                "inputs it is conditioned on, but none were supplied. "
                "`svgp_elbo` threads X through automatically; the CVI / "
                "natural-gradient, sparse-Markov and multi-output ELL "
                "paths do not, so use `svgp_elbo` (or plain MAP/VI) with "
                "a conditional warp, and an unconditional warp elsewhere."
            )
        cond = jnp.broadcast_to(
            jnp.reshape(X, (-1, *warp.cond_shape)),
            (flat.shape[0], *warp.cond_shape),
        )
        out = jax.vmap(fn)(flat, cond)
    return jnp.reshape(out, jnp.shape(f))


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
        X: Float[Array, " ..."] | None = None,
    ) -> Float[Array, ""]:
        r"""Sum of per-point Gaussian log-densities about $G(f)$.

        ``X`` is required when the warp is conditional (``cond_shape``
        set) and ignored otherwise. Note the cost: the expected
        log-likelihood evaluates the warp at every quadrature node, so a
        conditional warp pays ``order`` conditioner forward passes per
        data point.
        """
        g = _apply_warp(self.warp, f, X)
        return nd.Normal(g, jnp.sqrt(self.noise_var)).log_prob(y).sum()


def warped_predictive_moments(
    lik: WarpedGaussianLikelihood,
    f_loc: Float[Array, " N"],
    f_var: Float[Array, " N"],
    X: Float[Array, "N D"] | None = None,
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
        X: Inputs of shape ``(N, D)``; required when the warp is
            conditional, ignored otherwise. Broadcast over the quadrature
            nodes.
        order: Gauss-Hermite nodes; at least 2, since one node carries no
            spread. Values above ~256 are numerically unreliable
            (``hermegauss`` overflows to ``NaN`` by order 512).

    Returns:
        Tuple of ``(mean, variance)`` in observation space, both ``(N,)``.
    """
    if not 2 <= order <= 256:
        # order == 1 places a single node at the mean, so the centered
        # second moment is identically zero and every bit of latent
        # uncertainty would be silently discarded.
        raise ValueError(
            f"order must be in [2, 256]; got {order}. A single quadrature "
            "node cannot represent any spread."
        )
    x, w = np.polynomial.hermite_e.hermegauss(order)
    x = jnp.asarray(x)
    w = jnp.asarray(w) / np.sqrt(2.0 * np.pi)
    fs = f_loc[None, :] + jnp.sqrt(f_var)[None, :] * x[:, None]
    # _apply_warp flattens (order, N) row-major, so the per-point inputs
    # tile across the node axis.
    g = _apply_warp(
        lik.warp,
        fs,
        None if X is None else jnp.broadcast_to(X, (order, *X.shape)),
    )
    m1 = jnp.sum(w[:, None] * g, axis=0)
    # Centered second moment. Subtracting m1**2 from the raw second moment
    # cancels catastrophically once G's output carries a large offset
    # relative to its spread (an Affine warp centred near 1e4 loses the
    # whole variance in float32, and can go negative).
    m2 = lik.noise_var + jnp.sum(w[:, None] * (g - m1[None, :]) ** 2, axis=0)
    return m1, m2
