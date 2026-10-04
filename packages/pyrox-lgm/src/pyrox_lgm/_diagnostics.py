r"""Model diagnostics without refits: DIC, WAIC, CPO and PIT (P9).

All four are expectations over the posterior marginals of the linear
predictor $\eta_i$, which `inla()` keeps per design point as Gaussians
$\mathcal N(\bar\eta_{ik}, v_{ik})$ (means from the mode, variances from the
Takahashi selected inverse), mixed over the design with weights $w_k$. Each
site's expectation is a 1-D Gauss–Hermite quadrature per design point, so no
model is refitted:

Leave-one-out uses $\pi(\eta_i\mid y_{-i}) \propto \pi(\eta_i\mid
y)/\pi(y_i\mid\eta_i)$, as R-INLA does, keeping the hyperparameters' full-data
posterior. Dividing the Gaussian marginal by $\pi(y_i\mid\eta_i)$ itself is
an integral that diverges for most likelihoods ($1/\pi$ grows like
$e^{e^\eta}$ for a Poisson), so the site is removed at the order of the
Laplace approximation that built the marginal: with $g_{ik}$ and $W_{ik}$ the
gradient and negative Hessian of $\log\pi(y_i\mid\eta)$ at $\bar\eta_{ik}$,
the cavity is

$$
\pi_k(\eta_i\mid y_{-i}) \approx \mathcal N\big(\bar\eta_{ik} - g_{ik}c_{ik},\;
c_{ik}\big),\qquad c_{ik} = \frac{v_{ik}}{1 - W_{ik}v_{ik}},
$$

and, with $Z_{ik} = \int \pi(y_i\mid\eta)\,\pi_k(\eta\mid y_{-i})\,d\eta$
(the cavity carries design point $k$'s leave-one-out weight $\propto w_k /
Z_{ik}$),

$$
\mathrm{CPO}_i = \Big(\sum_k w_k / Z_{ik}\Big)^{-1},
\qquad
\mathrm{PIT}_i = \mathrm{CPO}_i\sum_k \frac{w_k}{Z_{ik}}
    \int F(y_i\mid\eta)\,\pi_k(\eta\mid y_{-i})\,d\eta,
$$

both exact for a Gaussian likelihood at fixed hyperparameters. WAIC
(Watanabe, 2010) uses
$\mathrm{lppd}_i = \log\mathbb E[\pi(y_i\mid\eta_i)]$ and
$p_{\mathrm{WAIC},i} = \operatorname{var}[\log\pi(y_i\mid\eta_i)]$; DIC
(Spiegelhalter et al., 2002) the mean deviance and the deviance at the
posterior mean of $\eta$ and the hyperparameter mode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float


if TYPE_CHECKING:
    from pyrox_lgm._result import INLAResult


class Diagnostics(NamedTuple):
    """Leave-one-out and information criteria of an `INLAResult`.

    Attributes:
        cpo: Conditional predictive ordinates $\\pi(y_i\\mid y_{-i})$.
        pit: Probability integral transforms $P(Y_i \\le y_i \\mid y_{-i})$.
        log_score: $-\\frac1n\\sum_i \\log\\mathrm{CPO}_i$ (lower is better).
        waic: $-2\\sum_i(\\mathrm{lppd}_i - p_{\\mathrm{WAIC},i})$.
        p_waic: Effective number of parameters of WAIC.
        dic: $\\bar D + p_D$.
        p_d: Effective number of parameters of DIC, $\\bar D - \\hat D$.
    """

    cpo: Float[Array, " n"]
    pit: Float[Array, " n"]
    log_score: Float[Array, ""]
    waic: Float[Array, ""]
    p_waic: Float[Array, ""]
    dic: Float[Array, ""]
    p_d: Float[Array, ""]


def diagnostics(result: INLAResult, *, order: int = 80) -> Diagnostics:
    """DIC, WAIC, CPO and PIT of an `INLAResult`, with no refits.

    Args:
        result: What `inla()` returned.
        order: Gauss–Hermite nodes per site and design point.

    Returns:
        A `Diagnostics`.

    Examples:
        >>> import numpy as np
        >>> import pyrox_lgm as lgm
        >>> t = np.arange(25)
        >>> y = np.sin(t / 4.0) + 0.2 * np.cos(5.0 * t)
        >>> model = lgm.LGM((lgm.RW2(25, name="t"),), lgm.FixedEffects(("intercept",)))
        >>> d = lgm.inla(model, {"y": y, "t": t}).diagnostics()
        >>> d.cpo.shape, bool(((d.pit > 0) & (d.pit < 1)).all())
        ((25,), True)
    """
    model, data = result.model, result.data
    y = jnp.asarray(data["y"])
    nodes, gh = np.polynomial.hermite_e.hermegauss(order)
    log_g = jnp.log(jnp.asarray(gh / gh.sum()))
    z = jnp.asarray(nodes)
    lik = model.likelihood

    def nodes_of(m, v):  # (n, order) quadrature nodes of N(m, v)
        return m[:, None] + jnp.sqrt(v)[:, None] * z[None, :]

    def per_point(u, m, v):
        theta = model._likelihood_theta(model.unflatten(u))
        logp = lik.site_log_prob(y[:, None], nodes_of(m, v), theta, data)
        # Sites are independent given eta: the gradient of the sum is the
        # per-site gradient, and so on for the curvature.
        site = lambda e: jnp.sum(lik.site_log_prob(y, e, theta, data))
        g = jax.grad(site)(m)
        W = -jax.grad(lambda e: jnp.sum(jax.grad(site)(e)))(m)
        a = jnp.maximum(1.0 - W * v, 1e-12)
        c = v / a
        cav_mean = m - g * c
        # Z_ik on the marginal's own nodes, as E[p * cavity / marginal]: the
        # ratio is constant for a Gaussian likelihood and smooth for a
        # log-concave one, where the wider cavity's nodes under-resolve p.
        # Its log, expanded in the standard node z (eta = m + sqrt(v) z), has
        # no division by v, so a deterministic predictor (v = 0) gives p(y|m).
        zz = z[None, :]
        log_ratio = (
            0.5 * zz**2 * (1.0 - a)[:, None]
            - (jnp.sqrt(v) * g)[:, None] * zz
            - (0.5 * g**2 * c)[:, None]
            + 0.5 * jnp.log(a)[:, None]
        )
        log_z = logsumexp(log_g[None, :] + logp + log_ratio, axis=1)
        # F is bounded, so its cavity expectation takes the cavity's nodes.
        cdf = lik.site_cdf(y[:, None], nodes_of(cav_mean, c), theta, data)
        return logp, log_z, jnp.sum(gh / gh.sum() * cdf, axis=1)

    logp, log_z, cav_pit = jax.vmap(per_point)(
        result.theta_points, result.predictor_means, result.predictor_variances
    )  # (K, n, order), (K, n), (K, n)
    log_w = jnp.log(result.theta_weights)[:, None, None] + log_g[None, None, :]

    def expect_log(values):  # log E[exp(values)] over (k, node)
        return logsumexp(log_w + values, axis=(0, 2))

    # Design point k's leave-one-out weight is proportional to w_k / Z_ik.
    log_loo = jnp.log(result.theta_weights)[:, None] - log_z
    log_cpo = -logsumexp(log_loo, axis=0)
    cpo = jnp.exp(log_cpo)
    pit = jnp.exp(log_cpo + logsumexp(log_loo, axis=0, b=cav_pit))
    w = jnp.exp(log_w)
    mean_logp = jnp.sum(w * logp, axis=(0, 2))
    # Centred, so a large common log-density does not cancel (float32).
    var_logp = jnp.sum(w * (logp - mean_logp[None, :, None]) ** 2, axis=(0, 2))
    lppd = expect_log(logp)
    p_waic = jnp.sum(var_logp)
    waic = -2.0 * (jnp.sum(lppd) - p_waic)

    d_bar = -2.0 * jnp.sum(mean_logp)
    theta_hat = model._likelihood_theta(model.unflatten(result.theta_mode))
    eta_bar = result.linear_predictor.mean
    d_hat = -2.0 * jnp.sum(lik.site_log_prob(y, eta_bar, theta_hat, data))
    p_d = d_bar - d_hat
    return Diagnostics(
        cpo=cpo,
        pit=jnp.clip(pit, 0.0, 1.0),
        log_score=-jnp.mean(log_cpo),
        waic=waic,
        p_waic=p_waic,
        dic=d_bar + p_d,
        p_d=p_d,
    )
