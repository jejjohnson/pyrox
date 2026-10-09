"""INLAResult: posterior marginals and the marginal likelihood (P8)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr
from jaxtyping import Array, ArrayLike, Float


_PROBS = (0.025, 0.5, 0.975)


class Summary(NamedTuple):
    """Posterior summary: mean, sd and the 2.5 / 50 / 97.5 % quantiles."""

    mean: Array
    sd: Array
    q025: Array
    q50: Array
    q975: Array


# A skew-normal's skewness is below (4 - pi) / 2 * (2 / (pi - 2))^1.5 ~ 0.9953.
_MAX_SKEW = 0.99
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(40)
# Simpson points (odd) for the moments of a range-limited mixture.
_N_LIMIT = 801
# Columns per Simpson chunk, bounding the (801, chunk) grids.
_CHUNK = 2048


def skew_normal_params(mean, var, skew):
    """``(xi, omega, alpha)`` of the skew-normal with these three moments.

    The skewness is clipped to the family's range (|skew| < 0.9953).
    """
    s = jnp.clip(skew, -_MAX_SKEW, _MAX_SKEW)
    r = (2.0 * jnp.abs(s) / (4.0 - math.pi)) ** (2.0 / 3.0)
    delta = jnp.sign(s) * jnp.sqrt(0.5 * math.pi * r / (1.0 + r))
    omega = jnp.sqrt(var / (1.0 - 2.0 * delta**2 / math.pi))
    xi = mean - omega * delta * math.sqrt(2.0 / math.pi)
    return xi, omega, delta / jnp.sqrt(1.0 - delta**2)


def skew_normal_cdf(x, xi, omega, alpha):
    """``Phi(z) - 2 T(z, alpha)`` with Owen's T by 40-node Gauss-Legendre."""
    z = (x - xi) / omega
    u = jnp.asarray(0.5 * (_GL_NODES + 1.0))
    w = jnp.asarray(0.5 * _GL_WEIGHTS)
    t2 = 1.0 + (alpha[..., None] * u) ** 2
    owen = (
        alpha
        / (2.0 * math.pi)
        * jnp.sum(w * jnp.exp(-0.5 * z[..., None] ** 2 * t2) / t2, axis=-1)
    )
    return ndtr(z) - 2.0 * owen


def _truncated_gaussian_moments(means, s, weights, a, b, center):
    r"""Mean and variance of $\sum_k w_k\,\mathcal N(m_k, s_k^2)$ on ``[a, b]``.

    Per component, with standardised bounds $\alpha, \beta$ and
    $Z = \Phi(\beta) - \Phi(\alpha)$: $\int_a^b (x - m) f =
    s(\varphi(\alpha) - \varphi(\beta))$ and $\int_a^b (x - m)^2 f =
    s^2[Z + \alpha\varphi(\alpha) - \beta\varphi(\beta)]$, combined about
    ``center`` (the mixture mean) to avoid cancellation.
    """
    s = jnp.maximum(s, 1e-150)
    al, be = (a[None, :] - means) / s, (b[None, :] - means) / s
    pa = jnp.exp(-0.5 * al**2) / math.sqrt(2.0 * math.pi)
    pb = jnp.exp(-0.5 * be**2) / math.sqrt(2.0 * math.pi)
    z = ndtr(be) - ndtr(al)
    d = means - center[None, :]
    first = s * (pa - pb)
    second = s**2 * (z + al * pa - be * pb)
    w = weights[:, None]
    m0 = jnp.sum(w * z, axis=0)
    m1 = jnp.sum(w * (d * z + first), axis=0) / m0
    m2 = jnp.sum(w * (second + 2.0 * d * first + d**2 * z), axis=0) / m0
    return center + m1, m2 - m1**2


def _truncated_grid_moments(weights, xi, omega, alpha, a, b):
    """Mean and variance of a skew-normal mixture on ``[a, b]``: Simpson's
    rule on ``_N_LIMIT`` points, in column chunks of at most ``_CHUNK``."""
    n = a.shape[0]
    chunk = min(n, _CHUNK)
    pad = -n % chunk
    t = jnp.linspace(0.0, 1.0, _N_LIMIT)
    simpson = np.ones(_N_LIMIT)
    simpson[1:-1:2], simpson[2:-1:2] = 4.0, 2.0
    simpson = jnp.asarray(simpson)[:, None]

    def moments(cols):
        xi_c, om_c, al_c, lo, hi = cols  # (K, chunk) x3, (chunk,) x2
        x = lo[None, :] + (hi - lo)[None, :] * t[:, None]

        def add(k, acc):
            z = (x - xi_c[k]) / om_c[k]
            pdf = 2.0 * jnp.exp(-0.5 * z**2) * ndtr(al_c[k] * z)
            return acc + weights[k] * pdf / (math.sqrt(2.0 * math.pi) * om_c[k])

        f = jax.lax.fori_loop(0, xi_c.shape[0], add, jnp.zeros_like(x)) * simpson
        m0 = jnp.sum(f, axis=0)
        mean_t = jnp.sum(f * x, axis=0) / m0
        return mean_t, jnp.sum(f * (x - mean_t) ** 2, axis=0) / m0

    def split(v):  # (..., n) -> (n_chunks, ..., chunk); pads repeat the last column
        v = jnp.pad(v, [(0, 0)] * (v.ndim - 1) + [(0, pad)], mode="edge")
        return jnp.moveaxis(v.reshape(*v.shape[:-1], -1, chunk), -2, 0)

    mean_t, var_t = jax.lax.map(
        moments, (split(xi), split(omega), split(alpha), split(a), split(b))
    )
    return mean_t.reshape(-1)[:n], var_t.reshape(-1)[:n]


def mixture_summary(
    means: Float[Array, "K n"],
    variances: Float[Array, "K n"],
    weights: Float[Array, " K"],
    skewness: Float[Array, "K n"] | None = None,
    limit: float | None = None,
) -> Summary:
    """Summary of the mixture $\\sum_k w_k\\,\\mathcal N(m_k, v_k)$ per column.

    With ``skewness``, each component is the skew-normal with that mean,
    variance and skewness (``strategy="sla"``); the mean and sd are the
    same either way. Quantiles by 80 bisection steps on the mixture CDF.

    With ``limit``, the summary is of the mixture restricted to its mean
    $\\pm$ ``limit`` sds, as R-INLA reports it (its combined marginal is a
    spline-corrected Gaussian on $\\pm 5$ sds, ``GMRFLib_density_combine``):
    a component far wider than the mixture, from a design point in a long
    tail of $\\theta$, then loses the mass outside that range. Moments by
    a closed form for Gaussian components (memory $O(Kn)$), and for
    skew-normal ones Simpson's rule on ``_N_LIMIT`` points in column chunks
    of bounded memory.
    """
    w = weights[:, None]
    mean = jnp.sum(w * means, axis=0)
    second = jnp.sum(w * (variances + means**2), axis=0)
    sd = jnp.sqrt(jnp.maximum(second - mean**2, 0.0))
    s = jnp.sqrt(variances)
    lo0 = jnp.min(means - 10.0 * s, axis=0)
    hi0 = jnp.max(means + 10.0 * s, axis=0)
    if skewness is None:

        def component_cdf(x):
            return ndtr((x[None, :] - means) / s)

    else:
        xi, omega, alpha = skew_normal_params(means, variances, skewness)

        def component_cdf(x):
            return skew_normal_cdf(x[None, :], xi, omega, alpha)

    p_lo = p_mass = None
    if limit is not None:
        a, b = mean - limit * sd, mean + limit * sd
        if skewness is None:
            mean_t, var_t = _truncated_gaussian_moments(means, s, weights, a, b, mean)
        else:
            mean_t, var_t = _truncated_grid_moments(weights, xi, omega, alpha, a, b)
        p_lo = jnp.sum(w * component_cdf(a), axis=0)
        p_mass = jnp.sum(w * component_cdf(b), axis=0) - p_lo
        ok = sd > 0
        mean = jnp.where(ok, mean_t, mean)
        sd = jnp.where(ok, jnp.sqrt(jnp.maximum(var_t, 0.0)), sd)

    def quantile(p):
        if p_lo is not None:
            p = p_lo + p * p_mass

        def body(_, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            cdf = jnp.sum(w * component_cdf(mid), axis=0)
            below = cdf < p
            return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

        lo, hi = jax.lax.fori_loop(0, 80, body, (lo0, hi0))
        return 0.5 * (lo + hi)

    q = [quantile(p) for p in _PROBS]
    return Summary(mean, sd, *q)


class INLAResult(eqx.Module):
    """What `inla()` returns.

    Attributes:
        fixed: Fixed effect name to its `Summary` (scalars).
        random: Component name to the `Summary` of its whole field (arrays
            over ``n_nodes``; a `BYM2` field is ``(b, u*)``).
        hyperpar: Hyperparameter key (``f"{name}.{k}"``) to its `Summary`,
            on the user (constrained) scale.
        theta_mode: Unconstrained mode $u^\\ast$.
        theta_points: Design points ``(K, m)``, unconstrained.
        theta_weights: Their normalised weights ``(K,)``.
        latent_means: Per-point latent means ``(K, N)`` (VB-corrected under
            ``strategy="vb"``).
        latent_variances: Per-point marginal variances ``(K, N)``.
        latent_skewness: Per-point marginal skewness ``(K, N)``: zero but
            under ``strategy="sla"``, whose summaries are skew-normal
            mixtures (`sample_latent` stays Gaussian, at the shifted means).
        predictor_means: Per-point means of the linear predictor
            $\\eta = Ax + o$ at the observations ``(K, n_obs)``.
        predictor_variances: Their variances ``(K, n_obs)``, from the
            Takahashi selected inverse (no solves).
        linear_predictor: `Summary` of the mixed predictor marginals.
        log_marginal_likelihood: $\\log\\tilde\\pi(y)$, Gaussian approximation
            over $\\theta$ (R-INLA's "Gaussian" mlik).
        n_dropped: Design points dropped because their inner Newton did not
            converge.
        newton_iters: The Newton budget each kept point converged with
            (``max_newton``, or four times it after a retry).
    """

    fixed: dict[str, Summary]
    random: dict[str, Summary]
    hyperpar: dict[str, Summary]
    theta_mode: Float[Array, " m"]
    theta_points: Float[Array, "K m"]
    theta_weights: Float[Array, " K"]
    latent_means: Float[Array, "K N"]
    latent_variances: Float[Array, "K N"]
    latent_skewness: Float[Array, "K N"]
    predictor_means: Float[Array, "K M"]
    predictor_variances: Float[Array, "K M"]
    linear_predictor: Summary
    log_marginal_likelihood: Float[Array, ""]
    n_dropped: int = eqx.field(static=True)
    model: Any
    data: Mapping[str, Any]
    newton_iters: tuple[int, ...] = eqx.field(static=True)

    def sample_latent(self, key: jax.Array, n: int) -> Float[Array, "n N"]:
        """Draws of the latent vector from the mixture over design points.

        Each draw picks a design point by its weight, then samples that
        point's Gaussian approximation $\\mathcal N(\\bar x_k, H_k^{-1})$
        (kriged onto the hard constraints) through the sparse Cholesky
        factor of $H_k$.
        """
        k_pick, k_draw = jax.random.split(key)
        picks = np.asarray(
            jax.random.choice(
                k_pick, self.theta_weights.shape[0], (n,), p=self.theta_weights
            )
        )
        out = np.empty((n, self.latent_means.shape[1]))
        keys = jax.random.split(k_draw, len(self.theta_weights))
        A = self.model.projector(self.data)
        for k in np.unique(picks):
            rows = np.flatnonzero(picks == k)
            fit = self.model.laplace(
                self.theta_points[k],
                self.data,
                projector=A,
                max_newton=self.newton_iters[k],
            )
            if not bool(fit.converged):
                raise RuntimeError(f"refit of design point {k} did not converge")
            prior = self.model.latent_prior(self.model.unflatten(self.theta_points[k]))
            z = jax.random.normal(keys[k], (rows.size, fit.mode.shape[0]))
            draws = jax.vmap(fit.factor.solve_lower_transpose)(z)
            if isinstance(prior, gx.IntrinsicGMRF):
                V = prior.null_space
                HV = jax.vmap(fit.factor.solve, in_axes=1, out_axes=1)(V)
                S = V.T @ HV
                draws = draws - (HV @ jnp.linalg.solve(S, V.T @ draws.T)).T
            out[rows] = np.asarray(self.latent_means[k] + draws)
        return jnp.asarray(out)

    def diagnostics(self, order: int = 80):
        """DIC, WAIC, CPO and PIT, without refits (see `pyrox_lgm.diagnostics`)."""
        from pyrox_lgm._diagnostics import diagnostics

        return diagnostics(self, order=order)

    def predict(
        self, new_data: Mapping[str, ArrayLike], key: jax.Array, n_samples: int = 1000
    ) -> Summary:
        """Monte Carlo summary of the linear predictor $\\eta = Ax + o$ at ``new_data``.

        ``new_data`` has the component indices and covariates of the new
        locations (and ``"y"``, any placeholder of the right length, which
        sizes the projector); ``"offset"`` is optional.
        """
        A = self.model.projector(new_data)
        x = self.sample_latent(key, n_samples)
        eta = jax.vmap(A.mv)(x)
        if "offset" in new_data:
            eta = eta + jnp.asarray(new_data["offset"])
        q = jnp.quantile(eta, jnp.asarray(_PROBS), axis=0)
        return Summary(eta.mean(0), eta.std(0), q[0], q[1], q[2])

    def to_xarray(self):
        """The summaries as an ``xarray.Dataset`` (needs the ``xarray`` extra)."""
        try:
            import xarray as xr  # ty: ignore[unresolved-import]
        except ImportError as err:  # pragma: no cover - optional extra
            raise ImportError(
                "to_xarray needs `pip install pyrox-lgm[xarray]`"
            ) from err
        variables = {}
        for group in ("fixed", "random", "hyperpar"):
            for name, s in getattr(self, group).items():
                for field, value in s._asdict().items():
                    arr = np.asarray(value)
                    dims = (
                        ()
                        if arr.ndim == 0
                        else tuple(f"{name}_{i}" for i in range(arr.ndim))
                    )
                    variables[f"{group}.{name}.{field}"] = (dims, arr)
        for field, value in self.linear_predictor._asdict().items():
            variables[f"linear_predictor.{field}"] = ("obs", np.asarray(value))
        ds = xr.Dataset(variables)
        ds.attrs["log_marginal_likelihood"] = float(self.log_marginal_likelihood)
        ds.attrs["n_dropped"] = self.n_dropped
        return ds
