"""INLAResult: posterior marginals and the marginal likelihood (P8)."""

from __future__ import annotations

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


def mixture_summary(
    means: Float[Array, "K n"],
    variances: Float[Array, "K n"],
    weights: Float[Array, " K"],
) -> Summary:
    """Summary of the Gaussian mixture $\\sum_k w_k\\,\\mathcal N(m_k, v_k)$ per column.

    Quantiles by 80 bisection steps on the mixture CDF.
    """
    w = weights[:, None]
    mean = jnp.sum(w * means, axis=0)
    second = jnp.sum(w * (variances + means**2), axis=0)
    sd = jnp.sqrt(jnp.maximum(second - mean**2, 0.0))
    s = jnp.sqrt(variances)
    lo0 = jnp.min(means - 10.0 * s, axis=0)
    hi0 = jnp.max(means + 10.0 * s, axis=0)

    def quantile(p):
        def body(_, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            cdf = jnp.sum(w * ndtr((mid[None, :] - means) / s), axis=0)
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
        log_marginal_likelihood: $\\log\\tilde\\pi(y)$, Gaussian approximation
            over $\\theta$ (R-INLA's "Gaussian" mlik).
        n_dropped: Design points dropped because their inner Newton did not
            converge.
    """

    fixed: dict[str, Summary]
    random: dict[str, Summary]
    hyperpar: dict[str, Summary]
    theta_mode: Float[Array, " m"]
    theta_points: Float[Array, "K m"]
    theta_weights: Float[Array, " K"]
    latent_means: Float[Array, "K N"]
    latent_variances: Float[Array, "K N"]
    log_marginal_likelihood: Float[Array, ""]
    n_dropped: int = eqx.field(static=True)
    model: Any
    data: Mapping[str, Any]
    max_newton: int = eqx.field(static=True)

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
                self.theta_points[k], self.data, projector=A, max_newton=self.max_newton
            )
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
            import xarray as xr
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
        ds = xr.Dataset(variables)
        ds.attrs["log_marginal_likelihood"] = float(self.log_marginal_likelihood)
        ds.attrs["n_dropped"] = self.n_dropped
        return ds
