"""Concrete Gaussian-expectation integrators for scalar latents.

Each integrator implements the :class:`pyrox.gp.Integrator` protocol::

    integrate(fn, mean, var) -> E_{q(f)}[fn(f)]

where ``q(f) = N(mean, var)`` is a *scalar* Gaussian (``mean``, ``var``
are scalars or per-point arrays). For full-covariance Gaussian
propagation use :mod:`gaussx`'s ``GaussianState``-based stack directly.

The non-trivial work — generating quadrature nodes, sampling — is
delegated to ``gaussx`` (:func:`gaussx.gauss_hermite_points`); pyrox
just exposes the simple per-site API that the advanced inference
strategies in :mod:`pyrox.gp._inference_nongauss` consume.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from gaussx import gauss_hermite_points
from jaxtyping import Array, Float

from pyrox.gp._protocols import Integrator


_INV_SQRT_2PI = 1.0 / jnp.sqrt(2.0 * jnp.pi)


class GaussHermite(Integrator):
    r"""Tensor-product Gauss-Hermite quadrature for scalar Gaussians.

    Computes ``E[fn(F)]`` with ``F ~ N(mean, var)`` as
    ``sum_i w_i' fn(mean + sqrt(var) * x_i)`` where ``(x_i, w_i)`` are
    standard probabilists' Hermite nodes from
    :func:`gaussx.gauss_hermite_points` and ``w_i' = w_i / sqrt(2*pi)``.
    Order ``20`` is exact for polynomials up to degree ``39``;
    accurate to ~ ``1e-12`` for smooth integrands at typical scales.

    Vectorizes over per-point ``mean`` / ``var`` so a single call
    handles a whole site sweep.

    Attributes:
        deg: Number of quadrature nodes. Default ``20``.
    """

    deg: int = eqx.field(static=True, default=20)

    def integrate(
        self,
        fn: Callable[[Float[Array, " ..."]], Float[Array, " ..."]],
        mean: Float[Array, " ..."],
        var: Float[Array, " ..."],
    ) -> Float[Array, " ..."]:
        nodes, weights = gauss_hermite_points(self.deg, dim=1)
        x = nodes[:, 0]  # (deg,)
        w = weights * _INV_SQRT_2PI
        std = jnp.sqrt(var)
        # Broadcast: mean (...,), x (deg,) -> samples (deg, ...)
        samples = mean[None, ...] + std[None, ...] * x.reshape((-1,) + (1,) * mean.ndim)
        vals = jax.vmap(fn)(samples)  # (deg, ...)
        return jnp.tensordot(w, vals, axes=1)


class MonteCarlo(Integrator):
    r"""Plain Monte Carlo integration for scalar Gaussians.

    Draws ``n`` samples per site from ``N(mean, var)`` and averages
    ``fn`` evaluated at the samples. Uses a fixed key drawn at
    construction time; pass a fresh ``key`` for each new estimator
    instance if you need stochastic estimates across calls.

    Cheaper than Gauss-Hermite for non-smooth integrands, but
    higher variance. For most non-conjugate likelihoods (Bernoulli,
    Poisson, StudentT) prefer :class:`GaussHermite` with ``deg=20``.

    Attributes:
        n: Number of Monte Carlo samples. Default ``64``.
        key: PRNG key. Default ``jax.random.PRNGKey(0)``.
    """

    n: int = eqx.field(static=True, default=64)
    key: jax.Array = eqx.field(default_factory=lambda: jax.random.PRNGKey(0))

    def integrate(
        self,
        fn: Callable[[Float[Array, " ..."]], Float[Array, " ..."]],
        mean: Float[Array, " ..."],
        var: Float[Array, " ..."],
    ) -> Float[Array, " ..."]:
        std = jnp.sqrt(var)
        eps = jax.random.normal(self.key, (self.n, *mean.shape), dtype=mean.dtype)
        samples = mean[None, ...] + std[None, ...] * eps
        vals = jax.vmap(fn)(samples)
        return jnp.mean(vals, axis=0)
