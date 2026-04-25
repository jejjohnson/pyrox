"""NumPyro ``Distribution`` wrapper for the spatiotemporal Hawkes process."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Float, PRNGKeyArray
from numpyro.distributions import constraints

from xtremax._rng import check_prng_key
from xtremax.point_processes._domain import RectangularDomain, TemporalDomain
from xtremax.point_processes.operators.hawkes_spatiotemporal import (
    SpatioTemporalHawkes as _StHawkesOp,
)


class SpatioTemporalHawkes(dist.Distribution):
    """NumPyro wrapper for a separable exp-Gaussian spatiotemporal Hawkes process.

    Args:
        mu: Baseline rate (scalar > 0).
        alpha: Excitation amplitude (scalar; ``alpha < 1`` sub-critical).
        beta: Temporal decay (scalar > 0).
        sigma: Spatial bandwidth (scalar > 0).
        spatial: Spatial domain.
        temporal: Temporal interval.
        max_events: Static buffer for ``sample``.
        boundary_correction: Whether the compensator uses the
            axis-wise erf-CDF for in-domain spatial-kernel mass
            (default ``True``).
        validate_args: Forwarded to ``numpyro``.
    """

    arg_constraints = {
        "mu": constraints.positive,
        "alpha": constraints.positive,
        "beta": constraints.positive,
        "sigma": constraints.positive,
    }
    support = constraints.dependent
    reparametrized_params: list[str] = []

    def __init__(
        self,
        mu: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        sigma: ArrayLike,
        spatial: RectangularDomain,
        temporal: TemporalDomain,
        max_events: int = 512,
        boundary_correction: bool = True,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.mu = jnp.asarray(mu)
        self.alpha = jnp.asarray(alpha)
        self.beta = jnp.asarray(beta)
        self.sigma = jnp.asarray(sigma)
        for name, val in (
            ("mu", self.mu),
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("sigma", self.sigma),
        ):
            if val.shape != ():
                raise ValueError(
                    f"SpatioTemporalHawkes distribution requires scalar `{name}`; "
                    f"got shape {val.shape}. Vmap over parameters at the "
                    "operator layer for batched draws."
                )
        self.spatial = spatial
        self.temporal = temporal
        self._max_events = int(max_events)
        self._op = _StHawkesOp(
            mu=self.mu,
            alpha=self.alpha,
            beta=self.beta,
            sigma=self.sigma,
            spatial=spatial,
            temporal=temporal,
            boundary_correction=boundary_correction,
        )
        super().__init__(batch_shape=(), validate_args=validate_args)

    @property
    def max_events(self) -> int:
        return self._max_events

    @property
    def n_dims(self) -> int:
        return self.spatial.n_dims

    def sample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[Float[Array, ...], Float[Array, ...], Bool[Array, ...]]:
        """Return ``(locations, times, mask)``."""
        check_prng_key(key)
        if sample_shape != ():
            raise NotImplementedError(
                "SpatioTemporalHawkes supports sample_shape=() only; "
                "vmap over PRNG keys at the operator layer for batched draws."
            )
        locations, times, mask, _ = self._op.sample(key, self._max_events)
        return locations, times, mask

    def log_prob(
        self,
        value: tuple[Float[Array, ...], Float[Array, ...], Bool[Array, ...]],
    ) -> Float[Array, ...]:
        """Log-likelihood :math:`\\sum_i \\log \\lambda^*(s_i, t_i) - \\Lambda`."""
        locations, times, mask = value
        return self._op.log_prob(locations, times, mask)

    def _validate_sample(self, value) -> None:
        return None
