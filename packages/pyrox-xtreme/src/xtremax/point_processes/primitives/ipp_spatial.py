"""Pure-JAX primitives for the inhomogeneous spatial Poisson process.

For a domain :math:`D \\subset \\mathbb{R}^d` and intensity
:math:`\\lambda(s) \\geq 0` the integrated intensity is
:math:`\\Lambda(D) = \\int_D \\lambda(s)\\, ds`, the count is
:math:`N \\sim \\mathrm{Poisson}(\\Lambda)`, and the joint
log-likelihood factorises as

.. math::
    \\log L = \\sum_{i=1}^n \\log \\lambda(s_i) - \\Lambda(D).

The intensity is provided in log-space (``log_intensity_fn``) — the
same convention as the temporal IPP. Sampling uses Lewis–Shedler
thinning with an upper bound :math:`\\lambda_{\\max}` over the box.
The default conservative bound :math:`2 \\Lambda / |D|` is fine when
the intensity is roughly unimodal; for sharply-peaked surfaces pass a
tighter bound.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._domain import RectangularDomain


def ipp_spatial_log_prob(
    locations: Float[Array, ...],
    mask: Bool[Array, ...],
    log_intensity_fn: Callable[[Array], Array],
    integrated_intensity: Float[Array, ...],
) -> Float[Array, ...]:
    """Log-likelihood of an observed inhomogeneous spatial pattern.

    Args:
        locations: Padded locations ``(..., max_events, d)``. Padding
            rows are evaluated by ``log_intensity_fn`` for shape
            stability and masked out of the sum — choose padding
            inside the domain so ``log_intensity_fn`` does not raise.
        mask: Event mask shape ``(..., max_events)``.
        log_intensity_fn: Callable mapping ``(..., n, d)`` locations to
            ``(..., n)`` log-intensities.
        integrated_intensity: ``Λ(D)`` (computed by quadrature or
            analytically by the caller).

    Returns:
        Log-likelihood with shape ``mask.shape[:-1]``.
    """
    log_intensities = log_intensity_fn(locations)
    masked = jnp.where(mask, log_intensities, 0.0)
    return jnp.sum(masked, axis=-1) - jnp.asarray(integrated_intensity)


def ipp_spatial_sample_thinning(
    key: PRNGKeyArray,
    log_intensity_fn: Callable[[Array], Array],
    domain: RectangularDomain,
    lambda_max: Float[Array, ...],
    max_candidates: int,
) -> tuple[Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
    """Lewis–Shedler thinning sampler with a static buffer.

    Algorithm:

    1. Generate ``n`` candidates from a homogeneous Poisson process at
       rate ``λ_max`` over ``D`` (uniform on the box).
    2. Accept candidate ``s`` with probability
       :math:`\\lambda(s) / \\lambda_{\\max}`.

    Args:
        key: JAX PRNG key.
        log_intensity_fn: Log-intensity callable.
        domain: Spatial domain.
        lambda_max: Upper bound on :math:`\\lambda(s)` over ``D``.
            Tighter bounds give higher acceptance.
        max_candidates: Static cap on the candidate pool — choose
            comfortably above ``λ_max · |D|``.

    Returns:
        Tuple ``(locations, accepted_mask, n_candidates_uncapped)``:

        * ``locations`` shape ``(max_candidates, d)`` — sorted by the
          first coordinate to give a stable, deterministic order.
        * ``accepted_mask`` shape ``(max_candidates,)`` — ``True`` at
          positions that are real candidates *and* accepted by
          thinning.
        * ``n_candidates_uncapped`` — Poisson draw before capping;
          useful for diagnosing buffer over-runs.
    """
    key_n, key_locs, key_thin = random.split(key, 3)

    lambda_max = jnp.asarray(lambda_max)
    expected = lambda_max * domain.volume()
    n_uncapped = random.poisson(key_n, expected)

    ranks = jnp.arange(max_candidates)
    candidate_mask = ranks < n_uncapped
    raw = domain.sample_uniform(key_locs, shape=(max_candidates,))

    # Sort by first coordinate so the buffer order is deterministic and
    # padding rows trail the real candidates (push padding to +inf).
    primary_for_sort = jnp.where(candidate_mask, raw[:, 0], jnp.inf)
    order = jnp.argsort(primary_for_sort)
    sorted_locs = raw[order]
    sorted_mask = candidate_mask[order]

    # Replace padding rows with ``domain.lo`` before evaluating the
    # log-intensity so the call stays inside the support.
    safe_locs = jnp.where(sorted_mask[:, None], sorted_locs, domain.lo)
    log_intensities = log_intensity_fn(safe_locs)

    log_accept = log_intensities - jnp.log(lambda_max)
    u = random.uniform(key_thin, shape=(max_candidates,))
    thinning_accept = jnp.log(u) < log_accept
    accepted_mask = sorted_mask & thinning_accept

    locations = safe_locs
    return locations, accepted_mask, n_uncapped


def ipp_spatial_intensity(
    locations: Float[Array, ...],
    log_intensity_fn: Callable[[Array], Array],
) -> Float[Array, ...]:
    """Pointwise intensity ``λ(s) = exp(log_intensity_fn(s))``."""
    return jnp.exp(log_intensity_fn(jnp.asarray(locations)))


def ipp_spatial_predict_count(
    log_intensity_fn: Callable[[Array], Array],
    subdomain: RectangularDomain,
    n_integration_points: int = 2048,
    method: str = "qmc",
) -> Float[Array, ...]:
    """Expected count over a sub-domain ``A``.

    :math:`\\mathbb{E}[N(A)] = \\int_A \\lambda(s) ds` — same quadrature
    machinery as the :math:`\\Lambda(D)` calculation.
    """
    # Local import to avoid a top-level cycle: the integration helper
    # itself depends on RectangularDomain, which we already imported.
    from xtremax.point_processes._integration_spatial import (
        integrate_log_intensity_spatial,
    )

    return integrate_log_intensity_spatial(
        log_intensity_fn,
        subdomain,
        n_points=n_integration_points,
        method=method,  # type: ignore[arg-type]
    )
