r"""Self-exciting spatiotemporal point process (ETAS-style Hawkes).

The conditional intensity given the history :math:`\mathcal{H}_t` is

.. math::
    \lambda^*(s, t) = \mu + \sum_{t_i < t} \alpha \,
                     k_t(t - t_i) \, k_s(s - s_i),

with separable exponential-Gaussian triggering kernel

.. math::
    k_t(\tau) = \beta e^{-\beta \tau}, \qquad
    k_s(\Delta s) = \frac{1}{(2\pi)^{d/2} \sigma^d}
                    \exp\!\left(-\frac{\|\Delta s\|^2}{2 \sigma^2}\right).

Both kernels integrate to one over their unbounded supports, so the
compensator on :math:`D \times [t_0, t_1)` admits the closed-form

.. math::
    \Lambda = \mu |D| (t_1 - t_0) + \alpha \sum_{i}
              \big(1 - e^{-\beta (t_1 - t_i)}\big)
              \, \kappa_i,

where :math:`\kappa_i = \int_D k_s(s - s_i)\, ds \in (0, 1]` measures
the fraction of the spatial kernel mass that falls inside ``D``. We
provide both the **infinite-domain approximation** :math:`\kappa_i = 1`
(fast, exact when :math:`\sigma` is small relative to the domain and
events are interior) and a **boundary-corrected** form using the
product of axis-wise normal-CDF differences (still O(n)).

Sub-criticality requires :math:`\alpha < 1` — the spatial and temporal
kernels each integrate to one over :math:`\mathbb{R}^d` and
:math:`(0, \infty)`, so the branching ratio collapses to :math:`\alpha`.

Following the temporal Hawkes module
(:mod:`xtremax.point_processes.primitives.hawkes`), the user-facing
log-prob takes padded ``(locations, times, mask)`` inputs and the
sampler returns the same triple plus an uncapped count.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import erf
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from xtremax.point_processes._domain import RectangularDomain, TemporalDomain


def _gaussian_kernel(
    delta: Float[Array, ...],
    sigma: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Isotropic Gaussian density :math:`k_s(\Delta s)` on :math:`\mathbb{R}^d`."""
    delta = jnp.asarray(delta)
    sigma = jnp.asarray(sigma)
    d = delta.shape[-1]
    sq = jnp.sum(delta * delta, axis=-1)
    log_norm = -0.5 * d * jnp.log(2.0 * jnp.pi) - d * jnp.log(sigma)
    return jnp.exp(log_norm - 0.5 * sq / (sigma * sigma))


def _gaussian_box_mass(
    centre: Float[Array, ...],
    spatial: RectangularDomain,
    sigma: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Mass of a centered isotropic Gaussian inside the rectangular domain.

    Uses the axis-wise factorisation
    :math:`\int_D k_s(s - c)\, ds = \prod_j \tfrac{1}{2}
    [\mathrm{erf}((b_j - c_j)/(\sigma\sqrt 2))
     - \mathrm{erf}((a_j - c_j)/(\sigma\sqrt 2))]`,
    which is exact for the isotropic Gaussian on an axis-aligned box.
    """
    sigma_sqrt2 = sigma * jnp.sqrt(2.0)
    upper = (spatial.hi - centre) / sigma_sqrt2
    lower = (spatial.lo - centre) / sigma_sqrt2
    per_axis = 0.5 * (erf(upper) - erf(lower))
    return jnp.prod(per_axis, axis=-1)


def stpp_hawkes_intensity(
    s: Float[Array, ...],
    t: Float[Array, ...],
    event_locations: Float[Array, ...],
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    mu: Float[Array, ...],
    alpha: Float[Array, ...],
    beta: Float[Array, ...],
    sigma: Float[Array, ...],
) -> Float[Array, ...]:
    r"""Conditional intensity :math:`\lambda^*(s, t)` of the ETAS Hawkes process.

    Args:
        s: ``(d,)`` query location (scalar event) or ``(N, d)`` batch.
        t: Scalar or ``(N,)`` query time.
        event_locations: ``(max_events, d)`` past event locations.
        event_times: ``(max_events,)`` past event times.
        mask: ``(max_events,)`` boolean event mask.
        mu: Baseline rate (positive scalar).
        alpha: Excitation amplitude (must satisfy ``alpha < 1`` for
            sub-criticality).
        beta: Temporal decay rate (positive).
        sigma: Spatial bandwidth (positive).

    Returns:
        Intensity at the query, with shape matching the broadcast of
        ``t`` and ``s``'s leading axes.
    """
    s = jnp.asarray(s)
    t_arr = jnp.asarray(t)

    if t_arr.ndim == 0:
        past = mask & (event_times < t_arr)
        # Clip ``dt`` to non-negative before the exponential. With
        # negative ``dt`` (future / padding events) ``exp(-β·dt)`` can
        # overflow to ``+inf``, and ``inf · 0`` from the masked-out
        # branch then poisons the result with ``nan`` even though the
        # term should contribute zero. The mask is still applied below
        # to zero out the future events; ``dt_pos`` only keeps
        # intermediates finite.
        dt_pos = jnp.clip(t_arr - event_times, 0.0, jnp.inf)
        delta = s[..., None, :] - event_locations
        spatial = _gaussian_kernel(delta, sigma)
        temporal = beta * jnp.exp(-beta * dt_pos)
        contribs = jnp.where(past, alpha * temporal * spatial, 0.0)
        return mu + jnp.sum(contribs, axis=-1)

    past = mask & (event_times < t_arr[..., None])
    dt_pos = jnp.clip(t_arr[..., None] - event_times, 0.0, jnp.inf)
    delta = s[..., None, :] - event_locations
    spatial = _gaussian_kernel(delta, sigma)
    temporal = beta * jnp.exp(-beta * dt_pos)
    contribs = jnp.where(past, alpha * temporal * spatial, 0.0)
    return mu + jnp.sum(contribs, axis=-1)


def stpp_hawkes_compensator(
    event_locations: Float[Array, ...],
    event_times: Float[Array, ...],
    mask: Bool[Array, ...],
    mu: Float[Array, ...],
    alpha: Float[Array, ...],
    beta: Float[Array, ...],
    sigma: Float[Array, ...],
    spatial: RectangularDomain,
    temporal: TemporalDomain,
    boundary_correction: bool = True,
) -> Float[Array, ...]:
    r"""Closed-form compensator :math:`\Lambda` of the ETAS Hawkes process.

    .. math::
        \Lambda = \mu |D| (t_1 - t_0) + \alpha \sum_i
                  (1 - e^{-\beta (t_1 - t_i)}) \kappa_i,

    where :math:`\kappa_i = \int_D k_s(s - s_i)\, ds` is exactly
    :math:`1` if ``boundary_correction=False`` (infinite-domain
    approximation, valid when :math:`\sigma` is small relative to the
    box and events are interior) or computed via axis-wise erf when
    ``boundary_correction=True``.
    """
    dt = jnp.clip(temporal.t1 - event_times, 0.0, jnp.inf)
    temporal_factor = 1.0 - jnp.exp(-beta * dt)

    if boundary_correction:
        spatial_factor = jax.vmap(_gaussian_box_mass, in_axes=(0, None, None))(
            event_locations, spatial, sigma
        )
    else:
        spatial_factor = jnp.ones_like(temporal_factor)

    excitation = alpha * jnp.sum(
        jnp.where(mask, temporal_factor * spatial_factor, 0.0), axis=-1
    )
    baseline = mu * spatial.volume() * temporal.duration
    return baseline + excitation


def stpp_hawkes_log_prob(
    locations: Float[Array, ...],
    times: Float[Array, ...],
    mask: Bool[Array, ...],
    mu: Float[Array, ...],
    alpha: Float[Array, ...],
    beta: Float[Array, ...],
    sigma: Float[Array, ...],
    spatial: RectangularDomain,
    temporal: TemporalDomain,
    boundary_correction: bool = True,
) -> Float[Array, ...]:
    r"""Log-likelihood :math:`\sum_i \log \lambda^*(s_i, t_i) - \Lambda`.

    Computed by O(n²) pairwise construction of the per-event
    intensities (matches the temporal :func:`general_hawkes_log_prob`
    pattern); for typical event counts under a few thousand this is
    competitive with the recursive Ozaki form and far simpler to
    extend to non-exponential temporal kernels in a follow-up.
    """
    locations = jnp.asarray(locations)
    times = jnp.asarray(times)

    dt = times[..., :, None] - times[..., None, :]
    delta = locations[..., :, None, :] - locations[..., None, :, :]
    spatial_vals = _gaussian_kernel(delta, sigma)
    temporal_vals = beta * jnp.exp(-beta * jnp.clip(dt, 0.0, jnp.inf))

    # Define causal precedence by *timestamp*, not buffer position.
    # Position-based masking (``idx_j < idx_i``) silently treats
    # whichever rows the caller wrote first as "earlier" — for an
    # unsorted buffer this turns future events into spurious parents,
    # and the ``clip(dt, 0, inf)`` then injects them at the maximum
    # temporal-kernel value. Timestamp comparison is correct
    # regardless of the row order chosen by the caller (Codex P2 in
    # PR #13 review).
    strictly_before = times[..., None, :] < times[..., :, None]
    pair_valid = strictly_before & mask[..., None, :]
    contribs = alpha * temporal_vals * spatial_vals
    per_event = jnp.sum(jnp.where(pair_valid, contribs, 0.0), axis=-1)
    lam_at_events = mu + per_event
    log_lam = jnp.log(jnp.clip(lam_at_events, 1e-30, jnp.inf))
    sum_log_lam = jnp.sum(jnp.where(mask, log_lam, 0.0), axis=-1)

    Lambda = stpp_hawkes_compensator(
        locations,
        times,
        mask,
        mu,
        alpha,
        beta,
        sigma,
        spatial,
        temporal,
        boundary_correction=boundary_correction,
    )
    return sum_log_lam - Lambda


def stpp_hawkes_lambda_max(
    n_events: Int[Array, ...],
    mu: Float[Array, ...],
    alpha: Float[Array, ...],
    beta: Float[Array, ...],
    sigma: Float[Array, ...],
    n_dims: int,
) -> Float[Array, ...]:
    r"""Adaptive upper bound :math:`\bar\lambda = \mu + N \alpha \beta k_s(0)`.

    The temporal kernel peaks at :math:`\beta` and the spatial kernel
    peaks at :math:`(2\pi)^{-d/2} / \sigma^d`. For a sub-critical
    process this bound is finite and tight just after the most recent
    event; it loosens between events but stays a true upper bound.
    """
    n = jnp.asarray(n_events).astype(jnp.asarray(mu).dtype)
    peak_spatial = (2.0 * jnp.pi) ** (-0.5 * n_dims) / (sigma**n_dims)
    return mu + n * alpha * beta * peak_spatial


def stpp_hawkes_sample(
    key: PRNGKeyArray,
    mu: Float[Array, ...],
    alpha: Float[Array, ...],
    beta: Float[Array, ...],
    sigma: Float[Array, ...],
    spatial: RectangularDomain,
    temporal: TemporalDomain,
    max_events: int,
) -> tuple[Float[Array, ...], Float[Array, ...], Bool[Array, ...], Int[Array, ...]]:
    r"""Ogata-style spatiotemporal thinning sample.

    Algorithm (per Daley & Vere-Jones, §7.3):

    1. Maintain a running history ``(locs, times, mask)``.
    2. At each step, draw a candidate inter-event time from
       ``Exp(λ̄)`` with adaptive bound :math:`\bar\lambda` from
       :func:`stpp_hawkes_lambda_max`.
    3. Draw a candidate location uniformly on ``D``.
    4. Accept with probability
       :math:`\lambda^*(s, t)/\bar\lambda`; otherwise discard and try
       again.

    Loops via :func:`jax.lax.scan` over a fixed budget of
    ``5 * max_events`` candidates so the function is jit-compatible.

    Args:
        key: PRNG key.
        mu: Baseline rate.
        alpha: Excitation amplitude (sub-critical: ``alpha < 1``).
        beta: Temporal decay.
        sigma: Spatial bandwidth.
        spatial: Spatial domain.
        temporal: Temporal interval.
        max_events: Static buffer cap.

    Returns:
        ``(locations, times, mask, n_events)`` with ``times`` sorted
        ascending. ``n_events`` is uncapped — if it exceeds
        ``max_events`` the buffer is truncated; if it exceeds the
        candidate budget the sampler stops early and the user should
        increase ``max_events``.
    """
    mu_arr = jnp.asarray(mu)
    alpha_arr = jnp.asarray(alpha)
    beta_arr = jnp.asarray(beta)
    sigma_arr = jnp.asarray(sigma)

    d = spatial.n_dims
    n_max = max_events
    candidate_budget = 5 * n_max

    init_locs = jnp.zeros((n_max, d), dtype=mu_arr.dtype)
    init_times = jnp.full((n_max,), temporal.t1, dtype=mu_arr.dtype)
    init_mask = jnp.zeros((n_max,), dtype=jnp.bool_)
    init_count = jnp.asarray(0, dtype=jnp.int32)
    init_t = jnp.asarray(temporal.t0, dtype=mu_arr.dtype)

    keys = random.split(key, candidate_budget)

    box_vol = spatial.volume()

    def step(carry, k_step):
        locs, times, mask, count, t_curr = carry
        k_dt, k_loc, k_acc, k_thin = random.split(k_step, 4)

        # Per-unit-volume peak intensity, valid as an upper bound on
        # ``λ*(s, t)`` for any ``s ∈ D`` from the current step onward.
        # Floor with ``eps`` so that a μ=0, count=0 history does not
        # divide by zero in either ``dt`` or the acceptance ratio
        # (Codex P1 in PR #13 review). The clamp matches the pattern in
        # :func:`thinning_sample`.
        eps = jnp.asarray(1e-30, dtype=mu_arr.dtype)
        lam_bar_per_vol = jnp.maximum(
            stpp_hawkes_lambda_max(count, mu_arr, alpha_arr, beta_arr, sigma_arr, d),
            eps,
        )
        # The inter-event waiting time over the whole domain is
        # exponential with rate ``λ̄·|D|``: integrating the per-volume
        # bound across ``D`` gives the rate at which space-time events
        # arrive anywhere in the box.
        lam_bar_total = jnp.maximum(lam_bar_per_vol * box_vol, eps)
        u = random.uniform(k_dt, minval=1e-30, maxval=1.0)
        dt = -jnp.log(u) / lam_bar_total
        t_next = t_curr + dt

        s_cand = spatial.sample_uniform(k_loc, shape=())
        # Intensity uses the *current* history (events before t_next).
        lam = stpp_hawkes_intensity(
            s_cand,
            t_next,
            locs,
            times,
            mask,
            mu_arr,
            alpha_arr,
            beta_arr,
            sigma_arr,
        )
        u_acc = random.uniform(k_acc, minval=1e-30, maxval=1.0)
        # Acceptance is against the per-volume bound — the candidate
        # location is uniform on ``D`` so the proposal already accounts
        # for the spatial integration in ``λ_bar_total``.
        accept_in_box = u_acc < (lam / lam_bar_per_vol)
        del k_thin

        in_window = t_next < temporal.t1
        room = count < n_max
        accept = accept_in_box & in_window & room

        # Clip the write index to ``[0, n_max - 1]``. Using ``n_max`` as
        # a sentinel for rejected candidates is undefined: in JAX the
        # default ``promise_in_bounds`` semantics make scatter/gather at
        # that index backend-dependent and can corrupt buffer state on
        # rejection-heavy runs (Copilot/Codex P1 in PR #13 review). The
        # ``accept`` flag in ``jnp.where`` already keeps rejected
        # candidates from changing the slot — the clip just keeps the
        # index in range for the no-op read-modify-write.
        safe_idx = jnp.clip(count, 0, n_max - 1)
        locs = locs.at[safe_idx].set(jnp.where(accept, s_cand, locs[safe_idx]))
        times = times.at[safe_idx].set(jnp.where(accept, t_next, times[safe_idx]))
        mask = mask.at[safe_idx].set(jnp.where(accept, True, mask[safe_idx]))
        count = count + accept.astype(jnp.int32)
        # Always advance t_curr by dt — even rejected candidates consume
        # the exponential clock per Ogata's argument.
        t_curr = jnp.minimum(t_next, temporal.t1)
        return (locs, times, mask, count, t_curr), None

    (locs, times, mask, count, _), _ = jax.lax.scan(
        step,
        (init_locs, init_times, init_mask, init_count, init_t),
        keys,
    )

    # Buffer is already filled in event-arrival order; events are
    # naturally time-sorted because the Ogata sampler walks ``t``
    # monotonically forward.
    locations = jnp.where(mask[:, None], locs, spatial.lo)
    times = jnp.where(mask, times, temporal.t1)
    return locations, times, mask, count
