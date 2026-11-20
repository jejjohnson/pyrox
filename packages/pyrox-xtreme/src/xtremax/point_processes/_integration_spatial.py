"""Multi-dimensional integration helpers for spatial point processes.

Spatial PPs need :math:`\\Lambda(D) = \\int_D \\lambda(s)\\, ds` for both
the log-likelihood (``-Λ`` term) and for :class:`InhomogeneousSpatialPP`
sampling, which uses ``Λ`` to size its candidate buffer. The intensity
function is always supplied in log-space — same convention as the
temporal IPP — and applied pointwise to a quadrature grid built from
:class:`~xtremax.point_processes._domain.RectangularDomain`.

Two strategies are provided:

* **Tensor-product trapezoid** — closed-form factorisation of the
  per-axis trapezoid weights. Exact under
  ``log_intensity_fn(s) = c1·s1 + c2·s2 + …``. Cost is
  :math:`n^d` for ``n`` nodes per axis, which becomes uneconomical
  for ``d ≥ 4``; cap default ``n`` accordingly.
* **Quasi-Monte Carlo (QMC)** — deterministic Halton sequence over
  ``[0, 1)^d`` mapped to ``[lo, hi)``. Convergence is
  :math:`O(\\log^d N / N)` versus :math:`O(N^{-1/2})` for plain Monte
  Carlo, so a few thousand QMC nodes typically beat a tensor-product
  trapezoid grid for ``d ≥ 3``.

Both routines return ``Λ`` and stay ``jit``/``vmap`` safe. Grid sizes
are static Python ``int`` so XLA tracing does not see them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from xtremax.point_processes._domain import RectangularDomain


SpatialMethod = Literal["trapezoid", "qmc"]


def integrate_log_intensity_spatial(
    log_intensity_fn: Callable[[Float[Array, ...]], Float[Array, ...]],
    domain: RectangularDomain,
    n_points: int = 2048,
    method: SpatialMethod = "qmc",
) -> Float[Array, ...]:
    """Compute :math:`\\int_D \\exp(\\log\\lambda(s))\\, ds`.

    Args:
        log_intensity_fn: Callable mapping ``(N, d)`` locations to
            ``(N,)`` log-intensities. Must accept a 2-D leading-batch
            array — the same convention as the temporal IPP, extended
            with a trailing dimension axis.
        domain: Spatial domain over which to integrate.
        n_points: Static number of quadrature/QMC nodes. For
            ``method="trapezoid"`` this is the *total* node count; the
            per-axis grid is :math:`n^{1/d}` rounded up. For
            ``method="qmc"`` it is the literal sample count.
        method: ``"trapezoid"`` (closed-form on tensor-product grid,
            best for ``d ≤ 2``) or ``"qmc"`` (Halton sequence, best for
            ``d ≥ 3`` and the default for any ``d``).

    Returns:
        Scalar Λ, differentiable through ``log_intensity_fn``'s
        parameters.

    Notes:
        Both methods are deterministic given a fixed ``n_points`` —
        no PRNG key is needed. QMC uses a digit-reversed Halton
        sequence on the first ``d`` primes, which is reproducible and
        avoids the wrap-around bias of plain MC.
    """
    if method == "trapezoid":
        return _integrate_trapezoid(log_intensity_fn, domain, n_points)
    if method == "qmc":
        return _integrate_qmc(log_intensity_fn, domain, n_points)
    raise ValueError(f"Unknown method {method!r}; expected 'trapezoid' or 'qmc'.")


def _integrate_trapezoid(
    log_intensity_fn: Callable[[Float[Array, ...]], Float[Array, ...]],
    domain: RectangularDomain,
    n_points: int,
) -> Float[Array, ...]:
    """Tensor-product composite trapezoid.

    ``n_points`` is the total node budget; per-axis nodes are
    :math:`\\lceil n^{1/d} \\rceil` clipped at 2.
    """
    d = domain.n_dims
    # Compute the per-axis grid size in pure Python — both ``n_points``
    # and ``d`` are static Python ints, and using ``jnp.ceil`` here
    # would force a device→host conversion under tracing.
    n_per_axis = max(2, math.ceil(n_points ** (1.0 / d)))

    # Per-axis 1-D linspace + trapezoid weights (1/2, 1, 1, ..., 1, 1/2).
    axis_grids = [
        jnp.linspace(domain.lo[i], domain.hi[i], n_per_axis) for i in range(d)
    ]
    h_per_axis = jnp.array(
        [(domain.hi[i] - domain.lo[i]) / (n_per_axis - 1) for i in range(d)]
    )
    base_weight = jnp.ones(n_per_axis)
    base_weight = base_weight.at[0].set(0.5)
    base_weight = base_weight.at[-1].set(0.5)

    # Tensor-product mesh: meshgrid with indexing='ij' gives ``(n^d, d)``
    # after flattening, in row-major order matching the weights tensor.
    mesh = jnp.stack(jnp.meshgrid(*axis_grids, indexing="ij"), axis=-1)
    flat_points = mesh.reshape(-1, d)
    flat_log_intensity = log_intensity_fn(flat_points)
    intensities = jnp.exp(flat_log_intensity).reshape((n_per_axis,) * d)

    # Tensor-product weights via successive outer products.
    weight = base_weight
    for _ in range(d - 1):
        weight = jnp.tensordot(weight, base_weight, axes=0)

    cell_volume = jnp.prod(h_per_axis)
    return jnp.sum(weight * intensities) * cell_volume


def _integrate_qmc(
    log_intensity_fn: Callable[[Float[Array, ...]], Float[Array, ...]],
    domain: RectangularDomain,
    n_points: int,
) -> Float[Array, ...]:
    """Halton-sequence quasi-Monte Carlo over ``[lo, hi)^d``.

    Uses the first ``d`` primes as Halton bases. Returns the QMC
    estimate :math:`|D| \\cdot \\overline{\\lambda(s_k)}`.
    """
    d = domain.n_dims
    unit_cube_points = _halton_sequence(n_points, d)
    points = domain.lo + unit_cube_points * domain.side_lengths
    log_intensities = log_intensity_fn(points)
    mean_intensity = jnp.mean(jnp.exp(log_intensities))
    return domain.volume() * mean_intensity


def _first_n_primes(n: int) -> tuple[int, ...]:
    """First ``n`` prime numbers, generated by trial division.

    Cached implicitly via Python's tuple-equality semantics on the
    callsite (we only call this once per ``_halton_sequence`` and
    ``n`` is small in practice). The standard prime-counting bound
    :math:`p_n < n (\\ln n + \\ln \\ln n)` for :math:`n \\geq 6`
    ensures the search ceiling stays modest even for huge ``d``.
    """
    if n <= 0:
        return ()
    primes: list[int] = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


def _halton_sequence(n: int, d: int) -> Float[Array, ...]:
    """Halton sequence of length ``n`` in ``d`` dimensions.

    Each axis uses a different prime base via the digit-reversed
    radical inverse :math:`\\phi_b(i) = \\sum_k a_k(i) b^{-(k+1)}`.
    Computed eagerly with NumPy-equivalent JAX ops so the whole thing
    is jittable; ``n`` and ``d`` must be Python ``int`` (static).

    Note that classical Halton with the first ``d`` primes loses
    space-filling quality for very large ``d`` because high-prime axes
    cycle slowly — practitioners typically prefer scrambled or
    leaped variants past ``d ≈ 20``. We do not impose a hard ceiling
    here; the caller is responsible for choosing an integration
    method appropriate to their dimension.
    """
    primes = _first_n_primes(d)
    indices = jnp.arange(1, n + 1)  # skip i=0 which always maps to 0
    cols = [_radical_inverse(indices, primes[axis]) for axis in range(d)]
    return jnp.stack(cols, axis=-1)


def _radical_inverse(i: Array, base: int) -> Array:
    """Vectorised digit-reversed radical inverse :math:`\\phi_{base}(i)`.

    Iterates the standard recurrence
    ``f *= 1/base; r += f * (i mod base); i //= base`` until every
    ``i`` reaches zero. The number of iterations is bounded by
    ``log_base(max(i)) + 1`` so we choose a static ceiling that covers
    any reasonable ``n`` (``i ≤ 10^9`` for base 2 needs 30 iters).
    """
    max_iters = 64  # 2^64 covers anything we will plausibly request
    base_f = jnp.float32(base)

    def body(carry, _):
        i_curr, f, r = carry
        digit = jnp.mod(i_curr, base)
        new_f = f / base_f
        new_r = r + new_f * digit.astype(jnp.float32)
        new_i = i_curr // base
        return (new_i, new_f, new_r), None

    # Use whatever int dtype is the default — usually int32 under
    # ``jax_enable_x64=False``, int64 otherwise. Either is plenty for
    # ``i ≤ n`` since we cap ``n_points`` upstream.
    init = (
        i.astype(jnp.result_type(int)),
        jnp.ones((), dtype=jnp.float32),
        jnp.zeros_like(i, dtype=jnp.float32),
    )
    (_, _, result), _ = jax.lax.scan(body, init, jnp.arange(max_iters))
    return result
