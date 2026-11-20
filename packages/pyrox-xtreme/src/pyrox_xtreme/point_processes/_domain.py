"""Axis-aligned rectangular domain and temporal interval for point processes.

A spatial point process needs three things from its domain: dimension
``d``, total volume ``|D|``, and the ability to draw uniform candidate
locations. ``RectangularDomain`` packages those into an ``equinox``
PyTree so it threads through ``jit`` / ``vmap`` like any other operator
state. Polygon, sphere, and manifold domains are out of scope for v1 —
this is the same restriction the upstream snippet enforces.

For spatiotemporal processes the time axis carries a different semantic
contract (it is causally ordered and supports time rescaling), so it
gets its own :class:`TemporalDomain` rather than reusing
:class:`RectangularDomain` with ``d = 1``. Existing temporal-only
processes still accept a bare ``t_max: float`` for backwards
compatibility — the new type is opt-in via the spatiotemporal API.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.typing import ArrayLike
from jaxtyping import Array, Float, PRNGKeyArray


class RectangularDomain(eqx.Module):
    """Axis-aligned rectangle ``[lo, hi] ⊂ ℝᵈ``.

    Both ``lo`` and ``hi`` are real PyTree leaves so the domain can
    be vmapped over (e.g. for batches with different bounding boxes)
    and so that, in principle, the bounds could be optimised through
    gradient descent.

    Args:
        lo: Lower corner ``(d,)``. Defaults to the origin when only
            ``hi`` is provided via :meth:`from_size`.
        hi: Upper corner ``(d,)``. Must be strictly above ``lo``
            elementwise.

    Notes:
        The convention is half-open ``[lo, hi)`` for sampling — the
        upper edge has measure zero so any choice is fine, but uniform
        draws on ``[lo, hi)`` are the natural pairing with ``lo + U·side``.
    """

    lo: Float[Array, ...]
    hi: Float[Array, ...]

    def __init__(self, lo: ArrayLike, hi: ArrayLike) -> None:
        lo_arr = jnp.asarray(lo)
        hi_arr = jnp.asarray(hi)
        # Validate eagerly at construction. Shape and ordering errors
        # are user-facing input bugs that should surface here rather
        # than producing silently negative volumes / out-of-box draws
        # downstream. Using ``bool(...)`` forces concretisation, which
        # is fine because RectangularDomain is invariably built from
        # static (non-traced) bounds; tracing through __init__ would
        # be a misuse.
        if lo_arr.shape != hi_arr.shape:
            raise ValueError(
                f"RectangularDomain: `lo` shape {lo_arr.shape} does not "
                f"match `hi` shape {hi_arr.shape}."
            )
        try:
            if not bool(jnp.all(hi_arr > lo_arr)):
                raise ValueError(
                    f"RectangularDomain requires `hi > lo` elementwise; "
                    f"got lo={lo_arr.tolist()}, hi={hi_arr.tolist()}."
                )
        except jax.errors.ConcretizationTypeError as exc:  # pragma: no cover
            # Building a domain inside a trace (e.g. inside ``jit``) is
            # not the supported path; skip the eager check rather than
            # erroring out of an otherwise valid jit-compiled function.
            del exc
        self.lo = lo_arr
        self.hi = hi_arr

    @classmethod
    def from_size(cls, size: ArrayLike) -> RectangularDomain:
        """Box anchored at the origin: ``lo = 0``, ``hi = size``.

        Mirrors the source-snippet convention where users pass a
        ``domain_size`` array directly. Always returns a 1-D ``hi``
        array even for scalar input.
        """
        size_arr = jnp.atleast_1d(jnp.asarray(size))
        return cls(jnp.zeros_like(size_arr), size_arr)

    @property
    def n_dims(self) -> int:
        """Spatial dimension ``d``.

        Static at PyTree-construction time so callers can use it as a
        Python ``int`` (e.g. for kernel-grid construction).
        """
        return int(self.hi.shape[-1])

    @property
    def side_lengths(self) -> Float[Array, ...]:
        """Per-axis box widths ``hi - lo`` with shape ``(d,)``."""
        return self.hi - self.lo

    def volume(self) -> Float[Array, ...]:
        """Lebesgue measure :math:`|D| = \\prod_i (hi_i - lo_i)`."""
        return jnp.prod(self.side_lengths, axis=-1)

    def contains(self, x: Float[Array, ...]) -> Array:
        """Boolean ``True`` where every coordinate lies in the half-open box."""
        return jnp.all((x >= self.lo) & (x < self.hi), axis=-1)

    def sample_uniform(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...] = (),
    ) -> Float[Array, ...]:
        """Draw uniform points in the box.

        The output shape is ``shape + (d,)`` — ``shape`` is the leading
        sample/batch axes and ``d`` is the spatial dimension.
        """
        u = random.uniform(key, shape=(*tuple(shape), self.n_dims))
        return self.lo + u * self.side_lengths


class TemporalDomain(eqx.Module):
    """Half-open temporal interval ``[t0, t1) ⊂ ℝ``.

    Parallel to :class:`RectangularDomain` but specialised to one
    causally-ordered axis. Used by spatiotemporal point processes so the
    time domain stays explicit (``[t0, t1)`` instead of an implicit
    ``[0, T]``) and so callers can shift the origin without re-deriving
    quadrature offsets.

    Args:
        t0: Inclusive start time.
        t1: Exclusive end time. Must satisfy ``t1 > t0``.
    """

    t0: Float[Array, ...]
    t1: Float[Array, ...]

    def __init__(self, t0: ArrayLike, t1: ArrayLike) -> None:
        t0_arr = jnp.asarray(t0)
        t1_arr = jnp.asarray(t1)
        if t0_arr.shape != t1_arr.shape:
            raise ValueError(
                f"TemporalDomain: `t0` shape {t0_arr.shape} does not match "
                f"`t1` shape {t1_arr.shape}."
            )
        try:
            if not bool(jnp.all(t1_arr > t0_arr)):
                raise ValueError(
                    f"TemporalDomain requires `t1 > t0`; got "
                    f"t0={t0_arr.tolist()}, t1={t1_arr.tolist()}."
                )
        except jax.errors.ConcretizationTypeError as exc:  # pragma: no cover
            del exc
        self.t0 = t0_arr
        self.t1 = t1_arr

    @classmethod
    def from_duration(cls, duration: ArrayLike) -> TemporalDomain:
        """Interval anchored at zero: ``t0 = 0``, ``t1 = duration``.

        Mirrors the source-snippet convention where users pass an
        ``observation_window`` scalar directly.
        """
        dur = jnp.asarray(duration)
        return cls(jnp.zeros_like(dur), dur)

    @property
    def duration(self) -> Float[Array, ...]:
        """Length of the interval ``t1 - t0``."""
        return self.t1 - self.t0

    def volume(self) -> Float[Array, ...]:
        """Lebesgue measure of the interval — alias for :attr:`duration`.

        Provided for API symmetry with :class:`RectangularDomain` so
        joint-domain code can call ``spatial.volume() * temporal.volume()``
        without special-casing the temporal axis.
        """
        return self.duration

    def contains(self, t: Float[Array, ...]) -> Array:
        """Boolean ``True`` where ``t ∈ [t0, t1)``."""
        return (t >= self.t0) & (t < self.t1)

    def sample_uniform(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...] = (),
    ) -> Float[Array, ...]:
        """Draw uniform times in the interval.

        Output shape is exactly ``shape``; no trailing dimension axis,
        because time is scalar.
        """
        u = random.uniform(key, shape=tuple(shape))
        return self.t0 + u * self.duration
