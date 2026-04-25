"""Axis-aligned rectangular domain for spatial point processes.

A spatial point process needs three things from its domain: dimension
``d``, total volume ``|D|``, and the ability to draw uniform candidate
locations. ``RectangularDomain`` packages those into an ``equinox``
PyTree so it threads through ``jit`` / ``vmap`` like any other operator
state. Polygon, sphere, and manifold domains are out of scope for v1 —
this is the same restriction the upstream snippet enforces.
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
