"""Support-constraint dispatch shared by the EVT distribution classes.

NumPyro's inference machinery unconstrains latent sites through
``biject_to(dist.support)``. ``constraints.interval(lo, hi)`` maps to a
``Sigmoid ∘ Affine(scale=hi - lo)`` bijector, which returns ``inf``/``nan``
the moment a bound is infinite — so a support built from ``interval`` with
``±inf`` bounds breaks every model that uses the distribution as a latent
site, even though the membership check itself works. The classes therefore
dispatch on the *statically known* sign of the concentration and return a
constraint whose registered bijector genuinely handles the (half-)unbounded
geometry (``greater_than`` / ``less_than`` / ``real`` / ``nonnegative``),
falling back to an unbounded constraint when the sign cannot be known at
trace time.
"""

from __future__ import annotations

import jax
import numpy as np
from jaxtyping import ArrayLike


def concrete_sign(concentration: ArrayLike) -> int | None:
    """Return the statically known sign of ``concentration``, else ``None``.

    Returns ``1`` / ``-1`` / ``0`` when every element of a *concrete*
    (non-traced) concentration is positive / negative / zero. Returns
    ``None`` for traced values (inside ``jit`` / inference, where the sign
    cannot be known at trace time) and for mixed-sign batches (where no
    single constraint geometry fits every element).
    """
    if isinstance(concentration, jax.core.Tracer):
        return None
    arr = np.asarray(concentration)
    if not np.all(np.isfinite(arr)):
        return None
    if np.all(arr > 0):
        return 1
    if np.all(arr < 0):
        return -1
    if np.all(arr == 0):
        return 0
    return None
