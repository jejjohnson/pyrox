"""Shared helpers for marked point-process mark likelihoods.

JAX's reverse-mode autodiff threads through **both** branches of
``jnp.where``, so evaluating ``d.log_prob(mark)`` on an out-of-support
padding mark (zero-padding vs. a positive-support Gamma/LogNormal law,
say) produces ``-inf`` values and NaN gradients even though the result
is masked out afterwards. Every marked variant (temporal, spatial,
spatiotemporal) must therefore substitute an in-support value at padding
positions *before* calling ``log_prob``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array


def safe_padding_mark(d: dist.Distribution, m_i: Array) -> Array:
    """Return a value inside ``d``'s support, dtype-matching ``m_i``.

    Uses ``d.support.feasible_like(m_i)`` when available — NumPyro's
    ``Constraint`` API ships this for every standard support
    (positive, unit interval, real, simplex, integer interval, ...) —
    and produces a value that is *strictly* inside the support and
    of the same dtype as the prototype, so it works for both
    continuous and discrete marks.

    Falls back to ``ones_like(m_i)`` when the constraint lacks
    ``feasible_like``. Custom user constraints that don't implement
    it must therefore have ``1`` (cast to the mark dtype) inside
    their support, or the caller must supply marks with safe padding
    already in place.
    """
    fl = getattr(d.support, "feasible_like", None)
    if fl is None:
        return jnp.ones_like(m_i)
    return fl(m_i)
