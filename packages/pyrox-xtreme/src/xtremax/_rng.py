"""Small helpers for validating JAX PRNG keys used by Distribution.sample."""

from __future__ import annotations

import jax.dtypes
import jax.numpy as jnp


_BAD_KEY_MSG = (
    "`key` must be a JAX PRNG key. Pass either `jax.random.key(...)` "
    "or the legacy `jax.random.PRNGKey(...)`."
)


def check_prng_key(key) -> None:
    """Reject anything that is not a JAX PRNG key.

    Accepts both the modern typed keys (``jax.random.key(...)``) and the
    legacy ``uint32[..., 2]`` keys (``jax.random.PRNGKey(...)``). Raises
    ``TypeError`` with guidance otherwise — never a bare ``assert``, which
    is stripped under ``python -O`` and gives an unhelpful message.
    """
    if not hasattr(key, "dtype") or not hasattr(key, "shape"):
        raise TypeError(_BAD_KEY_MSG)
    dtype = key.dtype
    # Typed keys carry a special PRNG dtype; checking `not integer` would
    # incorrectly accept any non-integer array (e.g. float32) as a key.
    is_typed = jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key)
    is_legacy = (
        jnp.issubdtype(dtype, jnp.integer)
        and dtype == jnp.uint32
        and key.shape[-1:] == (2,)
    )
    if not (is_typed or is_legacy):
        raise TypeError(_BAD_KEY_MSG)
