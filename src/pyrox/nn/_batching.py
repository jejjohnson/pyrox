"""Flatten-vmap-restore helper shared by the geonnax-core wrapper layers.

geonnax cores are single-example ``(D,) -> out`` callables; pyrox layers
document support for arbitrary leading batch dims ``(*batch, D)``. Every
wrapper used to hand-roll the same flatten / ``jax.vmap`` / restore dance —
this module owns it once.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import einx
import jax
from jaxtyping import Array, Float


def vmap_over_flat_batch(
    fn: Callable[[Float[Array, " D"]], Any],
    x: Float[Array, "*batch D"],
) -> Any:
    """Apply a single-example callable over arbitrary leading batch dims.

    Flattens ``x`` from ``(*batch, D)`` to ``(B, D)``, maps ``fn`` with
    `jax.vmap`, and restores ``(*batch, ...)`` on every output leaf,
    so pytree outputs such as ``(mean, var)`` are restored leaf-wise.
    Unbatched ``(D,)`` inputs round-trip to unbatched outputs.

    Args:
        fn: Single-example callable ``(D,) -> leaf | pytree of leaves``;
            each output leaf may have any trailing shape.
        x: Inputs with zero or more leading batch axes.

    Returns:
        ``fn`` mapped over the batch, with the original batch shape in
        place of the flattened axis on every leaf.
    """
    batch_shape = x.shape[:-1]
    flat = einx.id("b... d -> (b...) d", x)
    out = jax.vmap(fn)(flat)
    return jax.tree.map(
        lambda leaf: einx.id("(b...) k... -> b... k...", leaf, b=batch_shape),
        out,
    )
