"""Per-parameter-group optimizer helper.

Thin wrapper over ``optax.multi_transform`` — see
[`param_group_optimizer`][pyrox.inference.param_group_optimizer].
``optax`` is required (``pip install pyrox[optax]``), following the same
lazy-import convention as the ensemble primitives.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax

from pyrox.inference._ensemble import _require_optax


if TYPE_CHECKING:  # pragma: no cover - typing only
    import optax


def param_group_optimizer(
    groups: dict[str, optax.GradientTransformation],
    label_fn: Callable[[tuple, Any], str],
) -> optax.GradientTransformation:
    """Apply a different optimizer to each labelled parameter group.

    Models that mix a large block of free latent parameters with a few kernel
    hyperparameters usually want different step sizes for each: the latents
    sit on a well-conditioned objective and tolerate a large step, while
    lengthscales and noise live on a log scale and destabilize under one. A
    10x ratio is a common starting point.

    A ``label_fn`` returning a label that is not a key of ``groups`` fails
    loudly at ``init`` time — ``optax.multi_transform`` raises a
    ``ValueError`` naming the offending labels.

    !!! warning "Label by path, never by leaf value"
        ``optax.multi_transform`` evaluates a callable ``param_labels`` on
        the **parameter** tree at ``init`` and on the **update** tree at
        every ``update``. A ``label_fn`` that inspects leaf *values* (a
        sign test, say) can therefore assign one group at init and another
        afterwards, silently applying the wrong transform or tripping a
        masked-state structure error. Depend only on ``path`` and on
        update-invariant leaf metadata such as ``shape`` / ``dtype``, which
        are identical for parameters and their updates.

    Args:
        groups: Maps a group label to the optimizer for that group. Every
            label returned by ``label_fn`` must be a key here.
        label_fn: Called as ``label_fn(path, leaf)`` for each parameter,
            where ``path`` is the ``jax.tree_util`` key path. Returns the
            group label for that parameter. Must be a function of ``path``
            (or update-invariant leaf metadata) only — see the warning
            above.

    Returns:
        A single ``optax.GradientTransformation`` suitable for
        `pyrox.inference.EnsembleMAP` or ``numpyro.infer.SVI``.

    Examples:
        >>> import optax
        >>> def by_name(path, _):
        ...     return "latents" if "Z_T" in str(path) else "globals"
        >>> opt = param_group_optimizer(
        ...     {"latents": optax.adam(1e-2), "globals": optax.adam(1e-3)},
        ...     by_name,
        ... )
    """
    optax_mod = _require_optax()
    return optax_mod.multi_transform(
        groups,
        param_labels=lambda tree: jax.tree_util.tree_map_with_path(label_fn, tree),
    )
