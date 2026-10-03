"""NumPyro faces of LGM components.

`sample_component` is what `AbstractComponent.sample` runs: it lets any
component sit inside an ordinary NumPyro model and be sampled by NUTS, with or
without ``inla()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro
from jaxtyping import Array, ArrayLike, Int


if TYPE_CHECKING:
    from pyrox_lgm._components._base import AbstractComponent


def sample_component(
    component: AbstractComponent,
    index: Int[ArrayLike, " n_obs"] | None = None,
) -> Array:
    """Sample a component's hyperparameters and field inside a NumPyro model.

    Each hyperparameter ``k`` is the site ``f"{name}_{k}"`` with its prior
    from `theta_spec`; the field is the site ``name``, drawn from
    `prior(theta, constraint="soft")` (a tight Gaussian on the null space of
    an intrinsic field, since NUTS cannot sample a hard constraint).

    Args:
        component: The component.
        index: Node of each observation; ``None`` returns every addressable
            node (padding nodes dropped).

    Returns:
        The field at ``index``.

    Examples:
        >>> import jax
        >>> from numpyro import handlers
        >>> import pyrox_lgm as lgm
        >>> rw1 = lgm.RW1(10, name="trend")
        >>> tr = handlers.trace(handlers.seed(rw1.sample, 0)).get_trace()
        >>> list(tr)
        ['trend_tau', 'trend']
        >>> tr["trend"]["value"].shape
        (10,)
    """
    theta = {
        k: jnp.asarray(numpyro.sample(f"{component.name}_{k}", prior))
        for k, (prior, _) in component.theta_spec().items()
    }
    gmrf = component.prior(theta, constraint="soft")
    x = jnp.asarray(numpyro.sample(component.name, gmrf))
    if index is None:
        return x[: component.n_index]
    return x[jnp.asarray(index)]
