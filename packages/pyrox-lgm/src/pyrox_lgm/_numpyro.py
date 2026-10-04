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
    *,
    soft_constraint_scale: float = 1e-2,
) -> Array:
    r"""Sample a component's hyperparameters and field inside a NumPyro model.

    Each hyperparameter ``k`` is the site ``f"{name}_{k}"`` with its prior
    from `theta_spec`; the field is the site ``name``, drawn from
    `prior(theta, constraint="soft")`: a tight Gaussian
    $V^\top x \sim \mathcal N(0, s^2 I)$ on the null space of an intrinsic
    field, since NUTS cannot sample a hard constraint.

    The default $s = 10^{-2}$ is looser than gaussx's $10^{-3}$: the
    constraint is a narrow direction NUTS's diagonal mass matrix cannot
    adapt to, so its width sets the step size. On a 12 x 12 BYM2 Poisson
    model, $s = 10^{-3}$ hits the maximum tree depth (1023 leapfrog steps per
    iteration), $10^{-2}$ takes 255 and $10^{-1}$ 30, with the same posterior.
    For a connected field on $n$ nodes, $s = 10^{-2}$ pins the mean to a
    standard deviation of $10^{-2}/\sqrt n$, about Stan's BYM2 soft
    constraint (0.001 on the mean) at $n = 100$.

    Args:
        component: The component.
        index: Node of each observation; ``None`` returns every addressable
            node (padding nodes dropped).
        soft_constraint_scale: $s$ above.

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
    gmrf = component.prior(
        theta, constraint="soft", soft_constraint_scale=soft_constraint_scale
    )
    x = jnp.asarray(numpyro.sample(component.name, gmrf))
    if index is None:
        return x[: component.n_index]
    return x[jnp.asarray(index)]
