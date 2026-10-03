r"""Generic components from a user-supplied structure (P7)."""

from __future__ import annotations

import equinox as eqx
import gaussx as gx
import jax.numpy as jnp
import lineax as lx
import numpyro.distributions as dist
from jaxtyping import Array, ArrayLike, Float
from numpyro.distributions.transforms import Transform

from pyrox_lgm._components._base import (
    AbstractComponent,
    Constraint,
    default_transform,
    scale_operator,
)
from pyrox_lgm._priors import PCPrecision


class Generic(AbstractComponent):
    r"""A field with precision $\tau R$ for a user-supplied structure $R$.

    With ``null_space=None``, $R$ must be positive definite and the field is
    the proper GMRF $\mathcal N(0, (\tau R)^{-1})$. Otherwise $R$ is
    positive semidefinite with the given null space and the field is
    intrinsic, constrained to be orthogonal to it (R-INLA's ``generic0``
    with ``constr``).

    Pass $R$ as a gaussx / lineax operator: a `gaussx.SparseOperator` keeps
    the sparse-Cholesky path.

    Args:
        structure: $R$, symmetric, size ``n``.
        null_space: Basis of $\ker R$, ``(n, c)`` or ``(n,)``; ``None`` for a
            proper field.
        name: Site-name prefix.
        tau_prior: Prior on $\tau$; default ``PCPrecision(1, 0.01)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> import pyrox_lgm as lgm
        >>> R = lx.MatrixLinearOperator(
        ...     jnp.array([[2.0, -1.0], [-1.0, 2.0]]), lx.positive_semidefinite_tag
        ... )
        >>> comp = lgm.Generic(R, name="g")
        >>> comp.prior({"tau": jnp.asarray(3.0)}).precision.as_matrix().tolist()
        [[6.0, -3.0], [-3.0, 6.0]]
    """

    structure: lx.AbstractLinearOperator
    null_space: Float[Array, "n c"] | None
    name: str = eqx.field(static=True)
    tau_prior: dist.Distribution

    def __init__(
        self,
        structure: lx.AbstractLinearOperator,
        null_space: ArrayLike | None = None,
        *,
        name: str = "generic",
        tau_prior: dist.Distribution | None = None,
    ) -> None:
        if structure.in_size() != structure.out_size():
            raise ValueError("structure must be square")
        self.structure = structure
        if null_space is None:
            self.null_space = None
        else:
            V = jnp.asarray(null_space)
            V = V[:, None] if V.ndim == 1 else V
            if V.shape[0] != structure.in_size():
                raise ValueError(
                    f"null_space has {V.shape[0]} rows for a structure of "
                    f"size {structure.in_size()}"
                )
            self.null_space = V
        self.name = name
        self.tau_prior = PCPrecision(1.0, 0.01) if tau_prior is None else tau_prior

    @property
    def n_nodes(self) -> int:
        return self.structure.in_size()

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        return {"tau": (self.tau_prior, default_transform(self.tau_prior))}

    def prior(
        self,
        theta: dict[str, Array],
        *,
        constraint: Constraint = "hard",
        soft_constraint_scale: float = 1e-3,
    ) -> gx.GaussianMRF | gx.IntrinsicGMRF:
        loc = jnp.zeros(self.n_nodes)
        if self.null_space is None:
            return gx.GaussianMRF(loc, scale_operator(self.structure, theta["tau"]))
        return gx.IntrinsicGMRF(
            loc,
            theta["tau"],
            self.structure,
            self.null_space,
            constraint=constraint,
            soft_constraint_scale=soft_constraint_scale,
        )
