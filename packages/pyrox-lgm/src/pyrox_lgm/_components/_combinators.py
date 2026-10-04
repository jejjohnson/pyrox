r"""Combinators: separable Kronecker products and replicates of components (P7).

- `Kronecker` (R-INLA's ``group``): $Q = Q_\text{main} \otimes Q_\text{group}$,
  e.g. a spatial field correlated over time. Node $(i, j)$ of the product
  sits at flat index ``i * group.n_nodes + j``.
- `Replicate` (R-INLA's ``replicate``): independent copies sharing
  hyperparameters, $Q = I_r \otimes Q$; copy $r$, node $i$ at
  ``r * component.n_nodes + i``.

At most one factor may be intrinsic; its null space is lifted to the
product as $V \otimes I$ (or $I \otimes V$), one constraint per node of the
other factor.
"""

from __future__ import annotations

import equinox as eqx
import gaussx as gx
import jax.numpy as jnp
import lineax as lx
import numpyro.distributions as dist
from jaxtyping import Array, Float
from numpyro.distributions.transforms import Transform

from pyrox_lgm._components._base import AbstractComponent, Constraint


def _parts(
    gmrf: gx.GaussianMRF | gx.IntrinsicGMRF,
) -> tuple[lx.AbstractLinearOperator, Float[Array, "n c"] | None, Array]:
    """(operator, null space or None, scale) with precision = scale * operator."""
    if isinstance(gmrf, gx.BYM2GMRF):
        raise NotImplementedError("BYM2 cannot be a Kronecker / Replicate factor")
    if isinstance(gmrf, gx.IntrinsicGMRF):
        return gmrf.structure, gmrf.null_space, gmrf.precision_scale
    return gmrf.precision, None, jnp.asarray(1.0)


def _check_addressable(comp: AbstractComponent, role: str) -> None:
    if comp.n_index != comp.n_nodes:
        raise ValueError(
            f"{role} {type(comp).__name__} has padding nodes "
            f"(n_index={comp.n_index}, n_nodes={comp.n_nodes}); it cannot be a "
            "Kronecker / Replicate factor"
        )


def _product(
    left: gx.GaussianMRF | gx.IntrinsicGMRF,
    right: gx.GaussianMRF | gx.IntrinsicGMRF,
    constraint: Constraint,
    soft_constraint_scale: float,
) -> gx.GaussianMRF | gx.IntrinsicGMRF:
    op_l, V_l, s_l = _parts(left)
    op_r, V_r, s_r = _parts(right)
    n_l, n_r = op_l.in_size(), op_r.in_size()
    loc = jnp.zeros(n_l * n_r)
    if V_l is None and V_r is None:
        return gx.GaussianMRF(loc, gx.Kronecker(op_l, op_r))
    if V_l is not None and V_r is not None:
        raise NotImplementedError(
            "a Kronecker product of two intrinsic fields is not supported"
        )
    if V_l is not None:
        null = jnp.kron(V_l, jnp.eye(n_r))
    else:
        assert V_r is not None
        null = jnp.kron(jnp.eye(n_l), V_r)
    return gx.IntrinsicGMRF(
        loc,
        s_l * s_r,
        gx.Kronecker(op_l, op_r),
        null,
        constraint=constraint,
        soft_constraint_scale=soft_constraint_scale,
    )


class Kronecker(AbstractComponent):
    r"""Separable product $Q = Q_\text{main} \otimes Q_\text{group}$ (R-INLA ``group``).

    The precisions of the two factors multiply, so only the product of their
    $\tau$'s is identified; as R-INLA does, the group factor's ``tau`` is
    fixed at 1 and the group contributes only its correlation structure (an
    `AR1` group keeps ``rho``). Hyperparameter ``k`` of a factor named ``f``
    is ``f"{f}_{k}"``.

    Args:
        main: The main component (e.g. a spatial field).
        group: The grouping component (e.g. an `AR1` over time).
        name: Site-name prefix.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> import pyrox_lgm as lgm
        >>> st = lgm.Kronecker(
        ...     lgm.Leroux(kl.grid_graph((3, 3)), name="space"),
        ...     lgm.AR1(4, name="time"),
        ... )
        >>> sorted(st.theta_spec())
        ['space_rho', 'space_tau', 'time_rho']
        >>> st.n_nodes
        36
    """

    main: AbstractComponent
    group: AbstractComponent
    name: str = eqx.field(static=True)

    def __init__(
        self,
        main: AbstractComponent,
        group: AbstractComponent,
        *,
        name: str = "kronecker",
    ) -> None:
        _check_addressable(main, "main")
        _check_addressable(group, "group")
        if "tau" not in group.theta_spec():
            raise ValueError("the group component must have a 'tau' to fix at 1")
        self.main, self.group, self.name = main, group, name

    @property
    def n_nodes(self) -> int:
        return self.main.n_nodes * self.group.n_nodes

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        spec = {f"{self.main.name}_{k}": v for k, v in self.main.theta_spec().items()}
        spec |= {
            f"{self.group.name}_{k}": v
            for k, v in self.group.theta_spec().items()
            if k != "tau"
        }
        return spec

    def _split(self, theta: dict[str, Array]):
        main = {k: theta[f"{self.main.name}_{k}"] for k in self.main.theta_spec()}
        group = {
            k: (jnp.asarray(1.0) if k == "tau" else theta[f"{self.group.name}_{k}"])
            for k in self.group.theta_spec()
        }
        return main, group

    def prior(
        self,
        theta: dict[str, Array],
        *,
        constraint: Constraint = "hard",
        soft_constraint_scale: float = 1e-3,
    ) -> gx.GaussianMRF | gx.IntrinsicGMRF:
        main, group = self._split(theta)
        return _product(
            self.main.prior(main, constraint=constraint),
            self.group.prior(group, constraint=constraint),
            constraint,
            soft_constraint_scale,
        )


class Replicate(AbstractComponent):
    r"""``n_rep`` independent copies of a component, sharing its hyperparameters.

    $Q = I_{n_\text{rep}} \otimes Q_\text{component}$ (R-INLA's
    ``replicate``); an intrinsic component keeps its constraints in every
    copy.

    Args:
        component: The replicated component.
        n_rep: Number of copies.
        name: Site-name prefix; defaults to the component's.

    Examples:
        >>> import pyrox_lgm as lgm
        >>> rep = lgm.Replicate(lgm.RW1(10, name="trend"), 3)
        >>> rep.n_nodes, list(rep.theta_spec())
        (30, ['tau'])
    """

    component: AbstractComponent
    n_rep: int = eqx.field(static=True)
    name: str = eqx.field(static=True)

    def __init__(
        self, component: AbstractComponent, n_rep: int, *, name: str | None = None
    ) -> None:
        _check_addressable(component, "replicated")
        self.component = component
        self.n_rep = int(n_rep)
        self.name = component.name if name is None else name

    @property
    def n_nodes(self) -> int:
        return self.n_rep * self.component.n_nodes

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        return self.component.theta_spec()

    def prior(
        self,
        theta: dict[str, Array],
        *,
        constraint: Constraint = "hard",
        soft_constraint_scale: float = 1e-3,
    ) -> gx.GaussianMRF | gx.IntrinsicGMRF:
        identity = gx.GaussianMRF(
            jnp.zeros(self.n_rep), gx.iid_precision(self.n_rep, 1.0)
        )
        return _product(
            identity,
            self.component.prior(theta, constraint=constraint),
            constraint,
            soft_constraint_scale,
        )
