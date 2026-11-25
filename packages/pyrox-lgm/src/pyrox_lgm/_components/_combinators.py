r"""Combinators: separable Kronecker products and replicates of components (P7).

- `Kronecker` (R-INLA's ``group``): $Q = Q_\text{main} \otimes Q_\text{group}$,
  e.g. a spatial field correlated over time. Node $(i, j)$ of the product
  sits at flat index ``i * group.n_nodes + j``; an observation indexes
  ``i * group.n_index + j`` over the factors' addressable nodes.
- `Replicate` (R-INLA's ``replicate``): independent copies sharing
  hyperparameters, $Q = I_r \otimes Q$; copy $r$, node $i$ at
  ``r * component.n_nodes + i``; an observation indexes
  ``r * component.n_index + i``.

Both build their projector from their factors' projectors, so a factor whose
nodes are not its addressable indices (an odd `RW2`'s padding, a
`MarkovGP`'s states) reads the same values inside the product as alone.

At most one factor may be intrinsic; its null space is lifted to the
product as $V \otimes I$ (or $I \otimes V$), one constraint per node of the
other factor.
"""

from __future__ import annotations

import math

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import numpyro.distributions as dist
from jaxtyping import Array, ArrayLike, Float, Int
from numpyro.distributions.transforms import Transform

from pyrox_lgm._components._areal import BYM2
from pyrox_lgm._components._base import AbstractComponent, Constraint, probe_theta


_LOG_2PI = math.log(2.0 * math.pi)


class KroneckerIntrinsicGMRF(gx.IntrinsicGMRF):
    r"""Intrinsic $\mathcal N(0, (s\,R \otimes Q)^+)$ with its exact normaliser.

    One factor is intrinsic ($R$, rank $r_R$, ``n_R`` nodes) and the other
    proper ($Q$, ``n_Q`` nodes). The proper factor depends on $\theta$ (the
    group's $\rho$), so the normaliser is not a constant and is always
    included, from
    $\log|R \otimes Q|_+ = n_Q \log|R|_+ + r_R \log|Q|$.
    """

    intrinsic_op: lx.AbstractLinearOperator
    intrinsic_null: Float[Array, "n c"]
    proper_op: lx.AbstractLinearOperator

    def __init__(
        self, loc, scale, structure, null, intrinsic_op, intrinsic_null, proper_op, **kw
    ):
        super().__init__(loc, scale, structure, null, **kw)
        self.intrinsic_op = intrinsic_op
        self.intrinsic_null = intrinsic_null
        self.proper_op = proper_op

    @property
    def normalized(self) -> bool:
        """The log-density includes its normaliser (see `pyrox_lgm.LGM`)."""
        return True

    def log_prob(self, value):
        n_q = self.proper_op.in_size()
        rank_r = self.intrinsic_op.in_size() - self.intrinsic_null.shape[1]
        rank = rank_r * n_q
        log_pdet = n_q * gx.pseudo_logdet(
            self.intrinsic_op, null_space=self.intrinsic_null
        ) + rank_r * gx.logdet(self.proper_op)
        return super().log_prob(value) + 0.5 * log_pdet - 0.5 * rank * _LOG_2PI


def _parts(
    gmrf: gx.GaussianMRF | gx.IntrinsicGMRF,
) -> tuple[lx.AbstractLinearOperator, Float[Array, "n c"] | None, Array]:
    """(operator, null space or None, scale) with precision = scale * operator."""
    if isinstance(gmrf, gx.BYM2GMRF):
        raise NotImplementedError("BYM2 cannot be a Kronecker / Replicate factor")
    if isinstance(gmrf, gx.IntrinsicGMRF):
        return gmrf.structure, gmrf.null_space, gmrf.precision_scale
    return gmrf.precision, None, jnp.asarray(1.0)


def _probe_basis(comp: AbstractComponent):
    """A component's own constraint basis (theta-free for the intrinsic ones)."""
    return comp.constraint_basis(comp.prior(probe_theta(comp.theta_spec())))


def _check_factor(comp: AbstractComponent, role: str) -> None:
    if isinstance(comp, BYM2):
        raise ValueError(
            f"{role} BYM2 cannot be a Kronecker / Replicate factor: its prior "
            "is a (b, u*) pair, not one precision"
        )


def _check_index(index: ArrayLike, n_index: int) -> np.ndarray:
    index = np.asarray(index)
    if index.ndim != 1:
        raise ValueError(f"index must be 1-D, got shape {index.shape}")
    if index.size and (index.min() < 0 or index.max() >= n_index):
        raise ValueError(
            f"index out of range for {n_index} nodes: [{index.min()}, {index.max()}]"
        )
    return index


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
        intrinsic = (op_l, V_l, op_r)
    else:
        assert V_r is not None
        null = jnp.kron(jnp.eye(n_l), V_r)
        intrinsic = (op_r, V_r, op_l)
    return KroneckerIntrinsicGMRF(
        loc,
        s_l * s_r,
        gx.Kronecker(op_l, op_r),
        null,
        *intrinsic,
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
        _check_factor(main, "main")
        _check_factor(group, "group")
        if "tau" not in group.theta_spec():
            raise ValueError("the group component must have a 'tau' to fix at 1")
        self.main, self.group, self.name = main, group, name

    @property
    def n_nodes(self) -> int:
        return self.main.n_nodes * self.group.n_nodes

    @property
    def n_index(self) -> int:
        return self.main.n_index * self.group.n_index

    def projector(self, index: Int[ArrayLike, " n_obs"]) -> gx.SparseOperator:
        """Observation ``i * group.n_index + j`` reads main ``i`` and group ``j``."""
        i, j = np.divmod(_check_index(index, self.n_index), self.group.n_index)
        return gx.row_kron(self.main.projector(i), self.group.projector(j))

    def field_at(self, x, index=None):
        X = x.reshape(self.main.n_nodes, self.group.n_nodes)
        X = jax.vmap(self.group.field_at)(X)
        X = jax.vmap(self.main.field_at, in_axes=1, out_axes=1)(X)
        flat = X.reshape(-1)
        return flat if index is None else flat[jnp.asarray(index)]

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

    def constraint_basis(self, gmrf):
        # The intrinsic factor's own constraints, lifted (an RW2 main keeps
        # its sum-only rule), not the product's whole null space.
        main, group = _probe_basis(self.main), _probe_basis(self.group)
        if main is not None:
            return jnp.kron(main, jnp.eye(self.group.n_nodes))
        if group is not None:
            return jnp.kron(jnp.eye(self.main.n_nodes), group)
        return None

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
        _check_factor(component, "replicated")
        self.component = component
        self.n_rep = int(n_rep)
        self.name = component.name if name is None else name

    @property
    def n_nodes(self) -> int:
        return self.n_rep * self.component.n_nodes

    @property
    def n_index(self) -> int:
        return self.n_rep * self.component.n_index

    def projector(self, index: Int[ArrayLike, " n_obs"]) -> gx.SparseOperator:
        """Observation ``r * component.n_index + i`` reads node ``i`` of copy ``r``."""
        r, i = np.divmod(_check_index(index, self.n_index), self.component.n_index)
        m = r.shape[0]
        copy = gx.SparseOperator.from_coo(np.arange(m), r, jnp.ones(m), (m, self.n_rep))
        return gx.row_kron(copy, self.component.projector(i))

    def field_at(self, x, index=None):
        xs = x.reshape(self.n_rep, self.component.n_nodes)
        flat = jax.vmap(self.component.field_at)(xs).reshape(-1)
        return flat if index is None else flat[jnp.asarray(index)]

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        return self.component.theta_spec()

    def constraint_basis(self, gmrf):
        inner = _probe_basis(self.component)
        return None if inner is None else jnp.kron(jnp.eye(self.n_rep), inner)

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
