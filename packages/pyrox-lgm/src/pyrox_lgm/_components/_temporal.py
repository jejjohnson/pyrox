r"""Temporal (and unstructured) components: IID, RW1, RW2 and AR1 (P7).

| Component | Field | Hyperparameters | gaussx builder |
|---|---|---|---|
| `IID` | $x_i \sim \mathcal N(0, 1/\tau)$ | $\tau$ | `iid_precision` |
| `RW1` | $x_{t+1}-x_t \sim \mathcal N(0, 1/\tau)$ | $\tau$ | `rw1_structure` |
| `RW2` | $x_{t+1}-2x_t+x_{t-1} \sim \mathcal N(0, 1/\tau)$ | $\tau$ | `rw2_structure` |
| `AR1` | stationary AR(1), marginal precision $\tau$ | $\tau, \rho$ | `ar1_precision` |

The random walks are intrinsic: their null spaces (constants, plus linear
trends for RW2) are removed by sum-to-zero (and zero-slope) constraints,
hard in ``inla()`` and soft under NUTS. With ``scale_model=True`` (the
default, as R-INLA's ``scale.model``) the structure is rescaled so the
geometric mean of the constrained marginal variances is 1, which makes
$\tau$ mean the same for every ``n`` (Sørbye & Rue, 2014).
"""

from __future__ import annotations

import equinox as eqx
import gaussx as gx
import jax.numpy as jnp
import lineax as lx
import numpyro.distributions as dist
from jaxtyping import Array, Float
from numpyro.distributions.transforms import Transform

from pyrox_lgm._components._base import (
    AbstractComponent,
    Constraint,
    default_transform,
)
from pyrox_lgm._priors import PCAR1Rho, PCPrecision


def _tau_prior(prior: dist.Distribution | None) -> dist.Distribution:
    return PCPrecision(1.0, 0.01) if prior is None else prior


class IID(AbstractComponent):
    r"""Independent effects $x_i \sim \mathcal N(0, 1/\tau)$, $i < n$.

    Args:
        n: Number of nodes.
        name: Site-name prefix.
        tau_prior: Prior on $\tau$; default ``PCPrecision(1, 0.01)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import pyrox_lgm as lgm
        >>> iid = lgm.IID(4, name="u")
        >>> q = iid.prior({"tau": jnp.asarray(2.0)}).precision.as_matrix()
        >>> [float(v) for v in jnp.diag(q)]
        [2.0, 2.0, 2.0, 2.0]
    """

    n: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    tau_prior: dist.Distribution

    def __init__(
        self, n: int, *, name: str = "iid", tau_prior: dist.Distribution | None = None
    ) -> None:
        self.n = int(n)
        self.name = name
        self.tau_prior = _tau_prior(tau_prior)

    @property
    def n_nodes(self) -> int:
        return self.n

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        return {"tau": (self.tau_prior, default_transform(self.tau_prior))}

    def prior(
        self,
        theta: dict[str, Array],
        *,
        constraint: Constraint = "hard",
        soft_constraint_scale: float = 1e-3,
    ) -> gx.GaussianMRF:
        del constraint, soft_constraint_scale  # proper
        return gx.GaussianMRF(jnp.zeros(self.n), gx.iid_precision(self.n, theta["tau"]))


class _IntrinsicWalk(AbstractComponent):
    """Shared machinery of the random walks: a cached scaled structure."""

    n: eqx.AbstractVar[int]
    name: eqx.AbstractVar[str]
    tau_prior: eqx.AbstractVar[dist.Distribution]
    structure: eqx.AbstractVar[lx.AbstractLinearOperator]
    null_space: eqx.AbstractVar[Float[Array, "N c"]]
    scale: eqx.AbstractVar[Float[Array, ""]]

    @property
    def n_nodes(self) -> int:
        return self.structure.in_size()

    @property
    def n_index(self) -> int:
        return self.n

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        return {"tau": (self.tau_prior, default_transform(self.tau_prior))}

    def prior(
        self,
        theta: dict[str, Array],
        *,
        constraint: Constraint = "hard",
        soft_constraint_scale: float = 1e-3,
    ) -> gx.IntrinsicGMRF:
        # tau * (s R): scale the precision, not the operator, so the
        # structure keeps its banded solver.
        return gx.IntrinsicGMRF(
            jnp.zeros(self.n_nodes),
            theta["tau"] * self.scale,
            self.structure,
            self.null_space,
            constraint=constraint,
            soft_constraint_scale=soft_constraint_scale,
        )


class RW1(_IntrinsicWalk):
    r"""First-order random walk, $x_{t+1} - x_t \sim \mathcal N(0, 1/\tau)$.

    Intrinsic of order 1: the constant is its null space, removed by a
    sum-to-zero constraint. A cyclic walk also links $x_{n-1}$ and $x_0$.

    Args:
        n: Number of time points.
        name: Site-name prefix.
        tau_prior: Prior on $\tau$; default ``PCPrecision(1, 0.01)``.
        scale_model: Rescale the structure so the geometric mean of the
            constrained marginal variances is 1 (R-INLA's ``scale.model``).
        cyclic: Join the ends.

    Examples:
        >>> import jax.numpy as jnp
        >>> import pyrox_lgm as lgm
        >>> rw = lgm.RW1(20)
        >>> var = rw.prior({"tau": jnp.asarray(1.0)}).marginal_variances()
        >>> round(float(jnp.exp(jnp.mean(jnp.log(var)))), 4)  # scale_model
        1.0
    """

    n: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    tau_prior: dist.Distribution
    structure: lx.AbstractLinearOperator
    null_space: Float[Array, "N c"]
    scale: Float[Array, ""]

    def __init__(
        self,
        n: int,
        *,
        name: str = "rw1",
        tau_prior: dist.Distribution | None = None,
        scale_model: bool = True,
        cyclic: bool = False,
    ) -> None:
        self.n = int(n)
        self.name = name
        self.tau_prior = _tau_prior(tau_prior)
        self.structure = gx.rw1_structure(self.n, cyclic=cyclic)
        self.null_space = jnp.ones((self.n, 1)) / jnp.sqrt(self.n)
        self.scale = (
            gx.generalized_variance_scale(self.structure, self.null_space)
            if scale_model
            else jnp.asarray(1.0)
        )


class RW2(_IntrinsicWalk):
    r"""Second-order random walk, $x_{t+1} - 2x_t + x_{t-1} \sim \mathcal N(0, 1/\tau)$.

    Intrinsic of order 2: constants and linear trends form its null space.
    Inside ``inla()`` only the sum-to-zero constraint is imposed, as R-INLA's
    ``rw2`` does, and the linear trend is left to the data (the density's
    rank stays ``n - 2``); the NumPyro face soft-constrains both
    directions, so add a linear fixed effect there if the data have a trend.
    For odd ``n``,
    gaussx's banded structure carries one decoupled padding node
    (``n_nodes = n + 1``); it never enters a projector, and its own scale
    is taken out of ``scale_model``.

    Args:
        n: Number of time points (at least 3).
        name: Site-name prefix.
        tau_prior: Prior on $\tau$; default ``PCPrecision(1, 0.01)``.
        scale_model: Rescale as for `RW1`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import pyrox_lgm as lgm
        >>> rw = lgm.RW2(7)  # odd: one padding node
        >>> rw.n_nodes, rw.n_index
        (8, 7)
        >>> var = rw.prior({"tau": jnp.asarray(1.0)}).marginal_variances()[:7]
        >>> round(float(jnp.exp(jnp.mean(jnp.log(var)))), 4)
        1.0
    """

    n: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    tau_prior: dist.Distribution
    structure: lx.AbstractLinearOperator
    null_space: Float[Array, "N c"]
    scale: Float[Array, ""]

    def __init__(
        self,
        n: int,
        *,
        name: str = "rw2",
        tau_prior: dist.Distribution | None = None,
        scale_model: bool = True,
    ) -> None:
        if n < 3:
            raise ValueError(f"RW2 needs at least 3 time points, got {n}")
        self.n = int(n)
        self.name = name
        self.tau_prior = _tau_prior(tau_prior)
        self.structure = gx.rw2_structure(self.n)
        N = self.structure.in_size()
        t = jnp.arange(self.n, dtype=float)
        basis = jnp.zeros((N, 2)).at[: self.n, 0].set(1.0)
        basis = basis.at[: self.n, 1].set(t - t.mean())
        self.null_space, _ = jnp.linalg.qr(basis)  # zero on the padding row
        if scale_model:
            s_all = gx.generalized_variance_scale(self.structure, self.null_space)
            # The padding node has unit variance under R, so it pulls the
            # geometric mean over all N nodes towards 1: undo it to get the
            # scale over the n real nodes, s_n = s_N^(N/n).
            self.scale = s_all ** (N / self.n)
        else:
            self.scale = jnp.asarray(1.0)

    def constraint_basis(self, gmrf):
        # R-INLA's rw2 constrains the sum only: the linear trend stays in the
        # model (flat, identified by the data) instead of being removed.
        return self.null_space[:, :1]


class AR1(AbstractComponent):
    r"""Stationary AR(1), $x_t = \rho x_{t-1} + \varepsilon_t$.

    The marginal precision is $\tau$: every marginal variance is $1/\tau$
    (R-INLA's ``ar1`` convention, so $\tau$ and $\rho$ are separately
    interpretable).

    Args:
        n: Number of time points (at least 2).
        name: Site-name prefix.
        tau_prior: Prior on the marginal precision; default
            ``PCPrecision(1, 0.01)``.
        rho_prior: Prior on $\rho$; default ``PCAR1Rho(0.5, 0.5)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import pyrox_lgm as lgm
        >>> ar = lgm.AR1(30)
        >>> gmrf = ar.prior({"tau": jnp.asarray(4.0), "rho": jnp.asarray(0.7)})
        >>> [round(float(v), 4) for v in gmrf.marginal_variances()[:3]]
        [0.25, 0.25, 0.25]
    """

    n: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    tau_prior: dist.Distribution
    rho_prior: dist.Distribution

    def __init__(
        self,
        n: int,
        *,
        name: str = "ar1",
        tau_prior: dist.Distribution | None = None,
        rho_prior: dist.Distribution | None = None,
    ) -> None:
        if n < 2:
            raise ValueError(f"AR1 needs at least 2 time points, got {n}")
        self.n = int(n)
        self.name = name
        self.tau_prior = _tau_prior(tau_prior)
        self.rho_prior = PCAR1Rho(0.5, 0.5) if rho_prior is None else rho_prior

    @property
    def n_nodes(self) -> int:
        return self.n

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        return {
            "tau": (self.tau_prior, default_transform(self.tau_prior)),
            "rho": (self.rho_prior, default_transform(self.rho_prior)),
        }

    def prior(
        self,
        theta: dict[str, Array],
        *,
        constraint: Constraint = "hard",
        soft_constraint_scale: float = 1e-3,
    ) -> gx.GaussianMRF:
        del constraint, soft_constraint_scale  # proper
        Q = gx.ar1_precision(self.n, theta["rho"], theta["tau"])
        return gx.GaussianMRF(jnp.zeros(self.n), Q)
