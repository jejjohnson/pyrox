r"""SPDE (Matérn) components on a finite-element mesh or a regular grid (P7).

The SPDE approach (Lindgren, Rue & Lindström, 2011) represents a Matérn
field with smoothness $\nu$ in dimension $d$ as the solution of
$(\kappa^2 - \Delta)^{\alpha/2}(\tau x) = \mathcal W$, $\alpha = \nu + d/2$,
whose finite-element discretisation is a sparse GMRF. The component is
parameterised by the interpretable practical range $\rho$ and marginal
standard deviation $\sigma$ (`gaussx.matern_spde_params`), with the joint PC
prior `PCMatern` (R-INLA's ``inla.spde2.pcmatern``).
"""

from __future__ import annotations

import equinox as eqx
import gaussx as gx
import jax.numpy as jnp
import lineax as lx
import numpy as np
import numpyro.distributions as dist
from jaxtyping import Array, ArrayLike, Float, Int
from numpyro.distributions.transforms import Transform

from pyrox_lgm._assembly import coo_power, full_coo
from pyrox_lgm._components._base import (
    AbstractComponent,
    Constraint,
    default_transform,
)
from pyrox_lgm._priors import PCMatern


class SPDE(AbstractComponent):
    r"""Matérn field via the SPDE, on a triangle mesh or a regular grid.

    Exactly one of ``mesh`` and ``grid`` is given:

    - ``mesh=(vertices, triangles)``: P1 finite elements on a triangulated
      surface (``vertices`` of shape ``(V, 2)`` or ``(V, 3)`` on a manifold),
      with lumped mass ``C̃`` and stiffness ``G`` from
      `gaussx.fem_matrices`; the precision is `gaussx.spde_precision`
      (sparse). Nodes are the vertices; `project_points` builds the
      barycentric projector for arbitrary locations. The field is 2-D
      (``d = 2``).
    - ``grid=shape``: the finite-difference SPDE on a regular lattice with
      ``spacing`` (`gaussx.spde_precision_grid`, diagonalised exactly by its
      Kronecker structure). Nodes are the grid cells in C order; ``d`` is
      ``len(shape)``.

    The hyperparameter ``"range_sigma"`` is the pair ``[range, sigma]``
    under the joint `PCMatern` prior.

    Args:
        mesh: ``(vertices, triangles)``.
        grid: Lattice shape.
        spacing: Grid spacing (``grid`` only).
        alpha: SPDE order $\alpha = \nu + d/2$, an integer (2 is the
            default Matérn-1 field in 2-D).
        name: Site-name prefix.
        prior: Joint prior on ``[range, sigma]``; default
            ``PCMatern(range0, 0.5, 1, 0.01, d)`` with ``range0`` a fifth of
            the domain's diameter.

    Examples:
        >>> import jax.numpy as jnp
        >>> import pyrox_lgm as lgm
        >>> spde = lgm.SPDE(grid=(30, 30), alpha=2)
        >>> g = spde.prior({"range_sigma": jnp.array([8.0, 2.0])})
        >>> var = g.marginal_variances().reshape(30, 30)
        >>> # ~ sigma^2 = 4 away from the border, up to the lattice discretisation
        >>> bool(abs(float(var[15, 15]) - 4.0) < 0.4)
        True
    """

    alpha: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    shape: tuple[int, ...] | None = eqx.field(static=True)
    spacing: float = eqx.field(static=True)
    vertices: Float[Array, "V D"] | None
    triangles: Int[Array, "T 3"] | None
    mass: lx.DiagonalLinearOperator | None
    stiffness: gx.SparseOperator | None
    name: str = eqx.field(static=True)
    prior_range_sigma: dist.Distribution

    def __init__(
        self,
        *,
        mesh: tuple[ArrayLike, ArrayLike] | None = None,
        grid: tuple[int, ...] | None = None,
        spacing: float = 1.0,
        alpha: int = 2,
        name: str = "spde",
        prior: dist.Distribution | None = None,
    ) -> None:
        if (mesh is None) == (grid is None):
            raise ValueError(
                "give exactly one of mesh=(vertices, triangles) or grid=shape"
            )
        self.alpha = int(alpha)
        self.spacing = float(spacing)
        self.name = name
        if mesh is not None:
            V = jnp.asarray(mesh[0])
            T = jnp.asarray(mesh[1])
            self.vertices, self.triangles = V, T
            self.mass, self.stiffness = gx.fem_matrices(V, T)
            self.shape = None
            self.d = 2
            extent = np.asarray(V.max(axis=0) - V.min(axis=0))
        else:
            assert grid is not None
            self.shape = tuple(int(s) for s in grid)
            self.vertices = self.triangles = None
            self.mass = self.stiffness = None
            self.d = len(self.shape)
            extent = (np.asarray(self.shape) - 1) * self.spacing
        if self.alpha - self.d / 2 <= 0:
            raise ValueError(
                f"alpha={self.alpha} gives smoothness nu = alpha - d/2 <= 0 "
                f"in d={self.d}"
            )
        if prior is None:
            range0 = float(np.linalg.norm(extent)) / 5.0
            prior = PCMatern(range0, 0.5, 1.0, 0.01, d=self.d)
        self.prior_range_sigma = prior

    @property
    def nu(self) -> float:
        """Smoothness $\\nu = \\alpha - d/2$."""
        return self.alpha - self.d / 2

    @property
    def n_nodes(self) -> int:
        if self.shape is not None:
            return int(np.prod(self.shape))
        assert self.vertices is not None
        return int(self.vertices.shape[0])

    def theta_spec(self) -> dict[str, tuple[dist.Distribution, Transform]]:
        p = self.prior_range_sigma
        return {"range_sigma": (p, default_transform(p))}

    def prior(
        self,
        theta: dict[str, Array],
        *,
        constraint: Constraint = "hard",
        soft_constraint_scale: float = 1e-3,
    ) -> gx.GaussianMRF:
        del constraint, soft_constraint_scale  # proper
        range_, sigma = theta["range_sigma"][0], theta["range_sigma"][1]
        kappa, tau, alpha = gx.matern_spde_params(range_, sigma, self.nu, self.d)
        if self.shape is not None:
            Q = gx.spde_precision_grid(
                self.shape, kappa, tau, alpha, spacing=self.spacing
            )
        else:
            assert self.mass is not None and self.stiffness is not None
            Q = gx.spde_precision(self.mass, self.stiffness, kappa, tau, alpha)
        return gx.GaussianMRF(jnp.zeros(self.n_nodes), Q)

    def assembly_precision(self, theta, gmrf):
        """The grid precision as a `gaussx.SparseOperator` (mesh: unchanged).

        ``Q = tau^2 h^d (kappa^2 I + h^-2 L)^alpha``, ``L`` the Kronecker sum
        of the axes' path-graph Laplacians, exactly `gaussx.spde_precision_grid`
        written entry-wise: ``alpha`` sparse products on a theta-free pattern.
        """
        if self.shape is None:
            return gmrf.precision
        range_, sigma = theta["range_sigma"][0], theta["range_sigma"][1]
        kappa, tau, alpha = gx.matern_spde_params(range_, sigma, self.nu, self.d)
        h = self.spacing
        laplacian = _path_laplacians(self.shape)
        r, c, v = full_coo(laplacian)
        n = self.n_nodes
        diag = np.arange(n)
        base = (
            np.concatenate([r, diag]),
            np.concatenate([c, diag]),
            jnp.concatenate([v / h**2, jnp.full(n, kappa**2)]),
        )
        rows, cols, vals = coo_power(base, alpha, n)
        return gx.SparseOperator.from_coo(rows, cols, tau**2 * h**self.d * vals, (n, n))

    def project_points(self, points: Float[ArrayLike, "n D"]) -> gx.SparseOperator:
        """Barycentric ``(n, V)`` projector from mesh vertices to ``points``."""
        if self.vertices is None or self.triangles is None:
            raise ValueError("project_points needs a mesh; index grid cells directly")
        return gx.fem_projector(self.vertices, self.triangles, points)


def _path_laplacians(shape: tuple[int, ...]) -> lx.AbstractLinearOperator:
    """``L_1 ⊕ ... ⊕ L_d`` of path-graph Laplacians (natural boundary)."""

    def path(n: int) -> gx.SparseOperator:
        deg = np.full(n, 2.0)
        deg[0] -= 1.0  # one neighbour at each end; a single node has none
        deg[-1] -= 1.0
        i = np.arange(n - 1)
        return gx.SparseOperator.from_coo(
            np.concatenate([np.arange(n), i + 1, i]),
            np.concatenate([np.arange(n), i, i + 1]),
            jnp.asarray(np.concatenate([deg, -np.ones(2 * (n - 1))])),
            (n, n),
        )

    ops = [path(n) for n in shape]
    out = ops[-1]
    for op in reversed(ops[:-1]):
        out = gx.KroneckerSum(op, out)
    return out
